# Standard
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import logging
import os
import sys

# Third Party
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    OcrOptions,
    PdfPipelineOptions,
    TesseractOcrOptions,
)

# First Party
from instructlab.sdg.utils.model_formats import is_model_gguf, is_model_safetensors

logger = logging.getLogger(__name__)
_DEFAULT_CHUNK_OVERLAP = 100
SUPPORTED_FILETYPES = [".pdf", ".md"]


def _num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def _num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def resolve_ocr_options(
    docling_model_path: Optional[Path] = None,
) -> Optional[OcrOptions]:
    # Declare ocr_options explicitly as Optional[OcrOptions]
    ocr_options: Optional[OcrOptions] = None

    # First, attempt to use tesserocr
    try:
        ocr_options = TesseractOcrOptions()
        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling.models.tesseract_ocr_model import TesseractOcrModel

        _ = TesseractOcrModel(
            enabled=True,
            artifacts_path=docling_model_path,
            options=ocr_options,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        )
        return ocr_options
    except ImportError:
        # No tesserocr, so try something else
        logger.warning("Tesseract not found, falling back to EasyOCR.")

    try:
        ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=None,
            confidence_threshold=0.5,
            model_storage_directory=str(docling_model_path),
            recog_network="standard",
            download_enabled=True,
        )
        # triggers torch loading, import lazily
        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling.models.easyocr_model import EasyOcrModel

        _ = EasyOcrModel(
            enabled=True,
            artifacts_path=None,
            options=ocr_options,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        )
        return ocr_options
    except ImportError:
        # no easyocr either, so don't use any OCR
        logger.error(
            "Failed to load Tesseract and EasyOCR - disabling optical character recognition in PDF documents"
        )
        return None


def split_docs_by_filetype(document_paths: List[Path]) -> Dict[str, List[Path]]:
    """Split document paths into a dict of lists based on their file extension."""
    document_dict = defaultdict(list)
    for path in document_paths:
        filetype = path.suffix
        if filetype not in SUPPORTED_FILETYPES:
            raise ValueError(f"Provided unsupported filetype {filetype}")

        document_dict[filetype].append(path)

    return dict(document_dict)


class DocumentChunker:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        document_paths: List[Path],
        output_dir: Path,
        tokenizer_model_name: str | Path,
        docling_model_path: Optional[Path] = None,
        server_ctx_size: int = 4096,
        chunk_word_count: int = 1024,
    ):
        if not document_paths:
            raise ValueError("Provided empty list of documents")

        document_dict = split_docs_by_filetype(document_paths)

        if len(document_dict) > 1:
            raise ValueError("Provided multiple document types")

        # We know there is only 1 key, value pair, so we take the first
        self.document_filetype, self.document_paths = next(iter(document_dict.items()))
        self.docling_model_path = docling_model_path
        self.converter = self._init_docling_converter()

        self.output_dir = self._path_validator(output_dir)
        self.server_ctx_size = server_ctx_size
        self.chunk_word_count = chunk_word_count
        self.tokenizer = self.create_tokenizer(tokenizer_model_name)

    def _init_docling_converter(self):
        """Initialize docling converter with filetype-specific configurations"""
        # triggers torch loading, import lazily
        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        if self.docling_model_path is None:
            logger.info("Docling models not found on disk, downloading models...")
            self.docling_model_path = StandardPdfPipeline.download_models_hf()
        else:
            logger.info("Found the docling models")

        pipeline_options = PdfPipelineOptions(
            artifacts_path=self.docling_model_path,
            do_ocr=False,
        )

        # deactivate MPS acceleration on Github CI
        if os.getenv("CI") and sys.platform == "darwin":
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU
            )
        ocr_options = resolve_ocr_options(docling_model_path=self.docling_model_path)
        if ocr_options is not None:
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = ocr_options

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def chunk_documents(self) -> List:
        """Split a list of documents into chunks

        Returns:
            List: a list of chunks from the documents
        """
        # Move docling_core import inside method where it's used to avoid importing transformers at top level
        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        parsed_documents = self.converter.convert_all(self.document_paths)
        all_chunks = []
        for conversion_result in parsed_documents:
            doc = conversion_result.document
            chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=500)
            try:
                chunk_iter = chunker.chunk(dl_doc=doc)
                chunks = [chunker.serialize(chunk=chunk) for chunk in chunk_iter]
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    f"Error chunking document {conversion_result.input.file}: {e}"
                )
                chunks = []
            all_chunks.extend(chunks)
        return all_chunks

    def _path_validator(self, path) -> Path:
        """
        Validate the path and return a Path object.
        Args:
            path (str): Path to be validated.
        Returns:
            Path: Path object.
        """
        if isinstance(path, str):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
        return path

    @staticmethod
    def create_tokenizer(model_path: str | Path):
        """
        Create a tokenizer instance from a pre-trained model or a local directory.

        Args:
            model_name (str): The name of the model or the path to the local directory.

        Returns:
            AutoTokenizer: The tokenizer instance.
        """
        # import lazily to not load transformers at top level
        # pylint: disable=import-outside-toplevel
        # Third Party
        from transformers import AutoTokenizer

        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        error_info_message = (
            "Please run `ilab model download {download_args}` and try again"
        )
        try:
            if is_model_safetensors(model_path):
                error_info_message = error_info_message.format(
                    download_args=f"--repository {model_path}"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)

            elif is_model_gguf(model_path):
                model_dir, model_filename = model_path.parent, model_path.name
                error_info_message = error_info_message.format(
                    download_args=f"--repository {model_dir} --filename {model_filename}"
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir, gguf_file=model_filename
                )

            else:
                error_info_message = "Please provide a path to a valid model format. For help on downloading models, run `ilab model download --help`."
                raise ValueError()

            logger.info(f"Successfully loaded tokenizer from: {model_path}")
            return tokenizer

        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to load tokenizer as no valid model was not found at {model_path}. {error_info_message}"
            )
            raise e

    def get_token_count(self, text, tokenizer):
        """
        Get the number of tokens in a text using the provided tokenizer.
        Args:
            text (str): The text to tokenize.
            tokenizer (AutoTokenizer): The tokenizer to use.
        Returns:
            int: Number of tokens.
        """
        return len(tokenizer.tokenize(text))

    def export_documents(self, converted_docs: Iterable[ConversionResult]):
        """Write converted documents to json files

        Check for successful conversions and write those to the docling artifacts directory.
        Returns:
            Path: path to directory with docling json artifacts
        """
        # triggers torch loading, import lazily
        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling.document_converter import ConversionStatus

        docling_artifacts_path = self.output_dir / "docling-artifacts"
        docling_artifacts_path.mkdir(parents=True, exist_ok=True)

        success_count = 0
        failure_count = 0

        for doc in converted_docs:
            if doc.status == ConversionStatus.SUCCESS:
                success_count += 1
                doc_filename = doc.input.file.stem

                # Export Deep Search document JSON format:
                with (docling_artifacts_path / f"{doc_filename}.json").open("w") as fp:
                    fp.write(json.dumps(doc.document.export_to_dict()))

                # Export Markdown format:
                with (docling_artifacts_path / f"{doc_filename}.md").open("w") as fp:
                    fp.write(doc.document.export_to_markdown())
            else:
                logger.info(f"Document {doc.input.file} failed to convert.")
                failure_count += 1

        logger.info(
            f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
        )

        return docling_artifacts_path
