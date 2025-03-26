# Standard
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import logging
import os
import re
import sys

# Third Party
from datasets import Dataset
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
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tabulate import tabulate

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

        if self.document_paths == []:
            return []

        parsed_documents = self.converter.convert_all(self.document_paths)

        docling_artifacts_path = self.export_documents(parsed_documents)

        docling_json_paths = list(docling_artifacts_path.glob("*.json"))
        chunks = []
        for json_fp in docling_json_paths:
            chunks.extend(self._process_parsed_docling_json(json_fp))

        return chunks

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

    def _process_parsed_docling_json(self, json_fp: Path) -> Dataset:
        """
        Process the parsed docling json file and return a dataset.
        Args:
            json_fp (Path): Path to the parsed docling json file.
        Returns:
            List: a list of chunks built from the provided json file
        """
        logger.info(f"Processing parsed docling json file: {json_fp}")
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = self.build_chunks_from_docling_json(
            data,
            max_token_per_chunk=500,
            tokenizer=self.tokenizer,
        )
        fused_texts = self.fuse_texts(chunks, 200)

        num_tokens_per_doc = _num_tokens_from_words(self.chunk_word_count)
        chunk_size = _num_chars_from_tokens(num_tokens_per_doc)
        return chunk_markdowns(fused_texts, chunk_size)

    def fuse_texts(
        self, text_list: List, short_length_threshold: int = 130
    ) -> List[str]:
        """
        Fuse short texts with preceding longer texts if their token count is below the threshold.
        Args:
            text_list (list): List of text chunks to process.
            short_length_threshold (int): The token count threshold for determining short texts.
                                      Default is 130, tuned specifically for the Mixtral tokenizer.
                                      Update this value if changing the tokenizer model.
        Returns:
            list: List of fused texts.
        """
        fused_texts: List[str] = []
        previous_long_text = ""

        for text in text_list:
            token_count = self.get_token_count(
                text, self.tokenizer
            )  # Use tokenizer for token count

            if token_count <= short_length_threshold and previous_long_text:
                # Append the short text to the last long text
                fused_texts[-1] += "\n\n" + text
            else:
                # This is a long text, so add it to the list and remember it
                fused_texts.append(text)
                previous_long_text = text

        return fused_texts

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

    def add_heading_formatting(self, text):
        """
        Add heading formatting to the text if the first part is short.
        Args:
            text (str): The input text to format.
        Returns:
            str: Formatted text with headings applied.
        """
        text = text.split(".")

        # Change this from hardcoded to something more flexible
        if len(text) > 1 and len(text[0].split(" ")) < 3:
            text = f"**{text[0]}**" + ".".join(text[1:])
        else:
            text = ".".join(text)
        return text

    def generate_table_from_parsed_rep(self, item):
        """
        Generate the table from the parsed representation and return as a string.
        Args:
            item (dict): Parsed representation of a table.
        Returns:
            str: Formatted table as a string.
        """
        caption = ""
        if "text" in item:
            caption = item["text"]

        data = item["data"]

        if len(data) <= 1 or len(data[0]) <= 1:
            return ""

        table = []
        for _, row in enumerate(data):
            trow = []
            for _, cell in enumerate(row):
                trow.append(cell["text"])
            table.append(trow)

        table_text = tabulate(table, tablefmt="github")
        if caption:
            table_text += f"\nCaption: {caption}\n"
        return table_text

    def get_table(self, json_book, table_ref):
        """
        Retrieve a table from a document based on a reference string.
        Args:
            json_book (dict): JSON representation of the document.
            table_ref (str): Reference path to the table within the document.
        Returns:
            str: Formatted table string.
        """
        parts = table_ref.split("/")
        table_text = self.generate_table_from_parsed_rep(
            json_book[parts[1]][int(parts[2])]
        )
        return table_text

    def get_table_page_number(self, json_book, idx):
        """
        Get the page number of a table or other document element.
        Args:
            json_book (dict): JSON representation of the document.
            idx (int): Index of the element in the document.
        Returns:
            int: Page number of the element.
        """
        prev_page_num, next_page_num = None, None
        for book_element in json_book["main-text"][idx - 1 :: -1]:
            if "prov" in book_element and book_element["prov"]:
                prev_page_num = book_element["prov"][0]["page"]
                break
        for book_element in json_book["main-text"][idx:]:
            if "prov" in book_element and book_element["prov"]:
                next_page_num = book_element["prov"][0]["page"]
                break
        if prev_page_num is not None and next_page_num is not None:
            if prev_page_num == next_page_num:
                return prev_page_num
            return next_page_num
        if prev_page_num is not None:
            return prev_page_num
        if next_page_num is not None:
            return next_page_num

    def build_chunks_from_docling_json(
        self,
        json_book,
        max_token_per_chunk,
        tokenizer,
        keep_same_page_thing_together=False,
        chunking_criteria=None,
    ):
        """
        Build document chunks from a docling JSON representation.
        Args:
            json_book (dict): JSON document to process.
            max_token_per_chunk (int): Maximum token count per chunk.
            tokenizer (AutoTokenizer): Tokenizer instance to use.
            keep_same_page_thing_together (bool): Whether to keep content on the same page together.
            chunking_criteria (callable): Custom function for determining chunk breaks.
        Returns:
            list: List of document chunks.
        """
        current_buffer = []
        document_chunks = []
        prev_page_number = None
        book_title = None

        for idx, book_element in enumerate(json_book["main-text"]):
            if book_element["type"] in [
                "page-footer",
                "picture",
                "reference",
                "meta-data",
                "figure",
                "page-header",
            ]:
                continue
            if book_element["type"] == "footnote":
                current_book_page_number = book_element["prov"][0]["page"]
            elif book_element["type"] in [
                "subtitle-level-1",
                "paragraph",
                "table",
                "title",
                "equation",
            ]:  # 'page-header',
                if book_element["type"] == "table":
                    current_book_page_number = self.get_table_page_number(
                        json_book, idx
                    )
                    book_text = self.get_table(json_book, book_element["$ref"])
                elif book_element["prov"]:
                    current_book_page_number = book_element["prov"][0][
                        "page"
                    ]  # TODO export to function to handle empty ["prov"]
                    book_text = book_element["text"]
                else:
                    current_book_page_number = None
                    book_text = book_element["text"]

                if book_element["type"] == "subtitle-level-1":
                    if book_title is None:
                        book_title = book_text
                        book_text = f"# Title: **{book_text}**"
                    else:
                        book_text = f"## **{book_text}**"

                if book_element["type"] == "title":
                    book_text = f"# **{book_text}**"
                if book_element["type"] == "page-header":
                    book_text = f"Page Header: **{book_text}**\n\n"

                if chunking_criteria is not None:
                    # custom break function that can be used to chunk document
                    if chunking_criteria(book_text):
                        document_chunks.append("\n\n".join(current_buffer))
                        current_buffer = []
                elif (
                    prev_page_number is not None
                    and prev_page_number != current_book_page_number
                ) and keep_same_page_thing_together:
                    document_chunks.append("\n\n".join(current_buffer))
                    current_buffer = []
                else:
                    if (
                        self.get_token_count("\n\n".join(current_buffer), tokenizer)
                        >= max_token_per_chunk
                        and len(current_buffer) > 1
                    ):
                        chunk_text = "\n\n".join(current_buffer[:-1])
                        logger.debug(
                            f"Current chunk size {self.get_token_count(chunk_text, tokenizer)} and max is {max_token_per_chunk}"
                        )

                        document_chunks.append("\n\n".join(current_buffer[:-1]))

                        if (
                            self.get_token_count(current_buffer[-1], tokenizer)
                            >= max_token_per_chunk
                        ):
                            logger.debug(
                                f"The following text was dropped from the document because it was too long to fit into a single context for synthetic data generation: {current_buffer[-1]}"
                            )
                            document_chunks.append(current_buffer[-1])
                            current_buffer = []
                        else:
                            current_buffer = current_buffer[-1:]

                if book_element["type"] == "paragraph":
                    book_text = self.add_heading_formatting(book_text)
                if "## References" in book_text or "## Acknowledgements" in book_text:
                    # For research papers we ignore everything after this sections
                    break
                current_buffer.append(book_text)

            try:
                prev_page_number = current_book_page_number
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error processing book element: {book_element}, {str(e)}")

        if "\n\n".join(current_buffer) not in document_chunks:
            document_chunks.append("\n\n".join(current_buffer))
        return document_chunks

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
                    fp.write(json.dumps(doc.legacy_document.export_to_dict()))

                # Export Markdown format:
                with (docling_artifacts_path / f"{doc_filename}.md").open("w") as fp:
                    fp.write(doc.legacy_document.export_to_markdown())
            else:
                logger.info(f"Document {doc.input.file} failed to convert.")
                failure_count += 1

        logger.info(
            f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
        )

        return docling_artifacts_path


def chunk_markdowns(documents: List | Dataset, chunk_size) -> Dataset:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (list): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """

    # Checks for input type error
    content = []
    # chunk_size = _num_chars_from_tokens(no_tokens_per_doc)
    chunk_overlap = _DEFAULT_CHUNK_OVERLAP

    # Using Markdown as default, document-specific chunking will be implemented in separate pr.
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Determine file type for heuristics, default with markdown
    for docs in documents:
        # Use regex to remove unnecessary dashes in front of pipe characters in a markdown table.
        docs = re.sub(r"-{2,}\|", "-|", docs)
        # Remove unnecessary spaces in front of pipe characters in a markdown table.
        docs = re.sub(r"\  +\|", " |", docs)
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])
    return content
