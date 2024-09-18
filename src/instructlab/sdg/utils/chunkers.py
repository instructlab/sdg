import json
import logging
import re
import yaml
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Tuple, DefaultDict

# Third Party
from datasets import Dataset, concatenate_datasets
from docling.datamodel.base_models import PipelineOptions
from docling.datamodel.document import ConvertedDocument, DocumentConversionInput
from docling.document_converter import ConversionStatus, DocumentConverter
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tabulate import tabulate
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
_DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TAXONOMY_PATH = Path("~/.local/share/instructlab/taxonomy").expanduser()


class FileTypes(Enum):
    MD = ".md"
    PDF = ".pdf"


class ChunkerBase(ABC):
    @abstractmethod
    def chunk_documents():
        pass


class DocumentChunker:
    """A factory chunker class that instantiates the applicable chunker
    
    Currently, only Markdown and PDF are supported. For Markdown, returns
    TextSplitChunker, and for PDF, returns ContextAwareChunker"""
    def __new__(
        cls,
        leaf_node = None,
        output_dir: Path = None,
        server_ctx_size=4096,
        chunk_word_count=1024,
        tokenizer_model_name: str = None,
    ):
        """Insantiate the appropriate chunker for the provided document
        
        Args:
            leaf_node: a leaf node dict containing "documents",
                "filepaths", and "taxonomy_path" keys
            output_dir (Path): directory where artifacts should be stored
            server_ctx_size (int): Context window size of server
            chunk_word_count (int): Maximum number of words to chunk a document
            tokenizer_model_name (str): name of huggingface model to get
                tokenizer from 
        Returns:
            TextSplitChunker | ContextAwareChunker: Object of the appropriate
                chunker class for the provided filetype
        """
        documents = leaf_node[0]["documents"]
        assert type(documents) == list
        filepaths = leaf_node[0]["filepaths"]
        leaf_node_path = Path(leaf_node[0]["taxonomy_path"].replace("->", "/"))

        doc_dict = cls._split_docs_by_filetype(documents, filepaths)
        if len(doc_dict.keys()) > 1:
            raise ValueError(f"Received multiple document types")

        if FileTypes.MD in doc_dict:
            return TextSplitChunker(
                doc_dict[FileTypes.MD],
                server_ctx_size,
                chunk_word_count,
                output_dir,
            )

        if FileTypes.PDF in doc_dict:
            return ContextAwareChunker(
                doc_dict[FileTypes.PDF],
                filepaths,
                DEFAULT_TAXONOMY_PATH / leaf_node_path / "qna.yaml",
                output_dir, 
                tokenizer_model_name,
            )

    @staticmethod
    def _split_docs_by_filetype(documents: List[str], filepaths: List[Path]) -> defaultdict[any, List]:
        """Separate documents into lists based on their filetype.

        Currently, only Markdown and PDF are supported.
        Args:
            documents (List[str]): A list of the document contents as strings
            filepaths (List[Path]): Corresponding document filepaths
        Returns:
            defaultdict: Dictionary with either ".md" or ".pdf" as a key. 
                Markdown items contain document contents, PDF items contain
                paths to documents.
        """
        doc_dict = defaultdict(list)
        for doc, path in zip(documents, filepaths):
            if path.suffix == ".md":
                # append doc contents
                doc_dict[FileTypes.MD].append(doc)
            elif path.suffix == ".pdf":
                # append doc paths
                doc_dict[FileTypes.PDF].append(path)
            else:
                raise ValueError(
                    f"Received document of type .{path.suffix}, which is not a supported filetype"
                )
        return doc_dict


class TextSplitChunker(ChunkerBase):
    def __init__(
        self,
        document_contents: List | str,
        server_ctx_size: int,
        chunk_word_count: int,
        output_dir: Path,
    ):
        self.document_contents = document_contents
        self.server_ctx_size = server_ctx_size
        self.chunk_word_count = chunk_word_count
        self.output_dir = output_dir

    def chunk_documents(self) -> Dataset:
        """Naively chunk markdown documents based on the word count provided by the user.
        Returns:
            List[str]: List of chunked documents.
        """
        num_tokens_per_doc = self._num_tokens_from_words(self.chunk_word_count)
        if num_tokens_per_doc > int(self.server_ctx_size - 1024):
            raise ValueError(
                "Error: {}".format(
                    str(
                        f"Given word count ({self.chunk_word_count}) per doc will exceed the server context window size ({self.server_ctx_size})"
                    )
                )
            )
        if self.document_contents == []:
            return []

        # Placeholder for params
        content = []
        chunk_size = self._num_chars_from_tokens(num_tokens_per_doc)
        chunk_overlap = _DEFAULT_CHUNK_OVERLAP

        # Using Markdown as default, document-specific chunking will be implemented in separate pr.
        md_text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Determine file type for heuristics, default with markdown
        for doc in self.document_contents:
            # Use regex to remove unnecessary dashes in front of pipe characters in a markdown table.
            doc = re.sub(r"-{2,}\|", "-|", doc)
            # Remove unnecessary spaces in front of pipe characters in a markdown table.
            doc = re.sub(r"\  +\|", " |", doc)
            temp = md_text_splitter.create_documents([doc])
            content.extend([item.page_content for item in temp])

        return content

    @staticmethod
    def _num_tokens_from_words(num_words) -> int:
        return int(num_words * 1.3)  # 1 word ~ 1.3 token

    @staticmethod
    def _num_chars_from_tokens(num_tokens) -> int:
        return int(num_tokens * 4)  # 1 token ~ 4 English character


class ContextAwareChunker(ChunkerBase):
    def __init__(
        self,
        document_paths,
        filepaths,
        leaf_node_path,
        output_dir: Path,
        tokenizer_model_name=None,
    ):
        self.document_paths = document_paths
        self.filepaths = filepaths
        self.leaf_node_path = leaf_node_path
        self.output_dir = self._path_validator(output_dir)
        if tokenizer_model_name is None:
            self.tokenizer_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        else:
            self.tokenizer_model_name = tokenizer_model_name
        self.qna_yaml = self._load_qna_yaml(
            self._path_validator(leaf_node_path) if leaf_node_path else None
        )

        if tokenizer_model_name is None:
            tokenizer_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.tokenizer = self.create_tokenizer(tokenizer_model_name)

    def chunk_documents(self) -> Dataset:
        """Semantically chunk PDF documents.

        Returns:
            List: a list of chunks from the documents
        """
        if self.document_paths == []:
            return []

        model_artifacts_path = DocumentConverter.download_models_hf()
        converter = DocumentConverter(artifacts_path=model_artifacts_path)
        inputs = DocumentConversionInput.from_paths(self.filepaths)
        parsed_documents = converter.convert(inputs)

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

    def _load_qna_yaml(self, qna_yaml_path: Path) -> dict:
        """
        Load the qna YAML file.
        Args:
            qna_yaml_path (Path): Path to the knowledge qna YAML file.
        Returns:
            dict: Dictionary corresponding to knowledge qna YAML file.
        """
        if qna_yaml_path:
            with open(qna_yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

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

        file_name = json_fp.stem
        chunks = self.build_chunks_from_docling_json(
            data,
            max_token_per_chunk=500,
            tokenizer=self.tokenizer,
        )
        return self.fuse_texts(chunks, 200)

    def fuse_texts(self, text_list, short_length_threshold=100):
        """
        Fuse short texts with preceding longer texts if their word count is below the threshold.
        Args:
            text_list (list): List of text chunks to process.
            short_length_threshold (int): The word count threshold for determining short texts.
        Returns:
            list: List of fused texts.
        """
        fused_texts = []
        previous_long_text = ""

        for text in text_list:
            word_count = len(text.split())

            if word_count <= short_length_threshold and previous_long_text:
                # Append the short text to the last long text
                fused_texts[-1] += "\n\n" + text
            else:
                # This is a long text, so add it to the list and remember it
                fused_texts.append(text)
                previous_long_text = text

        return fused_texts


    def create_tokenizer(self, model_name: str):
        """
        Create a tokenizer instance from a pre-trained model or a local directory.

        Args:
            model_name (str): The name of the model or the path to the local directory.

        Returns:
            AutoTokenizer: The tokenizer instance.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Successfully loaded tokenizer from: {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_name}: {str(e)}")
            raise


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
        for i, row in enumerate(data):
            trow = []
            for j, cell in enumerate(row):
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
        table_text = self.generate_table_from_parsed_rep(json_book[parts[1]][int(parts[2])])
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
            if "prov" in book_element:
                prev_page_num = book_element["prov"][0]["page"]
                break
        for book_element in json_book["main-text"][idx:]:
            if "prov" in book_element:
                next_page_num = book_element["prov"][0]["page"]
                break
        if prev_page_num is not None and next_page_num is not None:
            if prev_page_num == next_page_num:
                return prev_page_num
            else:
                return next_page_num
        elif prev_page_num is not None:
            return prev_page_num
        elif next_page_num is not None:
            return next_page_num


    def build_chunks_from_docling_json(self, 
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
            elif book_element["type"] == "footnote":
                current_book_page_number = book_element["prov"][0]["page"]
            elif book_element["type"] in [
                "subtitle-level-1",
                "paragraph",
                "table",
                "title",
                "equation",
            ]:
                if book_element["type"] == "table":
                    current_book_page_number = self.get_table_page_number(json_book, idx)
                else:
                    current_book_page_number = book_element["prov"][0]["page"]
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
                        document_chunks.append("\n\n".join(current_buffer[:-1]))

                        if (
                            self.get_token_count(current_buffer[-1], tokenizer)
                            >= max_token_per_chunk
                        ):
                            document_chunks.append(current_buffer[-1])
                            current_buffer = []
                        else:
                            current_buffer = current_buffer[-1:]

                if book_element["type"] == "paragraph":
                    book_text = self.add_heading_formatting(book_text)
                elif book_element["type"] == "table":
                    book_text = self.get_table(json_book, book_element["$ref"])

                if "## References" in book_text or "## Acknowledgements" in book_text:
                    # For research papers we ignore everything after these sections
                    break
                current_buffer.append(book_text)

            try:
                prev_page_number = current_book_page_number
            except Exception as e:
                logger.error(f"Error processing book element: {book_element}, {str(e)}")

        if "\n\n".join(current_buffer) not in document_chunks:
            document_chunks.append("\n\n".join(current_buffer))
        return document_chunks

    def export_documents(self, converted_docs: Iterable[ConvertedDocument]):
        """Write converted documents to json files
        
        Check for successful conversions and write those to the docling artifacts directory.
        Returns:
            Path: path to directory with docling json artifacts
        """
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
                    fp.write(json.dumps(doc.render_as_dict()))

                # Export Markdown format:
                with (docling_artifacts_path / f"{doc_filename}.md").open("w") as fp:
                    fp.write(doc.render_as_markdown())
            else:
                logger.info(f"Document {doc.input.file} failed to convert.")
                failure_count += 1

        logger.info(
            f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
        )

        return docling_artifacts_path
