# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Iterable, List, Tuple
import json
import logging
import re

# Third Party
from datasets import Dataset, concatenate_datasets
from docling.datamodel.base_models import PipelineOptions
from docling.datamodel.document import ConvertedDocument, DocumentConversionInput
from docling.document_converter import ConversionStatus, DocumentConverter
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tabulate import tabulate
from transformers import AutoTokenizer
import yaml

logger = logging.getLogger(__name__)
DOC_FILEPATH = Path("~/.local/share/instructlab/datasets").expanduser()
_DEFAULT_CHUNK_OVERLAP = 100


def fuse_texts(text_list, short_length_threshold=100):
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


def create_tokenizer(model_name: str):
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


def get_token_count(text, tokenizer):
    """
    Get the number of tokens in a text using the provided tokenizer.
    Args:
        text (str): The text to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.
    Returns:
        int: Number of tokens.
    """
    return len(tokenizer.tokenize(text))


def add_heading_formatting(text):
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


def generate_table_from_parsed_rep(item):
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


def get_table(json_book, table_ref):
    """
    Retrieve a table from a document based on a reference string.
    Args:
        json_book (dict): JSON representation of the document.
        table_ref (str): Reference path to the table within the document.
    Returns:
        str: Formatted table string.
    """
    parts = table_ref.split("/")
    table_text = generate_table_from_parsed_rep(json_book[parts[1]][int(parts[2])])
    return table_text


def get_table_page_number(json_book, idx):
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


def build_chunks_from_docling_json(
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
                current_book_page_number = get_table_page_number(json_book, idx)
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
                    get_token_count("\n\n".join(current_buffer), tokenizer)
                    >= max_token_per_chunk
                    and len(current_buffer) > 1
                ):
                    document_chunks.append("\n\n".join(current_buffer[:-1]))

                    if (
                        get_token_count(current_buffer[-1], tokenizer)
                        >= max_token_per_chunk
                    ):
                        document_chunks.append(current_buffer[-1])
                        current_buffer = []
                    else:
                        current_buffer = current_buffer[-1:]

            if book_element["type"] == "paragraph":
                book_text = add_heading_formatting(book_text)
            elif book_element["type"] == "table":
                book_text = get_table(json_book, book_element["$ref"])

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


def safe_concatenate_datasets(datasets: list):
    """
    Concatenate datasets safely, ignoring any datasets that are None or empty.
    """
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    if not filtered_datasets:
        return None

    return concatenate_datasets(filtered_datasets)


class DocProcessor:
    def __init__(
        self,
        parsed_doc_dir: Path,
        qna_yaml_path: Path = None,
        tokenizer_model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ):
        """
        Initialize the DocProcessor.
        Args:
            parsed_doc_dir (Path): Directory containing parsed docling JSON files.
            tokenizer_model_name (str): The name of the model or path to the tokenizer.
            qna_yaml_path (Path, optional): Path to the qna YAML file.
        """
        self.parsed_doc_dir = self._path_validator(parsed_doc_dir)
        self.qna_yaml = self._load_qna_yaml(
            self._path_validator(qna_yaml_path) if qna_yaml_path else None
        )
        self.docling_jsons = list(self.parsed_doc_dir.glob("*.json"))

        if tokenizer_model_name is None:
            tokenizer_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.tokenizer = create_tokenizer(tokenizer_model_name)
        print(f"""THIS IS KHALED: INIT DOCPROCESSOR:
              {self.parsed_doc_dir=}, 
              {self.qna_yaml=}, 
              {self.docling_jsons=},
              {self.tokenizer}
        """)

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
            Dataset: Dataset object.
        """
        logger.info(f"Processing parsed docling json file: {json_fp}")
        print(f"THIS IS KHALED: {json_fp=}")
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_name = json_fp.stem
        chunks = build_chunks_from_docling_json(
            data,
            max_token_per_chunk=500,
            tokenizer=self.tokenizer,
        )
        chunks = fuse_texts(chunks, 200)
        return Dataset.from_dict(
            {
                "document": chunks,
                "document_outline": [self.qna_yaml.get("document_outline", "")]
                * len(chunks),
                "document_title": [file_name] * len(chunks),
                "domain": [self.qna_yaml.get("domain", "")] * len(chunks),
            }
        )

    def _add_icls(self, chunked_document: Dataset) -> Dataset:
        """
        Add the ICLS label to the dataset.
        Args:
            chunked_document (Dataset): Dataset object.
        Returns:
            Dataset: Dataset object with ICLS label.
        """
        icl = self.qna_yaml.get("seed_examples", [])
        chunked_document_all_icl = []
        for icl_ in icl:
            chunked_document_all_icl.append(
                chunked_document.map(
                    lambda x: {
                        "icl_document": icl_["context"],
                        "icl_query_1": icl_["questions_and_answers"][0]["question"],
                        "icl_response_1": icl_["questions_and_answers"][0]["answer"],
                        "icl_query_2": icl_["questions_and_answers"][1]["question"],
                        "icl_response_2": icl_["questions_and_answers"][1]["answer"],
                        "icl_query_3": icl_["questions_and_answers"][2]["question"],
                        "icl_response_3": icl_["questions_and_answers"][2]["answer"],
                    }
                )
            )
        chunked_document_all_icl = safe_concatenate_datasets(chunked_document_all_icl)
        chunked_document_all_icl = chunked_document_all_icl.map(
            lambda x: {
                "chunks": chunk_markdowns(
                    [x["document"]], server_ctx_size=4096, chunk_word_count=1024
                )
                if get_token_count(x["document"], self.tokenizer) > 1024
                else [x["document"]]
            }
        )
        df = chunked_document_all_icl.to_pandas()
        df_exploded = df.explode("chunks").reset_index(drop=True)
        new_ds = Dataset.from_pandas(df_exploded)
        new_ds = new_ds.remove_columns("document").rename_columns(
            {"chunks": "document"}
        )

        # Only keep document greater than 100 tokens
        new_ds = new_ds.filter(
            lambda x: get_token_count(x["document"], self.tokenizer) > 100
        )
        return new_ds

    def get_processed_dataset(self) -> Dataset:
        """
        Process all the parsed docling json files and return a dataset.
        Returns:
            Dataset: Dataset object.
        """
        datasets = []
        for json_fp in self.docling_jsons:
            chunk_ds = self._process_parsed_docling_json(json_fp)
            chunk_ds_with_icls = self._add_icls(chunk_ds)
            datasets.append(chunk_ds_with_icls)
        return safe_concatenate_datasets(datasets)


def _num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def _num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def _split_docs_by_filetype(doc_tuples: List[Tuple[str, Path]]):
    """Separate documents into lists based on their filetype.

    Currently, only Markdown and PDF are supported.
    Args:
        TODO
    Returns:
        TODO
         (List[str], List[str]): Lists of Markdown and PDF documents, respectively
    """
    md_docs = []
    pdf_docs = []

    for doc, path in doc_tuples:
        if path.suffix == ".md":
            md_docs.append(doc)
        elif path.suffix == ".pdf":
            pdf_docs.append(path)
        else:
            raise ValueError(
                f"Received document of type .{path.suffix}, which is not a supported filetype"
            )

    return md_docs, pdf_docs


def chunk_documents(
    leaf_node, server_ctx_size, chunk_word_count, output_dir, model_name
) -> List[str]:
    """
    Iterate over the documents of a leaf node and split them into chunks based on the word count provided by the user.
    Args:
        leaf_node (TODO): TODO CHANGE List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """
    filepaths = leaf_node[0]["filepaths"]
    documents = leaf_node[0]["documents"]
    leaf_node_path = Path(leaf_node[0]["taxonomy_path"].replace("->", "/"))

    # Check for single document
    if isinstance(documents, str):
        documents = [documents]
        logger.info(
            "Converted single string into list of strings. Assumed the string passed in is the document. Normally, chunk_document() should take a list as input."
        )

    assert type(documents) == list

    md_docs, pdf_docs = _split_docs_by_filetype(zip(documents, filepaths))

    print(f"THIS IS KHALED: {md_docs=}\n {pdf_docs=}")
    chunked_mds = chunk_markdowns(md_docs, server_ctx_size, chunk_word_count)
    chunked_pdfs = chunk_pdfs(pdf_docs, filepaths, leaf_node_path, model_name)

    return chunked_mds + chunked_pdfs


def chunk_markdowns(
    documents: List | str, server_ctx_size, chunk_word_count
) -> List[str]:
    """Naively chunk markdown documents based on the word count provided by the user.
    Args:
        documents (list): List of markdown documents.
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """
    num_tokens_per_doc = _num_tokens_from_words(chunk_word_count)
    if num_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )
    if documents == []:
        return []

    # Placeholder for params
    content = []
    chunk_size = _num_chars_from_tokens(num_tokens_per_doc)
    chunk_overlap = _DEFAULT_CHUNK_OVERLAP

    # Using Markdown as default, document-specific chunking will be implemented in separate pr.
    md_text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Determine file type for heuristics, default with markdown
    for doc in documents:
        # Use regex to remove unnecessary dashes in front of pipe characters in a markdown table.
        doc = re.sub(r"-{2,}\|", "-|", doc)
        # Remove unnecessary spaces in front of pipe characters in a markdown table.
        doc = re.sub(r"\  +\|", " |", doc)
        temp = md_text_splitter.create_documents([doc])
        content.extend([item.page_content for item in temp])
    return content


def chunk_pdfs(
    pdf_docs: List, filepaths: List, leaf_node_path: Path, model_name: str):
    """Semantically chunk PDF documents.

    TODO
    """
    if pdf_docs == []:
        return []

    print(f"""THIS IS KHALED: CHUNKING PDF DOCS
        {pdf_docs[0]=}
    """)
    model_artifacts_path = DocumentConverter.download_models_hf()
    converter = DocumentConverter(artifacts_path=model_artifacts_path)
    inputs = DocumentConversionInput.from_paths(filepaths)
    parsed_pdfs = converter.convert(inputs)

    docling_artifacts_path = export_documents(parsed_pdfs)
    print(f"THIS IS KHALED {docling_artifacts_path=}")

    dp = DocProcessor(
        parsed_doc_dir=str(docling_artifacts_path),
        tokenizer_model_name=model_name,
        qna_yaml_path=Path("~/.local/share/instructlab/taxonomy").expanduser()
        / leaf_node_path
        / "qna.yaml",
    )

    chunked_pdfs = dp.get_processed_dataset()
    for k, v in chunked_pdfs.to_dict().items():
        print(f"{k=}: ; {type(v)=}\n")
        print(f"{v[:5]=}\n\n")
    print(f"THIS IS KHALED: {type(chunked_pdfs)=}")
    print(f"THIS IS KHALED: {chunked_pdfs.shape=}")

    raise Exception('STOPPING')
    return chunked_pdfs


def export_documents(converted_docs: Iterable[ConvertedDocument]):
    """TODO
    
    """
    docling_artifacts_path = DOC_FILEPATH / "docling-artifacts"
    docling_artifacts_path.mkdir(parents=True, exist_ok=True)
    print(f"THIS IS KHALED IN EXPORT DOCUMENTS: {docling_artifacts_path}")

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
