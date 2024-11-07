# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List
import logging
import re

# Third Party
from transformers import AutoTokenizer
from instructlab.sdg.utils.models import get_model_family

# Local
from instructlab.sdg.utils import GenerateException

_DEFAULT_CHUNK_OVERLAP = 100

logger = logging.getLogger(__name__)


def _num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def _num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def chunk_document(documents: List, server_ctx_size, chunk_word_count, model_path: str) -> List[str]:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (list): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
        model_path (str): Path to the downloaded teacher model.
    Returns:
         List[str]: List of chunked documents.
    """

    # Checks for input type error
    if isinstance(documents, str):
        documents = [documents]
        logger.info(
            "Converted single string into a list of string. Assumed the string passed in is the document. Normally, chunk_document() should take a list as input."
        )
    elif not isinstance(documents, list):
        raise TypeError(
            "Expected: documents to be a list, but got {}".format(type(documents))
        )

    no_tokens_per_doc = _num_tokens_from_words(chunk_word_count)
    if no_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )

    # Load the tokenizer from the local model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise GenerateException(f"Failed to load tokenizer from {model_path}: {str(e)}")

    # Determine the model family
    model_family = get_model_family(None, model_path)
    # Placeholder for params
    content = []
    chunk_size = no_tokens_per_doc * 4  # Approximate character count
    chunk_overlap = _DEFAULT_CHUNK_OVERLAP

    # Using Markdown as default, document-specific chunking will be implemented in separate pr.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(tokenizer.encode(x))
    )
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,



    for docs in documents:
        # Use regex to remove unnecessary dashes in front of pipe characters in a markdown table.
        docs = re.sub(r"-{2,}\|", "-|", docs)
        # Remove unnecessary spaces in front of pipe characters in a markdown table.
        docs = re.sub(r"\  +\|", " |", docs)
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])
    return content
