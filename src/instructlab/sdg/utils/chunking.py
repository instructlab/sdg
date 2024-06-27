# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List

# Third Party
from langchain_text_splitters import RecursiveCharacterTextSplitter

_DEFAULT_CHUNK_OVERLAP = 100


def _num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def _num_tokens_from_chars(num_chars) -> int:
    return int(num_chars / 4)  # 1 token ~ 4 English character


def max_seed_example_tokens(server_ctx_size, prompt_num_chars) -> int:
    """
    Estimates the maximum number of tokens any seed example can have based
    on the server context size and number of characters in the selected prompt.

    A lot has to fit into the given server context size:
      - The prompt itself, which can vary in size a bit based on model family and knowledge vs skill
      - Two seed examples, which we append to the prompt template.
      - A knowledge document chunk, if this is a knowledge example.
      - The generated completion, which can vary substantially in length.

    This is an attempt to roughly estimate the maximum size any seed example
    (question + answer + context values from the yaml) should be to even have
    a hope of not often exceeding the server's maximum context size.

    NOTE: This does not take into account knowledge document chunks. It's meant
    to calculate the maximum size that any seed example should be, whether knowledge
    or skill. Knowledge seed examples will want to stay well below this limit.

    NOTE: This is a very simplistic calculation, and examples with lots of numbers
    or punctuation may have quite a different token count than the estimates here,
    depending on the model (and thus tokenizer) in use. That's ok, as it's only
    meant to be a rough estimate.

    Args:
        server_ctx_size (int): Size of the server context, in tokens.
        prompt_num_chars (int): Number of characters in the prompt (not including the examples)
    """
    # Ensure we have at least 1024 tokens available for a response.
    max_seed_tokens = server_ctx_size - 1024
    # Subtract the number of tokens in our prompt template
    max_seed_tokens = max_seed_tokens - _num_tokens_from_chars(prompt_num_chars)
    # Divide number of characters by 2, since we insert 2 examples
    max_seed_tokens = int(max_seed_tokens / 2)
    return max_seed_tokens


def chunk_document(documents: List, server_ctx_size, chunk_word_count) -> List[str]:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (dict): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """
    no_tokens_per_doc = _num_tokens_from_words(chunk_word_count)
    if no_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )
    content = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=num_chars_from_tokens(no_tokens_per_doc),
        chunk_overlap=_DEFAULT_CHUNK_OVERLAP,
    )

    for docs in documents:
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])

    return content
