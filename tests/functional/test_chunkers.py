# Standard
from pathlib import Path
import os

# Third Party
import pytest

# First Party
from instructlab.sdg.utils.chunkers import DocumentChunker

# Constants for Test Directory and Test Documents
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "testdata")
TEST_DOCUMENTS = {
    "pdf": "sample_documents/phoenix.pdf",
    "md": "sample_documents/phoenix.md",
}


@pytest.fixture(scope="module")
def test_paths():
    """Fixture to return paths to test documents."""
    return {
        doc_type: Path(os.path.join(TEST_DATA_DIR, path))
        for doc_type, path in TEST_DOCUMENTS.items()
    }


@pytest.fixture
def tokenizer_model_name():
    """Fixture to return the path to the tokenizer model."""
    return os.path.join(TEST_DATA_DIR, "models/instructlab/granite-7b-lab")


@pytest.mark.parametrize(
    "doc_type, expected_chunks, contains_text",
    [
        ("pdf", 9, "Phoenix is a minor constellation"),
        ("md", 7, None),  # Assuming there's no specific text to check in Markdown
    ],
)
def test_chunk_documents(
    tmp_path, tokenizer_model_name, test_paths, doc_type, expected_chunks, contains_text
):
    """
    Generalized test function for chunking documents.
    """
    document_path = test_paths[doc_type]
    chunker = DocumentChunker(
        document_paths=[document_path],
        output_dir=tmp_path,
        tokenizer_model_name=tokenizer_model_name,
        server_ctx_size=4096,
        chunk_word_count=500,
    )
    chunks = chunker.chunk_documents()
    assert len(chunks) > expected_chunks
    if contains_text:
        # Normalize spaces and remove newlines for more flexible text comparison
        normalized_chunk = " ".join(chunks[0].replace("\n", " ").split())
        normalized_text = " ".join(contains_text.split())
        assert normalized_text in normalized_chunk
    for chunk in chunks:
        assert len(chunk) < 2500
