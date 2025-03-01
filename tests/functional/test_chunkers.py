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


@pytest.fixture(scope="module")
def tokenizer_model_name():
    """Fixture to return the path to the tokenizer model."""
    return os.path.join(TEST_DATA_DIR, "models/instructlab/granite-7b-lab")


@pytest.mark.parametrize(
    "document_type, expected_chunks",
    [
        ("pdf", 9),
        ("md", 7),
    ],
)
def test_chunk_documents(
    tmp_path, tokenizer_model_name, test_paths, document_type, expected_chunks
):
    """
    Generalized test function for chunking documents.

    Verifies that:
      - The number of chunks is greater than the expected minimum.
      - No chunk is empty.
      - Each chunk's length is less than 2500 characters.
    """
    document_path = test_paths[document_type]
    chunker = DocumentChunker(
        document_paths=[document_path],
        output_dir=tmp_path,
        tokenizer_model_name=tokenizer_model_name,
        server_ctx_size=4096,
        chunk_word_count=500,
    )
    chunks = chunker.chunk_documents()

    # Check that we have more chunks than expected.
    assert (
        len(chunks) > expected_chunks
    ), f"Expected more than {expected_chunks} chunks, got {len(chunks)}"

    # Check that no chunk is empty and each chunk's length is within the allowed limit.
    for chunk in chunks:
        assert chunk, "Chunk should not be empty"
        assert len(chunk) < 2500, f"Chunk length {len(chunk)} exceeds maximum allowed"
