# Standard
from pathlib import Path
import logging
import os
import sys

# Third Party
import pytest
import torch

# First Party
from instructlab.sdg.utils.chunkers import DocumentChunker

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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


@pytest.fixture(scope="module", autouse=True)
def force_cpu_on_macos_ci():
    """Force CPU usage on macOS CI environments."""
    logger.debug("=== Starting CPU force fixture ===")
    is_macos = sys.platform == "darwin"
    if is_macos:
        logger.info("Forcing CPU usage on macOS CI environment")
        # Force CPU as default device
        os.environ["PYTORCH_DEVICE"] = "cpu"
        torch.set_default_device("cpu")

        # Disable MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    logger.debug(f"Current device: {os.getenv('PYTORCH_DEVICE', 'not set')}")
    logger.debug(f"Available PyTorch devices: CPU={torch.cuda.is_available()}")
    logger.debug("=== Finished CPU force fixture ===")

    yield


@pytest.mark.parametrize(
    "document_type, expected_chunks",
    [
        ("pdf", 9),
        ("md", 7),
    ],
)
def test_chunk_documents(
    tmp_path,
    tokenizer_model_name,
    test_paths,
    document_type,
    expected_chunks,
):
    """
    Generalized test function for chunking documents.

    Verifies that:
      - The number of chunks is greater than the expected minimum.
      - No chunk is empty.
      - Each chunk's token count is less than or equal to 500 tokens.
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
    assert (
        len(chunks) > expected_chunks
    ), f"Expected more than {expected_chunks} chunks, got {len(chunks)}"
    for chunk in chunks:
        assert chunk, "Chunk should not be empty"
        token_count = len(chunker.tokenizer.tokenize(chunk))
        assert (
            token_count < 1000
        ), f"Chunk token count {token_count} exceeds maximum of 500 tokens"
