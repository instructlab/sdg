# Standard
from pathlib import Path
import os
import sys
import logging

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
    is_ci = bool(os.getenv("CI"))  # Convert to boolean explicitly
    is_macos = sys.platform == "darwin"
    logger.debug(f"CI environment: {is_ci}, Platform: {sys.platform}")

    if is_ci and is_macos:
        logger.info("Forcing CPU usage on macOS CI environment")
        # Disable MPS
        torch.backends.mps.enabled = False
        
        # Force CPU as default device
        os.environ["PYTORCH_DEVICE"] = "cpu"
        torch.set_default_device("cpu")
        
        # Ensure map_location is CPU for torch.load operations
        def cpu_loader(storage, *args, **kwargs):
            return storage.cpu()

        torch.load = lambda f, *a, **kw: torch._load_global(
            f, map_location=cpu_loader, *a, **kw
        )

        # Additional MPS disabling
        os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "cpu"
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    logger.debug(f"After setup - MPS enabled: {torch.backends.mps.enabled}")
    logger.debug(f"Current device: {os.getenv('PYTORCH_DEVICE', 'not set')}")
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
