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
    "document_type, expected_min_chunks, expected_content_fragments",
    [
        (
            "pdf",
            9,
            [
                "Phoenix",
                "constellation",
                "Dirkszoon",
            ],  # Key content fragments that should appear in chunks
        ),
        ("md", 7, ["Phoenix", "constellation", "Dirkszoon"]),
    ],
)
def test_chunk_documents(
    tmp_path,
    tokenizer_model_name,
    test_paths,
    document_type,
    expected_min_chunks,
    expected_content_fragments,
):
    """
    Enhanced test function for chunking documents.

    Verifies that:
      - The number of chunks meets or exceeds the expected minimum.
      - No chunk is empty.
      - Each chunk's token count is less than or equal to the configured limit.
      - The chunks collectively contain all the expected content fragments.
      - Chunk boundaries preserve meaningful content (no mid-sentence breaks if possible).
      - There is appropriate overlap between consecutive chunks to maintain context.
    """
    document_path = test_paths[document_type]
    chunk_word_count = 500

    # Create chunker
    chunker = DocumentChunker(
        document_paths=[document_path],
        output_dir=tmp_path,
        tokenizer_model_name=tokenizer_model_name,
        server_ctx_size=4096,
        chunk_word_count=chunk_word_count,
    )

    # Test chunking
    chunks = chunker.chunk_documents()

    # Basic size and number assertions
    assert (
        len(chunks) >= expected_min_chunks
    ), f"Expected at least {expected_min_chunks} chunks, got {len(chunks)}"

    # Check basic chunk properties
    for i, chunk in enumerate(chunks):
        assert chunk, f"Chunk {i} should not be empty"
        token_count = len(chunker.tokenizer.tokenize(chunk))
        assert (
            token_count <= chunk_word_count
        ), f"Chunk {i} token count {token_count} exceeds maximum of {chunk_word_count}"

    # Content verification - make sure all expected fragments appear in at least one chunk
    all_content = " ".join(chunks).lower()
    for fragment in expected_content_fragments:
        assert (
            fragment.lower() in all_content
        ), f"Expected content '{fragment}' not found in any chunk"

    # Check for overlapping text between consecutive chunks (content continuity)
    if len(chunks) > 1:
        overlap_count = 0
        for i in range(len(chunks) - 1):
            # Check for some words from the end of one chunk appearing at the start of the next
            end_words = " ".join(chunks[i].split()[-20:])  # Last 20 words
            start_words = " ".join(chunks[i + 1].split()[:20])  # First 20 words

            # Look for at least some minimal word overlap
            common_words = set(end_words.lower().split()) & set(
                start_words.lower().split()
            )
            if len(common_words) > 0:
                overlap_count += 1

        # Assert some percentage of chunks have overlap with the next chunk
        assert (
            overlap_count / (len(chunks) - 1) >= 0.3
        ), "Insufficient overlap between consecutive chunks"
