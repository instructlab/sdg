# Standard
from pathlib import Path
import os

# Third Party
import pytest

# First Party
from instructlab.sdg.utils.chunkers import DocumentChunker

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "testdata")


@pytest.fixture
def tokenizer_model_name():
    return os.path.join(TEST_DATA_DIR, "models/instructlab/granite-7b-lab")


def test_chunk_pdf(tmp_path, tokenizer_model_name):
    pdf_path = Path(os.path.join(TEST_DATA_DIR, "sample_documents", "phoenix.pdf"))
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "filepaths": [pdf_path],
            "taxonomy_path": "knowledge",
        }
    ]
    chunker = DocumentChunker(
        leaf_node=leaf_node,
        taxonomy_path=tmp_path,
        output_dir=tmp_path,
        server_ctx_size=4096,
        chunk_word_count=500,
        tokenizer_model_name=tokenizer_model_name,
    )
    chunks = chunker.chunk_documents()
    assert len(chunks) > 9
    assert "Phoenix is a minor constellation" in chunks[0]
    for chunk in chunks:
        # inexact sanity-checking of chunk max length
        assert len(chunk) < 2500


def test_chunk_md(tmp_path, tokenizer_model_name):
    markdown_path = Path(os.path.join(TEST_DATA_DIR, "sample_documents", "phoenix.md"))
    leaf_node = [
        {
            "documents": [markdown_path.read_text(encoding="utf-8")],
            "filepaths": [markdown_path],
            "taxonomy_path": "knowledge",
        }
    ]
    chunker = DocumentChunker(
        leaf_node=leaf_node,
        taxonomy_path=tmp_path,
        output_dir=tmp_path,
        server_ctx_size=4096,
        chunk_word_count=500,
        tokenizer_model_name=tokenizer_model_name,
    )
    chunks = chunker.chunk_documents()
    assert len(chunks) > 7
    for chunk in chunks:
        # inexact sanity-checking of chunk max length
        assert len(chunk) < 2500
