# Standard
from pathlib import Path
import os

# First Party
from instructlab.sdg.utils.chunkers import DocumentChunker

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")

# TODO: Apparently we don't really need any contents in the qna.yaml?
knowledge_qna = """
version: 3
domain: astronomy
"""


def test_chunk_pdf(tmp_path):
    qna_dir = os.path.join(tmp_path, "knowledge")
    os.makedirs(qna_dir)
    with open(os.path.join(qna_dir, "qna.yaml"), "w", encoding="utf-8") as f:
        f.write(knowledge_qna)

    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "filepaths": [Path(os.path.join(TEST_DATA_DIR, "phoenix.pdf"))],
            "taxonomy_path": "knowledge",
        }
    ]
    chunker = DocumentChunker(
        leaf_node=leaf_node,
        taxonomy_path=tmp_path,
        output_dir=tmp_path,
        server_ctx_size=4096,
        chunk_word_count=500,
        tokenizer_model_name="instructlab/merlinite-7b-lab",
    )
    chunks = chunker.chunk_documents()
    assert len(chunks) > 9
    assert "Phoenix is a minor constellation" in chunks[0]
    for chunk in chunks:
        # inexact sanity-checking of chunk max length
        assert len(chunk) < 2500


def test_chunk_md(tmp_path):
    qna_dir = os.path.join(tmp_path, "knowledge")
    os.makedirs(qna_dir)
    with open(os.path.join(qna_dir, "qna.yaml"), "w", encoding="utf-8") as f:
        f.write(knowledge_qna)

    markdown_path = Path(os.path.join(TEST_DATA_DIR, "phoenix.md"))
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
        tokenizer_model_name="instructlab/merlinite-7b-lab",
    )
    chunks = chunker.chunk_documents()
    assert len(chunks) > 7
    for chunk in chunks:
        # inexact sanity-checking of chunk max length
        assert len(chunk) < 2500
