# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import tempfile

# Third Party
import pytest

# First Party
from instructlab.sdg.utils.chunkers import (
    ContextAwareChunker,
    DocumentChunker,
    FileTypes,
    TextSplitChunker,
)

# Local
from .testdata import testdata


@pytest.fixture
def documents_dir():
    return Path(__file__).parent / "testdata" / "sample_documents"


@pytest.mark.parametrize(
    "filepaths, chunker_type",
    [
        ([Path("document.md")], TextSplitChunker),
        ([Path("document.pdf")], ContextAwareChunker),
    ],
)
def test_chunker_factory(filepaths, chunker_type, documents_dir):
    """Test that the DocumentChunker factory class returns the proper Chunker type"""
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "taxonomy_path": "",
            "filepaths": filepaths,
        }
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        chunker = DocumentChunker(
            leaf_node=leaf_node,
            taxonomy_path=documents_dir,
            output_dir=temp_dir,
            tokenizer_model_name="instructlab/merlinite-7b-lab",
        )
        assert isinstance(chunker, chunker_type)


def test_chunker_factory_unsupported_filetype(documents_dir):
    """Test that the DocumentChunker factory class fails when provided an unsupported document"""
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "taxonomy_path": "",
            "filepaths": [Path("document.jpg")],
        }
    ]
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = DocumentChunker(
                leaf_node=leaf_node,
                taxonomy_path=documents_dir,
                output_dir=temp_dir,
                tokenizer_model_name="instructlab/merlinite-7b-lab",
            )


def test_chunker_factory_empty_filetype(documents_dir):
    """Test that the DocumentChunker factory class fails when provided no document"""
    leaf_node = [
        {
            "documents": [],
            "taxonomy_path": "",
            "filepaths": [],
        }
    ]
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = DocumentChunker(
                leaf_node=leaf_node,
                taxonomy_path=documents_dir,
                output_dir=temp_dir,
                tokenizer_model_name="instructlab/merlinite-7b-lab",
            )
