# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
from docling.datamodel.base_models import PipelineOptions
from docling.datamodel.document import ConvertedDocument, DocumentConversionInput
from docling.document_converter import ConversionStatus, DocumentConverter
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


# def read_document_contents(document_path: Path):
#     # TODO
#     if document_path.suffix == ".md":
#         pass
#     if document_path.suffix == ".pdf":
#         pass


def build_leaf_node(document_paths: str | list):
    # TODO maybe check for directory
    if not isinstance(document_paths, list):
        document_paths = [document_paths]

    return [
        {
            "taxonomy_path": "",  # TODO
            "filepaths": document_paths,
            "documents": [read_document_contents(d) for d in document_paths],
        }
    ]


@pytest.fixture
def documents_dir():
    return Path(__file__) / "testdata" / "sample_documents"


@pytest.mark.parametrize(
    "filepaths, chunker_type",
    [
        ([Path("document.md")], TextSplitChunker),
        ([Path("document.pdf")], ContextAwareChunker),
    ],
)
def test_chunker_factory(filepaths, chunker_type):
    """Test that the DocumentChunker factory class returns the proper Chunker type"""
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "taxonomy_path": "sample/path",
            "filepaths": filepaths,
        }
    ]
    chunker = DocumentChunker(leaf_node=leaf_node)
    assert isinstance(chunker, chunker_type)


def test_chunker_factory_unsupported_filetype():
    """Test that the DocumentChunker factory class fails when provided an unsupported document"""
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "taxonomy_path": "sample/path",
            "filepaths": [Path("document.jpg")],
        }
    ]
    with pytest.raises(ValueError):
        _ = DocumentChunker(leaf_node=leaf_node)


# class TestTextSplitChunker():
#     @pytest.fixture
#     def chunker():
#         pass

#     def test_chunk_documents():
#         pass

#     def test_chunk_docs_wc_exceeds_ctx_window(self):
#         with pytest.raises(ValueError) as exc:
#             chunking.chunk_document(
#                 documents=testdata.documents,
#                 chunk_word_count=1000,
#                 server_ctx_size=1034,
#             )
#         assert (
#             "Given word count (1000) per doc will exceed the server context window size (1034)"
#             in str(exc.value)
#         )

#     def test_chunk_docs_long_lines(self):
#         # TODO see if this is applicable to context-aware
#         chunk_words = 50
#         chunks = chunking.chunk_document(
#             documents=testdata.long_line_documents,
#             chunk_word_count=chunk_words,
#             server_ctx_size=4096,
#         )
#         max_tokens = chunking._num_tokens_from_words(chunk_words)
#         max_chars = chunking._num_chars_from_tokens(max_tokens)
#         max_chars += chunking._DEFAULT_CHUNK_OVERLAP  # add in the chunk overlap
#         max_chars += 50  # and a bit extra for some really long words
#         for chunk in chunks:
#             assert len(chunk) <= max_chars

#     def test_chunk_docs_chunk_overlap_error(self):
#         # TODO check if applicable to context-aware
#         with pytest.raises(ValueError) as exc:
#             chunking.chunk_document(
#                 documents=testdata.documents,
#                 chunk_word_count=5,
#                 server_ctx_size=1034,
#             )
#         assert (
#             "Got a larger chunk overlap (100) than chunk size (24), should be smaller"
#             in str(exc.value)
#         )


# class TestContextAwareChunker():
#     @pytest.fixture
#     def chunker(documents_dir):
#         pass

#     def test_chunk_documents():
#         pass

#     def test_path_validator():
#         pass

#     def test_load_qna_yaml():
#         pass

#     def test_process_parsed_docling_json():
#         pass

#     def test_fuse_texts():
#         pass

#     def test_create_tokenizer():
#         pass

#     def test_get_token_count():
#         pass

#     def test_add_heading_formatting():
#         pass

#     def test_generate_table_from_parsed_rep():
#         pass

#     def test_get_table():
#         pass

#     def test_get_table_page_number():
#         pass

#     def test_build_chunks_from_docling_json():
#         pass

#     def test_export_document():
#         pass

