# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import tempfile

# Third Party
from docling.datamodel.pipeline_options import EasyOcrOptions, TesseractOcrOptions
import pytest

# First Party
from instructlab.sdg.utils.chunkers import (
    ContextAwareChunker,
    DocumentChunker,
    FileTypes,
    TextSplitChunker,
    resolve_ocr_options,
)

# Local
from .testdata import testdata

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


@pytest.fixture
def documents_dir():
    return Path(TEST_DATA_DIR) / "sample_documents"


@pytest.fixture
def tokenizer_model_name():
    return os.path.join(TEST_DATA_DIR, "models/instructlab/granite-7b-lab")


@pytest.mark.parametrize(
    "filepaths, chunker_type",
    [
        ([Path("document.md")], TextSplitChunker),
        ([Path("document.pdf")], ContextAwareChunker),
    ],
)
def test_chunker_factory(filepaths, chunker_type, documents_dir, tokenizer_model_name):
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
            tokenizer_model_name=tokenizer_model_name,
        )
        assert isinstance(chunker, chunker_type)


def test_chunker_factory_unsupported_filetype(documents_dir, tokenizer_model_name):
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
                tokenizer_model_name=tokenizer_model_name,
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
                tokenizer_model_name=tokenizer_model_name,
            )


def test_resolve_ocr_options_is_not_none():
    """
    Test that resolve_ocr_options does not return None, which means it
    found a valid OCR library on the machine running this test
    """
    ocr_options = resolve_ocr_options()
    assert ocr_options is not None


@patch("docling.models.tesseract_ocr_model.TesseractOcrModel")
def test_resolve_ocr_options_prefers_tessserocr(mock_tesseract):
    """
    Ensure resolve_ocr_options defaults to tesserocr if we're able
    to load that library without error.
    """
    mock_tesseract.return_value = MagicMock()
    ocr_options = resolve_ocr_options()
    assert isinstance(ocr_options, TesseractOcrOptions)


@patch("docling.models.tesseract_ocr_model.TesseractOcrModel")
def test_resolve_ocr_options_falls_back_to_easyocr(mock_tesseract):
    """
    Ensure resolve_ocr_options falls back to easyocr if we cannot
    load tesserocr.
    """
    mock_tesseract.side_effect = ImportError("mock import error")
    ocr_options = resolve_ocr_options()
    assert isinstance(ocr_options, EasyOcrOptions)


@patch("docling.models.tesseract_ocr_model.TesseractOcrModel")
@patch("docling.models.easyocr_model.EasyOcrModel")
@patch("logging.Logger.error")
def test_resolve_ocr_options_none_found_logs_error(
    mock_logger, mock_easyocr, mock_tesseract
):
    """
    If we cannot load tesserocr or easyocr, ensure
    resolve_ocr_options logs an error so that users are aware optical
    character recognition in PDFs will be disabled.
    """
    mock_tesseract.side_effect = ImportError("mock import error")
    mock_easyocr.side_effect = ImportError("mock import error")
    ocr_options = resolve_ocr_options()
    assert ocr_options is None
    mock_logger.assert_called()


def test_create_tokenizer(tokenizer_model_name):
    ContextAwareChunker.create_tokenizer(tokenizer_model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "models/invalid_gguf.gguf",
        "models/invalid_safetensors_dir/",
        "bad_path",
    ],
)
def test_invalid_tokenizer(model_name):
    model_path = os.path.join(TEST_DATA_DIR, model_name)
    with pytest.raises(ValueError):
        ContextAwareChunker.create_tokenizer(model_path)
