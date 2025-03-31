# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import os
import tempfile

# Third Party
from docling.datamodel.pipeline_options import EasyOcrOptions, TesseractOcrOptions
import git
import pytest

# First Party
from instructlab.sdg.utils.chunkers import DocumentChunker, resolve_ocr_options
from instructlab.sdg.utils.taxonomy import _get_documents

# Local
from .testdata import testdata

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


@pytest.fixture
def documents_dir():
    return Path(TEST_DATA_DIR) / "sample_documents"


@pytest.fixture
def tokenizer_model_name():
    return os.path.join(TEST_DATA_DIR, "models/instructlab/granite-7b-lab")


def test_init_document_chunker_unsupported_filetype(
    documents_dir, tokenizer_model_name
):
    """Test that the DocumentChunker factory class fails when provided an unsupported document"""
    document_paths = [documents_dir / "document.jpg"]
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = DocumentChunker(
                document_paths=document_paths,
                output_dir=temp_dir,
                tokenizer_model_name=tokenizer_model_name,
            )


def test_chunker_factory_empty_document_paths(tokenizer_model_name):
    """Test that the DocumentChunker factory class fails when provided no document"""
    document_paths = []
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = DocumentChunker(
                document_paths=document_paths,
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
    DocumentChunker.create_tokenizer(tokenizer_model_name)


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
        DocumentChunker.create_tokenizer(model_path)


def test_get_documents_basic():
    """Test successful document retrieval with basic inputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source = {
            "repo": "https://fake-repo-url.git",
            "commit": "abc123",
            "patterns": ["*.md", "*.pdf"],
        }

        mock_repo = Mock()
        mock_repo.working_dir = temp_dir

        # Create test files
        test_md = Path(temp_dir) / "test.md"
        test_md.write_text("# Test content")

        with patch("git.Repo.clone_from", return_value=mock_repo):
            result = _get_documents(source, document_output_dir=Path(temp_dir))

        assert len(result) == 1
        assert result[0].name == "test.md"


def test_get_documents_html_warning():
    """Test warning is logged when markdown contains HTML"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source = {"repo": "https://fake-repo-url.git", "patterns": ["*.md"]}

        mock_repo = Mock()
        mock_repo.working_dir = temp_dir

        # Create test file with HTML
        test_md = Path(temp_dir) / "test.md"
        test_md.write_text("# Test\n<div>Some HTML</div>")

        with (
            patch("git.Repo.clone_from", return_value=mock_repo),
            patch("logging.Logger.warning") as mock_warning,
        ):
            result = _get_documents(source, document_output_dir=Path(temp_dir))

        mock_warning.assert_called_once()
        assert len(result) == 1


def test_get_documents_no_files():
    """Test error when no valid documents are found"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source = {"repo": "https://fake-repo-url.git", "patterns": ["*.md"]}

        mock_repo = Mock()
        mock_repo.working_dir = temp_dir

        with (
            patch("git.Repo.clone_from", return_value=mock_repo),
            pytest.raises(SystemExit),
        ):
            _get_documents(source, document_output_dir=Path(temp_dir))


def test_get_documents_git_error():
    """Test handling of git errors"""
    source = {"repo": "https://fake-repo-url.git", "patterns": ["*.md"]}

    with patch("git.Repo.clone_from") as mock_clone:
        mock_clone.side_effect = git.exc.GitCommandError("clone", "error")
        with pytest.raises(git.exc.GitCommandError):
            _get_documents(source)


def test_get_documents_skip_checkout():
    """Test that commit checkout is skipped when specified"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source = {
            "repo": "https://fake-repo-url.git",
            "commit": "abc123",
            "patterns": ["*.md"],
        }

        mock_repo = Mock()
        mock_repo.working_dir = temp_dir

        # Create a test file so the function finds something
        test_md = Path(temp_dir) / "test.md"
        test_md.write_text("# Test content")

        with patch("git.Repo.clone_from", return_value=mock_repo) as mock_clone:
            result = _get_documents(
                source, skip_checkout=True, document_output_dir=Path(temp_dir)
            )

        mock_repo.git.checkout.assert_not_called()
        assert len(result) == 1
        assert result[0].name == "test.md"
