# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for top-level subset selection module.
"""

# Standard
from unittest.mock import patch
import os

# Third Party
import h5py
import pytest
import torch

# First Party
from instructlab.sdg.subset_selection import (
    BasicConfig,
    DataProcessor,
    ProcessingConfig,
)


@pytest.fixture
def mock_gpu_environment():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
    ):
        yield


@pytest.fixture
def mock_encoder():
    class MockEncoder:
        def __init__(self, model_name, testing_mode=False):
            self.model_name = model_name
            self.testing_mode = testing_mode

        def encode(self, inputs, instruction):
            # Return mock embeddings
            return torch.randn(len(inputs), 768)

    return MockEncoder


@pytest.fixture
def data_processor(mock_encoder, mock_gpu_environment):
    config = ProcessingConfig(
        input_files=["test.jsonl"],
        subset_sizes=[10, 20.5],
    )
    return DataProcessor(config, mock_encoder)


def test_format_text(data_processor):
    """Test text formatting with different templates"""
    example = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }
    formatted = data_processor.format_text(example, "conversation")
    assert "user: Hello" in formatted
    assert "assistant: Hi there" in formatted


def test_calculate_subset_size(data_processor):
    """Test subset size calculation for both percentage and absolute values"""
    # Test percentage
    assert data_processor.calculate_subset_size(1000, 10.5) == 105
    # Test absolute
    assert data_processor.calculate_subset_size(1000, 50) == 50
    # Test percentage rounding
    assert data_processor.calculate_subset_size(1000, 0.1) == 1
    # Test absolute capping
    assert data_processor.calculate_subset_size(10, 20) == 10


def test_get_subset_name(data_processor):
    """Test subset name generation"""
    assert data_processor.get_subset_name(10.5, 105) == "percent_10.5"
    assert data_processor.get_subset_name(50, 50) == "samples_50"


def test_valid_config_initialization(mock_gpu_environment):
    """Test configuration initialization with valid parameters"""
    config = ProcessingConfig(
        input_files=["test.jsonl"],
        subset_sizes=[10, 20.5],
        basic=BasicConfig(output_dir="test_output"),
    )
    assert config.input_files == ["test.jsonl"]
    assert config.subset_sizes == [10, 20.5]
    assert config.basic.output_dir == "test_output"


def test_invalid_subset_sizes(mock_gpu_environment):
    """Test configuration validation with invalid subset sizes"""
    with pytest.raises(
        ValueError, match="subset_sizes must contain only integers or floats"
    ):
        ProcessingConfig(
            input_files=["test.jsonl"],
            subset_sizes=[10, "invalid"],
        )

    with pytest.raises(
        ValueError, match="Percentage values in subset_sizes must be between 0 and 100"
    ):
        ProcessingConfig(
            input_files=["test.jsonl"],
            subset_sizes=[10, 150.5],
        )

    with pytest.raises(
        ValueError, match="Absolute values in subset_sizes must be positive"
    ):
        ProcessingConfig(
            input_files=["test.jsonl"],
            subset_sizes=[-10],
        )


def test_process_batch(mock_gpu_environment, data_processor, tmp_path):
    """Test batch processing of texts"""

    batch_texts = ["text1", "text2", "text3"]
    output_file = str(tmp_path / "test_batch.h5")

    embedding_dim = data_processor.process_batch(batch_texts, output_file)

    assert embedding_dim is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as f:
        embeddings = f["embeddings"][:]
        assert embeddings.shape == (3, embedding_dim)


def test_generate_embeddings_resume(mock_gpu_environment, data_processor, tmp_path):
    """Test embedding generation with resume capability"""
    output_dir = str(tmp_path / "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # Create pre-existing batch files to simulate partial processing
    mock_embeddings = torch.randn(10, 768)
    batch_0_file = os.path.join(output_dir, "batch_0.h5")

    with h5py.File(batch_0_file, "w") as f:
        f.create_dataset("embeddings", data=mock_embeddings.numpy())

    # dummy dataset
    dataset = [{"text": f"text{i}"} for i in range(30)]

    merged_file = data_processor.generate_embeddings(dataset, output_dir)

    # Verify resume behavior
    assert os.path.exists(merged_file)

    # Verify the merged file exists and contains embeddings
    with h5py.File(merged_file, "r") as f:
        final_embeddings = f["embeddings"][:]
        # Just verify the embedding dimension is correct
        assert final_embeddings.shape[1] == 768

    # Run again with same dataset - should skip processing
    new_merged_file = data_processor.generate_embeddings(dataset, output_dir)

    # Verify the file is the same and has same number of embeddings
    with h5py.File(new_merged_file, "r") as f:
        new_embeddings = f["embeddings"][:]
        assert new_embeddings.shape == final_embeddings.shape
