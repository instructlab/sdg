# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for top-level subset selection module.
"""

# Standard
from unittest.mock import patch
import os

# Third Party
from datasets import Dataset
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
    # Test percentage (now using 0-1 range instead of 0-100)
    assert data_processor.calculate_subset_size(1000, 0.105) == 105
    # Test absolute
    assert data_processor.calculate_subset_size(1000, 50) == 50
    # Test percentage rounding
    assert data_processor.calculate_subset_size(1000, 0.001) == 1
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


def test_generate_embeddings_parallel(mock_gpu_environment, tmp_path, mock_encoder):
    """Test the parallelized embedding generation feature."""
    # Create a sample dataset
    sample_texts = [f"text{i}" for i in range(50)]
    dataset = Dataset.from_dict({"text": sample_texts})

    # Create output directory
    output_dir = str(tmp_path / "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # Create mock embeddings file to test the early return
    merged_file = os.path.join(output_dir, "embeddings.h5")
    with h5py.File(merged_file, "w") as f:
        f.create_dataset("embeddings", data=torch.randn(50, 768).numpy())

    # Create config
    basic_config = BasicConfig(output_dir=str(tmp_path))
    config = ProcessingConfig(
        input_files=["test.jsonl"],
        subset_sizes=[10, 0.2],  # Both absolute and percentage
        basic=basic_config,
    )
    config.system.num_gpus = 2

    # Create processor
    processor = DataProcessor(config, mock_encoder)

    # Test case 1: File exists, should return early
    result_path = processor.generate_embeddings(dataset, output_dir)
    assert result_path == merged_file

    # Test case 2: File doesn't exist, should process data
    # First remove the existing file
    os.remove(merged_file)

    # We need to patch the Pool.map call instead of the functions directly
    # This avoids the pickling issue
    with patch("multiprocessing.pool.Pool.map") as mock_map:
        # Set up mock return values
        shard_files = [
            os.path.join(output_dir, f"shard_{i}", f"embeddings_shard_{i}.h5")
            for i in range(2)
        ]
        mock_map.return_value = shard_files

        # Also mock the merge function to avoid actually creating files
        with patch("instructlab.sdg.subset_selection._merge_shard_files") as mock_merge:
            mock_merge.side_effect = lambda shard_files, merged_path: None

            # Call the function
            result_path = processor.generate_embeddings(dataset, output_dir)

            # Verify Pool.map was called
            mock_map.assert_called_once()

            # Verify the correct path is returned
            assert result_path == os.path.join(output_dir, "embeddings.h5")

            # Verify merge was called with the right arguments
            mock_merge.assert_called_once_with(shard_files, merged_file)
