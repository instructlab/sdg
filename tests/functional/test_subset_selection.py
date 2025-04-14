# Standard
from multiprocessing import set_start_method
from pathlib import Path
from unittest.mock import patch
import json
import logging
import os
import tempfile
import uuid

# Third Party
from datasets import Dataset
import h5py
import pytest
import torch


def create_test_data(num_samples=50):
    """Create synthetic conversation data similar to the real dataset."""
    test_data = []

    # Create conversation examples
    topics = ["stars", "galaxies", "planets", "nebulae", "black holes"]
    for i in range(num_samples):
        topic = topics[i % len(topics)]
        conversation = {
            "messages": [
                {"content": "", "role": "system"},
                {
                    "content": f"Document:\nThis is a test document about {topic} in astronomy.\nIt contains synthetic data for testing purposes.\nThe document discusses various properties of {topic}.\n\nWhat are the main characteristics of {topic}?",
                    "role": "user",
                },
                {
                    "content": f"This is a test response about {topic} characteristics.",
                    "role": "assistant",
                },
            ],
            "metadata": json.dumps(
                {
                    "sdg_document": f"Test document about {topic}",
                    "domain": "astronomy",
                    "dataset": "test_dataset",
                    "dataset_type": "test",
                }
            ),
            "id": str(uuid.uuid4()),
        }
        test_data.append(conversation)

    return test_data


# Define mock process_folds_with_gpu function at module level
def mock_process_folds_with_gpu(args):
    # Extract subset_sizes from args
    subset_sizes = args[3]
    # Return a consistent result for each subset size
    return [
        (
            0,
            {
                size: {"indices": list(range(10)), "gains": [0.5] * 10}
                for size in subset_sizes
            },
        )
    ]


# Mock Pool class that runs functions directly without multiprocessing
class MockPool:
    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, func, iterable):
        # For this test, we care about process_folds_with_gpu so we use our mock function
        return [mock_process_folds_with_gpu(item) for item in iterable]


@pytest.mark.gpu
def test_subset_datasets_functional():
    """Functional test for subset_datasets."""

    # Lazy import down here to not trigger the import except when we're running GPU tests
    # First Party
    from instructlab.sdg.subset_selection import subset_datasets

    logger = logging.getLogger(__name__)

    # Create a mock encoder class
    class MockEncoder:
        def __init__(self, model_name=None, device=None, testing_mode=False, **kwargs):
            self.model_name = model_name
            self.device = device
            self.testing_mode = testing_mode

        def encode(self, inputs, instruction=None, **kwargs):
            # Return random embeddings of the right shape
            if isinstance(inputs, str):
                inputs = [inputs]
            return torch.randn(len(inputs), 768)

    # Create mock embeddings file
    def create_mock_embeddings(dataset, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        merged_path = os.path.join(output_dir, "embeddings.h5")

        # Create actual mock embeddings file
        with h5py.File(merged_path, "w") as f:
            f.create_dataset("embeddings", data=torch.randn(len(dataset), 768).numpy())

        return merged_path

    # Mock FacilityLocationFunction for subset selection
    class MockFacilityLocationFunction:
        def __init__(self, n, sijs, mode, separate_rep):
            self.n = n
            self.sijs = sijs
            self.mode = mode
            self.separate_rep = separate_rep

        def maximize(
            self,
            budget,
            optimizer,
            epsilon,
            stopIfZeroGain,
            stopIfNegativeGain,
            verbose,
        ):
            # Return mock subset results with indices and gains
            return [(i, 0.5) for i in range(budget)]

    # Setup all the mocks
    with (
        patch(
            "instructlab.sdg.subset_selection.get_encoder_class",
            return_value=MockEncoder,
        ),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "instructlab.sdg.subset_selection.DataProcessor.generate_embeddings",
            side_effect=create_mock_embeddings,
        ),
        patch("submodlib.FacilityLocationFunction", MockFacilityLocationFunction),
        patch(
            "instructlab.sdg.subset_selection.compute_pairwise_dense",
            return_value=torch.randn(50, 50),
        ),
        patch(
            "instructlab.sdg.subset_selection.process_folds_with_gpu",
            mock_process_folds_with_gpu,
        ),
        patch("multiprocessing.set_start_method"),
    ):
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate test data
                test_data = create_test_data(num_samples=50)

                # Save as JSONL
                input_file = Path(temp_dir) / "test_data.jsonl"
                with open(input_file, "w") as f:
                    for item in test_data:
                        f.write(json.dumps(item) + "\n")

                # Run subset selection with fast testing mode
                subset_datasets(
                    input_files=[str(input_file)],
                    output_dir=os.path.join(temp_dir, "output"),
                    batch_size=10,
                    num_folds=2,
                    subset_sizes=[10, 0.2],  # Test both absolute and percentage
                    num_gpus=1,
                    testing_mode=True,
                )

                # Verify outputs exist
                dataset_name = "test_data"
                output_dir = os.path.join(temp_dir, "output")
                dataset_output_dir = os.path.join(output_dir, dataset_name)

                # Check embeddings file
                assert os.path.exists(
                    os.path.join(dataset_output_dir, "embeddings", "embeddings.h5")
                )

                # Check subset files
                assert os.path.exists(
                    os.path.join(
                        dataset_output_dir, f"{dataset_name}_samples_10_subset.jsonl"
                    )
                )
                percent_file = os.path.join(
                    dataset_output_dir, f"{dataset_name}_percent_0.2_subset.jsonl"
                )
                assert os.path.exists(percent_file)

                # Check metadata files
                assert os.path.exists(
                    os.path.join(
                        output_dir,
                        f"{dataset_name}_fl_2_partitions_samples_10_metadata.npz",
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        output_dir,
                        f"{dataset_name}_fl_2_partitions_percent_0.2_metadata.npz",
                    )
                )

        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
