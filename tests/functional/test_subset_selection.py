# Standard
from multiprocessing import set_start_method
from pathlib import Path
import json
import logging
import os
import tempfile
import uuid

# Third Party
from datasets import Dataset
import pytest
import torch

# First Party
from instructlab.sdg.subset_selection import subset_datasets


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


def test_subset_datasets_functional():
    """Functional test for subset_datasets."""
    set_start_method("spawn", force=True)
    logger = logging.getLogger(__name__)

    try:
        # Create a temporary directory for input/output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate synthetic test data
            test_data = create_test_data(num_samples=50)

            # Save as JSONL file
            input_file = Path(temp_dir) / "test_data.jsonl"
            with open(input_file, "w") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            logger.info(f"Created test file with {len(test_data)} samples")

            # Configure subset selection
            input_files = [str(input_file)]
            output_dir = os.path.join(temp_dir, "output")

            # Run subset selection
            subset_datasets(
                input_files=input_files,
                output_dir=output_dir,
                batch_size=10,  # Small batch size for testing
                num_folds=2,  # Fewer folds for faster testing
                subset_sizes=[20],  # Select 20 samples
                num_gpus=2,  # Use 2 threads
                encoder_type="arctic",
                encoder_model="Snowflake/snowflake-arctic-embed-l-v2.0",
                epsilon=0.1,  # Small epsilon for small dataset
                testing_mode=True,  # Enable testing mode
            )

            # Verify outputs
            dataset_name = "test_data"
            dataset_output_dir = os.path.join(output_dir, dataset_name)

            # Check if embeddings were generated
            assert os.path.exists(
                os.path.join(dataset_output_dir, "embeddings", "embeddings.h5")
            ), "Embeddings file not found"

            # Check if subset file was created
            assert os.path.exists(
                os.path.join(
                    dataset_output_dir, f"{dataset_name}_samples_20_subset.jsonl"
                )
            ), "20-sample subset file not found"

            # Check if metadata file was created
            assert os.path.exists(
                os.path.join(
                    output_dir,
                    f"{dataset_name}_fl_2_partitions_samples_20_metadata.npz",
                )
            ), "Metadata file for 20-sample subset not found"

    finally:
        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
