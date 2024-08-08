# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import uuid

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.data_files import EmptyDatasetError

# First Party
from instructlab.sdg.utils import pandas

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, checkpoint_dir=None, save_freq=1):
        self._checkpoint_dir = checkpoint_dir

        self._save_freq = save_freq
        self._cache = []

    def checkpoint(self, dataset):
        if len(dataset) != 0:
            self._cache.append(dataset)
        if len(self._cache) < self._save_freq:
            return
        self.save()
        self._cache.clear()

    def done(self):
        if self._cache:
            self.save()
            self._cache.clear()

    def save(self):
        if self._checkpoint_dir is None:
            return
        checkpoint_id = uuid.uuid4().hex
        checkpoint_file = (
            f"{self._checkpoint_dir}/data_checkpoint_{checkpoint_id}.jsonl"
        )
        logger.info(f"Saving checkpoint to {checkpoint_file}")
        # Saves all the current records to new file in the checkpoint dir
        concatenate_datasets(self._cache).to_json(
            checkpoint_file, orient="records", lines=True
        )

    def load(self, dataset: Dataset) -> Dataset:
        if self._checkpoint_dir is None:
            return dataset, None

        try:
            pre_generated_data = load_dataset(
                "json", data_dir=self._checkpoint_dir, split="train"
            )
        except EmptyDatasetError:
            logger.info(
                f"No existing checkpoints found in {self._checkpoint_dir}, generating from scratch"
            )
            return dataset, None

        logger.info(
            f"Loading existing checkpoints from {self._checkpoint_dir}, with {pre_generated_data.num_rows} rows"
        )
        seed_data = self._get_missing_data(dataset, pre_generated_data)
        logger.info(f"Found {seed_data.num_rows} missing rows in the dataset")
        return seed_data, pre_generated_data

    def _get_missing_data(self, seed_data, generated_data):
        # Get the common columns between the two datasets
        common_columns = list(
            set(seed_data.column_names) & set(generated_data.column_names)
        )

        # Extract the relevant data based on common columns
        seed_data_common = seed_data.select_columns(common_columns)
        generated_data_common = generated_data.select_columns(common_columns)

        # Convert to Pandas DataFrames for easier comparison
        seed_df = seed_data_common.to_pandas()
        generated_df = generated_data_common.to_pandas()

        # Identify missing rows
        missing_rows = ~seed_df.apply(tuple, 1).isin(generated_df.apply(tuple, 1))

        missing_df = seed_data.to_pandas()[missing_rows]
        return pandas.dataset_from_pandas_dataframe(missing_df)
