# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Third Party
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

# Local
from .pipeline import Pipeline


def split_dataset(dataset: Dataset, batch_size: int) -> List[Dataset]:
    """Split the dataset into smaller batches."""
    total_size = len(dataset)
    num_batches = (total_size + batch_size - 1) // batch_size
    batches = [
        dataset.select(range(i * batch_size, min((i + 1) * batch_size, total_size)))
        for i in range(num_batches)
    ]
    return batches


class SDG:
    def __init__(
        self, pipelines: list[Pipeline], num_workers=1, batch_size=None
    ) -> None:
        self.pipelines = pipelines
        self.num_workers = num_workers
        self.batch_size = batch_size

    @staticmethod
    def generate_split(pipelines, input_split, output_splits, i):
        for pipeline in pipelines:
            input_split = pipeline.generate(input_split)
        output_splits[i] = input_split

    def generate(self, dataset: Dataset):
        """
        Generate the dataset by running the chained pipeline steps.
        dataset: the input dataset
        """
        input_splits = (
            split_dataset(dataset, self.batch_size) if self.batch_size else [dataset]
        )
        output_splits = [None] * len(input_splits)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self.generate_split, self.pipelines, input_split, output_splits, i
                )
                for i, input_split in enumerate(input_splits)
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # Ensure each future completes

        return concatenate_datasets(output_splits)
