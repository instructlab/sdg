# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional
import math

# Third Party
from datasets import Dataset, concatenate_datasets

# Local
from .pipeline import Pipeline


# This is part of the public API.
class SDG:
    def __init__(
        self,
        pipelines: list[Pipeline],
        num_workers: Optional[int] = 1,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize a new instance of SDG with a collection of pipelines to run
        in sequence.

        pipelines (list[Pipeline]): A list of pipeline objects to run in
            sequence.
        num_workers (int | None): The number of workers to use. None implies the
            default behavior of min(32, os.cpu_count() + 4)
            https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
        batch_size (int | None): The batch size to use when processing. None
            implies no batching.
        """
        self.pipelines = pipelines
        self.num_workers = num_workers
        self.batch_size = batch_size

    def generate(self, dataset: Dataset) -> Dataset:
        """
        Generate the dataset by running the chained pipeline steps.
        dataset (Dataset): the input dataset
        """
        input_splits = self._split_dataset(dataset)
        output_splits = [None] * len(input_splits)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_split, self.pipelines, input_split, output_splits, i
                )
                for i, input_split in enumerate(input_splits)
            ]

            # Ensure all futures complete
            for future in as_completed(futures):
                future.result()

        return concatenate_datasets(output_splits)

    ## Implementation Details ##

    def _split_dataset(self, dataset: Dataset) -> list[Dataset]:
        """Split the dataset into smaller batches."""
        if self.batch_size is None:
            return [dataset]
        total_size = len(dataset)
        num_batches = math.ceil(total_size / self.batch_size)
        batches = [
            dataset.select(self._get_batch_indices(i, total_size))
            for i in range(num_batches)
        ]
        return batches

    def _get_batch_indices(self, batch_index: int, total_size: int) -> Iterable[int]:
        assert (
            self.batch_size is not None
        ), "Programming Error: Should not call _get_batch_indices if batching disabled"
        return range(
            # Start index offset by the batch size
            batch_index * self.batch_size,
            # End index is the next batch offset or the end of the dataset
            min((batch_index + 1) * self.batch_size, total_size),
        )

    @staticmethod
    def _generate_split(
        pipelines: list[Pipeline],
        input_split: list[Dataset],
        output_splits: list[Optional[Dataset]],
        i: int,
    ) -> None:
        """Helper to run inside a worker thread. Results will be placed into the
        output_splits list at the given index.
        """
        for pipeline in pipelines:
            input_split = pipeline.generate(input_split)
        output_splits[i] = input_split
