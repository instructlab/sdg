# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset
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

    def generate(self, dataset: Dataset, cache_dataset_path=None):
        """
        Generate the dataset by running the chained pipeline steps.
        dataset: the input dataset
        """
        # if a partial cache is provided, remove the corresponding rows in dataset (based
        # on the keys in dataset) such that the pipeline only runs on the rest of the rows.
        # below assumes all columns in the original dataset is preserved in the synthetic.
        if cache_dataset_path is not None:
            cache_dataset = load_dataset(
                "json", data_files=[cache_dataset_path], split="train"
            )
            # get columns from the original dataset
            orig_cols = dataset.column_names
            # concate the values as the key
            orig_repr = lambda x: "-".join([x[col] for col in orig_cols])
            # build cache
            cache = set([orig_repr(x) for x in cache_dataset])
            # filter out the dataset to keep only the new ones
            dataset = dataset.filter(lambda x: orig_repr(x) not in cache)

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

        if cache_dataset_path is not None:
            output_splits += [cache_dataset]
        output_dataset = concatenate_datasets(output_splits)
        if cache_dataset_path is not None:
            output_dataset.to_json(cache_dataset_path, orient="records", lines=True)
        return output_dataset
