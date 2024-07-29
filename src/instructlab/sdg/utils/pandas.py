# SPDX-License-Identifier: Apache-2.0

# Third Party
from datasets import Dataset
from pandas import DataFrame


def dataset_from_pandas_dataframe(data_frame: DataFrame) -> Dataset:
    """
    Convert a pandas DataFrame into a Hugging Face Dataset, ensuring that the index is
    dropped to avoid introducing __index_level_0__ column
    """

    data_frame = data_frame.reset_index(drop=True)
    return Dataset.from_pandas(data_frame)
