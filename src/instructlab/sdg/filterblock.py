# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


# Note - this is not a method on the class below in order to avoid
# serializing the object itself when multi-processing is used.
# In particular, SSLContext - embedded in the OpenAI client object -
# cannot be pickled.
def _filter_by_values(samples, column, op, values, num_proc=1):
    return samples.filter(
        lambda x: any(op(x[column], value) for value in values),
        num_proc=num_proc,
    )


def _map_dtype(samples, column, dtype, num_proc=1):
    def convert_column(sample):
        try:
            sample[column] = dtype(sample[column])
        except ValueError as e:
            logger.error(
                "Error converting dtype: %s, filling with None to be filtered later", e
            )
            sample[column] = None
        return sample

    # FIXME: it appears multiprocessing map has issues with
    # None columns. If we pass num_proc>1 here and the error
    # case is triggered above, we get:
    #   ValueError: The features can't be aligned ...
    # because the column is still considered a string not
    # the new dtype.
    num_proc = 1

    return samples.map(convert_column, num_proc=num_proc)


class FilterByValueBlock(Block):
    def __init__(
        self,
        ctx,
        filter_column,
        filter_value,
        operation,
        convert_dtype=None,
        **batch_kwargs,
    ) -> None:
        """
        Initializes a new instance of the FilterByValueBlock class.

        Parameters:
        - ctx (PipelineContext): A PipelineContext object containing runtime parameters.
        - filter_column (str): The name of the column in the dataset to apply the filter on.
        - filter_value (any or list of any): The value(s) to filter by.
        - operation (callable): A function that takes two arguments (column value and filter value) and returns a boolean indicating whether the row should be included in the filtered dataset.
        - convert_dtype (callable, optional): A function to convert the data type of the filter column before applying the filter. Defaults to None.
        - **batch_kwargs: Additional kwargs for batch processing.

        Returns:
        None
        """
        super().__init__(ctx, block_name=self.__class__.__name__)
        self.value = filter_value if isinstance(filter_value, list) else [filter_value]
        self.column_name = filter_column
        self.operation = operation
        self.convert_dtype = convert_dtype
        self.num_procs = batch_kwargs.get("num_procs", 1)

    def generate(self, samples) -> Dataset:
        if self.convert_dtype:
            samples = _map_dtype(
                samples, self.column_name, self.convert_dtype, self.num_procs
            )

        return _filter_by_values(
            samples, self.column_name, self.operation, self.value, self.num_procs
        )
