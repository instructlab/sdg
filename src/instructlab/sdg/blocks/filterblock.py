# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import operator

# Third Party
from datasets import Dataset

# Local
from .block import Block

logger = logging.getLogger(__name__)


# This is part of the public API.
class FilterByValueBlockError(Exception):
    """An exception raised by the FilterByValue block."""


def _get_operator_func(op):
    if not op in dir(operator):
        raise FilterByValueBlockError(f"Unknown FilterByValueBlock operation '{op}'")
    return getattr(operator, op)


# Note - this is not a method on the class below in order to avoid
# serializing the object itself when multi-processing is used.
# In particular, SSLContext - embedded in the OpenAI client object -
# cannot be pickled.
def _filter_by_values(samples, column, op, values, num_proc=1):
    return samples.filter(
        lambda x: any(op(x[column], value) for value in values),
        num_proc=num_proc,
    )


class DTypeConverter:
    def __init__(self, dtype, default_value=None):
        self.dtype = dtype
        self.default_value = default_value

    def __call__(self, value):
        if self.dtype is None:
            return value
        try:
            return self.dtype(value)
        except ValueError as e:
            logger.debug(
                f"Error converting to {self.dtype}: {e}, filling with {self.default_value}"
            )
            return self.default_value

    @classmethod
    def get(cls, dtype, default_value):
        if not dtype:
            return DTypeConverter(None, None)

        type_mapping = {
            "int": (int, 0),
            "float": (float, 0.0),
            "bool": (bool, False),
        }
        if not dtype in type_mapping:
            raise FilterByValueBlockError(
                f"Unknown FilterByValueBlock convert_dtype '{dtype}'"
            )

        if default_value is None:
            return DTypeConverter(*type_mapping[dtype])

        dtype = type_mapping[dtype][0]
        return DTypeConverter(dtype, dtype(default_value))


# Note - this is not a method on the class below in order to avoid
# serializing the object itself when multi-processing is used.
# In particular, SSLContext - embedded in the OpenAI client object -
# cannot be pickled.
def _map_dtype(samples, column, dtype, num_proc=1):
    def convert_column(sample):
        sample[column] = dtype(sample[column])
        return sample

    return samples.map(convert_column, num_proc=num_proc)


# This is part of the public API.
class FilterByValueBlock(Block):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        filter_column,
        filter_value,
        operation,
        convert_dtype=None,
        default_value=None,
    ) -> None:
        """
        Initializes a new instance of the FilterByValueBlock class.

        Parameters:
        - ctx (PipelineContext): A PipelineContext object containing runtime parameters.
        - pipe (Pipeline): The Pipeline containing this block in its chain.
        - block_name (str): An identifier for this block.
        - filter_column (str): The name of the column in the dataset to apply the filter on.
        - filter_value (any or list of any): The value(s) to filter by.
        - operation (string): The name of a function provided by the "operator"
          Python package that takes two arguments (column value and filter value)
          and returns a boolean indicating whether the row should be included in
          the filtered dataset.
        - convert_dtype (string, optional): the name of a Python type to convert
          the column values to. Supported values are "int", "float", and "bool".
          Defaults to None.
        - default_value (string, optional): a default value that should be used
          if convert_dtype fails. Defaults to 0 for int/float, and False for bool.

        Returns:
        None

        For supported values of `operation`, see the "operator" package
        documentation: https://docs.python.org/3/library/operator.html

        Only a subset of the "operator" package is relevant. It has to
        follow the semantics of taking two parameters and returning a boolean.
        Some operations that work include:
        - eq: equal to
        - ne: not equal to
        - gt: greater than
        - ge: greater than or equal to
        - lt: less than
        - le: less than or equal to
        - contains: filter_column contains filter_value (only for string columns)

        Note that the semantics of all operations are:
          - filter_column operation filter_value

        Example: FilterByValueBlock(ctx, "filter_by_age", "age", 30, "eq", "int")
            - This block will filter the dataset to only include rows where the
              "age" column is equal to 30.

        The `contains` operator is only supported for string columns. This is
        useful if you want to ensure that a string column contains a specific
        substring.

        Example: FilterByValueBlock(ctx, "filter_by_name", "full_name", "John", "contains")
            - This block will filter the dataset to only include rows where the
              "full_name" column contains the substring "John".

        `filter_value` does not have to be a single value. It can also be a list of values.
        In that case, the operation will be applied to each value in the list. The result is
        considered True if the operation is True for any of the values in the list.

        Example: FilterByValueBlock(ctx, "filter_by_age", "age", [30, 35], "eq", "int", "0")
            - This block will filter the dataset to only include rows where the
              "age" column is equal to 30 or 35. Non-integer values will be treated like zero.

        Example: FilterByValueBlock(ctx, "filter_by_city", "city", ["boston", "charleston", "dublin", "new york"], "eq")
            - This block will filter the dataset to only include rows where the
              "city" column is equal to "boston", "charleston", "dublin", or "new york".

        Example: FilterByValueBlock(ctx, "filter_by_name", "full_name", ["John", "Jane"], "contains")
            - This block will filter the dataset to only include rows where the
              "full_name" column contains the substring "John" or "Jane".
        """
        super().__init__(ctx, pipe, block_name)
        self.value = filter_value if isinstance(filter_value, list) else [filter_value]
        self.column_name = filter_column
        self.operation = _get_operator_func(operation)
        self.dtype = DTypeConverter.get(convert_dtype, default_value)
        if self.dtype:
            self.value = [self.dtype(value) for value in self.value]

    def generate(self, samples) -> Dataset:
        if self.dtype:
            samples = _map_dtype(
                samples,
                self.column_name,
                self.dtype,
                self.ctx.dataset_num_procs,
            )

        return _filter_by_values(
            samples,
            self.column_name,
            self.operation,
            self.value,
            self.ctx.dataset_num_procs,
        )
