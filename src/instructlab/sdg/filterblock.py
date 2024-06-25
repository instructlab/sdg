# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


class FilterByValueBlock(Block):
    def __init__(
        self, filter_column, filter_value, operation, convert_dtype=None, **batch_kwargs
    ) -> None:
        self.value = filter_value
        self.column_name = filter_column
        self.operation = operation
        self.convert_dtype = convert_dtype
        self.num_procs = batch_kwargs.get("num_procs", 1)

    def generate(self, samples) -> Dataset:
        if self.convert_dtype:
            samples = samples.map(
                lambda x: {
                    **x,
                    self.column_name: self.convert_dtype(x[self.column_name]),
                },
                num_proc=self.num_procs,
            )

        return samples.filter(
            lambda x: self.operation(x[self.column_name], self.value),
            num_proc=self.num_procs,
        )
