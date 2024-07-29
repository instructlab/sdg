# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg.utils import pandas

# Local
from .block import Block

logger = logging.getLogger(__name__)


# This is part of the public API.
class SamplePopulatorBlock(Block):
    def __init__(
        self, ctx, pipe, block_name, config_paths, column_name, post_fix=""
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.configs = {}
        for config in config_paths:
            if post_fix:
                config_name = config.replace(".yaml", f"_{post_fix}.yaml")
            else:
                config_name = config
            config_key = config.split("/")[-1].split(".")[0]
            self.configs[config_key] = self._load_config(config_name)
        self.column_name = column_name

    # Using a static method to avoid serializing self when using multiprocessing
    @staticmethod
    def _map_populate(samples, configs, column_name, num_proc=1):
        def populate(sample):
            return {**sample, **configs[sample[column_name]]}

        return samples.map(populate, num_proc=num_proc)

    def generate(self, samples) -> Dataset:
        return self._map_populate(
            samples, self.configs, self.column_name, self.ctx.dataset_num_procs
        )


# This is part of the public API.
class SelectorBlock(Block):
    def __init__(
        self, ctx, pipe, block_name, choice_map, choice_col, output_col
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.choice_map = choice_map
        self.choice_col = choice_col
        self.output_col = output_col

    # Using a static method to avoid serializing self when using multiprocessing
    @staticmethod
    def _map_select_choice(samples, choice_map, choice_col, output_col, num_proc=1):
        def select_choice(sample) -> dict:
            sample[output_col] = sample[choice_map[sample[choice_col]]]
            return sample

        return samples.map(select_choice, num_proc=num_proc)

    def generate(self, samples: Dataset) -> Dataset:
        return self._map_select_choice(
            samples,
            self.choice_map,
            self.choice_col,
            self.output_col,
            self.ctx.dataset_num_procs,
        )


# This is part of the public API.
class CombineColumnsBlock(Block):
    def __init__(
        self, ctx, pipe, block_name, columns, output_col, separator="\n\n"
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.columns = columns
        self.output_col = output_col
        self.separator = separator

    # Using a static method to avoid serializing self when using multiprocessing
    @staticmethod
    def _map_combine(samples, columns, output_col, separator, num_proc=1):
        def combine(sample):
            sample[output_col] = separator.join([sample[col] for col in columns])
            return sample

        return samples.map(combine, num_proc=num_proc)

    def generate(self, samples: Dataset) -> Dataset:
        return self._map_combine(
            samples,
            self.columns,
            self.output_col,
            self.separator,
            self.ctx.dataset_num_procs,
        )


class FlattenColumnsBlock(Block):
    """Melt/transform a data from a wide format to a long format see pandas.melt for a description

    Args:
            var_cols (list): Column(s) to unpivot. All other columns are set to use as identifier variables.
            value_name (str): Name to use for the ‘value’ column, can’t be an existing column label.
            var_name (str):  Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.
    """

    def __init__(
        self, ctx, pipe, block_name: str, var_cols: list, value_name: str, var_name: str
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.var_cols = var_cols
        self.value_name = value_name
        self.var_name = var_name

    def generate(self, samples: Dataset) -> Dataset:
        df = samples.to_pandas()
        id_cols = [col for col in samples.column_names if col not in self.var_cols]
        flatten_df = df.melt(
            id_vars=id_cols,
            value_vars=self.var_cols,
            value_name=self.value_name,
            var_name=self.var_name,
        )
        return pandas.dataset_from_pandas_dataframe(flatten_df)


class DuplicateColumnsBlock(Block):
    def __init__(self, ctx, pipe, block_name: str, columns_map: dict) -> None:
        """Create duplicate of columns specified in column map.

        Args:
            columns_map (dict): mapping of existing column to new column names
        """
        super().__init__(ctx, pipe, block_name)
        self.columns_map = columns_map

    def generate(self, samples: Dataset):
        for col_to_dup in self.columns_map:
            samples = samples.add_column(
                self.columns_map[col_to_dup], samples[col_to_dup]
            )
        return samples


class RenameColumnsBlock(Block):
    def __init__(self, ctx, pipe, block_name: str, columns_map: dict) -> None:
        """Rename dataset columns.

        Args:
            columns_map (dict): mapping of existing column to new column names
        """
        self.columns_map = columns_map
        super().__init__(ctx, pipe, block_name)

    def generate(self, samples: Dataset):
        samples = samples.rename_columns(self.columns_map)
        return samples


class SetToMajorityValueBlock(Block):
    """Set the value of the specified column to the most common value (the mode)

    Args:
        col_name (str): the column to find the "mode" of and then set universally
    """

    def __init__(self, ctx, pipe, block_name: str, col_name) -> None:
        self.col_name = col_name
        super().__init__(ctx, pipe, block_name)

    def generate(self, samples: Dataset):
        samples = samples.to_pandas()
        samples[self.col_name] = samples[self.col_name].mode()[0]
        return pandas.dataset_from_pandas_dataframe(samples)
