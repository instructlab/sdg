# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


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
