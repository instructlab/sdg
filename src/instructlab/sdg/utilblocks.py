from .block import Block
from .logger_config import setup_logger

from datasets import Dataset

logger = setup_logger(__name__)


class SamplePopulatorBlock(Block):
    def __init__(self, config_paths, column_name, **batch_kwargs) -> None:
        self.configs = {}
        for config in config_paths: 
            config_key = config.split("/")[-1].split(".")[0]
            self.configs[config_key] = self._load_config(config)
        self.column_name = column_name
        self.num_procs = batch_kwargs.get('num_procs', 8)

    def _generate(self, sample) -> dict:
        sample = {**sample, **self.configs[sample[self.column_name]]}
        return sample

    def generate(self, samples) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples


class SelectorBlock(Block):
    def __init__(self, choice_map, choice_col, output_col, **batch_kwargs) -> None:
        self.choice_map = choice_map
        self.choice_col = choice_col
        self.output_col = output_col
        self.num_procs = batch_kwargs.get('num_procs', 8)
    
    def _generate(self, sample) -> dict:
        sample[self.output_col] = sample[self.choice_map[sample[self.choice_col]]]
        return sample
    
    def generate(self, samples: Dataset) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples


class CombineColumnsBlock(Block):
    def __init__(self, columns, output_col, separator="\n\n", **batch_kwargs) -> None:
        self.columns = columns
        self.output_col = output_col
        self.separator = separator
        self.num_procs = batch_kwargs.get('num_procs', 8)
    
    def _generate(self, sample) -> dict:
        sample[self.output_col] = self.separator.join([sample[col] for col in self.columns])
        return sample
    
    def generate(self, samples: Dataset) -> Dataset:
        samples = samples.map(self._generate, num_proc=self.num_procs)
        return samples
