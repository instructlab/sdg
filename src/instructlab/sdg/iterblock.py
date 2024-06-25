import re
from .block import Block
from .logger_config import setup_logger
from datasets import Dataset

logger = setup_logger(__name__)

class IterBlock(Block):
    def __init__(self, block_name, num_iters, block_type, block_kwargs, **kwargs):
        super().__init__(block_name)
        self.num_iters = num_iters
        self.block = block_type(**block_kwargs)
        self.gen_kwargs = kwargs.get('gen_kwargs', {})
        self.gen_kwargs = kwargs.get("gen_kwargs", {})

    def generate(self, samples, **gen_kwargs) -> Dataset:
        generated_samples = []
        num_iters = self.num_iters

        for _ in range(num_iters):
            batch_generated = self.block.generate(
                samples, **{**self.gen_kwargs, **gen_kwargs}
            )
            generated_samples.extend(batch_generated)

        return Dataset.from_list(generated_samples)
