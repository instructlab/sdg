# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from datasets import Dataset

# Local
from ..pipeline import _lookup_block_type
from ..registry import BlockRegistry
from .block import Block

logger = logging.getLogger(__name__)


# This is part of the public API.
@BlockRegistry.register("IterBlock")
class IterBlock(Block):
    """
    Call another block multiple times for a single set of input
    samples, concatening the results of each iteration's call to that
    other block in the final returned output.

    Args:
        num_iters: The number of times to iterate over the block
        block_type: The type of the other block to call (ie LLMBlock)
        block_config: Any necessary configuration that will get passed to the
                      other block to properly configure it.

    Returns:
        A Dataset containing all output samples from each iteration
    """

    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        num_iters,
        block_type,
        **block_config,
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.num_iters = num_iters
        block_type = _lookup_block_type(block_type)
        self.block = block_type(ctx, pipe, block_name, **block_config)

    def generate(self, samples: Dataset) -> Dataset:
        generated_samples = []
        num_iters = self.num_iters

        for _ in range(num_iters):
            batch_generated = self.block.generate(samples)
            generated_samples.extend(batch_generated)

        return Dataset.from_list(generated_samples)
