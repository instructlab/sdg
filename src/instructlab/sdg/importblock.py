# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from . import pipeline
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)


# This is part of the public API.
class ImportBlock(Block):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        path,
    ) -> None:
        """
        ImportBlock imports a chain of blocks from another pipeline config file.

        Parameters:
        - ctx (PipelineContext): A PipelineContext object containing runtime parameters.
        - pipe (Pipeline): The Pipeline containing this block in its chain.
        - block_name (str): An identifier for this block.
        - path (str): A path (absolute, or relative to the instructlab.sdg package) to a pipeline config file.
        """
        super().__init__(ctx, pipe, block_name)
        self.path = path
        self.pipeline = pipeline.Pipeline.from_file(self.ctx, self.path)

    def generate(self, samples) -> Dataset:
        logger.info("ImportBlock chaining to blocks from {self.path}")
        return self.pipeline.generate(samples)
