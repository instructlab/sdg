# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from datasets import Dataset

# Local
from .block import Block

logger = logging.getLogger(__name__)


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

        # FIXME: find a better fix for this circular import error:
        #
        #   src/instructlab/sdg/__init__.py:29: in <module>
        #     from .importblock import ImportBlock
        #   src/instructlab/sdg/importblock.py:6: in <module>
        #     from . import pipeline
        #   src/instructlab/sdg/pipeline.py:102: in <module>
        #     "ImportBlock": importblock.ImportBlock,
        #   E   AttributeError: partially initialized module 'src.instructlab.sdg.importblock' has no attribute 'ImportBlock' (most likely due to a circular import)
        #
        # pylint: disable=C0415
        # Local
        from . import pipeline

        self.pipeline = pipeline.Pipeline.from_file(self.ctx, self.path)

    def generate(self, samples) -> Dataset:
        logger.info("ImportBlock chaining to blocks from {self.path}")
        return self.pipeline.generate(samples)
