# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import unittest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from instructlab.sdg import Block, BlockRegistry, IterBlock


class TestIterBlock(unittest.TestCase):
    @BlockRegistry.register("TestCounterBlock")
    class TestCounterBlock(Block):
        def __init__(
            self,
            ctx,
            pipe,
            block_name,
            column,
            increment=1,
        ) -> None:
            super().__init__(ctx, pipe, block_name)
            self.column = column
            self.increment = increment
            self.counter = 0

        def generate(self, samples: Dataset):
            samples = samples.map(
                lambda x: {self.column: x[self.column] + self.counter}
            )
            self.counter += self.increment
            return samples

    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.dataset_num_procs = 1
        self.pipe = MagicMock()
        self.block = IterBlock(
            self.ctx,
            self.pipe,
            "iter_test",
            num_iters=4,
            block_type="TestCounterBlock",
            column="counter",
        )
        self.dataset = Dataset.from_dict(
            {"counter": [0]},
            features=Features({"counter": Value("int32")}),
        )

    def test_simple_iterate(self):
        iterated_dataset = self.block.generate(self.dataset)
        # We iterated 4 times, so 4 items in our dataset - one from
        # each iteration
        self.assertEqual(len(iterated_dataset), 4)
        # Each iteration increment our counter, because of our custom
        # block used that just increments counters
        self.assertEqual(iterated_dataset["counter"], [0, 1, 2, 3])
