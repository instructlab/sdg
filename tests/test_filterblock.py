# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import operator
import unittest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from instructlab.sdg.filterblock import FilterByValueBlock
from instructlab.sdg.pipeline import PipelineContext


class TestFilterByValueBlock(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.dataset_num_procs = 1
        self.pipe = MagicMock()
        self.block = FilterByValueBlock(
            self.ctx,
            self.pipe,
            "filter_by_age",
            filter_column="age",
            filter_value="30",
            operation="eq",
            convert_dtype="int",
        )
        self.block_with_list = FilterByValueBlock(
            self.ctx,
            self.pipe,
            "filter_by_age_list",
            filter_column="age",
            filter_value=["30", "35"],
            operation="eq",
            convert_dtype="int",
        )
        self.dataset = Dataset.from_dict(
            {"age": ["25", "30", "35", "forty", "45"]},
            features=Features({"age": Value("string")}),
        )

    def test_generate_mixed_types(self):
        filtered_dataset = self.block.generate(self.dataset)
        self.assertEqual(len(filtered_dataset), 1)
        self.assertEqual(filtered_dataset["age"], [30])

    def test_generate_mixed_types_multi_value(self):
        filtered_dataset = self.block_with_list.generate(self.dataset)
        self.assertEqual(len(filtered_dataset), 2)
        self.assertEqual(filtered_dataset["age"], [30, 35])
