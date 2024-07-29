# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from instructlab.sdg.utilblocks import SamplePopulatorBlock


class TestSamplePopulatorBlock(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.dataset_num_procs = 1
        self.pipe = MagicMock()

    @patch("instructlab.sdg.block.Block._load_config")
    def test_generate(self, mock_load_config):
        def load_config(file_name):
            if file_name == "coffee.yaml" or file_name == "tea.yaml":
                return {"preferred_temperature": "hot"}
            else:
                return {"preferred_temperature": "cold"}

        mock_load_config.side_effect = load_config

        block = SamplePopulatorBlock(
            self.ctx,
            self.pipe,
            "populate_preferred_temperature",
            config_paths=["coffee.yaml", "tea.yaml", "water.yaml"],
            column_name="beverage",
        )
        dataset = Dataset.from_dict(
            {"beverage": ["coffee", "tea", "water"]},
            features=Features({"beverage": Value("string")}),
        )

        dataset = block.generate(dataset)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset["preferred_temperature"], ["hot", "hot", "cold"])
