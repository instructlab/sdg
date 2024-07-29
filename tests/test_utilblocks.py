# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from src.instructlab.sdg.utilblocks import (
    DuplicateColumnsBlock,
    FlattenColumnsBlock,
    RenameColumnsBlock,
    SetToMajorityValueBlock,
)


class TestUtilBlock(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_pipe = MagicMock()

        self.dataset = Dataset.from_dict(
            {
                "name": ["Tomato", "Rose", "Carrot", "Basil"],
                "type": ["Fruit", "Flower", "Vegetable", "Herb"],
                "edible": ["yes", "no", "yes", "yes"],
            }
        )

    def test_dup(self):
        """Test that the DuplicateColumnsBlock Block can correctly duplicate a column"""
        dup_block = DuplicateColumnsBlock(
            self.mock_ctx, self.mock_pipe, "blockname", columns_map={"edible": "caneat"}
        )
        assert "caneat" not in self.dataset.column_names
        new_samples = dup_block.generate(self.dataset)
        assert "edible" in new_samples.column_names
        assert "caneat" in new_samples.column_names

        assert new_samples["edible"] == new_samples["caneat"]

    def test_RenameColumnsBlock(self):
        """Test that the RenameColumnsBlock Block can rename a column"""
        rename_block = RenameColumnsBlock(
            self.mock_ctx, self.mock_pipe, "blockname", columns_map={"edible": "caneat"}
        )
        new_samples = rename_block.generate(self.dataset)
        assert "caneat" in new_samples.column_names
        assert "edible" not in new_samples.column_names

        assert new_samples["caneat"] == ["yes", "no", "yes", "yes"]

    def test_setmajority(self):
        """Test that the SetToMajorityValueBlock Block can correctly set the value of the specified column to the most common value (the mode)"""
        mv_block = SetToMajorityValueBlock(
            self.mock_ctx, self.mock_pipe, "blockname", col_name="edible"
        )

        new_samples = mv_block.generate(self.dataset)

        assert new_samples["edible"] == ["yes", "yes", "yes", "yes"]

    def test_flatten(self):
        """Test that the FlattenColumnsBlock can correctly melt/transform a data from a wide format to a long format
        see pandas.melt for a description"""
        flatten_block = FlattenColumnsBlock(
            self.mock_ctx,
            self.mock_pipe,
            "blockname",
            var_cols=["edible", "type"],
            value_name="val",
            var_name="vname",
        )

        new_samples = flatten_block.generate(self.dataset)

        new_data_dict = {
            "name": [
                "Tomato",
                "Rose",
                "Carrot",
                "Basil",
                "Tomato",
                "Rose",
                "Carrot",
                "Basil",
            ],
            "vname": [
                "edible",
                "edible",
                "edible",
                "edible",
                "type",
                "type",
                "type",
                "type",
            ],
            "val": ["yes", "no", "yes", "yes", "Fruit", "Flower", "Vegetable", "Herb"],
        }

        assert new_samples.to_dict() == new_data_dict
