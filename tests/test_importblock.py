# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import os
import tempfile
import unittest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from instructlab.sdg.importblock import ImportBlock
from instructlab.sdg.pipeline import Pipeline

# Local
from .conftest import get_single_threaded_ctx


class TestImportBlockWithMockPipeline(unittest.TestCase):
    @patch("instructlab.sdg.pipeline.Pipeline")
    def setUp(self, mock_pipeline):
        self.ctx = get_single_threaded_ctx()
        self.pipe = MagicMock()
        self.block_name = "test_block"
        self.path = "/path/to/config"
        self.mock_pipeline = mock_pipeline
        self.import_block = ImportBlock(self.ctx, self.pipe, self.block_name, self.path)
        self.dataset = Dataset.from_dict({})

    def test_initialization(self):
        self.assertEqual(self.import_block.block_name, self.block_name)
        self.assertEqual(self.import_block.path, self.path)
        self.mock_pipeline.from_file.assert_called_once_with(self.ctx, self.path)

    def test_generate(self):
        self.mock_pipeline.from_file.return_value.generate.return_value = self.dataset
        samples = self.import_block.generate(self.dataset)
        self.mock_pipeline.from_file.return_value.generate.assert_called_once_with(
            samples
        )
        self.assertEqual(samples, self.dataset)


_CHILD_YAML = """\
version: "1.0"
blocks:
- name: greater_than_thirty
  type: FilterByValueBlock
  config:
    filter_column: age
    filter_value: 30
    operation: gt
    convert_dtype: int
"""


_PARENT_YAML_FMT = """\
version: "1.0"
blocks:
- name: forty_or_under
  type: FilterByValueBlock
  config:
    filter_column: age
    filter_value: 40
    operation: le
    convert_dtype: int
    default_value: 1000
- name: import_child
  type: ImportBlock
  config:
    path: %s
- name: big_bdays
  type: FilterByValueBlock
  config:
    filter_column: age
    filter_value:
    - 30
    - 40
    operation: eq
    convert_dtype: int
"""


class TestImportBlockWithFilterByValue(unittest.TestCase):
    def setUp(self):
        self.ctx = get_single_threaded_ctx()
        self.child_yaml = self._write_tmp_yaml(_CHILD_YAML)
        self.parent_yaml = self._write_tmp_yaml(_PARENT_YAML_FMT % self.child_yaml)
        self.dataset = Dataset.from_dict(
            {"age": ["25", "30", "35", "40", "45"]},
            features=Features({"age": Value("string")}),
        )

    def tearDown(self):
        os.remove(self.parent_yaml)
        os.remove(self.child_yaml)

    def _write_tmp_yaml(self, content):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
        tmp_file.write(content)
        tmp_file.close()
        return tmp_file.name

    def test_generate(self):
        pipeline = Pipeline.from_file(self.ctx, self.parent_yaml)
        filtered_dataset = pipeline.generate(self.dataset)
        self.assertEqual(len(filtered_dataset), 1)
        self.assertEqual(filtered_dataset["age"], [40])
