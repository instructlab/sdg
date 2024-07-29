# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import resources
from unittest.mock import patch
import unittest

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg.filterblock import FilterByValueBlock
from instructlab.sdg.llmblock import ConditionalLLMBlock, LLMBlock
from instructlab.sdg.pipeline import Pipeline, PipelineContext
from instructlab.sdg.utilblocks import (
    CombineColumnsBlock,
    DuplicateColumnsBlock,
    FlattenColumnsBlock,
    RenameColumnsBlock,
    SamplePopulatorBlock,
    SelectorBlock,
)


def _noop_generate(self, samples):
    return samples


@patch.object(CombineColumnsBlock, "generate", _noop_generate)
@patch.object(ConditionalLLMBlock, "generate", _noop_generate)
@patch.object(DuplicateColumnsBlock, "generate", _noop_generate)
@patch.object(FilterByValueBlock, "generate", _noop_generate)
@patch.object(FlattenColumnsBlock, "generate", _noop_generate)
@patch.object(LLMBlock, "generate", _noop_generate)
@patch.object(RenameColumnsBlock, "generate", _noop_generate)
@patch.object(SamplePopulatorBlock, "generate", _noop_generate)
@patch.object(SelectorBlock, "generate", _noop_generate)
@patch("instructlab.sdg.llmblock.server_supports_batched", lambda c, m: True)
@patch.object(Pipeline, "_drop_duplicates", lambda self, dataset, cols: dataset)
class TestDefaultPipelineConfigs(unittest.TestCase):
    def setUp(self):
        self._yaml_files = [
            file
            for package in [
                "instructlab.sdg.pipelines.simple",
                "instructlab.sdg.pipelines.full",
            ]
            for file in resources.files(package).iterdir()
            if file.suffix == ".yaml"
        ]

    def test_pipeline_from_config(self):
        ctx = PipelineContext(
            client=None,
            model_family="mixtral",
            model_id="model",
            num_instructions_to_generate=1,
        )
        for pipeline_yaml in self._yaml_files:
            pipeline = Pipeline.from_file(ctx, pipeline_yaml)
            output = pipeline.generate(Dataset.from_list([{"test": "test"}]))
            self.assertIsNotNone(output)
