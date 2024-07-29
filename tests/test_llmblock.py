# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from datasets import Dataset, Features, Value

# First Party
from src.instructlab.sdg.llmblock import LLMBlock


class TestLLMBlockModelPrompt(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.config_return_value = {
            "system": "{fruit}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }
        self.dataset = Dataset.from_dict(
            {"fruit": ["apple", "pear", "mango"]},
            features=Features({"fruit": Value("string")}),
        )

    @patch("src.instructlab.sdg.block.Block._load_config")
    def test_model_prompt_empty_string(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        # Ensure that if an empty model_prompt is not specified, no model prompt is used.
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="test_block",
            config_path="",
            output_cols=[],
            model_prompt="",
        )
        prompt = block._format_prompt(self.dataset[0])
        self.assertEqual(
            prompt,
            "apple\nintroduction\nprinciples\nexamples\ngeneration",
            "no model prompt should be used when explicitly set to an empty string",
        )

    @patch("src.instructlab.sdg.block.Block._load_config")
    def test_model_prompt_none(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        # Ensure that if a custom model_prompt is not specified, it defaults to setting it to
        # something based on the model family (i.e. mixtral).
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="test_block",
            config_path="",
            output_cols=[],
            model_prompt=None,  # Or simply omit model_prompt as it defaults to None
        )
        prompt = block._format_prompt(self.dataset[1])
        self.assertEqual(
            prompt,
            "<s> [INST] pear\nintroduction\nprinciples\nexamples\ngeneration [/INST]",
            "model_prompt based on model_family should be used set to None",
        )

    @patch("src.instructlab.sdg.block.Block._load_config")
    def test_model_prompt_none(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        # Ensure that if a custom model_prompt is specified, it is used correctly
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="test_block",
            config_path="",
            output_cols=[],
            model_prompt="FOO {prompt} BAR",
        )
        prompt = block._format_prompt(self.dataset[1])
        self.assertEqual(
            prompt,
            "FOO pear\nintroduction\nprinciples\nexamples\ngeneration BAR",
            "model_prompt should be a non-empty string when set to None",
        )
