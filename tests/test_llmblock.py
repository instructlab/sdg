# Standard
from unittest.mock import MagicMock, patch
import unittest

# First Party
from src.instructlab.sdg.llmblock import LLMBlock


class TestLLMBlockModelPrompt(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.config_return_value = {
            "system": "system",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }

    @patch("src.instructlab.sdg.block.Block._load_config")
    def test_model_prompt_empty_string(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="test_block",
            config_path="",
            output_cols=[],
            model_prompt="",
        )
        self.assertEqual(
            block.model_prompt,
            "",
            "model_prompt should be an empty string when explicitly set to an empty string",
        )

    @patch("src.instructlab.sdg.block.Block._load_config")
    def test_model_prompt_none(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        # Ensure that if a custom model_prompt is not specified, it defaults to setting it to
        # something based on the model family. For this we just make sure it's not an empty string.
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="test_block",
            config_path="",
            output_cols=[],
            model_prompt=None,  # Or simply omit model_prompt as it defaults to None
        )
        self.assertNotEqual(
            block.model_prompt,
            "",
            "model_prompt should be a non-empty string when set to None",
        )
