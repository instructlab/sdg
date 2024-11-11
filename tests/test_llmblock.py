# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from datasets import Dataset, Features, Value
from httpx import URL
from openai import InternalServerError, NotFoundError, OpenAI

# First Party
from src.instructlab.sdg.llmblock import LLMBlock, server_supports_batched


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

    @patch("src.instructlab.sdg.block.Block._load_config")
    def test_max_num_tokens_override(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        self.mock_ctx.max_num_tokens = 512
        # Ensure that if a custom model_prompt is specified, it is used correctly
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_path="",
            output_cols=[],
            model_prompt="",
            gen_kwargs={"max_tokens": 2048},
        )
        num_tokens = block.gen_kwargs["max_tokens"]
        assert num_tokens == 512

    def test_server_supports_batched_llama_cpp(self):
        resp_text = """{"message":"Hello from InstructLab! Visit us at https://instructlab.ai"}"""
        mock_client = MagicMock()
        mock_client.server_supports_batched = None
        mock_client.base_url = URL("http://localhost:8000/v1")
        mock_client.get = MagicMock()
        mock_client.get.return_value = MagicMock()
        mock_client.get().text = resp_text
        self.mock_ctx.client = mock_client
        supports_batched = server_supports_batched(self.mock_ctx.client, "my-model")
        assert not supports_batched

    def test_server_supports_batched_other_llama_cpp(self):
        resp_text = "another server"
        mock_client = MagicMock()
        mock_client.server_supports_batched = None
        mock_client.base_url = URL("http://localhost:8000/v1")
        mock_client.get = MagicMock()
        mock_client.get.return_value = MagicMock()
        mock_client.get().text = resp_text
        mock_completion = MagicMock()
        mock_completion.create = MagicMock()
        mock_completion.create.side_effect = InternalServerError(
            "mock error",
            response=MagicMock(),
            body=MagicMock(),
        )
        mock_client.completions = mock_completion
        self.mock_ctx.client = mock_client
        supports_batched = server_supports_batched(self.mock_ctx.client, "my-model")
        assert not supports_batched

    def test_server_supports_batched_vllm(self):
        mock_client = MagicMock()
        mock_client.server_supports_batched = None
        mock_client.base_url = URL("http://localhost:8000/v1")
        mock_client.get = MagicMock()
        mock_client.get.side_effect = NotFoundError(
            "mock error",
            response=MagicMock(),
            body=MagicMock(),
        )
        mock_completion_resp = MagicMock()
        mock_completion_resp.choices = ["a", "b", "c", "d", "e", "f"]
        mock_completion = MagicMock()
        mock_completion.create = MagicMock()
        mock_completion.create.return_value = mock_completion_resp
        mock_client.completions = mock_completion
        self.mock_ctx.client = mock_client
        supports_batched = server_supports_batched(self.mock_ctx.client, "my-model")
        assert supports_batched
