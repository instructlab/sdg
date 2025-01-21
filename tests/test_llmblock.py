# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import resources
from unittest.mock import MagicMock, patch
import os
import unittest

# Third Party
from datasets import Dataset, Features, Value
from httpx import URL
from jinja2 import StrictUndefined, Template, UndefinedError
from openai import InternalServerError, NotFoundError
import pytest

# First Party
from src.instructlab.sdg import (
    BlockConfigParserError,
    ConditionalLLMBlock,
    LLMBlock,
    LLMLogProbBlock,
    LLMMessagesBlock,
)
from src.instructlab.sdg.blocks.llmblock import server_supports_batched


@patch("src.instructlab.sdg.blocks.block.Block._load_config")
class TestLLMBlockModelPrompt(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.config_return_value = {
            "system": "{{fruit}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }
        self.dataset = Dataset.from_dict(
            {"fruit": ["apple", "pear", "mango"]},
            features=Features({"fruit": Value("string")}),
        )

    def test_model_prompt_empty_string(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        # Ensure that if an empty model_prompt is specified, no model prompt is used.
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

    def test_model_prompt_custom(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        # Ensure that if a custom model_prompt is specified, it is used correctly
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="test_block",
            config_path="",
            output_cols=[],
            model_prompt="FOO {{prompt}} BAR",
        )
        prompt = block._format_prompt(self.dataset[1])
        self.assertEqual(
            prompt,
            "FOO pear\nintroduction\nprinciples\nexamples\ngeneration BAR",
            "custom model_prompt was not used when explicitly set",
        )


class TestLLMBlockWithRealConfigs(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()

    def test_configs_with_invalid_sample(self):
        for config_type in ["knowledge", "skills"]:
            for config_yaml in resources.files(
                f"instructlab.sdg.configs.{config_type}"
            ).iterdir():
                if config_yaml.suffix != ".yaml":
                    continue
                block = LLMBlock(
                    ctx=self.mock_ctx,
                    pipe=self.mock_pipe,
                    block_name=config_yaml.stem,
                    config_path=config_yaml,
                    output_cols=[],
                )
                sample = {"foo": "bar"}
                assert not block._validate(
                    block.prompt_template, sample
                ), f"{config_type} config {config_yaml.name} validated even though it was given a sample with none of the expected fields"

    def test_simple_generate_qa_with_valid_sample(self):
        config_yaml = os.path.join(
            resources.files("instructlab.sdg.configs.knowledge"),
            "simple_generate_qa.yaml",
        )
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_path=config_yaml,
            output_cols=[],
        )
        sample = {
            "domain": "domain goes here",
            "document": "document goes here",
            "document_outline": "document outline goes here",
            "icl_document": "context goes here",
            "icl_query_1": "query 1 goes here",
            "icl_response_1": "response 1 goes here",
            "icl_query_2": "query 2 goes here",
            "icl_response_2": "response 2 goes here",
            "icl_query_3": "query 3 goes here",
            "icl_response_3": "response 3 goes here",
        }
        assert block._validate(block.prompt_template, sample)


@patch("src.instructlab.sdg.blocks.block.Block._load_config")
class TestLLMBlockOtherFunctions(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.config_return_value = {
            "system": "{{fruit}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }

    def test_max_num_tokens_override(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        self.mock_ctx.max_num_tokens = 512
        # Ensure that if max_tokens is specified, it is used correctly
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

    def test_validate(self, mock_load_config):
        mock_load_config.return_value = {
            "system": "{{var1}} {{var2}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }
        block = LLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_path="",
            output_cols=[],
        )

        assert not block._validate(block.prompt_template, {})
        assert block._validate(block.prompt_template, {"var1": "foo", "var2": "bar"})


class TestLLMBlockBatching(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.mock_client = MagicMock()
        self.mock_client.server_supports_batched = None
        self.mock_client.base_url = URL("http://localhost:8000/v1")
        self.mock_client.get = MagicMock()
        self.mock_ctx.client = self.mock_client

    def test_server_supports_batched_llama_cpp(self):
        resp_text = """{"message":"Hello from InstructLab! Visit us at https://instructlab.ai"}"""
        self.mock_client.get.return_value = MagicMock()
        self.mock_client.get().text = resp_text
        supports_batched = server_supports_batched(self.mock_ctx.client, "my-model")
        assert not supports_batched

    def test_server_supports_batched_other_llama_cpp(self):
        resp_text = "another server"
        self.mock_client.get.return_value = MagicMock()
        self.mock_client.get().text = resp_text
        mock_completion = MagicMock()
        mock_completion.create = MagicMock()
        mock_completion.create.side_effect = InternalServerError(
            "mock error",
            response=MagicMock(),
            body=MagicMock(),
        )
        self.mock_client.completions = mock_completion
        supports_batched = server_supports_batched(self.mock_ctx.client, "my-model")
        assert not supports_batched

    def test_server_supports_batched_vllm(self):
        self.mock_client.get.side_effect = NotFoundError(
            "mock error",
            response=MagicMock(),
            body=MagicMock(),
        )
        mock_completion_resp = MagicMock()
        mock_completion_resp.choices = ["a", "b", "c", "d", "e", "f"]
        mock_completion = MagicMock()
        mock_completion.create = MagicMock()
        mock_completion.create.return_value = mock_completion_resp
        self.mock_client.completions = mock_completion
        supports_batched = server_supports_batched(self.mock_ctx.client, "my-model")
        assert supports_batched


@patch("src.instructlab.sdg.blocks.block.Block._load_config")
class TestConditionalLLMBlock(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()

    def test_invalid_config_paths(self, _mock_load_config):
        with pytest.raises(BlockConfigParserError) as exc:
            ConditionalLLMBlock(
                ctx=self.mock_ctx,
                pipe=self.mock_pipe,
                block_name="gen_knowledge",
                config_paths=[],
                output_cols=[],
                selector_column_name="selector",
            )
        assert "at least one entry" in str(exc.value)

        with pytest.raises(BlockConfigParserError) as exc:
            ConditionalLLMBlock(
                ctx=self.mock_ctx,
                pipe=self.mock_pipe,
                block_name="gen_knowledge",
                config_paths=[["foo"]],
                output_cols=[],
                selector_column_name="selector",
            )
        assert "config path and selector" in str(exc.value)

    def test_validate(self, mock_load_config):
        mock_load_config.return_value = {
            "system": "{{var1}} {{var2}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }
        block = ConditionalLLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_paths=[["/foo/bar", "_A_"]],
            output_cols=[],
            selector_column_name="selector",
        )

        assert not block._validate(block.prompt_template, {})
        assert not block._validate(
            block.prompt_template, {"selector": "_B_", "var1": "foo", "var2": "bar"}
        )
        assert block._validate(
            block.prompt_template, {"selector": "_A_", "var1": "foo", "var2": "bar"}
        )

    def test_format_prompt_with_jinja_templates(self, mock_load_config):
        mock_load_config.return_value = {
            "system": "{{var1}} {{var2}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }
        # Create an instance of the block
        block = ConditionalLLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_paths=[["/foo/bar", "_A_"]],
            output_cols=[],
            selector_column_name="selector",
        )

        # Mock prompt_template as a Jinja template
        block.prompt_template = {
            "case1": Template("Hello, {{ name }}!", undefined=StrictUndefined),
            "case2": Template("Goodbye, {{ name }}.", undefined=StrictUndefined),
        }

        # Sample input
        sample1 = {"selector": "case1", "name": "Alice"}
        sample2 = {"selector": "case2", "name": "Bob"}

        # Test case 1
        result1 = block._format_prompt(sample1)
        self.assertEqual(result1, "Hello, Alice!")

        # Test case 2
        result2 = block._format_prompt(sample2)
        self.assertEqual(result2, "Goodbye, Bob.")

        # Verify error with missing keys in sample
        with self.assertRaises(UndefinedError):
            block._format_prompt({"selector": "case1"})  # 'name' is missing

    def test_format_prompt_without_selector(self, mock_load_config):
        mock_load_config.return_value = {
            "system": "{{var1}} {{var2}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }
        # Create an instance of the block
        block = ConditionalLLMBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_paths=[["/foo/bar", "_A_"]],
            output_cols=[],
            selector_column_name=None,  # No selector
        )

        # Mock prompt_template as a Jinja template
        block.prompt_template = Template(
            "Welcome, {{ user }}!", undefined=StrictUndefined
        )

        # Sample input
        sample = {"user": "Connor"}

        # Test
        result = block._format_prompt(sample)
        self.assertEqual(result, "Welcome, Connor!")

        # Verify error with missing keys in sample
        with self.assertRaises(UndefinedError):
            block._format_prompt({})  # 'user' is missing


@patch("src.instructlab.sdg.blocks.block.Block._load_config")
class TestLLMLogProbBlock(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.config_return_value = {
            "system": "{{fruit}}",
            "introduction": "introduction",
            "principles": "principles",
            "examples": "examples",
            "generation": "generation",
        }

    def test_constructor_works(self, mock_load_config):
        mock_load_config.return_value = self.config_return_value
        block = LLMLogProbBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            config_path="",
            output_cols=[],
        )
        assert block is not None


class TestLLMMessagesBlock(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.model_family = "mixtral"
        self.mock_ctx.model_id = "test_model"
        self.mock_pipe = MagicMock()
        self.mock_client = MagicMock()
        self.mock_ctx.client = self.mock_client

    def test_constructor_works(self):
        block = LLMMessagesBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            input_col="messages",
            output_col="output",
        )
        assert block is not None

    def test_temperature_validation(self):
        block = LLMMessagesBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            input_col="messages",
            output_col="output",
            gen_kwargs={"n": 5, "temperature": 0},
        )
        assert block.gen_kwargs["temperature"] != 0

        block = LLMMessagesBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            input_col="messages",
            output_col="output",
            gen_kwargs={"n": 1, "temperature": 0},
        )
        assert block.gen_kwargs["temperature"] == 0

    def test_calls_chat_completion_api(self):
        # Mock the OpenAI client so we don't actually hit a server here
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = "generated response"
        mock_completion_resp = MagicMock()
        mock_completion_resp.choices = [mock_choice]
        mock_completion = MagicMock()
        mock_completion.create = MagicMock()
        mock_completion.create.return_value = mock_completion_resp
        mock_chat = MagicMock()
        mock_chat.completions = mock_completion
        self.mock_client.chat = mock_chat
        block = LLMMessagesBlock(
            ctx=self.mock_ctx,
            pipe=self.mock_pipe,
            block_name="gen_knowledge",
            input_col="messages",
            output_col="output",
            gen_kwargs={"n": 1, "temperature": 0},
        )
        samples = Dataset.from_dict(
            {"messages": ["my message"]},
            features=Features({"messages": Value("string")}),
        )
        output = block.generate(samples)
        assert len(output) == 1
        assert "output" in output.column_names
        mock_completion.create.assert_called()
        assert mock_completion.create.call_args.kwargs["messages"] == "my message"
