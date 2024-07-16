# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict
import re

# Third Party
from datasets import Dataset
import openai

# Local
from .block import Block
from .logger_config import setup_logger

logger = setup_logger(__name__)

MODEL_FAMILY_MIXTRAL = "mixtral"
MODEL_FAMILY_MERLINITE = "merlinite"

_MODEL_PROMPT_MIXTRAL = "<s> [INST] {prompt} [/INST]"
_MODEL_PROMPT_MERLINITE = "'<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{prompt}\n<|assistant|>\n'"

_MODEL_PROMPTS = {
    MODEL_FAMILY_MIXTRAL: _MODEL_PROMPT_MIXTRAL,
    MODEL_FAMILY_MERLINITE: _MODEL_PROMPT_MERLINITE,
}


def _get_model_prompt(model_family):
    if model_family not in _MODEL_PROMPTS:
        raise ValueError(f"Unknown model family: {model_family}")
    return _MODEL_PROMPTS[model_family]


def server_supports_batched(client, model_id: str) -> bool:
    supported = getattr(client, "server_supports_batched", None)
    if supported is not None:
        return supported
    try:
        # Make a test call to the server to determine whether it supports
        # multiple input prompts per request and also the n parameter
        response = client.completions.create(
            model=model_id, prompt=["test1", "test2"], max_tokens=1, n=3
        )
        # Number outputs should be 2 * 3 = 6
        supported = len(response.choices) == 6
    except openai.InternalServerError:
        supported = False
    client.server_supports_batched = supported
    logger.info(f"LLM server supports batched inputs: {client.server_supports_batched}")
    return supported


# This is part of the public API.
# pylint: disable=dangerous-default-value
class LLMBlock(Block):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        config_path,
        output_cols,
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = (
            """{system}\n{introduction}\n{principles}\n{examples}\n{generation}"""
        )
        self.prompt_template = self.prompt_struct.format(**self.block_config)
        self.model_prompt = _get_model_prompt(self.ctx.model_family)
        self.output_cols = output_cols
        self.batch_params = batch_kwargs
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)
        self.defaults = {
            "model": self.ctx.model_id,
            "temperature": 0,
            "max_tokens": 4096,
        }

        # Whether the LLM server supports a list of input prompts
        # and supports the n parameter to generate n outputs per input
        self.server_supports_batched = server_supports_batched(
            self.ctx.client, self.ctx.model_id
        )

    def _parse(self, generated_string) -> dict:
        matches = {}

        if self.parser_name is not None and self.parser_name == "custom":
            pattern = re.compile(self.parsing_pattern, re.DOTALL)
            all_matches = pattern.findall(generated_string)
            matches = {column_name: [] for column_name in self.output_cols}
            if all_matches and isinstance(all_matches[0], tuple):
                for match in all_matches:
                    for column_name, value in zip(self.output_cols, match):
                        value = value.strip()
                        for clean_tag in self.parser_cleanup_tags:
                            value = value.replace(clean_tag, "")
                        matches[column_name].append(value)
            else:
                matches[self.output_cols[0]] = (
                    [match.strip() for match in all_matches] if all_matches else []
                )
        else:
            for start_tag, end_tag, output_col in zip(
                self.block_config.get("start_tags", []),
                self.block_config.get("end_tags", []),
                self.output_cols,
            ):
                if not start_tag and not end_tag:
                    matches[output_col] = [
                        generated_string.strip() if generated_string else None
                    ]
                else:
                    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
                    all_matches = re.findall(pattern, generated_string, re.DOTALL)
                    matches[output_col] = (
                        [match.strip() for match in all_matches] if all_matches else []
                    )

        return matches

    def _format_prompt(self, sample: Dict) -> str:
        return self.prompt_template.format(**sample).strip()

    def _gen_kwargs(self, **gen_kwargs):
        gen_kwargs = {**self.defaults, **gen_kwargs}
        if "max_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        if "temperature" in gen_kwargs:
            gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        return gen_kwargs

    def _generate(self, samples, **gen_kwargs) -> list:
        prompts = [
            self.model_prompt.format(prompt=self._format_prompt(sample))
            for sample in samples
        ]
        generate_args = self._gen_kwargs(**gen_kwargs)

        if self.server_supports_batched:
            response = self.ctx.client.completions.create(
                prompt=prompts, **generate_args
            )
            return [choice.text.strip() for choice in response.choices]

        n = gen_kwargs.get("n", 1)
        results = []
        for prompt in prompts:
            for _ in range(n):
                response = self.ctx.client.completions.create(
                    prompt=prompt, **generate_args
                )
                results.append(response.choices[0].text.strip())
        return results

    def generate(self, samples: Dataset, **gen_kwargs) -> Dataset:
        """
        Generate the output from the block. This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.

        :return: The parsed output after generation.
        """
        num_samples = self.batch_params.get("num_samples", None)
        logger.debug("Generating outputs for {} samples".format(len(samples)))

        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))

        # validate each sample
        # Log errors and remove invalid samples
        valid_samples = []

        for sample in samples:
            if self._validate(self.prompt_template, sample):
                valid_samples.append(sample)
            else:
                logger.warning(
                    f"Sample failed validation: {sample}"
                )  # Log details of the failed sample

        samples = valid_samples

        if len(samples) == 0:
            return Dataset.from_list([])

        # generate the output

        outputs = self._generate(samples, **gen_kwargs)
        logger.debug("Generated outputs: %s", outputs)

        num_parallel_samples = gen_kwargs.get("n", 1)
        extended_samples = []

        # Duplicate each input sample n times, where n is the number
        # of output sequences generated per input, so that we can
        # pair up the inputs and outputs.
        for item in samples:
            extended_samples.extend([item] * num_parallel_samples)

        new_data = []
        for sample, output in zip(extended_samples, outputs):
            parsed_outputs = self._parse(output)
            max_length = max(len(value) for value in parsed_outputs.values())
            for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})

        return Dataset.from_list(new_data)


# This is part of the public API.
class ConditionalLLMBlock(LLMBlock):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        config_paths,
        output_cols,
        selector_column_name,
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(
            ctx,
            pipe,
            block_name,
            config_paths[0][0],
            output_cols,
            parser_kwargs=parser_kwargs,
            batch_kwargs=batch_kwargs,
        )
        self.selector_column_name = selector_column_name
        self.prompt_template = {}
        if len(config_paths) == 1 and config_paths[0][1] == "All":
            self.prompt_template = self.prompt_struct.format(**self.block_config)
        else:
            for config, config_key in config_paths:
                self.prompt_template[config_key] = self.prompt_struct.format(
                    **self._load_config(config)
                )

    def _format_prompt(self, sample: Dict) -> str:
        if isinstance(self.prompt_template, dict):
            return (
                self.prompt_template[sample[self.selector_column_name]]
                .format(**sample)
                .strip()
            )

        return self.prompt_template.format(**sample).strip()

    def validate(self, prompt_template: str, input_dict: Dict[str, Any]) -> bool:
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template[input_dict[self.selector_column_name]]
        return super()._validate(prompt_template, input_dict)
