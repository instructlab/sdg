# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict
import logging
import re

# Third Party
from datasets import Dataset
from tqdm import tqdm
import httpx
import openai

# Local
# Import prompts to register default chat templates
from .. import prompts as default_prompts  # pylint: disable=unused-import
from ..registry import BlockRegistry, PromptRegistry
from .block import Block, BlockConfigParserError

logger = logging.getLogger(__name__)

DEFAULT_MAX_NUM_TOKENS = 4096


def server_supports_batched(client, model_id: str) -> bool:
    supported = getattr(client, "server_supports_batched", None)
    if supported is not None:
        return supported
    # Start looking for InstructLab's default llama-cpp-python so we
    # can avoid throwing an assertion error in the server, as
    # llama-cpp-python does not like us explicitly testing batches
    if "/v1" in client.base_url.path:
        try:
            # The root (without /v1) will have InstructLab's welcome
            # message
            http_res = client.get("../", cast_to=httpx.Response)
            if "Hello from InstructLab" in http_res.text:
                # The server is llama-cpp-python, so disable batching
                supported = False
        except openai.APIStatusError:
            # The server is not InstructLab's llama-cpp-python
            pass
    if supported is None:
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


def template_from_struct_and_config(struct, config):
    # replace None with empty strings
    filtered_config = {k: (v if v is not None else "") for k, v in config.items()}
    return PromptRegistry.template_from_string(struct.format(**filtered_config))


# This is part of the public API.
@BlockRegistry.register("LLMBlock")
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
        model_prompt=None,
        gen_kwargs={},
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = (
            """{system}\n{introduction}\n{principles}\n{examples}\n{generation}"""
        )
        self.prompt_template = template_from_struct_and_config(
            self.prompt_struct, self.block_config
        )
        self.model_prompt = model_prompt
        self.output_cols = output_cols
        self.batch_params = batch_kwargs
        max_num_token_override = ctx.max_num_tokens
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)
        # max_num_tokens should only be applicable to knowledge blocks
        # gen_knowledge if the full/simple pipeline's knowledge generation block
        if block_name != "gen_knowledge":
            logger.debug(
                f"Not applying max_num_tokens to block {block_name}. This is only applicable for gen_knowledge."
            )
            max_num_token_override = DEFAULT_MAX_NUM_TOKENS
        self.gen_kwargs = self._gen_kwargs(
            max_num_token_override,
            gen_kwargs,
            model=self.ctx.model_id,
            temperature=0,
            max_tokens=DEFAULT_MAX_NUM_TOKENS,
        )
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

    # There are three cases to handle for self.model_prompt
    # 1. None - no model_prompt specified, look one up based on model family
    # 2. Non-empty string - the pipeline has specified a custom model prompt
    # 3. Empty string - the pipeline has specified that no model prompt is needed
    def _format_prompt(self, sample: Dict) -> str:
        prompt_templated_str = self.prompt_template.render(sample).strip()

        model_prompt = None
        if self.model_prompt is None:
            model_prompt = PromptRegistry.get_template(self.ctx.model_family)
        elif self.model_prompt:
            model_prompt = PromptRegistry.template_from_string(self.model_prompt)
        else:
            # Our model prompt is an empty string, which we'll render
            # verbatim without wrapping in the messages format
            model_prompt = PromptRegistry.get_template("blank")

        messages = [{"role": "user", "content": prompt_templated_str}]

        return model_prompt.render(
            messages=messages,
            prompt=prompt_templated_str,
            add_generation_prompt=True,
        ).strip()

    def _gen_kwargs(self, max_num_token_override, gen_kwargs, **defaults):
        gen_kwargs = {**defaults, **gen_kwargs}
        if (
            "n" in gen_kwargs
            and isinstance(gen_kwargs["n"], str)
            and gen_kwargs["n"] == "scaled"
        ):
            gen_kwargs["n"] = self.ctx.num_instructions_to_generate
        if "temperature" in gen_kwargs:
            gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        if max_num_token_override != DEFAULT_MAX_NUM_TOKENS:
            gen_kwargs["max_tokens"] = max_num_token_override
        elif "max_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        return gen_kwargs

    def _generate(self, samples) -> list:
        prompts = [self._format_prompt(sample) for sample in samples]
        logger.debug(f"STARTING GENERATION FOR LLMBlock USING PROMPTS: {prompts}")
        logger.debug(f"Generation arguments: {self.gen_kwargs}")
        if self.server_supports_batched:
            response = self.ctx.client.completions.create(
                prompt=prompts, **self.gen_kwargs
            )
            return [choice.text.strip() for choice in response.choices]

        results = []
        progress_bar = tqdm(
            range(len(prompts)), desc=f"{self.block_name} Prompt Generation"
        )
        for prompt in prompts:
            logger.debug(f"CREATING COMPLETION FOR PROMPT: {prompt}")
            for _ in range(self.gen_kwargs.get("n", 1)):
                response = self.ctx.client.completions.create(
                    prompt=prompt, **self.gen_kwargs
                )
                results.append(response.choices[0].text.strip())
                progress_bar.update(1)
        return results

    def generate(self, samples: Dataset) -> Dataset:
        """
        Generate the output from the block. This method should first validate the input data,
        then generate the output, and finally parse the generated output before returning it.

        Args:
            samples (Dataset): The samples used as input data

        Returns:
            The parsed output after generation.
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

        outputs = self._generate(samples)
        logger.debug("Generated outputs: %s", outputs)

        num_parallel_samples = self.gen_kwargs.get("n", 1)
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
@BlockRegistry.register("ConditionalLLMBlock")
class ConditionalLLMBlock(LLMBlock):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        config_paths,
        output_cols,
        selector_column_name,
        model_prompt=None,
        gen_kwargs={},
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        if not config_paths:
            raise BlockConfigParserError(
                f"ConditionalLLMBlock config_paths of block {block_name} requires at least one entry"
            )
        for config_path in config_paths:
            if len(config_path) != 2:
                raise BlockConfigParserError(
                    f"ConditionalLLMBlock config_paths of block {block_name} should be a list of config path and selector column names"
                )
        super().__init__(
            ctx,
            pipe,
            block_name,
            config_paths[0][0],
            output_cols,
            model_prompt=model_prompt,
            gen_kwargs=gen_kwargs,
            parser_kwargs=parser_kwargs,
            batch_kwargs=batch_kwargs,
        )
        self.selector_column_name = selector_column_name
        self.prompt_template = {}
        if len(config_paths) == 1 and config_paths[0][1] == "All":
            self.prompt_template = template_from_struct_and_config(
                self.prompt_struct, self.block_config
            )
        else:
            for config, config_key in config_paths:
                self.prompt_template[config_key] = template_from_struct_and_config(
                    self.prompt_struct, self._load_config(config)
                )

    def _format_prompt(self, sample: Dict) -> str:
        # If prompt_template is a dict, use the selector column to select the correct template
        if isinstance(self.prompt_template, dict):
            return (
                self.prompt_template[sample[self.selector_column_name]]
                .render(sample)
                .strip()
            )
        # Otherwise, use the sample to render the prompt without any selection
        return self.prompt_template.render(sample).strip()

    def _validate(self, prompt_template: str, input_dict: Dict[str, Any]) -> bool:
        if isinstance(prompt_template, dict):
            if not self.selector_column_name in input_dict:
                logger.error(
                    f"ConditionalLLMBlock {self.block_name} missing key: {self.selector_column_name}"
                )
                return False
            config_key = input_dict[self.selector_column_name]
            if not config_key in prompt_template:
                logger.error(
                    f"ConditionalLLMBlock {self.block_name} selector key {config_key} not found in block config"
                )
                return False
            prompt_template = prompt_template[config_key]
        return super()._validate(prompt_template, input_dict)


# This is part of the public API.
@BlockRegistry.register("LLMLogProbBlock")
class LLMLogProbBlock(LLMBlock):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        config_path,
        output_cols,
        model_prompt=None,
        gen_kwargs={},
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(
            ctx,
            pipe,
            block_name,
            config_path,
            output_cols,
            model_prompt=model_prompt,
            gen_kwargs=gen_kwargs,
            parser_kwargs=parser_kwargs,
            batch_kwargs=batch_kwargs,
        )

    # def _generate_logprobs(self, samples, **gen_kwargs):
    #     prompts = [
    #         self.model_prompt.format(prompt=self._format_prompt(sample))
    #         for sample in samples
    #     ]
    #     generate_args = {**self.defaults, **gen_kwargs}

    #     # verify if logprobs is mentioned in the generate_args, if not add it and return top10 logprobs
    #     if "logprobs" not in generate_args:
    #         generate_args["logprobs"] = 10

    #     if self.server_supports_batched:
    #         response = self.client.completions.create(prompt=prompts, **generate_args)
    #         return [choice.logprobs.top_logprobs for choice in response.choices]

    #     n = gen_kwargs.get("n", 1)
    #     results = []
    #     for prompt in prompts:
    #         for _ in range(n):
    #             response = self.client.completions.create(
    #                 prompt=prompt, **generate_args
    #             )
    #             results.append(response.choices[0].logprobs.top_logprobs)
    #     return results

    # def _parse(self, generations: List[List[Dict]]) -> List[List[str]]:
    #     # override the parse method to convert the generations to json string
    #     # convert the generations to json string to save as dataset
    #     # this is because the dataset can only store key value pairs which are consistent
    #     return [[json.dumps(item) for item in sublist] for sublist in generations]

    # def generate(self, samples: Dataset, **gen_kwargs) -> Dataset:
    #     """
    #     Generate the output from the block. This method should first validate the input data,
    #     then generate the output, and finally parse the generated output before returning it.

    #     Returns:
    #         The parsed output after generation.
    #     """
    #     num_samples = self.block_config.get("num_samples", None)
    #     logger.debug("Generating outputs for {} samples".format(len(samples)))

    #     if (num_samples is not None) and ("num_samples" not in samples.column_names):
    #         samples = samples.add_column("num_samples", [num_samples] * len(samples))

    #     # validate each sample
    #     # Log errors and remove invalid samples
    #     valid_samples = []

    #     for sample in samples:
    #         if self._validate(self.prompt_template, sample):
    #             valid_samples.append(sample)
    #         else:
    #             logger.warning(
    #                 f"Sample failed validation: {sample}"
    #             )  # Log details of the failed sample

    #     samples = valid_samples

    #     if len(samples) == 0:
    #         logger.warning(
    #             "No valid samples to generate outputs for, returning empty dataset"
    #         )
    #         return Dataset.from_list([])

    #     # generate the output

    #     outputs = self._generate_logprobs(samples, **gen_kwargs)
    #     logger.debug("Generated outputs: %s", outputs)

    #     output_dataset = Dataset.from_list(samples)
    #     output_dataset = output_dataset.add_column(
    #         self.output_cols[0],
    #         self._parse(outputs),  # pylint: disable=no-value-for-parameter
    #     )

    #     return output_dataset


# This is part of the public API.
@BlockRegistry.register("LLMMessagesBlock")
class LLMMessagesBlock(Block):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        input_col,
        output_col,
        gen_kwargs={},
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.input_col = input_col
        self.output_col = output_col
        self.gen_kwargs = self._gen_kwargs(
            gen_kwargs,
            model=self.ctx.model_id,
            temperature=0,
            max_tokens=DEFAULT_MAX_NUM_TOKENS,
        )

    def _gen_kwargs(self, gen_kwargs, **defaults):
        gen_kwargs = {**defaults, **gen_kwargs}
        if "temperature" in gen_kwargs:
            gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        if (
            "n" in gen_kwargs
            and gen_kwargs["n"] > 1
            and gen_kwargs.get("temperature", 0) <= 0
        ):
            gen_kwargs["temperature"] = 0.7
            logger.warning(
                "Temperature should be greater than 0 for n > 1, setting temperature to 0.7"
            )
        return gen_kwargs

    def _generate(self, samples) -> list:
        messages = samples[self.input_col]
        logger.debug("STARTING GENERATION FOR LLMMessagesBlock")
        logger.debug(f"Generation arguments: {self.gen_kwargs}")
        results = []
        progress_bar = tqdm(
            range(len(samples)), desc=f"{self.block_name} Chat Completion Generation"
        )
        n = self.gen_kwargs.get("n", 1)
        for message in messages:
            logger.debug(f"CREATING CHAT COMPLETION FOR MESSAGE: {message}")
            responses = self.ctx.client.chat.completions.create(
                messages=message, **self.gen_kwargs
            )
            if n > 1:
                results.append([choice.message.content for choice in responses.choices])
            else:
                results.append(responses.choices[0].message.content)
            progress_bar.update(n)
        return results

    def generate(self, samples: Dataset) -> Dataset:
        outputs = self._generate(samples)
        logger.debug("Generated outputs: %s", outputs)
        samples = samples.add_column(self.output_col, outputs)
        return samples
