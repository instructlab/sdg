import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Any, Dict, List
import logging
import re

from datasets import Dataset
from tqdm import tqdm
import httpx
import openai

from .. import prompts as default_prompts  # pylint: disable=unused-import
from ..registry import BlockRegistry, PromptRegistry
from ..utils import models
from .block import Block, BlockConfigParserError

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

DEFAULT_MAX_NUM_TOKENS = 8192


def template_from_struct_and_config(struct, config):
    # Replace None values with empty strings
    filtered_config = {k: (v if v is not None else "") for k, v in config.items()}
    return PromptRegistry.template_from_string(struct.format(**filtered_config))


# This is part of the public API.


def _resolve_model_id(model_id, ctx_model_id, block):
    # If a model id was passed in the PipelineContext, use that
    if ctx_model_id:
        return ctx_model_id

    # If we have no model id at all, raise an error
    if not model_id:
        raise BlockConfigParserError(
            f"{type(block).__name__} {block.block_name} requires a model_id but none was specified in the block config nor passed via the PipelineContext"
        )

    # Otherwise fallback to the model_id specified in the block config
    return model_id


def _resolve_model_family(model_family, ctx_model_family):
    # If a model family was passed in the PipelineContext, use that
    if ctx_model_family:
        return ctx_model_family
    return model_family


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


@BlockRegistry.register("TranslationBlock")
# pylint: disable=dangerous-default-value
class TranslationBlock(Block):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        config_path,
        output_cols,
        model_id=None,
        model_family=None,
        model_prompt=None,
        trans_model_id=None,
        trans_model=None,
        source_lang="eng_Latn",
        target_lang="hin_Deva",
        gen_kwargs={},
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = """{question}\n{response}"""
        self.prompt_template = template_from_struct_and_config(
            self.prompt_struct, self.block_config
        )
        self.model_id = _resolve_model_id(model_id, self.ctx.model_id, self)
        self.model_family = models.get_model_family(
            _resolve_model_family(model_family, self.ctx.model_family),
            self.model_id,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model_prompt = model_prompt
        self.trans_model_id = f"facebook/nllb-200-3.3B"
        # self.trans_model_id = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.output_cols = output_cols
        self.batch_params = batch_kwargs
        max_num_token_override = ctx.max_num_tokens
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)

        # Load tokenizer and model for translation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.trans_model_id, src_lang=self.source_lang
            )
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.trans_model_id
            ).to(DEVICE)
        except Exception as e:
            raise ValueError(f"Error loading TRANSLATOIN model {self.model_id}: {e}")
        # try:
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.trans_model_id)
        #     self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(self.trans_model_id).to(DEVICE)
        # except Exception as e:
        #     raise ValueError(f"Error loading TRANSLATOIN model {self.model_id}: {e}")
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
            model=self.model_id,
            temperature=0,
            max_tokens=DEFAULT_MAX_NUM_TOKENS,
        )
        # Whether the LLM server supports a list of input prompts
        # and supports the n parameter to generate n outputs per input
        self.server_supports_batched = server_supports_batched(
            self.ctx.client, self.model_id
        )

    def _translate(self, text: str) -> str:
        """Translates a single string and returns the translated text."""
        logging.debug(f"Translating text using model {self.model_id}")
        encoded_input = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        # encoded_input = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            translated_tokens = self.trans_model.generate(
                **encoded_input,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                    self.target_lang
                ),
                max_length=1024,
            )
            # translated_tokens = self.trans_model.generate(**encoded_input)

        translation = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        return translation

    # def _translate(self, samples: List[Dict]) -> List[Dict]:
    #     """Translates a batch of input samples and returns structured output."""
    #     # prompts = [sample["output"] for sample in samples]'
    #     logging.debug(f"STARTING TRANSLATION USING MODEL {self.model_id}")

    #     encoded_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    #     with torch.no_grad():
    #         translated_tokens = self.trans_model.generate(**encoded_inputs)

    #     translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

    #     # Attach translations back to inputs (if needed)
    #     return [
    #         {**sample, "translated_output": translation}
    #         for sample, translation in zip(samples, translations)
    #     ]

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

    def _gen_kwargs(self, max_num_token_override, gen_kwargs, **defaults):
        gen_kwargs = {**defaults, **gen_kwargs}
        if (
            "n" in gen_kwargs
            and isinstance(gen_kwargs["n"], str)
            and gen_kwargs["n"] == "scaled"
        ):
            if not self.ctx.num_instructions_to_generate:
                raise BlockConfigParserError(
                    f"""LLMBlock {self.block_name} has a gen_kwargs["n"] value of "scaled" but num_instructions_to_generate was not set in the PipelineContext"""
                )
            gen_kwargs["n"] = self.ctx.num_instructions_to_generate
        if "temperature" in gen_kwargs:
            gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        if max_num_token_override != DEFAULT_MAX_NUM_TOKENS:
            gen_kwargs["max_tokens"] = max_num_token_override
        elif "max_tokens" in gen_kwargs:
            gen_kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        return gen_kwargs

    def _generate(self, samples) -> list:
        prompts = [sample for sample in samples]
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
                text = response.choices[0].text.strip()
                text = response.choices[0].text.strip()
                print(f"Generated text: {text}")
                translated_text = text
                # firest strip the text ,if it begins with "Question:" or "Answer:"
                if text.startswith("Question:") or text.startswith("Answer:"):
                    q_part, a_part = None, None
                    if "Question:" in text:
                        q_start = text.find("Question:")
                        a_start = text.find("Answer:")
                        if a_start != -1 and a_start > q_start:
                            q_part = text[q_start + len("Question:") : a_start].strip()
                            a_part = text[a_start + len("Answer:") :].strip()
                        else:
                            q_part = text[q_start + len("Question:") :].strip()
                    elif "Answer:" in text:
                        a_part = text[len("Answer:") :].strip()

                    translated_q = self._translate(q_part) if q_part else None
                    translated_a = self._translate(a_part) if a_part else None

                    translated_text = ""
                    if translated_q:
                        translated_text += "Question: " + translated_q + "\n"
                    if translated_a:
                        translated_text += "Answer: " + translated_a
                else:
                    translated_text = self._translate(text)

                print(f"Translated text: {translated_text}")

                results.append(translated_text)
                # logger.debug(f"RESULT: {translated_text}")
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

        import pdb

        pdb.set_trace()

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
        logger.debug("Generated Translated outputs: %s", outputs)

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
            # parsed_outputs = self._parse(output)
            max_length = max(len(value) for value in parsed_outputs.values())
            for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})

        return Dataset.from_list(new_data)
