# Standard
from typing import Any, Dict, List
import logging
import re

# Third Party
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import httpx
import openai
import torch

# Local
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
        trans_model_id=None,
        source_lang="eng_Latn",
        target_lang="hin_Deva",
        prompt_struct=None,
        gen_kwargs={},
        parser_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.block_config = self._load_config(config_path)
        self.prompt_struct = prompt_struct
        self.prompt_template = template_from_struct_and_config(
            self.prompt_struct, self.block_config
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.trans_model_id = trans_model_id
        # self.trans_model_id = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.output_cols = output_cols
        self.batch_params = batch_kwargs
        max_num_token_override = ctx.max_num_tokens
        self.parser_name = parser_kwargs.get("parser_name", None)
        self.parsing_pattern = parser_kwargs.get("parsing_pattern", None)
        self.parser_cleanup_tags = parser_kwargs.get("parser_cleanup_tags", None)

        # Load tokenizer and model for translation
        try:
            logger.warn(f"Loading {self.trans_model_id} model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.trans_model_id, src_lang=self.source_lang
            )
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.trans_model_id
            ).to(DEVICE)
        except Exception as e:
            raise ValueError(
                f"Error loading TRANSLATOIN model {self.trans_model_id}: {e}"
            )

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
            model=self.trans_model_id,
            temperature=0,
            max_tokens=DEFAULT_MAX_NUM_TOKENS,
        )
        # Whether the LLM server supports a list of input prompts
        # and supports the n parameter to generate n outputs per input
        self.server_supports_batched = False

    def _translate(self, text: str) -> str:
        """Translates a single string and returns the translated text."""
        logging.debug(f"Translating text using model {self.trans_model_id}")
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
        logger.debug(f"STARTING GENERATION FOR TRANSLATION Block:")
        logger.debug(f"Generation arguments: {self.gen_kwargs}")

        results = []
        progress_bar = tqdm(
            range(len(samples)), desc=f"{self.block_name} Prompt Generation"
        )
        for sample in samples:

            columns_to_translate = [sample[key] for key in self.block_config.keys()]

            for _ in range(self.gen_kwargs.get("n", 1)):
                translated_texts = []

                for text in columns_to_translate:
                    translated_texts.append(self._translate(text))

                results.append(translated_texts)
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
            is_valid = True
            for key in self.block_config.keys():
                if key not in sample:
                    is_valid = False

            if is_valid:
                valid_samples.append(sample)

        samples = valid_samples

        if len(samples) == 0:
            return Dataset.from_list([])

        # generate the output

        outputs = self._generate(samples)

        num_parallel_samples = self.gen_kwargs.get("n", 1)
        extended_samples = []

        # Duplicate each input sample n times, where n is the number
        # of output sequences generated per input, so that we can
        # pair up the inputs and outputs.
        for item in samples:
            extended_samples.extend([item] * num_parallel_samples)

        new_data = []
        for sample, output in zip(extended_samples, outputs):

            translated_data = {}

            index = 0
            for key in self.output_cols:
                translated_data[key] = output[index]
                index = index + 1

            new_data.append({**sample, **translated_data})

        return Dataset.from_list(new_data)
