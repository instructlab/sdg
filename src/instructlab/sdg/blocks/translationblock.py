# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from IndicTransToolkit import IndicProcessor
# from .block import Block, BlockConfigParserError

# # Local
# # Import prompts to register default chat templates
# from .. import prompts as default_prompts  # pylint: disable=unused-import
# from ..registry import BlockRegistry, PromptRegistry
# from ..utils import models
# from .block import Block, BlockConfigParserError
# # recommended to run this on a gpu with flash_attn installed
# # don't set attn_implemetation if you don't have flash_attn
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # make a translation block 
# # This is part of the public API.

# @BlockRegistry.register("TranslationBlock")
# class TranslationBlock(Block):
#     def __init__(self, config):
#         super().__init__(config)

#         model_name = config.get("model_name", "ai4bharat/indictrans2-en-indic-1B")
#         self.src_lang = config.get("source_lang", "eng_Latn")
#         self.tgt_lang = config.get("target_lang", "hin_Deva")
#         self.input_col = config.get("input_col", "output")  # Default input column
#         self.output_col = config.get("output_col", "translated_output")  # Default output column

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Load tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name, 
#             trust_remote_code=True, 
#             torch_dtype=torch.float16,
#             attn_implementation="flash_attention_2"
#         ).to(self.device)
        
#         # Load IndicProcessor
#         self.ip = IndicProcessor(inference=True)

#     def translate(self, input_sentences):
#         # Preprocess batch
#         batch = self.ip.preprocess_batch(input_sentences, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        
#         # Tokenize the sentences
#         inputs = self.tokenizer(
#             batch,
#             truncation=True,
#             padding="longest",
#             return_tensors="pt",
#             return_attention_mask=True,
#         ).to(self.device)
        
#         # Generate translations
#         with torch.no_grad():
#             generated_tokens = self.model.generate(
#                 **inputs,
#                 use_cache=True,
#                 min_length=0,
#                 max_length=256,
#                 num_beams=5,
#                 num_return_sequences=1,
#             )
        
#         # Decode tokens to text
#         with self.tokenizer.as_target_tokenizer():
#             generated_tokens = self.tokenizer.batch_decode(
#                 generated_tokens.detach().cpu().tolist(),
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True,
#             )
        
#         # Postprocess translations
#         translations = self.ip.postprocess_batch(generated_tokens, lang=self.tgt_lang)
#         return translations

#     def __call__(self, data):
#         if self.input_col not in data:
#             raise ValueError(f"Column '{self.input_col}' not found in input data")

#         # Translate the text
#         translated_texts = self.translate(data[self.input_col])

#         # Add translated text to the pipeline output
#         data[self.output_col] = translated_texts
#         return data




import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .block import Block, BlockConfigParserError

from datasets import Dataset
from tqdm import tqdm
import httpx
import openai

# Recommended to run this on a GPU with flash_attn installed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        model_id=None,
        model_family=None,
        gen_kwargs={},
        batch_kwargs={},
    ) -> None:
        super().__init__(ctx, pipe, block_name)
        self.block_config = self._load_config(config_path)
        self.model_id = model_id or "facebook/nllb-200-3.3B"
        self.model_family = models.get_model_family(
            _resolve_model_family(model_family, self.ctx.model_family),
            self.model_id,
        )
        self.output_cols = output_cols
        self.batch_params = batch_kwargs
        self.gen_kwargs = self._gen_kwargs(gen_kwargs)
        self.server_supports_batched = server_supports_batched(
            self.ctx.client, self.model_id
        )
    
    def _gen_kwargs(self, gen_kwargs):
        defaults = {"temperature": 0, "max_tokens": DEFAULT_MAX_NUM_TOKENS}
        return {**defaults, **gen_kwargs}
    
    def _translate(self, samples) -> list:
        prompts = [sample["text"] for sample in samples]
        logger.debug(f"STARTING TRANSLATION USING PROMPTS: {prompts}")
        
        if self.server_supports_batched:
            response = self.ctx.client.completions.create(
                prompt=prompts, **self.gen_kwargs
            )
            return [choice.text.strip() for choice in response.choices]
        
        results = []
        for prompt in tqdm(prompts, desc=f"{self.block_name} Translation"):
            response = self.ctx.client.completions.create(
                prompt=prompt, **self.gen_kwargs
            )
            results.append(response.choices[0].text.strip())
        return results
    
    def generate(self, samples: Dataset) -> Dataset:
        num_samples = self.batch_params.get("num_samples", None)
        logger.debug(f"Generating translations for {len(samples)} samples")
        
        if (num_samples is not None) and ("num_samples" not in samples.column_names):
            samples = samples.add_column("num_samples", [num_samples] * len(samples))
        
        translations = self._translate(samples)
        
        new_data = []
        for sample, translation in zip(samples, translations):
            new_data.append({**sample, self.output_cols[0]: translation})
        
        return Dataset.from_list(new_data)