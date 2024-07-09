# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from importlib import resources
import operator
import os

# Local
from .filterblock import FilterByValueBlock
from .llmblock import LLMBlock
from .utilblocks import CombineColumnsBlock

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


class Flow(ABC):
    def __init__(
        self, client, model_family, model_id, num_instructions_to_generate
    ) -> None:
        self.client = client
        self.model_family = model_family
        self.model_id = model_id
        self.num_instructions_to_generate = num_instructions_to_generate
        self.sdg_base = resources.files(__package__)

    @abstractmethod
    def get_flow(self) -> list:
        pass


class _SimpleFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "",  # must be set by subclass
                    "config_path": "",  # must be set by subclass
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["output"],
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "n": self.num_instructions_to_generate,
                },
                "drop_duplicates": ["output"],
            }
        ]


class SimpleKnowledgeFlow(_SimpleFlow):
    def get_flow(self) -> list:
        flow = super().get_flow()
        flow[0]["block_config"]["config_path"] = os.path.join(
            self.sdg_base, "configs/knowledge/simple_generate_qa.yaml"
        )
        flow[0]["block_config"]["block_name"] = "gen_knowledge"
        return flow


class SimpleFreeformSkillFlow(_SimpleFlow):
    def get_flow(self) -> list:
        flow = super().get_flow()
        flow[0]["block_config"]["config_path"] = os.path.join(
            self.sdg_base, "configs/skills/simple_generate_qa_freeform.yaml"
        )
        flow[0]["block_config"]["block_name"] = "gen_skill_freeform"
        flow[0]["block_config"]["block_name"] = "gen_skill_freeform"
        return flow


class SimpleGroundedSkillFlow(_SimpleFlow):
    def get_flow(self) -> list:
        flow = super().get_flow()
        flow[0]["block_config"]["config_path"] = os.path.join(
            self.sdg_base, "configs/skills/simple_generate_qa_grounded.yaml"
        )
        flow[0]["block_config"]["block_name"] = "gen_skill_grounded"
        return flow


class MMLUBenchFlow(Flow):
    def get_flow(self) -> list:
        self.sdg_base = resources.files(__package__)
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_mmlu_knowledge",
                    "config_path": os.path.join(
                        self.sdg_base, "configs/knowledge/mcq_generation.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["mmlubench_question", "mmlubench_answer"],
                },
                "gen_kwargs": {
                    "temperature": 0,
                    "max_tokens": 2048,
                },
                "drop_duplicates": ["mmlubench_question"],
            },
        ]


class SynthKnowledgeFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_knowledge",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/knowledge/generate_questions_responses.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["question", "response"],
                    "parser_kwargs": {
                        "parser_name": "custom",
                        "parsing_pattern": r"\[(?:Question|QUESTION)\]\s*(.*?)\s*\[(?:Answer|ANSWER)\]\s*(.*?)\s*(?=\[(?:Question|QUESTION)\]|$)",
                        "parser_cleanup_tags": ["[END]"],
                    },
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_faithfulness_qa_pair",
                    "config_path": os.path.join(
                        self.sdg_base, "configs/knowledge/evaluate_faithfulness.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["explanation", "judgment"],
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_faithfulness",
                    "filter_column": "judgment",
                    "filter_value": "YES",
                    "operation": operator.eq,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["judgment", "explanation"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_relevancy_qa_pair",
                    "config_path": os.path.join(
                        self.sdg_base, "configs/knowledge/evaluate_relevancy.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["feedback", "score"],
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_relevancy",
                    "filter_column": "score",
                    "filter_value": 2.0,
                    "operation": operator.eq,
                    "convert_dtype": float,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["feedback", "score"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_verify_question",
                    "config_path": os.path.join(
                        self.sdg_base, "configs/knowledge/evaluate_question.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["explanation", "rating"],
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_verify_question",
                    "filter_column": "rating",
                    "filter_value": 1.0,
                    "operation": operator.eq,
                    "convert_dtype": float,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["explanation", "rating", "__index_level_0__"],
            },
        ]


class SynthSkillsFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_questions",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/freeform_questions.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["question"],
                    "batch_kwargs": {
                        "num_samples": self.num_instructions_to_generate,
                    },
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_questions",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/evaluate_freeform_questions.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["evaluation", "score"],
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_questions",
                    "filter_column": "score",
                    "filter_value": 1.0,
                    "operation": operator.eq,
                    "convert_dtype": float,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["evaluation", "score", "num_samples"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_responses",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/freeform_responses.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["response"],
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "evaluate_qa_pair",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/evaluate_freeform_pair.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["evaluation", "score"],
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_qa_pair",
                    "filter_column": "score",
                    "filter_value": 2.0,
                    "operation": operator.ge,
                    "convert_dtype": float,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["evaluation", "score"],
            },
        ]


class SynthGroundedSkillsFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_contexts",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/contexts.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["context"],
                },
                "gen_kwargs": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "n": self.num_instructions_to_generate,
                },
                "drop_duplicates": ["context"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_grounded_questions",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/grounded_questions.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["question"],
                    "batch_kwargs": {
                        "num_samples": 3,
                    },
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_grounded_questions",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/evaluate_grounded_questions.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["evaluation", "score"],
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_grounded_questions",
                    "filter_column": "score",
                    "filter_value": 1.0,
                    "operation": operator.eq,
                    "convert_dtype": float,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["evaluation", "score", "num_samples"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_grounded_responses",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/grounded_responses.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["response"],
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "evaluate_grounded_qa_pair",
                    "config_path": os.path.join(
                        self.sdg_base,
                        "configs/skills/evaluate_grounded_pair.yaml",
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_family),
                    "output_cols": ["evaluation", "score"],
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_grounded_qa_pair",
                    "filter_column": "score",
                    "filter_value": 2.0,
                    "operation": operator.ge,
                    "convert_dtype": float,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
            },
            {
                "block_type": CombineColumnsBlock,
                "block_config": {
                    "block_name": "combine_question_and_context",
                    "columns": ["context", "question"],
                    "output_col": "question",
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": True,
                    },
                },
            },
        ]
