# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
import operator

# Local
from .filterblock import FilterByValueBlock
from .llmblock import LLMBlock
from .utilblocks import CombineColumnsBlock


class Flow(ABC):
    def __init__(self, ctx) -> None:
        self.ctx = ctx

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
                    "output_cols": ["output"],
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "n": self.ctx.num_instructions_to_generate,
                },
                "drop_duplicates": ["output"],
            }
        ]


class SimpleKnowledgeFlow(_SimpleFlow):
    def get_flow(self) -> list:
        flow = super().get_flow()
        flow[0]["block_config"]["config_path"] = (
            "configs/knowledge/simple_generate_qa.yaml"
        )
        flow[0]["block_config"]["block_name"] = "gen_knowledge"
        return flow


class SimpleFreeformSkillFlow(_SimpleFlow):
    def get_flow(self) -> list:
        flow = super().get_flow()
        flow[0]["block_config"]["config_path"] = (
            "configs/skills/simple_generate_qa_freeform.yaml"
        )
        flow[0]["block_config"]["block_name"] = "gen_skill_freeform"
        return flow


class SimpleGroundedSkillFlow(_SimpleFlow):
    def get_flow(self) -> list:
        flow = super().get_flow()
        flow[0]["block_config"]["config_path"] = (
            "configs/skills/simple_generate_qa_grounded.yaml"
        )
        flow[0]["block_config"]["block_name"] = "gen_skill_grounded"
        return flow


class MMLUBenchFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_mmlu_knowledge",
                    "config_path": "configs/knowledge/mcq_generation.yaml",
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
                    "config_path": "configs/knowledge/generate_questions_responses.yaml",
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
                    "config_path": "configs/knowledge/evaluate_faithfulness.yaml",
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
                },
                "drop_columns": ["judgment", "explanation"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_relevancy_qa_pair",
                    "config_path": "configs/knowledge/evaluate_relevancy.yaml",
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
                },
                "drop_columns": ["feedback", "score"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_verify_question",
                    "config_path": "configs/knowledge/evaluate_question.yaml",
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
                    "config_path": "configs/skills/freeform_questions.yaml",
                    "output_cols": ["question"],
                    "add_num_samples": True,
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_questions",
                    "config_path": "configs/skills/evaluate_freeform_questions.yaml",
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
                },
                "drop_columns": ["evaluation", "score", "num_samples"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_responses",
                    "config_path": "configs/skills/freeform_responses.yaml",
                    "output_cols": ["response"],
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "evaluate_qa_pair",
                    "config_path": "configs/skills/evaluate_freeform_pair.yaml",
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
                    "config_path": "configs/skills/contexts.yaml",
                    "output_cols": ["context"],
                },
                "gen_kwargs": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "n": self.ctx.num_instructions_to_generate,
                },
                "drop_duplicates": ["context"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_grounded_questions",
                    "config_path": "configs/skills/grounded_questions.yaml",
                    "output_cols": ["question"],
                    "add_num_samples": True,
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_grounded_questions",
                    "config_path": "configs/skills/evaluate_grounded_questions.yaml",
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
                },
                "drop_columns": ["evaluation", "score", "num_samples"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_grounded_responses",
                    "config_path": "configs/skills/grounded_responses.yaml",
                    "output_cols": ["response"],
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "evaluate_grounded_qa_pair",
                    "config_path": "configs/skills/evaluate_grounded_pair.yaml",
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
                },
            },
            {
                "block_type": CombineColumnsBlock,
                "block_config": {
                    "block_name": "combine_question_and_context",
                    "columns": ["context", "question"],
                    "output_col": "question",
                },
            },
        ]
