# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
import operator

# Local
from .filterblock import FilterByValueBlock
from .llmblock import LLMBlock
from .utilblocks import SamplePopulatorBlock, SelectorBlock


class Flow(ABC):
    def __init__(self, client, model_id) -> None:
        self.client = client
        self.model_id = model_id

    @abstractmethod
    def get_flow(self) -> list:
        pass


class MMLUBenchFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_mmlu_knowledge",
                    "config_path": "src/instructlab/sdg/configs/knowledge/mcq_generation.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": "<s> [INST] {prompt} [/INST]",
                    "output_cols": ["mmlubench_question", "mmlubench_answer"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": True,
                    },
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
                    "config_path": "src/instructlab/sdg/configs/knowledge/generate_questions_responses.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": "<s> [INST] {prompt} [/INST]",
                    "output_cols": ["question", "response"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": True,
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
                    "config_path": "src/instructlab/sdg/configs/knowledge/evaluate_faithfulness.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": "<s> [INST] {prompt} [/INST]",
                    "output_cols": ["explanation", "judgment"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": True,
                    },
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
                    "config_path": "src/instructlab/sdg/configs/knowledge/evaluate_relevancy.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": "<s> [INST] {prompt} [/INST]",
                    "output_cols": ["feedback", "score"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": True,
                    },
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
                    "filter_value": "2",
                    "operation": operator.eq,
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
                    "config_path": "src/instructlab/sdg/configs/knowledge/evaluate_question.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": "<s> [INST] {prompt} [/INST]",
                    "output_cols": ["explanation", "rating"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": True,
                    },
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
                    "filter_value": "1",
                    "operation": operator.eq,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["explanation", "rating"],
            },
        ]
