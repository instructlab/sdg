# SPDX-License-Identifier: Apache-2.0

# Standard
import random
import string

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg import LLMBlock


def _random_string(size):
    return "".join(random.choices(string.ascii_lowercase, k=size))


def _add_mocked_cols(sample, block_name):
    match block_name:
        case "gen_questions" | "gen_grounded_questions":
            sample["question"] = f"Is this a question {_random_string(8)}?"
        case "eval_questions" | "eval_grounded_questions":
            sample["evaluation"] = "This is an evaluation."
            sample["score"] = "1"
        case "gen_responses" | "gen_grounded_responses":
            sample["response"] = "This is a response."
        case "evaluate_qa_pair" | "evaluate_grounded_qa_pair":
            sample["evaluation"] = "This is an evaluation."
            sample["score"] = "2"
        case "gen_contexts":
            sample["context"] = f"This is a context {_random_string(8)}."
        case "gen_spellcheck":
            sample["spellcheck"] = sample["document"]
        case "gen_knowledge":
            sample["question"] = f"Is this a question {_random_string(8)}?"
            sample["response"] = "This is a response."
        case "eval_faithfulness_qa_pair":
            sample["explanation"] = "This is an explanation."
            sample["judgment"] = "YES"
        case "eval_relevancy_qa_pair":
            sample["feedback"] = "This is some feedback."
            sample["score"] = "2"
        case "eval_verify_question":
            sample["explanation"] = "This is an explanation."
            sample["rating"] = "1"
        case _:
            raise Exception(
                f"Received an un-mocked LLMBlock: {block_name}. Add code in {__file__} to handle this block."
            )
    return sample


class MockLLMBlock(LLMBlock):
    def generate(self, samples: Dataset):
        return samples.map(_add_mocked_cols, fn_kwargs={"block_name": self.block_name})
