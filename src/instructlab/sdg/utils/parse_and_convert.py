# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
import json
import uuid

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg import utils
from instructlab.sdg.logger_config import setup_logger


logger = setup_logger(__name__)

class TaxonomyType(Enum):
    KNOWLEDGE = "knowledge"
    SKILL = "skill"


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question(synth_example: dict):
    if "question" in synth_example:
        return synth_example["question"]

    if not synth_example.get("output"):
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[0].strip() + "?" if len(parts) == 2 else ""


# This is also a hack. See the comment above _get_question.
def _get_response(synth_example: dict):
    if "response" in synth_example:
        return synth_example["response"]

    if "output" not in synth_example:
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()



def _convert_to_hack_fmt(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain 'system', 'user', and 'assistant' columns.

    Note: We should remove this function in the future when we resolve this issue and
    standardize the format to messages.
    """
    # Create user query message
    user_query = _unescape(_get_question(sample))
    response = _unescape(_get_response(sample))
    if "context" in sample:
        user_query = f"{sample['context']}\n\n{user_query}"

    sample["id"] = str(uuid.uuid4())
    sample["system"] = sys_prompt
    sample["user"] = user_query
    sample["assistant"] = response

    metadata = {
        key: value
        for key, value in sample.items()
        if key not in ["system", "user", "assistant"]
    }
    sample["metadata"] = json.dumps(metadata)

    return sample


def _convert_to_messages(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain 'messages' 
    and 'metadata' columns required for training.
    """
    # Create user query message
    user_query = _unescape(_get_question(sample))
    response = _unescape(_get_response(sample))
    
    sample["id"] = str(uuid.uuid4())
    sample["messages"] = [
        {"content": sys_prompt, "role": "system"},
        {"content": user_query, "role": "user"},
        {"content": response, "role": "assistant"},
    ]
    metadata = {
        key: value
        for key, value in sample.items()
        if key not in ["inputs", "targets"]
    }
    sample["metadata"] = json.dumps(metadata)

    return sample
