# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
import json
import uuid
import random
import yaml
import os

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg import utils
from instructlab.sdg.logger_config import setup_logger
from datasets import Dataset, concatenate_datasets

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

    return sample


def create_auxiliary_dataset(generated_dataset: Dataset):

    if "dataset_type" not in generated_dataset.column_names:
        return None
    if os.path.isfile("src/instructlab/sdg/config/kowledge/auxilary_instructions.yaml"):
        with open("src/instructlab/sdg/config/kowledge/auxilary_instructions.yaml", "r") as fp:
            auxiliary_inst = yaml.safe_load(fp)
    else:
        return None
    auxiliary_ds = generated_dataset.filter(lambda x: x["dataset_type"] != "base_document")
    unique_document_auxiliary = auxiliary_ds.to_pandas().drop_duplicates(subset=["document"])
    unique_document_auxiliary = Dataset.from_pandas(unique_document_auxiliary).remove_columns(
        [col for col in unique_document_auxiliary.column_names if col not in ['raw_document', 'document_outline', 'domain', 'dataset_type', 'document']])
    print(unique_document_auxiliary)
    unique_document_auxiliary = unique_document_auxiliary.rename_columns({"raw_document": "context", "document": "response"})
    def __create_auxiliary_ds(rec):
        instruction = random.choice(auxiliary_inst[rec['dataset_type']])
        messages = [{"role": "user", "content": f"{rec['context']}\n\n{instruction}"},
                    {"role": "assistant", "content": rec["response"]}]
        metadata = json.dumps({
            "dataset_type": rec["dataset_type"],
            "raw_document": rec["context"],
            "dataset": f"document_{rec['dataset_type']}",
            "domain": rec["domain"]
        })
        return {"messages": messages, "metadata": metadata, "id":  str(uuid.uuid4())}
    unique_document_auxiliary = unique_document_auxiliary.map(__create_auxiliary_ds, remove_columns=unique_document_auxiliary.column_names)
    return unique_document_auxiliary


def generate_knowledge_qa_dataset(generated_dataset: Dataset, keep_context_separate=False):
    def __create_qa_row(rec):
        context = rec["document"]
        instruction = rec["question"]
        response = rec["response"]
        metadata = {
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": f"document_knowledge_qa"
        }
        if "raw_document" in rec and "dataset_type" in rec:
            metadata.update({"raw_document": rec["raw_document"],
            "dataset_type": rec["dataset_type"],})
        metadata = json.dumps(metadata)
        if keep_context_separate:
            messages = [{"role": "user", "content": f"{instruction}"},
                            {"role": "assistant", "content": response}]
            return {"messages": messages, "metadata": metadata, "id":  str(uuid.uuid4()), "context": context}
        else:
            messages = [{"role": "user", "content": f"{context}\n\n{instruction}"},
                        {"role": "assistant", "content": response}]
       
            return {"messages": messages, "metadata": metadata, "id":  str(uuid.uuid4())}
    knowledge_ds = generated_dataset.map(__create_qa_row, remove_columns=generated_dataset.column_names)
    return knowledge_ds 


def build_raft_dataset(ds: Dataset, p, num_doc_in_context=4):
    all_context = ds["context"]
    all_context = [" ".join(e.split(" ")[:random.randint(100, 500)]) for e in all_context]
    ds = ds.add_column("row_idx", range(ds.num_rows))
    def __pick_documents(rec, p):
        while True:
            selected_docs = random.choices(range(ds.num_rows), k=num_doc_in_context)
            if rec["row_idx"] not in selected_docs:
                break
        if random.uniform(0, 1) < p:
            docs = [all_context[idx] for idx in selected_docs[:num_doc_in_context-1]] + [rec["context"]]
            # rec['indicator'] ='golden'
        else:
            docs = [all_context[idx] for idx in selected_docs] 
            # rec['indicator'] = 'distractor'
        random.shuffle(docs)
        docs = "\n".join(([f"Document:\n{e}\n\n" for idx, e in enumerate(docs)]))
        user_idx, user_msg = [(idx, rec_msg) for idx, rec_msg in enumerate(rec["messages"]) if rec_msg["role"] == "user"][0]
        user_inst = user_msg["content"]
        rec["messages"][user_idx]["content"] = f"{docs}\n\n{user_inst}"
        rec["messages"] = rec["messages"]
        metadata = json.loads(rec['metadata'])
        metadata['dataset'] += f"_raft_p{p}"
        rec['metadata'] = json.dumps(metadata)
        return rec
    ds = ds.map(__pick_documents, fn_kwargs={"p": p}, remove_columns=["context"])
    return ds

def _conv_pretrain(rec):
    rec["messages"] = [
        {
        "role": "pretraining",
        "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}"
    }]
    return rec


def create_phase10_ds(generated_dataset: Dataset):
    # Phase 1.0
    knowledge_ds = generate_knowledge_qa_dataset(generated_dataset, keep_context_separate=True)
    knowledge_ds = build_raft_dataset(knowledge_ds, p=0.4)
    
    auxiliary_dataset = create_auxiliary_dataset(generated_dataset)
    if auxiliary_dataset is not None:
        phase10 = concatenate_datasets([knowledge_ds, auxiliary_dataset])
    else:
        phase10 = knowledge_ds
    return phase10


def create_phase07_ds(generated_dataset: Dataset):
    # Phase 0.7
    knowledge_ds = generate_knowledge_qa_dataset(generated_dataset, keep_context_separate=False)
    knowledge_ds = knowledge_ds.map(_conv_pretrain)
    
    auxiliary_dataset = create_auxiliary_dataset(generated_dataset)
    if auxiliary_dataset is not None:
        auxiliary_dataset = auxiliary_dataset.map(_conv_pretrain)
        phase07 = concatenate_datasets([knowledge_ds, auxiliary_dataset])
    else:
        phase07 = knowledge_ds
    return phase07
