# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
import json
import uuid
import random

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


def create_summary_dataset(generated_dataset: Dataset):
    detailed_summary_inst = [
    "Provide me with a comprehensive summary of the given document.",
    "Prepare a detailed breakdown of the contents of the document for me.",
    "Summarize the document thoroughly, covering all important points.",
    "Create a detailed executive summary of the provided document.",
    "Compose a comprehensive overview of the document's content.",
    "Deliver a detailed synopsis of the material presented in the document.",
    "Furnish me with a detailed analysis of the document's key points.",
    "Generate a thorough summary of the main ideas in the document.",
    "Offer a detailed digest of the information contained in the document.",
    "Supply me with a comprehensive rundown of the document's contents."
]

    extractive_summary_inst = [
    "Provide me with a summary of the document using extractive methods.",
    "Create an extractive summary for the given document.",
    "Generate an extractive summary from the document that was given to you.",
    "Summarize the document using extractive techniques.",
    "Create a summary of the provided document using extractive methods.",
    "Generate an extractive summary for the document provided.",
    "Using extractive techniques, summarize the given document.",
    "Create a summary of the document using extractive summarization.",
    "Generate an extractive summary of the document that was provided.",
    "Summarize the provided document using extractive summarization techniques."
    ]

    atomic_facts_inst = [
    "Identify and list all atomic facts from the document.",
    "Extract all key facts from the given document.",
    "List all the important facts from the provided document.",
    "Highlight all the atomic facts present in the document.",
    "Identify and enumerate all key facts from the given text.",
    "List out all the critical information from the document.",
    "Highlight all the essential facts from the provided text.",
    "Identify and summarize all the important details from the document.",
    "Extract all the atomic facts from the given document.",
    "List all the key takeaways from the provided text."
    ]
    summary_ds = generated_dataset.filter(lambda x: x["summary_type"] != "summary_base_document")
    unique_document_summary = summary_ds.to_pandas().drop_duplicates(subset=["document"])
    unique_document_summary = Dataset.from_pandas(unique_document_summary).remove_columns(["icl_query_1", "icl_response_1", "icl_query_2", "icl_response_2", "icl_query_3", "icl_response_3", "route", "__index_level_0__", "question", "response"])
    unique_document_summary = unique_document_summary.rename_columns({"raw_document": "context", "document": "response"})
    def __create_summary_ds(rec):
        if rec["summary_type"] == "summary_detailed":
            instruction = random.choice(detailed_summary_inst)
        elif rec["summary_type"] == "summary_extractive":
            instruction = random.choice(extractive_summary_inst)
        else:
            # Its probably atomic facts
            instruction = random.choice(atomic_facts_inst)
        messages = [{"role": "user", "content": f"{rec['context']}\n\n{instruction}"},
                    {"role": "assistant", "content": rec["response"]}]
        metadata = json.dumps({
            "summary_type": rec["summary_type"],
            "raw_document": rec["context"],
            "dataset": f"document_summary_{rec['summary_type']}"
        })
        return {"messages": messages, "metadata": metadata, "id":  str(uuid.uuid4())}
    unique_document_summary = unique_document_summary.map(__create_summary_ds, remove_columns=unique_document_summary.column_names)
    return unique_document_summary


def generate_knowledge_qa_dataset(generated_dataset: Dataset, keep_context_separate=False):
    def __create_qa_row(rec):
        context = rec["document"]
        instruction = rec["question"]
        response = rec["response"]
        metadata = json.dumps({
            "raw_document": rec["raw_document"],
            "summary_type": rec["summary_type"],
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": f"document_knowledge_qa"
        })
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
    summary_dataset = create_summary_dataset(generated_dataset)
    knowledge_ds = generate_knowledge_qa_dataset(generated_dataset, keep_context_separate=True)
    knowledge_ds = build_raft_dataset(knowledge_ds, p=0.4)
    phase10 = concatenate_datasets([knowledge_ds, summary_dataset])
    return phase10


def create_phase07_ds(generated_dataset: Dataset):
    # Phase 0.7
    summary_dataset = create_summary_dataset(generated_dataset)
    knowledge_ds = generate_knowledge_qa_dataset(generated_dataset, keep_context_separate=False)
    summary_dataset = summary_dataset.map(_conv_pretrain)
    knowledge_ds = knowledge_ds.map(_conv_pretrain)
    phase07 = concatenate_datasets([knowledge_ds, summary_dataset])
    return phase07
