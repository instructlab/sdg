# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import os
import time

# Third Party
# instructlab - All of these need to go away (other than sdg) - issue #6
from datasets import concatenate_datasets, Dataset
import httpx
import openai

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg import SDG, utils
from instructlab.sdg.default_flows import (
    DEFAULT_FLOW_FILE_MAP,
    MODEL_FAMILY_MERLINITE,
    MODEL_FAMILY_MIXTRAL,
    Flow,
)
from instructlab.sdg.pipeline import Pipeline
from instructlab.sdg.utils import models, datamixing
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)

_SYS_PROMPT = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."


class TaxonomyType(Enum):
    KNOWLEDGE = "knowledge"
    SKILL = "skill"


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8")


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question(logger, synth_example):
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
def _get_response(logger, synth_example):
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


def _convert_to_messages(sample):
    """
    Convert a sample dictionary to contain 'messages' and 'metadata' columns required for training.
    """
    # Create user query message
    user_query = sample["inputs"]
    # TODO: in the future we can remove the combinecolumnsblock and combine them here for simplicity
    # if "context" in sample:
    #     user_query = f"{sample['context']}\n\n{sample['inputs']}"

    sample["messages"] = [
        {"content": user_query, "role": "user"},
        {"content": sample["targets"], "role": "assistant"},
    ]
    metadata = {
        key: value
        for key, value in sample.items()
        if key not in ["messages", "inputs", "targets"]
    }
    sample["metadata"] = json.dumps(metadata)

    # keeping required keys for messages training format
    sample = {"messages": sample["messages"], "metadata": sample["metadata"]}

    return sample


def _gen_train_data(
    logger, machine_instruction_data, output_file_train, output_file_messages
):
    train_data = []
    messages_data = []

    for output_dataset in machine_instruction_data:
        for synth_example in output_dataset:
            logger.debug(synth_example)
            user = _get_question(logger, synth_example)
            if len(synth_example.get("context", "")) > 0:
                user += "\n" + synth_example["context"]
            assistant = _unescape(_get_response(logger, synth_example))
            train_entry = {
                "system": _SYS_PROMPT,
                "user": _unescape(user),
                "assistant": assistant,
            }
            train_data.append(train_entry)
            sample = {
                "inputs": _unescape(user),
                "targets": assistant,
                "system": _SYS_PROMPT,
            }
            messages_data.append(_convert_to_messages(sample))

    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for entry in train_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")

    with open(output_file_messages, "w", encoding="utf-8") as outfile:
        for entry in messages_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


def _gen_test_data(
    leaf_nodes,
    output_file_test,
):
    test_data = []
    for _, leaf_node in leaf_nodes.items():
        for seed_example in leaf_node:
            user = seed_example["instruction"]  # question

            if len(seed_example["input"]) > 0:
                user += "\n" + seed_example["input"]  # context

            test_data.append(
                {
                    "system": _SYS_PROMPT,
                    "user": _unescape(user),
                    "assistant": _unescape(seed_example["output"]),  # answer
                }
            )

    with open(output_file_test, "w", encoding="utf-8") as outfile:
        for entry in test_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


def _sdg_init(pipeline, client, num_instructions_to_generate):
    knowledge_flows = []
    freeform_skill_flows = []
    grounded_skill_flows = []

    if pipeline == "full":
        knowledge_flows.append(
            Flow(client).get_flow_from_file(DEFAULT_FLOW_FILE_MAP["MMLUBenchFlow"])
        )
        knowledge_flows.append(
            Flow(client).get_flow_from_file(DEFAULT_FLOW_FILE_MAP["SynthKnowledgeFlow"])
        )
        freeform_skill_flows.append(
            Flow(client, num_instructions_to_generate).get_flow_from_file(
                DEFAULT_FLOW_FILE_MAP["SynthSkillsFlow"]
            )
        )
        grounded_skill_flows.append(
            Flow(client, num_instructions_to_generate).get_flow_from_file(
                DEFAULT_FLOW_FILE_MAP["SynthGroundedSkillsFlow"]
            )
        )
    elif pipeline == "simple":
        knowledge_flows.append(
            Flow(client).get_flow_from_file(
                DEFAULT_FLOW_FILE_MAP["SimpleKnowledgeFlow"]
            )
        )
        freeform_skill_flows.append(
            Flow(client, num_instructions_to_generate).get_flow_from_file(
                DEFAULT_FLOW_FILE_MAP["SimpleFreeformSkillFlow"]
            )
        )
        grounded_skill_flows.append(
            Flow(client, num_instructions_to_generate).get_flow_from_file(
                DEFAULT_FLOW_FILE_MAP["SimpleGroundedSkillFlow"]
            )
        )
    elif os.path.isfile(pipeline):
        flow = Flow(client, num_instructions_to_generate).get_flow_from_file(pipeline)
        knowledge_flows = [flow]
        freeform_skill_flows = [flow]
        grounded_skill_flows = [flow]
    else:
        raise utils.GenerateException(f"Error: pipeline ({pipeline}) is not supported.")

    sdg_knowledge = SDG([Pipeline(flow) for flow in knowledge_flows])
    sdg_freeform_skill = SDG([Pipeline(flow) for flow in freeform_skill_flows])
    sdg_grounded_skill = SDG([Pipeline(flow) for flow in grounded_skill_flows])
    return sdg_knowledge, sdg_freeform_skill, sdg_grounded_skill


# TODO - parameter removal needs to be done in sync with a CLI change.
# pylint: disable=unused-argument
def generate_data(
    logger,
    api_base,
    api_key: Optional[str] = None,
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
    # TODO - not used -- when batching is enabled, this is relevant.
    # Right now the code hard codes 8 cpus for batching
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
    output_dir: Optional[str] = None,
    # TODO - not used and should be removed from the CLI
    prompt_file_path: Optional[str] = None,
    # TODO - probably should be removed
    rouge_threshold: Optional[float] = None,
    console_output=True,
    yaml_rules: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    tls_insecure=False,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
    # TODO need to update the CLI to specify which pipeline to use (simple or full at the moment)
    pipeline: Optional[str] = "simple",
    configs_dir: Path = Path(os.path.abspath(__file__)) / "configs",
):
    generate_start = time.time()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (taxonomy and os.path.exists(taxonomy)):
        raise utils.GenerateException(f"Error: taxonomy ({taxonomy}) does not exist.")

    leaf_nodes = read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules)
    if not leaf_nodes:
        raise utils.GenerateException("Error: No new leaf nodes found in the taxonomy.")

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file_messages = f"messages_{name}_{date_suffix}.jsonl"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    # train data in messages format that will be mixed and split up into train test eventually
    output_file_train = f"train_{name}_{date_suffix}.jsonl"

    _gen_test_data(
        leaf_nodes,
        os.path.join(output_dir, output_file_test),
    )

    logger.debug(f"Generating to: {os.path.join(output_dir, output_file_test)}")

    orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
    cert = tuple(item for item in orig_cert if item)
    verify = not tls_insecure
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(cert=cert, verify=verify),
    )

    if models.get_model_family(model_family, model_name) == "mixtral":
        model_family = MODEL_FAMILY_MIXTRAL
    else:
        model_family = MODEL_FAMILY_MERLINITE

    # TODO -- llama-cpp doesn't support batching, we need to get a hint from the CLI
    # about whether we can turn this on (whether vllm is used or not)

    sdg_knowledge, sdg_freeform_skill, sdg_grounded_skill = _sdg_init(
        pipeline,
        client,
        num_instructions_to_generate,
    )

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    generated_knowledge = []
    generated_skills = []
    for leaf_node in leaf_nodes.values():
        samples = leaf_node_to_samples(leaf_node, server_ctx_size, chunk_word_count)

        if not samples:
            raise utils.GenerateException("Error: No samples found in leaf node.")

        if samples[0].get("document"):
            sdg = sdg_knowledge
            generated_data = generated_knowledge
        elif samples[0].get("context"):
            sdg = sdg_grounded_skill
            generated_data = generated_skills
        else:
            sdg = sdg_freeform_skill
            generated_data = generated_skills

        logger.debug("Samples: %s" % samples)
        ds = Dataset.from_list(samples)
        logger.debug("Dataset: %s" % ds)
        new_generated_data = sdg.generate(ds)
        generated_data.append(new_generated_data)

        logger.info("Generated %d samples" % len(new_generated_data))
        logger.debug("Generated data: %s" % new_generated_data)

    if generated_data is None:
        generated_data = []

    _gen_train_data(
        logger,
        generated_data,
        os.path.join(output_dir, output_file_train),
        os.path.join(output_dir, output_file_messages),
    )

    # TODO
    # This is for backwards compatibility. The file existing previously, so we'll keep it for now.
    # I believe the github bot assumes it is present for presenting generated data to a taxonomy
    # reviewer or contributor. Otherwise, I don't see a consumer of it in this repo or the
    # `ilab` CLI.

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")

    logger.info("Sampling mixed dataset for training")

    generated_knowledge.to_json(os.path.join(output_dir, "generated_knowledge.jsonl"))
    generated_skills.to_json(os.path.join(output_dir, "generated_skills.jsonl"))

    knowledge_recipe = datamixing.get_populated_recipe(
        recipe_path=configs_dir / "data_recipes" / "knowledge" / "default_recipe.yaml",
        generated_data_path=os.path.join(output_dir, "generated_knowledge.jsonl"),
        recipe_output_path=os.path.join(output_dir, "knowledge_recipe.yaml"),
    )

    skills_recipe = datamixing.get_populated_recipe(
        recipe_path=configs_dir / "data_recipes" / "skills" / "default_recipe.yaml",
        generated_data_path=os.path.join(output_dir, "generated_skills.jsonl"),
        recipe_output_path=os.path.join(output_dir, "skills_recipe.yaml"),
    )

    for recipe in [knowledge_recipe, skills_recipe]:
        if recipe["sys_prompt"]:
            sys_prompt = recipe.get("sys_prompt", None).strip()

        if recipe["drop_columns"]:
            drop_cols = recipe.get("drop_columns", None)

        datasets = []
        for dataset in recipe["datasets"]:
            path = dataset["path"]
            sampling_ratio = dataset["sampling_ratio"]
            ds = datamixing.load_ds(path, sampling_ratio)
            datasets.append(ds)

        ds = concatenate_datasets(datasets)
        ds = ds.map(
            datamixing.add_system_message,
            # pylint: disable-next=possibly-used-before-assignment
            fn_kwargs={"sys_prompt": sys_prompt},
            num_proc=32,
        )
        ds_train, ds_test = datamixing.generate_train_test_splits(ds)

        # pylint: disable-next=possibly-used-before-assignment
        datamixing.save(ds_train, drop_cols, recipe["save_dir"], f"train_{recipe["save_name"]}")
        datamixing.save(ds_test, drop_cols, recipe["save_dir"], f"test_{recipe["save_name"]}")
