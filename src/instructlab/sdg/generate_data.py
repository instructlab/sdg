# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Optional
import dataclasses
import json
import logging
import os
import time

# Third Party
# instructlab - All of these need to go away (other than sdg) - issue #6
from datasets import Dataset
from xdg_base_dirs import xdg_data_dirs, xdg_data_home
import openai

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg.datamixing import DataMixer, _get_question_hack, _get_response_hack
from instructlab.sdg.eval_data import generate_eval_task_data, mmlubench_pipe_init
from instructlab.sdg.llmblock import MODEL_FAMILY_MERLINITE, MODEL_FAMILY_MIXTRAL
from instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    SIMPLE_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)
from instructlab.sdg.utils import GenerateException, models
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)

logger = logging.getLogger(__name__)

_SYS_PROMPT = "I am, Red Hat® Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant."


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


def _convert_to_messages(sample):
    """
    Convert a sample dictionary to contain 'messages' and 'metadata' columns required for training.

    Note that this is for the legacy messages format, used before data
    mixing was introduced. Once we can drop the older `messages_*.jsonl`
    output files, this can go away.
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


def _gen_train_data(machine_instruction_data, output_file_train, output_file_messages):
    """
    Generate training data in the legacy system/user/assistant format
    used in train_*.jsonl as well as the legacy messages format used
    in messages_*.jsonl files.

    This can be dropped once we no longer need those formats and are fully
    using the new data mixing messages format.
    """
    train_data = []
    messages_data = []

    for output_dataset in machine_instruction_data:
        for synth_example in output_dataset:
            logger.debug(synth_example)
            user = _get_question_hack(synth_example)
            if len(synth_example.get("context", "")) > 0:
                user += "\n" + synth_example["context"]
            assistant = _unescape(_get_response_hack(synth_example))
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


def _knowledge_seed_example_to_test_data(seed_example):
    res = []
    for qna in seed_example["questions_and_answers"]:
        user = qna["question"] + "\n" + seed_example["context"]
        res.append(
            {
                "system": _SYS_PROMPT,
                "user": _unescape(user),
                "assistant": _unescape(qna["answer"]),
            }
        )
    return res


def _gen_test_data(
    leaf_nodes,
    output_file_test,
):
    """
    Generate test data in the format needed by the legacy Linux training
    in instructlab/instructlab.
    """
    test_data = []
    for _, leaf_node in leaf_nodes.items():
        for seed_example in leaf_node:
            if "questions_and_answers" in seed_example:
                test_data.extend(_knowledge_seed_example_to_test_data(seed_example))
                continue

            # skill seed example

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


def _check_pipeline_dir(pipeline):
    for file in ["knowledge.yaml", "freeform_skills.yaml", "grounded_skills.yaml"]:
        if not os.path.exists(os.path.join(pipeline, file)):
            raise GenerateException(
                f"Error: pipeline directory ({pipeline}) does not contain {file}."
            )


def _context_init(
    client: openai.OpenAI,
    model_family: str,
    model_id: str,
    num_instructions_to_generate: int,
    checkpoint_dir: str,
    save_freq: int,
    batch_num_workers: Optional[int],
    batch_size: Optional[int],
):
    extra_kwargs = {}
    if batch_size is not None:
        extra_kwargs["batch_size"] = batch_size
        extra_kwargs["batch_num_workers"] = batch_num_workers

    return PipelineContext(
        client=client,
        model_family=model_family,
        model_id=model_id,
        num_instructions_to_generate=num_instructions_to_generate,
        checkpoint_dir=checkpoint_dir,
        save_freq=save_freq,
        **extra_kwargs,
    )


def _sdg_init(ctx, pipeline):
    pipeline_pkg = None

    # Search for the pipeline in User and Site data directories
    # then for a package defined pipeline
    # and finally pipelines referenced by absolute path
    data_dirs = [os.path.join(xdg_data_home(), "instructlab", "sdg")]
    data_dirs.extend(os.path.join(dir, "instructlab", "sdg") for dir in xdg_data_dirs())

    for d in data_dirs:
        pipeline_path = os.path.join(d, "pipelines", pipeline)
        if os.path.exists(pipeline_path):
            _check_pipeline_dir(pipeline_path)
            break
    else:
        if pipeline == "full":
            pipeline_pkg = FULL_PIPELINES_PACKAGE
        elif pipeline == "simple":
            pipeline_pkg = SIMPLE_PIPELINES_PACKAGE
        else:
            # Validate that pipeline is a valid directory and that it contains the required files
            if not os.path.exists(pipeline):
                raise GenerateException(
                    f"Error: pipeline directory ({pipeline}) does not exist."
                )
            _check_pipeline_dir(pipeline)

    def load_pipeline(yaml_basename):
        if pipeline_pkg:
            with resources.as_file(
                resources.files(pipeline_pkg).joinpath(yaml_basename)
            ) as yaml_path:
                return Pipeline.from_file(ctx, yaml_path)
        else:
            return Pipeline.from_file(ctx, os.path.join(pipeline, yaml_basename))

    return (
        load_pipeline("knowledge.yaml"),
        load_pipeline("freeform_skills.yaml"),
        load_pipeline("grounded_skills.yaml"),
    )


def _mixer_init(ctx, output_dir, date_suffix, knowledge_auxiliary_inst):
    data_dirs = [os.path.join(xdg_data_home(), "instructlab", "sdg")]
    data_dirs.extend(os.path.join(dir, "instructlab", "sdg") for dir in xdg_data_dirs())

    return DataMixer(
        data_dirs,
        output_dir,
        date_suffix,
        _SYS_PROMPT,
        ctx.dataset_num_procs,
        knowledge_auxiliary_inst,
    )


# This is part of the public API, and used by instructlab.
# TODO - parameter removal needs to be done in sync with a CLI change.
# to be removed: logger, prompt_file_path, rouge_threshold, tls_*
def generate_data(
    client: openai.OpenAI,
    logger: logging.Logger = logger,  # pylint: disable=redefined-outer-name
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
    output_dir: Optional[str] = None,
    # TODO - not used and should be removed from the CLI
    prompt_file_path: Optional[str] = None,  # pylint: disable=unused-argument
    # TODO - probably should be removed
    rouge_threshold: Optional[float] = None,  # pylint: disable=unused-argument
    console_output=True,
    yaml_rules: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    pipeline: Optional[str] = "simple",
    batch_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """Generate data for training and testing a model.

    This currently serves as the primary interface from the `ilab` CLI to the `sdg` library.
    It is somewhat a transitionary measure, as this function existed back when all of the
    functionality was embedded in the CLI. At some stage, we expect to evolve the CLI to
    use the SDG library constructs directly, and this function will likely be removed.

    Args:
        pipeline: This argument may be either an alias defined in a user or site "data directory"
                  or an alias defined by the sdg library ("simple", "full")(if the data directory has no matches),
                  or an absolute path to a directory containing the pipeline YAML files.
                  We expect three files to be present in this directory: "knowledge.yaml",
                    "freeform_skills.yaml", and "grounded_skills.yaml".
    """
    generate_start = time.time()

    # FIXME: remove this when ilab knows to pass batch_size=0 with llama.cpp
    if batch_size is None:
        batch_size = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (taxonomy and os.path.exists(taxonomy)):
        raise GenerateException(f"Error: taxonomy ({taxonomy}) does not exist.")

    leaf_nodes = read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules)
    if not leaf_nodes:
        raise GenerateException("Error: No new leaf nodes found in the taxonomy.")

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file_messages = f"messages_{name}_{date_suffix}.jsonl"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    output_file_train = f"train_{name}_{date_suffix}.jsonl"

    _gen_test_data(
        leaf_nodes,
        os.path.join(output_dir, output_file_test),
    )

    logger.debug(f"Generating to: {os.path.join(output_dir, output_file_test)}")

    if models.get_model_family(model_family, model_name) == "mixtral":
        model_family = MODEL_FAMILY_MIXTRAL
    else:
        model_family = MODEL_FAMILY_MERLINITE

    ctx = _context_init(
        client,
        model_family,
        model_name,
        num_instructions_to_generate,
        checkpoint_dir,
        1,  # save_freq
        batch_size=batch_size,
        batch_num_workers=num_cpus,
    )

    knowledge_pipe, freeform_skills_pipe, grounded_skills_pipe = _sdg_init(
        ctx, pipeline
    )

    # Make sure checkpointing is disabled (we don't want this pipeline to load checkpoints from the main pipeline)
    mmlu_ctx = dataclasses.replace(ctx, checkpoint_dir=None)
    mmlu_bench_pipe = mmlubench_pipe_init(mmlu_ctx)

    mixer = _mixer_init(ctx, output_dir, date_suffix, knowledge_pipe.auxiliary_inst)

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    generated_data = None
    empty_sdg_leaf_nodes = []
    for leaf_node in leaf_nodes.values():
        is_knowledge = False
        leaf_node_path = leaf_node[0]["taxonomy_path"].replace("->", "_")
        samples = leaf_node_to_samples(leaf_node, server_ctx_size, chunk_word_count)

        if not samples:
            raise GenerateException("Error: No samples found in leaf node.")

        if samples[0].get("document"):
            pipe = knowledge_pipe
            is_knowledge = True

        elif samples[0].get("seed_context"):
            pipe = grounded_skills_pipe

        else:
            pipe = freeform_skills_pipe

        logger.debug("Samples: %s", samples)
        ds = Dataset.from_list(samples)
        logger.debug("Dataset: %s", ds)
        new_generated_data = pipe.generate(ds, leaf_node_path)
        if len(new_generated_data) == 0:
            empty_sdg_leaf_nodes.append(leaf_node_path)
            logger.warning("Empty dataset for qna node: %s", leaf_node_path)
            continue
        generated_data = (
            [new_generated_data]
            if generated_data is None
            else generated_data + [new_generated_data]
        )
        logger.info("Generated %d samples", len(generated_data))
        logger.debug("Generated data: %s", generated_data)

        if is_knowledge:
            # generate mmlubench data for the current leaf node
            generate_eval_task_data(
                mmlu_bench_pipe,
                leaf_node_path,
                ds,
                output_dir,
                date_suffix,
            )

        mixer.collect(leaf_node_path, new_generated_data, is_knowledge)

    if generated_data is None:
        generated_data = []

    _gen_train_data(
        generated_data,
        os.path.join(output_dir, output_file_train),
        os.path.join(output_dir, output_file_messages),
    )

    mixer.generate()

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
    if len(empty_sdg_leaf_nodes) > 0:
        logger.warning(
            "Leaf nodes with empty sdg output: {}".format(
                " ".join(empty_sdg_leaf_nodes)
            )
        )
