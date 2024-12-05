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
from instructlab.sdg.blocks.llmblock import DEFAULT_MAX_NUM_TOKENS
from instructlab.sdg.datamixing import DataMixer, _get_question_hack, _get_response_hack
from instructlab.sdg.eval_data import generate_eval_task_data, mmlubench_pipe_init
from instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    SIMPLE_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)
from instructlab.sdg.taxonomy import taxonomy_to_samples
from instructlab.sdg.utils import GenerateException, models
from instructlab.sdg.utils.json import jldump, jlload

logger = logging.getLogger(__name__)

_SYS_PROMPT = "I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model. My primary role is to serve as a chat assistant."


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


def _gen_train_data(
    machine_instruction_data, output_file_train, output_file_messages, system_prompt
):
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
                "system": system_prompt,
                "user": _unescape(user),
                "assistant": assistant,
            }
            train_data.append(train_entry)
            sample = {
                "inputs": _unescape(user),
                "targets": assistant,
                "system": system_prompt,
            }
            messages_data.append(_convert_to_messages(sample))

    jldump(train_data, output_file_train)

    jldump(messages_data, output_file_messages)


def _knowledge_seed_example_to_test_data(seed_example, system_prompt):
    res = []
    for i in range(3):
        idx = i + 1
        user = seed_example[f"icl_query_{idx}"] + "\n" + seed_example["icl_document"]
        res.append(
            {
                "system": system_prompt,
                "user": _unescape(user),
                "assistant": _unescape(seed_example[f"icl_response_{idx}"]),
            }
        )
    return res


def _gen_test_data(
    seed_examples,
    output_file_test,
    system_prompt,
):
    """
    Generate test data in the format needed by the legacy Linux training
    in instructlab/instructlab.
    """
    test_data = []
    for seed_example in seed_examples:
        if "icl_query_1" in seed_example:
            test_data.extend(
                _knowledge_seed_example_to_test_data(seed_example, system_prompt)
            )
            continue

        # skill seed example

        user = seed_example["seed_question"]  # question

        if seed_example["leaf_node_type"] == "grounded_skill":
            user += "\n" + seed_example["seed_context"]  # context

        test_data.append(
            {
                "system": system_prompt,
                "user": _unescape(user),
                "assistant": _unescape(seed_example["seed_response"]),  # answer
            }
        )

    jldump(test_data, output_file_test)


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
    max_num_tokens: Optional[int] = DEFAULT_MAX_NUM_TOKENS,
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
        max_num_tokens=max_num_tokens,
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


def _mixer_init(
    ctx,
    output_dir,
    date_suffix,
    knowledge_auxiliary_inst,
    system_prompt,
):
    data_dirs = [os.path.join(xdg_data_home(), "instructlab", "sdg")]
    data_dirs.extend(os.path.join(dir, "instructlab", "sdg") for dir in xdg_data_dirs())

    return DataMixer(
        data_dirs,
        output_dir,
        date_suffix,
        system_prompt,
        ctx.dataset_num_procs,
        knowledge_auxiliary_inst,
    )


# This is part of the public API, and used by instructlab.
# TODO - parameter removal needs to be done in sync with a CLI change.
# to be removed: logger
def generate_data(
    client: openai.OpenAI,
    logger: logging.Logger = logger,  # pylint: disable=redefined-outer-name
    system_prompt: Optional[str] = None,
    use_legacy_pretraining_format: Optional[bool] = True,
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    taxonomy: Optional[str] = None,  # TODO rename to taxonomy_path to match config
    taxonomy_base: Optional[str] = None,
    output_dir: Optional[str] = None,
    console_output=True,
    yaml_rules: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    pipeline: Optional[str] = "simple",
    batch_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    max_num_tokens: Optional[int] = DEFAULT_MAX_NUM_TOKENS,
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

    system_prompt = system_prompt if system_prompt is not None else _SYS_PROMPT

    # FIXME: remove this when ilab knows to pass batch_size=0 with llama.cpp
    if batch_size is None:
        batch_size = 0

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    preprocessed_output_dir = output_dir.joinpath(f"preprocessed_{date_suffix}")

    # This writes samples to disk in our output_dir and returns the
    # list of files created
    sample_files = taxonomy_to_samples(
        taxonomy,
        preprocessed_output_dir,
        chunk_word_count=chunk_word_count,
        server_ctx_size=server_ctx_size,
        taxonomy_base=taxonomy_base,
        yaml_rules=yaml_rules,
    )

    name = Path(model_name).stem  # Just in case it is a file path
    output_file_messages = f"messages_{name}_{date_suffix}.jsonl"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    output_file_train = f"train_{name}_{date_suffix}.jsonl"

    all_samples = []
    for sample_file in sample_files:
        all_samples.extend(jlload(sample_file))
    _gen_test_data(
        all_samples,
        os.path.join(output_dir, output_file_test),
        system_prompt,
    )

    logger.debug(f"Generating to: {os.path.join(output_dir, output_file_test)}")

    model_family = models.get_model_family(model_family, model_name)

    ctx = _context_init(
        client,
        model_family,
        model_name,
        num_instructions_to_generate,
        checkpoint_dir,
        1,  # save_freq
        batch_size=batch_size,
        batch_num_workers=num_cpus,
        max_num_tokens=max_num_tokens,
    )

    knowledge_pipe, freeform_skills_pipe, grounded_skills_pipe = _sdg_init(
        ctx, pipeline
    )

    # Make sure checkpointing is disabled (we don't want this pipeline to load checkpoints from the main pipeline)
    mmlu_ctx = dataclasses.replace(ctx, checkpoint_dir=None)
    mmlu_bench_pipe = mmlubench_pipe_init(mmlu_ctx)

    mixer = _mixer_init(
        ctx,
        output_dir,
        date_suffix,
        knowledge_pipe.auxiliary_inst,
        system_prompt,
    )

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    generated_data = []
    empty_input_sample_files = []
    for sample_file in sample_files:
        logger.debug("Generating data from input sample file: %s", sample_file)
        samples = jlload(sample_file)
        if not samples:
            raise GenerateException(
                "Error: No samples found in input file {sample_file}"
            )
        # For now we assume every sample in the file is the same type
        first_sample = samples[0]
        leaf_node_path = first_sample["leaf_node_path"]
        leaf_node_type = first_sample["leaf_node_type"]
        is_knowledge = False
        if leaf_node_type == "knowledge":
            pipe = knowledge_pipe
            is_knowledge = True
        elif leaf_node_type == "grounded_skill":
            pipe = grounded_skills_pipe
        else:
            pipe = freeform_skills_pipe

        samples_ds = Dataset.from_list(samples)
        logger.debug("Samples: %s", samples_ds)

        new_generated_data = pipe.generate(samples_ds, leaf_node_path)
        if len(new_generated_data) == 0:
            empty_input_sample_files.append(sample_file)
            logger.warning("Empty generated dataset for sample file: %s", sample_file)
            continue
        generated_data.append(new_generated_data)

        logger.info("Generated %d samples", len(generated_data))
        logger.debug("Generated data: %s", generated_data)

        if is_knowledge:
            # generate mmlubench data for the current leaf node
            generate_eval_task_data(
                mmlu_bench_pipe,
                leaf_node_path,
                samples,
                output_dir,
                date_suffix,
            )

        mixer.collect(
            leaf_node_path,
            new_generated_data,
            is_knowledge,
            use_legacy_pretraining_format,
        )

    _gen_train_data(
        generated_data,
        os.path.join(output_dir, output_file_train),
        os.path.join(output_dir, output_file_messages),
        system_prompt,
    )

    mixer.generate()

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
    if len(empty_input_sample_files) > 0:
        logger.warning(
            "Input sample files with empty sdg output: {}".format(
                " ".join(empty_input_sample_files)
            )
        )
