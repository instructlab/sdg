# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Optional
import glob
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
from instructlab.sdg.datamixing import (
    DataMixer,
    Recipe,
    _get_question_hack,
    _get_response_hack,
)
from instructlab.sdg.eval_data import generate_eval_task_data, mmlubench_pipe_init
from instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    SIMPLE_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)
from instructlab.sdg.taxonomy import preprocess_taxonomy
from instructlab.sdg.utils import GenerateException
from instructlab.sdg.utils.json import jldump, jlload
from instructlab.sdg.utils.taxonomy import _unescape

logger = logging.getLogger(__name__)

_SYS_PROMPT = "I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model. My primary role is to serve as a chat assistant."


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
    num_procs,
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
        num_procs,
        knowledge_auxiliary_inst,
    )


def _extract_leaf_node_path_and_type(sample):
    leaf_node_path = sample.get("leaf_node_path", "unknown")
    leaf_node_type = sample.get("leaf_node_type")
    return leaf_node_path, leaf_node_type


def generate_taxonomy(
    client: openai.OpenAI,
    input_dir: str,
    output_dir: str,
    logger: logging.Logger = logger,  # pylint: disable=redefined-outer-name
    model_family: Optional[str] = None,
    model_id: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    console_output=True,
    pipeline: Optional[str] = "simple",
    batch_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    max_num_tokens: Optional[int] = DEFAULT_MAX_NUM_TOKENS,
):
    ctx = _context_init(
        client,
        model_family,
        model_id,
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

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    input_files = glob.glob(f"{input_dir}/*.jsonl")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    empty_input_files = []
    for input_file in input_files:
        logger.debug("Generating data from input file: %s", input_file)
        samples = jlload(input_file)
        if not samples:
            raise GenerateException(
                "Error: No samples found in input file {input_file}"
            )
        # For now we assume every sample in the file is the same type
        first_sample = samples[0]
        leaf_node_path, leaf_node_type = _extract_leaf_node_path_and_type(first_sample)
        if leaf_node_type == "knowledge":
            pipe = knowledge_pipe
        elif leaf_node_type == "grounded_skill":
            pipe = grounded_skills_pipe
        else:
            pipe = freeform_skills_pipe

        samples_ds = Dataset.from_list(samples)
        logger.debug("Generating from samples: %s", samples_ds)

        new_generated_data = pipe.generate(samples_ds, leaf_node_path)
        if len(new_generated_data) == 0:
            empty_input_files.append(input_file)
            logger.warning("Empty generated dataset for input file: %s", input_file)
            continue

        output_file = os.path.join(output_dir, os.path.basename(input_file))
        jldump(new_generated_data, output_file)
        logger.info("Generated %d samples", len(new_generated_data))
        logger.debug("Generated data: %s", new_generated_data)

    if len(empty_input_files) > 0:
        logger.warning(
            "Input sample files with empty sdg output: {}".format(
                " ".join(empty_input_files)
            )
        )


def generate_taxonomy_eval(
    client: openai.OpenAI,
    input_dir: str,
    output_dir: str,
    date_suffix: str,
    model_family: Optional[str] = None,
    model_id: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    batch_size: Optional[int] = None,
    max_num_tokens: Optional[int] = DEFAULT_MAX_NUM_TOKENS,
):
    ctx = _context_init(
        client,
        model_family,
        model_id,
        num_instructions_to_generate,
        None,  # disable checkpoints for eval pipeline
        1,  # save_freq
        batch_size=batch_size,
        batch_num_workers=num_cpus,
        max_num_tokens=max_num_tokens,
    )
    mmlu_bench_pipe = mmlubench_pipe_init(ctx)

    input_files = glob.glob(f"{input_dir}/*.jsonl")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for input_file in input_files:
        logger.debug("Generating eval data from input file: %s", input_file)
        samples = jlload(input_file)
        if not samples:
            raise GenerateException(
                "Error: No samples found in input file {input_file}"
            )
        samples_ds = Dataset.from_list(samples)
        # For now we assume every sample in the file is the same type
        first_sample = samples[0]
        leaf_node_path, leaf_node_type = _extract_leaf_node_path_and_type(first_sample)
        is_knowledge = False
        if leaf_node_type == "knowledge":
            is_knowledge = True

        if is_knowledge:
            generate_eval_task_data(
                mmlu_bench_pipe,
                leaf_node_path,
                samples_ds,
                output_dir,
                date_suffix,
            )


def postprocess_taxonomy(
    input_dir: str,
    output_dir: str,
    date_suffix: str,
    pipeline: Optional[str] = "simple",
    num_procs: Optional[int] = PipelineContext.DEFAULT_DATASET_NUM_PROCS,
    system_prompt: Optional[str] = _SYS_PROMPT,
    use_legacy_pretraining_format: Optional[bool] = True,
):
    knowledge_pipe, _, _ = _sdg_init(None, pipeline)
    mixer = _mixer_init(
        num_procs,
        output_dir,
        date_suffix,
        knowledge_pipe.auxiliary_inst,
        system_prompt,
    )

    input_files = glob.glob(f"{input_dir}/*.jsonl")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_file_messages = f"messages_{date_suffix}.jsonl"
    output_file_train = f"train_{date_suffix}.jsonl"

    all_generated_data = []
    for input_file in input_files:
        logger.debug(
            "Postprocessing generated taxonomy date in input file: %s", input_file
        )
        samples = jlload(input_file)
        if not samples:
            raise GenerateException(
                "Error: No samples found in input file {input_file}"
            )
        # For now we assume every sample in the file is the same type
        first_sample = samples[0]
        leaf_node_path, leaf_node_type = _extract_leaf_node_path_and_type(first_sample)
        is_knowledge = False
        if leaf_node_type == "knowledge":
            is_knowledge = True

        samples_ds = Dataset.from_list(samples)
        logger.debug("Postprocessing from samples: %s", samples_ds)
        all_generated_data.append(samples_ds)

        mixer.collect(
            leaf_node_path,
            samples_ds,
            is_knowledge,
            use_legacy_pretraining_format,
        )

    _gen_train_data(
        all_generated_data,
        os.path.join(output_dir, output_file_train),
        os.path.join(output_dir, output_file_messages),
        system_prompt,
    )

    mixer.write_recipes()


def mix_datasets(
    recipe_file: str,
    output_file: str,
    num_proc: Optional[int] = 8,
):
    recipe = Recipe(recipe_file)
    if recipe.datasets:
        recipe.save_mixed_dataset(output_file, num_proc)
    else:
        logger.info("Not mixing empty recipe file: %s", recipe_file)


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

    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_test = output_dir.joinpath(f"test_{date_suffix}.jsonl")
    preprocessed_dir = output_dir.joinpath(f"preprocessed_{date_suffix}")
    generated_dir = output_dir.joinpath(f"generated_{date_suffix}")

    # This writes samples to disk in our output_dir and returns the
    # list of files created
    preprocess_taxonomy(
        taxonomy,
        output_dir=preprocessed_dir,
        chunk_word_count=chunk_word_count,
        server_ctx_size=server_ctx_size,
        taxonomy_base=taxonomy_base,
        yaml_rules=yaml_rules,
        test_output_file=output_file_test,
        system_prompt=system_prompt,
    )

    generate_taxonomy(
        client,
        input_dir=preprocessed_dir,
        output_dir=generated_dir,
        logger=logger,
        model_family=model_family,
        model_id=model_name,
        num_cpus=num_cpus,
        num_instructions_to_generate=num_instructions_to_generate,
        console_output=console_output,
        pipeline=pipeline,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        max_num_tokens=max_num_tokens,
    )

    generate_taxonomy_eval(
        input_dir=preprocessed_dir,
        output_dir=output_dir,
        date_suffix=date_suffix,
        client=client,
        model_family=model_family,
        model_id=model_name,
        num_cpus=num_cpus,
        num_instructions_to_generate=num_instructions_to_generate,
        batch_size=batch_size,
        max_num_tokens=max_num_tokens,
    )

    postprocess_taxonomy(
        input_dir=generated_dir,
        output_dir=output_dir,
        date_suffix=date_suffix,
        pipeline=pipeline,
        system_prompt=system_prompt,
        use_legacy_pretraining_format=use_legacy_pretraining_format,
    )

    mix_datasets(
        recipe_file=f"{output_dir}/skills_recipe_{date_suffix}.yaml",
        output_file=f"{output_dir}/skills_train_msgs_{date_suffix}.jsonl",
    )
    mix_datasets(
        recipe_file=f"{output_dir}/knowledge_recipe_{date_suffix}.yaml",
        output_file=f"{output_dir}/knowledge_train_msgs_{date_suffix}.jsonl",
    )

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
