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
import yaml

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
from instructlab.sdg.utils import GenerateException
from instructlab.sdg.utils.json import jldump, jlload
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)

logger = logging.getLogger(__name__)

_SYS_PROMPT = "I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model. My primary role is to serve as a chat assistant."

DEFAULT_CHUNK_WORD_COUNT = 1000
DEFAULT_TAXONOMY_BASE = "empty"
DEFAULT_SERVER_CTX_SIZE = 4096


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


def _knowledge_seed_example_to_test_data(seed_example, system_prompt, num_iterations=3):
    res = []
    for i in range(num_iterations):
        idx = i + 1
        user = seed_example[f"icl_query_{idx}"] + "\n" + seed_example["icl_document"]
        test_sample = {
            "user": _unescape(user),
            "assistant": _unescape(seed_example[f"icl_response_{idx}"]),
        }
        if system_prompt:
            test_sample["system"] = system_prompt
        res.append(test_sample)
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

        test_sample = {
            "user": _unescape(user),
            "assistant": _unescape(seed_example["seed_response"]),  # answer
        }
        if system_prompt:
            test_sample["system"] = system_prompt
        test_data.append(test_sample)

    jldump(test_data, output_file_test)


def _check_pipeline_dir(pipeline):
    for file in ["knowledge.yaml", "freeform_skills.yaml", "grounded_skills.yaml"]:
        if not os.path.exists(os.path.join(pipeline, file)):
            raise GenerateException(
                f"Error: pipeline directory ({pipeline}) does not contain {file}."
            )


def _locate_docling_models():
    # Search for the models in User and Site data directories
    data_dirs = [os.path.join(xdg_data_home(), "instructlab", "sdg")]
    data_dirs.extend(os.path.join(dir, "instructlab", "sdg") for dir in xdg_data_dirs())

    docling_model_path = None
    sdg_models_path = docling_model_path
    for d in data_dirs:
        if os.path.exists(os.path.join(d, "models")):
            sdg_models_path = os.path.join(d, "models")
            break

    if sdg_models_path is not None:
        try:
            with open(
                os.path.join(sdg_models_path, "config.yaml"), "r", encoding="utf-8"
            ) as file:
                config = yaml.safe_load(file)
                docling_model_path = config["models"][0]["path"]
        except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
            logger.warning(f"unable to read docling models path from config.yaml {e}")

    return docling_model_path


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


def preprocess_taxonomy(
    taxonomy_dir,
    output_dir,
    chunk_word_count=DEFAULT_CHUNK_WORD_COUNT,
    server_ctx_size=DEFAULT_SERVER_CTX_SIZE,
    taxonomy_base=DEFAULT_TAXONOMY_BASE,
    teacher_model_path: Optional[str] = None,
    yaml_rules: Optional[str] = None,
    test_output_file: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """
    Preprocess a taxonomy into input samples suitable for use with
    data generation pipelines. This does the following steps:

    - Determine changed leaf nodes in the taxonomy
    - Retrieve knowledge documents for changed taxonomy leaf nodes
    - Convert any non-markdown knowledge documents to markdown
    - Write the Docling json and markdown outputs from this conversion to
      disk for other processes to consume if needed.
    - Chunk the converted knowledge documents to the desired chunk sizes.
    - Turn the qna.yaml and knowledge documents into samples in the format
      expected by the `simple` and `full` data generation pipelines shipped
      in SDG.
    - Write these samples to disk, with one file per taxonomy leaf node.

    Args:
        taxonomy_dir: The path to the taxonomy
        output_dir: Where to write the samples create for use with data generation
        test_output_file: Path to write the test samples jsonl file
        chunk_word_count: The target number of words per document chunk
        server_ctx_size: The maximum number of tokens the inference server used
                         during data generation can handle
        taxonomy_base: Determines how we calculate what has changed. This should
                       be a git reference or the special value of 'empty' which
                       means assume the entire taxonomy has changed.
        teacher_model_path: Path to the teacher model on disk, which we'll use to
                            load its tokenizer for use with document chunking.
        yaml_rules: Path to a custom YAML rules file for YAML linting.
        test_output_file: Path to write a file with generated test samples
        system_prompt: System prompt to use when generating test samples

    Returns:
        List[str]: The list of output sample files written to disk.

    """
    logging.info("Converting taxonomy to samples")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_files = []

    if not (taxonomy_dir and os.path.exists(taxonomy_dir)):
        raise GenerateException(f"Error: taxonomy ({taxonomy_dir}) does not exist.")

    document_output_dir = output_dir.joinpath("documents")
    docling_model_path = _locate_docling_models()

    leaf_nodes = read_taxonomy_leaf_nodes(
        taxonomy_dir, taxonomy_base, yaml_rules, document_output_dir
    )
    if not leaf_nodes:
        raise GenerateException("Error: No new leaf nodes found in the taxonomy.")

    # TODO: This is all a temporary hack here, as we either need to
    # remove, deprecate, or otherwise determine the right way to
    # support test samples
    all_samples = []
    for leaf_node in leaf_nodes.values():
        leaf_node_path = leaf_node[0]["taxonomy_path"].replace("->", "_")
        samples = leaf_node_to_samples(
            leaf_node,
            server_ctx_size,
            chunk_word_count,
            document_output_dir,
            teacher_model_path,
            docling_model_path=docling_model_path,
        )

        if not samples:
            raise GenerateException("Error: No samples found in leaf node.")

        logger.debug("Samples: %s", samples)

        output_file = output_dir.joinpath(f"{leaf_node_path}.jsonl")
        all_samples.extend(samples)
        jldump(samples, output_file)
        output_files.append(str(output_file))

    if test_output_file:
        _gen_test_data(
            all_samples,
            test_output_file,
            system_prompt,
        )
        logger.debug(f"Generating test data to: {test_output_file}")
    logger.info("Taxonomy converted to samples and written to %s", output_dir)
    return output_files


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
    system_prompt: Optional[str] = None,
):
    recipe = Recipe(recipe_file, system_prompt)
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
        teacher_model_path=model_name,
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
        system_prompt=system_prompt,
    )
    mix_datasets(
        recipe_file=f"{output_dir}/knowledge_recipe_{date_suffix}.yaml",
        output_file=f"{output_dir}/knowledge_train_msgs_{date_suffix}.jsonl",
        system_prompt=system_prompt,
    )

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
