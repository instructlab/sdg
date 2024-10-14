# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import resources
from typing import Any
import logging
import re

# Third Party
from datasets import Dataset
import yaml

# First Party
from instructlab.sdg.pipeline import EVAL_PIPELINES_PKG, Pipeline

logger = logging.getLogger(__name__)


def _extract_options(text: str) -> list[Any]:
    """regex to extract options from mcq

    Args:
        text (str): question with options/mcq choices

    Returns:
        list[Any]: options under question that match the pattern.
    """
    # Use a regular expression to find patterns and capture the text after the letter and parenthesis
    pattern = r"\b[A-Z]\) (.+)"
    matches = re.findall(pattern, text)
    return matches


def _format_mmlu_style(ds: Dataset) -> Dataset:
    """Format the dataset according to lm-harness mmlu requirement.

    Args:
        ds (Dataset): input dataset

    Returns:
        Dataset: formated hf dataset
    """
    ds = ds.map(
        lambda x: {"answer": x["mmlubench_answer"][: x["mmlubench_answer"].index(")")]}
    )
    ds = ds.map(lambda x: {"choices": _extract_options(x["mmlubench_question"])})
    ds = ds.map(
        lambda x: {
            "question": x["mmlubench_question"][
                : x["mmlubench_question"].index("A)")
            ].strip()
        }
    )
    ds = ds.rename_columns({"domain": "subject"})
    ds = ds.filter(lambda x: x["choices"])
    ds = ds.filter(lambda x: len(x["choices"]) == 4)
    ds = ds.filter(lambda x: x["answer"] in ["A", "B", "C", "D"])
    # We filter out a lot of the dataset above (and in _post_process_mcq)
    # if we've managed to filter out all of the results we don't want to run class_encode_column
    # as the answer column might not exist
    if len(ds):
        ds = ds.class_encode_column("answer")
    return ds


def _post_process_mcq(ds: Dataset) -> Dataset:
    """Filter, format and return Multiple Choice Question (MCQ) dataset that
    is compatible with lm-harness for doing mmlu-style evaluation

    Filters out badly generated data, adds dataset type column

    Args:
        ds (Dataset): mcq generated dataset from mmlu pipeline

    Returns:
        Dataset: Hf Dataset with new column, filtered dataset
    """
    ds = ds.filter(lambda x: ")" in x["mmlubench_answer"])
    ds = ds.filter(lambda x: "A)" in x["mmlubench_question"])
    ds = ds.add_column("dataset_type", ["mcq_qa"] * ds.num_rows)
    return _format_mmlu_style(ds)


def _create_mmlu_evaluation_task(task_name, eval_data_file_path, yaml_file_path):
    """
    Prepare Task Yaml that will be used in by the instructlab.sdg library using lm_eval_harness to evaluate knowledge using mmlu style metric
    see: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
         https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#writing-a-prompt-template
    """

    # The features we know to be required are question, choices, and answer
    # TODO: Remove excess features its more clear what isn't required
    task_yaml = {
        "task": task_name,
        "dataset_path": "json",
        "dataset_name": None,
        "test_split": "test",
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": "{{[choices[0], choices[1], choices[2], choices[3]]}}",
        "doc_to_target": "{{answer}}",
        "output_type": "multiple_choice",
        "metric_list": [
            {
                "metric": "acc",
                "aggregation": "mean",
                "higher_is_better": "true",
            }
        ],
        "dataset_kwargs": {"data_files": {"test": eval_data_file_path}},
        "tag": "mmlu_pr",
    }
    with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(task_yaml, yaml_file, default_flow_style=False)


def generate_eval_task_data(
    mmlubench_pipe, task_name, samples, output_dir, date_suffix
):
    mmlubench_data = mmlubench_pipe.generate(samples)
    if len(mmlubench_data):
        mmlubench_data = _post_process_mcq(mmlubench_data)

    eval_data_file_path = (
        f"{output_dir}/node_datasets_{date_suffix}/mmlubench_{task_name}.jsonl"
    )
    logger.info(f"Saving MMLU Dataset {eval_data_file_path}")
    mmlubench_data.to_json(eval_data_file_path, orient="records", lines=True)

    yaml_file_path = f"{output_dir}/node_datasets_{date_suffix}/{task_name}_task.yaml"
    logger.info(f"Saving MMLU Task yaml {yaml_file_path}")
    _create_mmlu_evaluation_task(
        task_name=task_name,
        eval_data_file_path=eval_data_file_path,
        yaml_file_path=yaml_file_path,
    )


def mmlubench_pipe_init(ctx):
    with resources.as_file(
        resources.files(EVAL_PIPELINES_PKG).joinpath("mmlu_bench.yaml")
    ) as yaml_path:
        return Pipeline.from_file(ctx, yaml_path)
