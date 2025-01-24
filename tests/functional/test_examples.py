# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

# Third Party
from docling.document_converter import DocumentConverter

# First Party
from instructlab.sdg.utils.json import jlload


def test_example_mixing(tmp_path: Path, examples_path: Path):
    example_copy_path = tmp_path.joinpath("mix_datasets")
    shutil.copytree(examples_path.joinpath("mix_datasets"), example_copy_path)
    script = example_copy_path.joinpath("example_mixing.py")
    subprocess.check_call([sys.executable, str(script)], text=True)

    concatenated = jlload(example_copy_path.joinpath("output", "concatenated.jsonl"))
    assert len(concatenated) == 10
    from_ds_1 = []
    from_ds_2 = []
    for sample in concatenated:
        if sample["id"].startswith("dataset_1"):
            from_ds_1.append(sample)
        else:
            from_ds_2.append(sample)
    assert len(from_ds_1) == len(from_ds_2) == 5

    weighted = jlload(example_copy_path.joinpath("output", "weighted.jsonl"))
    assert len(weighted) == 11
    from_ds_1 = []
    from_ds_2 = []
    for sample in weighted:
        if sample["id"].startswith("dataset_1"):
            from_ds_1.append(sample)
        else:
            from_ds_2.append(sample)
    assert len(from_ds_1) == 10
    assert len(from_ds_2) == 1


def _test_example_run_pipeline(example_dir):
    doc_converter = DocumentConverter()
    readme_path = example_dir.joinpath("README.md")
    assert readme_path.exists()
    conv_result = doc_converter.convert(readme_path)
    command_to_run = ""
    for item, _level in conv_result.document.iterate_items():
        if item.label == "code" and "run_pipeline" in item.text:
            command_to_run = item.text
    assert command_to_run
    # Turn the generic command into a list of shell arguments
    shell_args = shlex.split(command_to_run)
    # Ensure we use the proper Python - ie from tox environment
    shell_args[0] = sys.executable
    # Run the command with the current working directory set to our
    # example's subdirectory
    subprocess.check_call(shell_args, text=True, cwd=example_dir)


def test_example_iterblock(tmp_path: Path, examples_path: Path):
    shutil.copytree(
        examples_path.joinpath("blocks", "iterblock"), tmp_path, dirs_exist_ok=True
    )
    iterblock_path = tmp_path
    _test_example_run_pipeline(iterblock_path)
    output_jsonl = iterblock_path.joinpath("output.jsonl")
    assert output_jsonl.exists()
    output = jlload(output_jsonl)
    assert len(output) == 5
    assert output[4]["baz"] == "bar"
