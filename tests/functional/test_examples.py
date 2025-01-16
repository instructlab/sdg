# SPDX-License-Identifier: Apache-2.0

# Standard
import pathlib
import shutil
import subprocess
import sys

# First Party
from instructlab.sdg.utils.json import jlload


def test_example_mixing(tmp_path: pathlib.Path, examples_path: pathlib.Path):
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
