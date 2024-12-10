# SPDX-License-Identifier: Apache-2.0

# Standard
import pathlib
import subprocess
import sys


def test_custom_block(testdata_path: pathlib.Path):
    script = testdata_path.joinpath("custom_block.py")
    subprocess.check_call([sys.executable, str(script)], text=True)


def test_custom_prompt(testdata_path: pathlib.Path):
    script = testdata_path.joinpath("custom_prompt.py")
    subprocess.check_call([sys.executable, str(script)], text=True)
