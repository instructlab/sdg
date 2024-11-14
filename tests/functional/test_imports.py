# SPDX-License-Identifier: Apache-2.0

# Standard
import pathlib
import subprocess
import sys


def test_sdg_imports(testdata_path: pathlib.Path):
    script = testdata_path / "leanimports.py"
    subprocess.check_call([sys.executable, str(script)], text=True)
