"""Helper for test_sdg_imports"""

# Standard
import sys

# block slow imports
for unwanted in ["deepspeed", "llama_cpp", "torch", "transformers", "vllm"]:
    # importlib raises ModuleNotFound when sys.modules value is None.
    assert unwanted not in sys.modules
    sys.modules[unwanted] = None  # type: ignore[assignment]

# First Party
# This will trigger errors if any of the import chain tries to load
# the unwanted modules
from instructlab.sdg.generate_data import generate_data
