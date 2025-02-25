"""Helper for test_sdg_imports"""

# Standard
import sys

# block slow imports
for unwanted in ["deepspeed", "llama_cpp", "torch", "vllm"]:
    # importlib raises ModuleNotFound when sys.modules value is None.
    assert unwanted not in sys.modules
    sys.modules[unwanted] = None  # type: ignore[assignment]

# Try to import in your PR to see if this works around the issue. If not, print an error
try:
    # Third Party
    import docling_core
except ImportError as e:
    print(f"Could not import `docling_core` because: {e}")

# First Party
# This will trigger errors if any of the import chain tries to load
# the unwanted modules
from instructlab.sdg.generate_data import generate_data
