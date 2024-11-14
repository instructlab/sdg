# Standard
import json
import logging
import pathlib
import struct

# Third Party
from gguf.constants import GGUF_MAGIC

logger = logging.getLogger(__name__)


def is_model_safetensors(model_path: pathlib.Path) -> bool:
    """Check if model_path is a valid safe tensors directory

    Directory must contain a specific set of files to qualify as a safetensors model directory
    Args:
        model_path (Path): The path to the model directory
    Returns:
        bool: True if the model is a safetensors model, False otherwise.
    """
    try:
        files = list(model_path.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.debug("Failed to read directory: %s", e)
        return False

    # directory should contain either .safetensors or .bin files to be considered valid
    filetypes = [file.suffix for file in files]
    if not ".safetensors" in filetypes and not ".bin" in filetypes:
        logger.debug("'%s' has no .safetensors or .bin files", model_path)
        return False

    basenames = {file.name for file in files}
    requires_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    diff = requires_files.difference(basenames)
    if diff:
        logger.debug("'%s' is missing %s", model_path, diff)
        return False

    for file in model_path.glob("*.json"):
        try:
            with file.open(encoding="utf-8") as f:
                json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            logger.debug("'%s' is not a valid JSON file: e", file, e)
            return False

    return True


def is_model_gguf(model_path: pathlib.Path) -> bool:
    """
    Check if the file is a GGUF file.
    Args:
        model_path (Path): The path to the file.
    Returns:
        bool: True if the file is a GGUF file, False otherwise.
    """
    try:
        with model_path.open("rb") as f:
            first_four_bytes = f.read(4)

        # Convert the first four bytes to an integer
        first_four_bytes_int = int(struct.unpack("<I", first_four_bytes)[0])

        return first_four_bytes_int == GGUF_MAGIC
    except struct.error as e:
        logger.debug(
            f"Failed to unpack the first four bytes of {model_path}. "
            f"The file might not be a valid GGUF file or is corrupted: {e}"
        )
        return False
    except IsADirectoryError as e:
        logger.debug(f"GGUF Path {model_path} is a directory, returning {e}")
        return False
    except OSError as e:
        logger.debug(f"An unexpected error occurred while processing {model_path}: {e}")
        return False
