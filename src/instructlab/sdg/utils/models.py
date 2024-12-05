# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import re

# First Party
from instructlab.sdg.registry import PromptRegistry
from instructlab.sdg.utils import GenerateException

# When otherwise unknown, ilab uses this as the default family
DEFAULT_MODEL_FAMILY = "merlinite"


def get_model_family(model_family, model_path):
    registry = PromptRegistry.get_registry()

    # A model_family was given, so use it explicitly
    if model_family:
        if model_family not in registry:
            raise GenerateException("Unknown model family: %s" % model_family)
        return model_family

    # Try to guess the model family based on the model's filename
    if model_path:
        guess = re.match(r"^\w*", os.path.basename(model_path)).group(0).lower()
        if guess in registry:
            return guess

    # Nothing was found, so just return the default
    return DEFAULT_MODEL_FAMILY
