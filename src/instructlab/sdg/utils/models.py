# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import re

# First Party
from instructlab.sdg.utils import GenerateException

# When otherwise unknown, ilab uses this as the default family
DEFAULT_MODEL_FAMILY = "merlinite"

# Model families understood by ilab
MODEL_FAMILIES = set(("merlinite", "mixtral"))

# Map model names to their family
MODEL_FAMILY_MAPPINGS = {"granite": "merlinite", "mistral": "mixtral"}


def get_model_family(model_family, model_path):
    model_family_retrieved = MODEL_FAMILY_MAPPINGS.get(model_family, model_family)
    if model_family_retrieved and model_family_retrieved.lower() not in MODEL_FAMILIES:
        raise GenerateException("Unknown model family: %s" % model_family_retrieved)

    # Try to guess the model family based on the model's filename
    guess = re.match(r"^\w*", os.path.basename(model_path)).group(0).lower()
    guess = MODEL_FAMILY_MAPPINGS.get(guess, guess)

    return guess if guess in MODEL_FAMILIES else DEFAULT_MODEL_FAMILY
