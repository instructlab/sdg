# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# First Party
from instructlab.sdg import mix_datasets

output_dir = Path(__file__).parent.joinpath("output")
output_dir.mkdir(exist_ok=True)

concatenate_recipe_yaml = Path(__file__).parent.joinpath("concatenate_recipe.yaml")
concatenated_output_jsonl = output_dir.joinpath("concatenated.jsonl")
mix_datasets(concatenate_recipe_yaml, concatenated_output_jsonl)

weighted_recipe_yaml = Path(__file__).parent.joinpath("weighted_recipe.yaml")
weighted_output_jsonl = output_dir.joinpath("weighted.jsonl")
mix_datasets(weighted_recipe_yaml, weighted_output_jsonl)
