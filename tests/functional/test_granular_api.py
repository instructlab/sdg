# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from unittest.mock import MagicMock
import glob
import pathlib

# Third Party
import git

# First Party
from instructlab.sdg import BlockRegistry
from instructlab.sdg.generate_data import (
    generate_taxonomy,
    mix_datasets,
    postprocess_taxonomy,
    preprocess_taxonomy,
)

# Local
from ..mockllmblock import MockLLMBlock


def _clone_instructlab_taxonomy(taxonomy_dir):
    taxonomy_repo_url = "https://github.com/instructlab/taxonomy"
    taxonomy_commit = "dfa3afaf26f40f923cf758389719619ec9b1ddb1"
    repo = git.Repo.clone_from(taxonomy_repo_url, taxonomy_dir, no_checkout=True)
    repo.git.checkout(taxonomy_commit)


def test_granular_api_end_to_end(testdata_path: pathlib.Path, tmp_path: pathlib.Path):
    # Registry our mock block so we can reference it in pipelines
    BlockRegistry.register("MockLLMBlock")(MockLLMBlock)

    # Clone a taxonomy and edit 1 file in it
    taxonomy_dir = tmp_path.joinpath("taxonomy")
    _clone_instructlab_taxonomy(taxonomy_dir)
    changed_qna_yaml = taxonomy_dir.joinpath(
        "knowledge", "science", "animals", "birds", "black_capped_chickadee", "qna.yaml"
    )
    with open(changed_qna_yaml, "a", encoding="utf-8") as file:
        file.write("")

    pipeline_dir = testdata_path.joinpath("mock_pipelines")
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")

    preprocessed_dir = tmp_path.joinpath("preprocessed")
    preprocess_taxonomy(
        taxonomy_dir=taxonomy_dir,
        output_dir=preprocessed_dir,
    )
    chickadee_docs = glob.glob(
        str(
            preprocessed_dir.joinpath(
                "documents", "knowledge_science_*", "chickadee.md"
            )
        )
    )
    assert chickadee_docs
    chickadee_samples_path = preprocessed_dir.joinpath(
        "knowledge_science_animals_birds_black_capped_chickadee.jsonl"
    )
    assert chickadee_samples_path.is_file()

    client = MagicMock()
    client.server_supports_batched = False
    generated_dir = tmp_path.joinpath("generated")
    generate_taxonomy(
        client=client,
        input_dir=preprocessed_dir,
        output_dir=generated_dir,
        pipeline=pipeline_dir,
    )
    generated_chickadee_samples_path = generated_dir.joinpath(
        "knowledge_science_animals_birds_black_capped_chickadee.jsonl"
    )
    assert generated_chickadee_samples_path.is_file()

    postprocessed_dir = tmp_path.joinpath("postprocessed")
    postprocess_taxonomy(
        input_dir=generated_dir,
        output_dir=postprocessed_dir,
        date_suffix=date_suffix,
        pipeline=pipeline_dir,
    )
    knowledge_recipe_file = postprocessed_dir.joinpath(
        f"knowledge_recipe_{date_suffix}.yaml"
    )
    assert knowledge_recipe_file.is_file()
    skills_recipe_file = postprocessed_dir.joinpath(f"skills_recipe_{date_suffix}.yaml")
    assert skills_recipe_file.is_file()

    mixed_skills_output_file = (
        f"{postprocessed_dir}/skills_train_msgs_{date_suffix}.jsonl"
    )
    mix_datasets(
        recipe_file=f"{postprocessed_dir}/skills_recipe_{date_suffix}.yaml",
        output_file=mixed_skills_output_file,
    )
    assert pathlib.Path(mixed_skills_output_file).is_file()
