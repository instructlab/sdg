# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from unittest.mock import MagicMock
import glob
import os
import pathlib
import unittest

# Third Party
import git
import pytest

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
from ..taxonomy import load_test_skills


def _clone_instructlab_taxonomy(taxonomy_dir):
    taxonomy_repo_url = "https://github.com/instructlab/taxonomy"
    taxonomy_commit = "dfa3afaf26f40f923cf758389719619ec9b1ddb1"
    repo = git.Repo.clone_from(taxonomy_repo_url, taxonomy_dir, no_checkout=True)
    repo.git.checkout(taxonomy_commit)


class TestGranularAPI(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir, testdata_path, tmp_path):
        self.test_taxonomy = taxonomy_dir
        self.testdata_path = testdata_path
        self.tmp_path = tmp_path

    def setUp(self):
        test_valid_knowledge_skill_file = self.testdata_path.joinpath(
            "test_valid_knowledge_skill.yaml"
        )
        untracked_knowledge_file = os.path.join("knowledge", "new", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file, test_valid_knowledge_skill
        )

    def file_list(self):
        return glob.glob(str(self.tmp_path.joinpath("**/*")), recursive=True)

    def test_granular_api_end_to_end(self):
        # Registry our mock block so we can reference it in pipelines
        BlockRegistry.register("MockLLMBlock")(MockLLMBlock)

        # Clone a taxonomy and edit 1 file in it
        taxonomy_dir = self.tmp_path

        pipeline_dir = self.testdata_path.joinpath("mock_pipelines")
        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )

        preprocessed_dir = self.tmp_path.joinpath("preprocessed")
        teacher_model_path = self.testdata_path.joinpath(
            "models/instructlab/granite-7b-lab"
        )
        preprocess_taxonomy(
            taxonomy_dir=taxonomy_dir,
            output_dir=preprocessed_dir,
            teacher_model_path=teacher_model_path,
        )
        docs = glob.glob(
            str(preprocessed_dir.joinpath("documents", "knowledge_new_*", "phoenix.md"))
        )
        assert docs, f"Expected docs not found in {self.file_list()}"
        samples_path = preprocessed_dir.joinpath("knowledge_new.jsonl")
        assert (
            samples_path.is_file()
        ), f"Expected samples file not found in {self.file_list()}"

        client = MagicMock()
        client.server_supports_batched = False
        generated_dir = self.tmp_path.joinpath("generated")
        generate_taxonomy(
            client=client,
            input_dir=preprocessed_dir,
            output_dir=generated_dir,
            pipeline=pipeline_dir,
            num_cpus=1,  # Test is faster running on a single CPU vs forking
            batch_size=0,  # Disable batch for tiny dataset and fastest test
        )
        generated_samples_path = generated_dir.joinpath("knowledge_new.jsonl")
        assert (
            generated_samples_path.is_file()
        ), f"Generated samples not found in {self.file_list()}"

        postprocessed_dir = self.tmp_path.joinpath("postprocessed")
        postprocess_taxonomy(
            input_dir=generated_dir,
            output_dir=postprocessed_dir,
            date_suffix=date_suffix,
            pipeline=pipeline_dir,
        )
        knowledge_recipe_file = postprocessed_dir.joinpath(
            f"knowledge_recipe_{date_suffix}.yaml"
        )
        assert (
            knowledge_recipe_file.is_file()
        ), f"Generated knowledge recipe file not found in {self.file_list()}"
        skills_recipe_file = postprocessed_dir.joinpath(
            f"skills_recipe_{date_suffix}.yaml"
        )
        assert (
            skills_recipe_file.is_file()
        ), f"Generated skills recipe file not found in {self.file_list()}"

        mixed_skills_output_file = (
            f"{postprocessed_dir}/skills_train_msgs_{date_suffix}.jsonl"
        )
        mix_datasets(
            recipe_file=f"{postprocessed_dir}/skills_recipe_{date_suffix}.yaml",
            output_file=mixed_skills_output_file,
        )
        assert pathlib.Path(
            mixed_skills_output_file
        ).is_file(), f"Generated mixed output not found in {self.file_list()}"
