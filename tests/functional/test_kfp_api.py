# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, List, NamedTuple
from unittest.mock import MagicMock
import os
import unittest

# Third Party
import kfp
import pytest

# First Party
from instructlab.sdg import BlockRegistry
from instructlab.sdg.kfp.components import (
    generate_taxonomy,
    mix_taxonomy_datasets,
    postprocess_taxonomy,
    preprocess_taxonomy,
)
from instructlab.sdg.kfp.pipelines import (
    e2e_pipeline,
    full_knowledge_pipeline,
)
from instructlab.sdg.utils.json import jlload

# Local
from ..taxonomy import load_test_skills


@kfp.dsl.component()
def validate_files_in_directory(
    directory: kfp.dsl.InputPath("Directory"),
    file_patterns: List[str],
):
    # Standard
    import glob
    import os

    for file_pattern in file_patterns:
        matches = glob.glob(os.path.join(directory, file_pattern))
        assert (
            matches
        ), f"File matching pattern {file_pattern} not found in {directory}."


class TestKubeflowPipelinesAPI(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir, testdata_path, tmp_path):
        self.test_taxonomy = taxonomy_dir
        self.testdata_path = testdata_path
        self.tmp_path = tmp_path

    def setUp(self):
        kfp.local.init(runner=kfp.local.SubprocessRunner(use_venv=False))
        test_valid_knowledge_skill_file = self.testdata_path.joinpath(
            "test_valid_knowledge_skill.yaml"
        )
        untracked_knowledge_file = os.path.join("knowledge", "new", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file, test_valid_knowledge_skill
        )

    def test_kfp_single_pipeline(self):
        if "true" == "true":
            return
        results = full_knowledge_pipeline(
            dataset_path=str(self.testdata_path.joinpath("datasets", "knowledge_new_short.jsonl")),
        )
        assert results.output
        assert results.output.path
        output_samples = jlload(results.output.path)
        assert output_samples
        for sample in output_samples:
            assert sample.get("document", None)
            assert sample.get("icl_document", None)
            assert sample.get("question", None)
            assert sample.get("response", None)

    def test_kfp_e2e_pipeline(self):
        taxonomy_repo = "https://github.com/RedHatOfficial/rhelai-sample-taxonomy"
        pipeline = str(self.testdata_path.joinpath("mock_pipelines"))
        teacher_model_path = str(
            self.testdata_path.joinpath("models/instructlab/granite-7b-lab")
        )

        kfp.compiler.Compiler().compile(e2e_pipeline, package_path="e2e_pipeline.yaml")

        results = e2e_pipeline(
            taxonomy_repo=taxonomy_repo,
            pipeline=pipeline,
            teacher_model_path=teacher_model_path,
        )

        assert os.path.isdir(results.outputs["taxonomy_path"].path)
        assert os.path.isdir(results.outputs["preprocessed_path"].path)
        assert os.path.isdir(results.outputs["generated_path"].path)
        assert os.path.isdir(results.outputs["postprocessed_path"].path)
        assert os.path.isfile(results.outputs["mixed_skills"].path)
        assert os.path.isfile(results.outputs["mixed_knowledge"].path)

    def test_kfp_granular_components(self):
        taxonomy_dir = str(self.tmp_path)
        pipeline_dir = str(self.testdata_path.joinpath("mock_pipelines"))
        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )

        teacher_model_path = str(
            self.testdata_path.joinpath("models/instructlab/granite-7b-lab")
        )

        @kfp.dsl.pipeline
        def test_pipeline() -> (
            NamedTuple(
                "outputs",
                [
                    ("mixed_skills", kfp.dsl.Dataset),
                    ("mixed_knowledge", kfp.dsl.Dataset),
                ],
            )
        ):
            taxonomy_importer = kfp.dsl.importer(
                artifact_uri=taxonomy_dir,
                artifact_class=kfp.dsl.Artifact,
                reimport=True,
            )

            preprocess_task = preprocess_taxonomy(
                taxonomy_path=taxonomy_importer.output,
                teacher_model_path=teacher_model_path,
                taxonomy_base="empty",
            )
            validate_files_in_directory(
                directory=preprocess_task.output,
                file_patterns=[
                    os.path.join("documents", "knowledge_new_*", "phoenix.md"),
                    "knowledge_new.jsonl",
                ],
            )

            generate_task = generate_taxonomy(
                preprocessed_path=preprocess_task.output,
                pipeline=pipeline_dir,
                model_id="mock",
                num_cpus=1,  # Test is faster running on a single CPU vs forking
                batch_size=0,  # Disable batch for tiny dataset and fastest test
            )
            validate_files_in_directory(
                directory=generate_task.output,
                file_patterns=["knowledge_new.jsonl"],
            )

            postprocess_task = postprocess_taxonomy(
                generated_path=generate_task.output,
                date_suffix=date_suffix,
                pipeline=pipeline_dir,
            )
            validate_files_in_directory(
                directory=postprocess_task.output,
                file_patterns=[
                    f"knowledge_recipe_{date_suffix}.yaml",
                    f"skills_recipe_{date_suffix}.yaml",
                ],
            )

            mix_task = mix_taxonomy_datasets(
                postprocessed_path=postprocess_task.output,
                date_suffix=date_suffix,
            )

            outputs = namedtuple("outputs", ["mixed_skills", "mixed_knowledge"])
            return outputs(
                mix_task.outputs["mixed_skills"], mix_task.outputs["mixed_knowledge"]
            )

        results = test_pipeline()

        assert os.path.isfile(results.outputs["mixed_skills"].path)
        assert os.path.isfile(results.outputs["mixed_knowledge"].path)
