# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, Union
import os
import pathlib

# Third Party
import pytest
import yaml

# First Party
from instructlab.sdg.utils import taxonomy

TEST_SEED_EXAMPLE = "Can you help me debug this failing unit test?"

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def load_test_skills(skills_file_path) -> Union[Dict[str, Any], None]:
    with open(skills_file_path, "r", encoding="utf-8") as skills_file:
        return yaml.safe_load(skills_file)


class TestTaxonomy:
    """Test taxonomy in instructlab.sdg.utils.taxonomy."""

    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    @pytest.mark.parametrize(
        "taxonomy_base, create_tracked_file, create_untracked_file, check_leaf_node_keys",
        [
            ("main", True, True, ["compositional_skills->new"]),
            ("main", False, True, ["compositional_skills->new"]),
            ("main", True, False, []),
            ("main", False, False, []),
            ("main^", True, False, ["compositional_skills->tracked"]),
            (
                "main^",
                True,
                True,
                ["compositional_skills->new", "compositional_skills->tracked"],
            ),
            ("empty", True, False, ["compositional_skills->tracked"]),
            (
                "empty",
                True,
                True,
                ["compositional_skills->new", "compositional_skills->tracked"],
            ),
        ],
    )
    def test_read_taxonomy_leaf_nodes(
        self,
        taxonomy_base,
        create_tracked_file,
        create_untracked_file,
        check_leaf_node_keys,
    ):
        tracked_file = "compositional_skills/tracked/qna.yaml"
        untracked_file = "compositional_skills/new/qna.yaml"
        test_compositional_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_compositional_skill.yaml"
        )
        test_compositional_skill = load_test_skills(test_compositional_skill_file)
        if create_tracked_file:
            self.taxonomy.add_tracked(tracked_file, test_compositional_skill)
        if create_untracked_file:
            self.taxonomy.create_untracked(untracked_file, test_compositional_skill)

        leaf_nodes = taxonomy.read_taxonomy_leaf_nodes(
            self.taxonomy.root, taxonomy_base, None
        )
        assert len(leaf_nodes) == len(check_leaf_node_keys)

        for leaf_node_key in check_leaf_node_keys:
            assert leaf_node_key in leaf_nodes

            leaf_node_entries = leaf_nodes.get(leaf_node_key)
            seed_example_exists = False
            if any(
                entry["instruction"] == TEST_SEED_EXAMPLE for entry in leaf_node_entries
            ):
                seed_example_exists = True
            assert seed_example_exists is True
