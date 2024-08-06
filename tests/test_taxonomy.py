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

TEST_TAXONOMY_BASE = "main"

TEST_CUSTOM_YAML_RULES = b"""extends: relaxed

rules:
  line-length:
    max: 180
"""

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def load_test_skills(skills_file_path) -> Union[Dict[str, Any], None]:
    with open(skills_file_path, "r", encoding="utf-8") as skills_file:
        return yaml.safe_load(skills_file)


class TestTaxonomy:
    """Test taxonomy in instructlab.sdg.utils.taxonomy."""

    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_read_taxonomy_leaf_nodes(self):
        test_compositional_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_compositional_skill.yaml"
        )
        tracked_file = "compositional_skills/tracked/qna.yaml"
        untracked_file = "compositional_skills/new/qna.yaml"
        test_compositional_skill = load_test_skills(test_compositional_skill_file)
        self.taxonomy.add_tracked(tracked_file, test_compositional_skill)
        self.taxonomy.create_untracked(untracked_file, test_compositional_skill)

        leaf_node = taxonomy.read_taxonomy_leaf_nodes(
            self.taxonomy.root, TEST_TAXONOMY_BASE, TEST_CUSTOM_YAML_RULES
        )
        leaf_node_key = str(pathlib.Path(untracked_file).parent).replace(
            os.path.sep, "->"
        )
        assert leaf_node_key in leaf_node

        leaf_node_entries = leaf_node.get(leaf_node_key)
        seed_example_exists = False
        if any(
            entry["instruction"] == TEST_SEED_EXAMPLE for entry in leaf_node_entries
        ):
            seed_example_exists = True
        assert seed_example_exists is True
