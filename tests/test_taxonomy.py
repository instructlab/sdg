# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import pathlib

# Third Party
import pytest

# First Party
from instructlab.sdg.utils import taxonomy

TEST_VALID_COMPOSITIONAL_SKILL_YAML = """created_by: rafael-vasquez
version: 1
seed_examples:
- answer: "Sure thing!"
  context: "This is a valid YAML."
  question: "Can you help me debug this failing unit test?"
- answer: "answer2"
  context: "context2"
  question: "question2"
- answer: "answer3"
  context: "context3"
  question: "question3"
- answer: "answer4"
  context: "context4"
  question: "question4"
- answer: "answer5"
  context: "context5"
  question: "question5"
task_description: 'This is a task'
"""

TEST_SEED_EXAMPLE = "Can you help me debug this failing unit test?"

TEST_TAXONOMY_BASE = "main"

TEST_CUSTOM_YAML_RULES = b"""extends: relaxed

rules:
  line-length:
    max: 180
"""


class TestTaxonomy:
    """Test taxonomy in instructlab.sdg.utils.taxonomy."""

    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_read_taxonomy_leaf_nodes(self):
        tracked_file = "compositional_skills/tracked/qna.yaml"
        untracked_file = "compositional_skills/new/qna.yaml"
        self.taxonomy.add_tracked(tracked_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML)
        self.taxonomy.create_untracked(
            untracked_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
        )

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
