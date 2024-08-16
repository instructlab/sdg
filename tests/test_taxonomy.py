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
        if create_tracked_file:
            self.taxonomy.add_tracked(tracked_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML)
        if create_untracked_file:
            self.taxonomy.create_untracked(
                untracked_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
            )

        leaf_nodes = taxonomy.read_taxonomy_leaf_nodes(
            self.taxonomy.root, taxonomy_base, TEST_CUSTOM_YAML_RULES
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
