# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from instructlab.sdg.utils import taxonomy

# Local
from .taxonomy import TEST_VALID_COMPOSITIONAL_SKILL_YAML

TAXONOMY_BASE = "main"

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
        self.taxonomy.add_tracked(tracked_file)
        leaf_node = taxonomy.read_taxonomy_leaf_nodes(
            self.taxonomy.root, TAXONOMY_BASE, TEST_CUSTOM_YAML_RULES
        )
        assert TEST_VALID_COMPOSITIONAL_SKILL_YAML not in leaf_node
