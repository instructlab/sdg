"""
Unit tests for the top-level generate_data module.
"""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import glob
import os
import shutil
import tempfile
import unittest

# Third Party
from datasets import load_dataset
import pytest

# First Party
from instructlab.sdg.generate_data import _context_init, generate_data
from instructlab.sdg.llmblock import LLMBlock
from instructlab.sdg.pipeline import PipelineContext

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

TEST_TAXONOMY_BASE = "main"

TEST_CUSTOM_YAML_RULES = b"""extends: relaxed
rules:
  line-length:
    max: 180
"""

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _noop_llmblock_generate(self, samples):
    generated_output_list = []
    generated_output_file = os.path.join(
        TEST_DATA_DIR, "llmblock_compositional_generated.txt"
    )
    with open(generated_output_file, "r", encoding="utf-8") as listfile:
        for line in listfile:
            generated_output_list.append(line)
    return generated_output_list


@patch.object(LLMBlock, "_generate", _noop_llmblock_generate)
class TestGenerateData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.tmp_path = tempfile.TemporaryDirectory().name
        tracked_compositional_file = os.path.join(
            "compositional_skills", "tracked", "qna.yaml"
        )
        untracked_compositional_file = os.path.join(
            "compositional_skills", "new", "qna.yaml"
        )
        self.test_taxonomy.add_tracked(
            tracked_compositional_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
        )
        self.test_taxonomy.create_untracked(
            untracked_compositional_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
        )

    def test_generate(self):
        with patch("logging.Logger.warning") as mocked_logger:
            generate_data(
                mocked_logger,
                model_family="merlinite",
                model_name="models/merlinite-7b-lab-Q4_K_M.gguf",
                num_instructions_to_generate=10,
                taxonomy=self.test_taxonomy.root,
                taxonomy_base=TEST_TAXONOMY_BASE,
                output_dir=self.tmp_path,
                yaml_rules=TEST_CUSTOM_YAML_RULES,
                client=MagicMock(),
                pipeline="simple",
            )

        node_file = os.path.join("node_datasets_*", "compositional_skills_new.jsonl")
        for name in [
            "test_*.jsonl",
            "train_*.jsonl",
            "messages_*.jsonl",
            "skills_recipe_*.yaml",
            "skills_train_*.jsonl",
            node_file,
        ]:
            file_name = os.path.join(self.tmp_path, name)
            print(f"Testing that generated file ({file_name}) exists")
            files = glob.glob(file_name)
            assert len(files) == 1

    def teardown(self) -> None:
        """Recursively remove the temporary repository and all of its
        subdirectories and files.
        """
        shutil.rmtree(self.tmp_path)
        return

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()


def test_context_init_batch_size_optional():
    """Test that the _context_init function can handle a missing batch size by
    delegating to the default in PipelineContext.
    """
    ctx = _context_init(
        None,
        "mixtral",
        "foo.bar",
        1,
        "/checkpoint/dir",
        1,
        batch_size=None,
        batch_num_workers=None,
    )
    assert ctx.batch_size == PipelineContext.DEFAULT_BATCH_SIZE


def test_context_init_batch_size_optional():
    """Test that the _context_init function can handle a passed batch size"""
    ctx = _context_init(
        None,
        "mixtral",
        "foo.bar",
        1,
        "/checkpoint/dir",
        1,
        batch_size=20,
        batch_num_workers=32,
    )
    assert ctx.batch_size == 20
