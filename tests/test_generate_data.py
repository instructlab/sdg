"""
Unit tests for the top-level generate_data module.
"""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import shutil
import tempfile
import unittest

# Third Party
from datasets import Dataset
import pytest

# First Party
from instructlab.sdg.generate_data import _context_init, generate_data
from instructlab.sdg.pipeline import Pipeline, PipelineContext

# Local
from .conftest import get_single_threaded_ctx
from .taxonomy import MockTaxonomy

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


def _noop_generate(self, dataset):
    return dataset


def _noop_get_question_hack(synth_example):
    return ""


def _noop_get_response_hack(synth_example):
    return ""


def _noop_gen_train_data(
    machine_instruction_data, output_file_train, output_file_messages
):
    return


@patch.object(Pipeline, "generate", _noop_generate)
@patch("instructlab.sdg.datamixing._get_question_hack", _noop_get_question_hack)
@patch("instructlab.sdg.datamixing._get_response_hack", _noop_get_response_hack)
@patch("instructlab.sdg.generate_data._gen_train_data", _noop_gen_train_data)
class TestGenerateData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.ctx = get_single_threaded_ctx
        self.tmp_path = tempfile.TemporaryDirectory().name
        tracked_file = "compositional_skills/tracked/qna.yaml"
        untracked_file = "compositional_skills/new/qna.yaml"
        self.test_taxonomy.add_tracked(
            tracked_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
        )
        self.test_taxonomy.create_untracked(
            untracked_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
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

    def teardown(self) -> None:
        """Recursively remove the temporary repository and all of its
        subdirectories and files.
        """
        shutil.rmtree(self.tmp_path)

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
