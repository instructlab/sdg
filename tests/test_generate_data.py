"""
Unit tests for the top-level generate_data module.
"""

# Standard
from contextlib import contextmanager
from typing import Any, Dict, Union
from unittest.mock import MagicMock, patch
import glob
import json
import os
import re
import shutil
import tempfile
import unittest

# Third Party
from datasets import load_dataset
import pytest
import yaml

# First Party
from instructlab.sdg import LLMBlock, PipelineContext
from instructlab.sdg.generate_data import (
    _context_init,
    _gen_train_data,
    _locate_docling_models,
    _sdg_init,
    generate_data,
)
from instructlab.sdg.utils.json import jlload

# Local
from .taxonomy import load_test_skills

TEST_SYS_PROMPT = "I am, Red Hat® Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant."

TEST_TAXONOMY_BASE = "main"

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")

NUM_INSTRUCTIONS_TO_GENERATE = 10


def validate_legacy_dataset(dataset_file_name, expected_samples):
    """Test dataset in the "legacy message sample" format.

    See LegacyMessageSample in instructlab/instructlab.

      system: str
      user: str
      assistant: str

    This is what is currently used by the legacy training methods such as Linux training and MacOS training.
    """
    ds = load_dataset("json", data_files=dataset_file_name, split="train")
    features = ["system", "user", "assistant"]
    assert len(ds.features) == len(features)
    for feature in features:
        assert feature in ds.features
        assert ds.features[feature].dtype == "string"

    for idx, sample in enumerate(expected_samples):
        assert ds[idx]["system"] == TEST_SYS_PROMPT
        assert ds[idx]["user"] == sample["user"]
        assert ds[idx]["assistant"] == sample["assistant"]


def validate_messages_dataset(dataset_file_name, expected_samples):
    """Test dataset in the Hugging Face messages format

    See MessageSample in instructlab/instructlab.

      messages:
        content: str
        # one of: "user", "assistant", or "system"
        role: str
    """
    ds = load_dataset("json", data_files=dataset_file_name, split="train")
    assert len(ds.features) == 2
    assert len(ds.features["messages"]) == 1
    assert len(ds.features["messages"][0]) == 2
    assert ds.features["messages"][0]["content"].dtype == "string"
    assert ds.features["messages"][0]["role"].dtype == "string"
    assert ds.features["metadata"].dtype == "string"

    for idx, sample in enumerate(expected_samples):
        assert len(ds[idx]["messages"]) == 2
        assert ds[idx]["messages"][0]["role"] == "user"
        assert ds[idx]["messages"][0]["content"] == sample["user"]
        assert ds[idx]["messages"][1]["role"] == "assistant"
        assert ds[idx]["messages"][1]["content"] == sample["assistant"]
        assert ds[idx]["metadata"] == json.dumps({"system": TEST_SYS_PROMPT})


def validate_skill_leaf_node_dataset(dataset_file_name):
    ds = load_dataset("json", data_files=dataset_file_name, split="train")
    assert len(ds.features) == 10
    features = [
        ("task_description", "string"),
        ("seed_context", "string"),
        ("seed_question", "string"),
        ("seed_response", "string"),
        ("output", "string"),
        ("id", "string"),
        ("leaf_node_path", "string"),
        ("leaf_node_type", "string"),
        ("unmask", "bool"),
    ]
    for feature, dtype in features:
        assert feature in ds.features
        assert ds.features[feature].dtype == dtype
    assert "messages" in ds.features
    assert len(ds.features["messages"]) == 1
    assert len(ds.features["messages"][0]) == 2
    assert ds.features["messages"][0]["content"].dtype == "string"
    assert ds.features["messages"][0]["role"].dtype == "string"


def validate_phase_leaf_node_dataset(dataset_file_name):
    ds = load_dataset("json", data_files=dataset_file_name, split="train")
    assert len(ds.features) == 4
    features = [("metadata", "string"), ("id", "string"), ("unmask", "bool")]
    for feature, dtype in features:
        assert feature in ds.features
        assert ds.features[feature].dtype == dtype
    assert "messages" in ds.features
    assert len(ds.features["messages"]) == 1
    assert len(ds.features["messages"][0]) == 2
    assert ds.features["messages"][0]["content"].dtype == "string"
    assert ds.features["messages"][0]["role"].dtype == "string"


def validate_recipe(recipe_file_name, num_datasets):
    with open(recipe_file_name, encoding="utf-8") as fp:
        yaml_contents = yaml.safe_load(fp)
        assert len(yaml_contents["datasets"]) == num_datasets
        assert yaml_contents["datasets"][0]["path"].endswith(".jsonl")
        assert "sampling_size" in yaml_contents["datasets"][0]
        assert yaml_contents["metadata"]["sys_prompt"] == TEST_SYS_PROMPT


def validate_mixed_dataset(dataset_file_name):
    ds = load_dataset("json", data_files=dataset_file_name, split="train")
    assert "messages" in ds.features
    assert len(ds.features["messages"]) == 1
    assert len(ds.features["messages"][0]) == 2
    assert ds.features["messages"][0]["content"].dtype == "string"
    assert ds.features["messages"][0]["role"].dtype == "string"


def validate_lm_eval_task(lm_eval_task_file_name):
    with open(lm_eval_task_file_name, encoding="utf-8") as fp:
        yaml_contents = yaml.safe_load(fp)
        assert "task" in yaml_contents
        assert "dataset_kwargs" in yaml_contents
        assert "doc_to_text" in yaml_contents
        assert "doc_to_choice" in yaml_contents
        assert "doc_to_target" in yaml_contents


def validate_mmlubench_dataset(dataset_file_name):
    with open(dataset_file_name, encoding="utf-8") as fp:
        # FIXME: fix the mmlubench pipeline in this test
        assert fp.readlines() == []


def generate_test_samples(yaml_contents):
    """Convert questions and answers from the taxonomy format into the
    user/assistant format used by the legacy training methods such as
    Linux training and MacOS training.

    This mirrors what _gen_test_data() does.
    """
    test_samples = []
    is_knowledge = "document" in yaml_contents
    for seed_example in yaml_contents["seed_examples"]:
        if is_knowledge:
            for qna in seed_example["questions_and_answers"]:
                test_samples.append(
                    {
                        "user": qna["question"]
                        + "\n"
                        + seed_example["context"].strip(),
                        "assistant": qna["answer"].strip(),
                    }
                )

        else:
            # FIXME: handle freeform skills - no context
            test_samples.append(
                {
                    "user": seed_example["question"] + "\n" + seed_example["context"],
                    "assistant": seed_example["answer"],
                }
            )
    return test_samples


def generate_train_samples(yaml_contents):
    """Generate expected training samples in the user/assistant format
    used by the legacy training methods such as Linux training and MacOS
    training.

    Mirroring _noop_llmblock_generate() below, we generate 10 samples
    per input, and then follow _gen_train_data()'s output format.
    """

    def add_question_mark(q):
        return (q + "?") if not "?" in q else q

    train_samples = []
    is_knowledge = "document" in yaml_contents
    for seed_example in yaml_contents["seed_examples"]:
        for i in range(NUM_INSTRUCTIONS_TO_GENERATE):
            if is_knowledge:
                train_samples.append(
                    {
                        "user": seed_example["context"]
                        + f" (q{i}) "
                        + add_question_mark(
                            seed_example["questions_and_answers"][0]["question"].strip()
                        ),
                        "assistant": f"(a{i}) "
                        + seed_example["questions_and_answers"][0]["answer"].strip(),
                    }
                )
            else:
                # FIXME: handle freeform skills - no context
                train_samples.append(
                    {
                        "user": seed_example["context"]
                        + f" (q{i}) "
                        + add_question_mark(seed_example["question"]),
                        "assistant": f"(a{i}) " + seed_example["answer"],
                    }
                )
    return train_samples


def _noop_llmblock_generate(self, samples):
    """Generate mock output based on input samples.

    Simply return the seed question and response from the input sample,
    joined using '?' and with an integer discriminator.

    _get_question_hack() and _get_response_hack() is the code that later
    splits these using the '?' separator.

    Return 10 output samples per input samples, since the LLMBlock in the
    simple pipeline is configured with 'n: scaled' and we pass
    num_instructions_to_generate=10 to generate_data.
    """

    def strip_q(q):
        return q.strip().rstrip("?")

    output = []
    for sample in samples:
        for i in range(NUM_INSTRUCTIONS_TO_GENERATE):
            if "domain" in sample:  # knowledge
                output.append(
                    sample["icl_document"]
                    + f" (q{i}) "
                    + strip_q(sample["icl_query_1"])
                    + f" ? (a{i}) "
                    + sample["icl_response_1"]
                )
            else:
                output.append(
                    sample["seed_context"]
                    + f" (q{i}) "
                    + strip_q(sample["seed_question"])
                    + f" ? (a{i}) "
                    + sample["seed_response"]
                )
    return output


def _empty_llmblock_generate(self, samples):
    """Return an empty set of generated samples."""
    return []


@patch.object(LLMBlock, "_generate", _noop_llmblock_generate)
class TestGenerateCompositionalData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.tmp_path = tempfile.TemporaryDirectory().name
        test_valid_compositional_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_compositional_skill.yaml"
        )
        tracked_compositional_file = os.path.join(
            "compositional_skills", "tracked", "qna.yaml"
        )
        untracked_compositional_file = os.path.join(
            "compositional_skills", "new", "qna.yaml"
        )
        test_valid_compositional_skill = load_test_skills(
            test_valid_compositional_skill_file
        )
        self.test_taxonomy.add_tracked(
            tracked_compositional_file, test_valid_compositional_skill
        )
        self.test_taxonomy.create_untracked(
            untracked_compositional_file, test_valid_compositional_skill
        )
        self.expected_test_samples = generate_test_samples(
            test_valid_compositional_skill
        )
        self.expected_train_samples = generate_train_samples(
            test_valid_compositional_skill
        )

    def test_generate(self):
        with patch("logging.Logger.info") as mocked_logger:
            generate_data(
                client=MagicMock(),
                logger=mocked_logger,
                model_family="granite",
                model_name=os.path.join(
                    TEST_DATA_DIR, "models/instructlab/granite-7b-lab"
                ),
                num_instructions_to_generate=10,
                taxonomy=self.test_taxonomy.root,
                taxonomy_base=TEST_TAXONOMY_BASE,
                output_dir=self.tmp_path,
                pipeline="simple",
                system_prompt=TEST_SYS_PROMPT,
            )

        for name in ["test_*.jsonl", "train_*.jsonl", "messages_*.jsonl"]:
            matches = glob.glob(os.path.join(self.tmp_path, name))
            assert len(matches) == 1
            if name.startswith("test_"):
                validate_legacy_dataset(matches[0], self.expected_test_samples)
            elif name.startswith("train_"):
                validate_legacy_dataset(matches[0], self.expected_train_samples)
            elif name.startswith("messages_"):
                validate_messages_dataset(matches[0], self.expected_train_samples)

        node_file = os.path.join("node_datasets_*", "compositional_skills_new.jsonl")
        for name in [
            "skills_recipe_*.yaml",
            "skills_train_msgs_*.jsonl",
            node_file,
        ]:
            matches = glob.glob(os.path.join(self.tmp_path, name))
            assert len(matches) == 1
            if name.endswith("compositional_skills_new.jsonl"):
                validate_skill_leaf_node_dataset(matches[0])
            elif name.startswith("skills_recipe_"):
                validate_recipe(matches[0], 1)
            elif name.startswith("skills_train_msgs_"):
                validate_mixed_dataset(matches[0])

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


@patch.object(LLMBlock, "_generate", _noop_llmblock_generate)
class TestGenerateKnowledgeData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.tmp_path = tempfile.TemporaryDirectory().name
        test_valid_knowledge_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_knowledge_skill.yaml"
        )
        tracked_knowledge_file = os.path.join("knowledge  ", "tracked", "qna.yaml")
        # Explicitly add 2 files here to ensure multiple knowledge leaf nodes
        # don't conflict in anything like document_output_dir for knowledge docs
        untracked_knowledge_file1 = os.path.join("knowledge", "new1", "qna.yaml")
        untracked_knowledge_file2 = os.path.join("knowledge", "new2", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.test_taxonomy.add_tracked(
            tracked_knowledge_file, test_valid_knowledge_skill
        )
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file1, test_valid_knowledge_skill
        )
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file2, test_valid_knowledge_skill
        )
        self.expected_test_samples = generate_test_samples(test_valid_knowledge_skill)
        self.expected_train_samples = generate_train_samples(test_valid_knowledge_skill)

    def test_generate(self):
        with patch("logging.Logger.info") as mocked_logger:
            generate_data(
                client=MagicMock(),
                logger=mocked_logger,
                model_family="granite",
                model_name=os.path.join(
                    TEST_DATA_DIR, "models/instructlab/granite-7b-lab"
                ),
                num_instructions_to_generate=10,
                taxonomy=self.test_taxonomy.root,
                taxonomy_base=TEST_TAXONOMY_BASE,
                output_dir=self.tmp_path,
                chunk_word_count=1000,
                server_ctx_size=4096,
                pipeline="simple",
                system_prompt=TEST_SYS_PROMPT,
            )

        for name in ["test_*.jsonl", "train_*.jsonl", "messages_*.jsonl"]:
            matches = glob.glob(os.path.join(self.tmp_path, name))
            assert len(matches) == 1
            if name.startswith("test_"):
                validate_legacy_dataset(matches[0], self.expected_test_samples)
            elif name.startswith("train_"):
                validate_legacy_dataset(matches[0], self.expected_train_samples)
            elif name.startswith("messages_"):
                validate_messages_dataset(matches[0], self.expected_train_samples)

        node1_p07_file = os.path.join("node_datasets_*", "knowledge_new1_p07.jsonl")
        node1_p10_file = os.path.join("node_datasets_*", "knowledge_new1_p10.jsonl")
        node2_p07_file = os.path.join("node_datasets_*", "knowledge_new2_p07.jsonl")
        node2_p10_file = os.path.join("node_datasets_*", "knowledge_new2_p10.jsonl")
        for name in [
            "skills_recipe_*.yaml",
            "skills_train_*.jsonl",
            "knowledge_recipe_*.yaml",
            "knowledge_train_msgs_*.jsonl",
            node1_p07_file,
            node1_p10_file,
            node2_p07_file,
            node2_p10_file,
        ]:
            matches = glob.glob(os.path.join(self.tmp_path, name))
            assert len(matches) == 1
            if name.endswith("knowledge_new1_p07.jsonl") or name.endswith(
                "knowledge_new1_p10.jsonl"
            ):
                validate_phase_leaf_node_dataset(matches[0])
            elif name.startswith("skills_recipe_") or name.startswith(
                "knowledge_recipe_"
            ):
                validate_recipe(matches[0], 2)
            elif name.startswith("skills_train_msgs_") or name.startswith(
                "knowledge_train_msgs_"
            ):
                validate_mixed_dataset(matches[0])

        for name in [
            "knowledge_new1_task.yaml",
            "mmlubench_knowledge_new1.jsonl",
            "knowledge_new2_task.yaml",
            "mmlubench_knowledge_new2.jsonl",
        ]:
            matches = glob.glob(os.path.join(self.tmp_path, "node_datasets_*", name))
            assert len(matches) == 1
            if name == "knowledge_new1_task.yaml":
                validate_lm_eval_task(matches[0])
            elif name == "mmlubench_knowledge_new1.jsonl":
                validate_mmlubench_dataset(matches[0])

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


@patch.object(LLMBlock, "_generate", _empty_llmblock_generate)
class TestGenerateEmptyDataset(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.tmp_path = tempfile.TemporaryDirectory().name
        test_valid_knowledge_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_knowledge_skill.yaml"
        )
        tracked_knowledge_file = os.path.join("knowledge  ", "tracked", "qna.yaml")
        untracked_knowledge_file = os.path.join("knowledge", "new", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.test_taxonomy.add_tracked(
            tracked_knowledge_file, test_valid_knowledge_skill
        )
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file, test_valid_knowledge_skill
        )

    def test_generate(self):
        with patch("logging.Logger.info") as mocked_logger:
            generate_data(
                client=MagicMock(),
                logger=mocked_logger,
                model_family="granite",
                model_name=os.path.join(
                    TEST_DATA_DIR, "models/instructlab/granite-7b-lab"
                ),
                num_instructions_to_generate=10,
                taxonomy=self.test_taxonomy.root,
                taxonomy_base=TEST_TAXONOMY_BASE,
                output_dir=self.tmp_path,
                chunk_word_count=1000,
                server_ctx_size=4096,
                pipeline="simple",
                system_prompt=TEST_SYS_PROMPT,
            )
        mocked_logger.warning.assert_called()
        assert re.search(
            "empty sdg output: .+knowledge_new.jsonl",
            mocked_logger.warning.call_args.args[0],
        )

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


def test_locate_docling_models_config_found(testdata_path):
    with patch.dict(os.environ):
        os.environ["XDG_DATA_HOME"] = str(testdata_path.joinpath("mock_xdg_data_dir"))
        docling_model_path = _locate_docling_models()
        assert docling_model_path == "/mock/docling-models"


def test_locate_docling_models_config_not_found(testdata_path):
    with patch.dict(os.environ):
        os.environ["XDG_DATA_HOME"] = str(testdata_path.joinpath("nonexistent_dir"))
        docling_model_path = _locate_docling_models()
        assert docling_model_path is None


class TestGenTrainData(unittest.TestCase):
    """Test the _gen_train_data function with small synthetic examples."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.system_prompt = "Test system prompt"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_gen_train_data_with_empty_response(self):
        """Test _gen_train_data with synthetic examples with blank responses."""
        # Create mock synthetic examples with blank responses
        machine_instruction_data = [
            [
                {"question": "Q1", "response": "", "context": "C1"},
                {"question": "Q2", "response": "A2", "context": "C2"},
            ]
        ]

        output_file_train = os.path.join(self.test_dir, "train_test.jsonl")
        output_file_messages = os.path.join(self.test_dir, "messages_test.jsonl")

        # Call the function
        _gen_train_data(
            machine_instruction_data,
            output_file_train,
            output_file_messages,
            self.system_prompt,
        )

        # Verify train file was created and only has a single sample
        self.assertTrue(os.path.exists(output_file_train))
        train_data = jlload(output_file_train)
        self.assertEqual(len(train_data), 1)

        # Check first sample
        first_sample = train_data[0]
        self.assertEqual(first_sample["system"], self.system_prompt)
        self.assertEqual(first_sample["user"], "Q2\nC2")
        self.assertEqual(first_sample["assistant"], "A2")

        # Verify messages file was created and has correct content
        self.assertTrue(os.path.exists(output_file_messages))
