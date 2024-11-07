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
from instructlab.sdg.generate_data import _context_init, generate_data
from instructlab.sdg.llmblock import LLMBlock
from instructlab.sdg.pipeline import PipelineContext

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
    assert len(ds.features) == 7
    features = [
        "task_description",
        "seed_context",
        "seed_question",
        "seed_response",
        "output",
        "id",
    ]
    for feature in features:
        assert feature in ds.features
        assert ds.features[feature].dtype == "string"
    assert "messages" in ds.features
    assert len(ds.features["messages"]) == 1
    assert len(ds.features["messages"][0]) == 2
    assert ds.features["messages"][0]["content"].dtype == "string"
    assert ds.features["messages"][0]["role"].dtype == "string"


def validate_phase_leaf_node_dataset(dataset_file_name):
    ds = load_dataset("json", data_files=dataset_file_name, split="train")
    assert len(ds.features) == 3
    features = ["metadata", "id"]
    for feature in features:
        assert feature in ds.features
        assert ds.features[feature].dtype == "string"
    assert "messages" in ds.features
    assert len(ds.features["messages"]) == 1
    assert len(ds.features["messages"][0]) == 2
    assert ds.features["messages"][0]["content"].dtype == "string"
    assert ds.features["messages"][0]["role"].dtype == "string"


def validate_recipe(recipe_file_name):
    with open(recipe_file_name, encoding="utf-8") as fp:
        yaml_contents = yaml.safe_load(fp)
        assert len(yaml_contents["datasets"]) == 1
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


def load_test_skills(skills_file_path) -> Union[Dict[str, Any], None]:
    with open(skills_file_path, "r", encoding="utf-8") as skills_file:
        return yaml.safe_load(skills_file)


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
                model_family="merlinite",
                model_name="models/merlinite-7b-lab-Q4_K_M.gguf",
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
                validate_recipe(matches[0])
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
        untracked_knowledge_file = os.path.join("knowledge", "new", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.test_taxonomy.add_tracked(
            tracked_knowledge_file, test_valid_knowledge_skill
        )
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file, test_valid_knowledge_skill
        )
        self.expected_test_samples = generate_test_samples(test_valid_knowledge_skill)
        self.expected_train_samples = generate_train_samples(test_valid_knowledge_skill)

    def test_generate(self):
        with patch("logging.Logger.info") as mocked_logger:
            generate_data(
                client=MagicMock(),
                logger=mocked_logger,
                model_family="merlinite",
                model_name="models/merlinite-7b-lab-Q4_K_M.gguf",
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

        node_p07_file = os.path.join("node_datasets_*", "knowledge_new_p07.jsonl")
        node_p10_file = os.path.join("node_datasets_*", "knowledge_new_p10.jsonl")
        for name in [
            "skills_recipe_*.yaml",
            "skills_train_*.jsonl",
            "knowledge_recipe_*.yaml",
            "knowledge_train_msgs_*.jsonl",
            node_p07_file,
            node_p10_file,
        ]:
            matches = glob.glob(os.path.join(self.tmp_path, name))
            assert len(matches) == 1
            if name.endswith("knowledge_new_p07.jsonl") or name.endswith(
                "knowledge_new_p10.jsonl"
            ):
                validate_phase_leaf_node_dataset(matches[0])
            elif name.startswith("skills_recipe_") or name.startswith(
                "knowledge_recipe_"
            ):
                validate_recipe(matches[0])
            elif name.startswith("skills_train_msgs_") or name.startswith(
                "knowledge_train_msgs_"
            ):
                validate_mixed_dataset(matches[0])

        for name in [
            "knowledge_new_task.yaml",
            "mmlubench_knowledge_new.jsonl",
        ]:
            matches = glob.glob(os.path.join(self.tmp_path, "node_datasets_*", name))
            assert len(matches) == 1
            if name == "knowledge_new_task.yaml":
                validate_lm_eval_task(matches[0])
            elif name == "mmlubench_knowledge_new.jsonl":
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
                model_family="merlinite",
                model_name="models/merlinite-7b-lab-Q4_K_M.gguf",
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
            "empty sdg output: knowledge_new", mocked_logger.warning.call_args.args[0]
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
