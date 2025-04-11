# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the top-level datamixing module.
"""

# Standard
from importlib import resources
from unittest.mock import patch
import os

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset

# First Party
from instructlab.sdg.datamixing import (
    DataMixer,
    Recipe,
    _add_extra_contexts_to_samples,
    _conv_pretrain,
    _create_auxiliary_dataset,
    _create_phase07_ds,
    _create_phase10_ds,
)
from instructlab.sdg.utils.json import jldump, jlload

# We mock out the actual things that use num_procs anyway, but just
# for a consistent value in the tests...
TEST_NUM_PROCS = 4
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")
TEST_RECIPE_PATH = os.path.join(TEST_DATA_DIR, "relative_path_recipe.yaml")
TEST_SAMPLES_ABS_PATH = os.path.join(TEST_DATA_DIR, "datasets/samples.jsonl")
TEST_KNOWLEDGE_PATH = os.path.join(TEST_DATA_DIR, "datasets/knowledge.jsonl")
TEST_KNOWLEDGE_SKILLS_PATH = os.path.join(
    TEST_DATA_DIR, "datasets/knowledge_skills.jsonl"
)
TEST_AUXILIARY_PATH = os.path.join(TEST_DATA_DIR, "datasets/auxiliary.jsonl")
TEST_PRECOMPUTED_07X_PATH = os.path.join(
    TEST_DATA_DIR, "datasets/precomputed_skills_07x.jsonl"
)


auxiliary_inst = {
    "spellcheck": [
        "Correct any spelling errors in the document and output the corrected version.",
        "Rewrite the document to remove any spelling errors.",
    ]
}


def _empty_recipe(self):
    return {}


def _noop_sample(dataset, _sampling_size, _num_procs):
    return dataset


def load_knowledge_dataset():
    return load_dataset("json", data_files=TEST_KNOWLEDGE_PATH, split="train")


def load_knowledge_skills_ds():
    return load_dataset("json", data_files=TEST_KNOWLEDGE_SKILLS_PATH, split="train")


def load_auxiliary_dataset():
    return load_dataset("json", data_files=TEST_AUXILIARY_PATH, split="train")


def _fake_context(msg_id):
    return {
        "context": f"context {msg_id}",
        "id": msg_id,
        "messages": [{"role": "user", "content": f"user content {msg_id}"}],
        "metadata": '{"dataset": []}',
    }


def test_datamixer_can_load_default_recipes():
    """
    Test that DataMixer can load default recipe files by pointing
    it at a simple set of test recipe files under the testdata/
    directory.
    """
    date_suffix = "2024-07-25T15_52_10"
    prompt = "You are a useful AI assistant."
    mixer = DataMixer(
        [TEST_DATA_DIR], TEST_DATA_DIR, date_suffix, prompt, TEST_NUM_PROCS
    )
    assert mixer.knowledge_recipe.datasets[0]["path"] == "test/knowledge.jsonl"
    assert mixer.skills_recipe.datasets[0]["path"] == "test/skills.jsonl"


def test_recipe_init_with_empty_params_adds_dataset():
    """
    Test that an empty-initialized recipe can add datasets
    """
    recipe = Recipe()
    recipe.add_dataset("testdata/datasets/samples.jsonl", 1.0)
    assert recipe.dataset_added


def test_recipe_init_with_empty_params_loads_abs_dataset():
    """
    Test that an empty-initialized recipe can load datasets from
    absolute file paths.
    """
    recipe = Recipe()
    dataset = recipe._load_ds(TEST_SAMPLES_ABS_PATH)
    assert dataset is not None


def test_recipe_init_with_empty_params_loads_rel_dataset():
    """
    Test that an empty-initialized recipe looks for dataset files relative
    to the current working directory (as opposed to blowing up because of
    no recipe_path given).
    """
    recipe = Recipe()
    rel_path = os.path.relpath(TEST_SAMPLES_ABS_PATH)
    dataset = recipe._load_ds(rel_path)
    assert dataset is not None


@patch.object(Recipe, "_load_recipe", _empty_recipe)
def test_init_with_empty_recipe_files():
    """
    Test that we can initialize a Recipe that points to a recipe
    file that does not contain one or more of our expected keys, and
    that instead of blowing up (like with a KeyError) we just use sane
    defaults.
    """
    recipe = Recipe(recipe_path=TEST_RECIPE_PATH)
    assert len(recipe.datasets) == 0
    assert recipe.sys_prompt == ""


@patch("instructlab.sdg.datamixing._sample_ds", _noop_sample)
def test_load_ds_with_relative_jsonl_path():
    """
    Test that the _load_ds function can load from datasets from jsonl
    files referenced with a path relative to the recipe file
    """
    recipe = Recipe(recipe_path=TEST_RECIPE_PATH)
    dataset = recipe._load_and_sample_datasets(TEST_NUM_PROCS)
    assert dataset is not None


@patch("instructlab.sdg.datamixing._sample_ds", _noop_sample)
def test_load_ds_with_absolute_jsonl_path():
    """
    Test that the _load_ds function can load from datasets from jsonl
    files referenced with an absolute dataset path
    """
    recipe = Recipe(recipe_path=TEST_RECIPE_PATH)
    # Patch an absolute path into our recipe before loading it
    recipe.datasets[0]["path"] = TEST_SAMPLES_ABS_PATH
    dataset = recipe._load_and_sample_datasets(TEST_NUM_PROCS)
    assert dataset is not None


def test_add_extra_contexts_to_samples_with_one_sample():
    """
    Test _add_extra_contexts_to_samples doesn't error out when
    given only one sample
    """
    samples = Dataset.from_list([_fake_context("context1")])
    dataset = _add_extra_contexts_to_samples(samples, p=0.4)
    assert len(dataset) == 1
    assert "context context1" in dataset[0]["messages"][0]["content"]


def test_add_extra_contexts_to_samples_with_two_samples_golden_path():
    """
    Test _add_extra_contexts_to_samples doesn't error out and adds
    both expected contexts when given only two samples
    """
    samples = Dataset.from_list(
        [
            _fake_context("context1"),
            _fake_context("context2"),
        ]
    )
    p = 1.0  # 1.0 to test the golden/answer document path of this logic
    num_doc_in_context = 4
    dataset = _add_extra_contexts_to_samples(
        samples, p=p, num_doc_in_context=num_doc_in_context
    )
    assert len(dataset) == 2
    # ensure both contexts end up in both samples
    assert "context context1" in dataset[0]["messages"][0]["content"]
    assert "context context2" in dataset[0]["messages"][0]["content"]
    assert "context context1" in dataset[1]["messages"][0]["content"]
    assert "context context2" in dataset[1]["messages"][0]["content"]


def test_add_extra_contexts_to_samples_with_six_samples_golden_path():
    """
    Test _add_extra_contexts_to_samples doesn't error out and adds
    the expected number of contexts when given six samples
    """
    samples = Dataset.from_list(
        [
            _fake_context("context1"),
            _fake_context("context2"),
            _fake_context("context3"),
            _fake_context("context4"),
            _fake_context("context5"),
            _fake_context("context6"),
        ]
    )
    p = 1.0  # 1.0 to test the golden/answer document path of this logic
    num_doc_in_context = 2
    dataset = _add_extra_contexts_to_samples(
        samples, p=p, num_doc_in_context=num_doc_in_context
    )
    assert len(dataset) == 6
    for i, sample in enumerate(dataset):
        sample_content = sample["messages"][0]["content"]
        # ensure every sample contains its own context
        assert f"context context{i + 1}" in sample_content
        # ensure we have the expected number of contexts
        assert sample_content.count("Document:\ncontext") == num_doc_in_context


def test_add_extra_contexts_to_samples_with_six_samples_distractor_path():
    """
    Test _add_extra_contexts_to_samples doesn't error out and does
    not add the answer document as a distractor when given six samples
    """
    samples = Dataset.from_list(
        [
            _fake_context("context1"),
            _fake_context("context2"),
            _fake_context("context3"),
            _fake_context("context4"),
            _fake_context("context5"),
            _fake_context("context6"),
        ]
    )
    p = 0.0  # 0.0 to test the distractor path of this logic
    num_doc_in_context = 2
    dataset = _add_extra_contexts_to_samples(
        samples, p=p, num_doc_in_context=num_doc_in_context
    )
    assert len(dataset) == 6
    for i, sample in enumerate(dataset):
        sample_content = sample["messages"][0]["content"]
        # ensure no sample contains its own context
        assert f"context context{i + 1}" not in sample_content
        # ensure we have the expected number of contexts
        assert sample_content.count("Document:\ncontext") == num_doc_in_context


@patch("instructlab.sdg.datamixing._create_auxiliary_dataset")
def test_phase07_creation(mock_auxiliary_dataset):
    """
    Test Phase 0.7 dataset creation.

    Phase 0.7 should include knowledge and auxiliary datasets.
    """
    knowledge_dataset = load_knowledge_dataset()
    auxiliary_dataset = load_auxiliary_dataset()
    mock_auxiliary_dataset.return_value = auxiliary_dataset

    # Create Phase 0.7 dataset
    phase07_ds = _create_phase07_ds(
        generated_dataset=knowledge_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # Check if Phase 0.7 contains knowledge and auxiliary datasets
    expected_phase07_size = len(knowledge_dataset) + len(auxiliary_dataset)
    assert (
        len(phase07_ds) == expected_phase07_size
    ), "Phase 0.7 should contain knowledge and auxiliary datasets."

    # Verify that the content from all datasets is present in Phase 0.7
    auxiliary_ids = {item["id"] for item in auxiliary_dataset}
    phase07_ids = {item["id"] for item in phase07_ds}

    assert auxiliary_ids.issubset(
        phase07_ids
    ), "Phase 0.7 should include all auxiliary dataset entries."


@patch("instructlab.sdg.datamixing._create_auxiliary_dataset")
def test_phase10_creation(mock_auxiliary_dataset):
    """
    Test Phase 1.0 dataset creation.

    Phase 1.0 should include the content of Phase 0.7, along with auxiliary and knowledge_skills datasets.
    """
    knowledge_dataset = load_knowledge_dataset()
    auxiliary_dataset = load_auxiliary_dataset()
    knowledge_skills_ds = load_knowledge_skills_ds()
    mock_auxiliary_dataset.return_value = auxiliary_dataset

    # Create Phase 1.0 dataset
    phase10_ds = _create_phase10_ds(
        generated_dataset=knowledge_skills_ds,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # Expected size calculation for Phase 1.0
    phase10_expected_size = (
        len(knowledge_dataset) + len(knowledge_skills_ds) + len(auxiliary_dataset)
    )

    # Check if Phase 1.0 includes knowledge, auxiliary, and knowledge_skills content
    assert (
        len(phase10_ds) == phase10_expected_size
    ), "Phase 1.0 should contain the expected number of entries, including Phase 0.7 content."


def test_all_samples_have_unmask_field():
    """
    Test that all samples have an unmask field after mixing, regardless of
    whether they are knowledge or skills samples.
    """
    # create a knowledge dataset
    knowledge_dataset = load_knowledge_dataset()

    # create both phase07 and phase10 datasets
    phase07_ds = _create_phase07_ds(
        generated_dataset=knowledge_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    phase10_ds = _create_phase10_ds(
        generated_dataset=knowledge_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # verify every sample in both datasets has an unmask field
    for sample in phase07_ds:
        assert "unmask" in sample, "Sample missing unmask field in phase07"

    for sample in phase10_ds:
        assert "unmask" in sample, "Sample missing unmask field in phase10"


def test_phase07_knowledge_samples_have_unmask_true():
    """
    Test that all samples in phase07 knowledge dataset have unmask=True.
    This is important as phase07 is used for pretraining.
    """

    # Create a knowledge dataset
    knowledge_dataset = load_knowledge_dataset()

    # Create phase07 dataset
    phase07_ds = _create_phase07_ds(
        generated_dataset=knowledge_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # create phase10 dataset
    phase10_ds = _create_phase10_ds(
        generated_dataset=knowledge_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # verify every sample has unmask=True
    for sample in phase07_ds:
        assert sample["unmask"] is True, "Phase07 sample does not have unmask=True"

    # also verify the auxiliary dataset samples if present
    auxiliary_dataset = _create_auxiliary_dataset(knowledge_dataset, auxiliary_inst)
    if auxiliary_dataset is not None:
        auxiliary_ds = auxiliary_dataset.map(
            lambda rec: _conv_pretrain(rec, use_legacy_pretraining_format=False)
        )
        for sample in auxiliary_ds:
            assert (
                sample["unmask"] is True
            ), "Auxiliary sample does not have unmask=True"

    # verify that at least ONE sample in phase10 has unmask=True
    assert any(
        sample["unmask"] for sample in phase10_ds
    ), "No samples in phase10 have unmask=True"


def test_mix_instructlab_07x_precomputed_skills_with_unmask(tmp_path):
    """
    Test that we can mix the precomputed skills data format used in
    InstructLab 0.7.x with new data that has the unmask field.
    """

    # Create a knowledge dataset
    knowledge_dataset = load_knowledge_dataset()

    # Create phase07 dataset
    phase10_ds = _create_phase10_ds(
        generated_dataset=knowledge_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )
    phase10_path = os.path.join(tmp_path, "knowledge_p10.jsonl")
    jldump(phase10_ds, phase10_path)

    output_path = os.path.join(tmp_path, "output.jsonl")
    recipe = Recipe()
    # Add an old precomputed skills dataset in that does NOT have an unmask field
    recipe.add_dataset(TEST_PRECOMPUTED_07X_PATH, 1.0)
    # Add in our new phase10 dataset that does have unmask fields
    recipe.add_dataset(phase10_path, 1.0)
    # Mix the two datasets, ensuring nothing errors out
    recipe.save_mixed_dataset(output_path, TEST_NUM_PROCS)

    # Ensure all the mixed samples have an unmask field
    mixed_samples = load_dataset("json", data_files=output_path, split="train")
    for sample in mixed_samples:
        assert (
            sample.get("unmask", None) is not None
        ), "Mixed sample does not have unmask"
