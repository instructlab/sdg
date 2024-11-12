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
    _create_phase07_ds,
    _create_phase10_ds,
)

# We mock out the actual things that use num_procs anyway, but just
# for a consistent value in the tests...
TEST_NUM_PROCS = 4
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")
TEST_RECIPE_PATH = os.path.join(TEST_DATA_DIR, "relative_path_recipe.yaml")
TEST_SAMPLES_ABS_PATH = os.path.join(TEST_DATA_DIR, "datasets/samples.jsonl")
TEST_PRETRAINING_PATH = os.path.join(TEST_DATA_DIR, "datasets/pretraining.jsonl")
TEST_PRECOMPUTED_PATH = os.path.join(TEST_DATA_DIR, "datasets/precomputed.jsonl")
TEST_KNOWLEDGE_SKILLS_PATH = os.path.join(
    TEST_DATA_DIR, "datasets/knowledge_skills.jsonl"
)
TEST_AUXILIARY_PATH = os.path.join(TEST_DATA_DIR, "datasets/auxiliary.jsonl")


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


def load_generated_dataset():
    return load_dataset("json", data_files=TEST_SAMPLES_ABS_PATH, split="train")


def load_pretraining_dataset():
    return load_dataset("json", data_files=TEST_PRETRAINING_PATH, split="train")


def load_knowledge_skills_ds():
    return load_dataset("json", data_files=TEST_KNOWLEDGE_SKILLS_PATH, split="train")


def load_precomputed_ds():
    return load_dataset("json", data_files=TEST_PRECOMPUTED_PATH, split="train")


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
        assert f"context context{i+1}" in sample_content
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
        assert f"context context{i+1}" not in sample_content
        # ensure we have the expected number of contexts
        assert sample_content.count("Document:\ncontext") == num_doc_in_context


def test_phase07_and_phase10_creation():
    """
    Test Phase 0.7 and Phase 1.0 dataset creation functions.

    Phase 0.7 should include both generated, auxiliary, and pretraining datasets.
    Phase 1.0 should include the content of Phase 0.7, along with generated and auxiliary datasets.
    """
    generated_dataset = load_generated_dataset()
    auxiliary_dataset = load_auxiliary_dataset()
    pretraining_dataset = load_pretraining_dataset()
    knowledge_skills_ds = load_knowledge_skills_ds()
    precomputed_ds = load_precomputed_ds()

    # Concatenate generated and pretraining datasets to simulate the input for phase creation
    combined_generated_and_pretraining = concatenate_datasets(
        [generated_dataset, pretraining_dataset]
    )

    # Test Phase 0.7 dataset creation
    phase07_ds = _create_phase07_ds(
        generated_dataset=combined_generated_and_pretraining,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # Check if Phase 0.7 contains generated, auxiliary, and pretraining datasets
    assert len(phase07_ds) == len(auxiliary_dataset) + len(
        pretraining_dataset
    ), "Phase 0.7 should contain generated, auxiliary, and pretraining datasets."

    # Verify that the content from all datasets is present in Phase 0.7
    auxiliary_ids = {item["id"] for item in auxiliary_dataset}
    pretraining_ids = {item["id"] for item in pretraining_dataset}
    phase07_ids = {item["id"] for item in phase07_ds}

    assert auxiliary_ids.issubset(
        phase07_ids
    ), "Phase 0.7 should include all auxiliary dataset entries."
    assert pretraining_ids.issubset(
        phase07_ids
    ), "Phase 0.7 should include all pretraining dataset entries."

    # Test Phase 1.0 dataset creation, which should include Phase 0.7, raft, and additional pretraining
    phase10_ds = _create_phase10_ds(
        generated_dataset=generated_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    pre_computed_ds_size = len(precomputed_ds)
    # Check if Phase 1.0 includes generated, auxiliary, pretraining, and content from Phase 0.7
    assert (
        len(phase10_ds)
        == len(phase10_ds) + len(knowledge_skills_ds) - pre_computed_ds_size
    ), "Phase 1.0 should contain generated, auxiliary, and pretraining datasets, including Phase 0.7 content."

    # Verify that Phase 0.7 content is included in Phase 1.0
    phase10_ids = {item["id"] for item in phase10_ds}
    assert phase07_ids.issubset(
        phase10_ids
    ), "Phase 1.0 should include all entries from Phase 0.7."

    print("All tests for Phase 0.7 and Phase 1.0 dataset creation passed.")


def test_phase07_creation():
    """
    Test Phase 0.7 dataset creation.

    Phase 0.7 should include generated, auxiliary, and pretraining datasets.
    """
    generated_dataset = load_generated_dataset()
    auxiliary_dataset = load_auxiliary_dataset()
    pretraining_dataset = load_pretraining_dataset()

    # Concatenate generated and pretraining datasets to simulate the input for phase creation
    combined_generated_and_pretraining = concatenate_datasets(
        [generated_dataset, pretraining_dataset]
    )

    # Create Phase 0.7 dataset
    phase07_ds = _create_phase07_ds(
        generated_dataset=combined_generated_and_pretraining,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # Check if Phase 0.7 contains generated, auxiliary, and pretraining datasets
    expected_phase07_size = (
        len(generated_dataset) + len(auxiliary_dataset) + len(pretraining_dataset)
    )
    assert (
        len(phase07_ds) == expected_phase07_size
    ), "Phase 0.7 should contain generated, auxiliary, and pretraining datasets."

    # Verify that the content from all datasets is present in Phase 0.7
    generated_ids = {item["id"] for item in generated_dataset}
    auxiliary_ids = {item["id"] for item in auxiliary_dataset}
    pretraining_ids = {item["id"] for item in pretraining_dataset}
    phase07_ids = {item["id"] for item in phase07_ds}

    assert generated_ids.issubset(
        phase07_ids
    ), "Phase 0.7 should include all generated dataset entries."
    assert auxiliary_ids.issubset(
        phase07_ids
    ), "Phase 0.7 should include all auxiliary dataset entries."
    assert pretraining_ids.issubset(
        phase07_ids
    ), "Phase 0.7 should include all pretraining dataset entries."


def test_phase10_creation():
    """
    Test Phase 1.0 dataset creation.

    Phase 1.0 should include the content of Phase 0.7, along with generated, auxiliary, knowledge_skills, and precomputed datasets.
    """
    generated_dataset = load_generated_dataset()
    auxiliary_dataset = load_auxiliary_dataset()
    knowledge_skills_ds = load_knowledge_skills_ds()
    precomputed_ds = load_precomputed_ds()

    # Create Phase 0.7 dataset as part of Phase 1.0 creation
    phase07_ds = _create_phase07_ds(
        generated_dataset=concatenate_datasets(
            [generated_dataset, load_pretraining_dataset()]
        ),
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # Create Phase 1.0 dataset
    phase10_ds = _create_phase10_ds(
        generated_dataset=generated_dataset,
        auxiliary_inst=auxiliary_inst,
        use_legacy_pretraining_format=False,
    )

    # Expected size calculation for Phase 1.0
    phase10_expected_size = (
        len(phase07_ds)
        + len(knowledge_skills_ds)
        + len(auxiliary_dataset)
        + len(generated_dataset)
        - len(precomputed_ds)
    )

    # Check if Phase 1.0 includes generated, auxiliary, knowledge_skills, and Phase 0.7 content
    assert (
        len(phase10_ds) == phase10_expected_size
    ), "Phase 1.0 should contain the expected number of entries, including Phase 0.7 content."

    # Verify that Phase 0.7 content is included in Phase 1.0
    phase07_ids = {item["id"] for item in phase07_ds}
    phase10_ids = {item["id"] for item in phase10_ds}
    assert phase07_ids.issubset(
        phase10_ids
    ), "Phase 1.0 should include all entries from Phase 0.7."
