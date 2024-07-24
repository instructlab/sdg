"""
Unit tests for the top-level datamixing module.
"""

# Standard
from importlib import resources
from unittest.mock import patch
import os

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg.datamixing import DataMixer, Recipe, _add_extra_contexts_to_samples

# We mock out the actual things that use num_procs anyway, but just
# for a consistent value in the tests...
TEST_NUM_PROCS = 4
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")
TEST_RECIPE_PATH = os.path.join(TEST_DATA_DIR, "relative_path_recipe.yaml")
TEST_SAMPLES_ABS_PATH = os.path.join(TEST_DATA_DIR, "datasets/samples.jsonl")


def _empty_recipe(self):
    return {}


def _noop_sample(dataset, _sampling_size, _num_procs):
    return dataset


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
    samples = Dataset.from_list([_fake_context("abc123")])
    dataset = _add_extra_contexts_to_samples(samples, p=0.4)
    assert len(dataset) == 1


def test_add_extra_contexts_to_samples_with_two_samples():
    """
    Test _add_extra_contexts_to_samples doesn't error out when
    given only two samples
    """
    samples = Dataset.from_list(
        [
            _fake_context("abc123"),
            _fake_context("bcd234"),
        ]
    )
    dataset = _add_extra_contexts_to_samples(samples, p=0.4)
    assert len(dataset) == 2


def test_add_extra_contexts_to_samples_with_six_samples():
    """
    Test _add_extra_contexts_to_samples doesn't error out when
    given more samples
    """
    samples = Dataset.from_list(
        [
            _fake_context("s1"),
            _fake_context("s2"),
            _fake_context("s3"),
            _fake_context("s4"),
            _fake_context("s5"),
            _fake_context("s6"),
        ]
    )
    dataset = _add_extra_contexts_to_samples(samples, p=0.4)
    assert len(dataset) == 6
