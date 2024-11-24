# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the top-level datamixing module.
"""

# Standard
from unittest.mock import patch
import os

# First Party
from instructlab.sdg.utils.datamixing import Recipe, _load_and_sample_datasets

# We mock out the actual things that use num_procs anyway, but just
# for a consistent value in the tests...
TEST_NUM_PROCS = 4
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")
TEST_RECIPE_PATH = os.path.join(TEST_DATA_DIR, "relative_path_recipe.yaml")
TEST_SAMPLES_ABS_PATH = os.path.join(TEST_DATA_DIR, "datasets/samples.jsonl")
TEST_SAMPLE_SIZE = 1.0


def _empty_recipe(self):
    return {}


def test_load_default_recipes():
    """
    Test that we can load default recipe files by pointing
    at a simple set of test recipe files under the testdata/
    directory.
    """
    test_recipe = Recipe(TEST_RECIPE_PATH)
    print(test_recipe.recipe)
    assert test_recipe.recipe["datasets"][0]["path"] == "datasets/samples.jsonl"


@patch.object(Recipe, "_load_recipe", _empty_recipe)
def test_init_with_empty_recipe_files():
    """
    Test that we can initialize a Recipe that points to a recipe
    file that does not contain one or more of our expected keys, and
    that instead of blowing up (like with a KeyError) we just 
    get an empty recipe object.
    """
    recipe = Recipe(recipe_path=TEST_RECIPE_PATH)
    assert recipe.recipe == {}


def test_load_ds_with_absolute_jsonl_path():
    """
    Test that the _load_and_sample_datasets function can load from datasets from jsonl
    files referenced with a path relative to the recipe file
    """
    dataset = _load_and_sample_datasets(TEST_SAMPLES_ABS_PATH, TEST_SAMPLE_SIZE)
    assert dataset is not None



# The tests for _add_extra_contexts_to_samples are commented out.  The method is not in the new code base.  If we wind up needing it then, we may need to bring back this test.  If not, we should delete the commented out tests.

# def test_add_extra_contexts_to_samples_with_one_sample():
#     """
#     Test _add_extra_contexts_to_samples doesn't error out when
#     given only one sample
#     """
#     samples = Dataset.from_list([_fake_context("context1")])
#     dataset = _add_extra_contexts_to_samples(samples, p=0.4)
#     assert len(dataset) == 1
#     assert "context context1" in dataset[0]["messages"][0]["content"]


# def test_add_extra_contexts_to_samples_with_two_samples_golden_path():
#     """
#     Test _add_extra_contexts_to_samples doesn't error out and adds
#     both expected contexts when given only two samples
#     """
#     samples = Dataset.from_list(
#         [
#             _fake_context("context1"),
#             _fake_context("context2"),
#         ]
#     )
#     p = 1.0  # 1.0 to test the golden/answer document path of this logic
#     num_doc_in_context = 4
#     dataset = _add_extra_contexts_to_samples(
#         samples, p=p, num_doc_in_context=num_doc_in_context
#     )
#     assert len(dataset) == 2
#     # ensure both contexts end up in both samples
#     assert "context context1" in dataset[0]["messages"][0]["content"]
#     assert "context context2" in dataset[0]["messages"][0]["content"]
#     assert "context context1" in dataset[1]["messages"][0]["content"]
#     assert "context context2" in dataset[1]["messages"][0]["content"]



# def test_add_extra_contexts_to_samples_with_six_samples_golden_path():
#     """
#     Test _add_extra_contexts_to_samples doesn't error out and adds
#     the expected number of contexts when given six samples
#     """
#     samples = Dataset.from_list(
#         [
#             _fake_context("context1"),
#             _fake_context("context2"),
#             _fake_context("context3"),
#             _fake_context("context4"),
#             _fake_context("context5"),
#             _fake_context("context6"),
#         ]
#     )
#     p = 1.0  # 1.0 to test the golden/answer document path of this logic
#     num_doc_in_context = 2
#     dataset = _add_extra_contexts_to_samples(
#         samples, p=p, num_doc_in_context=num_doc_in_context
#     )
#     assert len(dataset) == 6
#     for i, sample in enumerate(dataset):
#         sample_content = sample["messages"][0]["content"]
#         # ensure every sample contains its own context
#         assert f"context context{i+1}" in sample_content
#         # ensure we have the expected number of contexts
#         assert sample_content.count("Document:\ncontext") == num_doc_in_context


# def test_add_extra_contexts_to_samples_with_six_samples_distractor_path():
#     """
#     Test _add_extra_contexts_to_samples doesn't error out and does
#     not add the answer document as a distractor when given six samples
#     """
#     samples = Dataset.from_list(
#         [
#             _fake_context("context1"),
#             _fake_context("context2"),
#             _fake_context("context3"),
#             _fake_context("context4"),
#             _fake_context("context5"),
#             _fake_context("context6"),
#         ]
#     )
#     p = 0.0  # 0.0 to test the distractor path of this logic
#     num_doc_in_context = 2
#     dataset = _add_extra_contexts_to_samples(
#         samples, p=p, num_doc_in_context=num_doc_in_context
#     )
#     assert len(dataset) == 6
#     for i, sample in enumerate(dataset):
#         sample_content = sample["messages"][0]["content"]
#         # ensure no sample contains its own context
#         assert f"context context{i+1}" not in sample_content
#         # ensure we have the expected number of contexts
#         assert sample_content.count("Document:\ncontext") == num_doc_in_context
