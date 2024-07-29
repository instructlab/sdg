# SPDX-License-Identifier: Apache-2.0

"""
Common fixtures and testing utilities
"""

# Standard
from unittest import mock

# Third Party
from datasets import Dataset
import pytest

# First Party
from instructlab.sdg.pipeline import PipelineContext

# Local
from .taxonomy import MockTaxonomy


def get_ctx(**kwargs) -> PipelineContext:
    kwargs.setdefault("client", mock.MagicMock())
    kwargs.setdefault("model_family", "test")
    kwargs.setdefault("model_id", "test-model")
    kwargs.setdefault("num_instructions_to_generate", 10)
    kwargs.setdefault("dataset_num_procs", 1)
    return PipelineContext(**kwargs)


def get_single_threaded_ctx(**kwargs) -> PipelineContext:
    kwargs["batch_size"] = 0
    return get_ctx(**kwargs)


def get_threaded_ctx(**kwargs) -> PipelineContext:
    kwargs["batch_size"] = 6
    kwargs["batch_num_workers"] = 2
    return get_ctx(**kwargs)


@pytest.fixture
def single_threaded_ctx() -> PipelineContext:
    return get_single_threaded_ctx()


@pytest.fixture
def sample_dataset():
    return Dataset.from_list([{"foo": i} for i in range(10)])


@pytest.fixture
def taxonomy_dir(tmp_path):
    with MockTaxonomy(tmp_path) as taxonomy:
        yield taxonomy


@pytest.fixture
def threaded_ctx() -> PipelineContext:
    return get_threaded_ctx()
