"""
Common fixtures and testing utilities
"""

# Standard
from unittest import mock

# Third Party
import pytest

# First Party
from instructlab.sdg.pipeline import PipelineContext


def get_single_threaded_ctx() -> PipelineContext:
    return PipelineContext(
        client=mock.MagicMock(),
        model_family="test",
        model_id="test-model",
        num_instructions_to_generate=10,
        dataset_num_procs=1,
        batch_size=None,
    )


def get_threaded_ctx() -> PipelineContext:
    return PipelineContext(
        client=mock.MagicMock(),
        model_family="test",
        model_id="test-model",
        num_instructions_to_generate=10,
        dataset_num_procs=1,
    )


@pytest.fixture
def single_threaded_ctx() -> PipelineContext:
    return get_single_threaded_ctx()


@pytest.fixture
def threaded_ctx() -> PipelineContext:
    return get_threaded_ctx()
