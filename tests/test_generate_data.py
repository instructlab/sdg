"""
Unit tests for the top-level generate_data module.
"""

# Standard
from unittest import mock

# First Party
from instructlab.sdg.generate_data import _sdg_init
from instructlab.sdg.pipeline import PipelineContext


def test_sdg_init_batch_size_optional():
    """Test that the _sdg_init function can handle a missing batch size by
    delegating to the default in PipelineContext.
    """
    sdgs = _sdg_init(
        "simple",
        None,
        "mixtral",
        "foo.bar",
        1,
        batch_size=None,
        batch_num_workers=None,
    )
    assert all(
        pipe.ctx.batch_size == PipelineContext.DEFAULT_BATCH_SIZE
        for sdg in sdgs
        for pipe in sdg.pipelines
    )


def test_sdg_init_batch_size_optional():
    """Test that the _sdg_init function can handle a passed batch size"""
    sdgs = _sdg_init(
        "simple",
        None,
        "mixtral",
        "foo.bar",
        1,
        batch_size=20,
        batch_num_workers=32,
    )
    assert all(pipe.ctx.batch_size == 20 for sdg in sdgs for pipe in sdg.pipelines)
