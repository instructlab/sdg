"""
Unit tests for the top-level generate_data module.
"""

# Standard
from unittest import mock

# First Party
from instructlab.sdg.generate_data import _context_init
from instructlab.sdg.pipeline import PipelineContext


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
