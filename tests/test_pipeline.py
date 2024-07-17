"""
Unit tests for common Pipeline functionality
"""

# Standard
from unittest import mock

# Third Party
from datasets import Dataset
import pytest

# First Party
from instructlab.sdg.block import Block
from instructlab.sdg.pipeline import BlockGenerationError, Pipeline


def test_pipeline_named_errors_match_type():
    """Validate that a BlockGenerationError is raised to wrap exceptions raised
    in a Block's generate method
    """
    mock_dataset = ["not empty"]
    working_block = mock.MagicMock()
    working_block().generate.return_value = mock_dataset
    failure_block = mock.MagicMock()
    failure_block.__name__ = "BadBlock"
    failure_exc = RuntimeError("Oh no!")
    failure_block().generate = mock.MagicMock(side_effect=failure_exc)
    pipe_cfg = [
        {"name": "I work", "type": "working", "config": {}},
        {"name": "I don't", "type": "failure", "config": {}},
    ]
    with mock.patch(
        "instructlab.sdg.pipeline._block_types",
        {
            "working": working_block,
            "failure": failure_block,
        },
    ):
        pipe = Pipeline(None, None, pipe_cfg)
        with pytest.raises(BlockGenerationError) as exc_ctx:
            pipe.generate(None)

        assert exc_ctx.value.__cause__ is failure_exc
        assert exc_ctx.value.exception is failure_exc
        assert exc_ctx.value.block is failure_block()


def test_block_generation_error_properties():
    """Make sure the BlockGenerationError exposes its properties and string form
    correctly
    """

    class TestBlock(Block):
        def generate(self, dataset: Dataset) -> Dataset:
            return dataset

    block_name = "my-block"
    block = TestBlock(None, None, block_name)
    inner_err = TypeError("Not the right type")
    gen_err = BlockGenerationError(block, inner_err)
    assert gen_err.block is block
    assert gen_err.exception is inner_err
    assert gen_err.block_name is block_name
    assert gen_err.block_type == TestBlock.__name__
    assert (
        str(gen_err)
        == f"{BlockGenerationError.__name__}({TestBlock.__name__}/{block_name}): {inner_err}"
    )
