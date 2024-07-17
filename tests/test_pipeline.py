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
from instructlab.sdg.pipeline import Pipeline, PipelineBlockError


def test_pipeline_named_errors_match_type():
    """Validate that a PipelineBlockError is raised to wrap exceptions raised
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
        with pytest.raises(PipelineBlockError) as exc_ctx:
            pipe.generate(None)

        assert exc_ctx.value.__cause__ is failure_exc
        assert exc_ctx.value.exception is failure_exc
        assert exc_ctx.value.block is failure_block()


def test_pipeline_config_error_handling():
    """Validate that a PipelineBlockError is raised when block config is
    incorrect
    """
    pipe_cfg = [
        {"name_not_there": "I work", "type": "working", "config": {}},
        {"name": "I don't", "type": "failure", "config": {}},
    ]
    pipe = Pipeline(None, None, pipe_cfg)
    with pytest.raises(PipelineBlockError) as exc_ctx:
        pipe.generate(None)

    assert isinstance(exc_ctx.value.__cause__, KeyError)


def test_block_generation_error_properties_from_block():
    """Make sure the PipelineBlockError exposes its properties and string form
    correctly when pulled from a Block instance
    """

    class TestBlock(Block):
        def generate(self, dataset: Dataset) -> Dataset:
            return dataset

    block_name = "my-block"
    block = TestBlock(None, None, block_name)
    inner_err = TypeError("Not the right type")
    gen_err = PipelineBlockError(inner_err, block=block)
    assert gen_err.block is block
    assert gen_err.exception is inner_err
    assert gen_err.block_name is block_name
    assert gen_err.block_type == TestBlock.__name__
    assert (
        str(gen_err)
        == f"{PipelineBlockError.__name__}({TestBlock.__name__}/{block_name}): {inner_err}"
    )


def test_block_generation_error_properties_from_strings():
    """Make sure the PipelineBlockError exposes its properties and string form
    correctly when pulled from strings
    """
    inner_err = TypeError("Not the right type")
    block_name = "my-block"
    block_type = "TestBlock"
    gen_err = PipelineBlockError(
        inner_err, block_name=block_name, block_type=block_type
    )
    assert gen_err.block is None
    assert gen_err.exception is inner_err
    assert gen_err.block_name is block_name
    assert gen_err.block_type == block_type
    assert (
        str(gen_err)
        == f"{PipelineBlockError.__name__}({block_type}/{block_name}): {inner_err}"
    )
