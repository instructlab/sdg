# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for common Pipeline functionality
"""

# Standard
from contextlib import contextmanager
from threading import Event
from unittest import mock

# Third Party
from datasets import Dataset
import pytest

# First Party
from instructlab.sdg import Block, Pipeline, PipelineBlockError

## Helpers ##


@contextmanager
def block_types(block_types_dict):
    get_registry_mock = mock.MagicMock()
    get_registry_mock.return_value = block_types_dict
    with mock.patch(
        "instructlab.sdg.registry.BlockRegistry.get_registry",
        get_registry_mock,
    ):
        yield


## Pipeline Batching ##


def test_pipeline_no_batching(sample_dataset, single_threaded_ctx):
    """Test that with no batching enabled, the block is called once"""
    block_type_mock = mock.MagicMock()
    block_type_mock().generate.return_value = sample_dataset
    pipe_cfg = [
        {
            "name": "block-one",
            "type": "test",
            "config": {},
        }
    ]
    with block_types({"test": block_type_mock}):
        Pipeline(single_threaded_ctx, "", pipe_cfg).generate(sample_dataset)
    block_type_mock().generate.assert_called_once_with(sample_dataset)


def test_pipeline_with_batching(sample_dataset, threaded_ctx):
    """Test that when configured with batching enabled, the block is called
    multiple times, once for each batch
    """
    block_type_mock = mock.MagicMock()
    block_type_mock().generate.return_value = sample_dataset
    pipe_cfg = [
        {
            "name": "block-one",
            "type": "test",
            "config": {},
        }
    ]
    with block_types({"test": block_type_mock}):
        Pipeline(threaded_ctx, "", pipe_cfg).generate(sample_dataset)
    block_type_mock().generate.call_count > 1


def test_pipeline_batching_order_correct(sample_dataset, threaded_ctx):
    """Make sure that batches are recombined in the correct order"""

    class MockBlockType:
        # NOTE: This needs to be a class variable because it will be different
        #   instances of the block for each batch
        _second_half_event = Event()

        def __init__(self, *_, **__):
            pass

        def generate(self, dataset):
            # Make sure the second half is processed before the first half
            if dataset[0]["foo"] == 0:
                print("A")
                self._second_half_event.wait()
                print("B")
            else:
                print("C")
                self._second_half_event.set()
                print("D")
            return dataset.map(lambda r: {"foo": r["foo"] * 2})

    pipe_cfg = [
        {
            "name": "block-one",
            "type": "test",
            "config": {},
        }
    ]
    with block_types({"test": MockBlockType}):
        res = Pipeline(threaded_ctx, "", pipe_cfg).generate(sample_dataset)
    assert res.to_list() == [{"foo": i * 2} for i in range(10)]


def test_pipeline_batching_after_each_block(sample_dataset, threaded_ctx):
    """Test that batching occurs after each block in the pipeline."""

    class MockBlockOne:
        def __init__(self, ctx, pipeline, block_name, **block_config):
            self.ctx = ctx  # Save the context for use in generate if needed

        def generate(self, dataset):
            # Assert that the dataset entering Block 1 is properly batched
            assert (
                len(dataset) <= self.ctx.batch_size
            ), f"Dataset size {len(dataset)} entering block 1 exceeds batch size {self.ctx.batch_size}"
            # Simulate dataset explosion in Block 1

            exploded_data = []
            for _ in range(10):  # Repeat each entry 10 times
                exploded_data.extend(dataset)

            # Create a new Dataset from the exploded data
            output = Dataset.from_list(exploded_data)

            return output

    class MockBlockTwo:
        def __init__(self, ctx, pipeline, block_name, **block_config):
            self.ctx = ctx  # Save the context for use in generate if needed

        def generate(self, dataset):
            # Assert that the dataset entering Block 2 is properly batched (this will fail if batching is not done after each block)
            assert (
                len(dataset) <= self.ctx.batch_size
            ), f"Dataset size {len(dataset)} entering block 2 exceeds batch size {self.ctx.batch_size}"
            return dataset

    # Define the pipeline configuration with two blocks
    pipe_cfg = [
        {
            "name": "block-one",
            "type": "block_one",
            "config": {},
        },
        {
            "name": "block-two",
            "type": "block_two",
            "config": {},
        },
    ]

    # Patch block types to use the mock implementations
    with block_types({"block_one": MockBlockOne, "block_two": MockBlockTwo}):
        # Run the pipeline
        result = Pipeline(threaded_ctx, "", pipe_cfg).generate(sample_dataset)
    # Assertions for the final output dataset:
    # 1. Check the final dataset length is the expected value
    expected_len = (
        len(sample_dataset) * 10
    )  # Since Block 1 multiplies the dataset by 10
    assert (
        len(result) == expected_len
    ), f"Expected dataset length {expected_len}, but got {len(result)}"

    # 2. Check the dataset features: Ensure the feature structure is consistent with the input
    assert "foo" in result[0], "Feature 'foo' not found in the final dataset"


## Pipeline Error Handling ##


def test_pipeline_named_errors_match_type(single_threaded_ctx):
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
    with block_types(
        {
            "working": working_block,
            "failure": failure_block,
        },
    ):
        pipe = Pipeline(single_threaded_ctx, "", pipe_cfg)
        with pytest.raises(PipelineBlockError) as exc_ctx:
            pipe.generate(None)

        assert exc_ctx.value.__cause__ is failure_exc
        assert exc_ctx.value.exception is failure_exc
        assert exc_ctx.value.block is failure_block()


def test_pipeline_config_error_handling(single_threaded_ctx):
    """Validate that a PipelineBlockError is raised when block config is
    incorrect
    """
    pipe_cfg = [
        {"name_not_there": "I work", "type": "working", "config": {}},
        {"name": "I don't", "type": "failure", "config": {}},
    ]
    pipe = Pipeline(single_threaded_ctx, "", pipe_cfg)
    with pytest.raises(PipelineBlockError) as exc_ctx:
        pipe.generate(None)

    assert isinstance(exc_ctx.value.__cause__, KeyError)


## PipelineBlockError ##


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
