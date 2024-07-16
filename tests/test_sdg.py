"""
Unit tests for the SDG wrapper class
"""

# Standard
from threading import Event
from unittest import mock

# Third Party
from datasets import Dataset
import pytest

# First Party
from instructlab.sdg.sdg import SDG


@pytest.fixture
def sample_dataset():
    return Dataset.from_list([{"foo": i} for i in range(10)])


def test_sdg_no_batching(sample_dataset):
    """Test that by default SDG does not split the dataset into batches"""
    pipe_mock = mock.MagicMock()
    pipe_mock.generate.return_value = sample_dataset
    SDG([pipe_mock]).generate(sample_dataset)
    pipe_mock.generate.assert_called_once_with(sample_dataset)


def test_sdg_with_batching(sample_dataset):
    """Test that when configured, SDG splits the dataset into batches and runs
    them in parallel
    """
    pipe_mock = mock.MagicMock()
    pipe_mock.generate.return_value = sample_dataset
    SDG([pipe_mock], batch_size=6).generate(sample_dataset)
    assert pipe_mock.generate.call_count == 2


def test_sdg_batching_order_correct(sample_dataset):
    """Make sure that batches are recombined in the correct order"""

    class MockPipe:
        def __init__(self):
            self._second_half_event = Event()

        def generate(self, dataset):
            # Make sure the second half is processed before the first half
            if dataset[0]["foo"] == 0:
                # DEBUG
                print("Waiting on second half")
                self._second_half_event.wait()
                print("Done waiting on second half")
            else:
                print("Setting on second half")
                self._second_half_event.set()
                print("Done setting on second half")
            return dataset.map(lambda r: {"foo": r["foo"] * 2})

    res = SDG([MockPipe()], num_workers=2, batch_size=6).generate(sample_dataset)
    assert res.to_list() == [{"foo": i * 2} for i in range(10)]
