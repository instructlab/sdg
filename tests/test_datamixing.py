"""
Unit tests for the top-level datamixing module.
"""

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg.datamixing import _add_extra_contexts_to_samples


def _fake_context(msg_id):
    return {
        "context": f"context {msg_id}",
        "id": msg_id,
        "messages": [{"role": "user", "content": f"user content {msg_id}"}],
        "metadata": '{"dataset": []}',
    }


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
