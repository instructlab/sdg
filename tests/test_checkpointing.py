# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import patch
import json
import os

# Third Party
from datasets import Dataset
import pytest

# First Party
from instructlab.sdg.checkpointing import Checkpointer


def _add_bar(sample, add_value=100):
    sample["bar"] = sample["foo"] + add_value
    return sample


def _populate_checkpoints(tmpdir, dataset, checkpoints_count, remove_column):
    for i in range(0, checkpoints_count):
        checkpoint_dataset = dataset.select(range(i * 10, (i + 1) * 10))
        checkpoint_dataset = checkpoint_dataset.map(
            lambda x: _add_bar(x, add_value=100)
        )
        if remove_column:
            checkpoint_dataset = checkpoint_dataset.remove_columns("foo")
        checkpoint_dataset.to_json(
            os.path.join(tmpdir, f"data_checkpoint_abcde{i}.jsonl"),
            orient="records",
            lines=True,
        )


def _validate_checkpoints(tmpdir, expected_files_count, expected_length, remove_column):
    saved_files = os.listdir(tmpdir)
    assert len(saved_files) == expected_files_count
    assert all(f.startswith("data_checkpoint_") for f in saved_files)
    assert all(f.endswith(".jsonl") for f in saved_files)

    for f in saved_files:
        with open(os.path.join(tmpdir, f), "r") as f:
            l = list(f)
            if isinstance(expected_length, list):
                expected_length.remove(len(l))
            else:
                assert len(l) == expected_length
            for s in l:
                data = json.loads(s)
                if remove_column:
                    assert "foo" not in data and "bar" in data
                else:
                    assert "foo" in data and "bar" in data


@pytest.mark.parametrize(
    "save_freq, remove_column, dataset_size, init_checkpoints, splits, final_checkpoints, checkpoint_length",
    [
        (1, False, 10, 0, 0, 1, 10),
        (1, True, 10, 0, 0, 1, 10),
        (1, False, 100, 1, 9, 10, 10),
        (1, True, 100, 1, 9, 10, 10),
        (1, False, 100, 2, 8, 10, 10),
        (3, False, 100, 2, 8, 5, [10, 10, 30, 30, 20]),
    ],
)
def test_checkpointing(
    tmpdir,
    save_freq,
    remove_column,
    dataset_size,
    init_checkpoints,
    splits,
    final_checkpoints,
    checkpoint_length,
):
    # Our initial dataset
    dataset = Dataset.from_list([{"idx": i, "foo": i} for i in range(dataset_size)])

    # Generate and save some checkpoints to disk
    _populate_checkpoints(tmpdir, dataset, init_checkpoints, remove_column)

    # Load checkpoints, giving us the remaining dataset to process and
    # the generated data loaded from the checkpoints
    checkpointer = Checkpointer(checkpoint_dir=tmpdir, save_freq=save_freq)
    dataset, pre_generated_data = checkpointer.load(dataset)

    # Should be present, even if removed from the checkpoint (remove_column=True)
    assert "foo" in dataset.features

    # When testing save_freq, we will have checkpoints of different lengths
    if isinstance(checkpoint_length, list):
        checkpoints_total = sum(checkpoint_length[:init_checkpoints])
    else:
        checkpoints_total = checkpoint_length * init_checkpoints

    # Validate pre-generated data loaded from the checkpoints
    assert len(dataset) == (dataset_size - checkpoints_total)
    if init_checkpoints > 0:
        assert len(pre_generated_data) == checkpoints_total

    # Apply pipeline to the remaining dataset and save checkpoints
    if splits:
        for i in range(0, splits):
            split = dataset.select(range(i * 10, (i + 1) * 10))
            split = split.map(lambda x: _add_bar(x, add_value=100))
            if remove_column:
                split = split.remove_columns("foo")
            checkpointer.checkpoint(split)
    else:
        dataset = dataset.map(lambda x: _add_bar(x, add_value=10))
        if remove_column:
            dataset = dataset.remove_columns("foo")
        checkpointer.checkpoint(dataset)

    checkpointer.done()

    # Validate that all checkpoints are now saved to disk
    _validate_checkpoints(tmpdir, final_checkpoints, checkpoint_length, remove_column)


@patch.object(Checkpointer, "save")
def test_checkpoint_empty_dataset_doesnt_save_empty_files(mock_save):
    checkpointer = Checkpointer()
    empty_dataset = Dataset.from_list([])
    checkpointer.checkpoint(empty_dataset)
    checkpointer.done()
    mock_save.assert_not_called()


@patch.object(Checkpointer, "save")
def test_checkpoint_nonempty_dataset_saves_files(mock_save):
    checkpointer = Checkpointer()
    dataset = Dataset.from_list([{"idx": 1, "foo": "bar"}])
    checkpointer.checkpoint(dataset)
    checkpointer.done()
    mock_save.assert_called()
