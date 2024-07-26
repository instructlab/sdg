# Standard
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


def _populate_checkpoints(tmpdir, dataset, checkpoints_count):
    for i in range(0, checkpoints_count):
        checkpoint_dataset = dataset.select(range(i * 10, (i + 1) * 10))
        checkpoint_dataset = checkpoint_dataset.map(
            lambda x: _add_bar(x, add_value=100)
        )
        checkpoint_dataset.to_json(
            os.path.join(tmpdir, f"data_checkpoint_abcde{i}.jsonl"),
            orient="records",
            lines=True,
        )


def _validate_checkpoints(tmpdir, expected_files_count, expected_length):
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
                assert "foo" in data and "bar" in data


@pytest.mark.parametrize(
    "save_freq, dataset_size, init_checkpoints, splits, final_checkpoints, checkpoint_length",
    [
        (1, 10, 0, 0, 1, 10),
        (1, 100, 1, 9, 10, 10),
        (1, 100, 2, 8, 10, 10),
        (3, 100, 2, 8, 5, [10, 10, 30, 30, 20]),
    ],
)
def test_checkpointing(
    tmpdir,
    save_freq,
    dataset_size,
    init_checkpoints,
    splits,
    final_checkpoints,
    checkpoint_length,
):
    # Our initial dataset
    dataset = Dataset.from_list([{"foo": i} for i in range(dataset_size)])

    # Generate and save some checkpoints to disk
    _populate_checkpoints(tmpdir, dataset, init_checkpoints)

    # Load checkpoints, giving us the remaining dataset to process and
    # the generated data loaded from the checkpoints
    checkpointer = Checkpointer(checkpoint_dir=tmpdir, save_freq=save_freq)
    dataset, pre_generated_data = checkpointer.load(dataset)

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
            checkpointer.checkpoint(split)
    else:
        dataset = dataset.map(lambda x: _add_bar(x, add_value=10))
        checkpointer.checkpoint(dataset)

    checkpointer.done()

    # Validate that all checkpoints are now saved to disk
    _validate_checkpoints(tmpdir, final_checkpoints, checkpoint_length)
