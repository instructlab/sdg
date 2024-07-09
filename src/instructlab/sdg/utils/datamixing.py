import json

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset

from ..logger_config import setup_logger

LOGGER = setup_logger(__name__)


def load_ds(path, sampling_ratio):
    LOGGER.info(f"Loading dataset from {path} ...")
    dataset = load_dataset("json", data_files=path, split="train")
    LOGGER.info(f"Dataset columns: {dataset.column_names}")
    LOGGER.info(f"Dataset loaded with {len(dataset)} samples")

    if sampling_ratio != 1.0:
        num_samples = int(len(dataset) * sampling_ratio)
        dataset = adjust_train_sample_size(dataset, num_samples)

    # check if metadata column is string if not convert it using json.dumps
    if not isinstance(dataset["metadata"][0], str):
        dataset = dataset.map(
            lambda x: {"metadata": json.dumps(x["metadata"])}, num_proc=32
        )

    return dataset


def add_system_message(sample: dict, sys_prompt: str) -> dict:
    # check if the messages have role system
    has_system = False
    for msg in sample["messages"]:
        if msg["role"] == "system":
            has_system = True
            msg["content"] = sys_prompt

    if not has_system:
        sample["messages"].insert(0, {"role": "system", "content": sys_prompt})

    return sample


def adjust_train_sample_size(ds: Dataset, num_samples: int):
    LOGGER.info(f"Rebalancing dataset to have {num_samples} samples ...")
    df = ds.to_pandas()
    df = df.sample(n=num_samples, random_state=42, replace=True)
    ds = Dataset.from_pandas(df)
    return ds


def convert_metadata(sample):
    sample["metadata"] = json.dumps(sample["metadata"])
    return sample


def make_frequency_table(dataset, key):
    cat, cnt = np.unique(dataset[key], return_counts=True)
    df = pd.DataFrame({key: cat, "count": cnt})
    df = df.sort_values(by="count", ascending=False)
    df = df.reset_index(drop=True)
    return df


def subplots(*args, **kwargs):
    fig, ax_or_axes = plt.subplots(*args, tight_layout=True, **kwargs)
    plt.close(fig)
    return fig, ax_or_axes


def plot_counts_group(ax, dataset, key):
    df = make_frequency_table(dataset, key)
    LOGGER.info(f"Plotting {key} distribution ...")
    LOGGER.info(df)
    x, y = df[key][::-1], df["count"][::-1]
    bars = ax.barh(x, y)

    # plot the counts on top of the bars
    for b in bars:
        xval = b.get_width()
        yval = b.get_y() + b.get_height() / 2
        ax.text(xval + 0.1, yval, round(xval, 1), ha="left", va="center")

    ax.set_xlabel("count")
    ax.set_ylabel(key)


def save(ds, drop_cols, save_dir, filename):
    save_path = f"{save_dir}/{filename}"
    LOGGER.info(f"Saving datasets and plots to {save_path}")

    if drop_cols:
        drop_columns_in_ds = [e for e in drop_cols if e in ds.column_names]
        ds = ds.remove_columns(drop_columns_in_ds)

    ds.to_json(save_path, orient="records", lines=True)

    fig1, ax1 = subplots(figsize=(7, 0.3 * (len(ds.unique("group")) - 1)))
    plot_counts_group(ax1, ds, "group")

    fig2, ax2 = subplots(figsize=(7, 0.3 * (len(ds.unique("dataset")) - 1)))
    plot_counts_group(ax2, ds, "dataset")

    fig1.savefig(f"{save_dir}/group_dist.png", bbox_inches="tight")
    fig2.savefig(f"{save_dir}/dataset_dist.png", bbox_inches="tight")

    # load dataset back to make sure it loads correctly
    new_ds = load_dataset("json", data_files=save_path, split="train")
    LOGGER.info(f"Dataset loaded with {len(new_ds)} samples")
    LOGGER.info(f"Dataset columns: {new_ds.column_names}")
    LOGGER.info(f"Dataset Sample: {ds[0]}")


def get_populated_recipe(recipe_path, generated_data_path, recipe_output_path):
    with open(recipe_path, encoding="utf-8") as fp:
        recipe = yaml.safe_load(fp)

    recipe["datasets"].append({"path": generated_data_path, "sampling_ratio": 1.0})

    with open(recipe_output_path, encoding="utf-8") as fp:
        yaml.dump(recipe, fp)

    return recipe


def generate_train_test_splits(ds):
    ds = ds.train_test_split(test_size=0.1)

    return ds["train"], ds["test"]
