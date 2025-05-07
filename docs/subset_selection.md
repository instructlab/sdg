# Subset Selection Documentation

## Overview

Subset Selection is a feature in InstructLab - SDG that allows you to reduce the number of samples in your dataset to a representative and diverse subset. This is useful for:

- Making training times manageable in subsequent InstructLab workflow steps after SDG
- Creating benchmarking subsets that preserve original dataset characteristics
- Reducing storage and processing requirements while maintaining diversity

The feature uses a facility location method to select samples that best represent the full dataset.

## Prerequisites

Required embedding model must be downloaded before use:

```shell
ilab model download -rp Snowflake/snowflake-arctic-embed-l-v2.0
```

## Basic Usage

```shell
python -m instructlab.sdg.subset_select --input_files <input_files> --output_dir <output_dir> --subset_sizes <sizes>
```

### Example

```shell
python -m instructlab.sdg.subset_select \
    --input_files data.jsonl \
    --output_dir output/ \
    --subset_sizes 0.1 1000
```

This will:

- Process `data.jsonl` from the local directory
- Save results to the output/ directory
- Create two subsets:
  - One with 10% of the original data
  - One with 1000 samples

## Command Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input_files` | One or more input files to process (space-separated) Supports: jsonl, json, csv, parquet |
| `--output_dir` | Directory where output files will be saved |
| `--subset_sizes` | One or more subset sizes (space-separated). Percentages (0-1): e.g., 0.1 for 10%. Absolute counts: e.g., 1000 for 1000 samples |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_folds` | 50 | Number of folds for subset selection |
For small datasets (<1000), use 1-5 |
| `--batch_size` | 100000 | Batch size for processing embeddings |
| `--epsilon` | 160.0 | Parameter for the optimizer | 
For small datasets, use smaller values (start from 0.01) |
| `--num_gpus` | all available | Number of GPUs to use |
| `--encoder_type` | "arctic" | Type of encoder for generating embeddings |
| `--encoder_model` | "Snowflake/snowflake-arctic-embed-l-v2.0" | Model to use for embeddings |
| `--template_name` | "conversation" | Template for formatting examples. Options: default, conversation, qa |
| `--combine_files` | false | Combines all input files into a single dataset |
| `--log_dir` | None | Directory to store log files |
| `--log_level` | "INFO" | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--testing_mode` | false | Run in testing mode (for development only) |

## Python API

You can also use Subset Selection as a Python API:

```shell
from instructlab.sdg.subset_selection import subset_datasets

# All CLI arguments are supported as parameters
subset_datasets(
    input_files=["data.jsonl"], 
    subset_sizes=[0.1, 1000],
    output_dir="output/"
    # Any other optional parameters
)
```

## Implementation Details

- The algorithm uses a facility location method to select diverse samples
- Embeddings are generated using the specified encoder model
- For datasets >100k samples, the default parameters are optimized
- For smaller datasets, use smaller epsilon values
- Multi-GPU support provides faster processing for large datasets

## Known Limitations

- The algorithm may hang if epsilon value and batch_size are not appropriate for the input dataset size
- Uses a fixed seed for randomization
- Potential sampling bias for the first 32768 samples in datasets larger than that size
