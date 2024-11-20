# Synthetic Data Generation (SDG)

![Lint](https://github.com/instructlab/sdg/actions/workflows/lint.yml/badge.svg?branch=main)
![Build](https://github.com/instructlab/sdg/actions/workflows/pypi.yaml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/instructlab/sdg)
![License](https://img.shields.io/github/license/instructlab/sdg)

![`e2e-nvidia-t4-x1.yaml` on `main`](https://github.com/instructlab/sdg/actions/workflows/e2e-nvidia-t4-x1.yml/badge.svg?branch=main)
![`e2e-nvidia-l4-x1.yaml` on `main`](https://github.com/instructlab/sdg/actions/workflows/e2e-nvidia-l4-x1.yml/badge.svg?branch=main)
![`e2e-nvidia-l40s-x4.yml` on `main`](https://github.com/instructlab/sdg/actions/workflows/e2e-nvidia-l40s-x4.yml/badge.svg?branch=main)

Python library for Synthetic Data Generation

## Introduction

Synthetic Data Generation (SDG) is a process that creates an artificially generated dataset that mimics real data based on provided examples. SDG uses a YAML file containing question-and-answer pairs as input data.

## Installing the SDG library

Clone the library and navigate to the repo:

```bash
git clone https://github.com/instructlab/sdg
cd sdg
```

Install the library:

```bash
pip install .
```

### Using the library

You can import SDG into your Python files with the following items:

```python
 from instructlab.sdg.generate_data import generate_data
 from instructlab.sdg.utils import GenerateException
```

## Pipelines

A pipeline describes a series of steps to execute in-order to generate data.

There are three default pipelines shipped in SDG. These are the `simple`, `full`, and `eval` pipelines. Each pipeline requires specific hardware specifications

### Simple Pipeline

The [simple pipeline](src/instructlab/sdg/pipelines/simple) is designed to be used with [quantized Merlinite](https://huggingface.co/instructlab/merlinite-7b-lab-GGUF) as the teacher model. It exists to enable basic data generation results on lower end consumer grade hardware, such as laptops and desktops with small or no discrete GPUs.

### Full Pipeline

The [full pipeline](src/instructlab/sdg/pipelines/full) is designed to be used with [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) as the the teacher model, but has also been successfully tested with smaller models such as [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and even some quantized versions of the two above. This is the preferred data generation pipeline on higher end consumer grade hardware and on all enterprise hardware.

### Eval Pipeline

The [eval pipeline](src/instructlab/sdg/pipelines/eval) is used to generate [MMLU](https://en.wikipedia.org/wiki/MMLU) benchmark data that can be used to later evaluate a trained model on your knowledge dataset. It does not generate data for use during model training.

### Pipeline architecture

All the pipelines are written in a YAML format and must adhere to a [specific schema](src/instructlab/sdg/pipelines/schema/v1.json).

The pipelines that generate data for model training (simple and full pipelines) expect to have three different pipeline configs - one each for knowledge, grounded skills, and freeform skills. They are expected to exist in files called `knowledge.yaml`, `grounded_skills.yaml`, and `freeform_skills.yaml` respectively. For background on the difference in knowledge, grounded skills, and freeform skills, refer to the [InstructLab Taxonomy repository](https://github.com/instructlab/taxonomy).

## Repository structure

```bash
|-- src/instructlab/ (1)
|-- docs/ (2)
|-- scripts/ (3)
|-- tests/ (4)
```

1. Contains the SDG code that interacts with InstructLab.
2. Contains documentation on various SDG methodologies.
3. Contains some utility scripts, but not part of any supported API.
4. Contains all the tests for the SDG repository.
