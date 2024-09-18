# Synthetic Data Generation (SDG)

![Lint](https://github.com/instructlab/sdg/actions/workflows/lint.yml/badge.svg?branch=main)
![Build](https://github.com/instructlab/sdg/actions/workflows/pypi.yaml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/instructlab/sdg)
![License](https://img.shields.io/github/license/instructlab/sdg)

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

## Using the library

You can import SDG into your Python files with the following items: 

```python
 from instructlab.sdg.generate_data import generate_data
 from instructlab.sdg.utils import GenerateException
```

## Pipelines

There are four pipelines that are used in SDG. Each pipeline requires specific hardware specifications.
<!--TODO: Add explanations of pipelines-->

*Full* -

This pipeline is targeted for running SDG on consumer grade accelerators (GPUs).

*Simple* -

This pipeline is targeted for running SDG on CPUs or GPU enhanced CPUs.

### Pipeline architecture

All the pipelines are written in YAML format.

Knowledge:

Grounded Skills:

Freeform Skills:

<!--TODO: Add content here-->

## Repository structure

```bash
|-- sdg/src/instructlab/pipelines/ (1)
|-- sdg/src/instructlab/configs/ (2)
|-- sdg/src/instructlab/utils/ (3)
|-- sdg/docs/ (4)
|-- sdg/scripts/ (5)
|-- sgd/tests/ (6)
```

1. Contains the YAML code that configures the SDG pipelines
2. 
3.
4.
5.
6. Contains all the CI tests for the SDG repository