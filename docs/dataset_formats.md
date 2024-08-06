# Dataset Formats

The SDG library generates a synthetic dataset based on an input [taxonomy](https://github.com/instructlab/taxonomy) tree of skill and knowledge contributions.

The process involves transforming the input dataset in various ways and adding synthetic data to the input, producing output datasets in various formats. The goal of this document is to provide a reference for each of these input, intermediate, and output dataset formats.

## Input Format

These input datasets can be provided by the user as input to the data generation pipeline.

### Taxonomy (Input)

A taxonomy contribution - also referred to a "leaf node" - is defined by a directory containing a YAML file, usually named `qna.yaml`. Each such leaf node is identified by a path such as:

```text
compositional_skills->extraction->inference->qualitative->e2e-siblings
compositional_skills->extraction->answerability->e2e-yes_or_no
knowledge->tonsils->overview->e2-tonsils
```

The first element in this path is significant in that it defines whether the skills or knowledge input format is expected. Otherwise, the knowledge format is assumed if the file contains a "document" entry.

This taxonomy YAML format is documented in [the taxonomy repo](https://github.com/instructlab/taxonomy) and is validated using [a JSON schema (currently version 3)](https://github.com/instructlab/schema/tree/main/src/instructlab/schema/v3).

The fields in a skill contribution are:

* **created_by**: The GitHub username of the contributor.
* **task_description**: A description of the task which is used in prompts to the teacher model.
* **seed_examples**:
  * **question**: A question used for synthetic data generation.
  * **answer**: The desired response for the question.
  * **context** (optional): Information that the teacher model is expected to take into account during processing. Skills with this field set are referred to as "grounded skills", otherwise they are "freeform skills".

The fields in a knowledge contribution are:

* **created_by**: The GitHub username of the contributor.
* **domain**: The knowledge domain which is used in prompts to the teacher model.
* **document**: The knowledge documents.
  * **repo**: The URL to a Git repository holding the knowledge documents.
  * **commit**: The commit in the Git repository containing the knowledge documents.
  * **patterns**: An array of glob patterns of the knowledge documents in the Git repository.
* **document_outline**: A brief summary of the document.
* **seed_examples**:
  * **context**: Context from the document associated with this set of sample q&a pairs.
  * **questions_and_answers**: A list of 3 q&a pairs.
    * **question**: A question used for synthetic data generation.
    * **answer**: The desired response for the question.

Examples of these files used in CI are found [here](https://github.com/instructlab/instructlab/tree/main/scripts/test-data):

* [e2e-qna-freeform-skill](https://github.com/instructlab/instructlab/blob/main/scripts/test-data/e2e-qna-freeform-skill.yaml)
* [e2e-qna-grounded-skill](https://github.com/instructlab/instructlab/blob/main/scripts/test-data/e2e-qna-grounded-skill.yaml)
* [e2e-qna-knowledge](https://github.com/instructlab/instructlab/blob/main/scripts/test-data/e2e-qna-knowledge.yaml)

### Pregenerated Dataset (Input)

Previously generated synthetic datasets can be mixed into the output dataset in proportions specified by a "data mixing recipe".

The expected format of the dataset itself is the "messages" format described in [Messages Training Dataset (Output)](#messages-training-dataset-output) below. See the [InstructLab Community Dataset](https://huggingface.co/datasets/instructlab/InstructLabCommunity) for an example.

For more information, see [documentation on Data Mixing](data_mixing.md).

## Internal Formats

These internal dataset formats are only relevant to the library code and pipeline definitions.

### Seed Instruction Data (Internal)

The `read_taxonomy_leaf_nodes()` function transforms a collection of taxonomy contributions into the "seed instruction" dataset. This dataset is a dictionary, mapping each leaf node "taxonomy path" to an array of seed instructions.

The seed instruction dataset format is simple translation of the input taxonomy format. Essentially, every "seed example" in the taxonomy contribution is transformed into a seed instruction.

For skills, this is:

* **taxonomy_path**: the leaf node path, as described above.
* **task_description**: `task_description`.
* **instruction**: `seed_example.question`.
* **output**: `seed_example.answer`.
* **input**: `seed_example.context`.
* **domain**: `None` - not applicable for skills.
* **document**: `None` - not applicable for skills.

For knowledge, it is:

* **taxonomy_path**: the leaf node path, as described above.
* **domain**: `domain`
* **questions_and_answers**: the `seed_example.questions_and_answers` array.
* **context**: `seed_examples.context`.
* **document_outline**: `document_outline`.
* **document**: an array containing the full (unchunked) contents of the knowledge documents found in the git repository described in the taxonomy contribution.

### Leaf Node Samples (Internal)

Each array of seed instructions for a given leaf node is translated by the `leaf_node_to_samples()` function into a "samples" format that is then converted in a Hugging Face `Dataset` data structure before being passed to the data generation pipeline.

For skills, the samples format is:

* **task_description**: `seed_instruction.task_description`.
* **seed_question**: `seed_instruction.instruction`.
* **seed_response**: `seed_instruction.output`.
* **seed_context** (optional): `seed_instruction.input`.

For knowledge, the supplied documents are split into "chunks" based on a maximum word count and the context window size of the server using [Langchain's Text Splitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/). A sample is created for each chunk, with the following fields:

* **domain**: `seed_instruction.domain`.
* **icl_document**: `seed_instruction.context`. (ICL stands for "In-Context Learning")
* **icl_query_1**, **icl_query_2**, **icl_query_3**: The `question` from each element in `seed_instruction.questions_and_answers`.
* **icl_response_1**, **icl_response_2**, **icl_response_3**: The `answer` from each element in `seed_instruction.questions_and_answers`.
* **document_outline**: `seed_instruction.document_outline`.
* **document**: a chunk of the documents in `seed_instruction.document`.

### Generated Samples (Internal)

The leaf node samples dataset is passed to a data generation pipeline and a new dataset is returned.

The pipeline is a sequence of "blocks", and each block can add to and transform samples in the dataset in numerous ways:

* **LLMBlock**: may extends the dataset with additional samples, based on `LLMBlock.gen_kwargs.n`.
  * Adds new columns based on the `LLMBlock.output_cols` list, containing the parsed output from the teacher model
  * Add a`num_samples` column with the product of `LLMBlock.batch_kwargs.num_samples` and the total number of samples in this leaf node.
* **FilterByValueBlock**: drops any samples that do not pass a filter expression.
* **CombineColumnsBlock**: used to e.g. merge "question" and "answer" into the "question" column.
* **DuplicateColumnsBlock**: create a copy of an existing column with a new name, e.g. to retain a copy of the original value before modifying a column.
* **Block.drop_columns**: these columns are deleted after the block has completed.
* **Block.drop_duplicates**: a set of columns used to identify and delete duplicate samples.

It is assumed that at the end of the pipeline, each sample contains "question" and "response" columns, in addition to the original leaf node sample columns.

In the case of the `simple` pipeline, the "question" and "answer" fields are parsed from an "output" column using "?" as a separator - this is because the simple workflow returns a q/a pair as a single output, because the smaller teacher model used cannot follow the strict formatting instructions required to return these in separate fields.

## Output Files

Stored in e.g. `$HOME/.local/share/instructlab/datasets` with a unique identifier for this data generation run based on the teacher model name and the time of execution.

### Data Checkpoint (Output)

With inference servers that supports it (e.g. vLLM), the data generation pipeline is executed in parallel on "batches" or "splits" of the leaf node samples, and each generated dataset is saved as a checkpoint.

Checkpoints are stored in `.jsonl` files in the `$HOME/.local/share/instructlab/datasets/checkpoints/{leaf_node_name}/` sub-directory in files named `data_checkpoint_{uuid}.jsonl`. The format matches the "generated samples" format above.

Before executing a pipeline, existing checkpoints are loaded and previously generated data samples are re-used if they match the input samples.

### Legacy Test Dataset (Ouput)

This is a test dataset stored in the "legacy message sample" format. Each seed example q&a pair from a taxonomy leaf node is stored as follows.

For skills:

* **system**: the hard-coded system prompt, e.g. "I am, Red Hat® Instruct Model based on Granite 7B, ...".
* **user**: `seed_example.question`, combined with `seed_example.context` if supplied.
* **assistant**: `seed_example.answer`.

For knowledge:

* **system**: the hard-coded system prompt.
* **user**: `seed_example.questions_and_answers.question`, combined with `seed_example.context` if supplied.
* **assistant**: `seed_example.questions_and_answers.answer`.

This is what is currently used by the legacy training methods such as Linux training and MacOS training.

### Legacy Training Dataset (Output)

The generated samples are converted to a training dataset in the "legacy message sample" format. Each sample has the following fields:

* **system**: the hard-coded system prompt.
* **user**: the `question` field, with the `context` field appended (if one generated).
* **assistant**: the `response` field.

### Messages Training Dataset (Output)

The generated samples are converted to a training dataset in in the "messages" format. Each sample has the following fields:

* **messages**:
  * **role=user**, **content=**: the `question` and `context` columns.
  * **role=assistant**, **content=**: the `response` column.
* **metadata**:
  * **system_prompt**: the hard-coded system prompt.

### Leaf Node Dataset (Output)

In order to facilitate [data mixing](./data_mixing.md), the generated samples for each leaf node are stored at ```node_datasets_{self.date_suffix}/{leaf_node_path}.jsonl``. These datasets are suitable for either the "phase 1" (knowledge, aka "phase 0.7" or "p07") or the "phase 2" (skills, aka phase "1.0" or "p10") training phase, and are then referenced by the knowledge and skills data mixing recipes described below.

The contents of a dataset for a skill leaf node is straightforward - all of the the generated samples fields along with a "messages" column (as above in [Messages Training Dataset (Output)](#messages-training-dataset-output)) and an additional `id` column containing a unique UUID per sample.

For knowledge leaf nodes, a separate dataset is generated for each of the knowledge (phase 1) and skills (phase 2) phases. Both datasets are based on the "messages" format described in [Messages Training Dataset (Output)](#messages-training-dataset-output), except for:

1. In the phase 1 (knowledge training) dataset, the messages array contains an additional `role=pretraining` entry with the user and assistant messages combined into a single string separated by "<|user|>" and "<|assistant|>". This dataset also the context document in the user question/instruction message.
2. In the phase 2 (skills training) dataset, the user question/instruction message includes the original context combined with context selected from random other samples. This approach is inspired by the concepts of Retrieval Augmented FineTuning (RAFT) where the additional context are referred to as "distractor documents" and they are intended to help the model learn to differentiate between relevant and irrelevant information.

Finally, for knowledge leaf nodes, "auxiliary samples" will be included in both the phase 1 and phase 2 datasets. This is where we ask the model to generate some additional data samples with a different prompt than the standard dataset, along with some extra instruction prompts that will get matched to the auxiliary generated samples and used during training. In the case of the "full" pipeline, these auxiliary samples are generated using a spellcheck on the context document.

## Data-mixing Recipes and Mixed Dataset (Output)

As leaf node datasets are generated, they are added to "data-mixing recipes" for knowledge and skills training. Principally, these recipes merely provide the locations of the leaf node datasets, but in the case of samples generated from skills leaf nodes, the recipe also contains an instruction to only use 30 samples per leaf node. The expectation is that this is enough samples to sufficiently learn a new skill while also ensuring a balance of overall mixed data when learning multiple skills at once.

A final data-mixing process generates "mixed" datasets for knowledge and skills training - named e.g. `knowledge_train_msgs_<timestamp>.jsonl` and `skills_train_msgs_<timestamp>.jsonl` - according to the data-mixing recipes.

## MMLU Evaluation Benchmark Dataset (Output)

One final per-leaf-node dataset is generated under the `node_datasets` for knowledge leaf-nodes. This is the `mmlubench_knowledge_<leaf-node-path>.jsonl` dataset, and contains multiple-choice questions (MCQ) that are generated by a teacher LLM from the synthetic samples and used to test the trained model's performance using the MMLU Evaluation Benchmark. The `knowledge_<leaf-node-path>_task.yaml` file is configuration for an `lm_eval` harness task which will used this MCQ dataset for evaluation purposes.
