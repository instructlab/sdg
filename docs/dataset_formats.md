# Dataset Formats

The SDG library generates a synthetic dataset based on an input [taxonomy](https://github.com/instructlab/taxonomy) tree of skill and knowledge contributions.

The process involves transforming the input dataset in various ways and adding synthetic data to the input, producing output datasets in various formats. The goal of this document is to provide a reference for each of these input, intermediate, and output dataset formats.

## Input Format

Primarily taxonomy. But also data-mixing recipe and dataset.

### Taxonomy (Input)

This format is documented in [the taxonomy repo](https://github.com/instructlab/taxonomy) and is validated using [a JSON schema (currently version 3)](https://github.com/instructlab/schema/tree/main/src/instructlab/schema/v3).

Each taxonomy contribution is also referred to as a "leaf node" and is identified by a path such as:

```
compositional_skills->extraction->inference->qualitative->e2e-siblings
compositional_skills->extraction->answerability->e2e-yes_or_no
knowledge->tonsils->overview->e2-tonsils
```

The first element in this path is significant in that it defines whether the skills or knowledge input format is expected. Otherwise, the knowledge format is assumed if the file contains a "document" entry.

The fields in a skill contribution are:

* **created_by**: The GitHub username of the contributor.
* **task_description**: A description of the task which is used in prompts to the teacher model.
* **seed_examples**:
    - **question**: A question used for synthetic data generation.
    - **answer**: The desired response for the question.
    - **context** (optional): Information that the teacher model is expected to take into account during processing. Skills with this field set are referred to as "grounded skills", otherwise they are "freeform skills".

The fields in a knowledge contribution are:

* **created_by**: The GitHub username of the contributor.
* **domain**: The knowledge domain which is used in prompts to the teacher model.
* **document**: The knowledge documents.
    - **repo**: The URL to a Git repository holding the knowledge documents.
    - **commit**: The commit in the Git repository containing the knowledge documents.
    - **patterns**: An array of glob patterns of the knowledge documents in the Git repository.
* **document_outline**: A brief summary of the document.
* **seed_examples**:
    - **context**: Context from the document associated with this set of sample q&a pairs.
    - **questions_and_answers**: A list of 3 q&a pairs.
        + **question**: A question used for synthetic data generation.
        + **answer**: The desired response for the question.

Examples of these files used in CI are found [here](https://github.com/instructlab/instructlab/tree/main/scripts/test-data):

* [e2e-qna-freeform-skill](https://github.com/instructlab/instructlab/blob/main/scripts/test-data/e2e-qna-freeform-skill.yaml)
* [e2e-qna-grounded-skill](https://github.com/instructlab/instructlab/blob/main/scripts/test-data/e2e-qna-grounded-skill.yaml)
* [e2e-qna-knowledge](https://github.com/instructlab/instructlab/blob/main/scripts/test-data/e2e-qna-knowledge.yaml)

## Internal Formats

Only relevant to the library code and pipeline definitions.

### Seed Instruction Data (Internal)

The `read_taxonomy_leaf_nodes()` function transforms a collection of taxonomy contributions into the "seed instruction" dataset. This dataset is a dictionary, mapping each leaf node "taxonomy path" to an array of seed instructions.

The seed instruction dataset format is simple translation of the input taxonomy format. Essentially, every "seed example" in the taxonomy contribution is transformed into a seed instruction.

For skills, this is:

* **taxonomy_path**: the leaf node path, as described above.
* **task_description**: `task_description`.
* **instruction**: `seed_examples.question`.
* **output**: `seed_examples.answer`.
* **input**: `seed_examples.context`.
* **domain**: `None` - not applicable for skills.
* **document**: `None` - not applicable for skills.

For knowledge, it is:

* **taxonomy_path**: the leaf node path, as described above.
* **domain**: `domain`
* **questions_and_answers**: the `seed_examples.questions_and_answers` array.
* **context**: `seed_examples.context`.
* **document_outline**: `document_outline`.
* **document**: an array containing the full contents of the knowledge documents found in the git repository described in the taxonomy file.

### Leaf Node Samples (Internal)

Each array of seed instructions for a given leaf node is translated into a "samples" format in a Hugging Face `Dataset` data structure before being passed to the data generation pipeline.

For skills, the samples format is:

* **task_description**: `seed_instruction.task_description`.
* **seed_question**: `seed_instruction.instruction`.
* **seed_response**: `seed_instruction.output`.
* **seed_context** (optional): `seed_instruction.input`.

For knowledge, the supplied documents are split into "chunks" based on a maximum word count and the context window size of the server using [Langchain's Text Splitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/). A sample is created for each chunk, with the following fields:

* **domain**: `seed_instruction.domain`.
* **icl_document**: `seed_instruction.context`. (ICL stands for "In-Context Learning")
* **icl_query_1**, **icl_query_2**, **icl_query_3**: The `question` from `seed_instruction.questions_and_answers`.
* **icl_response_1**, **icl_response_2**, **icl_response_3**: The `answer` from `seed_instruction.questions_and_answers`.
* **document_outline**: `seed_instruction.document_outline`.
* **document**: a chunk of the documents in `seed_instruction.document`.

### Generated Samples (Internal)

The leaf node samples dataset is passed to a data generation pipeline and a new dataset is returned.

The pipeline is a sequence of "blocks", and each block can add to and transform the dataset in numerous ways:

* LLMBlock: may extends the dataset with additional samples, based on `LLMBlock.gen_kwargs.n`.
    - Adds new columns based on the `LLMBlock.output_cols` list, containing the parsed output from the teacher model
    - Add a`num_samples` column with the product of `LLMBlock.batch_kwargs.num_samples` and the total number of samples in this leaf node.
* FilterByValueBlock: drops any samples that do not pass a filter expression.
* CombineColumnsBlock: used to e.g. merge "question" and "answer" into the "question" column.
* DuplicateColumnsBlock: create a copy of an existing column with a new name, e.g. to retain a copy of the original value before modifying a column.
* Block.drop_columns: these columns are deleted after the block has completed.
* Block.drop_duplicates: a set of columns used to identify and delete duplicate samples.

It is assumed that at the end of the pipeline, each sample contains either "question" and "response" columns, or an "output" column. In the latter case, the question and response is parsed from "output" with "?" as a separator - this is because the simple workflow returns a q/a pair as a single output.

## Output Files

Stored in e.g. `$HOME/.local/share/instructlab/datasets` with a unique identifier for this data generation run based on the teacher model name and the time of execution.

### Data Checkpoint (Output)

Stored in `.jsonl` files in the `checkpoints/{leaf_node_name}/` sub-directory in files named `data_checkpoint_{uuid}.jsonl`.

The data generation pipeline is executed on "batches" or "splits" of the leaf node samples, and each generated dataset is saved as a checkpoint. The format matches the "generated samples" format above.

Before executing a pipeline, existing checkpoints are loaded and generated data samples are re-used if matches based on the input samples.

### Legacy Test Dataset (Ouput)

This is a test dataset stored in the "legacy message sample" format. Each seed example q&a pair from a taxonomy leaf node is stored as follows.

For skills:

* **system**: the hard-coded system prompt, e.g. "I am, Red Hat® Instruct Model based on Granite 7B, ...".
* **user**: `seed_examples.question`, comined with `seed_examples.context` if supplied.
* **assistant**: `seed_examples.answer`.

For knowledge:

* **system**: the hard-coded system prompt.
* **user**: `seed_examples.questions_and_answers.question`, comined with `seed_examples.context` if supplied.
* **assistant**: `seed_examples.questions_and_answers.answer`.
 
This is what is currently used by the legacy training methods such as Linux training and MacOS training.

### Legacy Training Dataset (Output)

The generated samples in the "legacy message sample" format - the "question" and "context" columns are used to construct the "user" field and the "response" column is used for the "assistant" field.

### Messages Training Dataset (Output)

The generated samples in the "messages" format. Each sample has the following fields:

* **messages**:
    - **role=user**, **content=**: the "question" and "context" columns.
    - **role=assistant**, **content=**: the "response" column.
* **metadata**:
    - **system_prompt**: the hard-coded system prompt.

### Leaf Node Dataset (Output)

For each skill leaf node, the generated samples dataset is stored at at ```node_datasets_{self.date_suffix}/{leaf_node_path}.jsonl``` including all of the the generated samples fields along with a "messages" column (as above) and an additional `id` column with a unique UUID per sample.

For each knowledge leaf node, "phase 0.7" and "phase 1.0" datasets are generated. FIXME: document these

## Data-mixing Recipes (Output)

FIXME.

## Mixed Dataset (Output)

FIXME.

## MMLU Evaluation Benchmark Dataset (Output)

FIXME.