# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict, List, Optional
import json
import logging
import os.path
import random
import uuid

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset
import yaml

# First Party
from instructlab.sdg.utils import GenerateException, pandas
from instructlab.sdg.utils.pandas import dataset_from_pandas_dataframe

ALLOWED_COLS = ["id", "messages", "metadata"]
logger = logging.getLogger(__name__)


def _adjust_train_sample_size(ds: Dataset, num_samples: int):
    """
    Return a dataset with num_samples random samples selected from the
    original dataset.
    """
    logger.info(f"Rebalancing dataset to have {num_samples} samples ...")
    df = ds.to_pandas()
    df = df.sample(n=num_samples, random_state=42, replace=True)
    return pandas.dataset_from_pandas_dataframe(df)


def _sample_ds(dataset, sampling_size, num_proc):
    """
    Select sampling_size number/ratio of samples from a dataset, ensuring
    the returned dataset has only ALLOWED_COLS columns in it with any
    additional columns moved to the metadata section.
    """
    if sampling_size != 1.0:
        if isinstance(sampling_size, int):
            num_samples = sampling_size
        else:
            num_samples = int(len(dataset) * sampling_size)
        dataset = _adjust_train_sample_size(dataset, num_samples)

    # move any column that is not in ALLOWED_COLS to metadata
    def _move_unallowed_cols_to_metadata(example):
        metadata = example.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        for col in dataset.column_names:
            if col not in ALLOWED_COLS:
                metadata[col] = example[col]
                example.pop(col)
        example["metadata"] = json.dumps(metadata)
        return example

    dataset = dataset.map(_move_unallowed_cols_to_metadata, num_proc=num_proc)

    # check if metadata column is string if not convert it using json.dumps
    if not isinstance(dataset["metadata"][0], str):
        dataset = dataset.map(
            lambda x: {"metadata": json.dumps(x["metadata"])}, num_proc=num_proc
        )

    return dataset


def _add_system_message(sample: dict, sys_prompt: str) -> dict:
    """
    Ensure every sample has a system message with the correct system prompt
    """
    # check if the messages have role system
    has_system = False
    for msg in sample["messages"]:
        if msg["role"] == "system":
            has_system = True
            msg["content"] = sys_prompt

    if not has_system:
        sample["messages"].insert(0, {"role": "system", "content": sys_prompt})

    return sample


class Recipe:
    """
    A Recipe describes how datasets were mixed, including the path and
    sampling size used for each included dataset as well as the system
    prompt used when generating the data in those datasets.
    """

    def __init__(
        self, recipe_path: Optional[str] = None, sys_prompt: Optional[str] = ""
    ):
        self.recipe_path = recipe_path or ""
        self.sys_prompt = sys_prompt

        # Defaults if no recipe path given or these values don't
        # exist in the given recipe file
        self.datasets = []
        if recipe_path is not None:
            recipe = self._load_recipe()
            if "datasets" in recipe:
                self.datasets = recipe["datasets"]

        self.dataset_added = False

    def _load_recipe(self):
        with open(self.recipe_path, encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    def _load_ds(self, path):
        """
        Load a dataset from the given location. If a jsonl file is
        given, we load the dataset from disk. Otherwise, we load the
        path given from HuggingFace. Relative paths are resolved
        respective to the directory the recipe yaml itself resides in.
        """
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(self.recipe_path), path)
        logger.info(f"Loading dataset from {path} ...")
        dataset = load_dataset("json", data_files=path, split="train")
        logger.info(f"Dataset columns: {dataset.column_names}")
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        return dataset

    def _load_and_sample_datasets(self, num_proc):
        """
        Load and sample all the datasets in this recipe, taking
        into account the desired sampling size from each individual
        dataset to control the overall mix of samples in the final
        dataset.
        """
        return [
            _sample_ds(
                self._load_ds(dataset["path"]), dataset["sampling_size"], num_proc
            )
            for dataset in self.datasets
        ]

    def _create_mixed_dataset(self, num_proc):
        """
        Create the final mixed dataset by loading, sampling, and
        concatenating all datasets in this recipe
        """
        if not self.dataset_added:
            logger.error("No dataset added to the recipe")

        mixed_ds = self._load_and_sample_datasets(num_proc)
        mixed_ds = concatenate_datasets(mixed_ds)
        mixed_ds = mixed_ds.map(
            _add_system_message,
            fn_kwargs={"sys_prompt": self.sys_prompt},
            num_proc=num_proc,
        )

        # assert that the dataset only has the allowed columns
        assert set(mixed_ds.column_names) == set(
            ALLOWED_COLS
        ), "Dataset has invalid columns"
        return mixed_ds

    def add_dataset(self, path, sampling_size):
        """
        Add a dataset to this recipe.

        Args:
            path: The file path to the dataset's samples, as jsonl
            sampling_size: An int or float that specifies the number of
                           samples (if int) or the ratio of samples (if
                           float) to include in the mixed dataset. A value
                           of 1.0 means include all samples, 0.5 means half
                           of the samples, and so on.
        """
        self.dataset_added = True
        self.datasets.append({"path": path, "sampling_size": sampling_size})

    def save_recipe(self, output_path):
        recipe = {
            "datasets": self.datasets,
            "metadata": {"sys_prompt": self.sys_prompt},
        }
        with open(output_path, "w", encoding="utf-8") as fp:
            yaml.dump(recipe, fp)
        # Update this instance's recipe_path to reflect the path we
        # just saved it to so that any subsequent loading of datasets
        # (like via save_mixed_dataset) pulls from relative to the
        # saved recipe_path.
        self.recipe_path = output_path

    def save_mixed_dataset(self, output_path, num_proc):
        """
        Create the mixed dataset and write it to the specified output path
        as a jsonl file.
        """
        mixed_ds = self._create_mixed_dataset(num_proc)
        mixed_ds.to_json(output_path, orient="records", lines=True)
        logger.info(f"Mixed Dataset saved to {output_path}")


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question_hack(synth_example):
    if "question" in synth_example:
        return synth_example["question"]

    if not synth_example.get("output"):
        raise GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[0].strip() + "?" if len(parts) == 2 else ""


# This is also a hack. See the comment above _get_question.
def _get_response_hack(synth_example):
    if "response" in synth_example:
        return synth_example["response"]

    if "output" not in synth_example:
        raise GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _generate_knowledge_qa_dataset(
    generated_dataset: Dataset, keep_context_separate=False
):
    """
    Generate question and answer pairs from the newly generated dataset
    for each taxonomy leaf node. Each row of the generated dataset gets
    converted to have messages, metadata, and id columns.

    If `keep_context_separate` is True, then a context column is also added.
    If `keep_context_separate` is False, the context colum is omitted and
    the context is instead added directly to the user message content.
    """

    def __create_qa_row(rec):
        msg_id = str(uuid.uuid4())
        context = rec["document"]
        instruction = _get_question_hack(rec)
        response = _get_response_hack(rec)
        metadata = {
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": "document_knowledge_qa",
        }
        if "raw_document" in rec and "dataset_type" in rec:
            metadata.update(
                {
                    "raw_document": rec["raw_document"],
                    "dataset_type": rec["dataset_type"],
                }
            )
        metadata = json.dumps(metadata)
        if keep_context_separate:
            messages = [
                {"role": "user", "content": f"{instruction}"},
                {"role": "assistant", "content": response},
            ]
            return {
                "messages": messages,
                "metadata": metadata,
                "id": msg_id,
                "context": context,
            }
        messages = [
            {"role": "user", "content": f"{context}\n\n{instruction}"},
            {"role": "assistant", "content": response},
        ]

        return {"messages": messages, "metadata": metadata, "id": msg_id}

    knowledge_ds = generated_dataset.map(
        __create_qa_row, remove_columns=generated_dataset.column_names
    )
    return knowledge_ds


def _add_extra_contexts_to_samples(ds: Dataset, p, num_doc_in_context=4):
    """
    Add additional context to each sample in a knowledge_qa_dataset by
    selecting the context from random other samples and adding that
    combined with this sample's original context all into the user content
    section of the sample's messages.

    This expects to be called with a dataset that has a `context` column,
    such as the output from _generate_knowledge_qa_dataset with the param
    `keep_context_separate` equal to True. When this finishes, the `context`
    column is removed from the dataset and all context moved to the user
    messages.

    This is inspired by the concepts of Retrieval Augmented FineTuning (RAFT)
    from https://arxiv.org/abs/2403.10131
    """
    all_context = list(set(ds["context"]))

    def __pick_documents(rec, p):
        answer_document = rec["context"]
        selected_docs = [e for e in all_context if e != answer_document]
        if len(selected_docs) > 0:
            if len(selected_docs) < num_doc_in_context:
                logger.debug(
                    f"Number of unique documents is {len(selected_docs)} which is less than {num_doc_in_context}. Using all the documents in the expanded context."
                )
            if random.uniform(0, 1) < p:
                # golden/answer + distractor documents
                docs = (
                    random.sample(selected_docs, k=num_doc_in_context - 1)
                    + [answer_document]
                    if len(selected_docs) >= (num_doc_in_context - 1)
                    else selected_docs + [answer_document]
                )
            else:
                # distractor documents
                docs = (
                    random.sample(selected_docs, k=num_doc_in_context)
                    if len(selected_docs) >= num_doc_in_context
                    else selected_docs
                )
        else:
            logger.warning(
                "Only 1 unique document found. Disabling expanded context injection, which may lead to poorer knowledge retention results."
            )
            docs = [answer_document]
        random.shuffle(docs)
        docs = "\n".join(([f"Document:\n{e}\n\n" for idx, e in enumerate(docs)]))
        user_idx_msgs = [
            (idx, rec_msg)
            for idx, rec_msg in enumerate(rec["messages"])
            if rec_msg["role"] == "user"
        ]
        assert len(user_idx_msgs) > 0, "No user role found in dataset"
        user_idx, user_msg = user_idx_msgs[0]
        user_inst = user_msg["content"]
        rec["messages"][user_idx]["content"] = f"{docs}\n\n{user_inst}"
        rec["messages"] = rec["messages"]
        metadata = json.loads(rec["metadata"])
        metadata["dataset"] += f"_raft_p{p}"
        rec["metadata"] = json.dumps(metadata)
        return rec

    ds = ds.map(__pick_documents, fn_kwargs={"p": p}, remove_columns=["context"])
    return ds


def _conv_pretrain(rec, use_legacy_pretraining_format: bool):
    """
    Convert a messages dataset that contains only user/assistant entries per
    message (and in that order) to a pretraining message used downstream by
    the training pipeline. `_generate_knowledge_qa_dataset` creates the type
    of dataset expected here.
    """
    if use_legacy_pretraining_format:
        user = "<|user|>"
        assistant = "<|assistant|>"
    else:
        user = "<|start_of_role|>user<|end_of_role|>"
        assistant = "<|start_of_role|>assistant<|end_of_role|>"

    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"{user}\n{rec['messages'][0]['content']}\n{assistant}\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def _create_auxiliary_dataset(
    generated_dataset: Dataset, auxiliary_inst: Optional[Dict[str, List[str]]]
):
    # Samples that went through the auxiliary generation pipeline will
    # have a dataset_type column created by that pipeline. If that's
    # not present, then we may be running in a pipeline without any
    # auxiliary dataset generation enabled.
    if "dataset_type" not in generated_dataset.column_names:
        return None
    # If we didn't find any auxiliary instructions to load, then
    # that's also another sign that we're not running with any
    # auxiliary datasets enabled.
    if auxiliary_inst is None:
        return None
    # This "base_document" dataset_type is set in the knowledge
    # pipeline config, and represents samples that do not have the
    # auxiliary generated document attached, so we filter those out.
    auxiliary_ds = generated_dataset.filter(
        lambda x: x["dataset_type"] != "base_document"
    )
    unique_document_auxiliary = auxiliary_ds.to_pandas().drop_duplicates(
        subset=["document"]
    )
    unique_document_auxiliary = dataset_from_pandas_dataframe(unique_document_auxiliary)
    unique_document_auxiliary = unique_document_auxiliary.select_columns(
        [
            "raw_document",
            "document_outline",
            "domain",
            "dataset_type",
            "document",
        ]
    )
    unique_document_auxiliary = unique_document_auxiliary.rename_columns(
        {"raw_document": "context", "document": "response"}
    )

    def __create_auxiliary_ds(rec):
        instruction = random.choice(auxiliary_inst[rec["dataset_type"]])
        messages = [
            {"role": "user", "content": f"{rec['context']}\n\n{instruction}"},
            {"role": "assistant", "content": rec["response"]},
        ]
        metadata = json.dumps(
            {
                "dataset_type": rec["dataset_type"],
                "raw_document": rec["context"],
                "dataset": f"document_{rec['dataset_type']}",
                "domain": rec["domain"],
            }
        )
        return {"messages": messages, "metadata": metadata, "id": str(uuid.uuid4())}

    unique_document_auxiliary = unique_document_auxiliary.map(
        __create_auxiliary_ds, remove_columns=unique_document_auxiliary.column_names
    )
    return unique_document_auxiliary


def _create_phase10_ds(
    generated_dataset: Dataset,
    auxiliary_inst: Optional[Dict[str, List[str]]],
    use_legacy_pretraining_format: bool,
):
    """
    Create a dataset for Phase 1.0 of downstream training.

    This dataset is in our messages format, with each sample having
    additional context mixed in from other samples to improve the
    training outcomes.
    """
    knowledge_ds = _generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=True
    )
    raft_knowledge_ds = _add_extra_contexts_to_samples(knowledge_ds, p=0.4)
    # Include phase07
    pretraining_knowledge_ds = _generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=False
    ).map(lambda rec: _conv_pretrain(rec, use_legacy_pretraining_format))

    auxiliary_dataset = _create_auxiliary_dataset(generated_dataset, auxiliary_inst)

    if auxiliary_dataset is not None:
        phase10 = concatenate_datasets(
            [raft_knowledge_ds, pretraining_knowledge_ds, auxiliary_dataset]
        )
    else:
        phase10 = concatenate_datasets([raft_knowledge_ds, pretraining_knowledge_ds])
    return phase10


def _create_phase07_ds(
    generated_dataset: Dataset,
    auxiliary_inst: Optional[Dict[str, List[str]]],
    use_legacy_pretraining_format: bool,
):
    """
    Create a dataset for Phase 0.7 of downstream training.

    Phase 0.7 is a pretraining phase, and this dataset contains messages
    with a special `pretraining` role used by downstream training before
    running the full training with the Phase 1.0 dataset.
    """
    # Phase 0.7
    knowledge_ds = _generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=False
    )
    knowledge_ds = knowledge_ds.map(
        lambda rec: _conv_pretrain(rec, use_legacy_pretraining_format)
    )

    auxiliary_dataset = _create_auxiliary_dataset(generated_dataset, auxiliary_inst)
    if auxiliary_dataset is not None:
        auxiliary_dataset = auxiliary_dataset.map(
            lambda rec: _conv_pretrain(rec, use_legacy_pretraining_format)
        )
        phase07 = concatenate_datasets([knowledge_ds, auxiliary_dataset])
    else:
        phase07 = knowledge_ds
    return phase07


def _convert_to_leaf_node_messages(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain a 'messages' column required
    for training.

    Note that this is for the new messages format, introduced with data
    mixing.
    """
    user_query = _unescape(_get_question_hack(sample))
    response = _unescape(_get_response_hack(sample))

    sample["id"] = str(uuid.uuid4())
    sample["messages"] = [
        {"content": sys_prompt, "role": "system"},
        {"content": user_query, "role": "user"},
        {"content": response, "role": "assistant"},
    ]

    return sample


class DataMixer:
    # pylint: disable=too-many-instance-attributes

    # This determines how many samples to pick from each skill when mixing
    # skill datasets. It's only used for skills, as knowledge may require
    # a variable number of samples depending on the length of the
    # knowledge documents in question. The expectation is that this is
    # enough samples to sufficiently learn a new skill while also ensuring
    # a balance of overall mixed data when learning multiple skills at
    # once.
    NUM_SYNTH_SKILLS = 30

    def __init__(
        self,
        data_dirs,
        output_dir,
        date_suffix,
        sys_prompt,
        num_procs,
        auxiliary_inst=None,
    ):
        self.data_dirs = data_dirs
        self.output_dir = output_dir
        self.sys_prompt = sys_prompt
        self.date_suffix = date_suffix
        self.num_procs = num_procs
        self.auxiliary_inst = auxiliary_inst

        self.knowledge_recipe = self._load_default_recipe("knowledge.yaml")
        self.skills_recipe = self._load_default_recipe("skills.yaml")

        self.output_file_knowledge_recipe = f"knowledge_recipe_{date_suffix}.yaml"
        self.output_file_skills_recipe = f"skills_recipe_{date_suffix}.yaml"
        self.output_file_mixed_knowledge = f"knowledge_train_msgs_{date_suffix}.jsonl"
        self.output_file_mixed_skills = f"skills_train_msgs_{date_suffix}.jsonl"

    def _load_default_recipe(self, yaml_basename):
        """
        Load a default system recipe from e.g. /usr/share/instructlab/sdg/default_data_recipes
        if it exists, otherwise return an empty recipe.
        """
        for d in self.data_dirs:
            default_recipe_path = os.path.join(d, "default_data_recipes", yaml_basename)
            if os.path.exists(default_recipe_path):
                return Recipe(
                    recipe_path=default_recipe_path, sys_prompt=self.sys_prompt
                )
        return Recipe(sys_prompt=self.sys_prompt)

    def _gen_leaf_node_data(
        self, leaf_node_data, recipe, output_file_leaf_node, sampling_size=1.0
    ):
        """
        Write the data generated from each taxonomy leaf node to a file.
        Later on, after all data is generated, the data mixing will read data
        from these files to generate the overall mixed dataset.
        """
        output_file = os.path.join(self.output_dir, output_file_leaf_node)
        leaf_node_data.to_json(output_file, orient="records", lines=True)
        recipe.add_dataset(output_file_leaf_node, sampling_size)

    def collect(
        self,
        leaf_node_path,
        new_generated_data,
        is_knowledge,
        use_legacy_pretraining_format,
    ):
        if is_knowledge:
            knowledge_phase_data = _create_phase07_ds(
                new_generated_data, self.auxiliary_inst, use_legacy_pretraining_format
            )
            output_file_leaf_knowledge = (
                f"node_datasets_{self.date_suffix}/{leaf_node_path}_p07.jsonl"
            )
            self._gen_leaf_node_data(
                knowledge_phase_data,
                self.knowledge_recipe,
                output_file_leaf_knowledge,
            )

            skills_phase_data = _create_phase10_ds(
                new_generated_data, self.auxiliary_inst, use_legacy_pretraining_format
            )
            output_file_leaf_skills = (
                f"node_datasets_{self.date_suffix}/{leaf_node_path}_p10.jsonl"
            )
            self._gen_leaf_node_data(
                skills_phase_data,
                self.skills_recipe,
                output_file_leaf_skills,
            )
        else:
            messages = new_generated_data.map(
                _convert_to_leaf_node_messages,
                fn_kwargs={"sys_prompt": self.sys_prompt},
                num_proc=self.num_procs,
            )
            output_file_leaf = (
                f"node_datasets_{self.date_suffix}/{leaf_node_path}.jsonl"
            )
            self._gen_leaf_node_data(
                messages,
                self.skills_recipe,
                output_file_leaf,
                sampling_size=self.NUM_SYNTH_SKILLS,
            )

    def _gen_mixed_data(self, recipe, output_file_recipe, output_file_data):
        """
        Mix the generated leaf node data into a single dataset and write it to
        disk. The heavy lifting is delegated to the Recipe class.
        """
        if recipe.dataset_added:
            full_recipe_path = os.path.join(self.output_dir, output_file_recipe)
            recipe.save_recipe(full_recipe_path)
            recipe.save_mixed_dataset(
                os.path.join(self.output_dir, output_file_data),
                self.num_procs,
            )

    def generate(self):
        self._gen_mixed_data(
            self.knowledge_recipe,
            self.output_file_knowledge_recipe,
            self.output_file_mixed_knowledge,
        )
        self._gen_mixed_data(
            self.skills_recipe,
            self.output_file_skills_recipe,
            self.output_file_mixed_skills,
        )
