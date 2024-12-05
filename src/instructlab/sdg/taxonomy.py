# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from pathlib import Path
from typing import Optional
import logging
import os

# Third Party
from xdg_base_dirs import xdg_data_dirs, xdg_data_home
import yaml

# First Party
from instructlab.sdg.utils import GenerateException
from instructlab.sdg.utils.json import jldump
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_WORD_COUNT = 1000
DEFAULT_TAXONOMY_BASE = "empty"
DEFAULT_SERVER_CTX_SIZE = 4096


def _locate_docling_models():
    # Search for the models in User and Site data directories
    data_dirs = [os.path.join(xdg_data_home(), "instructlab", "sdg")]
    data_dirs.extend(os.path.join(dir, "instructlab", "sdg") for dir in xdg_data_dirs())

    docling_model_path = None
    sdg_models_path = docling_model_path
    for d in data_dirs:
        if os.path.exists(os.path.join(d, "models")):
            sdg_models_path = os.path.join(d, "models")
            break

    if sdg_models_path is not None:
        try:
            with open(
                os.path.join(sdg_models_path, "config.yaml"), "r", encoding="utf-8"
            ) as file:
                config = yaml.safe_load(file)
                docling_model_path = config["models"][0]["path"]
        except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
            logger.warning(f"unable to read docling models path from config.yaml {e}")

    return docling_model_path


def taxonomy_to_samples(
    taxonomy_path,
    output_dir,
    chunk_word_count=DEFAULT_CHUNK_WORD_COUNT,  # TODO: Remove chunk_word_count param
    server_ctx_size=DEFAULT_SERVER_CTX_SIZE,  # TODO: Remove server_ctx_size param
    taxonomy_base=DEFAULT_TAXONOMY_BASE,
    teacher_model_path: Optional[str] = None,
    yaml_rules: Optional[str] = None,
):
    """
    Preprocess a taxonomy into input samples suitable for use with
    data generation pipelines. This does the following steps:

    - Determine changed leaf nodes in the taxonomy
    - Retrieve knowledge documents for changed taxonomy leaf nodes
    - Convert any non-markdown knowledge documents to markdown
    - Write the Docling json and markdown outputs from this conversion to
      disk for other processes to consume if needed.
    - Chunk the converted knowledge documents to the desired chunk sizes.
    - Turn the qna.yaml and knowledge documents into samples in the format
      expected by the `simple` and `full` data generation pipelines shipped
      in SDG.
    - Write these samples to disk, with one file per taxonomy leaf node.

    Args:
        taxonomy_path: The path to the taxonomy
        output_dir: Where to write the samples create for use with data generation
        chunk_word_count: The target number of words per document chunk
        server_ctx_size: The maximum number of tokens the inference server used
                         during data generation can handle
        taxonomy_base: Determines how we calculate what has changed. This should
                       be a git reference or the special value of 'empty' which
                       means assume the entire taxonomy has changed.
        teacher_model_path: Path to the teacher model on disk, which we'll use to
                            load its tokenizer for use with document chunking.
        yaml_rules: Path to a custom YAML rules file for YAML linting.

    Returns:
        List[str]: The list of output sample files written to disk.

    """
    logging.info("Converting taxonomy to samples")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_files = []

    if not (taxonomy_path and os.path.exists(taxonomy_path)):
        raise GenerateException(f"Error: taxonomy ({taxonomy_path}) does not exist.")

    document_output_dir = output_dir.joinpath("documents")
    docling_model_path = _locate_docling_models()

    leaf_nodes = read_taxonomy_leaf_nodes(
        taxonomy_path, taxonomy_base, yaml_rules, document_output_dir
    )
    if not leaf_nodes:
        raise GenerateException("Error: No new leaf nodes found in the taxonomy.")

    for leaf_node in leaf_nodes.values():
        leaf_node_path = leaf_node[0]["taxonomy_path"].replace("->", "_")
        samples = leaf_node_to_samples(
            leaf_node,
            taxonomy_path,
            server_ctx_size,
            chunk_word_count,
            document_output_dir,
            teacher_model_path,
            docling_model_path=docling_model_path,
        )

        if not samples:
            raise GenerateException("Error: No samples found in leaf node.")

        logger.debug("Samples: %s", samples)

        output_file = output_dir.joinpath(f"{leaf_node_path}.jsonl")
        jldump(samples, output_file)
        output_files.append(str(output_file))

    logger.info("Taxonomy converted to samples and written to %s", output_dir)
    return output_files
