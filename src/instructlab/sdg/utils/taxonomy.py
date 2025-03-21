# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Union
import glob
import logging
import os
import re

# Third Party
from datasets import Dataset
from instructlab.schema.taxonomy import DEFAULT_TAXONOMY_FOLDERS as TAXONOMY_FOLDERS
from instructlab.schema.taxonomy import (
    TaxonomyMessageFormat,
    TaxonomyParser,
    TaxonomyReadingException,
)
import git
import gitdb
import yaml

# Local
from .chunkers import DocumentChunker

logger = logging.getLogger(__name__)


def _is_taxonomy_file(fn: str) -> bool:
    path = Path(fn)
    if path.parts[0] not in TAXONOMY_FOLDERS:
        return False
    if path.name == "qna.yaml":
        return True
    if path.name.casefold() in {"qna.yml", "qna.yaml"}:
        # warning for incorrect extension or case variants
        logger.warning(
            "Found a '%s' file: %s: taxonomy files must be named 'qna.yaml'. File will not be checked.",
            path.name,
            path,
        )
    return False


def _get_taxonomy_diff(
    repo_path: str | Path = "taxonomy", base: str = "origin/main"
) -> list[str]:
    repo = git.Repo(repo_path)
    untracked_files = [u for u in repo.untracked_files if _is_taxonomy_file(u)]

    branches = [b.name for b in repo.branches]

    head_commit = None
    if "/" in base:
        re_git_branch = re.compile(f"remotes/{base}$", re.MULTILINE)
    elif base in branches:
        re_git_branch = re.compile(f"{base}$", re.MULTILINE)
    else:
        try:
            head_commit = repo.commit(base)
        except gitdb.exc.BadName as e:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy git ref "{base}" from the current HEAD'
                )
            ) from e

    # Move backwards from HEAD until we find the first commit that is part of base
    # then we can take our diff from there
    current_commit = repo.commit("HEAD")
    while not head_commit:
        branches = repo.git.branch("-a", "--contains", current_commit.hexsha)
        if re_git_branch.findall(branches):
            head_commit = current_commit
            break
        try:
            current_commit = current_commit.parents[0]
        except IndexError as e:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy base branch "{base}" from the current HEAD'
                )
            ) from e

    modified_files = [
        d.b_path
        for d in head_commit.diff(None)
        if not d.deleted_file and _is_taxonomy_file(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


def _get_taxonomy(repo="taxonomy"):
    repo = Path(repo)
    taxonomy_file_paths = []
    for root, _, files in os.walk(repo):
        for file in files:
            file_path = Path(root).joinpath(file).relative_to(repo)
            if _is_taxonomy_file(file_path):
                taxonomy_file_paths.append(str(file_path))
    return taxonomy_file_paths


def _string_contains_html(s: str) -> bool:
    """Detect HTML tags in a string.

    We use this to catch markdown files that may contain html elements since
    docling does not support this."""
    # Define a regex to detect HTML tags
    html_tag_pattern = re.compile(r"<\/?[a-zA-Z][\s\S]*?>")

    # Check for HTML tags in the content
    return bool(html_tag_pattern.search(s))


def _get_documents(
    source: Dict[str, Union[str, List[str]]],
    skip_checkout: bool = False,
    document_output_dir: Path = None,
) -> List[Path]:
    """
    Retrieve the file paths of files (Markdown and PDF) from a Git repository.

    Args:
        source (dict): Source info containing repository URL, commit hash, and list of file patterns.
        skip_checkout (bool, optional): If True, skips checking out the specific commit. Defaults to False.
        document_output_dir (Path, optional): Directory to clone the repository into. Defaults to current directory.

    Returns:
        Tuple[List[str], List[Path]]:
            - List of document contents (Markdown as text and PDFs as extracted text).
            - List of corresponding file paths.

    Raises:
        SystemExit: If no valid documents are found.
        OSError, GitCommandError, FileNotFoundError: For errors during Git operations or file access.
    """
    repo_url = source.get("repo")
    commit_hash = source.get("commit")
    file_patterns = source.get("patterns", [])
    # pylint: disable=too-many-nested-blocks
    try:
        repo = git.Repo.clone_from(repo_url, document_output_dir)

        if not skip_checkout and commit_hash:
            repo.git.checkout(commit_hash)

        filepaths = []

        logger.info("Processing files...")
        for pattern in file_patterns:
            # Use glob to find files matching the pattern
            matched_files = glob.glob(
                os.path.join(repo.working_dir, pattern), recursive=True
            )
            logger.info(f"Pattern '{pattern}' matched {len(matched_files)} files.")

            for file_path in matched_files:
                if os.path.isfile(file_path):
                    logger.info(f"Processing file: {file_path}")
                    try:
                        if file_path.lower().endswith(".md"):
                            with open(file_path, "r", encoding="utf-8") as file:
                                content = file.read()
                                if _string_contains_html(content):
                                    logging.warning(
                                        f"Provided markdown file {file_path} contains HTML contents, which is currently unsupported as a part of markdown"
                                        "NOTE: Continuing this might affect your data generation quality."
                                        "To get best results please format your markdown documents without the use of HTML or use a different document filetype."
                                    )
                        filepaths.append(Path(file_path))
                        logger.info(f"Collected filepath: {file_path}")
                    # pylint: disable=broad-exception-caught
                    except Exception as file_error:
                        logger.error(
                            f"Error processing file '{file_path}': {file_error}"
                        )
                        continue
                else:
                    logger.info(f"Skipping non-file path: {file_path}")

        if filepaths:
            return filepaths
        raise SystemExit("Couldn't find knowledge documents")

    except (OSError, git.exc.GitCommandError, FileNotFoundError) as e:
        logger.error("Error retrieving documents: %s", str(e))
        raise e


def _read_taxonomy_file(
    file_path: str | Path,
    yamllint_config: str | None = None,
    document_output_dir: Path = Path(),
):
    seed_instruction_data = []

    taxonomy_parser = TaxonomyParser(
        schema_version=0,  # Use version value in yaml
        message_format=TaxonomyMessageFormat.LOGGING,  # Report warnings and errors to the logger
        yamllint_config=yamllint_config,
        yamllint_strict=True,  # Report yamllint warnings as errors
    )
    taxonomy = taxonomy_parser.parse(file_path)

    if taxonomy.warnings or taxonomy.errors:
        return seed_instruction_data, taxonomy.warnings, taxonomy.errors

    try:
        # get seed instruction data
        tax_path = "->".join(taxonomy.path.parent.parts)
        leaf_node_path = tax_path.replace("->", "_")
        contents = taxonomy.contents
        task_description = contents.get("task_description", None)
        domain = contents.get("domain")
        documents = contents.get("document")
        doc_filepaths = None
        if documents:
            os.makedirs(document_output_dir, exist_ok=True)
            unique_output_dir = mkdtemp(
                prefix=f"{leaf_node_path}_", dir=document_output_dir
            )
            doc_filepaths = _get_documents(
                source=documents,
                document_output_dir=unique_output_dir,
            )
            logger.debug("Content from git repo fetched")

        for seed_example in contents.get("seed_examples"):
            context = seed_example.get("context", "")
            if "questions_and_answers" in seed_example:
                question_answer_list = seed_example.get("questions_and_answers")
                seed_instruction_data.append(
                    {
                        "questions_and_answers": question_answer_list,
                        "context": context,
                        "taxonomy_path": tax_path,
                        "filepaths": doc_filepaths,
                        "domain": domain,
                        "document_outline": contents.get("document_outline"),
                    }
                )
            else:
                question = seed_example.get("question")
                answer = seed_example.get("answer")

                seed_instruction_data.append(
                    {
                        "instruction": question,
                        "input": context,
                        "output": answer,
                        "taxonomy_path": tax_path,
                        "task_description": task_description,
                        "document": documents,
                        "domain": domain,
                    }
                )
    except Exception as e:
        raise TaxonomyReadingException(f"Exception {e} raised in {file_path}") from e

    return seed_instruction_data, 0, 0


def read_taxonomy(
    taxonomy: str | Path,
    taxonomy_base: str,
    yaml_rules: str | None = None,
    document_output_dir: Path | None = None,
):
    yamllint_config = None  # If no custom rules file, use default config
    if yaml_rules is not None:  # user attempted to pass custom rules file
        yaml_rules_path = Path(yaml_rules)
        if yaml_rules_path.is_file():  # file was found, use specified config
            logger.debug("Using YAML rules from %s", yaml_rules)
            yamllint_config = yaml_rules_path.read_text(encoding="utf-8")
        else:
            logger.debug("Cannot find %s. Using default rules.", yaml_rules)

    seed_instruction_data = []
    is_file = os.path.isfile(taxonomy)
    if is_file:  # taxonomy is file
        seed_instruction_data, warnings, errors = _read_taxonomy_file(
            taxonomy, yamllint_config, document_output_dir
        )
        if warnings:
            logger.warning(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
    else:  # taxonomy is dir
        if taxonomy_base == "empty":
            # Gather all the yamls - equivalent to a diff against "the null tree"
            taxonomy_files = _get_taxonomy(taxonomy)
        else:
            # Gather the new or changed YAMLs using git diff, including untracked files
            taxonomy_files = _get_taxonomy_diff(taxonomy, taxonomy_base)
        total_errors = 0
        total_warnings = 0
        if taxonomy_files:
            logger.debug("Found taxonomy files:")
            for e in taxonomy_files:
                logger.debug(f"* {e}")
        for f in taxonomy_files:
            file_path = os.path.join(taxonomy, f)
            data, warnings, errors = _read_taxonomy_file(
                file_path, yamllint_config, document_output_dir
            )
            total_warnings += warnings
            total_errors += errors
            if data:
                seed_instruction_data.extend(data)
        if total_warnings:
            logger.warning(
                f"{total_warnings} warnings (see above) due to taxonomy files that were not (fully) usable."
            )
        if total_errors:
            raise SystemExit(
                yaml.YAMLError(f"{total_errors} taxonomy files with errors! Exiting.")
            )
    return seed_instruction_data


def read_taxonomy_leaf_nodes(
    taxonomy, taxonomy_base, yaml_rules, document_output_dir=None
):
    seed_instruction_data = read_taxonomy(
        taxonomy, taxonomy_base, yaml_rules, document_output_dir
    )

    # Transform into a more convenient format to feed into our updated SDG library
    leaf_nodes = {}
    for seed in seed_instruction_data:
        node = leaf_nodes.setdefault(seed["taxonomy_path"], [])
        node.append(seed)
        leaf_nodes[seed["taxonomy_path"]] = node

    return leaf_nodes


def map_chunks_to_icls(chunks: List, leaf_node: Dict) -> Dataset:
    chunked_dataset = []

    # domain is the same for the whole leaf node
    domain = leaf_node[0].get("domain")
    for chunk in chunks:
        for icl_ in leaf_node:
            record = {
                "document": chunk,
                "icl_document": icl_.get("context", ""),
                "document_outline": icl_.get("document_outline", ""),
                "domain": domain,
                "leaf_node_type": "knowledge",
            }

            qna_pairs = icl_.get("questions_and_answers", [])
            for i, qna in enumerate(qna_pairs):
                record.update(
                    {
                        f"icl_query_{i+1}": qna.get("question", ""),
                        f"icl_response_{i+1}": qna.get("answer", ""),
                    }
                )

            chunked_dataset.append(record)

    return chunked_dataset


def _knowledge_leaf_node_to_samples(
    leaf_node,
    server_ctx_size,
    chunk_word_count,
    document_output_dir,
    model_name,
    docling_model_path=None,
):
    document_paths = leaf_node[0]["filepaths"]
    chunker = DocumentChunker(
        document_paths=document_paths,
        output_dir=document_output_dir,
        tokenizer_model_name=model_name,
        server_ctx_size=server_ctx_size,
        chunk_word_count=chunk_word_count,
        docling_model_path=docling_model_path,
    )
    chunks = chunker.chunk_documents()

    samples = map_chunks_to_icls(chunks, leaf_node)
    return samples


def _skill_leaf_node_to_samples(leaf_node):
    samples = []

    # pylint: disable=consider-using-enumerate
    for i in range(len(leaf_node)):
        samples.append({})
        samples[-1]["task_description"] = leaf_node[i]["task_description"]
        sample_type = "freeform_skill"
        if leaf_node[i].get("input"):
            sample_type = "grounded_skill"
            samples[-1]["seed_context"] = leaf_node[i]["input"]
        samples[-1]["seed_question"] = leaf_node[i]["instruction"]
        samples[-1]["seed_response"] = leaf_node[i]["output"]
        samples[-1]["leaf_node_type"] = sample_type

    return samples


def _enrich_metadata(samples, leaf_node):
    leaf_node_path = leaf_node[0]["taxonomy_path"].replace("->", "_")
    for i, sample in enumerate(samples):
        sample["leaf_node_path"] = leaf_node_path
        samples[i] = sample
    return samples


def leaf_node_to_samples(
    leaf_node,
    server_ctx_size,
    chunk_word_count,
    document_output_dir,
    model_name,
    docling_model_path=None,
):
    samples = []
    # check if the leaf node has document filepaths, if so, it's a knowledge leaf node
    if leaf_node and (leaf_node[0].get("filepaths")):
        samples = _knowledge_leaf_node_to_samples(
            leaf_node,
            server_ctx_size,
            chunk_word_count,
            document_output_dir,
            model_name,
            docling_model_path,
        )
    elif leaf_node:
        samples = _skill_leaf_node_to_samples(leaf_node)
    samples = _enrich_metadata(samples, leaf_node)
    return Dataset.from_list(samples)
