# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Dict, List, Union
import glob
import logging
import os
import re
import tempfile

# Third Party
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
from . import chunking

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


def _get_documents(
    source: Dict[str, Union[str, List[str]]],
    skip_checkout: bool = False,
) -> List[str]:
    """
    Retrieve the content of files from a Git repository.

    Args:
        source (dict): Source info containing repository URL, commit hash, and list of file patterns.

    Returns:
         List[str]: List of document contents.
    """ ""
    repo_url = source.get("repo")
    commit_hash = source.get("commit")
    file_patterns = source.get("patterns", [])
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            repo = git.Repo.clone_from(repo_url, temp_dir)
            if not skip_checkout:
                repo.git.checkout(commit_hash)

            file_contents = []

            logger.debug("Processing files...")
            for pattern in file_patterns:
                for file_path in glob.glob(os.path.join(repo.working_dir, pattern)):
                    if os.path.isfile(file_path) and file_path.endswith(".md"):
                        with open(file_path, "r", encoding="utf-8") as file:
                            file_contents.append(file.read())

            if file_contents:
                return file_contents
            raise SystemExit("Couldn't find knowledge documents")
        except (OSError, git.exc.GitCommandError, FileNotFoundError) as e:
            raise e


# pylint: disable=broad-exception-caught
def _read_taxonomy_file(file_path: str | Path, yamllint_config: str | None = None):
    seed_instruction_data = []

    parser = TaxonomyParser(
        schema_version=0,  # Use version value in yaml
        message_format=TaxonomyMessageFormat.LOGGING,  # Report warnings and errors to the logger
        yamllint_config=yamllint_config,
        yamllint_strict=True,  # Report yamllint warnings as errors
    )
    taxonomy = parser.parse(file_path)

    if taxonomy.warnings or taxonomy.errors:
        return seed_instruction_data, taxonomy.warnings, taxonomy.errors

    try:
        # get seed instruction data
        tax_path = "->".join(taxonomy.path.parent.parts)
        contents = taxonomy.contents
        task_description = contents.get("task_description", None)
        domain = contents.get("domain")
        documents = contents.get("document")
        if documents:
            documents = _get_documents(source=documents)
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
                        "document": documents,
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
    taxonomy: str | Path, taxonomy_base: str, yaml_rules: str | None = None
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
            taxonomy, yamllint_config
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
            data, warnings, errors = _read_taxonomy_file(file_path, yamllint_config)
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


def read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules):
    seed_instruction_data = read_taxonomy(taxonomy, taxonomy_base, yaml_rules)

    # Transform into a more convenient format to feed into our updated SDG library
    leaf_nodes = {}
    for seed in seed_instruction_data:
        node = leaf_nodes.setdefault(seed["taxonomy_path"], [])
        node.append(seed)
        leaf_nodes[seed["taxonomy_path"]] = node

    return leaf_nodes


def _knowledge_leaf_node_to_samples(leaf_node, server_ctx_size, chunk_word_count):
    samples = []
    # document is the same for the whole leaf node
    chunks = (
        chunking.chunk_document(
            documents=leaf_node[0]["document"],
            server_ctx_size=server_ctx_size,
            chunk_word_count=chunk_word_count,
        )
        if leaf_node[0].get("document")
        else []
    )

    # domain is the same for the whole leaf node
    domain = leaf_node[0].get("domain")

    for chunk in chunks:
        # pylint: disable=consider-using-enumerate
        for icl_ in leaf_node:
            icl_query = {
                f"icl_query_{idx+1}": val["question"]
                for idx, val in enumerate(icl_["questions_and_answers"])
            }
            icl_resp = {
                f"icl_response_{idx+1}": val["answer"]
                for idx, val in enumerate(icl_["questions_and_answers"])
            }
            samples_row = {
                "icl_document": icl_["context"],
                "document": chunk,
                "document_outline": icl_["document_outline"],
                "domain": domain,
            }
            samples_row.update(icl_query)
            samples_row.update(icl_resp)
            samples.append(samples_row)

    return samples


def _skill_leaf_node_to_samples(leaf_node):
    samples = []

    # pylint: disable=consider-using-enumerate
    for i in range(len(leaf_node)):
        samples.append({})
        samples[-1]["task_description"] = leaf_node[i]["task_description"]
        if leaf_node[i].get("input"):
            samples[-1]["seed_context"] = leaf_node[i]["input"]
        samples[-1]["seed_question"] = leaf_node[i]["instruction"]
        samples[-1]["seed_response"] = leaf_node[i]["output"]

    return samples


def leaf_node_to_samples(leaf_node, server_ctx_size, chunk_word_count):
    if not leaf_node:
        return []
    if leaf_node[0].get("document"):
        return _knowledge_leaf_node_to_samples(
            leaf_node, server_ctx_size, chunk_word_count
        )
    return _skill_leaf_node_to_samples(leaf_node)
