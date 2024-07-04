# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
import glob
import json
import logging
import os
import re
import subprocess
import tempfile

# Third Party
import git
import gitdb
import yaml

# First Party
from instructlab.sdg import utils
from instructlab.sdg.utils import chunking

logger = logging.getLogger(__name__)

DEFAULT_YAML_RULES = """\
extends: relaxed

rules:
  line-length:
    max: 120
"""


class TaxonomyReadingException(Exception):
    """An exception raised during reading of the taxonomy."""


TAXONOMY_FOLDERS: List[str] = ["compositional_skills", "knowledge"]
"""Taxonomy folders which are also the schema names"""


class HTMLTagStripper(HTMLParser):
    def reset(self):
        super().reset()
        self.clean_text = ""

    def strip_html_tags(self, text: str) -> str:
        """Returns text without html tags for given text input.

        :param text: The original text to be processed
        """
        self.reset()
        self.feed(text)
        self.close()
        return self.clean_text

    def handle_data(self, data):
        self.clean_text += data


def _istaxonomyfile(fn):
    path = Path(fn)
    if path.suffix == ".yaml" and path.parts[0] in TAXONOMY_FOLDERS:
        return True
    return False


def _get_taxonomy_diff(repo="taxonomy", base="origin/main"):
    repo = git.Repo(repo)
    untracked_files = [u for u in repo.untracked_files if _istaxonomyfile(u)]

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
        if not d.deleted_file and _istaxonomyfile(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


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
            tag_stripper = HTMLTagStripper()

            for pattern in file_patterns:
                for file_path in glob.glob(os.path.join(repo.working_dir, pattern)):
                    if os.path.isfile(file_path) and file_path.endswith(".md"):
                        with open(file_path, "r", encoding="utf-8") as file:
                            file_contents.append(tag_stripper.strip_html_tags(file.read()))

            if file_contents:
                return file_contents
            raise SystemExit("Couldn't find knowledge documents")
        except (OSError, git.exc.GitCommandError, FileNotFoundError) as e:
            raise e


@cache
def _load_schema(path: "importlib.resources.abc.Traversable") -> "referencing.Resource":
    """Load the schema from the path into a Resource object.

    Args:
        path (Traversable): Path to the schema to be loaded.

    Raises:
        NoSuchResource: If the resource cannot be loaded.

    Returns:
        Resource: A Resource containing the requested schema.
    """
    # pylint: disable=C0415
    # Third Party
    from referencing import Resource
    from referencing.exceptions import NoSuchResource
    from referencing.jsonschema import DRAFT202012

    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(
            contents=contents, default_specification=DRAFT202012
        )
    except Exception as e:
        raise NoSuchResource(ref=str(path)) from e
    return resource


def _validate_yaml(contents: Mapping[str, Any], taxonomy_path: Path) -> int:
    """Validate the parsed yaml document using the taxonomy path to
    determine the proper schema.

    Args:
        contents (Mapping): The parsed yaml document to validate against the schema.
        taxonomy_path (Path): Relative path of the taxonomy yaml document where the
        first element is the schema to use.

    Returns:
        int: The number of errors found during validation.
        Messages for each error have been logged.
    """
    # pylint: disable=C0415
    # Standard
    from importlib import resources

    # Third Party
    from jsonschema.protocols import Validator
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
    from referencing.exceptions import NoSuchResource
    from referencing.typing import URI

    errors = 0
    version = _get_version(contents)
    schemas_path = resources.files("instructlab.schema").joinpath(f"v{version}")

    def retrieve(uri: URI) -> Resource:
        path = schemas_path.joinpath(uri)
        return _load_schema(path)

    schema_name = taxonomy_path.parts[0]
    if schema_name not in TAXONOMY_FOLDERS:
        schema_name = "knowledge" if "document" in contents else "compositional_skills"
        logger.info(
            f"Cannot determine schema name from path {taxonomy_path}. Using {schema_name} schema."
        )

    try:
        schema_resource = retrieve(f"{schema_name}.json")
        schema = schema_resource.contents
        validator_cls = validator_for(schema)
        validator: Validator = validator_cls(
            schema, registry=Registry(retrieve=retrieve)
        )

        for validation_error in validator.iter_errors(contents):
            errors += 1
            yaml_path = validation_error.json_path[1:]
            if not yaml_path:
                yaml_path = "."
            if validation_error.validator == "minItems":
                # Special handling for minItems which can have a long message for seed_examples
                message = (
                    f"Value must have at least {validation_error.validator_value} items"
                )
            else:
                message = validation_error.message[-200:]
            logger.error(
                f"Validation error in {taxonomy_path}: [{yaml_path}] {message}"
            )
    except NoSuchResource as e:
        cause = e.__cause__ if e.__cause__ is not None else e
        errors += 1
        logger.error(f"Cannot load schema file {e.ref}. {cause}")

    return errors


def _get_version(contents: Mapping) -> int:
    version = contents.get("version", 1)
    if not isinstance(version, int):
        # schema validation will complain about the type
        try:
            version = int(version)
        except ValueError:
            version = 1  # fallback to version 1
    return version


# pylint: disable=broad-exception-caught
def _read_taxonomy_file(file_path: str, yaml_rules: Optional[str] = None):
    seed_instruction_data = []
    warnings = 0
    errors = 0
    file_path = Path(file_path).resolve()
    # file should end with ".yaml" explicitly
    if file_path.suffix != ".yaml":
        logger.warning(
            f"Skipping {file_path}! Use lowercase '.yaml' extension instead."
        )
        warnings += 1
        return None, warnings, errors
    for i in range(len(file_path.parts) - 1, -1, -1):
        if file_path.parts[i] in TAXONOMY_FOLDERS:
            taxonomy_path = Path(*file_path.parts[i:])
            break
    else:
        taxonomy_path = file_path
    # read file if extension is correct
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            contents = yaml.safe_load(file)
        if not contents:
            logger.warning(f"Skipping {file_path} because it is empty!")
            warnings += 1
            return None, warnings, errors
        if not isinstance(contents, Mapping):
            logger.error(
                f"{file_path} is not valid. The top-level element is not an object with key-value pairs."
            )
            errors += 1
            return None, warnings, errors

        # do general YAML linting if specified
        version = _get_version(contents)
        if version > 1:  # no linting for version 1 yaml
            if yaml_rules is not None:
                is_file = os.path.isfile(yaml_rules)
                if is_file:
                    logger.debug(f"Using YAML rules from {yaml_rules}")
                    yamllint_cmd = [
                        "yamllint",
                        "-f",
                        "parsable",
                        "-c",
                        yaml_rules,
                        file_path,
                        "-s",
                    ]
                else:
                    logger.debug(f"Cannot find {yaml_rules}. Using default rules.")
                    yamllint_cmd = [
                        "yamllint",
                        "-f",
                        "parsable",
                        "-d",
                        DEFAULT_YAML_RULES,
                        file_path,
                        "-s",
                    ]
            else:
                yamllint_cmd = [
                    "yamllint",
                    "-f",
                    "parsable",
                    "-d",
                    DEFAULT_YAML_RULES,
                    file_path,
                    "-s",
                ]
            try:
                subprocess.check_output(yamllint_cmd, text=True)
            except subprocess.SubprocessError as e:
                lint_messages = [f"Problems found in file {file_path}"]
                parsed_output = e.output.splitlines()
                for p in parsed_output:
                    errors += 1
                    delim = str(file_path) + ":"
                    parsed_p = p.split(delim)[1]
                    lint_messages.append(parsed_p)
                logger.error("\n".join(lint_messages))
                return None, warnings, errors

        validation_errors = _validate_yaml(contents, taxonomy_path)
        if validation_errors:
            errors += validation_errors
            return None, warnings, errors

        # get seed instruction data
        tax_path = "->".join(taxonomy_path.parent.parts)
        task_description = contents.get("task_description")
        domain = contents.get("domain")
        documents = contents.get("document")
        if documents:
            documents = _get_documents(source=documents)
            logger.debug("Content from git repo fetched")

        for seed_example in contents.get("seed_examples"):
            question = seed_example.get("question")
            answer = seed_example.get("answer")
            context = seed_example.get("context", "")
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
        errors += 1
        raise TaxonomyReadingException(f"Exception {e} raised in {file_path}") from e

    return seed_instruction_data, warnings, errors


def read_taxonomy(taxonomy, taxonomy_base, yaml_rules):
    seed_instruction_data = []
    is_file = os.path.isfile(taxonomy)
    if is_file:  # taxonomy is file
        seed_instruction_data, warnings, errors = _read_taxonomy_file(
            taxonomy, yaml_rules
        )
        if warnings:
            logger.warning(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
    else:  # taxonomy is dir
        # Gather the new or changed YAMLs using git diff
        updated_taxonomy_files = _get_taxonomy_diff(taxonomy, taxonomy_base)
        total_errors = 0
        total_warnings = 0
        if updated_taxonomy_files:
            logger.debug("Found new taxonomy files:")
            for e in updated_taxonomy_files:
                logger.debug(f"* {e}")
        for f in updated_taxonomy_files:
            file_path = os.path.join(taxonomy, f)
            data, warnings, errors = _read_taxonomy_file(file_path, yaml_rules)
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
    samples = [{}]

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
        for i in range(len(leaf_node)):
            samples[-1].setdefault("task_description", leaf_node[i]["task_description"])
            samples[-1].setdefault("domain", domain)
            samples[-1].setdefault("document", chunk)
            if samples[-1].get("document") and not samples[-1].get("domain"):
                raise utils.GenerateException(
                    "Error: No domain provided for knowledge document in leaf node"
                )
            if "icl_query_3" in samples[-1]:
                samples.append({})
            if "icl_query_1" not in samples[-1]:
                samples[-1]["icl_query_1"] = leaf_node[i]["instruction"]
                samples[-1]["icl_response_1"] = leaf_node[i]["output"]
            elif "icl_query_2" not in samples[-1]:
                samples[-1]["icl_query_2"] = leaf_node[i]["instruction"]
                samples[-1]["icl_response_2"] = leaf_node[i]["output"]
            else:
                samples[-1]["icl_query_3"] = leaf_node[i]["instruction"]
                samples[-1]["icl_response_3"] = leaf_node[i]["output"]

        # wrap back around to the beginning if the number of examples was not
        # evenly divisble by 3
        if "icl_query_2" not in samples[-1]:
            samples[-1]["icl_query_2"] = leaf_node[0]["instruction"]
            samples[-1]["icl_response_2"] = leaf_node[0]["output"]
        if "icl_query_3" not in samples[-1]:
            samples[-1]["icl_query_3"] = leaf_node[1 if len(leaf_node) > 1 else 0][
                "instruction"
            ]
            samples[-1]["icl_response_3"] = leaf_node[1 if len(leaf_node) > 1 else 0][
                "output"
            ]

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
