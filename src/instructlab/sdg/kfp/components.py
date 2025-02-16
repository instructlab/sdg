# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Optional
import os

# Third Party
from kfp import dsl

# Local
from .._version import __version__

if "dev" in __version__:
    tox_env_dir = os.getenv("TOX_ENV_DIR", None)
    if tox_env_dir:
        sdg_install_version = str(Path(tox_env_dir).parent.parent)
    else:
        sdg_install_version = str(Path(__file__).parent.parent)
else:
    sdg_install_version = f"instructlab-sdg=={__version__}"

KFP_BASE_IMAGE = "python:3.10"
KFP_PACKAGES_TO_INSTALL = [sdg_install_version]


@dsl.component(base_image=KFP_BASE_IMAGE)
def taxonomy_git_importer(
    taxonomy_repo: str,
    output: dsl.OutputPath("Directory"),  # type: ignore
):
    # Third Party
    import git

    git.Repo.clone_from(taxonomy_repo, output)


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def preprocess_taxonomy(
    taxonomy_path: dsl.InputPath("Directory"),  # type: ignore
    output_path: dsl.OutputPath("Directory"),  # type: ignore
    chunk_word_count: int = 1000,
    server_ctx_size: int = 4096,
    taxonomy_base: str = "empty",
    teacher_model_path: Optional[str] = None,
    yaml_rules: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    # First Party
    from instructlab.sdg.generate_data import preprocess_taxonomy as orig_preprocess

    orig_preprocess(
        taxonomy_dir=taxonomy_path,
        output_dir=output_path,
        chunk_word_count=chunk_word_count,
        server_ctx_size=server_ctx_size,
        taxonomy_base=taxonomy_base,
        teacher_model_path=teacher_model_path,
        yaml_rules=yaml_rules,
        system_prompt=system_prompt,
    )


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def generate_taxonomy(
    preprocessed_path: dsl.InputPath("Directory"),  # type: ignore
    output_path: dsl.OutputPath("Directory"),  # type: ignore
    model_family: Optional[str] = None,
    model_id: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    console_output: bool = True,
    pipeline: Optional[str] = "simple",
    batch_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    max_num_tokens: Optional[int] = 4096,
):
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from instructlab.sdg.generate_data import generate_taxonomy as orig_generate

    client = MagicMock()
    client.server_supports_batched = False

    orig_generate(
        client=client,
        input_dir=preprocessed_path,
        output_dir=output_path,
        model_family=model_family,
        model_id=model_id,
        num_cpus=num_cpus,
        num_instructions_to_generate=num_instructions_to_generate,
        console_output=console_output,
        pipeline=pipeline,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        max_num_tokens=max_num_tokens,
    )


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def postprocess_taxonomy(
    generated_path: dsl.InputPath("Directory"),  # type: ignore
    output_path: dsl.OutputPath("Directory"),  # type: ignore
    date_suffix: str,
    pipeline: Optional[str] = "simple",
):
    # First Party
    from instructlab.sdg.generate_data import postprocess_taxonomy as orig_postprocess

    orig_postprocess(
        input_dir=generated_path,
        output_dir=output_path,
        date_suffix=date_suffix,
        pipeline=pipeline,
    )


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def mix_taxonomy_datasets(
    postprocessed_path: dsl.InputPath("Directory"),  # type: ignore
    mixed_skills: dsl.Output[dsl.Dataset],
    mixed_knowledge: dsl.Output[dsl.Dataset],
    date_suffix: str,
):
    # Standard
    import os

    # First Party
    from instructlab.sdg.generate_data import mix_datasets as orig_mix

    skills_recipe = os.path.join(
        postprocessed_path,
        f"skills_recipe_{date_suffix}.yaml",
    )
    orig_mix(
        recipe_file=skills_recipe,
        output_file=mixed_skills.path,
    )

    knowledge_recipe = os.path.join(
        postprocessed_path,
        f"knowledge_recipe_{date_suffix}.yaml",
    )
    orig_mix(
        recipe_file=knowledge_recipe,
        output_file=mixed_knowledge.path,
    )
