# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

# Third Party
from kfp import dsl

# Local
from .._version import __version__

# if "dev" in __version__:
#     tox_env_dir = os.getenv("TOX_ENV_DIR", None)
#     if tox_env_dir:
#         sdg_install_version = str(Path(tox_env_dir).parent.parent)
#     else:
#         sdg_install_version = str(Path(__file__).parent.parent)
# else:
#     sdg_install_version = f"instructlab-sdg=={__version__}"

KFP_BASE_IMAGE = "quay.io/bbrowning/sdg-kfp:0.0.4"
# KFP_PACKAGES_TO_INSTALL = [sdg_install_version]
KFP_PACKAGES_TO_INSTALL = []


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
    # Third Party
    from openai import OpenAI

    # First Party
    from instructlab.sdg.generate_data import generate_taxonomy as orig_generate

    client = OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY")
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


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def duplicate_columns_block(
    input_ds: dsl.Input[dsl.Dataset],
    output_ds: dsl.Output[dsl.Dataset],
    columns_map: Dict[str, str],
):
    # Third Party
    import datasets as hf_datasets

    # First Party
    from instructlab.sdg.utils.json import jldump, jlload

    samples = hf_datasets.Dataset.from_list(jlload(input_ds.path))
    for col_to_dup in columns_map:
        samples = samples.add_column(
            columns_map[col_to_dup], samples[col_to_dup]
        )

    jldump(samples, output_ds.path)


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def filter_by_value_block(
    input_ds: dsl.Input[dsl.Dataset],
    output_ds: dsl.Output[dsl.Dataset],
    filter_column: str,
    filter_value: str,
    operation: str,
    convert_dtype: Optional[str] = None,
    default_value: Optional[str] = None,
    drop_columns: List[str] = None,
):
    # Third Party
    import datasets as hf_datasets

    # First Party
    from instructlab.sdg.blocks.filterblock import (
        _filter_by_values,
        _get_operator_func,
        _map_dtype,
        DTypeConverter,
    )
    from instructlab.sdg.utils import pandas
    from instructlab.sdg.utils.json import jldump, jlload

    samples = hf_datasets.Dataset.from_list(jlload(input_ds.path))

    value = filter_value if isinstance(filter_value, list) else [filter_value]
    column_name = filter_column
    operation = _get_operator_func(operation)
    dtype = DTypeConverter.get(convert_dtype, default_value)
    if dtype:
        value = [dtype(value) for value in value]

    num_procs = 2
    if dtype:
        samples = _map_dtype(
            samples,
            column_name,
            dtype,
            num_procs,
        )

    results = _filter_by_values(
        samples,
        column_name,
        operation,
        value,
        num_procs,
    )

    drop_columns_in_ds = [
        e for e in drop_columns if e in results.column_names
    ]
    if drop_columns_in_ds:
        results = results.remove_columns(drop_columns_in_ds)

    jldump(results, output_ds.path)


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def flatten_columns_block(
    input_ds: dsl.Input[dsl.Dataset],
    output_ds: dsl.Output[dsl.Dataset],
    var_cols: List[str],
    value_name: str,
    var_name: str,
):
    # Third Party
    import datasets as hf_datasets

    # First Party
    from instructlab.sdg.utils import pandas
    from instructlab.sdg.utils.json import jldump, jlload

    samples = hf_datasets.Dataset.from_list(jlload(input_ds.path))
    df = samples.to_pandas()
    id_cols = [col for col in samples.column_names if col not in var_cols]
    flatten_df = df.melt(
        id_vars=id_cols,
        value_vars=var_cols,
        value_name=value_name,
        var_name=var_name,
    )
    results = pandas.dataset_from_pandas_dataframe(flatten_df)
    jldump(results, output_ds.path)


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def rename_columns_block(
    input_ds: dsl.Input[dsl.Dataset],
    output_ds: dsl.Output[dsl.Dataset],
    columns_map: Dict[str, str],
):
    # Third Party
    import datasets as hf_datasets

    # First Party
    from instructlab.sdg.utils import pandas
    from instructlab.sdg.utils.json import jldump, jlload

    samples = hf_datasets.Dataset.from_list(jlload(input_ds.path))
    samples = samples.rename_columns(columns_map)
    jldump(samples, output_ds.path)


@dsl.component(
    base_image=KFP_BASE_IMAGE,
    packages_to_install=KFP_PACKAGES_TO_INSTALL,
)
def llm_block(
    input_ds: dsl.Input[dsl.Dataset],
    output_ds: dsl.Output[dsl.Dataset],
    block_name: str,
    output_cols: List[str],
    block_config: Dict[str, Any],
    gen_kwargs: Dict[str, Any] = {},
    parser_kwargs: Dict[str, Any] = {},
    drop_duplicates: List[str] = [],
):
    # Third Party
    from jinja2 import Environment, StrictUndefined
    from openai import OpenAI
    import datasets as hf_datasets

    # First Party
    from instructlab.sdg.blocks.llmblock import LLMBlock
    from instructlab.sdg.pipeline import PipelineContext
    from instructlab.sdg.utils import pandas
    from instructlab.sdg.utils.json import jldump, jlload

    samples = hf_datasets.Dataset.from_list(jlload(input_ds.path))

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    ctx = PipelineContext(client)

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    llmblock = LLMBlock(
        ctx=ctx,
        pipe=None,
        block_name=block_name,
        config_path=None,
        output_cols=output_cols,
        model_id=model_id,
        gen_kwargs=gen_kwargs,
        parser_kwargs=parser_kwargs,
        block_config=block_config,
    )
    results = llmblock.generate(samples)

    if drop_duplicates:
        df = results.to_pandas()
        df = df.drop_duplicates(subset=drop_duplicates)
        results = pandas.dataset_from_pandas_dataframe(df)

    jldump(results, output_ds.path)
