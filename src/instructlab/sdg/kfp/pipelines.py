# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import namedtuple
from datetime import datetime
from typing import NamedTuple

# Third Party
from kfp import dsl

# Local
from .components import (
    generate_taxonomy,
    mix_taxonomy_datasets,
    postprocess_taxonomy,
    preprocess_taxonomy,
    taxonomy_git_importer,
)


@dsl.pipeline
def e2e_pipeline(
    taxonomy_repo: str,
    pipeline: str,
    teacher_model_path: str,
) -> NamedTuple(  # type: ignore
    "outputs",
    [
        ("taxonomy_path", dsl.Artifact),
        ("preprocessed_path", dsl.Artifact),
        ("generated_path", dsl.Artifact),
        ("postprocessed_path", dsl.Artifact),
        ("mixed_skills", dsl.Dataset),
        ("mixed_knowledge", dsl.Dataset),
    ],
):
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")

    # TODO: Figure out if we can use kfp dsl.importer instead
    # of our own simplistic importer here
    #
    # taxonomy_importer = dsl.importer(
    #     artifact_uri=taxonomy_path,
    #     artifact_class=dsl.Artifact,
    #     reimport=True,
    # )
    taxonomy_importer = taxonomy_git_importer(
        taxonomy_repo=taxonomy_repo,
    )

    preprocess_task = preprocess_taxonomy(
        taxonomy_path=taxonomy_importer.output,
        teacher_model_path=teacher_model_path,
        taxonomy_base="empty",
    )

    generate_task = generate_taxonomy(
        preprocessed_path=preprocess_task.output,
        pipeline=pipeline,
        model_id="mock",
        num_cpus=1,  # Test is faster running on a single CPU vs forking
        batch_size=0,  # Disable batch for tiny dataset and fastest test
    )

    postprocess_task = postprocess_taxonomy(
        generated_path=generate_task.output,
        date_suffix=date_suffix,
        pipeline=pipeline,
    )

    mix_task = mix_taxonomy_datasets(
        postprocessed_path=postprocess_task.output,
        date_suffix=date_suffix,
    )

    outputs = namedtuple(
        "outputs",
        [
            "taxonomy_path",
            "preprocessed_path",
            "generated_path",
            "postprocessed_path",
            "mixed_skills",
            "mixed_knowledge",
        ],
    )
    return outputs(
        taxonomy_importer.output,
        preprocess_task.output,
        generate_task.output,
        postprocess_task.output,
        mix_task.outputs["mixed_skills"],
        mix_task.outputs["mixed_knowledge"],
    )
