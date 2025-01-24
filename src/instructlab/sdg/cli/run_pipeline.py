# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import os

# Third Party
from datasets import Dataset
import openai

# First Party
from instructlab.sdg.pipeline import Pipeline, PipelineContext
from instructlab.sdg.utils.json import jldump, jlload
from instructlab.sdg.utils.logging import setup_logger

if __name__ == "__main__":
    # Standard
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a synthetic data generation pipeline."
    )

    # Required args
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Path to the yaml file of the pipeline to execute.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input jsonl file containing samples used for data generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the generated samples to, in jsonl format.",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL endpoint of an OpenAI-compatible API server running your teacher model.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="mixtral",
        help="Model family of your teacher model. Valid values are granite, merlinite, mistral, or mixtral.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="The id of the teacher model to use, as recognized by your OpenAI-compatible API server.",
    )

    # Optional args
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logger(args.log_level)
    client = openai.OpenAI(base_url=args.endpoint_url, api_key=args.api_key)
    # TODO: Remove num_instructions_to_generate hardcode of 30 here,
    # but first we need to remove it as a required parameter of the
    # PipelineContext generally.
    #
    # https://github.com/instructlab/sdg/issues/491
    pipeline_context = PipelineContext(client, args.model_family, args.model_id, 30)
    pipeline_path = Path(args.pipeline).absolute()
    pipeline = Pipeline.from_file(pipeline_context, pipeline_path)
    input_path = Path(args.input).absolute()
    input_ds = Dataset.from_list(jlload(str(input_path)))
    output_ds = pipeline.generate(input_ds)
    output_path = Path(args.output).absolute()
    jldump(output_ds, str(output_path))
