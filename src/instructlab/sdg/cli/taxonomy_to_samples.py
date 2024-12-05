# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# First Party
from instructlab.sdg.taxonomy import (
    DEFAULT_CHUNK_WORD_COUNT,
    DEFAULT_SERVER_CTX_SIZE,
    DEFAULT_TAXONOMY_BASE,
    taxonomy_to_samples,
)
from instructlab.sdg.utils.logging import setup_logger

if __name__ == "__main__":
    # Standard
    import argparse

    parser = argparse.ArgumentParser(
        description="Turn a taxonomy into json samples suitable for use as input to data generate pipelines"
    )

    # Required args
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the processed dataset samples into",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        required=True,
        help="Path to your InstructLab taxonomy",
    )

    # Optional args
    parser.add_argument(
        "--chunk-word-count",
        type=int,
        default=DEFAULT_CHUNK_WORD_COUNT,
        help="Number of words per document chunk",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--server-ctx-size",
        type=int,
        default=DEFAULT_SERVER_CTX_SIZE,
        help="The maximum number of tokens the inference server can handle.",
    )
    parser.add_argument(
        "--taxonomy-base",
        type=str,
        default=DEFAULT_TAXONOMY_BASE,
        help="Taxonomy based used to determine what has changed - defaults to 'empty' which means consider all the taxonomy files as changed and process all of them",
    )
    parser.add_argument(
        "--yaml-rules",
        type=str,
        default=None,
        help="Path to custom rules file for YAML linting",
    )

    args = parser.parse_args()
    setup_logger(args.log_level)
    taxonomy_to_samples(
        args.taxonomy_path,
        args.output_dir,
        chunk_word_count=args.chunk_word_count,
        server_ctx_size=args.server_ctx_size,
        taxonomy_base=args.taxonomy_base,
        yaml_rules=args.yaml_rules,
    )

"""
python -m instructlab.sdg.cli.taxonomy_to_samples --taxonomy-path /path/to/my/taxonomy --output-dir /path/to/my/output
"""
