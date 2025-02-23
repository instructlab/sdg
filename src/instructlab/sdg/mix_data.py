#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging
import os
import glob

from instructlab.sdg.datamixing import DataMixer, Recipe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mix generated data using DataMixer")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="List of data directories containing node datasets",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where mixed data and recipes will be saved",
    )
    parser.add_argument(
        "--sys-prompt", default="", help="System prompt to use in the messages"
    )
    parser.add_argument(
        "--date-suffix",
        default=datetime.datetime.now().strftime("%Y%m%d"),
        help="Date suffix for output files (default: YYYYMMDD)",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="Number of processes to use for parallel processing",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create recipes from node datasets
    knowledge_recipe = Recipe(sys_prompt=args.sys_prompt)
    skills_recipe = Recipe(sys_prompt=args.sys_prompt)

    for data_dir in args.data_dirs:
        node_datasets_dir = os.path.join(data_dir, "node_datasets_*")
        for node_dir in glob.glob(node_datasets_dir):
            # Find phase07 files for knowledge recipe
            p07_files = glob.glob(os.path.join(node_dir, "*_p07.jsonl"))
            for p07_file in p07_files:
                knowledge_recipe.add_dataset(p07_file, 1.0)
                logger.info(f"Added {p07_file} to knowledge recipe")

            # Find phase10 files for skills recipe
            p10_files = glob.glob(os.path.join(node_dir, "*_p10.jsonl"))
            for p10_file in p10_files:
                skills_recipe.add_dataset(p10_file, 1.0)
                logger.info(f"Added {p10_file} to skills recipe")

    # Initialize DataMixer with our recipes
    mixer = DataMixer(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        date_suffix=args.date_suffix,
        sys_prompt=args.sys_prompt,
        num_procs=args.num_procs,
    )

    # Set our recipes
    mixer.knowledge_recipe = knowledge_recipe
    mixer.skills_recipe = skills_recipe

    # Generate mixed datasets
    logger.info("Starting data mixing process...")
    mixer.generate()
    logger.info("Data mixing complete!")


if __name__ == "__main__":
    main()
