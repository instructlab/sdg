# Standard
from datetime import datetime
from multiprocessing import set_start_method
import argparse
import logging
import os
import sys

# First Party
from instructlab.sdg.subset_selection import subset_datasets


def setup_logging(log_level="INFO", log_file=None):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Reset the root logger by removing all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger to INFO by default (to avoid excessive logging from third-party libraries)
    root_logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    app_loggers = [
        "instructlab",
        "scripts",
        "__main__",
    ]

    for logger_name in app_loggers:
        logging.getLogger(logger_name).setLevel(numeric_level)

    # suppress noisy libraries
    noisy_loggers = ["matplotlib", "PIL", "submodlib", "transformers", "torch", "numpy"]

    for logger_name in noisy_loggers:
        if logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def parse_size(value):
    """Parse a size value that can be either a float (percentage) or an int (absolute count)."""
    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size value: {value}. Must be a number."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run subset selection on datasets using facility location method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="One or more input files (space-separated) to process",
    )

    parser.add_argument(
        "--output_dir", required=True, help="Directory where output files will be saved"
    )

    parser.add_argument(
        "--subset_sizes",
        nargs="+",
        type=parse_size,
        required=True,
        help="One or more subset sizes (space-separated). Values between 0-1 represent percentages, integers represent absolute counts",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--num_folds",
        type=int,
        default=50,
        help="Number of folds to use for subset selection (launches separate processes for each fold)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size for processing embeddings",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. If not specified, all available GPUs will be used, if specified more than available, max available will be used",
    )

    parser.add_argument(
        "--encoder_type",
        default="arctic",
        help="Type of encoder to use for generating embeddings",
    )

    parser.add_argument(
        "--encoder_model",
        default="Snowflake/snowflake-arctic-embed-l-v2.0",
        help="Model to use for generating embeddings, please download using ilab model download prior to using",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=160.0,
        help="Epsilon parameter for the LazierThanLazyGreedy optimizer. Default is optimized for datasets >100k samples. For smaller datasets, use smaller values (starting from 0.1)",
    )

    parser.add_argument(
        "--template_name",
        default="conversation",
        help="Template name to use for formatting examples. Options: default, conversation, qa",
    )

    parser.add_argument(
        "--testing_mode",
        action="store_true",
        help="Run in testing mode (limited computation), not for actual use",
    )

    parser.add_argument(
        "--combine_files",
        action="store_true",
        help="Combine all input files into a single dataset",
    )

    # Logging arguments
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory to store log files. If not specified, logs will only be printed to console",
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_start_method("spawn")
    args = parse_args()

    # Setup logging
    output_log_file = None
    if args.log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"subset_selection_{timestamp}.log"
        output_log_file = os.path.join(args.log_dir, log_filename)
        os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logging(args.log_level, output_log_file)
    logger.info(f"Starting subset selection with arguments: {args}")

    kwargs = vars(args)

    kwargs.pop("log_dir", None)
    kwargs.pop("log_level", None)

    input_files = kwargs.pop("input_files")
    subset_sizes = kwargs.pop("subset_sizes")

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        # Run subset selection
        logger.info(
            f"Running subset selection on {input_files} with sizes {subset_sizes}"
        )
        subset_datasets(input_files=input_files, subset_sizes=subset_sizes, **kwargs)
        logger.info("Subset selection completed successfully")
    # pylint: disable=broad-exception-caught
    except Exception as e:
        logger.exception(f"Error during subset selection: {e}")
        sys.exit(1)
