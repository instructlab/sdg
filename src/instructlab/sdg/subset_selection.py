# Standard
from dataclasses import dataclass, field
from functools import wraps
from multiprocessing import Pool, set_start_method
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar, Union
import gc
import glob
import logging
import math
import os
import re
import time

# Third Party
from datasets import concatenate_datasets, load_dataset
from jinja2 import BaseLoader, Environment
from submodlib import FacilityLocationFunction
from tqdm import tqdm
import h5py
import numpy as np
import torch

# Local
from .encoders.arctic_encoder import ArcticEmbedEncoder
from .encoders.bge_unified_encoder import UnifiedBGEEncoder
from .utils.subset_selection_utils import compute_pairwise_dense, get_default_num_gpus

__DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Type variables
T = TypeVar("T")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BasicConfig:
    """Basic configuration parameters."""

    output_dir: str = "output"
    batch_size: int = 100000
    num_folds: int = 50
    combine_files: bool = False


@dataclass
class EncoderConfig:
    """Encoder-specific configuration parameters."""

    instruction: str = field(
        default="Generate embeddings that capture the core meaning of user-assistant conversations, ensuring the embeddings can be clustered based on semantic similarity for subset selection.",
        metadata={"advanced": True},
    )
    query_description: str = field(default="Conversation", metadata={"advanced": True})
    encoder_type: str = field(default="arctic", metadata={"advanced": True})
    encoder_model: str = field(
        default="Snowflake/snowflake-arctic-embed-l-v2.0", metadata={"advanced": True}
    )


@dataclass
class TemplateConfig:
    """Template-related configuration parameters."""

    template_name: str = field(default="conversation", metadata={"advanced": True})
    templates: Dict[str, str] = field(
        default_factory=lambda: {
            "default": "{{ text }}",
            "conversation": "{% for msg in messages if msg.role != 'system' %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}",
            "qa": "Question: {{ question }}\nAnswer: {{ answer }}",
        },
        metadata={"advanced": True},
    )


@dataclass
class SystemConfig:
    """System-related configuration parameters."""

    num_gpus: int = field(
        default_factory=get_default_num_gpus, metadata={"advanced": True}
    )
    seed: int = field(default=42, metadata={"advanced": True})
    max_retries: int = field(default=3, metadata={"advanced": True})
    retry_delay: int = field(default=30, metadata={"advanced": True})


@dataclass
class ProcessingConfig:
    """
    Configuration for subset selection with basic and advanced parameters.

    Required Parameters:
        input_files: List of input files to process
        subset_sizes: List of subset sizes - integers for absolute counts or floats for percentages

    Configuration Groups:
        basic: Basic processing parameters
        encoder: Encoder-specific parameters
        template: Template-related parameters
        system: System-related parameters
    """

    # Required parameters
    input_files: List[str]
    subset_sizes: List[Union[int, float]]

    # Configuration groups
    basic: BasicConfig = field(default_factory=BasicConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.input_files:
            raise ValueError("input_files cannot be empty")

        if not isinstance(self.subset_sizes, list):
            raise ValueError("subset_sizes must be a list")

        for size in self.subset_sizes:
            if not isinstance(size, (int, float)):
                raise ValueError("subset_sizes must contain only integers or floats")
            if isinstance(size, float) and not 0 < size <= 100:
                raise ValueError(
                    "Percentage values in subset_sizes must be between 0 and 100"
                )
            if isinstance(size, int) and size <= 0:
                raise ValueError("Absolute values in subset_sizes must be positive")


def retry_on_exception(func):
    """
    Decorator to retry a function upon exception up to a maximum number of retries.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        last_exception = None
        for attempt in range(self.config.system.max_retries):
            try:
                return func(self, *args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                # Happens when GPU runs out of memory during batch processing
                last_exception = e
                logger.error(f"GPU out of memory on attempt {attempt + 1}: {str(e)}")
            except RuntimeError as e:
                # Common PyTorch errors (including some OOM errors and model issues)
                last_exception = e
                logger.error(
                    f"PyTorch runtime error on attempt {attempt + 1}: {str(e)}"
                )
            except ValueError as e:
                # From tokenizer or input validation
                last_exception = e
                logger.error(f"Value error on attempt {attempt + 1}: {str(e)}")
            except TypeError as e:
                # From incorrect input types or model parameter mismatches
                last_exception = e
                logger.error(f"Type error on attempt {attempt + 1}: {str(e)}")
            except IndexError as e:
                # Possible during tensor operations or batch processing
                last_exception = e
                logger.error(f"Index error on attempt {attempt + 1}: {str(e)}")

            if attempt < self.config.system.max_retries - 1:
                logger.info(f"Retrying in {self.config.system.retry_delay} seconds...")
                time.sleep(self.config.system.retry_delay)
                gc.collect()
                torch.cuda.empty_cache()

        raise last_exception

    return wrapper


class DataProcessor:
    """
    Enhanced data processor with support for combined files and multiple selection methods.
    """

    def __init__(self, config: ProcessingConfig, encoder_cls):
        """
        Initializes the DataProcessor with the given configuration and encoder class.

        Args:
            config (ProcessingConfig): The processing configuration.
            encoder_cls: The encoder class to use for generating embeddings.
        """
        self.config = config
        self.encoder = encoder_cls(model_name=config.encoder.encoder_model)
        self.env = Environment(loader=BaseLoader())
        self.templates = {
            k: self.env.from_string(v) for k, v in config.template.templates.items()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        np.random.seed(config.system.seed)
        torch.manual_seed(config.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.system.seed)

    def format_text(self, example: Dict[str, Any], format_type: str) -> str:
        """
        Formats the text of an example using the specified template.

        Args:
            example (Dict[str, Any]): The data example to format.
            format_type (str): The key of the template to use.

        Returns:
            str: The formatted text.
        """
        template = self.templates.get(format_type)
        if not template:
            raise ValueError(f"Unknown format type: {format_type}")
        return template.render(**example)

    def load_and_combine_datasets(self, input_files: List[str]) -> Any:
        """
        Load and optionally combine multiple datasets.

        Args:
            input_files (List[str]): List of input file paths.

        Returns:
            Combined dataset or list of individual datasets.
        """
        datasets = []

        for input_file in input_files:
            file_extension = input_file.split(".")[-1]
            if file_extension == "jsonl":
                file_extension = "json"
            dataset = load_dataset(
                file_extension, data_files=input_file, split="train", cache_dir=None
            )
            datasets.append(dataset)

        if self.config.basic.combine_files:
            logger.info("Combining datasets...")
            return concatenate_datasets(datasets)

        if len(datasets) > 1:
            raise ValueError(
                "Multiple datasets provided but combine_files is not enabled"
            )
        return datasets[0]

    def calculate_subset_size(
        self, total_samples: int, size_spec: Union[int, float]
    ) -> int:
        """
        Calculate the actual subset size based on the specification.

        Args:
            total_samples (int): Total number of samples in the dataset.
            size_spec (Union[int, float]): Size specification (percentage if float, absolute if int).

        Returns:
            int: Actual number of samples to select.
        """
        if isinstance(size_spec, float):
            # Treat as percentage
            return max(1, int(size_spec / 100 * total_samples))
        # Treat as absolute number
        return min(size_spec, total_samples)

    def get_subset_name(self, size_spec: Union[int, float], actual_size: int) -> str:
        """
        Generate appropriate subset name based on selection method.

        Args:
            size_spec (Union[int, float]): Original size specification.
            actual_size (int): Actual number of samples selected.

        Returns:
            str: Descriptive name for the subset.
        """
        if isinstance(size_spec, float):
            return f"percent_{size_spec:.1f}"
        return f"samples_{actual_size}"

    def get_last_processed_batch(self, output_dir: str) -> Tuple[int, Optional[str]]:
        """
        Retrieves the last processed batch number and its file path from the output directory.

        Args:
            output_dir (str): The directory where batch files are stored.

        Returns:
            Tuple[int, Optional[str]]: The last batch number and the corresponding batch file path.
        """
        batch_files = glob.glob(os.path.join(output_dir, "batch_*.h5"))
        if not batch_files:
            return -1, None

        # Sort batch files by batch number
        batch_files.sort(key=self.extract_batch_number)
        max_batch_file = batch_files[-1]
        max_batch_number = self.extract_batch_number(max_batch_file)

        # Return the max batch number and the corresponding batch file path
        return max_batch_number, max_batch_file

    @retry_on_exception
    def process_batch(self, batch_texts: List[str], output_file: str) -> Optional[int]:
        """
        Processes a batch of texts by generating embeddings and saving them to a file.
        Returns the embedding dimension or None if no embeddings were generated.
        """
        embeddings = (
            self.encoder.encode(
                inputs=batch_texts,
                instruction=self.config.encoder.instruction,
                query_description=self.config.encoder.query_description,
            )
            .cpu()
            .numpy()
        )

        if embeddings.size == 0:
            logger.warning(
                f"No embeddings generated for batch, skipping file {output_file}"
            )
            return None

        embedding_dim = int(embeddings.shape[1])  # Cast to int
        logger.info(f"Embedding dimension for batch: {embedding_dim}")

        with h5py.File(output_file, "w") as h5f:
            h5f.create_dataset(
                "embeddings", data=embeddings, dtype="float32", chunks=True
            )
            h5f.flush()

        return embedding_dim

    @retry_on_exception
    def generate_embeddings(self, dataset, output_dir: str) -> str:
        """
        Generates embeddings for the dataset and saves them to the output directory.

        Args:
            dataset: The dataset to process.
            output_dir (str): The directory where embeddings will be saved.

        Returns:
            str: The path to the merged embeddings file.
        """
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, "embeddings.h5")):
            logger.info(f"Embeddings file already exists in {output_dir}, skipping")
            return os.path.join(output_dir, "embeddings.h5")
        last_batch, last_batch_file = self.get_last_processed_batch(output_dir)
        if last_batch >= 0:
            logger.info(f"Resuming from batch {last_batch} in {last_batch_file}")
        else:
            logger.info("Starting from scratch")
        batch_texts = []

        # Initialize total_processed based on last batch
        if last_batch >= 0:
            # For the last batch, we need to check the number of samples processed
            embedding_size, _ = self.get_embedding_size_dim_from_file(
                str(last_batch_file)
            )
            total_processed = (
                last_batch * int(self.config.basic.batch_size) + embedding_size
            )
        else:
            total_processed = 0

        batch_number = last_batch + 1

        # Initialize progress bar
        progress_bar = tqdm(
            desc="Generating embeddings",
            initial=total_processed,
            unit=" samples",
            total=len(dataset),
        )

        # Iterate over dataset examples
        for i, example in enumerate(dataset):
            if i < total_processed:
                continue  # Skip already processed samples

            text = self.format_text(
                example, format_type=self.config.template.template_name
            )
            if i < 5:
                logger.info(f"Example {i + 1}: {text}")
            batch_texts.append(text)

            if len(batch_texts) == self.config.basic.batch_size:
                # Process batch
                batch_file = os.path.join(output_dir, f"batch_{batch_number}.h5")
                self.process_batch(batch_texts, batch_file)
                total_processed += len(batch_texts)
                progress_bar.update(len(batch_texts))
                batch_texts = []
                batch_number += 1
                gc.collect()
                torch.cuda.empty_cache()

        # Process any remaining texts in the final batch
        if batch_texts:
            batch_file = os.path.join(output_dir, f"batch_{batch_number}.h5")
            self.process_batch(batch_texts, batch_file)
            total_processed += len(batch_texts)
            progress_bar.update(len(batch_texts))

        progress_bar.close()

        # Merge all batch embeddings into a single file
        merged_file = os.path.join(output_dir, "embeddings.h5")
        self.merge_embeddings(output_dir, merged_file, total_samples=total_processed)
        return merged_file

    def extract_batch_number(self, filename):
        """
        Extracts the batch number from the filename.
        Assumes the filename is in the format 'batch_<number>.h5'.

        Args:
            filename (str): The filename from which to extract the batch number.

        Returns:
            int: The batch number extracted from the filename.
        """
        basename = os.path.basename(filename)
        match = re.search(r"batch_(\d+)\.h5$", basename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Filename {filename} does not match expected pattern.")

    def get_embedding_size_dim_from_file(self, batch_file: str) -> Tuple[int, int]:
        """
        Reads the batch file to determine the embedding size (number of embeddings) and dimension.
        """
        with h5py.File(batch_file, "r") as h5f:
            if "embeddings" not in h5f:
                raise ValueError(
                    f"The file {batch_file} does not contain 'embeddings' dataset."
                )
            embeddings = h5f["embeddings"]
            embedding_size = int(embeddings.shape[0])  # Cast to int
            embedding_dim = int(embeddings.shape[1])  # Cast to int
            logger.info(f"Embedding dimension from {batch_file}: {embedding_dim}")
        return embedding_size, embedding_dim

    def merge_embeddings(self, output_dir, merged_file, total_samples):
        """
        Merges all batch embedding files into a single embeddings file.

        Args:
            output_dir (str): The directory where batch embedding files are stored.
            merged_file (str): The path to the merged embeddings file.
            total_samples (int): The total number of samples (embeddings).

        """
        # Find all batch files
        batch_files = glob.glob(os.path.join(output_dir, "batch_*.h5"))
        if not batch_files:
            logger.warning("No batch files found to merge")
            return

        # Sort batch files by batch number
        batch_files.sort(key=self.extract_batch_number)

        # Retrieve embedding_dim from the first batch file
        _, embedding_dim = self.get_embedding_size_dim_from_file(batch_files[0])

        if os.path.exists(merged_file):
            logger.info(f"Merged file {merged_file} already exists, skipping merge")
            return

        logger.info(
            f"Merging {len(batch_files)} batch files into {merged_file} with {total_samples} samples"
        )

        with h5py.File(merged_file, "w") as h5f_merged:
            # Initialize the dataset in the merged file with the retrieved embedding dimension
            embeddings_ds = h5f_merged.create_dataset(
                "embeddings", shape=(total_samples, embedding_dim), dtype="float32"
            )

            start_idx = 0
            for batch_file in batch_files:
                with h5py.File(batch_file, "r") as h5f_batch:
                    if "embeddings" not in h5f_batch:
                        logger.error(
                            f"File {batch_file} does not contain 'embeddings' dataset"
                        )
                        continue

                    embeddings = h5f_batch["embeddings"][:]
                    batch_size = embeddings.shape[0]
                    end_idx = start_idx + batch_size

                    # Check that each file's embedding dimension matches the retrieved embedding_dim
                    if embeddings.shape[1] != embedding_dim:
                        logger.error(
                            f"Embedding dimension mismatch in {batch_file}. Expected {embedding_dim}, got {embeddings.shape[1]}"
                        )
                        continue

                    # Copy embeddings into the merged dataset
                    embeddings_ds[start_idx:end_idx] = embeddings
                    start_idx = end_idx

                # Remove the batch file after processing
                os.remove(batch_file)
                logger.info(f"Processed and removed {batch_file}")

            gc.collect()

    def select_subsets(
        self, dataset_name: str, embeddings: torch.Tensor
    ) -> Dict[Union[int, float], List[int]]:
        """
        Enhanced subset selection supporting both percentage and absolute size specifications.
        """
        indices = np.arange(len(embeddings))
        np.random.shuffle(indices)

        fold_size = len(embeddings) // self.config.basic.num_folds
        remainder = len(embeddings) % self.config.basic.num_folds

        folds = []
        start_idx = 0
        for i in range(self.config.basic.num_folds):
            extra = 1 if i < remainder else 0
            end_idx = start_idx + fold_size + extra
            folds.append(indices[start_idx:end_idx])
            start_idx = end_idx

        gpu_assignments = []
        folds_per_gpu = self.config.basic.num_folds // self.config.system.num_gpus
        extra_folds = self.config.basic.num_folds % self.config.system.num_gpus

        start_fold = 0
        for gpu_id in range(self.config.system.num_gpus):
            num_folds_this_gpu = folds_per_gpu + (1 if gpu_id < extra_folds else 0)
            end_fold = start_fold + num_folds_this_gpu
            gpu_folds_info = [
                (fold_idx, folds[fold_idx]) for fold_idx in range(start_fold, end_fold)
            ]

            gpu_assignments.append(
                (
                    gpu_id,
                    gpu_folds_info,
                    embeddings,
                    self.config.subset_sizes,
                    len(embeddings),  # Pass total samples for absolute size calculation
                )
            )
            start_fold = end_fold

        with Pool(processes=self.config.system.num_gpus) as pool:
            gpu_results = pool.map(process_folds_with_gpu, gpu_assignments)

        all_results = []
        for gpu_result in gpu_results:
            all_results.extend(gpu_result)

        class SubsetData(TypedDict):
            indices: List[int]
            gains: List[float]

        combined_subsets: Dict[Union[int, float], SubsetData] = {
            size: {"indices": [], "gains": []} for size in self.config.subset_sizes
        }

        for fold_idx, result in all_results:
            for size in self.config.subset_sizes:
                combined_subsets[size]["indices"].extend(result[size]["indices"])
                combined_subsets[size]["gains"].extend(result[size]["gains"])

        base_name = dataset_name
        subsets = {}

        for size_spec in self.config.subset_sizes:
            actual_size = self.calculate_subset_size(len(embeddings), size_spec)
            sorted_indices_gains = sorted(
                zip(
                    combined_subsets[size_spec]["indices"],
                    combined_subsets[size_spec]["gains"],
                ),
                key=lambda x: x[1],
                reverse=True,
            )[:actual_size]  # Limit to actual_size

            sorted_indices = [x[0] for x in sorted_indices_gains]
            sorted_gains = [x[1] for x in sorted_indices_gains]

            subset_name = self.get_subset_name(size_spec, actual_size)
            metadata_file = os.path.join(
                self.config.basic.output_dir,
                f"{base_name}_fl_{self.config.basic.num_folds}_partitions_{subset_name}_metadata.npz",
            )

            np.savez(metadata_file, indices=sorted_indices, gains=sorted_gains)
            logger.info(f"Saved metadata to {metadata_file}")
            subsets[size_spec] = sorted_indices

        return subsets

    def get_dataset_name(self, input_file: str) -> str:
        """
        Get a clean dataset name from the input file path.

        Args:
            input_file (str): Input file path

        Returns:
            str: Clean dataset name
        """
        # Extract filename without extension and path
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # Clean the name to make it filesystem-friendly
        clean_name = re.sub(r"[^\w\-_]", "_", base_name)
        return clean_name

    def process_files(self, input_files: List[str], output_dir: str):
        """
        Process multiple input files with support for both combined and separate processing.

        Args:
            input_files (List[str]): List of input files to process
            output_dir (str): Output directory for results
        """
        try:
            if self.config.basic.combine_files:
                # Process combined datasets
                logger.info("Processing combined datasets...")
                dataset = self.load_and_combine_datasets(input_files)
                dataset_name = "combined_dataset"

                # Process combined dataset
                self._process_single_dataset(
                    dataset, dataset_name, output_dir, input_files[0]
                )
            else:
                # Process each dataset separately
                logger.info("Processing datasets separately...")
                for input_file in input_files:
                    dataset = self.load_and_combine_datasets([input_file])
                    dataset_name = self.get_dataset_name(input_file)
                    logger.info(f"Processing dataset: {dataset_name}")
                    self._process_single_dataset(
                        dataset, dataset_name, output_dir, input_file
                    )

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise

    def _process_single_dataset(
        self, dataset, dataset_name: str, output_dir: str, input_file: str
    ):
        """
        Process a single dataset (either combined or individual).

        Args:
            dataset: The dataset to process
            dataset_name (str): Name of the dataset
            output_dir (str): Output directory
            input_file (str): Original input file path (for extension)
        """
        try:
            # Create dataset-specific output directory
            dataset_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            logger.info(f"Generating embeddings for {dataset_name}")
            embedding_file = self.generate_embeddings(
                dataset, os.path.join(dataset_output_dir, "embeddings")
            )

            logger.info("Loading embeddings for subset selection")
            with h5py.File(embedding_file, "r") as f:
                embeddings_data = f["embeddings"][:]
                if embeddings_data.size == 0:
                    logger.warning(
                        f"No embeddings generated for dataset {dataset_name}, skipping subset selection"
                    )
                    return
                embeddings = torch.tensor(embeddings_data, dtype=torch.float32)

            logger.info("Selecting subsets")
            subsets = self.select_subsets(dataset_name, embeddings)

            logger.info("Saving subsets")
            for size_spec, indices in subsets.items():
                subset_data = dataset.select(indices)
                subset_name = self.get_subset_name(size_spec, len(indices))

                # Create subset filename with dataset name
                output_file = os.path.join(
                    dataset_output_dir,
                    f"{dataset_name}_{subset_name}_subset.{input_file.split('.')[-1]}",
                )

                self._save_subset(subset_data, output_file, input_file)
                logger.info(
                    f"Saved subset with {len(indices)} samples to {output_file}"
                )

            # Clean up resources
            del dataset, embeddings
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            raise

    def _save_subset(self, subset_data, output_file: str, input_file: str):
        """
        Save subset data to file in appropriate format.

        Args:
            subset_data: The dataset subset to save
            output_file (str): Output file path
            input_file (str): Original input file path (for determining format)
        """
        extension = input_file.split(".")[-1]
        if extension in ["json", "jsonl"]:
            subset_data.to_json(output_file, orient="records", lines=True)
        elif extension == "csv":
            subset_data.to_csv(output_file, index=False)
        elif extension == "parquet":
            subset_data.to_parquet(output_file)


def process_folds_with_gpu(args):
    """
    Process folds on GPU with support for both percentage and absolute size specifications.
    """
    gpu_id, gpu_folds_info, embeddings, subset_sizes, total_samples = args
    try:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        results = []
        for fold_idx, fold_indices in gpu_folds_info:
            try:
                logger.info(f"Processing fold {fold_idx + 1} on GPU {gpu_id}")

                fold_embeddings = embeddings[fold_indices].to(device)

                logger.info(f"Computing similarity matrix for fold {fold_idx + 1}")
                max_sim_mat = compute_pairwise_dense(
                    fold_embeddings,
                    batch_size=50,
                    metric="cosine",
                    device=device,
                    scaling="additive",
                )
                similarity_matrix = max_sim_mat.cpu().numpy()

                subsets = {}
                ds_func = FacilityLocationFunction(
                    n=similarity_matrix.shape[0],
                    sijs=similarity_matrix,
                    mode="dense",
                    separate_rep=False,
                )

                for size_spec in subset_sizes:
                    if isinstance(size_spec, float):
                        # Percentage-based selection
                        budget = max(
                            1, math.ceil((size_spec / 100) * similarity_matrix.shape[0])
                        )
                    else:
                        # Absolute number-based selection
                        budget = max(
                            1,
                            math.ceil(
                                size_spec * (similarity_matrix.shape[0] / total_samples)
                            ),
                        )

                    logger.info(
                        f"Selecting subset of size {budget} for fold {fold_idx + 1}"
                    )

                    subset_result = ds_func.maximize(
                        budget=budget,
                        optimizer="LazierThanLazyGreedy",
                        epsilon=0.1,
                        stopIfZeroGain=False,
                        stopIfNegativeGain=False,
                        verbose=False,
                    )

                    subset_indices = [fold_indices[x[0]] for x in subset_result]
                    subset_gains = [x[1] for x in subset_result]
                    subsets[size_spec] = {
                        "indices": subset_indices,
                        "gains": subset_gains,
                    }

                results.append((fold_idx, subsets))

            except Exception as e:
                logger.error(
                    f"Error processing fold {fold_idx + 1} on GPU {gpu_id}: {str(e)}"
                )
                raise
            finally:
                for var in ["ds_func", "similarity_matrix", "fold_embeddings"]:
                    if var in locals():
                        del locals()[var]
                gc.collect()
                torch.cuda.empty_cache()

        return results
    except Exception as e:
        logger.error(f"Error in process_folds_with_gpu on GPU {gpu_id}: {str(e)}")
        raise


def subset_datasets(
    input_files: List[str], subset_sizes: List[Union[int, float]], **kwargs: Any
) -> None:
    """Create subsets of datasets using facility location for diverse subset selection."""

    # Get system's available GPU count
    available_gpus = get_default_num_gpus()

    # Create configuration groups
    basic_config = BasicConfig()
    encoder_config = EncoderConfig()
    template_config = TemplateConfig()
    system_config = SystemConfig()

    # Update configuration groups from kwargs
    for key, value in kwargs.items():
        if hasattr(basic_config, key):
            setattr(basic_config, key, value)
        elif hasattr(encoder_config, key):
            setattr(encoder_config, key, value)
        elif hasattr(template_config, key):
            setattr(template_config, key, value)
        elif hasattr(system_config, key):
            setattr(system_config, key, value)

    # Ensure num_gpus doesn't exceed available GPUs
    if system_config.num_gpus > available_gpus:
        logger.warning(
            f"Requested {system_config.num_gpus} GPUs but only {available_gpus} available. "
            f"Falling back to using {available_gpus} GPUs."
        )
        system_config.num_gpus = available_gpus

    # Create configuration
    config = ProcessingConfig(
        input_files=input_files,
        subset_sizes=subset_sizes,
        basic=basic_config,
        encoder=encoder_config,
        template=template_config,
        system=system_config,
    )

    try:

        logger.info(f"Processing configuration: {config}")

        # Initialize data processor based on encoder type
        os.makedirs(config.basic.output_dir, exist_ok=True)

        if config.encoder.encoder_type == "bge":
            processor = DataProcessor(config, UnifiedBGEEncoder)
        elif config.encoder.encoder_type == "arctic":
            processor = DataProcessor(config, ArcticEmbedEncoder)
        else:
            raise ValueError(f"Unsupported encoder type: {config.encoder.encoder_type}")

        processor.process_files(input_files, config.basic.output_dir)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
