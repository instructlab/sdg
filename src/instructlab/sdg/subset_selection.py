# Standard
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar, Union
import gc
import glob
import importlib
import logging
import math
import os
import re
import sys

# Third Party
from datasets import concatenate_datasets, load_dataset
from jinja2 import BaseLoader, Environment
from tqdm import tqdm
import h5py
import numpy as np
import torch

# Local
from .utils.subset_selection_utils import (
    compute_pairwise_dense,
    get_default_num_gpus,
    retry_on_exception,
)

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
    epsilon: float = field(
        default=160.0,
        metadata={
            "advanced": True,
            "help": "Epsilon parameter for the LazierThanLazyGreedy optimizer in facility location maximization. "
            "Default of 160.0 is optimized for datasets >100k samples. "
            "For smaller datasets, consider using much smaller values (starting from 0.1).",
        },
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.epsilon <= 160:
            raise ValueError("epsilon must be between 0 and 160")

    def validate_epsilon_for_dataset_size(self, dataset_size: int) -> None:
        """
        Validate epsilon parameter based on dataset size and provide appropriate warnings.

        Args:
            dataset_size (int): Size of the dataset being processed
        """
        if dataset_size < 100000:
            logger.warning(
                "Subset selection is highly recommended to be used only with dataset sizes over 100k samples. "
                f"Your dataset has {dataset_size:,} samples."
            )
            if self.epsilon > 1.0:
                logger.warning(
                    f"Current epsilon value ({self.epsilon}) may be too high for a dataset of this size. "
                    "For smaller datasets, consider using much smaller values (starting from 0.1) "
                    "to ensure proper subset selection."
                )


@dataclass
class EncoderConfig:
    """Encoder-specific configuration parameters."""

    instruction: str = field(
        default="Generate embeddings that capture the core meaning of user-assistant conversations, ensuring the embeddings can be clustered based on semantic similarity for subset selection.",
        metadata={"advanced": True},
    )
    encoder_type: str = field(default="arctic", metadata={"advanced": True})
    encoder_model: str = field(
        default="Snowflake/snowflake-arctic-embed-l-v2.0", metadata={"advanced": True}
    )
    testing_mode: bool = False


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

    num_gpus: int = field(init=False)  # Don't initialize in __init__
    seed: int = field(default=42, metadata={"advanced": True})
    max_retries: int = field(default=3, metadata={"advanced": True})
    retry_delay: int = field(default=30, metadata={"advanced": True})
    testing_mode: bool = field(default=False, metadata={"advanced": True})

    def __post_init__(self):
        """Initialize num_gpus after other fields are set."""
        self.num_gpus = get_default_num_gpus(testing_mode=self.testing_mode)


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
        self.encoder = encoder_cls(
            model_name=config.encoder.encoder_model,
            testing_mode=config.encoder.testing_mode,
        )
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
            # if not in range of 0 to 1, raise error
            if size_spec <= 0 or size_spec > 1:
                raise ValueError(
                    "Percentage values must be between 0(non-inclusive) and 1(inclusive)"
                )
            # If between 0 and 1, treat as decimal percentage (0.5 = 50%)
            return max(1, int((size_spec) * total_samples))
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
        Generates embeddings for the dataset and saves them to the output directory, using multiple GPUs in parallel.

        Args:
            dataset: The dataset to process.
            output_dir (str): The directory where embeddings will be saved.

        Returns:
            str: The path to the merged embeddings file.
        """
        os.makedirs(output_dir, exist_ok=True)
        merged_path = os.path.join(output_dir, "embeddings.h5")

        # If embeddings already exist, return early
        if os.path.exists(merged_path):
            logger.info(f"Embeddings file already exists in {output_dir}, skipping")
            return merged_path

        # Get number of GPUs to use
        num_gpus = min(self.config.system.num_gpus, torch.cuda.device_count())
        logger.info(f"Using {num_gpus} GPUs for embedding generation")

        # Create dataset shards - one per GPU
        total_samples = len(dataset)
        per_gpu_samples = (total_samples + num_gpus - 1) // num_gpus  # Ceiling division

        # Prepare arguments for parallel processing
        args_list = []
        for gpu_id in range(num_gpus):
            # Calculate start and end indices for this shard
            start_idx = gpu_id * per_gpu_samples
            end_idx = min(start_idx + per_gpu_samples, total_samples)

            if start_idx >= total_samples:
                continue  # Skip if this GPU has no data to process

            # Create arguments for this GPU
            args_list.append(
                (
                    gpu_id,
                    dataset.select(range(start_idx, end_idx)),
                    output_dir,
                    self.config.encoder.encoder_type,
                    self.config.encoder.encoder_model,
                    self.config.encoder.instruction,
                    self.config.template.template_name,
                    self.config.template.templates,
                    self.config.basic.batch_size,
                    self.config.encoder.testing_mode,
                )
            )

        # Process dataset shards in parallel
        with Pool(processes=num_gpus) as pool:
            shard_files = pool.map(_process_dataset_shard, args_list)

        # Filter out None values (failed shards)
        shard_files = [f for f in shard_files if f is not None]

        if not shard_files:
            raise ValueError("No embeddings were generated from any GPU")

        # Merge all shard files
        _merge_shard_files(shard_files, merged_path)

        return merged_path

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
                    self.config.basic.epsilon,
                    self.config.system.testing_mode,  # Explicitly pass testing_mode
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
            logger.info(f"Actual subset size: {actual_size}")
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
            # Validate epsilon based on dataset size
            self.config.basic.validate_epsilon_for_dataset_size(len(dataset))

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


def _process_dataset_shard(args):
    """Process a dataset shard on a specific GPU."""
    (
        gpu_id,
        dataset_shard,
        output_dir,
        encoder_type,
        encoder_model,
        instruction,
        template_name,
        templates,
        batch_size,
        testing_mode,
    ) = args

    try:
        # Set the GPU for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        logger.info(f"GPU {gpu_id} started processing {len(dataset_shard)} samples")

        # Import the encoder directly using the system path
        # Standard

        sys.path.append(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            )
        )

        # Import the encoder class using string-based absolute import

        module_name = f"sdg.src.instructlab.sdg.encoders.{encoder_type}_encoder"
        module = importlib.import_module(module_name)
        encoder_cls = getattr(module, f"{encoder_type.capitalize()}EmbedEncoder")

        # Create encoder instance
        encoder = encoder_cls(
            model_name=encoder_model,
            device=torch.device(device),
            testing_mode=testing_mode,
        )

        # Set up Jinja environment for templating
        env = Environment(loader=BaseLoader())
        templates_dict = {k: env.from_string(v) for k, v in templates.items()}

        # Create shard-specific output directory
        shard_dir = os.path.join(output_dir, f"shard_{gpu_id}")
        os.makedirs(shard_dir, exist_ok=True)

        # Process batches
        all_embeddings = []
        batch_texts = []

        # Create progress bar
        progress_bar = tqdm(
            desc=f"GPU {gpu_id} generating embeddings",
            total=len(dataset_shard),
            unit=" samples",
            position=gpu_id,  # Stack progress bars
            leave=True,
        )

        # Process each example in the shard
        for example in dataset_shard:
            # Format the text using the template
            template = templates_dict.get(template_name)
            if not template:
                raise ValueError(f"Unknown format type: {template_name}")

            text = template.render(**example)
            batch_texts.append(text)

            # Process when batch is full or at the end
            if len(batch_texts) == batch_size or example == dataset_shard[-1]:
                # Generate embeddings for this batch
                with torch.no_grad():
                    batch_embeddings = (
                        encoder.encode(
                            inputs=batch_texts,
                            instruction=instruction,
                        )
                        .cpu()
                        .numpy()
                    )

                all_embeddings.append(batch_embeddings)
                progress_bar.update(len(batch_texts))
                batch_texts = []

                # Clean up GPU memory
                torch.cuda.empty_cache()

        progress_bar.close()

        # Concatenate all batches
        if not all_embeddings:
            logger.warning(f"No embeddings generated for shard on GPU {gpu_id}")
            return None

        embeddings = np.concatenate(all_embeddings, axis=0)

        # Save embeddings to file
        shard_file = os.path.join(shard_dir, f"embeddings_shard_{gpu_id}.h5")
        with h5py.File(shard_file, "w") as h5f:
            h5f.create_dataset("embeddings", data=embeddings, dtype="float32")

        logger.info(f"GPU {gpu_id} completed processing. Saved to {shard_file}")
        return shard_file

    # pylint: disable=broad-exception-caught
    except Exception as e:
        logger.error(f"Error processing shard on GPU {gpu_id}: {str(e)}")
        return None


def _merge_shard_files(shard_files, merged_file):
    """
    Merge all shard files into a single embeddings file.
    """
    logger.info(f"Merging {len(shard_files)} shard files into {merged_file}")

    # Get the shape and type of embeddings from the first shard
    with h5py.File(shard_files[0], "r") as f:
        first_embeddings = f["embeddings"]
        embedding_dim = first_embeddings.shape[1]
        dtype = first_embeddings.dtype

    # Count total samples across all shards
    total_samples = 0
    for shard_file in shard_files:
        with h5py.File(shard_file, "r") as f:
            total_samples += f["embeddings"].shape[0]

    # Create the merged file
    with h5py.File(merged_file, "w") as merged_f:
        merged_dataset = merged_f.create_dataset(
            "embeddings", shape=(total_samples, embedding_dim), dtype=dtype
        )

        # Copy embeddings from each shard
        start_idx = 0
        for shard_file in shard_files:
            with h5py.File(shard_file, "r") as shard_f:
                embeddings = shard_f["embeddings"][:]
                end_idx = start_idx + embeddings.shape[0]
                merged_dataset[start_idx:end_idx] = embeddings
                start_idx = end_idx

            # Remove shard file after merging
            os.remove(shard_file)
            # Remove shard directory if empty
            shard_dir = os.path.dirname(shard_file)
            if not os.listdir(shard_dir):
                os.rmdir(shard_dir)

    logger.info(
        f"Successfully merged embeddings from {len(shard_files)} GPUs with {total_samples} total samples"
    )


def process_folds_with_gpu(args):
    """
    Process folds on GPU or CPU with support for both percentage and absolute size specifications.
    """
    (
        gpu_id,
        gpu_folds_info,
        embeddings,
        subset_sizes,
        total_samples,
        epsilon,
        testing_mode,
    ) = args

    # Third Party
    # pylint: disable=import-error, import-outside-toplevel
    from submodlib import FacilityLocationFunction

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            if not testing_mode:
                raise RuntimeError("GPU processing required but CUDA is not available")
            logger.warning(
                "Running in CPU mode for testing. Production use requires GPU acceleration."
            )
            device = "cpu"

        results = []
        for fold_idx, fold_indices in gpu_folds_info:
            try:
                logger.info(f"Processing fold {fold_idx + 1} on GPU {gpu_id}")

                fold_embeddings = embeddings[fold_indices].to(device)

                logger.info(f"Computing similarity matrix for fold {fold_idx + 1}")
                max_sim_mat = compute_pairwise_dense(
                    fold_embeddings,
                    batch_size=50000,
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
                            1, math.ceil(size_spec * similarity_matrix.shape[0])
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
                        epsilon=epsilon,
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


def get_supported_encoders():
    """Get list of supported encoder types from the .encoders directory."""
    encoders_dir = os.path.join(os.path.dirname(__file__), "encoders")
    encoder_files = glob.glob(os.path.join(encoders_dir, "*_encoder.py"))
    return [
        os.path.basename(f).replace("_encoder.py", "")
        for f in encoder_files
        if not os.path.basename(f).startswith("__")
    ]


def get_encoder_class(encoder_type: str):
    """Get the encoder class based on the encoder type."""
    try:
        # Convert encoder_type to class name (e.g., 'arctic' -> 'ArcticEmbedEncoder')
        class_name = f"{encoder_type.capitalize()}EmbedEncoder"
        # Import the module dynamically
        module = __import__(
            f"instructlab.sdg.encoders.{encoder_type}_encoder", fromlist=[class_name]
        )
        # Get the class from the module
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        supported_encoders = get_supported_encoders()
        raise ValueError(
            f"Unsupported encoder type: '{encoder_type}'. "
            f"Supported types are: {[f'{t}' for t in supported_encoders]}"
        ) from e


def subset_datasets(
    input_files: List[str],
    subset_sizes: List[Union[int, float]],
    testing_mode: bool = False,
    **kwargs: Any,
) -> None:
    """Create subsets of datasets using facility location for diverse subset selection."""

    # Get system's available GPU count
    available_gpus = get_default_num_gpus(testing_mode=testing_mode)

    # Create configuration groups
    basic_config = BasicConfig()
    encoder_config = EncoderConfig(testing_mode=testing_mode)
    template_config = TemplateConfig()
    system_config = SystemConfig(testing_mode=testing_mode)

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
        processor = DataProcessor(
            config, get_encoder_class(config.encoder.encoder_type)
        )
        processor.process_files(input_files, config.basic.output_dir)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
