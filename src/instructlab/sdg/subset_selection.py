# Standard
from dataclasses import dataclass, field
from functools import wraps
from multiprocessing import Pool, set_start_method
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import glob
import json
import logging
import math
import os
import re
import time

# Third Party
from datasets import concatenate_datasets, load_dataset
from jinja2 import BaseLoader, Environment
from submodlib import FacilityLocationFunction
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import h5py
import numpy as np
import torch

__DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model-specific configurations
MODEL_CONFIGS = {
    "BAAI/bge-base-en": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-base-en-v1.5": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-large-en": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-large-en-v1.5": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 512,
        "default_instruction": "Represent this sentence for searching relevant passages:",
        "batch_size": 256,
    },
    "BAAI/bge-m3": {
        "pooling_method": "cls",
        "normalize_embeddings": True,
        "max_length": 4096,
        "default_instruction": "Use the following sentences to search for relevant passages:",
        "batch_size": 32,
    },
    "BAAI/bge-multilingual-gemma2": {
        "pooling_method": "last_token",
        "normalize_embeddings": True,
        "max_length": 4096,
        "default_instruction": "Represent this for searching:",
        "batch_size": 20,
    },
}


class UnifiedBGEEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[torch.device] = None,
        use_fp16: bool = True,
        use_default_instruction: bool = True,
    ):
        """
        Unified encoder supporting all BGE model variants with model-specific configurations.

        Args:
            model_name: Model identifier from supported BGE variants
            device: Computation device
            batch_size: Base batch size (will be adjusted for multi-GPU)
            use_fp16: Whether to use half-precision
            use_default_instruction: Whether to use the model's default instruction
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        # Load model configuration
        self.config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.use_default_instruction = use_default_instruction

        # Initialize device and basic parameters
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set up GPU configuration
        self.num_gpus = torch.cuda.device_count()
        batch_size = self.config["batch_size"]
        self.batch_size = (
            batch_size * self.num_gpus if self.num_gpus > 0 else batch_size
        )

        # Suppress tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Model configuration
        self.model.eval()
        if use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(self.device)

        # Multi-GPU setup
        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)

    def _pool_embeddings(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Model-specific pooling method"""
        if self.config["pooling_method"] == "cls":
            return hidden_states[:, 0]
        elif self.config["pooling_method"] == "mean":
            s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.config["pooling_method"] == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return hidden_states[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device), sequence_lengths
            ]

    def _prepare_inputs(
        self,
        texts: Union[str, List[str]],
        instruction: str = "",
        query_description: str = "",
    ) -> List[str]:
        """Prepare inputs with model-specific formatting"""
        if isinstance(texts, str):
            texts = [texts]

        # Use model's default instruction if enabled and no custom instruction provided
        if (
            not instruction
            and self.use_default_instruction
            and self.config["default_instruction"]
        ):
            instruction = self.config["default_instruction"]

        if instruction:
            if "bge-multilingual" in self.model_name.lower():
                texts = [
                    f"<instruct>{instruction}\n{query_description}{text}"
                    for text in texts
                ]
            elif "bge-m3" not in self.model_name.lower():
                texts = [f"{instruction} {text}" for text in texts]
        return texts

    @torch.no_grad()
    def encode(
        self,
        inputs: Union[str, List[str]],
        instruction: str = "",
        query_description: str = "",
        show_progress: bool = True,
        return_tensors: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode texts into embeddings.

        Args:
            inputs: Input text or list of texts
            instruction: Optional instruction (overrides default if provided)
            query_description: Optional query description (used for specialized instructions)
            show_progress: Whether to show progress bar
            return_tensors: Whether to return pytorch tensors (True) or numpy arrays (False)
        """
        # Input preparation
        input_was_string = isinstance(inputs, str)
        inputs = self._prepare_inputs(inputs, instruction, query_description)

        # Tokenization
        encodings = self.tokenizer(
            inputs,
            max_length=self.config["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(self.device)

        # Batch processing
        embeddings_list = []
        for i in tqdm(
            range(0, len(inputs), self.batch_size),
            disable=not show_progress or len(inputs) < 256,
        ):
            # Prepare batch
            batch = {k: v[i : i + self.batch_size] for k, v in encodings.items()}

            # Model forward pass
            outputs = self.model(**batch)
            hidden_states = outputs.last_hidden_state

            # Pool embeddings
            embeddings = self._pool_embeddings(hidden_states, batch["attention_mask"])

            # Normalize if configured
            if self.config["normalize_embeddings"]:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings_list.append(embeddings.cpu())

            # Clean up GPU memory
            del outputs, hidden_states, embeddings, batch
            torch.cuda.empty_cache()

        # Combine embeddings
        embeddings = torch.cat(embeddings_list, dim=0)

        # Handle single input case
        if input_was_string:
            embeddings = embeddings[0]

        # Return appropriate format
        if return_tensors:
            return embeddings
        return embeddings.numpy()

    def encode_queries(
        self, queries: Union[str, List[str]], instruction: str = "", **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """Specialized method for encoding queries"""
        return self.encode(queries, instruction=instruction, **kwargs)

    def encode_corpus(
        self, corpus: Union[str, List[str]], instruction: str = "", **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """Specialized method for encoding corpus documents"""
        return self.encode(corpus, instruction=instruction, **kwargs)

    def embed_dataset(
        self,
        dataset: Dataset,
        column_name: str = "text",
        instruction: str = "",
        query_description: str = "",
    ) -> Dataset:
        """Embed an entire dataset column"""
        texts = dataset[column_name]
        embeddings = self.encode(
            texts,
            instruction=instruction,
            query_description=query_description,
            return_tensors=False,
        )
        return dataset.add_column("embedding", embeddings.tolist())


def compute_pairwise_dense(
    tensor1,
    tensor2=None,
    batch_size=10000,
    metric="cosine",
    device=__DEVICE,
    scaling=None,
    kw=0.1,
):
    """
    Compute pairwise metric in batches between two sets of vectors.

    Args:
        tensor1 (Tensor): Data points for the first set (n_samples1, n_features).
        tensor2 (Tensor, optional): Data points for the second set (n_samples2, n_features).
                                    Defaults to None, which means tensor1 will be used for self-comparison.
        batch_size (int): Size of each batch for computation.
        metric (str): Metric to compute. Options are 'cosine', 'dot', and 'euclidean'.
        device (str): Device to perform computation ('cuda' or 'cpu').
        scaling (str, optional): Scaling method to apply on the results. Options are 'min-max' or 'additive'.
        kw (float, optional): Kernel width for rbf metric.
    Returns:
        Tensor: Pairwise computed metric as a tensor.

    Note:
    The function performs computations in batches to manage GPU memory usage efficiently.
    similarity measure returned is in cpu memory to save GPU memory.
    If 'tensor2' is None, the function computes the metric for 'tensor1' against itself.
    """

    assert batch_size > 0, "Batch size must be positive."

    if tensor2 is None:
        tensor2 = tensor1

    tensor1, tensor2 = tensor1.to(device), tensor2.to(device)

    n_samples1, n_samples2 = tensor1.size(0), tensor2.size(0)

    # Initialize a results matrix in the CPU memory to save GPU memory
    results = torch.zeros(n_samples1, n_samples2, device="cpu")

    # Normalizing tensors if metric is cosine for cosine similarity computation
    if metric == "cosine":
        tensor1, tensor2 = (
            F.normalize(tensor1, p=2, dim=1),
            F.normalize(tensor2, p=2, dim=1),
        )

    # Function to calculate the metric
    def calculate_metric(a, b, metric, kw):
        if metric in ["cosine", "dot"]:
            return torch.mm(a, b.T)
        elif metric == "euclidean":
            distances = torch.cdist(a, b, p=2)
            similarities = 1 / (1 + distances**2)
            return similarities
        elif metric == "rbf":
            distance = torch.cdist(a, b)
            squared_distance = distance**2
            avg_dist = torch.mean(squared_distance)
            torch.div(squared_distance, kw * avg_dist, out=squared_distance)
            torch.exp(-squared_distance, out=squared_distance)
            return squared_distance
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Process in batches
    for i in range(0, n_samples1, batch_size):
        end_i = min(i + batch_size, n_samples1)
        rows = tensor1[i:end_i]

        for j in range(0, n_samples2, batch_size):
            end_j = min(j + batch_size, n_samples2)
            cols = tensor2[j:end_j]

            batch_results = calculate_metric(rows, cols, metric, kw).cpu()
            results[i:end_i, j:end_j] = batch_results

    # Apply scaling if specified
    if scaling == "min-max":
        min_val, max_val = results.min(), results.max()
        if max_val != min_val:
            results = (results - min_val) / (max_val - min_val)
    elif scaling == "additive":
        results = (results + 1) / 2

    return results


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_default_num_gpus() -> int:
    """Get the default number of GPUs based on available CUDA devices.

    Raises:
        RuntimeError: If no CUDA devices are available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA devices detected. This functionality requires at least one GPU."
        )
    return torch.cuda.device_count()


@dataclass
class ProcessingConfig:
    """
    Configuration for subset selection with basic and advanced parameters.

    Basic Parameters:
        input_files: List of input files to process (required)
        subset_sizes: List of subset sizes - integers for absolute counts or floats for percentages (required)
        output_dir: Directory to save output files (default: "output")
        batch_size: Size of batches for processing (default: 100000)
        num_folds: Number of folds for subset selection (default: 50)
        combine_files: Whether to combine input files before processing (default: False)

    Advanced Parameters:
        instruction: Instruction for the encoder
        query_description: Description for queries
        templates: Dictionary of templates for formatting text
        template_name: Name of template to use
        num_gpus: Number of GPUs to use
        seed: Random seed
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries in seconds
        encoder_type: Type of encoder to use
        encoder_model: Specific model to use for encoding
    """

    # Basic parameters
    input_files: List[str]  # required
    subset_sizes: List[Union[int, float]]  # required
    output_dir: str = "output"
    batch_size: int = 100000
    num_folds: int = 50
    combine_files: bool = False

    # Advanced parameters
    instruction: str = field(
        default="Generate embeddings that capture the core meaning of user-assistant conversations, ensuring the embeddings can be clustered based on semantic similarity for subset selection.",
        metadata={"advanced": True},
    )
    query_description: str = field(default="Conversation", metadata={"advanced": True})
    templates: Dict[str, str] = field(
        default_factory=lambda: {
            "default": "{{ text }}",
            "conversation": "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}",
            "qa": "Question: {{ question }}\nAnswer: {{ answer }}",
        },
        metadata={"advanced": True},
    )
    template_name: str = field(default="conversation", metadata={"advanced": True})
    num_gpus: int = field(
        default_factory=get_default_num_gpus, metadata={"advanced": True}
    )
    seed: int = field(default=42, metadata={"advanced": True})
    max_retries: int = field(default=3, metadata={"advanced": True})
    retry_delay: int = field(default=30, metadata={"advanced": True})
    # TODO: change to arctic-snowflake model once it's ready
    encoder_type: str = field(default="bge", metadata={"advanced": True})
    encoder_model: str = field(default="BAAI/bge-m3", metadata={"advanced": True})

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.input_files:
            raise ValueError("input_files cannot be empty")

        if not isinstance(self.subset_sizes, list):
            raise ValueError("subset_sizes must be a list")

        for size in self.subset_sizes:
            if not isinstance(size, (int, float)):
                raise ValueError("subset_sizes must contain only integers or floats")
            if isinstance(size, float) and not (0 < size <= 100):
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
        for attempt in range(self.config.max_retries):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
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
        self.encoder = encoder_cls(model_name=config.encoder_model)
        self.env = Environment(loader=BaseLoader())
        self.templates = {
            k: self.env.from_string(v) for k, v in config.templates.items()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

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

        if self.config.combine_files:
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
        batch_files.sort(key=lambda x: self.extract_batch_number(x))
        max_batch_file = batch_files[-1]
        max_batch_number = self.extract_batch_number(max_batch_file)

        # Return the max batch number and the corresponding batch file path
        return max_batch_number, max_batch_file

    @retry_on_exception
    def process_batch(self, batch_texts: List[str], output_file: str) -> int:
        """
        Processes a batch of texts by generating embeddings and saving them to a file.

        Args:
            batch_texts (List[str]): The list of texts in the batch.
            output_file (str): The path to the output file where embeddings will be saved.

        Returns:
            int: The dimension of the embeddings generated.
        """
        embeddings = (
            self.encoder.encode(
                inputs=batch_texts,
                instruction=self.config.instruction,
                query_description=self.config.query_description,
            )
            .cpu()
            .numpy()
        )

        if embeddings.size == 0:
            logger.warning(
                f"No embeddings generated for batch, skipping file {output_file}"
            )
            return None  # Return None if there are no embeddings

        embedding_dim = embeddings.shape[1]
        logger.info(f"Embedding dimension for batch: {embedding_dim}")

        # Write embeddings to HDF5 file
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
            embedding_size, _ = self.get_embedding_size_dim_from_file(last_batch_file)
            total_processed = last_batch * self.config.batch_size + embedding_size
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

            text = self.format_text(example, format_type=self.config.template_name)
            if i < 5:
                logger.info(f"Example {i + 1}: {text}")
            batch_texts.append(text)

            if len(batch_texts) == self.config.batch_size:
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
        else:
            raise ValueError(f"Filename {filename} does not match expected pattern.")

    def get_embedding_size_dim_from_file(self, batch_file: str) -> Tuple[int, int]:
        """
        Reads the batch file to determine the embedding size (number of embeddings) and dimension.

        Args:
            batch_file (str): The path to the batch file.

        Returns:
            Tuple[int, int]: A tuple containing the number of embeddings and the embedding dimension.
        """
        with h5py.File(batch_file, "r") as h5f:
            if "embeddings" not in h5f:
                raise ValueError(
                    f"The file {batch_file} does not contain 'embeddings' dataset."
                )
            embeddings = h5f["embeddings"]
            embedding_size = embeddings.shape[0]  # Get the number of embeddings
            embedding_dim = embeddings.shape[1]  # Get the embedding dimension
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
        batch_files.sort(key=lambda x: self.extract_batch_number(x))

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

        fold_size = len(embeddings) // self.config.num_folds
        remainder = len(embeddings) % self.config.num_folds

        folds = []
        start_idx = 0
        for i in range(self.config.num_folds):
            extra = 1 if i < remainder else 0
            end_idx = start_idx + fold_size + extra
            folds.append(indices[start_idx:end_idx])
            start_idx = end_idx

        gpu_assignments = []
        folds_per_gpu = self.config.num_folds // self.config.num_gpus
        extra_folds = self.config.num_folds % self.config.num_gpus

        start_fold = 0
        for gpu_id in range(self.config.num_gpus):
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

        with Pool(processes=self.config.num_gpus) as pool:
            gpu_results = pool.map(process_folds_with_gpu, gpu_assignments)

        all_results = []
        for gpu_result in gpu_results:
            all_results.extend(gpu_result)

        combined_subsets = {
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
                self.config.output_dir,
                f"{base_name}_fl_{self.config.num_folds}_partitions_{subset_name}_metadata.npz",
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
            if self.config.combine_files:
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
    input_files: List[str], subset_sizes: List[Union[int, float]], **kwargs
) -> None:
    """
    Create subsets of datasets using facility location for diverse subset selection.

    Required Parameters:
        input_files: List of input files to process
        subset_sizes: List of subset sizes - integers for absolute counts or floats for percentages

    Optional Basic Parameters (via **kwargs):
        output_dir: Directory to save output files (default: "output")
        batch_size: Size of batches for processing (default: 100000)
        num_folds: Number of folds for subset selection (default: 50)
        combine_files: Whether to combine input files before processing (default: False)

    Advanced Parameters (via **kwargs):
        instruction: Instruction for the encoder
        query_description: Description for queries
        templates: Dictionary of templates for formatting text
        template_name: Name of template to use
        num_gpus: Number of GPUs to use
        seed: Random seed
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries in seconds
        encoder_type: Type of encoder to use
        encoder_model: Specific model to use for encoding
    """
    # Create config with required parameters
    config_params = {
        "input_files": input_files,
        "subset_sizes": subset_sizes,
    }

    # Get system's available GPU count
    available_gpus = get_default_num_gpus()

    # Update with any provided optional parameters
    config_params.update(kwargs)

    # Ensure num_gpus doesn't exceed available GPUs
    if "num_gpus" in config_params:
        requested_gpus = config_params["num_gpus"]
        if requested_gpus > available_gpus:
            logger.warning(
                f"Requested {requested_gpus} GPUs but only {available_gpus} available. "
                f"Falling back to using {available_gpus} GPUs."
            )
            config_params["num_gpus"] = available_gpus

    # Create configuration
    config = ProcessingConfig(**config_params)

    try:
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        try:
            set_start_method("spawn")
        except RuntimeError:
            # Method is already set, ignore the error
            pass

        logger.info(f"Processing configuration: {config}")

        # Initialize data processor based on encoder type
        os.makedirs(config.output_dir, exist_ok=True)

        if config.encoder_type == "bge":
            processor = DataProcessor(config, UnifiedBGEEncoder)
        processor.process_files(input_files, config.output_dir)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
