# Standard
from functools import wraps
from typing import Optional, Union
import gc
import logging
import time

# Third Party
from torch import Tensor
from torch.nn import functional as F
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def get_default_num_gpus(testing_mode: bool = False) -> int:
    """
    Get the default number of GPUs based on available CUDA devices.

    Args:
        testing_mode (bool): If True, allows CPU usage with warnings. For testing only.
    """
    if not torch.cuda.is_available():
        if testing_mode:
            logger.warning(
                "No CUDA devices detected. Running in testing mode with CPU. "
                "Production use requires GPU acceleration."
            )
            return 1
        raise RuntimeError(
            "No CUDA devices detected. This functionality requires at least one GPU."
        )
    return torch.cuda.device_count()


def compute_pairwise_dense(
    tensor1: Tensor,
    tensor2: Optional[Tensor] = None,
    batch_size: int = 10000,
    metric: str = "cosine",
    device: Optional[Union[str, torch.device]] = None,
    scaling: Optional[str] = None,
    kw: float = 0.1,
) -> Tensor:
    """Compute pairwise metric in batches between two sets of vectors."""
    assert batch_size > 0, "Batch size must be positive."

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tensor2 is None:
        tensor2 = tensor1

    tensor1, tensor2 = tensor1.to(device), tensor2.to(device)
    n_samples1, n_samples2 = tensor1.size(0), tensor2.size(0)
    results = torch.zeros(n_samples1, n_samples2, device="cpu")

    if metric == "cosine":
        tensor1, tensor2 = (
            F.normalize(tensor1, p=2, dim=1),
            F.normalize(tensor2, p=2, dim=1),
        )

    def calculate_metric(a: Tensor, b: Tensor, metric: str, kw: float) -> Tensor:
        if metric in ["cosine", "dot"]:
            return torch.mm(a, b.T)
        if metric == "euclidean":
            distances = torch.cdist(a, b, p=2)
            similarities = 1 / (1 + distances**2)
            return similarities
        if metric == "rbf":
            distance = torch.cdist(a, b)
            squared_distance = distance**2
            avg_dist = torch.mean(squared_distance)
            torch.div(squared_distance, kw * avg_dist, out=squared_distance)
            torch.exp(-squared_distance, out=squared_distance)
            return squared_distance
        raise ValueError(f"Unknown metric: {metric}")

    for i in range(0, n_samples1, batch_size):
        end_i = min(i + batch_size, n_samples1)
        rows = tensor1[i:end_i]

        for j in range(0, n_samples2, batch_size):
            end_j = min(j + batch_size, n_samples2)
            cols = tensor2[j:end_j]
            batch_results = calculate_metric(rows, cols, metric, kw).cpu()
            results[i:end_i, j:end_j] = batch_results

    if scaling == "min-max":
        min_val, max_val = results.min(), results.max()
        if max_val != min_val:
            results = (results - min_val) / (max_val - min_val)
    elif scaling == "additive":
        results = (results + 1) / 2

    return results
