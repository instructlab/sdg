# Standard
from typing import Optional
import logging

# Third Party
from torch import Tensor
from torch.nn import functional as F
import torch

__DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_default_num_gpus() -> int:
    """Get the default number of GPUs based on available CUDA devices."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA devices detected. This functionality requires at least one GPU."
        )
    return torch.cuda.device_count()


def compute_pairwise_dense(
    tensor1: Tensor,
    tensor2: Optional[Tensor] = None,
    batch_size: int = 10000,
    metric: str = "cosine",
    device: str = __DEVICE,
    scaling: Optional[str] = None,
    kw: float = 0.1,
) -> Tensor:
    """Compute pairwise metric in batches between two sets of vectors."""
    assert batch_size > 0, "Batch size must be positive."

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
