# SPDX-License-Identifier: Apache-2.0

# Standard
from difflib import SequenceMatcher
import logging

# Third Party
import pandas as pd
from datasets import Dataset

# Local
from ..registry import BlockRegistry
from ..utils.pandas import dataset_from_pandas_dataframe
from .block import Block

logger = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _deduplicate_group(group, col, threshold):
    """Remove near-duplicate rows within a single group.

    Returns a list of integer indices to keep.
    """
    kept_indices = []
    kept_texts = []

    for idx, row in group.iterrows():
        text = str(row[col])
        is_duplicate = any(
            _similarity(text, kept) > threshold for kept in kept_texts
        )
        if not is_duplicate:
            kept_indices.append(idx)
            kept_texts.append(text)

    return kept_indices


# This is part of the public API.
@BlockRegistry.register("SimilarityFilterBlock")
class SimilarityFilterBlock(Block):
    def __init__(
        self,
        ctx,
        pipe,
        block_name,
        filter_column,
        threshold=0.85,
        group_by=None,
    ) -> None:
        """
        Initializes a new instance of the SimilarityFilterBlock class.

        Parameters:
        - ctx (PipelineContext): A PipelineContext object containing runtime parameters.
        - pipe (Pipeline): The Pipeline containing this block in its chain.
        - block_name (str): An identifier for this block.
        - filter_column (str): The column containing text to compare for similarity.
        - threshold (float): Similarity ratio (0.0 to 1.0). Rows with similarity
          above this value to any previously kept row are dropped. Default 0.85.
        - group_by (str, optional): Column to group by before deduplication.
          If set, similarity is only compared within each group. Default None.
        """
        super().__init__(ctx, pipe, block_name)
        self.filter_column = filter_column
        self.threshold = threshold
        self.group_by = group_by

    def generate(self, samples) -> Dataset:
        if len(samples) == 0:
            return samples

        df = samples.to_pandas()
        original_len = len(df)

        if self.group_by and self.group_by in df.columns:
            groups = []
            for _, group in df.groupby(self.group_by):
                kept = _deduplicate_group(group, self.filter_column, self.threshold)
                groups.append(group.loc[kept])
            result = (
                pd.concat(groups, ignore_index=True)
                if groups
                else df.iloc[:0]
            )
        else:
            kept = _deduplicate_group(df, self.filter_column, self.threshold)
            result = df.loc[kept]

        removed = original_len - len(result)
        if removed > 0:
            logger.info(
                "SimilarityFilterBlock '%s': removed %d near-duplicates "
                "(threshold=%.2f), %d → %d rows",
                self.block_name,
                removed,
                self.threshold,
                original_len,
                len(result),
            )

        return dataset_from_pandas_dataframe(result)
