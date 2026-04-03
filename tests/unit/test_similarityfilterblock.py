# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import unittest

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg import SimilarityFilterBlock


class TestSimilarityFilterBlock(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.dataset_num_procs = 1
        self.pipe = MagicMock()

    def _make_block(self, filter_column="text", threshold=0.85, group_by=None):
        return SimilarityFilterBlock(
            self.ctx,
            self.pipe,
            "test_similarity_filter",
            filter_column=filter_column,
            threshold=threshold,
            group_by=group_by,
        )

    def test_keeps_unique_rows(self):
        block = self._make_block()
        ds = Dataset.from_dict(
            {"text": ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india"]}
        )
        result = block.generate(ds)
        self.assertEqual(len(result), 3)

    def test_removes_exact_duplicates(self):
        block = self._make_block(threshold=0.8)
        ds = Dataset.from_dict(
            {"text": ["hello world", "hello world", "hello world"]}
        )
        result = block.generate(ds)
        self.assertEqual(len(result), 1)

    def test_removes_near_duplicates(self):
        block = self._make_block(threshold=0.7)
        ds = Dataset.from_dict(
            {
                "text": [
                    "What is photosynthesis and how does it work?",
                    "What is photosynthesis and how does it function?",
                    "Explain the process of sourdough bread making.",
                ]
            }
        )
        result = block.generate(ds)
        self.assertEqual(len(result), 2)

    def test_group_by_isolates_groups(self):
        block = self._make_block(threshold=0.8, group_by="doc_id")
        ds = Dataset.from_dict(
            {
                "text": ["same text here", "same text here"],
                "doc_id": ["doc_a", "doc_b"],
            }
        )
        result = block.generate(ds)
        self.assertEqual(len(result), 2)

    def test_group_by_deduplicates_within_group(self):
        block = self._make_block(threshold=0.8, group_by="doc_id")
        ds = Dataset.from_dict(
            {
                "text": ["same text here", "same text here"],
                "doc_id": ["doc_a", "doc_a"],
            }
        )
        result = block.generate(ds)
        self.assertEqual(len(result), 1)

    def test_empty_dataset(self):
        block = self._make_block()
        ds = Dataset.from_dict({"text": []})
        result = block.generate(ds)
        self.assertEqual(len(result), 0)

    def test_low_threshold_more_aggressive(self):
        texts = [
            "What is photosynthesis?",
            "What is the process of photosynthesis?",
            "Explain sourdough bread.",
        ]
        strict = self._make_block(threshold=0.5)
        lenient = self._make_block(threshold=0.95)
        ds = Dataset.from_dict({"text": texts})
        self.assertLessEqual(len(strict.generate(ds)), len(lenient.generate(ds)))
