#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/03_embeddings/agront/scripts"))

from aggregate_results import bh
from analyze_shard import regression
from common import estimated_token_count, normalize_dna
from embed_shard import pooled_token_indexes
from prepare_catalog import parse_genotype


class WorkflowTests(unittest.TestCase):
    def test_genotype_parser_preserves_ploidy(self):
        self.assertEqual(parse_genotype("[3, 5, 5]"), (3, 5, 5))
        with self.assertRaises(ValueError): parse_genotype("[3, 4.5]")

    def test_sequence_normalization_and_token_estimate(self):
        self.assertEqual(normalize_dna("acgtry"), "ACGTNN")
        self.assertEqual(estimated_token_count("A" * 12), 3)
        self.assertEqual(estimated_token_count("A" * 6 + "N" + "C" * 6), 4)

    def test_regression_with_intercept(self):
        x = np.array([1.0, 2.0, 4.0]); y = np.stack((2 * x + 3, -x + 2), axis=1)
        slope, intercept, r2, _, _ = regression(x, y)
        np.testing.assert_allclose(slope, [2, -1]); np.testing.assert_allclose(intercept, [3, 2]); np.testing.assert_allclose(r2, [1, 1])

    def test_zero_repeat_pooling_uses_deletion_junction(self):
        spans = [(0, 6), (6, 12), (12, 18)]
        self.assertEqual(pooled_token_indexes(spans, 8, 8), [1])
        self.assertEqual(pooled_token_indexes(spans, 12, 12), [1, 2])
        self.assertEqual(pooled_token_indexes(spans, 7, 14), [1, 2])

    def test_bh_is_monotonic_by_pvalue(self):
        p = np.array([0.01, 0.04, 0.03, 0.8]); q = bh(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= 0)); self.assertTrue(np.all((q >= p) & (q <= 1)))


if __name__ == "__main__": unittest.main()
