#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/03_embeddings/plantcad2/scripts"))
from aggregate_results import bh
from embed_shard import combine_rcps, region_slice


class WorkflowTests(unittest.TestCase):
    def test_rcps_combination_reverses_position_axis(self):
        forward = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        reverse = 100 + forward
        raw = torch.cat([forward, reverse], dim=-1)
        expected = (forward + reverse.flip(dims=[1])) / 2
        torch.testing.assert_close(combine_rcps(raw, 4), expected)

    def test_zero_repeat_uses_deletion_junction(self):
        self.assertEqual(region_slice(500, 0), slice(499, 501))
        self.assertEqual(region_slice(500, 12), slice(500, 512))

    def test_bh_is_monotonic(self):
        p = np.array([.01, .04, .03, .8]); q = bh(p); order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= 0)); self.assertTrue(np.all((q >= p) & (q <= 1)))


if __name__ == "__main__": unittest.main()
