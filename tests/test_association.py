#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/06_trait_association/analyze_str_oil_association.py"
)
SPEC = importlib.util.spec_from_file_location("association", MODULE_PATH)
association = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = association
SPEC.loader.exec_module(association)


class AssociationTests(unittest.TestCase):
    def test_complete_case_association_excludes_missing_sample(self):
        rng = np.random.default_rng(11)
        n = 60
        g = rng.integers(0, 4, size=n).astype(float)
        g[[2, 17, 41]] = np.nan
        x = np.column_stack([np.ones(n), rng.normal(size=n)])
        observed = np.isfinite(g)
        y = np.where(observed, 1.5 * np.nan_to_num(g), 0) + 0.3 * x[:, 1]
        y += rng.normal(scale=0.7, size=n)
        beta, se, pvalue, n_tested = association.association_complete_case(
            g, y, x, np.eye(n), delta=1.0
        )
        self.assertEqual(n_tested, n - 3)
        self.assertGreater(beta, 1.0)
        self.assertGreater(se, 0)
        self.assertLess(pvalue, 1e-10)

    def test_bh_is_monotone_in_pvalue_order(self):
        p = np.array([0.04, 0.001, np.nan, 0.02])
        q = association.benjamini_hochberg(p)
        self.assertTrue(np.isnan(q[2]))
        ordered = np.argsort(p[np.isfinite(p)])
        finite_q = q[np.isfinite(p)][ordered]
        self.assertTrue(np.all(np.diff(finite_q) >= 0))

    def test_fixed_effect_meta(self):
        meta = association.fixed_effect_meta(
            np.array([1.0]), np.array([0.5]), np.array([1.0]), np.array([0.5])
        )
        self.assertAlmostEqual(meta["beta"][0], 1.0)
        self.assertAlmostEqual(meta["se"][0], np.sqrt(0.125))
        self.assertAlmostEqual(meta["q"][0], 0.0)
        self.assertAlmostEqual(meta["i2"][0], 0.0)

    def test_lmm_detects_simulated_effect(self):
        rng = np.random.default_rng(7)
        n = 80
        g = rng.integers(0, 4, size=n).astype(float)
        x = np.column_stack([np.ones(n), rng.normal(size=n)])
        k = np.eye(n)
        y = 2.0 * g + 0.4 * x[:, 1] + rng.normal(scale=0.7, size=n)
        null = association.fit_null_lmm(y, x, k)
        beta, se, p = association.association_chunk(g[None, :], null)
        self.assertGreater(beta[0], 1.5)
        self.assertLess(p[0], 1e-10)
        self.assertGreater(se[0], 0)


if __name__ == "__main__":
    unittest.main()
