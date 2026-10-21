#!/usr/bin/env python3
"""Regenerate and export feature-weight matrices for the selected K=9 dNMF fit."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR / "chr_str_var_gt_0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def fit_nmf(
    matrix: np.ndarray, k: int, seed: int, max_iterations: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    model = NMF(
        n_components=k,
        init="nndsvdar",
        solver="cd",
        max_iter=max_iterations,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sample_weights = model.fit_transform(matrix)
    row_sums = sample_weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("NMF produced a zero-sum sample-weight row")
    normalized = sample_weights / row_sums
    return (
        normalized,
        model.components_.copy(),
        float(model.reconstruction_err_),
        int(model.n_iter_),
    )


def load_expected(path: Path, component_columns: list[str]) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    required = {"sample_id", *component_columns}
    if missing := required - set(data.columns):
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    return data[["sample_id", *component_columns]]


def main() -> None:
    args = parse_args()
    source_dir = args.analysis_dir / "source_data"
    matrix_path = source_dir / "chr_str_var_gt_0.1.csv.gz"
    metrics_path = source_dir / "dnmf_repeat_metrics.tsv"

    data = pd.read_csv(matrix_path, index_col=0, low_memory=False)
    data.index = data.index.astype(str)
    if data.index.duplicated().any() or data.columns.duplicated().any():
        raise ValueError("dNMF input contains duplicate locus or sample IDs")
    imputed = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(data.T)
    standardized = StandardScaler().fit_transform(imputed)
    positive = np.clip(standardized, a_min=0, a_max=None)
    negative = np.clip(-standardized, a_min=0, a_max=None)

    metrics = pd.read_csv(metrics_path, sep="\t")
    candidates = metrics.loc[metrics["K"].eq(args.k)].sort_values(
        ["matched_components", "mean_absolute_correlation"],
        ascending=False,
        kind="stable",
    )
    if candidates.empty:
        raise ValueError(f"No K={args.k} fit is recorded in {metrics_path}")
    chosen = candidates.iloc[0]
    positive_q, positive_h, positive_error, positive_iterations = fit_nmf(
        positive, args.k, int(chosen["positive_seed"]), args.max_iterations
    )
    negative_q, negative_h, negative_error, negative_iterations = fit_nmf(
        negative, args.k, int(chosen["negative_seed"]), args.max_iterations
    )

    correlations = np.corrcoef(positive_q.T, negative_q.T)[: args.k, args.k :]
    positive_index, negative_index = linear_sum_assignment(-np.abs(correlations))
    if not np.array_equal(positive_index, np.arange(args.k)):
        order = np.argsort(positive_index)
        negative_index = negative_index[order]
    negative_q = negative_q[:, negative_index]
    negative_h = negative_h[negative_index, :]

    components = [f"Q{i}" for i in range(1, args.k + 1)]
    expected_positive = load_expected(source_dir / f"best_k{args.k}_pos.tsv", components)
    expected_negative = load_expected(source_dir / f"best_k{args.k}_neg.tsv", components)
    sample_ids = data.columns.astype(str).tolist()
    if expected_positive["sample_id"].tolist() != sample_ids:
        raise ValueError("Positive Q matrix sample order differs from the dNMF input")
    if expected_negative["sample_id"].tolist() != sample_ids:
        raise ValueError("Negative Q matrix sample order differs from the dNMF input")
    positive_difference = float(
        np.max(np.abs(positive_q - expected_positive[components].to_numpy(dtype=float)))
    )
    negative_difference = float(
        np.max(np.abs(negative_q - expected_negative[components].to_numpy(dtype=float)))
    )
    if positive_difference > args.tolerance or negative_difference > args.tolerance:
        raise ValueError(
            "Refitted Q matrices do not reproduce the saved results: "
            f"positive max difference={positive_difference:.3g}, "
            f"negative max difference={negative_difference:.3g}"
        )

    for direction, weights in (("pos", positive_h), ("neg", negative_h)):
        output = pd.DataFrame(weights.T, columns=components)
        output.insert(0, "str_id", data.index)
        output.to_csv(
            source_dir / f"best_k{args.k}_{direction}_weights.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

    metadata = pd.DataFrame(
        [
            {
                "K": args.k,
                "selected_repeat": int(chosen["repeat"]),
                "positive_seed": int(chosen["positive_seed"]),
                "negative_seed": int(chosen["negative_seed"]),
                "n_samples": data.shape[1],
                "n_loci": data.shape[0],
                "positive_reconstruction_error": positive_error,
                "negative_reconstruction_error": negative_error,
                "positive_iterations": positive_iterations,
                "negative_iterations": negative_iterations,
                "positive_q_max_abs_difference": positive_difference,
                "negative_q_max_abs_difference": negative_difference,
                "negative_original_component_order": ",".join(
                    f"Q{i + 1}" for i in negative_index
                ),
            }
        ]
    )
    metadata.to_csv(
        source_dir / f"best_k{args.k}_weight_matrix_metadata.tsv",
        sep="\t",
        index=False,
    )
    print(metadata.to_string(index=False))
    print(
        f"Wrote two {data.shape[0]:,} x {args.k} locus-weight matrices to {source_dir}"
    )


if __name__ == "__main__":
    main()
