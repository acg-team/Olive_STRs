#!/usr/bin/env python3
"""Aggregate canonical reference-anchored GPN permutation regressions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config


KEYS = ["pooling", "layer", "metric", "predictor"]


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(ranked, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["root"])
    files = sorted((root / "results/reference_anchored_shards").glob("regression_*.csv.gz"))
    if len(files) != config["expected_shards"]:
        raise RuntimeError(
            f"Expected {config['expected_shards']} regression shards, found {len(files)}"
        )
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    expected_loci = 62847
    expected_rows = expected_loci * 2 * len(config["layers"]) * 3
    if len(data) != expected_rows or data.str_id.nunique() != expected_loci:
        raise RuntimeError(
            f"Unexpected dimensions: rows={len(data)}, loci={data.str_id.nunique()}"
        )
    if data.groupby(["str_id", "pooling", "layer", "metric"]).size().ne(1).any():
        raise RuntimeError("Duplicate or missing locus/pooling/layer/metric row")
    if data[["slope", "intercept", "r2", "p_value"]].isna().any().any():
        raise RuntimeError("Non-finite regression result")
    data["q_value"] = np.nan
    for _, indexes in data.groupby(KEYS, sort=False).groups.items():
        indexes = np.asarray(list(indexes))
        data.loc[indexes, "q_value"] = bh(data.loc[indexes, "p_value"])
    data["fdr_0_05"] = data.q_value < 0.05
    output = root / "results/per_locus_regressions.csv.gz"
    data.to_csv(output, index=False, compression="gzip")

    summary = (
        data.groupby(KEYS, as_index=False)
        .agg(
            n_loci=("str_id", "size"),
            mean_slope=("slope", "mean"),
            median_slope=("slope", "median"),
            slope_q25=("slope", lambda x: float(x.quantile(0.25))),
            slope_q75=("slope", lambda x: float(x.quantile(0.75))),
            fraction_positive=("slope", lambda x: float((x > 0).mean())),
            median_r2=("r2", "median"),
            median_p=("p_value", "median"),
            median_q=("q_value", "median"),
            fdr_0_05=("fdr_0_05", "sum"),
            fraction_q_lt_005=("fdr_0_05", "mean"),
            exact_tests=("permutation_method", lambda x: int((x == "exact").sum())),
            monte_carlo_tests=(
                "permutation_method", lambda x: int((x == "monte_carlo").sum())
            ),
        )
    )
    summary_path = root / "results/genome_wide_summary.csv"
    summary.to_csv(summary_path, index=False)

    lines = [
        "# Genome-wide reference-anchored GPN STR embedding analysis",
        "",
        f"This canonical analysis tests **{expected_loci:,} loci** having at least three "
        "distinct observed non-reference allele lengths. The reference allele is used only "
        "as the embedding-distance anchor and is excluded from regression.",
        "",
        "For every observed alternate allele, the predictor is its absolute repeat-count "
        "difference from the reference. Outcomes are L1, L2, and cosine distance from the "
        "reference embedding. Regressions use an intercept. Two-sided exact allele-label "
        "permutations are used when the alternate-allele factorial is at most 10,000; "
        "otherwise 10,000 deterministic Monte Carlo permutations are used. Benjamini–"
        "Hochberg FDR is applied separately for each pooling, layer, and metric family.",
        "",
        "The sole available GPN representation is the 512-dimensional last hidden state "
        "(output of convolutional block 25), labeled layer 25. Whole-sequence and actual "
        "STR-region mean pooling are evaluated.",
        "",
        "## Results",
        "",
        "| Pooling | Metric | Median slope/repeat | Positive | Median R² | FDR < 0.05 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["pooling", "metric"]).itertuples():
        lines.append(
            f"| {row.pooling} | {row.metric.upper()} | {row.median_slope:.6g} | "
            f"{row.fraction_positive:.1%} | {row.median_r2:.3f} | "
            f"{int(row.fdr_0_05):,}/{row.n_loci:,} ({row.fraction_q_lt_005:.1%}) |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A positive slope means that alleles farther from the reference repeat count tend "
        "to produce larger embedding perturbations. It does not establish functional or "
        "phenotypic causality.",
        "",
        "The superseded parametric t-test analysis is archived under "
        "`archive/reference_distance_ttest_20260728/`.",
        "",
    ]
    report = root / "report/GENOME_WIDE_STR_EMBEDDING_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(
        f"Wrote {output}, {summary_path}, and {report}; "
        "GPN_REFERENCE_AGGREGATION=PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
