#!/usr/bin/env python3
"""Aggregate reference-anchored ArgoNT regressions and replace the canonical report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config


KEYS = ["pooling", "layer", "metric", "predictor"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(ranked, 1)
    return result


def main():
    args = arguments()
    config = load_config(args.config)
    root = Path(config["root"])
    files = sorted((root / "results/reference_anchored_shards").glob("regression_*.csv.gz"))
    manifest = json.loads((root / "results/preparation_manifest.json").read_text())
    if len(files) != manifest["shards"]:
        raise RuntimeError(f"Expected {manifest['shards']} regression shards, found {len(files)}")
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
        data.groupby(["pooling", "layer", "metric", "predictor"], as_index=False)
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
    summary.to_csv(root / "results/genome_wide_summary.csv", index=False)

    str_mean = summary[summary.pooling == "str_mean"]
    lines = [
        "# Genome-wide reference-anchored STR embedding analysis",
        "",
        f"This canonical analysis replaces the previous anchor-free pairwise analysis. It tests "
        f"**{expected_loci:,} loci** having at least three distinct observed non-reference allele "
        "lengths. The reference allele is used only as the embedding-distance anchor and is excluded "
        "from regression.",
        "",
        "For every observed alternate allele, the predictor is its absolute repeat-count difference "
        "from the reference. Outcomes are L1, L2, and cosine distance from the reference embedding. "
        "Regressions use an intercept. Two-sided exact allele-label permutations are used when the "
        "alternate-allele factorial is at most 10,000; otherwise 10,000 deterministic Monte Carlo "
        "permutations are used. Benjamini–Hochberg FDR is applied separately for each pooling, layer, "
        "and metric family.",
        "",
        "Sequences retain the established fixed 500-bp left and right flanks with an allele-specific "
        "STR. Whole-sequence mean and actual STR-tract mean embeddings are evaluated at ArgoNT layers "
        "11, 21, 31, and 40.",
        "",
        "## STR-mean results",
        "",
        "| Layer | Metric | Median slope/repeat | Positive | Median R² | FDR < 0.05 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in str_mean.sort_values(["layer", "metric"]).itertuples():
        lines.append(
            f"| {row.layer} | {row.metric.upper()} | {row.median_slope:.6g} | "
            f"{row.fraction_positive:.1%} | {row.median_r2:.3f} | "
            f"{int(row.fdr_0_05):,}/{row.n_loci:,} ({row.fraction_q_lt_005:.1%}) |"
        )
    lines += [
        "",
        "## Whole-sequence mean results",
        "",
        "| Layer | Metric | Median slope/repeat | Positive | Median R² | FDR < 0.05 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    whole = summary[summary.pooling == "whole_mean"]
    for row in whole.sort_values(["layer", "metric"]).itertuples():
        lines.append(
            f"| {row.layer} | {row.metric.upper()} | {row.median_slope:.6g} | "
            f"{row.fraction_positive:.1%} | {row.median_r2:.3f} | "
            f"{int(row.fdr_0_05):,}/{row.n_loci:,} ({row.fraction_q_lt_005:.1%}) |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A positive slope means that alternate alleles farther from the reference repeat count tend "
        "to produce larger embedding perturbations relative to that reference. It does not establish "
        "functional impact or the direction of a phenotypic effect. Exact p-value resolution depends "
        "strongly on the number of alternate alleles; this must be considered when interpreting FDR.",
        "",
        "## Canonical outputs",
        "",
        "- `results/per_locus_regressions.csv.gz`",
        "- `results/genome_wide_summary.csv`",
        "- `figures/genome_wide_l2_slopes.png`",
        "- `figures/reference_anchored_slope_by_layer.png`",
        "- `figures/reference_anchored_fdr_by_layer.png`",
        "",
        "The superseded anchor-free results are retained under `archive/anchor_free_20260724/`.",
        "",
    ]
    report = root / "report/GENOME_WIDE_STR_EMBEDDING_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(
        f"Wrote {output}, genome_wide_summary.csv, and {report}; "
        "ARGONT_REFERENCE_AGGREGATION=PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
