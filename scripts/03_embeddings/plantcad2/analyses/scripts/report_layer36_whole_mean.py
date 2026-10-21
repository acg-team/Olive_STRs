#!/usr/bin/env python3
"""Report slope and FDR distributions for PlantCAD2 layer-36 whole-mean tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plantcad2_layer36_analysis_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "analyses"
INPUT = ROOT / "results/per_locus_regressions.csv.gz"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
REPORT = HERE / "report/LAYER36_WHOLE_MEAN_SLOPE_FDR_REPORT.md"
METRICS = ("l1", "l2", "cosine")
PREDICTOR = "absolute_repeat_difference_from_reference"


def format_number(value):
    value = float(value)
    return f"{value:.6g}" if abs(value) >= 1e-3 else f"{value:.3e}"


def main():
    for directory in (RESULTS, FIGURES, REPORT.parent): directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    selected = data[(data.pooling == "whole_mean") & (data.layer == 36)].copy()
    expected_loci = 62_847
    expected = expected_loci * len(METRICS)
    if len(selected) != expected or selected.str_id.nunique() != expected_loci:
        raise RuntimeError(f"Unexpected layer-36 subset: rows={len(selected)}, loci={selected.str_id.nunique()}")
    if set(selected.predictor) != {PREDICTOR} or selected.reference_included.any():
        raise RuntimeError("Input is not the expected reference-anchored, reference-excluded analysis")
    if not np.isfinite(selected[["slope", "intercept", "r2", "p_value", "q_value"]]).all().all():
        raise RuntimeError("Non-finite regression values")
    selected.to_csv(RESULTS / "layer36_whole_mean_regressions.csv.gz", index=False, compression="gzip")

    distribution_rows = []
    for (metric, predictor), frame in selected.groupby(["metric", "predictor"], sort=True):
        slope = frame.slope.to_numpy()
        quantiles = np.quantile(slope, [.01, .05, .25, .5, .75, .95, .99])
        distribution_rows.append({
            "metric": metric, "predictor": predictor, "n_loci": len(frame),
            "mean": slope.mean(), "standard_deviation": slope.std(ddof=1), "minimum": slope.min(),
            "q01": quantiles[0], "q05": quantiles[1], "q25": quantiles[2], "median": quantiles[3],
            "q75": quantiles[4], "q95": quantiles[5], "q99": quantiles[6], "maximum": slope.max(),
            "negative_loci": int(np.sum(slope < 0)), "zero_loci": int(np.sum(slope == 0)),
            "positive_loci": int(np.sum(slope > 0)), "fraction_positive": float(np.mean(slope > 0)),
        })
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(RESULTS / "slope_distribution_summary.csv", index=False)

    fdr_rows = []
    for (metric, predictor), frame in selected.groupby(["metric", "predictor"], sort=True):
        q = frame.q_value.to_numpy(); p = frame.p_value.to_numpy()
        fdr_rows.append({
            "metric": metric, "predictor": predictor, "n_loci": len(frame),
            "q_lt_0.01": int(np.sum(q < .01)), "fraction_q_lt_0.01": float(np.mean(q < .01)),
            "q_lt_0.05": int(np.sum(q < .05)), "fraction_q_lt_0.05": float(np.mean(q < .05)),
            "q_lt_0.10": int(np.sum(q < .10)), "fraction_q_lt_0.10": float(np.mean(q < .10)),
            "median_p": np.median(p), "median_q": np.median(q),
            "q25": np.quantile(q, .25), "q75": np.quantile(q, .75), "q95": np.quantile(q, .95),
        })
    fdr = pd.DataFrame(fdr_rows)
    fdr.to_csv(RESULTS / "fdr_distribution_summary.csv", index=False)

    by_alleles = selected.groupby(["metric", "predictor", "n_alternate_alleles"], as_index=False).agg(
        loci=("str_id", "size"), median_slope=("slope", "median"),
        fraction_positive=("slope", lambda values: float(np.mean(values > 0))),
        fdr_significant=("q_value", lambda values: int(np.sum(values < .05))),
        fraction_fdr_significant=("q_value", lambda values: float(np.mean(values < .05))),
        median_p=("p_value", "median"), median_q=("q_value", "median"),
    )
    by_alleles.to_csv(RESULTS / "fdr_by_allele_count.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        values = selected[selected.metric == metric].slope.to_numpy()
        low, high = np.quantile(values, [.005, .995]); central = values[(values >= low) & (values <= high)]
        ax.hist(central, bins=80, color="#3b7a9e", edgecolor="none")
        ax.axvline(0, color="black", linewidth=.8); ax.axvline(np.median(values), color="#b22222", linestyle="--")
        ax.set_yscale("log"); ax.set_title(metric.upper())
        ax.set(xlabel="Slope per repeat difference (central 99%)", ylabel="Loci (log count)")
        ax.text(.98, .95, f"median={format_number(np.median(values))}\nclipped={len(values)-len(central):,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.suptitle("PlantCAD2-Large layer 36 whole-mean slope distributions", fontsize=14)
    fig.savefig(FIGURES / "layer36_whole_mean_slope_distributions.png", dpi=200); plt.close(fig)

    repeat = selected
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        frame = repeat[repeat.metric == metric]; q = frame.q_value.to_numpy()
        ax.hist(q, bins=np.linspace(0, 1, 51), color="#6b5ca5", edgecolor="white")
        ax.axvline(.05, color="#b22222", linestyle="--", label="FDR 0.05")
        ax.set(title=metric.upper(), xlabel="BH-adjusted q value", ylabel="Loci")
        ax.legend(); ax.text(.98, .92, f"q<0.05: {np.mean(q<.05):.1%}", transform=ax.transAxes, ha="right")
    fig.suptitle("PlantCAD2-Large layer 36 whole-mean FDR distributions", fontsize=14)
    fig.savefig(FIGURES / "layer36_whole_mean_fdr_distributions.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for metric in METRICS:
        frame = by_alleles[by_alleles.metric == metric]
        ax.plot(frame.n_alternate_alleles, frame.fraction_fdr_significant, marker="o", label=metric.upper())
    ax.set(xlabel="Observed non-reference allele lengths at locus", ylabel="Fraction FDR-significant",
           title="FDR discovery rate depends on allele count")
    ax.set_ylim(-.02, 1.02); ax.legend()
    fig.savefig(FIGURES / "layer36_fdr_by_allele_count.png", dpi=200); plt.close(fig)

    repeat_dist = distribution.set_index("metric")
    repeat_fdr = fdr.set_index("metric")
    lines = [
        "# PlantCAD2-Large layer-36 whole-mean slope and FDR distributions", "",
        f"This report contains all **{expected_loci:,} eligible loci** from the reference-anchored genome-wide analysis. It selects `pooling = whole_mean` and `layer = 36`. The reference allele is the distance anchor but is excluded as a regression observation; observations are distinct non-reference allele lengths.", "",
        "## Slopes per repeat", "",
        "| Metric | Mean | SD | 1% | Median | 99% | Range | Positive loci |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = repeat_dist.loc[metric]
        lines.append(f"| {metric.upper()} | {format_number(row['mean'])} | {format_number(row.standard_deviation)} | {format_number(row.q01)} | {format_number(row['median'])} | {format_number(row.q99)} | {format_number(row['minimum'])} to {format_number(row['maximum'])} | {int(row.positive_loci):,}/{int(row.n_loci):,} ({row.fraction_positive:.2%}) |")
    lines += ["", "## FDR distribution", "",
              "FDR is Benjamini–Hochberg corrected separately for each pooling/layer/metric/predictor family.", "",
              "| Metric | q < 0.01 | q < 0.05 | q < 0.10 | Median p | Median q |",
              "|---|---:|---:|---:|---:|---:|"]
    for metric in METRICS:
        row = repeat_fdr.loc[metric]
        lines.append(f"| {metric.upper()} | {int(row['q_lt_0.01']):,} ({row['fraction_q_lt_0.01']:.1%}) | {int(row['q_lt_0.05']):,} ({row['fraction_q_lt_0.05']:.1%}) | {int(row['q_lt_0.10']):,} ({row['fraction_q_lt_0.10']:.1%}) | {format_number(row.median_p)} | {format_number(row.median_q)} |")
    l2 = repeat_dist.loc["l2"]; l2f = repeat_fdr.loc["l2"]
    lines += ["", "## Interpretation", "",
              f"For the primary L2-per-repeat test, the mean slope is **{format_number(l2['mean'])}**, the median is **{format_number(l2['median'])}**, and **{l2.fraction_positive:.2%}** of loci are positive. **{int(l2f['q_lt_0.05']):,} loci ({l2f['fraction_q_lt_0.05']:.1%})** pass FDR 0.05.", "",
              "A positive slope means that alternate alleles farther from the reference repeat count tend to perturb the embedding more. Nonsignificance does not prove absence of an effect: it also reflects the number and spacing of alternate alleles and permutation resolution.", "",
              "The q-value histogram is discrete because tests use exact permutations when possible and 10,000 Monte Carlo permutations otherwise. Benjamini–Hochberg correction controls multiple testing across loci within this layer/pooling/metric family; it does not replace the within-locus permutation test.", "",
              "Only the repeat-count predictor is stored. A slope per bp can be derived as slope per repeat divided by motif length; this rescaling leaves each locus's p and q values unchanged.", "",
              "## Outputs", "",
              "- `results/layer36_whole_mean_regressions.csv.gz`", "- `results/slope_distribution_summary.csv`",
              "- `results/fdr_distribution_summary.csv`", "- `results/fdr_by_allele_count.csv`",
              "- `figures/layer36_whole_mean_slope_distributions.png`", "- `figures/layer36_whole_mean_fdr_distributions.png`",
              "- `figures/layer36_fdr_by_allele_count.png`", ""]
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")
    print("PLANTCAD2_LAYER36_WHOLE_MEAN_REPORT=PASS")


if __name__ == "__main__": main()
