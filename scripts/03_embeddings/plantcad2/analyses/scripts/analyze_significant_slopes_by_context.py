#!/usr/bin/env python3
"""Stratify significant layer-36 whole-mean L2 slopes by motif and annotation."""

from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
ANALYSES = ROOT / "analyses"
PROMOTER_ANNOTATION = Path(os.environ.get(
    "OLIVE_STR_ANNOTATION", "inputs/O_europeana_strs.functional.promoter2kb.annotated.bed"
))
REGRESSION = ANALYSES / "results/layer36_whole_mean_regressions.csv.gz"
LOCI = ROOT / "inputs/loci.csv.gz"
REGION_ORDER = ["CDS", "UTR", "other_exon", "intron", "promoter", "intergenic"]


def load_promoter_annotation():
    annotation = pd.read_csv(
        PROMOTER_ANNOTATION,
        sep="\t",
        usecols=["#chrom", "start", "functional_region"],
        dtype={"#chrom": "string"},
    )
    annotation["str_id"] = (
        annotation["#chrom"] + "_" + annotation["start"].astype(str)
    )
    if annotation["str_id"].duplicated().any():
        raise ValueError("Duplicate STR identifiers in promoter-aware annotation")
    annotation["genome_region"] = annotation["functional_region"].replace(
        {"five_prime_UTR": "UTR", "three_prime_UTR": "UTR"}
    )
    return annotation[["str_id", "genome_region"]]


def bh(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty(len(values)); output[order] = np.minimum(ranked, 1)
    return output


def group_summary(data, group):
    return (data.groupby(group, observed=True)
            .agg(n_loci=("str_id", "size"), mean_slope_per_bp=("slope_per_bp", "mean"),
                 sd_slope_per_bp=("slope_per_bp", "std"), median_slope_per_bp=("slope_per_bp", "median"),
                 q25_slope_per_bp=("slope_per_bp", lambda x: x.quantile(.25)),
                 q75_slope_per_bp=("slope_per_bp", lambda x: x.quantile(.75)),
                 mean_slope_per_repeat=("slope_per_repeat", "mean"),
                 median_slope_per_repeat=("slope_per_repeat", "median"))
            .reset_index())


def omnibus(data, group, unit):
    arrays = [x[unit].to_numpy() for _, x in data.groupby(group, observed=True) if len(x)]
    statistic, p = kruskal(*arrays)
    n, k = sum(map(len, arrays)), len(arrays)
    epsilon_squared = max(0.0, (statistic - k + 1) / (n - k))
    return {"factor": group, "slope_unit": unit, "n_loci": n, "groups": k,
            "kruskal_h": statistic, "p_value": p, "epsilon_squared": epsilon_squared}


def pairwise(data, group, unit):
    rows = []
    groups = {str(key): x[unit].to_numpy() for key, x in data.groupby(group, observed=True)}
    for a, b in combinations(groups, 2):
        x, y = groups[a], groups[b]
        result = mannwhitneyu(x, y, alternative="two-sided")
        # Positive means values in group a tend to exceed values in group b.
        rank_biserial = 2 * result.statistic / (len(x) * len(y)) - 1
        rows.append({"factor": group, "slope_unit": unit, "group_a": a, "group_b": b,
                     "n_a": len(x), "n_b": len(y), "median_a": np.median(x),
                     "median_b": np.median(y), "median_difference_a_minus_b": np.median(x) - np.median(y),
                     "mann_whitney_u": result.statistic, "p_value": result.pvalue,
                     "rank_biserial_a_vs_b": rank_biserial})
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result.p_value)
    return result


def boxplot(ax, data, group, order, value, ylabel, title):
    arrays = [data.loc[data[group].astype(str).eq(str(item)), value].to_numpy() for item in order]
    ax.boxplot(arrays, tick_labels=[str(x) for x in order], showfliers=False, patch_artist=True,
               medianprops={"color": "#b22222", "linewidth": 1.6},
               boxprops={"facecolor": "#7aa6c2", "alpha": .8})
    ax.set_xlabel("Motif length (bp)" if group == "period" else "Genome annotation")
    ax.set_ylabel(ylabel); ax.set_title(title)
    if group == "genome_region":
        ax.tick_params(axis="x", rotation=25)


def fmt_p(value):
    return "<1e-300" if value == 0 else f"{value:.3g}"


def main():
    result_dir = ANALYSES / "results"; figure_dir = ANALYSES / "figures"; report_dir = ANALYSES / "report"
    for directory in (result_dir, figure_dir, report_dir): directory.mkdir(parents=True, exist_ok=True)

    regressions = pd.read_csv(REGRESSION)
    significant = regressions[(regressions.metric == "l2") &
                              (regressions.predictor == "absolute_repeat_difference_from_reference") &
                              (regressions.q_value < .05)].copy()
    significant = significant[["str_id", "n_alternate_alleles", "n_unique_absolute_differences",
                               "slope", "r2", "p_value", "q_value"]]
    significant = significant.rename(columns={"slope": "slope_per_repeat"})
    loci = pd.read_csv(LOCI, usecols=["str_id", "contig", "start", "end", "period", "motif"])
    data = significant.merge(loci, on="str_id", validate="one_to_one")
    data["slope_per_bp"] = data.slope_per_repeat / data.period
    data = data.reset_index(drop=True)
    data = data.merge(
        load_promoter_annotation(),
        on="str_id",
        how="left",
        validate="one_to_one",
    )
    if data["genome_region"].isna().any():
        missing = int(data["genome_region"].isna().sum())
        raise ValueError(f"{missing:,} significant loci lack promoter-aware annotation")
    data["genome_region"] = pd.Categorical(data.genome_region, REGION_ORDER, ordered=True)
    # Percentile ranks remove the motif-specific location and scale before comparing regions.
    data["within_motif_slope_rank"] = data.groupby("period").slope_per_bp.rank(pct=True)
    if not np.allclose(data.slope_per_repeat, data.slope_per_bp * data.period, rtol=2e-5, atol=1e-5):
        raise RuntimeError("Per-repeat and per-bp slopes do not have the expected motif-length scaling")

    motif_summary = group_summary(data, "period")
    region_summary = group_summary(data, "genome_region")
    region_rank = (data.groupby("genome_region", observed=True).within_motif_slope_rank
                   .agg(mean_within_motif_rank="mean", median_within_motif_rank="median").reset_index())
    region_summary = region_summary.merge(region_rank, on="genome_region", validate="one_to_one")
    motif_region_counts = pd.crosstab(data.genome_region, data.period).reindex(REGION_ORDER, fill_value=0)
    tests = pd.DataFrame([omnibus(data, factor, unit) for factor in ("period", "genome_region")
                          for unit in ("slope_per_bp", "slope_per_repeat")])
    tests = pd.concat([tests, pd.DataFrame([omnibus(data, "genome_region", "within_motif_slope_rank")])], ignore_index=True)
    pairs = pd.concat([pairwise(data, factor, "slope_per_bp") for factor in ("period", "genome_region")] +
                      [pairwise(data, "genome_region", "within_motif_slope_rank")], ignore_index=True)

    data.to_csv(result_dir / "layer36_whole_mean_l2_fdr05_annotated.csv.gz", index=False, compression="gzip")
    motif_summary.to_csv(result_dir / "significant_slope_by_motif_length.csv", index=False)
    region_summary.to_csv(result_dir / "significant_slope_by_genome_region.csv", index=False)
    motif_region_counts.to_csv(result_dir / "significant_locus_counts_by_motif_and_region.csv")
    tests.to_csv(result_dir / "significant_slope_omnibus_tests.csv", index=False)
    pairs.to_csv(result_dir / "significant_slope_pairwise_tests.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    periods = sorted(data.period.unique())
    present_regions = [x for x in REGION_ORDER if (data.genome_region == x).any()]
    boxplot(axes[0, 0], data, "period", periods, "slope_per_bp", "L2 slope per bp", "Motif length: scale-normalized slope")
    boxplot(axes[0, 1], data, "genome_region", present_regions, "slope_per_bp", "L2 slope per bp", "Genome region: scale-normalized slope")
    boxplot(axes[1, 0], data, "period", periods, "slope_per_repeat", "L2 slope per repeat", "Motif length: per-repeat slope")
    boxplot(axes[1, 1], data, "genome_region", present_regions, "slope_per_repeat", "L2 slope per repeat", "Genome region: per-repeat slope")
    fig.suptitle(f"PlantCAD2-Large layer 36 whole-mean L2 slopes among FDR < 0.05 loci (n={len(data):,})", fontsize=15)
    fig.savefig(figure_dir / "significant_l2_slopes_by_motif_and_region.png", dpi=180)
    plt.close(fig)

    motif_test = tests[(tests.factor == "period") & (tests.slope_unit == "slope_per_bp")].iloc[0]
    region_test = tests[(tests.factor == "genome_region") & (tests.slope_unit == "slope_per_bp")].iloc[0]
    adjusted_region_test = tests[(tests.factor == "genome_region") & (tests.slope_unit == "within_motif_slope_rank")].iloc[0]
    lines = [
        "# Significant Layer 36 whole-mean slopes by motif length and genome region", "",
        f"This analysis includes the **{len(data):,} loci** significant at `q < 0.05` for the reference-anchored PlantCAD2 Layer 36 whole-sequence mean L2 regression. The reference is the distance anchor and is not a regression observation. Genome regions use the canonical strand-aware 2-kb upstream promoter annotation with hierarchy CDS > UTR > other exon > intron > promoter > intergenic.", "",
        "## Motif length", "",
        "| Motif bp | Loci | Median slope/bp | Mean slope/bp | Median slope/repeat |", "|---:|---:|---:|---:|---:|",
    ]
    for row in motif_summary.itertuples():
        lines.append(f"| {row.period} | {row.n_loci:,} | {row.median_slope_per_bp:.4f} | {row.mean_slope_per_bp:.4f} | {row.median_slope_per_repeat:.4f} |")
    lines += ["", f"Per-bp slopes differ by motif length (Kruskal–Wallis H={motif_test.kruskal_h:.1f}, p={fmt_p(motif_test.p_value)}, epsilon-squared={motif_test.epsilon_squared:.4f}).", "",
              "The per-bp slope is the appropriate comparison across motif lengths. The per-repeat slope equals the per-bp slope multiplied by motif length, so differences in per-repeat values are partly guaranteed by definition.", "",
              "## Genome region", "", "| Region | Loci | Median slope/bp | Mean slope/bp | Median slope/repeat |", "|---|---:|---:|---:|---:|"]
    for row in region_summary.itertuples():
        lines.append(f"| {row.genome_region} | {row.n_loci:,} | {row.median_slope_per_bp:.4f} | {row.mean_slope_per_bp:.4f} | {row.median_slope_per_repeat:.4f} |")
    lines += ["", f"Per-bp slopes differ by genome annotation (Kruskal–Wallis H={region_test.kruskal_h:.1f}, p={fmt_p(region_test.p_value)}, epsilon-squared={region_test.epsilon_squared:.4f}). After converting slopes to percentile ranks within each motif length, the region difference remains statistically detectable but becomes still smaller (H={adjusted_region_test.kruskal_h:.1f}, p={fmt_p(adjusted_region_test.p_value)}, epsilon-squared={adjusted_region_test.epsilon_squared:.4f}). This indicates that motif composition explains part, but not all, of the unadjusted region association.", "",
              "Within-motif percentile ranks are included in the result table so region differences can be assessed after removing the motif-length scale effect.", "",
              "With tens of thousands of loci, very small p-values can accompany modest distributional differences. Epsilon-squared and median differences should therefore be used to judge effect size; BH-adjusted pairwise tests and rank-biserial effects are provided in the results table.", "",
              "These are conditional comparisons among loci already selected at q < 0.05; they do not estimate context effects across all tested loci. Discovery also depends on the number and spacing of non-reference alleles and permutation resolution.", "",
              "## Outputs", "", "- `results/layer36_whole_mean_l2_fdr05_annotated.csv.gz`", "- `results/significant_slope_by_motif_length.csv`", "- `results/significant_slope_by_genome_region.csv`", "- `results/significant_locus_counts_by_motif_and_region.csv`", "- `results/significant_slope_omnibus_tests.csv`", "- `results/significant_slope_pairwise_tests.csv`", "- `figures/significant_l2_slopes_by_motif_and_region.png`", ""]
    report = report_dir / "SIGNIFICANT_SLOPES_BY_MOTIF_AND_REGION.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report}")
    print("SIGNIFICANT_SLOPE_CONTEXT_ANALYSIS=PASS")


if __name__ == "__main__":
    main()
