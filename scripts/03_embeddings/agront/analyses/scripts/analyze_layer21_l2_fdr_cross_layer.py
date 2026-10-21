#!/usr/bin/env python3
"""Test persistence of Layer 21 whole-mean L2 FDR-significant STR loci."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, hypergeom


ROOT = Path(__file__).resolve().parents[2]
ANALYSES = ROOT / "analyses"
REGRESSIONS = ROOT / "results/per_locus_regressions.csv.gz"
LAYERS = (11, 21, 31, 40)
FOCAL_LAYER = 21
TARGET_LAYERS = (11, 31, 40)


def format_p(value: float) -> str:
    return "<1e-300" if value == 0 else f"{value:.3g}"


def main():
    result_dir = ANALYSES / "results"
    figure_dir = ANALYSES / "figures"
    report_dir = ANALYSES / "report"
    for directory in (result_dir, figure_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        REGRESSIONS,
        usecols=[
            "str_id",
            "pooling",
            "layer",
            "metric",
            "predictor",
            "reference_included",
            "q_value",
            "fdr_0_05",
        ],
    )
    data = data[
        data.pooling.eq("whole_mean")
        & data.metric.eq("l2")
        & data.predictor.eq("absolute_repeat_difference_from_reference")
        & ~data.reference_included.astype(bool)
    ].copy()
    if set(data.layer.unique()) != set(LAYERS):
        raise RuntimeError(f"Unexpected ArgoNT layers: {sorted(data.layer.unique())}")
    if data.groupby(["str_id", "layer"]).size().ne(1).any():
        raise RuntimeError("Duplicate or missing locus/layer rows")
    if not data.fdr_0_05.astype(bool).eq(data.q_value.lt(0.05)).all():
        raise RuntimeError("Stored FDR flags disagree with q-values")

    wide = data.pivot(index="str_id", columns="layer", values="fdr_0_05")
    wide = wide.reindex(columns=LAYERS)
    if wide.isna().any().any():
        raise RuntimeError("The four layers do not contain identical STR locus sets")
    wide = wide.astype(bool)
    wide.columns = [f"layer_{layer}_significant" for layer in wide.columns]
    wide = wide.reset_index()

    focal = wide[f"layer_{FOCAL_LAYER}_significant"].to_numpy()
    n_loci = len(wide)
    n_focal = int(focal.sum())
    rows = []
    for layer in TARGET_LAYERS:
        target = wide[f"layer_{layer}_significant"].to_numpy()
        n_target = int(target.sum())
        overlap = int((focal & target).sum())
        focal_only = n_focal - overlap
        target_only = n_target - overlap
        neither = n_loci - overlap - focal_only - target_only
        union = overlap + focal_only + target_only
        expected = n_focal * n_target / n_loci
        odds_ratio, fisher_p = fisher_exact(
            [[overlap, focal_only], [target_only, neither]],
            alternative="greater",
        )
        rows.append(
            {
                "target_layer": layer,
                "common_tested_loci": n_loci,
                "layer21_significant_loci": n_focal,
                "target_significant_loci": n_target,
                "overlap": overlap,
                "layer21_replication_fraction": overlap / n_focal,
                "target_significant_fraction": n_target / n_loci,
                "fraction_target_set_from_layer21": overlap / n_target,
                "union": union,
                "jaccard": overlap / union,
                "expected_overlap_independence": expected,
                "observed_to_expected": overlap / expected,
                "fisher_odds_ratio": odds_ratio,
                "fisher_p_greater": fisher_p,
                "hypergeometric_p": hypergeom.sf(
                    overlap - 1, n_loci, n_target, n_focal
                ),
            }
        )
    overlap_table = pd.DataFrame(rows)
    overlap_table.to_csv(
        result_dir / "layer21_l2_fdr_overlap_by_layer.csv", index=False
    )

    persistence = wide.loc[wide[f"layer_{FOCAL_LAYER}_significant"]].copy()
    other_columns = [f"layer_{layer}_significant" for layer in TARGET_LAYERS]
    persistence["other_significant_layers"] = persistence[other_columns].sum(axis=1)
    persistence.to_csv(
        result_dir / "layer21_l2_fdr_cross_layer_membership.csv.gz",
        index=False,
        compression="gzip",
    )
    persistence_counts = (
        persistence.other_significant_layers.value_counts()
        .reindex(range(4), fill_value=0)
        .rename_axis("other_significant_layers")
        .reset_index(name="layer21_significant_loci")
    )
    persistence_counts["fraction_layer21_significant_loci"] = (
        persistence_counts.layer21_significant_loci / n_focal
    )
    persistence_counts.to_csv(
        result_dir / "layer21_l2_fdr_persistence_counts.csv", index=False
    )

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.4,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    positions = np.arange(len(overlap_table))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        overlap_table.layer21_replication_fraction * 100,
        width,
        color="#5e81ac",
        label="Observed overlap",
    )
    axes[0].bar(
        positions + width / 2,
        overlap_table.target_significant_fraction * 100,
        width,
        color="#d8dee9",
        edgecolor="#666666",
        linewidth=0.7,
        label="Expected under independence",
    )
    axes[0].set_xticks(
        positions, [str(layer) for layer in overlap_table.target_layer]
    )
    axes[0].set(
        xlabel="ArgoNT layer",
        ylabel="Layer 21 significant loci (%)",
        title="Replication at another layer",
    )
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    for position, row in enumerate(overlap_table.itertuples()):
        axes[0].text(
            position - width / 2,
            row.layer21_replication_fraction * 100 + 1,
            f"{row.overlap:,}",
            ha="center",
            va="bottom",
        )

    axes[1].bar(
        persistence_counts.other_significant_layers,
        persistence_counts.fraction_layer21_significant_loci * 100,
        color="#bf616a",
        width=0.7,
    )
    axes[1].set_xticks(range(4))
    axes[1].set(
        xlabel="Other significant layers (of 3)",
        ylabel="Layer 21 significant loci (%)",
        title="Cross-layer persistence",
    )
    for row in persistence_counts.itertuples():
        axes[1].text(
            row.other_significant_layers,
            row.fraction_layer21_significant_loci * 100 + 1,
            f"{row.layer21_significant_loci:,}",
            ha="center",
            va="bottom",
        )
    for axis in axes:
        sns.despine(ax=axis)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Whole-mean L2 FDR-significant loci at ArgoNT Layer 21")
    stem = "layer21_l2_fdr_cross_layer_consistency"
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    all_other = int(
        persistence_counts.loc[
            persistence_counts.other_significant_layers.eq(3),
            "layer21_significant_loci",
        ].iloc[0]
    )
    none_other = int(
        persistence_counts.loc[
            persistence_counts.other_significant_layers.eq(0),
            "layer21_significant_loci",
        ].iloc[0]
    )
    none_fraction_text = (
        "<0.01%" if 0 < none_other / n_focal < 0.0001
        else f"{none_other / n_focal:.1%}"
    )
    lines = [
        "# Layer 21 whole-mean L2 FDR consistency across ArgoNT layers",
        "",
        f"The analysis starts from the **{n_focal:,} STR loci** passing `q < 0.05` "
        "for the canonical reference-anchored, whole-mean L2 test at Layer 21. "
        "Existing per-locus regressions are used; no regressions are rerun.",
        "",
        "| Other layer | Significant loci | Shared with Layer 21 | Layer 21 replicated | Jaccard | Expected overlap | Enrichment | Fisher OR | p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overlap_table.itertuples():
        lines.append(
            f"| {row.target_layer} | {row.target_significant_loci:,} | "
            f"{row.overlap:,} | {row.layer21_replication_fraction:.1%} | "
            f"{row.jaccard:.3f} | {row.expected_overlap_independence:,.1f} | "
            f"{row.observed_to_expected:.2f}× | {row.fisher_odds_ratio:.2f} | "
            f"{format_p(row.fisher_p_greater)} |"
        )
    lines += [
        "",
        f"Among Layer 21 discoveries, **{all_other:,} ({all_other / n_focal:.1%})** "
        "are significant at all three other sampled layers and "
        f"**{none_other:,} ({none_fraction_text})** are unique to Layer 21.",
        "",
        "The expected-overlap bars and enrichment statistics use independence of "
        "layer-specific significance as the chance baseline. Shared allele sets and "
        "the same permutation-test power structure make this a descriptive null rather "
        "than proof that layer representations are biologically equivalent.",
        "",
        "## Outputs",
        "",
        "- `results/layer21_l2_fdr_overlap_by_layer.csv`",
        "- `results/layer21_l2_fdr_cross_layer_membership.csv.gz`",
        "- `results/layer21_l2_fdr_persistence_counts.csv`",
        "- `figures/layer21_l2_fdr_cross_layer_consistency.png`",
        "- `figures/layer21_l2_fdr_cross_layer_consistency.pdf`",
        "",
    ]
    report = report_dir / "LAYER21_L2_FDR_CROSS_LAYER_CONSISTENCY.md"
    report.write_text("\n".join(lines))
    print(overlap_table.to_string(index=False), flush=True)
    print(f"report={report}", flush=True)
    print("LAYER21_L2_FDR_CROSS_LAYER=PASS", flush=True)


if __name__ == "__main__":
    main()
