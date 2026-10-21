#!/usr/bin/env python3
"""Plot canonical reference-anchored PlantCAD2 regression summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts/03_embeddings/plantcad2/scripts"))
from common import load_config


COLORS = {"l1": "#5e81ac", "l2": "#bf616a", "cosine": "#a3be8c"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["root"])
    data = pd.read_csv(root / "results/genome_wide_summary.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, pooling in enumerate(("whole_mean", "str_mean")):
        for column, metric in enumerate(("l1", "l2", "cosine")):
            frame = data[(data.pooling == pooling) & (data.metric == metric)].sort_values(
                "layer"
            )
            lower = frame.median_slope - frame.slope_q25
            upper = frame.slope_q75 - frame.median_slope
            axes[row, column].errorbar(
                frame.layer,
                frame.median_slope,
                yerr=[lower, upper],
                marker="o",
                capsize=4,
                color=COLORS[metric],
            )
            axes[row, column].axhline(0, color="black", linewidth=0.8)
            axes[row, column].set(
                title=f"{pooling}: {metric.upper()}",
                xlabel="PlantCAD2 layer",
                ylabel="Median slope per repeat (IQR)",
            )
    fig.suptitle("Reference-anchored STR embedding slopes across layers", fontsize=15)
    fig.savefig(figures / "reference_anchored_slope_by_layer.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, pooling in zip(axes, ("whole_mean", "str_mean")):
        for metric in ("l1", "l2", "cosine"):
            frame = data[(data.pooling == pooling) & (data.metric == metric)].sort_values(
                "layer"
            )
            axis.plot(
                frame.layer,
                frame.fraction_q_lt_005 * 100,
                marker="o",
                label=metric.upper(),
                color=COLORS[metric],
            )
        axis.set(
            title=pooling,
            xlabel="PlantCAD2 layer",
            ylabel="Loci with FDR < 0.05 (%)",
            ylim=(0, 100),
        )
        axis.legend()
    fig.suptitle("Reference-anchored permutation discoveries by layer", fontsize=15)
    fig.savefig(figures / "reference_anchored_fdr_by_layer.png", dpi=200)
    plt.close(fig)

    l2 = data[data.metric == "l2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, pooling in zip(axes, ("whole_mean", "str_mean")):
        frame = l2[l2.pooling == pooling].sort_values("layer")
        axis.errorbar(
            frame.layer,
            frame.median_slope,
            yerr=[
                frame.median_slope - frame.slope_q25,
                frame.slope_q75 - frame.median_slope,
            ],
            marker="o",
            capsize=4,
            color=COLORS["l2"],
        )
        axis.set(
            title=pooling,
            xlabel="PlantCAD2 layer",
            ylabel="Median L2 slope per repeat (IQR)",
        )
    fig.suptitle("Genome-wide reference-anchored L2 response", fontsize=14)
    fig.savefig(figures / "genome_wide_l2_slopes.png", dpi=200)
    plt.close(fig)

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.6,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    whole_mean = data[data.pooling.eq("whole_mean")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for metric in ("l1", "l2", "cosine"):
        frame = whole_mean[whole_mean.metric.eq(metric)].sort_values("layer")
        axes[0].plot(
            frame.layer,
            frame.median_slope,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=COLORS[metric],
            label=metric.upper(),
        )
        axes[1].plot(
            frame.layer,
            frame.fraction_q_lt_005 * 100,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=COLORS[metric],
            label=metric.upper(),
        )
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="PlantCAD2 layer",
        ylabel="Median slope per repeat (log scale)",
        title="Effect of hidden layer and distance metric\non embedding perturbation",
    )
    axes[1].set(
        xlabel="PlantCAD2 layer",
        ylabel="Loci with FDR < 0.05 (%)",
        title="Fraction of significant STR loci",
        ylim=(0, 40),
    )
    for axis in axes:
        axis.set_xticks(sorted(whole_mean.layer.unique()))
        sns.despine(ax=axis)
    axes[1].legend(loc="lower left", frameon=False)
    fig.savefig(figures / "whole_mean_slope_fdr_by_layer.png", dpi=300)
    fig.savefig(figures / "whole_mean_slope_fdr_by_layer.pdf")
    plt.close(fig)
    print("PLANTCAD2_REFERENCE_PLOTS=PASS", flush=True)


if __name__ == "__main__":
    main()
