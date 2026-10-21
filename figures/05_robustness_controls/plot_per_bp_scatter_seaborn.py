#!/usr/bin/env python3
"""Render the primary per-bp matched-indel scatter comparison with Seaborn."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "scripts/05_robustness_controls"
RESULTS = ROOT / "matched_indel_control/results"
FIGURES = Path(__file__).resolve().parent / "output"
MODELS = {
    "ArgoNT": {"layer": 21, "color": "#0072B2"},
    "PlantCAD2": {"layer": 36, "color": "#E69F00"},
    "GPN": {"layer": 25, "color": "#009E73"},
}


def main():
    data = pd.read_csv(RESULTS / "str_vs_control_per_locus.csv.gz")
    primary = data[
        (data.pooling == "whole_mean")
        & (data.metric == "l2")
        & (data.predictor == "absolute_bp_difference_from_reference")
        & data.apply(lambda row: row.layer == MODELS[row.model]["layer"], axis=1)
    ].copy()
    if primary.groupby("model").str_id.nunique().to_dict() != {
        model: 1000 for model in MODELS
    }:
        raise RuntimeError("Expected 1,000 primary per-bp loci per model")

    sns.set_theme(style="ticks", context="notebook", font_scale=1.6)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharex=False, sharey=False)
    for ax, model in zip(axes, MODELS):
        frame = primary[primary.model == model]
        color = MODELS[model]["color"]
        sns.scatterplot(
            data=frame,
            x="control_mean_slope",
            y="str_slope",
            color=color,
            alpha=0.62,
            s=34,
            linewidth=0,
            ax=ax,
        )
        limits = [
            min(frame.control_mean_slope.min(), frame.str_slope.min()),
            max(frame.control_mean_slope.max(), frame.str_slope.max()),
        ]
        padding = 0.04 * max(limits[1] - limits[0], 1e-12)
        limits = [limits[0] - padding, limits[1] + padding]
        ax.plot(limits, limits, "--", color="black", linewidth=1.4)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        correlation = frame[["str_slope", "control_mean_slope"]].corr().iloc[0, 1]
        fraction_above = (frame.str_slope > frame.control_mean_slope).mean()
        ax.text(
            0.04,
            0.96,
            f"$r$ = {correlation:.2f}\n{fraction_above:.0%} above",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.set_title(model)
        ax.set_xlabel("Matched-indel mean L2 slope per bp")
        ax.set_ylabel("STR L2 slope per bp")
    sns.despine()
    fig.tight_layout()
    stem = FIGURES / "three_model_str_vs_matched_indel_l2_slopes_per_bp"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"output={stem} SEABORN_PER_BP_SCATTER=PASS")


if __name__ == "__main__":
    main()
