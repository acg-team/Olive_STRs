#!/usr/bin/env python3
"""Plot ArgoNT Layer 21 whole-mean L2 slope against permutation q-value."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "scripts/03_embeddings/agront"
REGRESSIONS = ROOT / "results/per_locus_regressions.csv.gz"
FIGURES = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(
        REGRESSIONS,
        usecols=[
            "str_id",
            "pooling",
            "layer",
            "metric",
            "slope",
            "q_value",
            "fdr_0_05",
        ],
    )
    data = data[
        data.pooling.eq("whole_mean")
        & data.layer.eq(21)
        & data.metric.eq("l2")
    ].copy()
    if len(data) != 62_847 or data.str_id.nunique() != 62_847:
        raise RuntimeError(
            f"Unexpected Layer 21 whole-mean L2 dimensions: {len(data):,} rows"
        )
    if data.q_value.le(0).any():
        raise RuntimeError("The −log10(q) transform requires positive q-values")
    if not data.fdr_0_05.astype(bool).eq(data.q_value.lt(0.05)).all():
        raise RuntimeError("Stored FDR flags disagree with q-values")
    data["negative_log10_q"] = -np.log10(data.q_value)

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
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    nonsignificant = data.loc[~data.fdr_0_05.astype(bool)]
    significant = data.loc[data.fdr_0_05.astype(bool)]
    ax.scatter(
        nonsignificant.slope,
        nonsignificant.negative_log10_q,
        s=7,
        alpha=0.18,
        color="#a7a9ac",
        edgecolors="none",
        rasterized=True,
        label=f"q ≥ 0.05 (n={len(nonsignificant):,})",
    )
    ax.scatter(
        significant.slope,
        significant.negative_log10_q,
        s=8,
        alpha=0.35,
        color="#0072b2",
        edgecolors="none",
        rasterized=True,
        label=f"q < 0.05 (n={len(significant):,})",
    )
    ax.axhline(
        -np.log10(0.05),
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        label="FDR = 0.05",
    )
    ax.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
    ax.set(
        xlabel="L2 slope per repeat",
        ylabel="−log10(BH-adjusted permutation q-value)",
        title="ArgoNT Layer 21 whole-mean L2",
    )
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = "layer21_whole_mean_l2_slope_vs_qvalue"
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)
    print(
        f"loci={len(data):,} significant={len(significant):,} "
        f"figure={FIGURES / (stem + '.png')}",
        flush=True,
    )
    print("ARGONT_LAYER21_SLOPE_QVALUE_PLOT=PASS", flush=True)


if __name__ == "__main__":
    main()
