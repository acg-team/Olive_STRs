#!/usr/bin/env python3
"""Plot canonical single-layer GPN regression summaries."""

import argparse
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts/03_embeddings/gpn/scripts"))
from common import load_config


COLORS = {"l1": "#5e81ac", "l2": "#bf616a", "cosine": "#a3be8c"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    root = Path(load_config(args.config)["root"])
    data = pd.read_csv(root / "results/genome_wide_summary.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, pooling in zip(axes, ("whole_mean", "str_mean")):
        frame = data[data.pooling == pooling].set_index("metric").loc[list(COLORS)]
        axis.bar(
            [x.upper() for x in frame.index],
            frame.fraction_q_lt_005 * 100,
            color=[COLORS[x] for x in frame.index],
        )
        axis.set(title=pooling, ylabel="Loci with FDR < 0.05 (%)", ylim=(0, 100))
    fig.suptitle("GPN reference-anchored permutation discoveries (final layer)")
    fig.savefig(figures / "reference_anchored_fdr_by_metric.png", dpi=200)
    plt.close(fig)

    l2 = data[data.metric == "l2"].set_index("pooling").loc[
        ["whole_mean", "str_mean"]
    ]
    fig, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    axis.errorbar(
        l2.index,
        l2.median_slope,
        yerr=[
            l2.median_slope - l2.slope_q25,
            l2.slope_q75 - l2.median_slope,
        ],
        fmt="o",
        capsize=5,
        color=COLORS["l2"],
    )
    axis.set(ylabel="Median L2 slope per repeat (IQR)")
    axis.set_title("GPN final-layer reference-anchored L2 response")
    fig.savefig(figures / "genome_wide_l2_slopes.png", dpi=200)
    plt.close(fig)
    print("GPN_REFERENCE_PLOTS=PASS", flush=True)


if __name__ == "__main__":
    main()
