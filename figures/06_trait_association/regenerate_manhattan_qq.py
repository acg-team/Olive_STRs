#!/usr/bin/env python3
"""Regenerate the main STR-association Manhattan/QQ figure in Nature style."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] / "scripts/06_trait_association"
sys.path.insert(0, str(ROOT))
from analyze_str_oil_association import manhattan_coordinates
TRAITS = [
    ("oleic_acid", "Oleic acid"),
    ("linoleic_acid", "Linoleic acid"),
    ("log_oleic_linoleic_ratio", "Log O/L ratio"),
]


def set_nature_style() -> None:
    """Set a compact, vector-friendly journal figure theme."""
    sns.set_theme(
        style="ticks",
        context="paper",
        font="DejaVu Sans",
        font_scale=1.4,
        rc={
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def main() -> None:
    set_nature_style()
    thresholds = json.loads(
        (ROOT / "source_data/multiple_testing_thresholds.json").read_text()
    )
    results = [
        pd.read_csv(ROOT / "results" / f"{trait}_association.tsv.gz", sep="\t")
        for trait, _ in TRAITS
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11.8, 9.8),
        gridspec_kw={"width_ratios": [3.35, 1], "wspace": 0.18, "hspace": 0.34},
    )
    chromosome_colors = np.array(["#3569A8", "#B4B4B4"])

    for row, ((_, title), result) in enumerate(zip(TRAITS, results)):
        frame, ticks, labels = manhattan_coordinates(result)
        manhattan = axes[row, 0]
        manhattan.scatter(
            frame["plot_position"],
            -np.log10(frame["pooled_p"].clip(lower=1e-300)),
            c=chromosome_colors[frame["chrom_index"].to_numpy() % 2],
            s=4.2,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        manhattan.axhline(
            -math.log10(thresholds["study_bonferroni"]),
            color="#B2182B",
            linewidth=0.9,
            linestyle=(0, (4, 2)),
        )
        manhattan.axhline(
            -math.log10(thresholds["suggestive"]),
            color="#555555",
            linewidth=0.8,
            linestyle=(0, (1.2, 2)),
        )
        manhattan.set_xticks(ticks, labels)
        manhattan.set_ylabel(r"$-\log_{10}(P)$")
        manhattan.set_title(title, loc="left", pad=5)
        if row == 2:
            manhattan.set_xlabel("Chromosome")
        sns.despine(ax=manhattan)

        qq = axes[row, 1]
        pvalues = np.sort(result["pooled_p"].dropna().to_numpy())
        expected = -np.log10(
            (np.arange(1, len(pvalues) + 1) - 0.5) / len(pvalues)
        )
        observed = -np.log10(pvalues)
        limit = max(expected.max(), observed.max()) * 1.03
        qq.scatter(
            expected,
            observed,
            s=6,
            color="#3569A8",
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        qq.plot([0, limit], [0, limit], color="#202020", linewidth=0.8)
        inflation = np.nanmedian(chi2.isf(pvalues, 1)) / chi2.ppf(0.5, 1)
        qq.text(
            0.06,
            0.94,
            rf"$\lambda_{{GC}}={inflation:.2f}$",
            transform=qq.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        qq.set_xlim(-0.03, limit)
        qq.set_ylim(-0.03, limit)
        qq.set_xlabel(r"Expected $-\log_{10}(P)$")
        qq.set_ylabel(r"Observed $-\log_{10}(P)$")
        sns.despine(ax=qq)

    output = ROOT / "figures/pooled_lmm_manhattan_qq"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(
            output.with_suffix(suffix),
            bbox_inches="tight",
            dpi=600 if suffix == ".png" else 300,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
