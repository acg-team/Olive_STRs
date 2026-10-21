#!/usr/bin/env python3
"""Plot genotype–phenotype relationships for four manuscript showcase STRs."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ASSOCIATION_DIR = REPO_ROOT / "scripts/06_trait_association"
PROJECT_DIR = REPO_ROOT / "scripts"

STR_MATRIX = Path(os.environ.get(
    "OLIVE_STR_MATRIX", ASSOCIATION_DIR / "inputs/merged_str_length.csv"
))
PHENOTYPES = ASSOCIATION_DIR / "source_data" / "phenotypes.tsv"
FOLLOWUP = ASSOCIATION_DIR / "results" / "fdr_0.05_followup_loci.tsv"

FIGURE_DIR = SCRIPT_DIR / "output"
SOURCE_DIR = SCRIPT_DIR / "output/source_data"

SHOWCASE_LOCI = [
    "CACTIH010001827.1_963000",
    "CACTIH010001808.1_854553",
    "19_16919767",
    "4_5035216",
]

TRAITS = {
    "oleic_acid": ("oleic_acid", "Oleic acid", "#4F6980"),
    "linoleic_acid": ("linoleic_acid", "Linoleic acid", "#F47942"),
}

DATASET_LABELS = {"center_1": "Dataset 1", "center_2": "Dataset 2"}


def load_selected_genotypes(sample_ids: list[str]) -> pd.DataFrame:
    selected = []
    for chunk in pd.read_csv(
        STR_MATRIX,
        usecols=["str_id", *sample_ids],
        chunksize=20_000,
    ):
        hit = chunk[chunk["str_id"].isin(SHOWCASE_LOCI)]
        if not hit.empty:
            selected.append(hit)
    if not selected:
        raise ValueError("None of the showcase loci were found in the STR matrix")
    matrix = pd.concat(selected, ignore_index=True)
    missing = sorted(set(SHOWCASE_LOCI) - set(matrix["str_id"]))
    if missing:
        raise ValueError(f"Showcase loci absent from STR matrix: {', '.join(missing)}")
    if matrix["str_id"].duplicated().any():
        raise ValueError("Duplicate showcase loci in the STR matrix")
    return matrix.set_index("str_id").loc[SHOWCASE_LOCI]


def prepare_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    phenotype = pd.read_csv(PHENOTYPES, sep="\t")
    sample_ids = phenotype["Sample_ID"].tolist()
    genotype = load_selected_genotypes(sample_ids)

    records = []
    phenotype_by_sample = phenotype.set_index("Sample_ID")
    for locus in SHOWCASE_LOCI:
        for sample_id in sample_ids:
            records.append(
                {
                    "str_id": locus,
                    "sample_id": sample_id,
                    "dataset": DATASET_LABELS[
                        phenotype_by_sample.at[sample_id, "center"]
                    ],
                    "center": phenotype_by_sample.at[sample_id, "center"],
                    "mean_repeat_count": genotype.at[locus, sample_id],
                    "oleic_acid": phenotype_by_sample.at[
                        sample_id, "OleicAcid_Percentage"
                    ],
                    "linoleic_acid": phenotype_by_sample.at[
                        sample_id, "LinoleicAcid_Percentage"
                    ],
                }
            )
    plot_data = pd.DataFrame.from_records(records)

    followup = pd.read_csv(FOLLOWUP, sep="\t")
    association = followup[
        followup["str_id"].isin(SHOWCASE_LOCI)
        & followup["trait"].isin(TRAITS)
    ].copy()
    expected = {(locus, trait) for locus in SHOWCASE_LOCI for trait in TRAITS}
    observed = set(zip(association["str_id"], association["trait"]))
    missing_results = sorted(expected - observed)
    if missing_results:
        raise ValueError(f"Missing follow-up results: {missing_results}")
    return plot_data, association


def p_text(value: float) -> str:
    return f"{value:.1e}"


def draw_trait_panel(
    ax: plt.Axes,
    locus: str,
    panel_data: pd.DataFrame,
    association: pd.DataFrame,
    rng: np.random.Generator,
    trait: str,
) -> None:
    called = panel_data.dropna(subset=["mean_repeat_count"]).copy()
    unique_lengths = np.sort(called["mean_repeat_count"].unique())
    span = max(float(np.ptp(unique_lengths)), 1.0)
    jitter_sd = 0.012 * span

    column, label, color = TRAITS[trait]
    jitter = rng.normal(0, jitter_sd, len(called))
    ax.scatter(
        called["mean_repeat_count"] + jitter,
        called[column],
        s=42,
        marker="o",
        facecolor=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.66,
        zorder=2,
    )
    valid = called[["mean_repeat_count", column]].dropna()
    if valid["mean_repeat_count"].nunique() >= 2:
        slope, intercept = np.polyfit(valid["mean_repeat_count"], valid[column], 1)
        xx = np.linspace(
            valid["mean_repeat_count"].min(),
            valid["mean_repeat_count"].max(),
            100,
        )
        ax.plot(xx, intercept + slope * xx, color=color, linewidth=2.3, zorder=3)

    result = association.set_index("trait").loc[trait]
    annotation = f"P={p_text(result.pooled_p)}, FDR={result.pooled_fdr:.3g}"
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
    )
    motif = result["motif"]
    ax.set_title(label, pad=8)
    ax.text(
        0.97,
        0.03,
        f"Motif: {motif}   n={len(called)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
    )
    ax.set_xlabel("Mean STR repeat count")
    ax.set_ylabel(f"{label} (%)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"dpi": 350} if suffix == ".png" else {}
        fig.savefig(Path(f"{stem}{suffix}"), bbox_inches="tight", **kwargs)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.4,
        rc={
            "axes.linewidth": 1.1,
            "font.family": "sans-serif",
            "legend.frameon": False,
        },
    )
    plot_data, association = prepare_plot_data()
    plot_data.to_csv(
        SOURCE_DIR / "showcase_genotypes_phenotypes.tsv",
        sep="\t",
        index=False,
        na_rep="NA",
    )
    association.to_csv(
        SOURCE_DIR / "showcase_association_statistics.tsv",
        sep="\t",
        index=False,
        na_rep="NA",
    )

    rng = np.random.default_rng(20260730)
    for locus in SHOWCASE_LOCI:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        locus_data = plot_data[plot_data["str_id"] == locus]
        locus_association = association[association["str_id"] == locus]
        for ax, trait in zip(axes, TRAITS):
            draw_trait_panel(
                ax,
                locus,
                locus_data,
                locus_association,
                rng,
                trait,
            )
        title = locus
        if locus == "4_5035216":
            title += " (near OMES)"
        fig.suptitle(title, fontsize=16, y=0.99)
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            bottom=0.15,
            top=0.84,
            wspace=0.28,
        )
        save_figure(fig, FIGURE_DIR / f"{locus}_genotype_phenotype")
        plt.close(fig)


if __name__ == "__main__":
    main()
