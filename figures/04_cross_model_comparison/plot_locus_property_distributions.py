#!/usr/bin/env python3
"""Plot allele-count and repeat-length-variance distributions by model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "scripts/04_cross_model_comparison"
CONFIG = ROOT / "config.json"
FIGURES = Path(__file__).resolve().parent / "output"
RESULTS = ROOT / "results"
VARIANCE = Path(os.environ.get("OLIVE_STR_VARIANCE", "inputs/str_length_variance.csv"))
MODELS = ("argont", "plantcad2", "gpn")
DISPLAY = {"argont": "ArgoNT", "plantcad2": "PlantCAD2", "gpn": "GPN"}
COLORS = {
    "argont": "#0173B2",
    "plantcad2": "#DE8F05",
    "gpn": "#029E73",
}


def load_data() -> pd.DataFrame:
    """Load the matched primary L2 tests and shared locus properties."""
    config = json.loads(CONFIG.read_text())
    frames = []
    for model in MODELS:
        frame = pd.read_csv(config["primary_results"][model])
        frame = frame[
            frame.pooling.eq(config["pooling"])
            & frame.layer.eq(config[f"{model}_layer"])
            & frame.metric.eq("l2")
        ][
            [
                "str_id",
                "n_alternate_alleles",
                "q_value",
                "fdr_0_05",
            ]
        ].copy()
        if len(frame) != config["expected_loci"]:
            raise RuntimeError(f"{model}: unexpected number of L2 loci")
        if frame.str_id.duplicated().any():
            raise RuntimeError(f"{model}: duplicate STR identifiers")
        if not frame.fdr_0_05.astype(bool).eq(frame.q_value.lt(0.05)).all():
            raise RuntimeError(f"{model}: inconsistent FDR flag")
        frame["model"] = model
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    variance = pd.read_csv(VARIANCE).rename(
        columns={"variance": "repeat_length_variance"}
    )
    if variance.str_id.duplicated().any():
        raise RuntimeError("Duplicate STR identifiers in variance table")
    data = data.merge(
        variance[["str_id", "repeat_length_variance"]],
        on="str_id",
        how="left",
        validate="many_to_one",
    )
    if data.repeat_length_variance.isna().any():
        missing = data.loc[data.repeat_length_variance.isna(), "str_id"].nunique()
        raise RuntimeError(f"Variance is missing for {missing} matched loci")
    if data.repeat_length_variance.le(0).any():
        raise RuntimeError("Expected strictly positive repeat-length variances")

    counts = data.pivot(
        index="str_id",
        columns="model",
        values="n_alternate_alleles",
    )
    if not counts.nunique(axis=1).eq(1).all():
        raise RuntimeError("The three models do not use identical allele counts")
    return data


def set_style() -> None:
    """Apply the same compact publication style as the slope distribution."""
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_distribution(
    data: pd.DataFrame,
    *,
    column: str,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    output_stem: str,
    log_x: bool = False,
) -> pd.DataFrame:
    """Draw a three-model stacked distribution with FDR highlighting."""
    set_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.0, 4.0),
        sharey=True,
    )
    summary_rows = []
    for ax, model in zip(axes, MODELS):
        part = data[data.model.eq(model)].copy()
        values = part[column].to_numpy(dtype=float)
        significant = part.fdr_0_05.astype(bool).to_numpy()
        color = COLORS[model]
        median = float(np.median(values))
        fdr_fraction = float(significant.mean())
        summary_rows.append(
            {
                "property": column,
                "model": DISPLAY[model],
                "loci": len(part),
                "minimum": float(values.min()),
                "median": median,
                "mean": float(values.mean()),
                "maximum": float(values.max()),
                "fdr_0_05_loci": int(significant.sum()),
                "fdr_0_05_fraction": fdr_fraction,
            }
        )
        ax.hist(
            [values[~significant], values[significant]],
            bins=bins,
            stacked=True,
            color=[(*plt.matplotlib.colors.to_rgb(color), 0.20), color],
            edgecolor="none",
        )
        ax.axvline(
            median,
            color="#B22222",
            linestyle="--",
            linewidth=1.1,
        )
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set(
            xlabel="",
            title=(
                f"{DISPLAY[model]}\n"
                f"FDR < 0.05: {fdr_fraction:.1%}"
            ),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    middle_color = plt.matplotlib.colors.to_rgb(COLORS["plantcad2"])
    axes[1].legend(
        handles=[
            Patch(
                facecolor=COLORS["plantcad2"],
                edgecolor="none",
                label="FDR < 0.05",
            ),
            Patch(
                facecolor=(*middle_color, 0.20),
                edgecolor="none",
                label="q ≥ 0.05",
            ),
        ],
        frameon=False,
        loc="upper right",
    )
    axes[0].set_ylabel("Loci (log scale)")
    fig.supxlabel(xlabel, y=0.025)
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(
        top=0.70,
        bottom=0.23,
        left=0.075,
        right=0.99,
        wspace=0.08,
    )
    fig.savefig(FIGURES / f"{output_stem}.png", dpi=300)
    fig.savefig(FIGURES / f"{output_stem}.pdf")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_data()

    minimum_count = int(data.n_alternate_alleles.min())
    maximum_count = int(data.n_alternate_alleles.max())
    count_bins = np.arange(minimum_count - 0.5, maximum_count + 1.5, 1)
    count_summary = draw_distribution(
        data,
        column="n_alternate_alleles",
        bins=count_bins,
        xlabel="Number of distinct observed alternate allele lengths",
        title="Distinct allele-count distributions across all tested loci",
        output_stem="all_tested_distinct_allele_count_distributions",
    )

    variance_values = data.repeat_length_variance.to_numpy(dtype=float)
    variance_bins = np.geomspace(
        variance_values.min(),
        variance_values.max(),
        81,
    )
    variance_summary = draw_distribution(
        data,
        column="repeat_length_variance",
        bins=variance_bins,
        xlabel="STR repeat-length variance (log scale)",
        title="STR repeat-length variance across all tested loci",
        output_stem="all_tested_str_variance_distributions",
        log_x=True,
    )

    pd.concat(
        [count_summary, variance_summary],
        ignore_index=True,
    ).to_csv(
        RESULTS / "all_tested_locus_property_distribution_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
