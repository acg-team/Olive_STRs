#!/usr/bin/env python3
"""Combine region-clustered SNP ADMIXTURE and STR dNMF ancestry plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/02_population_structure"
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"
SUPERFISHEL_STONE = [
    "#6388B4",
    "#FFAE34",
    "#EF6F6A",
    "#8CC2CA",
    "#55AD89",
    "#C3BC3F",
    "#BB7693",
    "#BAA094",
    "#A9B5AE",
    "#767676",
]
SUBSPECIES_ORDER = ("europaea", "sylvestris")
SUBSPECIES_COLORS = {"europaea": "#767676", "sylvestris": "#8CC2CA"}
REGION_ORDER = (
    "Western Mediterranean",
    "Central Mediterranean–Adriatic",
    "Eastern Mediterranean–Levant",
    "Iranian Plateau",
)
REGION_LABELS = {
    "Western Mediterranean": "West. Med.",
    "Central Mediterranean–Adriatic": "Central Med.",
    "Eastern Mediterranean–Levant": "East. Med.",
    "Iranian Plateau": "Iran",
}
REGION_COLORS = {
    "Western Mediterranean": "#6388B4",
    "Central Mediterranean–Adriatic": "#FFAE34",
    "Eastern Mediterranean–Levant": "#55AD89",
    "Iranian Plateau": "#BB7693",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admixture-data",
        type=Path,
        default=ANALYSIS_DIR / "source_data/admixture_k6_hierarchical_cluster_order.tsv",
    )
    parser.add_argument(
        "--dnmf-data",
        type=Path,
        default=ANALYSIS_DIR / "chr_str_var_gt_0.1/source_data/dnmf_k9_hierarchical_cluster_order.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--font-scale", type=float, default=1.4)
    parser.add_argument("--expected-samples", type=int, default=212)
    return parser.parse_args()


def set_style(font_scale: float) -> None:
    sns.set_theme(style="white", context="paper", font_scale=font_scale)
    base = 7 * font_scale
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5 * font_scale,
            "ytick.labelsize": 6.5 * font_scale,
            "legend.fontsize": 5.7 * font_scale,
            "legend.title_fontsize": 6 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def load_ordered_data(
    path: Path, component_prefix: str, expected_k: int, expected_samples: int
) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(path, sep="\t").sort_values(
        "clustered_position", kind="stable"
    )
    required = {
        "clustered_position",
        "sample_id",
        "subspecies",
        "geographic_region",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"{path} lacks columns: {sorted(required - set(data.columns))}")
    if len(data) != expected_samples or data["sample_id"].duplicated().any():
        raise ValueError(f"Expected {expected_samples} unique samples in {path}")
    components = [
        column for column in data.columns if column.startswith(component_prefix)
    ]
    if len(components) != expected_k:
        raise ValueError(f"Expected K={expected_k} in {path}, found {len(components)}")
    values = data[components].to_numpy(dtype=float)
    if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Invalid ancestry proportions in {path}")
    if tuple(data["geographic_region"].drop_duplicates()) != REGION_ORDER:
        raise ValueError(f"Unexpected geographic-region order in {path}")
    return data.reset_index(drop=True), components


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    components: list[str],
    title: str,
    font_scale: float,
) -> None:
    positions = data["clustered_position"].to_numpy(dtype=float)
    bottom = np.zeros(len(data))
    for color, component in zip(SUPERFISHEL_STONE, components):
        values = data[component].to_numpy(dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottom,
            width=1.0,
            color=color,
            linewidth=0,
        )
        bottom += values

    for position, subspecies, region in zip(
        positions, data["subspecies"], data["geographic_region"]
    ):
        ax.add_patch(
            Rectangle(
                (position - 0.5, 1.025),
                1.0,
                0.025,
                color=SUBSPECIES_COLORS[subspecies],
                clip_on=False,
                linewidth=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (position - 0.5, 1.060),
                1.0,
                0.025,
                color=REGION_COLORS[region],
                clip_on=False,
                linewidth=0,
            )
        )

    annotation_size = 5.5 * font_scale
    ax.text(
        -0.008,
        1.037,
        "Subspecies",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=annotation_size,
        clip_on=False,
    )
    ax.text(
        -0.008,
        1.072,
        "Region",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=annotation_size,
        clip_on=False,
    )
    ax.set_title(title, pad=24 * font_scale)
    ax.set_xlim(-0.5, positions.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ancestry proportion")
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    sns.despine(ax=ax, bottom=True)


def main() -> None:
    args = parse_args()
    set_style(args.font_scale)
    admixture, admixture_components = load_ordered_data(
        args.admixture_data, "ancestry_", 6, args.expected_samples
    )
    dnmf, dnmf_components = load_ordered_data(
        args.dnmf_data, "Q", 9, args.expected_samples
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(NATURE_DOUBLE_COLUMN_IN, 6.0),
        gridspec_kw={"hspace": 0.42},
    )
    draw_panel(
        axes[0],
        admixture,
        admixture_components,
        "SNP ADMIXTURE (K = 6)",
        args.font_scale,
    )
    draw_panel(
        axes[1],
        dnmf,
        dnmf_components,
        "STR dNMF (K = 9)",
        args.font_scale,
    )
    axes[1].set_xlabel("Samples hierarchically clustered within geographic region")

    subspecies_handles = [
        Patch(facecolor=SUBSPECIES_COLORS[label], label=label.capitalize())
        for label in SUBSPECIES_ORDER
    ]
    region_handles = [
        Patch(facecolor=REGION_COLORS[region], label=REGION_LABELS[region])
        for region in REGION_ORDER
    ]
    subspecies_legend = figure.legend(
        handles=subspecies_handles,
        title="Subspecies",
        loc="center left",
        bbox_to_anchor=(0.80, 0.64),
        frameon=False,
        ncol=1,
        handlelength=1.1,
    )
    figure.add_artist(subspecies_legend)
    figure.legend(
        handles=region_handles,
        title="Geographic region",
        loc="center left",
        bbox_to_anchor=(0.80, 0.37),
        frameon=False,
        ncol=1,
        handlelength=1.1,
    )
    figure.subplots_adjust(left=0.11, right=0.78, top=0.93, bottom=0.10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / "admixture_dnmf_hierarchical_clustered"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(f"{output_stem}.{extension}", bbox_inches="tight")
    plt.close(figure)
    print(f"Combined figure: {output_stem}")


if __name__ == "__main__":
    main()
