#!/usr/bin/env python3
"""Plot K=9 dNMF ancestry clustered independently within geographic regions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/02_population_structure"
DEFAULT_ANALYSIS_DIR = ANALYSIS_DIR / "chr_str_var_gt_0.1"
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"
ANCESTRY_COLORS = [
    "#6388B4", "#FFAE34", "#EF6F6A", "#8CC2CA", "#55AD89",
    "#C3BC3F", "#BB7693", "#BAA094", "#A9B5AE",
]
SUBSPECIES_COLORS = {"europaea": "#767676", "sylvestris": "#8CC2CA"}
SUBSPECIES_ORDER = ("europaea", "sylvestris")
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
REGION_GAP = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-data", type=Path,
        default=DEFAULT_ANALYSIS_DIR / "source_data/dnmf_k9_snp_plot_order.tsv",
    )
    parser.add_argument(
        "--admixture-order",
        type=Path,
        default=ANALYSIS_DIR / "source_data/admixture_k6_hierarchical_cluster_order.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--expected-samples", type=int, default=212)
    parser.add_argument("--expected-k", type=int, default=9)
    return parser.parse_args()


def set_nature_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.7,
            "legend.title_fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def cluster_samples(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, float, float]:
    data = pd.read_csv(args.plot_data, sep="\t")
    required = {"plot_position", "sample_id", "Country", "Subspecies", "subspecies"}
    q_columns = [column for column in data.columns if column.startswith("Q")]
    if not required.issubset(data.columns):
        raise ValueError(f"Plot data lacks columns: {sorted(required - set(data.columns))}")
    if len(data) != args.expected_samples or data["sample_id"].duplicated().any():
        raise ValueError(f"Expected {args.expected_samples} unique samples")
    if len(q_columns) != args.expected_k:
        raise ValueError(f"Expected K={args.expected_k}, observed {len(q_columns)} Q columns")
    q = data[q_columns].to_numpy(dtype=float)
    if np.any(q < 0) or not np.allclose(q.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Invalid ancestry proportions")

    order = pd.read_csv(args.admixture_order, sep="\t")
    order_required = {
        "clustered_position",
        "sample_id",
        "Country",
        "Subspecies",
        "subspecies",
        "geographic_region",
    }
    if not order_required.issubset(order.columns):
        raise ValueError(
            f"ADMIXTURE order lacks columns: {sorted(order_required - set(order.columns))}"
        )
    if len(order) != args.expected_samples or order["sample_id"].duplicated().any():
        raise ValueError(f"Expected {args.expected_samples} unique ADMIXTURE-order samples")
    if set(order["sample_id"]) != set(data["sample_id"]):
        raise ValueError("ADMIXTURE order and dNMF data contain different sample sets")

    ordered = order[list(order_required)].merge(
        data[["sample_id", *q_columns]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    ordered = ordered.sort_values("clustered_position", kind="stable").reset_index(drop=True)
    observed_regions = tuple(ordered["geographic_region"].drop_duplicates())
    if observed_regions != REGION_ORDER:
        raise ValueError(
            f"Unexpected geographic-region order: {observed_regions}"
        )
    q = ordered[q_columns].to_numpy(dtype=float)
    ordered_indices = []
    linkage_tables = []
    admixture_adjacent_distances = []
    clustered_adjacent_distances = []
    for region in REGION_ORDER:
        indices = np.flatnonzero(ordered["geographic_region"].to_numpy() == region)
        if len(indices) < 2:
            raise ValueError(f"Need at least two samples from {region} for clustering")
        distances = pdist(q[indices], metric="jensenshannon")
        tree = linkage(distances, method="average")
        ordered_tree = optimal_leaf_ordering(tree, distances)
        local_order = leaves_list(ordered_tree)
        ordered_indices.extend(indices[local_order])
        linkage_table = pd.DataFrame(
            ordered_tree, columns=["left_node", "right_node", "distance", "cluster_size"]
        )
        linkage_table.insert(0, "geographic_region", region)
        linkage_tables.append(linkage_table)

        distance_matrix = squareform(distances)
        admixture_adjacent_distances.extend(
            distance_matrix[np.arange(len(indices) - 1), np.arange(1, len(indices))]
        )
        clustered_adjacent_distances.extend(
            distance_matrix[local_order[:-1], local_order[1:]]
        )

    clustered = ordered.iloc[ordered_indices].copy().reset_index(drop=True)
    positions = []
    offset = 0.0
    for region in REGION_ORDER:
        count = int((clustered["geographic_region"] == region).sum())
        positions.extend(offset + np.arange(count))
        offset += count + REGION_GAP
    clustered["clustered_position"] = positions
    return (
        clustered,
        q_columns,
        pd.concat(linkage_tables, ignore_index=True),
        float(np.mean(admixture_adjacent_distances)),
        float(np.mean(clustered_adjacent_distances)),
    )


def plot_clustered(
    data: pd.DataFrame, q_columns: list[str], figure_dir: Path
) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COLUMN_IN, 3.75))
    positions = data["clustered_position"].to_numpy()
    bottom = np.zeros(len(data))
    for color, component in zip(ANCESTRY_COLORS, q_columns):
        values = data[component].to_numpy()
        ax.bar(positions, values, bottom=bottom, width=1.0, color=color, linewidth=0)
        bottom += values

    for position, subspecies, region in zip(
        positions, data["subspecies"], data["geographic_region"]
    ):
        ax.add_patch(
            Rectangle(
                (position - 0.5, 1.025), 1.0, 0.025,
                color=SUBSPECIES_COLORS[subspecies], clip_on=False, linewidth=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (position - 0.5, 1.060), 1.0, 0.025,
                color=REGION_COLORS[region], clip_on=False, linewidth=0,
            )
        )

    ax.text(
        -0.008, 1.037, "Subspecies", transform=ax.transAxes,
        ha="right", va="center", fontsize=5.5, clip_on=False,
    )
    ax.text(
        -0.008, 1.072, "Region", transform=ax.transAxes,
        ha="right", va="center", fontsize=5.5, clip_on=False,
    )

    ax.set_xlim(-0.5, positions.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("dNMF ancestry proportion")
    ax.set_xlabel("Samples hierarchically clustered by dNMF within geographic region")
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ancestry_handles = [
        Patch(facecolor=color, label=component)
        for component, color in zip(q_columns, ANCESTRY_COLORS)
    ]
    subspecies_handles = [
        Patch(facecolor=SUBSPECIES_COLORS[label], label=label.capitalize())
        for label in SUBSPECIES_ORDER
    ]
    region_handles = [
        Patch(facecolor=REGION_COLORS[region], label=REGION_LABELS[region])
        for region in REGION_ORDER
    ]
    ancestry_legend = fig.legend(
        handles=ancestry_handles, ncol=len(q_columns), frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, 0.125), columnspacing=0.75,
        handlelength=1.05,
    )
    fig.add_artist(ancestry_legend)
    subspecies_legend = fig.legend(
        handles=subspecies_handles, title="Subspecies", ncol=2, frameon=False,
        loc="lower left", bbox_to_anchor=(0.07, 0.015), columnspacing=0.8,
        handlelength=1.05,
    )
    fig.add_artist(subspecies_legend)
    fig.legend(
        handles=region_handles, title="Geographic region", ncol=4, frameon=False,
        loc="lower center", bbox_to_anchor=(0.63, 0.005), columnspacing=0.75,
        handlelength=1.05, handletextpad=0.35,
    )
    sns.despine(ax=ax, bottom=True)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.30)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            figure_dir / f"dnmf_admixture_k9_hierarchical_clustered.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data, q_columns, tree, admixture_distance, clustered_distance = cluster_samples(args)
    data.to_csv(
        source_dir / "dnmf_k9_hierarchical_cluster_order.tsv", sep="\t", index=False
    )
    tree.to_csv(source_dir / "dnmf_k9_hierarchical_linkage.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "ordering": [
                "SNP ADMIXTURE order within geographic region",
                "dNMF hierarchical cluster order within geographic region",
            ],
            "mean_adjacent_jensen_shannon_distance": [
                admixture_distance,
                clustered_distance,
            ],
        }
    ).to_csv(source_dir / "dnmf_k9_ordering_qc.tsv", sep="\t", index=False)
    annotation_colors = pd.DataFrame(
        [
            {"annotation": "subspecies", "label": label, "color": color}
            for label, color in SUBSPECIES_COLORS.items()
        ]
        + [
            {
                "annotation": "geographic_region",
                "label": REGION_LABELS[label],
                "color": REGION_COLORS[label],
            }
            for label in REGION_ORDER
        ]
    )
    annotation_colors.to_csv(
        source_dir / "dnmf_k9_annotation_colors.tsv", sep="\t", index=False
    )
    plot_clustered(data, q_columns, figure_dir)
    print(
        f"Mean adjacent Jensen-Shannon distance: {admixture_distance:.4f} "
        f"(ADMIXTURE order) -> {clustered_distance:.4f} (dNMF-clustered order)"
    )


if __name__ == "__main__":
    main()
