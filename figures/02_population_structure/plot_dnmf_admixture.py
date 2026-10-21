#!/usr/bin/env python3
"""Plot K=9 dNMF ancestry in the exact SNP ADMIXTURE sample order."""

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
DEFAULT_ANALYSIS_DIR = ANALYSIS_DIR / "chr_str_var_gt_0.1"
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"
ANCESTRY_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E6AB02",
    "#6A3D9A", "#56B4E9", "#A6761D", "#666666",
]
SUBSPECIES_COLORS = {"europaea": "#4D4D4D", "sylvestris": "#56B4E9"}
COUNTRY_CODES = {
    "Albania": "ALB", "Algeria": "DZA", "Croatia": "HRV", "Greece": "GRC",
    "Iran": "IRN", "Israel": "ISR", "Italy": "ITA", "Lebanon": "LBN",
    "Morocco": "MAR", "Spain": "ESP", "Syria": "SYR", "Tunisia": "TUN",
    "Turkey": "TUR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-matrix", type=Path,
        default=DEFAULT_ANALYSIS_DIR / "source_data/best_k9_pos.tsv",
    )
    parser.add_argument(
        "--snp-order", type=Path,
        default=ANALYSIS_DIR / "source_data/admixture_k6_plot_data.tsv",
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


def load_plot_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    order = pd.read_csv(args.snp_order, sep="\t")
    required = {"plot_position", "sample_id", "Country", "Subspecies", "subspecies"}
    if not required.issubset(order.columns):
        raise ValueError(f"SNP order table lacks columns: {sorted(required - set(order.columns))}")
    if len(order) != args.expected_samples or order["sample_id"].duplicated().any():
        raise ValueError(f"Expected {args.expected_samples} unique samples in SNP plot order")
    order = order.sort_values("plot_position", kind="stable").reset_index(drop=True)
    if not np.array_equal(order["plot_position"], np.arange(args.expected_samples)):
        raise ValueError("SNP plot positions are not contiguous from zero")

    q = pd.read_csv(args.q_matrix, sep="\t")
    q_columns = [column for column in q.columns if column.startswith("Q")]
    if len(q_columns) != args.expected_k or q["sample_id"].duplicated().any():
        raise ValueError(f"Expected a unique-sample K={args.expected_k} Q matrix")
    if set(order["sample_id"]) != set(q["sample_id"]):
        raise ValueError("SNP order and dNMF Q matrix contain different sample sets")
    data = order[list(required)].merge(q, on="sample_id", how="left", validate="one_to_one")
    data = data.sort_values("plot_position", kind="stable").reset_index(drop=True)
    if not np.allclose(data[q_columns].sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("dNMF ancestry coefficients do not sum to one")
    return data, q_columns


def contiguous_blocks(data: pd.DataFrame, column: str) -> list[tuple[str, int, int]]:
    values = data[column].tolist()
    blocks = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            blocks.append((values[start], start, index))
            start = index
    return blocks


def plot(data: pd.DataFrame, q_columns: list[str], figure_dir: Path) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COLUMN_IN, 3.35))
    positions = data["plot_position"].to_numpy()
    bottom = np.zeros(len(data))
    for color, component in zip(ANCESTRY_COLORS, q_columns):
        values = data[component].to_numpy()
        ax.bar(positions, values, bottom=bottom, width=1.0, color=color, linewidth=0)
        bottom += values

    for country, start, end in contiguous_blocks(data, "Country"):
        if start > 0:
            ax.axvline(start - 0.5, color="white", lw=0.6, zorder=5)
        ax.text(
            (start + end - 1) / 2, -0.025, COUNTRY_CODES.get(country, country),
            ha="right", va="center", rotation=90, rotation_mode="anchor", fontsize=5.5,
        )

    for subspecies, start, end in contiguous_blocks(data, "subspecies"):
        ax.add_patch(
            Rectangle(
                (start - 0.5, 1.025), end - start, 0.025,
                color=SUBSPECIES_COLORS[subspecies], clip_on=False, linewidth=0,
            )
        )
        ax.text(
            (start + end - 1) / 2, 1.075, subspecies, ha="center", va="bottom",
            fontstyle="italic", fontsize=7,
        )
        if start > 0:
            ax.axvline(start - 0.5, color="#222222", lw=0.8, zorder=6)

    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("dNMF ancestry proportion")
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(
        handles=[Patch(facecolor=color, label=component)
                 for component, color in zip(q_columns, ANCESTRY_COLORS)],
        ncol=len(q_columns), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23),
        columnspacing=0.8, handlelength=1.1,
    )
    sns.despine(ax=ax, bottom=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.83, bottom=0.31)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"dnmf_admixture_k9_snp_order.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data, q_columns = load_plot_data(args)
    output_columns = [
        "plot_position", "sample_id", "Country", "Subspecies", "subspecies", *q_columns
    ]
    data[output_columns].to_csv(
        source_dir / "dnmf_k9_snp_plot_order.tsv", sep="\t", index=False
    )
    plot(data, q_columns, figure_dir)
    print(f"Plotted {len(data)} samples at K={len(q_columns)} in SNP ADMIXTURE order")


if __name__ == "__main__":
    main()
