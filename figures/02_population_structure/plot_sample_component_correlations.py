#!/usr/bin/env python3
"""Plot raw sample ancestry proportions for matched SNP and STR components."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/02_population_structure"
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"
SUBSPECIES_COLORS = {"europaea": "#4D4D4D", "sylvestris": "#56B4E9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", type=Path,
        default=ANALYSIS_DIR / "source_data/shared_component_profiles.tsv",
    )
    parser.add_argument(
        "--matches", type=Path,
        default=ANALYSIS_DIR / "source_data/hungarian_component_matches.tsv",
    )
    parser.add_argument(
        "--correlations", type=Path,
        default=ANALYSIS_DIR / "source_data/component_correlations.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--expected-samples", type=int, default=212)
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
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def build_long_table(args: argparse.Namespace) -> pd.DataFrame:
    profiles = pd.read_csv(args.profiles, sep="\t")
    if len(profiles) != args.expected_samples or profiles["sample_id"].duplicated().any():
        raise ValueError(f"Expected {args.expected_samples} unique sample profiles")
    matches = pd.read_csv(args.matches, sep="\t").query("cohort == 'all_samples'").copy()
    correlations = pd.read_csv(args.correlations, sep="\t").query("cohort == 'all_samples'")
    source_lookup = (
        correlations[["snp_component", "snp_source_column"]]
        .drop_duplicates()
        .set_index("snp_component")["snp_source_column"]
    )
    matches["snp_source_column"] = matches["snp_component"].map(source_lookup)
    matches["str_source_column"] = matches["str_component"].str.replace("STR ", "", regex=False)
    matches["snp_number"] = matches["snp_component"].str.extract(r"(\d+)").astype(int)
    matches = matches.sort_values("snp_number")
    if len(matches) != 6:
        raise ValueError(f"Expected six optimized component pairs, found {len(matches)}")

    rows = []
    metadata = profiles[["sample_id", "subspecies", "Country"]]
    for match in matches.itertuples(index=False):
        pair = metadata.copy()
        pair["snp_component"] = match.snp_component
        pair["str_component"] = match.str_component
        pair["component_pair"] = (
            match.snp_component.replace("SNP ", "") + "–" + match.str_component.replace("STR ", "")
        )
        pair["snp_source_column"] = match.snp_source_column
        pair["str_source_column"] = match.str_source_column
        pair["snp_ancestry_proportion"] = profiles[match.snp_source_column].to_numpy()
        pair["str_ancestry_proportion"] = profiles[match.str_source_column].to_numpy()
        rows.append(pair)
    return pd.concat(rows, ignore_index=True)


def plot_raw_samples(data: pd.DataFrame, figure_dir: Path) -> None:
    set_nature_style()
    pairs = data["component_pair"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(NATURE_DOUBLE_COLUMN_IN, 4.6), sharex=True, sharey=True)
    for axis, pair_name in zip(axes.flat, pairs):
        subset = data[data["component_pair"] == pair_name]
        for subspecies in ("europaea", "sylvestris"):
            group = subset[subset["subspecies"] == subspecies]
            axis.scatter(
                group["snp_ancestry_proportion"], group["str_ancestry_proportion"],
                s=13 if subspecies == "europaea" else 22,
                color=SUBSPECIES_COLORS[subspecies], alpha=0.62 if subspecies == "europaea" else 0.9,
                edgecolor="white", linewidth=0.25, zorder=2,
            )
        sns.regplot(
            data=subset, x="snp_ancestry_proportion", y="str_ancestry_proportion",
            scatter=False, ci=None, ax=axis, line_kws={"color": "#222222", "lw": 0.8},
        )
        all_r = pearsonr(
            subset["snp_ancestry_proportion"], subset["str_ancestry_proportion"]
        ).statistic
        europaea = subset[subset["subspecies"] == "europaea"]
        europaea_r = pearsonr(
            europaea["snp_ancestry_proportion"], europaea["str_ancestry_proportion"]
        ).statistic
        axis.text(
            0.04, 0.96, f"All $r$ = {all_r:.2f}\neuropaea $r$ = {europaea_r:.2f}",
            transform=axis.transAxes, ha="left", va="top", fontsize=6.2,
        )
        axis.set_title(pair_name)
        axis.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
        axis.set_aspect("equal", adjustable="box")
        sns.despine(ax=axis)

    for axis in axes[-1, :]:
        axis.set_xlabel("SNP ancestry proportion")
    for axis in axes[0, :]:
        axis.set_xlabel("")
    for axis in axes[:, 0]:
        axis.set_ylabel("STR ancestry proportion")
    for axis in axes[0, 1:]:
        axis.set_ylabel("")
    for axis in axes[1, 1:]:
        axis.set_ylabel("")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color=SUBSPECIES_COLORS[label],
               label=label, markersize=4.5)
        for label in ("europaea", "sylvestris")
    ]
    fig.legend(handles=legend, title="Subspecies", frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.1, h_pad=1.0)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            figure_dir / f"snp_str_sample_component_correlations.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = build_long_table(args)
    data.to_csv(source_dir / "sample_component_proportions.tsv", sep="\t", index=False)
    plot_raw_samples(data, figure_dir)
    print(f"Plotted {data['sample_id'].nunique()} samples across {data['component_pair'].nunique()} raw component pairs")


if __name__ == "__main__":
    main()
