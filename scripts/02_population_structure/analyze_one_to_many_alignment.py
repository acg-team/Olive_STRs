#!/usr/bin/env python3
"""Infer directional one-to-many mappings from SNP to STR ancestry components."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
STRUCTURE_DIR = SCRIPT_DIR.parent
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"
SNP_COLUMNS = [f"ancestry_{index}" for index in range(1, 7)]
SNP_LABELS = [f"SNP C{i}" for i in range(1, 7)]
STR_COLUMNS = [f"Q{i}" for i in range(1, 10)]
MILLER_STONE_COLORS = [
    "#4F6980", "#849DB1", "#A2CEAA", "#638B66", "#BFBB60", "#F47942",
    "#FBB04E", "#B66353", "#D7CE9F", "#B9AA97", "#7E756D",
]
STR_COMPONENT_COLORS = {
    f"STR Q{index}": color
    for index, color in enumerate(MILLER_STONE_COLORS[:9], start=1)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snp-q",
        type=Path,
        default=STRUCTURE_DIR / "admixture/source_data/admixture_k6.csv",
    )
    parser.add_argument(
        "--sample-metadata",
        type=Path,
        default=STRUCTURE_DIR / "admixture/source_data/admixture_k6_plot_data.tsv",
    )
    parser.add_argument(
        "--str-q",
        type=Path,
        default=STRUCTURE_DIR / "dNMF/chr_str_var_gt_0.1/source_data/dnmf_pos9.csv",
    )
    parser.add_argument(
        "--correlations",
        type=Path,
        default=SCRIPT_DIR / "source_data/component_correlations.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--minimum-r", type=float, default=0.30)
    parser.add_argument("--maximum-fdr", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20_260_714)
    return parser.parse_args()


def set_nature_style(font_scale: float = 1.0) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7 * font_scale,
            "axes.titlesize": 8 * font_scale,
            "axes.labelsize": 7 * font_scale,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5 * font_scale,
            "ytick.labelsize": 6.5 * font_scale,
            "legend.fontsize": 6 * font_scale,
            "legend.title_fontsize": 6.5 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def load_str_ancestry(path: Path) -> pd.DataFrame:
    str_q = pd.read_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t")
    if "sample" in str_q.columns and "sample_id" not in str_q.columns:
        str_q = str_q.rename(columns={"sample": "sample_id"})
    numbered = [str(index) for index in range(9)]
    if all(column in str_q.columns for column in numbered):
        str_q = str_q.rename(
            columns={
                str(index): f"Q{index + 1}"
                for index in range(9)
            }
        )
    return str_q


def load_snp_ancestry(path: Path, metadata_path: Path) -> pd.DataFrame:
    snp = pd.read_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t")
    if "sample" in snp.columns and "sample_id" not in snp.columns:
        snp = snp.rename(columns={"sample": "sample_id"})
    numbered = [str(index) for index in range(6)]
    if all(column in snp.columns for column in numbered):
        snp = snp.rename(
            columns={
                str(index): f"ancestry_{index + 1}"
                for index in range(6)
            }
        )
    if "subspecies" not in snp.columns:
        metadata = pd.read_csv(metadata_path, sep="\t")[
            ["sample_id", "subspecies"]
        ].drop_duplicates()
        snp = snp.merge(metadata, on="sample_id", validate="one_to_one")
    return snp


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    snp = load_snp_ancestry(args.snp_q, args.sample_metadata)
    str_q = load_str_ancestry(args.str_q)
    correlations = pd.read_csv(args.correlations, sep="\t")
    data = snp[["sample_id", "subspecies", *SNP_COLUMNS]].merge(
        str_q[["sample_id", *STR_COLUMNS]], on="sample_id", validate="one_to_one"
    )
    if len(data) != 212:
        raise ValueError(f"Expected 212 shared samples, found {len(data)}")
    if not np.allclose(data[SNP_COLUMNS].sum(axis=1), 1, atol=1e-4):
        raise ValueError("SNP ancestry proportions do not sum to one")
    if not np.allclose(data[STR_COLUMNS].sum(axis=1), 1, atol=1e-8):
        raise ValueError("STR ancestry proportions do not sum to one")
    return data, correlations


def ordered_str_components(assignments: pd.DataFrame) -> list[str]:
    """Group assigned STR components by native SNP order."""
    rows = assignments.set_index("str_component")
    unassigned = [
        f"STR {column}"
        for column in STR_COLUMNS
        if rows.loc[f"STR {column}", "assigned_snp_component"] == "Unassigned"
    ]
    ordered = []
    for snp_label in SNP_LABELS:
        assigned = [
            f"STR {column}"
            for column in STR_COLUMNS
            if rows.loc[f"STR {column}", "assigned_snp_component"] == snp_label
        ]
        ordered.extend(assigned)
        if not assigned and unassigned:
            ordered.extend(unassigned)
            unassigned = []
    ordered.extend(unassigned)
    if len(ordered) != len(STR_COLUMNS) or len(set(ordered)) != len(ordered):
        raise ValueError("Could not derive a unique STR display order")
    return ordered


def correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    x_norm = np.sqrt((x_centered**2).sum(axis=0))
    y_norm = np.sqrt((y_centered**2).sum(axis=0))
    return (x_centered.T @ y_centered) / np.outer(x_norm, y_norm)


def infer_assignments(
    correlations: pd.DataFrame, minimum_r: float, maximum_fdr: float
) -> pd.DataFrame:
    all_data = correlations[correlations["cohort"].eq("all_samples")].set_index(
        ["snp_component", "str_component"]
    )
    europaea = correlations[correlations["cohort"].eq("europaea")].set_index(
        ["snp_component", "str_component"]
    )
    rows = []
    for str_column in STR_COLUMNS:
        str_label = f"STR {str_column}"
        values = all_data.xs(str_label, level="str_component").loc[SNP_LABELS]
        best_label = values["pearson_r"].idxmax()
        second_label = values["pearson_r"].drop(best_label).idxmax()
        best = values.loc[best_label]
        second = values.loc[second_label]
        europe = europaea.loc[(best_label, str_label)]
        retained = (
            best["pearson_r"] >= minimum_r
            and best["pearson_fdr"] <= maximum_fdr
            and europe["pearson_r"] >= minimum_r
            and europe["pearson_fdr"] <= maximum_fdr
        )
        rows.append(
            {
                "str_component": str_label,
                "assigned_snp_component": best_label if retained else "Unassigned",
                "strongest_snp_component": best_label,
                "pearson_r_all": best["pearson_r"],
                "pearson_fdr_all": best["pearson_fdr"],
                "pearson_r_europaea": europe["pearson_r"],
                "pearson_fdr_europaea": europe["pearson_fdr"],
                "second_snp_component": second_label,
                "second_pearson_r_all": second["pearson_r"],
                "correlation_margin": best["pearson_r"] - second["pearson_r"],
                "passes_alignment_criteria": retained,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_assignments(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    minimum_r: float,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    x = data[SNP_COLUMNS].to_numpy(dtype=float)
    y = data[STR_COLUMNS].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    assignment_counts = np.zeros((len(STR_COLUMNS), len(SNP_COLUMNS)), dtype=int)
    threshold_counts = np.zeros(len(STR_COLUMNS), dtype=int)
    composite_delta: dict[str, np.ndarray] = {
        label: np.empty(iterations)
        for label in SNP_LABELS
        if sum(assignments["assigned_snp_component"].eq(label)) > 1
    }
    mapping = {
        label: [
            STR_COLUMNS.index(value.replace("STR ", ""))
            for value in assignments.loc[
                assignments["assigned_snp_component"].eq(label), "str_component"
            ]
        ]
        for label in composite_delta
    }
    for iteration in range(iterations):
        indices = rng.integers(0, len(data), len(data))
        matrix = correlation_matrix(x[indices], y[indices])
        strongest = matrix.argmax(axis=0)
        assignment_counts[np.arange(len(STR_COLUMNS)), strongest] += 1
        threshold_counts += matrix.max(axis=0) >= minimum_r
        for label, str_indices in mapping.items():
            snp_index = SNP_LABELS.index(label)
            individual_best = matrix[snp_index, str_indices].max()
            composite = pearsonr(
                x[indices, snp_index], y[indices][:, str_indices].sum(axis=1)
            ).statistic
            composite_delta[label][iteration] = composite - individual_best
    rows = []
    for str_index, row in assignments.iterrows():
        observed_index = SNP_LABELS.index(row["strongest_snp_component"])
        rows.append(
            {
                "str_component": row["str_component"],
                "strongest_snp_component": row["strongest_snp_component"],
                "bootstrap_assignment_stability": assignment_counts[
                    str_index, observed_index
                ]
                / iterations,
                "bootstrap_probability_max_r_ge_threshold": threshold_counts[str_index]
                / iterations,
            }
        )
    return pd.DataFrame(rows), composite_delta


def summarize_snp_mappings(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    composite_delta: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    europaea = data[data["subspecies"].eq("europaea")]
    for snp_label, snp_column in zip(SNP_LABELS, SNP_COLUMNS):
        str_labels = assignments.loc[
            assignments["assigned_snp_component"].eq(snp_label), "str_component"
        ].tolist()
        str_columns = [label.replace("STR ", "") for label in str_labels]
        if str_columns:
            individual = [pearsonr(data[snp_column], data[column]).statistic for column in str_columns]
            individual_europe = [
                pearsonr(europaea[snp_column], europaea[column]).statistic
                for column in str_columns
            ]
            composite = data[str_columns].sum(axis=1)
            composite_europe = europaea[str_columns].sum(axis=1)
            composite_r = pearsonr(data[snp_column], composite).statistic
            composite_r_europe = pearsonr(europaea[snp_column], composite_europe).statistic
            composite_rho = spearmanr(data[snp_column], composite).statistic
            best_r = max(individual)
            delta = composite_r - best_r
        else:
            individual = []
            individual_europe = []
            composite_r = composite_r_europe = composite_rho = best_r = delta = np.nan
        if snp_label in composite_delta:
            distribution = composite_delta[snp_label]
            ci_low, ci_high = np.quantile(distribution, [0.025, 0.975])
            p_value = (int((distribution <= 0).sum()) + 1) / (len(distribution) + 1)
        else:
            ci_low = ci_high = p_value = np.nan
        rows.append(
            {
                "snp_component": snp_label,
                "snp_source_column": snp_column,
                "n_assigned_str_components": len(str_columns),
                "assigned_str_components": ",".join(str_labels) if str_labels else "None",
                "individual_pearson_r_all": ",".join(f"{value:.6f}" for value in individual),
                "individual_pearson_r_europaea": ",".join(
                    f"{value:.6f}" for value in individual_europe
                ),
                "best_individual_pearson_r_all": best_r,
                "summed_str_pearson_r_all": composite_r,
                "summed_str_spearman_rho_all": composite_rho,
                "summed_str_pearson_r_europaea": composite_r_europe,
                "sum_minus_best_delta_r": delta,
                "delta_r_bootstrap_ci_low": ci_low,
                "delta_r_bootstrap_ci_high": ci_high,
                "delta_r_bootstrap_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def plot_alignment(
    assignments: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path
) -> None:
    set_nature_style()
    component_colors = dict(
        zip(SNP_LABELS, MILLER_STONE_COLORS[:6])
    )
    str_order = ordered_str_components(assignments)
    fig, axes = plt.subplots(
        1, 2, figsize=(NATURE_DOUBLE_COLUMN_IN, 3.15), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )

    axis = axes[0]
    snp_y = dict(zip(SNP_LABELS, np.linspace(0.92, 0.08, len(SNP_LABELS))))
    str_y = dict(zip(str_order, np.linspace(0.95, 0.05, len(str_order))))
    for label, y_value in snp_y.items():
        axis.scatter(0.08, y_value, s=95, color=component_colors[label], zorder=3)
        axis.text(0.15, y_value, label.replace("SNP ", ""), va="center", ha="left")
    for label, y_value in str_y.items():
        row = assignments.set_index("str_component").loc[label]
        assigned = row["assigned_snp_component"]
        color = STR_COMPONENT_COLORS[label]
        axis.scatter(0.92, y_value, s=75, color=color, zorder=3)
        axis.text(0.85, y_value, label.replace("STR ", ""), va="center", ha="right")
        if assigned != "Unassigned":
            axis.plot(
                [0.25, 0.75],
                [snp_y[assigned], y_value],
                color=color,
                lw=0.5 + 2.2 * row["pearson_r_all"],
                alpha=0.82,
                solid_capstyle="round",
                zorder=1,
            )
    axis.text(0.08, 1.01, "SNP K=6", ha="center", va="bottom", fontweight="bold")
    axis.text(0.92, 1.01, "STR K=9", ha="center", va="bottom", fontweight="bold")
    axis.set(xlim=(0, 1), ylim=(0, 1), title="Directional component alignment")
    axis.axis("off")
    axis.text(-0.06, 1.05, "A", transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")

    axis = axes[1]
    plot_data = summary.copy()
    x = np.arange(len(plot_data))
    width = 0.35
    best = plot_data["best_individual_pearson_r_all"].fillna(0)
    summed = plot_data["summed_str_pearson_r_all"].fillna(0)
    axis.bar(x - width / 2, best, width, color="#8C8C8C", label="Best single STR component")
    axis.bar(x + width / 2, summed, width, color="#0072B2", label="Sum of assigned STR components")
    for index, row in plot_data.iterrows():
        if row["n_assigned_str_components"] > 1:
            axis.text(
                index + width / 2,
                row["summed_str_pearson_r_all"] + 0.035,
                f"Δr={row['sum_minus_best_delta_r']:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
            )
        elif row["n_assigned_str_components"] == 0:
            axis.text(index, 0.035, "No primary\nassignment", ha="center", va="bottom", fontsize=5.8)
    axis.set(
        title="Reconstruction of SNP ancestry",
        xlabel="SNP component",
        ylabel="Pearson r",
        xticks=x,
        xticklabels=[label.replace("SNP ", "") for label in plot_data["snp_component"]],
        ylim=(0, 1.08),
    )
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=1)
    sns.despine(ax=axis)
    axis.text(-0.13, 1.05, "B", transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.32, top=0.76, bottom=0.16, left=0.04, right=0.98)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"one_to_many_component_alignment.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_alignment_panel_a(
    assignments: pd.DataFrame, figure_dir: Path
) -> None:
    """Render the directional alignment as a standalone, enlarged panel."""
    set_nature_style(font_scale=1.4)
    component_colors = dict(
        zip(SNP_LABELS, MILLER_STONE_COLORS[:6])
    )
    str_order = ordered_str_components(assignments)
    figure, axis = plt.subplots(figsize=(4.6, 3.7))
    snp_y = dict(zip(SNP_LABELS, np.linspace(0.92, 0.08, len(SNP_LABELS))))
    str_y = dict(zip(str_order, np.linspace(0.95, 0.05, len(str_order))))
    assignment_lookup = assignments.set_index("str_component")

    for label, y_value in snp_y.items():
        axis.scatter(
            0.07, y_value, s=135, color=component_colors[label], zorder=3
        )
        axis.text(
            0.16, y_value, label.replace("SNP ", ""),
            va="center", ha="left",
        )
    for label, y_value in str_y.items():
        row = assignment_lookup.loc[label]
        assigned = row["assigned_snp_component"]
        color = STR_COMPONENT_COLORS[label]
        axis.scatter(0.93, y_value, s=105, color=color, zorder=3)
        axis.text(
            0.84, y_value, label.replace("STR ", ""),
            va="center", ha="right",
        )
        if assigned != "Unassigned":
            axis.plot(
                [0.29, 0.71],
                [snp_y[assigned], y_value],
                color=color,
                lw=0.7 + 2.5 * row["pearson_r_all"],
                alpha=0.82,
                solid_capstyle="round",
                zorder=1,
            )

    axis.text(
        0.07, 1.01, "SNP K=6",
        ha="center", va="bottom", fontweight="bold",
    )
    axis.text(
        0.93, 1.01, "STR K=9",
        ha="center", va="bottom", fontweight="bold",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        title="Directional component alignment",
    )
    axis.axis("off")
    figure.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.05)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            figure_dir / f"one_to_many_component_alignment_panel_a.{suffix}",
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    data, correlations = load_data(args)
    assignments = infer_assignments(correlations, args.minimum_r, args.maximum_fdr)
    stability, composite_delta = bootstrap_assignments(
        data, assignments, args.minimum_r, args.bootstrap, args.seed
    )
    assignments = assignments.merge(stability, on=["str_component", "strongest_snp_component"])
    summary = summarize_snp_mappings(data, assignments, composite_delta)
    assignments.to_csv(source_dir / "str_component_directional_assignments.tsv", sep="\t", index=False)
    summary.to_csv(source_dir / "one_to_many_component_alignment.tsv", sep="\t", index=False)
    plot_alignment(assignments, summary, figure_dir)
    plot_alignment_panel_a(assignments, figure_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
