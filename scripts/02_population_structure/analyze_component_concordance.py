#!/usr/bin/env python3
"""Measure component-level concordance between SNP ADMIXTURE and STR dNMF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
STRUCTURE_DIR = SCRIPT_DIR.parent
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snp-q", type=Path,
        default=STRUCTURE_DIR / "admixture/source_data/admixture_k6.csv",
    )
    parser.add_argument(
        "--sample-metadata", type=Path,
        default=STRUCTURE_DIR / "admixture/source_data/admixture_k6_plot_data.tsv",
    )
    parser.add_argument(
        "--str-q", type=Path,
        default=STRUCTURE_DIR / "dNMF/chr_str_var_gt_0.1/source_data/dnmf_pos9.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20_260_713)
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
            "legend.title_fontsize": 6.5,
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
    required_metadata = {"subspecies", "Country"}
    if not required_metadata.issubset(snp.columns):
        metadata = pd.read_csv(metadata_path, sep="\t")[
            ["sample_id", "subspecies", "Country"]
        ].drop_duplicates()
        snp = snp.merge(metadata, on="sample_id", validate="one_to_one")
    return snp


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
    snp = load_snp_ancestry(args.snp_q, args.sample_metadata)
    str_q = load_str_ancestry(args.str_q)
    snp_columns = [column for column in snp.columns if column.startswith("ancestry_")]
    str_columns = [column for column in str_q.columns if column.startswith("Q")]
    if len(snp_columns) != 6 or len(str_columns) != 9:
        raise ValueError(f"Expected SNP K=6 and STR K=9, observed {len(snp_columns)} and {len(str_columns)}")
    if snp["sample_id"].duplicated().any() or str_q["sample_id"].duplicated().any():
        raise ValueError("Ancestry inputs contain duplicate sample IDs")
    required = {"sample_id", "subspecies", "Country"}
    if not required.issubset(snp.columns):
        raise ValueError(f"SNP input lacks columns: {sorted(required - set(snp.columns))}")
    data = snp[["sample_id", "subspecies", "Country", *snp_columns]].merge(
        str_q[["sample_id", *str_columns]], on="sample_id", how="inner", validate="one_to_one"
    )
    if len(data) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} shared samples, found {len(data)}")
    if not np.allclose(data[snp_columns].sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("SNP ancestry proportions do not sum to one")
    if not np.allclose(data[str_columns].sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("STR ancestry proportions do not sum to one")
    return data, snp_columns, str_columns


def adjust_bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def correlation_table(
    data: pd.DataFrame,
    snp_columns: list[str],
    str_columns: list[str],
    cohort: str,
) -> pd.DataFrame:
    rows = []
    for snp_index, snp_column in enumerate(snp_columns, start=1):
        for str_column in str_columns:
            pearson = pearsonr(data[snp_column], data[str_column])
            spearman = spearmanr(data[snp_column], data[str_column])
            rows.append(
                {
                    "cohort": cohort,
                    "n_samples": len(data),
                    "snp_component": f"SNP C{snp_index}",
                    "snp_source_column": snp_column,
                    "str_component": f"STR {str_column}",
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_rho": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result["pearson_fdr"] = adjust_bh(result["pearson_p"].to_numpy())
    result["spearman_fdr"] = adjust_bh(result["spearman_p"].to_numpy())
    return result


def correlation_matrix(
    table: pd.DataFrame, snp_labels: list[str], str_labels: list[str]
) -> np.ndarray:
    return (
        table.pivot(index="snp_component", columns="str_component", values="pearson_r")
        .loc[snp_labels, str_labels]
        .to_numpy()
    )


def matched_components(
    matrix: np.ndarray,
    table: pd.DataFrame,
    snp_labels: list[str],
    str_labels: list[str],
    cohort: str,
) -> pd.DataFrame:
    row_indices, column_indices = linear_sum_assignment(-matrix)
    lookup = table.set_index(["snp_component", "str_component"])
    rows = []
    for row_index, column_index in zip(row_indices, column_indices):
        snp_label = snp_labels[row_index]
        str_label = str_labels[column_index]
        values = lookup.loc[(snp_label, str_label)]
        rows.append(
            {
                "cohort": cohort,
                "snp_component": snp_label,
                "str_component": str_label,
                "pearson_r": values["pearson_r"],
                "pearson_fdr": values["pearson_fdr"],
                "spearman_rho": values["spearman_rho"],
                "spearman_fdr": values["spearman_fdr"],
            }
        )
    return pd.DataFrame(rows)


def permutation_test(
    data: pd.DataFrame,
    snp_columns: list[str],
    str_columns: list[str],
    permutations: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    snp = data[snp_columns].to_numpy(dtype=float)
    str_q = data[str_columns].to_numpy(dtype=float)
    snp -= snp.mean(axis=0)
    str_q -= str_q.mean(axis=0)
    snp /= np.sqrt((snp ** 2).sum(axis=0))
    str_q /= np.sqrt((str_q ** 2).sum(axis=0))
    observed_matrix = snp.T @ str_q
    rows, columns = linear_sum_assignment(-observed_matrix)
    observed = float(observed_matrix[rows, columns].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        permuted = snp.T @ str_q[rng.permutation(len(str_q))]
        null_rows, null_columns = linear_sum_assignment(-permuted)
        null[index] = permuted[null_rows, null_columns].mean()
    p_value = (int((null >= observed).sum()) + 1) / (permutations + 1)
    return observed, p_value, null


def plot_results(
    all_table: pd.DataFrame,
    europaea_table: pd.DataFrame,
    all_matches: pd.DataFrame,
    snp_labels: list[str],
    str_labels: list[str],
    figure_dir: Path,
) -> None:
    set_nature_style()
    all_matrix = correlation_matrix(all_table, snp_labels, str_labels)
    europaea_matrix = correlation_matrix(europaea_table, snp_labels, str_labels)
    matched_pairs = list(zip(all_matches["snp_component"], all_matches["str_component"]))

    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE_COLUMN_IN, 2.55), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for axis, matrix, title, show_colorbar in (
        (axes[0], all_matrix, "All samples", False),
        (axes[1], europaea_matrix, "europaea only", True),
    ):
        sns.heatmap(
            matrix, ax=axis, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"fontsize": 5.2},
            xticklabels=[label.replace("STR ", "") for label in str_labels],
            yticklabels=[label.replace("SNP ", "") for label in snp_labels],
            cbar=show_colorbar,
            cbar_kws={"shrink": 0.72, "pad": 0.03},
            linewidths=0.25, linecolor="white",
        )
        axis.set(title=title, xlabel="STR dNMF component", ylabel="SNP ADMIXTURE component")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)
        for snp_label, str_label in matched_pairs:
            row = snp_labels.index(snp_label)
            column = str_labels.index(str_label)
            axis.add_patch(Rectangle((column, row), 1, 1, fill=False, edgecolor="black", lw=1.0))

    europe_lookup = europaea_table.set_index(["snp_component", "str_component"])["pearson_r"]
    plot_data = all_matches.copy()
    plot_data["europaea_r"] = [
        europe_lookup.loc[(row.snp_component, row.str_component)]
        for row in plot_data.itertuples(index=False)
    ]
    x = np.arange(len(plot_data))
    width = 0.36
    axes[2].bar(x - width / 2, plot_data["pearson_r"], width, color="#4D4D4D", label="All samples")
    axes[2].bar(x + width / 2, plot_data["europaea_r"], width, color="#D55E00", label="europaea only")
    axes[2].axhline(0, color="#222222", lw=0.6)
    axes[2].set(
        xlabel="Optimized component pair", ylabel="Pearson r",
        xticks=x,
        xticklabels=[
            f"{row.snp_component.replace('SNP ', '')}–{row.str_component.replace('STR ', '')}"
            for row in plot_data.itertuples(index=False)
        ],
        ylim=(0, 1.05),
    )
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1
    )
    sns.despine(ax=axes[2])

    for label, axis in zip("abc", axes):
        axis.text(-0.18, 1.06, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"snp_str_component_concordance.{extension}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    matches: pd.DataFrame,
    permutation_results: pd.DataFrame,
    table: pd.DataFrame,
) -> None:
    all_matches = matches[matches["cohort"] == "all_samples"]
    europe_matches = matches[matches["cohort"] == "europaea"]
    pair_lines = "\n".join(
        f"- {row.snp_component}–{row.str_component}: r = {row.pearson_r:.3f} "
        f"(europaea-only r = {europe_matches.set_index(['snp_component', 'str_component'])['pearson_r'].get((row.snp_component, row.str_component), float('nan')):.3f})"
        for row in all_matches.itertuples(index=False)
    )
    all_perm = permutation_results.set_index("cohort").loc["all_samples"]
    europe_perm = permutation_results.set_index("cohort").loc["europaea"]
    strongest = table[table["cohort"] == "all_samples"].sort_values("pearson_r", ascending=False).iloc[0]
    matched_pair_set = set(zip(all_matches["snp_component"], all_matches["str_component"]))
    secondary = table[
        (table["cohort"] == "all_samples") & (table["pearson_r"] >= 0.5)
    ].copy()
    secondary = secondary[
        ~secondary.apply(
            lambda row: (row["snp_component"], row["str_component"]) in matched_pair_set,
            axis=1,
        )
    ].sort_values("pearson_r", ascending=False)
    secondary_lines = "\n".join(
        f"- {row.snp_component}–{row.str_component}: r = {row.pearson_r:.3f}"
        for row in secondary.itertuples(index=False)
    )
    text = f"""# SNP–STR ancestry-component concordance

## Design

The six SNP ADMIXTURE K=6 components from `admixture_k6.csv` were used in their native column order (`0`–`5` mapped to C1–C6) and compared with the nine chromosome-STR dNMF K=9 components from `dnmf_pos9.csv` across the same 212 accessions. Pearson and Spearman correlations were calculated for all 54 component pairs with Benjamini-Hochberg correction. Positive one-to-one matches were selected with the Hungarian algorithm. The analysis was repeated within the 201 europaea accessions to identify relationships that persist after excluding the 11 sylvestris samples.

Because ancestry coefficients sum to one within each marker system, negative correlations can arise from compositional closure and are not interpreted as direct component matches. The complete matrices are retained so one-to-many component splitting is visible.

## Results

The optimized mean Pearson correlation was {all_perm['observed_mean_matched_r']:.3f} across all samples (permutation P = {all_perm['permutation_p']:.4g}) and {europe_perm['observed_mean_matched_r']:.3f} within europaea (P = {europe_perm['permutation_p']:.4g}). The strongest individual relationship was {strongest['snp_component']}–{strongest['str_component']} (r = {strongest['pearson_r']:.3f}). Optimized all-sample pairs were:

{pair_lines}

Secondary positive relationships (r >= 0.5) outside the one-to-one assignment were:

{secondary_lines}

Black outlines in the heatmaps identify the all-sample optimized pairs. A strong value outside an outline indicates a possible one-to-many relationship that cannot be represented by forced one-to-one matching.

## Outputs

- `figures/snp_str_component_concordance.*`: all-sample and europaea-only heatmaps plus matched-pair comparison
- `source_data/component_correlations.tsv`: all 108 cohort-specific component-pair tests
- `source_data/hungarian_component_matches.tsv`: optimized matches for both cohorts
- `source_data/matching_permutation_tests.tsv`: matching-aware permutation tests
- `source_data/shared_component_profiles.tsv`: aligned SNP and STR ancestry coefficients
"""
    (args.output_dir / "analysis_report.md").write_text(text)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    for directory in (args.output_dir, source_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    data, snp_columns, str_columns = load_data(args)
    data.to_csv(source_dir / "shared_component_profiles.tsv", sep="\t", index=False)
    snp_labels = [f"SNP C{index}" for index in range(1, len(snp_columns) + 1)]
    str_labels = [f"STR {column}" for column in str_columns]

    tables = []
    matches = []
    permutation_rows = []
    cohorts = {
        "all_samples": data,
        "europaea": data[data["subspecies"] == "europaea"].copy(),
    }
    for cohort_index, (cohort, subset) in enumerate(cohorts.items()):
        table = correlation_table(subset, snp_columns, str_columns, cohort)
        matrix = correlation_matrix(table, snp_labels, str_labels)
        tables.append(table)
        matches.append(matched_components(matrix, table, snp_labels, str_labels, cohort))
        observed, p_value, null = permutation_test(
            subset, snp_columns, str_columns, args.permutations, args.seed + cohort_index
        )
        permutation_rows.append(
            {
                "cohort": cohort,
                "n_samples": len(subset),
                "permutations": args.permutations,
                "observed_mean_matched_r": observed,
                "null_mean": null.mean(),
                "null_95th_percentile": np.quantile(null, 0.95),
                "permutation_p": p_value,
            }
        )
    table = pd.concat(tables, ignore_index=True)
    match_table = pd.concat(matches, ignore_index=True)
    permutation_table = pd.DataFrame(permutation_rows)
    table.to_csv(source_dir / "component_correlations.tsv", sep="\t", index=False)
    match_table.to_csv(source_dir / "hungarian_component_matches.tsv", sep="\t", index=False)
    permutation_table.to_csv(source_dir / "matching_permutation_tests.tsv", sep="\t", index=False)
    plot_results(tables[0], tables[1], matches[0], snp_labels, str_labels, figure_dir)
    write_report(args, match_table, permutation_table, table)
    print(permutation_table.to_string(index=False))
    print(match_table[match_table["cohort"] == "all_samples"].to_string(index=False))


if __name__ == "__main__":
    main()
