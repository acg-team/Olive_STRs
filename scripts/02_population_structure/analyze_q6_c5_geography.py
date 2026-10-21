#!/usr/bin/env python3
"""Test geographic support for SNP C5 and the unassigned STR Q6 component."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
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
        "--str-q", type=Path,
        default=STRUCTURE_DIR / "dNMF/chr_str_var_gt_0.1/source_data/dnmf_pos9.csv",
    )
    parser.add_argument(
        "--metadata", type=Path, default=STRUCTURE_DIR.parent / "metadata/Geo_loc.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--minimum-group-size", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20_260_714)
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    snp = pd.read_csv(
        args.snp_q,
        sep="," if args.snp_q.suffix.lower() == ".csv" else "\t",
    )
    if "sample" in snp.columns and "sample_id" not in snp.columns:
        snp = snp.rename(columns={"sample": "sample_id"})
    if "4" in snp.columns and "ancestry_5" not in snp.columns:
        snp = snp.rename(columns={"4": "ancestry_5"})
    snp = snp[["sample_id", "ancestry_5"]]
    str_q = pd.read_csv(
        args.str_q,
        sep="," if args.str_q.suffix.lower() == ".csv" else "\t",
    )
    if "sample" in str_q.columns and "sample_id" not in str_q.columns:
        str_q = str_q.rename(columns={"sample": "sample_id"})
    if "5" in str_q.columns and "Q6" not in str_q.columns:
        str_q = str_q.rename(columns={"5": "Q6"})
    str_q = str_q[["sample_id", "Q6"]]
    metadata = pd.read_csv(args.metadata, encoding="latin1").rename(
        columns={"ID": "sample_id", "Provience": "Province"}
    )
    data = snp.merge(str_q, on="sample_id", validate="one_to_one").merge(
        metadata[["sample_id", "Country", "Province", "Subspecies"]],
        on="sample_id", validate="one_to_one",
    )
    if len(data) != 212:
        raise ValueError(f"Expected 212 samples, found {len(data)}")
    return data.rename(columns={"ancestry_5": "SNP C5", "Q6": "STR Q6"})


def filter_groups(data: pd.DataFrame, label: str, minimum: int) -> pd.DataFrame:
    counts = data[label].value_counts()
    return data[data[label].isin(counts[counts >= minimum].index)].copy()


def anova_statistic(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    groups = np.unique(labels)
    total_ss = float(((values - values.mean()) ** 2).sum())
    within_ss = sum(
        float(((values[labels == group] - values[labels == group].mean()) ** 2).sum())
        for group in groups
    )
    among_ss = total_ss - within_ss
    pseudo_f = (among_ss / (len(groups) - 1)) / (within_ss / (len(values) - len(groups)))
    return pseudo_f, among_ss / total_ss


def test_component(
    data: pd.DataFrame,
    label: str,
    component: str,
    scale: str,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    values = data[component].to_numpy(dtype=float)
    labels = data[label].to_numpy()
    groups = np.unique(labels)
    observed_f, r_squared = anova_statistic(values, labels)
    observed_h = kruskal(*(values[labels == group] for group in groups)).statistic
    rng = np.random.default_rng(seed)
    f_exceedances = 0
    h_exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        permuted_f, _ = anova_statistic(values, permuted)
        permuted_h = kruskal(*(values[permuted == group] for group in groups)).statistic
        f_exceedances += permuted_f >= observed_f
        h_exceedances += permuted_h >= observed_h
    return {
        "geographic_scale": scale,
        "component": component,
        "n_samples": len(data),
        "n_groups": len(groups),
        "anova_f": observed_f,
        "anova_r_squared": r_squared,
        "anova_permutation_p": (f_exceedances + 1) / (permutations + 1),
        "kruskal_h": observed_h,
        "kruskal_permutation_p": (h_exceedances + 1) / (permutations + 1),
    }


def summarize(data: pd.DataFrame, label: str, scale: str) -> pd.DataFrame:
    long = data.melt(
        id_vars=["sample_id", label],
        value_vars=["SNP C5", "STR Q6"],
        var_name="component",
        value_name="ancestry_proportion",
    )
    return (
        long.groupby(["component", label], observed=True)["ancestry_proportion"]
        .agg(n_samples="size", mean="mean", median="median", maximum="max")
        .reset_index()
        .rename(columns={label: "geographic_group"})
        .assign(geographic_scale=scale)
    )


def plot_results(
    country: pd.DataFrame,
    province: pd.DataFrame,
    tests: pd.DataFrame,
    figure_dir: Path,
) -> None:
    set_style()
    country_order = ["Iran", "Greece", "Spain", "Syria"]
    province_order = ["Golestan", "Tarom", "Ilam", "Kermanshah", "Fars", "Kohgiluyeh", "Khuzestan"]
    country_palette = dict(zip(country_order, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    province_palette = dict(zip(province_order, sns.color_palette("colorblind", len(province_order))))
    fig, axes = plt.subplots(2, 2, figsize=(NATURE_DOUBLE_COLUMN_IN, 5.1))
    panels = [
        (axes[0, 0], country, "Country", country_order, country_palette, "SNP C5", "Country"),
        (axes[0, 1], country, "Country", country_order, country_palette, "STR Q6", "Country"),
        (axes[1, 0], province, "Province", province_order, province_palette, "SNP C5", "Iran province"),
        (axes[1, 1], province, "Province", province_order, province_palette, "STR Q6", "Iran province"),
    ]
    for panel_letter, (axis, frame, label, order, palette, component, scale) in zip("ABCD", panels):
        sns.boxplot(
            data=frame, x=label, y=component, order=order, color="#F2F2F2",
            width=0.58, fliersize=0, linewidth=0.7, ax=axis,
        )
        sns.stripplot(
            data=frame, x=label, y=component, order=order, hue=label, palette=palette,
            jitter=0.22, size=2.8, alpha=0.75, edgecolor="white", linewidth=0.2,
            legend=False, ax=axis,
        )
        result = tests[
            tests["geographic_scale"].eq(scale) & tests["component"].eq(component)
        ].iloc[0]
        axis.text(
            0.02, 0.96,
            f"R²={result['anova_r_squared']:.3f}; permutation P={result['anova_permutation_p']:.4f}",
            transform=axis.transAxes, ha="left", va="top", fontsize=6.2,
        )
        axis.set(title=f"{component}: {scale.lower()}", xlabel="", ylabel="Ancestry proportion")
        axis.set_ylim(bottom=-0.02, top=max(0.12, frame[component].max() * 1.13))
        axis.tick_params(axis="x", rotation=32 if scale == "Iran province" else 0)
        sns.despine(ax=axis)
        axis.text(-0.12, 1.04, panel_letter, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.subplots_adjust(hspace=0.42, wspace=0.27, top=0.94, bottom=0.13)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"q6_c5_geographic_support.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    country = filter_groups(
        data[data["Subspecies"].eq("Olea europaea subsp. europaea")],
        "Country", args.minimum_group_size,
    )
    province = filter_groups(data[data["Country"].eq("Iran")], "Province", args.minimum_group_size)
    rows = []
    for scale_index, (frame, label, scale) in enumerate(
        ((country, "Country", "Country"), (province, "Province", "Iran province"))
    ):
        for component_index, component in enumerate(("SNP C5", "STR Q6")):
            rows.append(
                test_component(
                    frame, label, component, scale, args.permutations,
                    args.seed + 10 * scale_index + component_index,
                )
            )
    tests = pd.DataFrame(rows)
    tests.to_csv(source_dir / "q6_c5_geographic_tests.tsv", sep="\t", index=False)
    summaries = pd.concat(
        [summarize(country, "Country", "Country"), summarize(province, "Province", "Iran province")],
        ignore_index=True,
    )
    summaries.to_csv(source_dir / "q6_c5_geographic_group_summary.tsv", sep="\t", index=False)
    plot_data = pd.concat(
        [
            country.assign(geographic_scale="Country", geographic_group=country["Country"]),
            province.assign(geographic_scale="Iran province", geographic_group=province["Province"]),
        ], ignore_index=True,
    )
    plot_data[["sample_id", "geographic_scale", "geographic_group", "SNP C5", "STR Q6"]].to_csv(
        source_dir / "q6_c5_geographic_plot_data.tsv", sep="\t", index=False
    )
    plot_results(country, province, tests, figure_dir)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
