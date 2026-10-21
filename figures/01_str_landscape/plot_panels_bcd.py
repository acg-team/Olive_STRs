#!/usr/bin/env python3
"""Create Panels B-D and the combined polymorphic STR landscape figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/01_str_landscape"
INPUT_DIR = ANALYSIS_DIR / "inputs"
MOTIF_ORDER = [1, 2, 3, 4, 5, 6]
REGION_ORDER = [
    "CDS",
    "five_prime_UTR",
    "three_prime_UTR",
    "other_exon",
    "intron",
    "promoter",
    "intergenic",
]
REGION_LABELS = {
    "CDS": "CDS",
    "five_prime_UTR": "5' UTR",
    "three_prime_UTR": "3' UTR",
    "other_exon": "Other exon",
    "intron": "Intron",
    "promoter": "Promoter",
    "intergenic": "Intergenic",
}

NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
NATURE_SINGLE_COLUMN_IN = 89 / 25.4
FONT_FAMILY = "Nimbus Sans"
MOTIF_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
REGION_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#7A7A7A"]


def set_nature_style(font_scale: float = 1.0) -> None:
    """Configure artwork for Nature's 183-mm, double-column format."""
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6 * font_scale,
            "axes.titlesize": 7 * font_scale,
            "axes.titleweight": "normal",
            "axes.labelsize": 6 * font_scale,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 5.5 * font_scale,
            "ytick.labelsize": 5.5 * font_scale,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 5.5 * font_scale,
            "legend.title_fontsize": 6 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=plt.rcParams["font.size"] * 4 / 3,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=INPUT_DIR / "O_europeana_strs.functional.promoter2kb.annotated.bed",
    )
    parser.add_argument(
        "--variance",
        type=Path,
        default=INPUT_DIR / "str_length_variance.csv",
    )
    parser.add_argument(
        "--panel-a-source",
        type=Path,
        default=SCRIPT_DIR / "output/source_data/panel_a_1mb_windows.tsv",
    )
    parser.add_argument("--min-evaluable", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Regenerate only Panel C and variance-by-genomic-context outputs.",
    )
    parser.add_argument(
        "--motif-only",
        action="store_true",
        help="Regenerate only Panels B and D.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all figure text (default: 1.0).",
    )
    return parser.parse_args()


def load_polymorphic_loci(args: argparse.Namespace) -> pd.DataFrame:
    annotation = pd.read_csv(
        args.annotation,
        sep="\t",
        usecols=["#chrom", "start", "motif_length", "motif", "functional_region"],
        dtype={"#chrom": "string", "start": "int64", "motif_length": "int64"},
    ).rename(columns={"#chrom": "chrom"})
    annotation["str_id"] = annotation["chrom"] + "_" + annotation["start"].astype(str)
    if annotation["str_id"].duplicated().any():
        raise ValueError("The functional annotation contains duplicate STR identifiers")

    variance = pd.read_csv(args.variance, dtype={"str_id": "string"})
    if variance["str_id"].duplicated().any():
        raise ValueError("Variance input contains duplicate STR identifiers")

    result = variance.merge(
        annotation, on="str_id", how="left", validate="one_to_one"
    )
    missing = result[["variance", "motif_length", "functional_region"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} polymorphic loci could not be fully annotated")
    if not (result["variance"] > 0).all():
        raise ValueError("Variance input contains non-polymorphic loci")
    if not result["motif_length"].isin(MOTIF_ORDER).all():
        raise ValueError("Unexpected motif length in polymorphic loci")
    if not result["functional_region"].isin(REGION_ORDER).all():
        raise ValueError("Unexpected functional-region category")
    result["log10_variance"] = np.log10(result["variance"])
    return result


def context_source_tables(loci: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    total = len(loci)

    region = (
        loci["functional_region"].value_counts().reindex(REGION_ORDER, fill_value=0).rename_axis("functional_region").reset_index(name="count")
    )
    region["label"] = region["functional_region"].map(REGION_LABELS)
    region["percentage"] = 100 * region["count"] / total
    region.to_csv(source_dir / "panel_c_functional_regions.tsv", sep="\t", index=False)

    loci[["str_id", "chrom", "start", "functional_region", "motif_length", "n_called", "variance", "log10_variance"]].to_csv(
        source_dir / "variance_by_genomic_context.tsv", sep="\t", index=False
    )
    context_summary = (
        loci.groupby("functional_region", observed=True)["variance"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reindex(REGION_ORDER)
        .reset_index()
    )
    context_summary["label"] = context_summary["functional_region"].map(REGION_LABELS)
    context_summary.to_csv(source_dir / "variance_by_genomic_context_summary.tsv", sep="\t", index=False)
    return region


def motif_source_tables(loci: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    total = len(loci)

    motif = (
        loci["motif_length"].value_counts().reindex(MOTIF_ORDER, fill_value=0).rename_axis("motif_length").reset_index(name="count")
    )
    motif["percentage"] = 100 * motif["count"] / total
    motif.to_csv(source_dir / "panel_b_motif_composition.tsv", sep="\t", index=False)

    loci[["str_id", "chrom", "start", "motif_length", "motif", "functional_region", "n_called", "variance", "log10_variance"]].to_csv(
        source_dir / "panel_d_variance_by_motif.tsv", sep="\t", index=False
    )
    return motif


def source_tables(loci: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    motif = motif_source_tables(loci, output_dir)
    region = context_source_tables(loci, output_dir)
    return motif, region


def label_vertical_bars(ax: plt.Axes, table: pd.DataFrame) -> None:
    offset = max(table["count"].max() * 0.018, 1)
    for patch, row in zip(ax.patches, table.itertuples(index=False)):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + offset,
            f"{row.percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["xtick.labelsize"],
        )
    ax.set_ylim(0, table["count"].max() * 1.17)


def draw_panel_b(
    ax: plt.Axes, motif: pd.DataFrame, title: str = "Motif-length composition", panel_label: str = "b"
) -> None:
    sns.barplot(data=motif, x="motif_length", y="count", hue="motif_length", palette=MOTIF_COLORS, legend=False, ax=ax)
    label_vertical_bars(ax, motif)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Motif length (bp)")
    ax.set_ylabel("Polymorphic STR loci")
    ax.ticklabel_format(axis="y", style="plain")
    ax.grid(False)
    add_panel_label(ax, panel_label)
    sns.despine(ax=ax)


def draw_panel_c(
    ax: plt.Axes, region: pd.DataFrame, title: str = "Genomic context", panel_label: str = "c"
) -> None:
    plot_data = region[~region["functional_region"].eq("other_exon")].copy()
    sns.barplot(data=plot_data, y="label", x="count", hue="label", palette=REGION_COLORS, legend=False, ax=ax)
    offset = max(plot_data["count"].max() * 0.015, 1)
    for patch, row in zip(ax.patches, plot_data.itertuples(index=False)):
        ax.text(
            patch.get_width() + offset,
            patch.get_y() + patch.get_height() / 2,
            f"{row.percentage:.1f}%",
            ha="left",
            va="center",
            fontsize=plt.rcParams["xtick.labelsize"],
        )
    ax.set_xlim(0, plot_data["count"].max() * 1.28)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Polymorphic STR loci")
    ax.set_ylabel("")
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(False)
    add_panel_label(ax, panel_label)
    sns.despine(ax=ax)


def draw_panel_d(
    ax: plt.Axes,
    loci: pd.DataFrame,
    title: str = "Variance by motif length",
    show_counts: bool = True,
    panel_label: str = "d",
) -> None:
    sns.violinplot(
        data=loci,
        x="motif_length",
        y="log10_variance",
        order=MOTIF_ORDER,
        hue="motif_length",
        palette=MOTIF_COLORS,
        legend=False,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=0.7,
        ax=ax,
    )
    sns.boxplot(
        data=loci,
        x="motif_length",
        y="log10_variance",
        order=MOTIF_ORDER,
        width=0.18,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.6},
        medianprops={"color": "#111111", "linewidth": 0.8},
        whiskerprops={"color": "#333333", "linewidth": 0.6},
        capprops={"color": "#333333", "linewidth": 0.6},
        ax=ax,
    )
    counts = loci["motif_length"].value_counts().reindex(MOTIF_ORDER)
    labels = [f"{value}\n(n={counts[value]:,})" for value in MOTIF_ORDER] if show_counts else MOTIF_ORDER
    ax.set_xticks(range(len(MOTIF_ORDER)), labels)
    if show_counts:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.axhline(np.log10(0.1), color="#333333", linestyle="--", linewidth=0.7, label="Variance = 0.1")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Motif length (bp)")
    ax.set_ylabel(r"$\log_{10}$(variance)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax.grid(False)
    add_panel_label(ax, panel_label)
    sns.despine(ax=ax)


def draw_variance_by_context(ax: plt.Axes, loci: pd.DataFrame) -> None:
    context_order = [value for value in REGION_ORDER if value != "other_exon"]
    context_palette = dict(zip(context_order, REGION_COLORS))
    plot_data = loci[loci["functional_region"].isin(context_order)].copy()
    sns.violinplot(
        data=plot_data,
        x="functional_region",
        y="log10_variance",
        order=context_order,
        hue="functional_region",
        hue_order=context_order,
        palette=context_palette,
        legend=False,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=0.7,
        ax=ax,
    )
    sns.boxplot(
        data=plot_data,
        x="functional_region",
        y="log10_variance",
        order=context_order,
        width=0.16,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.6},
        medianprops={"color": "#111111", "linewidth": 0.8},
        whiskerprops={"color": "#333333", "linewidth": 0.6},
        capprops={"color": "#333333", "linewidth": 0.6},
        ax=ax,
    )
    counts = plot_data["functional_region"].value_counts().reindex(context_order)
    labels = [f"{REGION_LABELS[value]}\n(n={counts[value]:,})" for value in context_order]
    ax.set_xticks(range(len(context_order)), labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.axhline(np.log10(0.1), color="#333333", linestyle="--", linewidth=0.7, label="Variance = 0.1")
    ax.set_title("Variance by genomic context", loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$(variance)")
    ax.legend(frameon=False, loc="lower left", handlelength=1.4)
    ax.grid(False)
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def individual_panels(loci: pd.DataFrame, motif: pd.DataFrame, region: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    for name, size, bottom, drawer, data in (
        ("panel_b_motif_composition", (NATURE_SINGLE_COLUMN_IN, 2.7), 0.18, draw_panel_b, motif),
        ("panel_d_variance_by_motif", (NATURE_SINGLE_COLUMN_IN, 3.0), 0.29, draw_panel_d, loci),
    ):
        fig, ax = plt.subplots(figsize=size)
        drawer(ax, data)
        fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=bottom)
        save_figure(fig, figure_dir / name)


def context_panels(loci: pd.DataFrame, region: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COLUMN_IN, 2.7))
    draw_panel_c(ax, region)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.18)
    save_figure(fig, figure_dir / "panel_c_functional_regions")

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COLUMN_IN, 3.0))
    draw_variance_by_context(ax, loci)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.89, bottom=0.29)
    save_figure(fig, figure_dir / "variance_by_genomic_context")


def draw_panel_a(ax_density: plt.Axes, windows: pd.DataFrame) -> None:
    chrom_order = [str(i) for i in range(1, 24)]
    density = windows.pivot(index="chrom", columns="window_index", values="polymorphic_density_per_mb").reindex(chrom_order)

    sns.heatmap(
        density, ax=ax_density, cmap="viridis", mask=density.isna(),
        linewidths=0, cbar_kws={"label": "pSTRs per Mb", "shrink": 0.72, "aspect": 18, "pad": 0.018}
    )

    ticks = np.arange(0, density.shape[1] + 1, 5)
    ax_density.set_xticks(ticks)
    ax_density.set_xticklabels([str(value) for value in ticks], rotation=0)
    ax_density.set_title("Distribution of polymorphic STRs across chromosomes 1-23", loc="left", fontweight="bold")
    ax_density.set_xlabel("Genomic position (Mb)")
    chromosome_ticks = np.arange(len(chrom_order)) + 0.5
    ax_density.set_yticks(chromosome_ticks, chrom_order)
    ax_density.set_ylabel("Chromosome")
    ax_density.set_yticklabels(ax_density.get_yticklabels(), rotation=0)
    ax_density.tick_params(length=0)
    ax_density.grid(False)
    add_panel_label(ax_density, "a", x=-0.055, y=1.04)


def combined_figure(
    loci: pd.DataFrame, motif: pd.DataFrame, region: pd.DataFrame, windows: pd.DataFrame,
    output_dir: Path
) -> None:
    fig = plt.figure(figsize=(NATURE_DOUBLE_COLUMN_IN, 6.15))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.18, 1.0], hspace=0.48, wspace=0.62)
    ax_density = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    draw_panel_a(ax_density, windows)
    draw_panel_b(ax_b, motif)
    draw_panel_c(ax_c, region)
    draw_panel_d(ax_d, loci, show_counts=False)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.955, bottom=0.095)
    save_figure(fig, output_dir / "figures/main_figure_str_landscape")


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be greater than zero")
    if args.context_only and args.motif_only:
        raise ValueError("--context-only and --motif-only cannot be used together")
    partial_run = args.context_only or args.motif_only
    required_paths = (args.annotation, args.variance) if partial_run else (
        args.annotation,
        args.variance,
        args.panel_a_source,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    set_nature_style(args.font_scale)
    loci = load_polymorphic_loci(args)
    if args.context_only:
        region = context_source_tables(loci, args.output_dir)
        context_panels(loci, region, args.output_dir)
        print(f"Created genomic-context figures from {len(loci):,} polymorphic loci")
        return
    if args.motif_only:
        motif = motif_source_tables(loci, args.output_dir)
        individual_panels(loci, motif, pd.DataFrame(), args.output_dir)
        print(f"Created motif figures from {len(loci):,} polymorphic loci")
        return

    motif, region = source_tables(loci, args.output_dir)
    windows = pd.read_csv(args.panel_a_source, sep="\t", dtype={"chrom": "string"})
    if int(windows["polymorphic_count"].sum()) != int(loci["chrom"].isin([str(i) for i in range(1, 24)]).sum()):
        raise ValueError("Panel A source counts do not match polymorphic chromosome loci")
    individual_panels(loci, motif, region, args.output_dir)
    context_panels(loci, region, args.output_dir)
    combined_figure(loci, motif, region, windows, args.output_dir)
    print(f"Created Panels B-D and combined main figure from {len(loci):,} polymorphic loci")


if __name__ == "__main__":
    main()
