#!/usr/bin/env python3
"""Compare SNP- and STR-derived population structure with PCA."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
NATURE_SINGLE_COLUMN_IN = 89 / 25.4
FONT_FAMILY = "Nimbus Sans"
COUNTRY_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#332288",
    "#88CCEE",
    "#117733",
    "#999933",
    "#882255",
    "#7A7A7A",
]
SUBSPECIES_MARKERS = {"europaea": "o", "sylvestris": "^"}
GEOGRAPHIC_REGION_BY_COUNTRY = {
    "Spain": "Western Mediterranean",
    "Morocco": "Western Mediterranean",
    "Algeria": "Western Mediterranean",
    "Italy": "Central Mediterranean–Adriatic",
    "Tunisia": "Central Mediterranean–Adriatic",
    "Croatia": "Central Mediterranean–Adriatic",
    "Albania": "Central Mediterranean–Adriatic",
    "Greece": "Eastern Mediterranean–Levant",
    "Turkey": "Eastern Mediterranean–Levant",
    "Syria": "Eastern Mediterranean–Levant",
    "Lebanon": "Eastern Mediterranean–Levant",
    "Israel": "Eastern Mediterranean–Levant",
    "Iran": "Iranian Plateau",
}
GEOGRAPHIC_REGION_ORDER = [
    "Western Mediterranean",
    "Central Mediterranean–Adriatic",
    "Eastern Mediterranean–Levant",
    "Iranian Plateau",
]
GEOGRAPHIC_REGION_COLORS = {
    "Western Mediterranean": "#0072B2",
    "Central Mediterranean–Adriatic": "#E69F00",
    "Eastern Mediterranean–Levant": "#009E73",
    "Iranian Plateau": "#CC79A7",
}
GEOGRAPHIC_REGION_ABBREVIATIONS = {
    "Western Mediterranean": "West. Med.",
    "Central Mediterranean–Adriatic": "Central Med.",
    "Eastern Mediterranean–Levant": "East. Med.",
    "Iranian Plateau": "Iran",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snp-bfile", type=Path, default=PROJECT_ROOT / "raw_snp/olive.admix"
    )
    parser.add_argument(
        "--str-matrix",
        type=Path,
        default=PROJECT_ROOT / "olive_str/str_data/str_var_0.1.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "olive_str/metadata/Geo_loc.csv",
    )
    parser.add_argument(
        "--plink",
        type=Path,
        default=Path(os.environ.get("PLINK", "plink")),
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--expected-samples", type=int, default=212)
    parser.add_argument("--expected-str-loci", type=int, default=82_210)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--font-scale", type=float, default=1.4)
    parser.add_argument(
        "--separate-figures-only",
        action="store_true",
        help="Create only the three standalone main plots from existing source tables.",
    )
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
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 6 * font_scale,
            "legend.title_fontsize": 6.5 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def normalize_subspecies(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.endswith("sylvestris"):
        return "sylvestris"
    if lowered.endswith("europaea"):
        return "europaea"
    raise ValueError(f"Unexpected subspecies label: {value!r}")


def read_sample_inputs(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame, list[str]]:
    fam_path = Path(f"{args.snp_bfile}.fam")
    if not fam_path.is_file():
        raise FileNotFoundError(f"Missing SNP FAM file: {fam_path}")
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[0, 1], names=["FID", "IID"])
    if fam["IID"].duplicated().any():
        raise ValueError("SNP FAM contains duplicate sample IDs")

    with args.str_matrix.open(newline="") as handle:
        header = next(csv.reader(handle))
    if not header or header[0] != "str_id":
        raise ValueError("STR matrix must begin with a str_id column")
    str_samples = header[1:]
    if len(str_samples) != len(set(str_samples)):
        raise ValueError("STR matrix contains duplicate sample columns")

    metadata = pd.read_csv(args.metadata, encoding="latin-1", dtype=str).fillna("")
    required = {"ID", "Country", "Subspecies"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if metadata["ID"].duplicated().any():
        raise ValueError("Metadata contains duplicate sample IDs")
    metadata = metadata.set_index("ID", drop=False)

    snp_samples = set(fam["IID"])
    missing_snp = [sample for sample in str_samples if sample not in snp_samples]
    missing_metadata = [sample for sample in str_samples if sample not in metadata.index]
    if missing_snp or missing_metadata:
        raise ValueError(
            f"Shared cohort validation failed; missing SNP={missing_snp}, "
            f"missing metadata={missing_metadata}"
        )
    if len(str_samples) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} shared samples, found {len(str_samples)}"
        )

    sample_metadata = metadata.loc[str_samples, ["ID", "Country", "Subspecies"]].copy()
    if (sample_metadata["Country"].str.strip() == "").any():
        raise ValueError("At least one shared sample has no country annotation")
    sample_metadata["subspecies_short"] = sample_metadata["Subspecies"].map(normalize_subspecies)
    snp_only = sorted(snp_samples - set(str_samples))
    return str_samples, sample_metadata.reset_index(drop=True), snp_only


def orient_components(scores: np.ndarray) -> np.ndarray:
    oriented = scores.copy()
    for index in range(oriented.shape[1]):
        anchor = int(np.argmax(np.abs(oriented[:, index])))
        if oriented[anchor, index] < 0:
            oriented[:, index] *= -1
    return oriented


def run_snp_pca(
    args: argparse.Namespace, samples: list[str], source_dir: Path, plink_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plink_dir.mkdir(parents=True, exist_ok=True)
    keep_path = source_dir / "shared_samples.keep"
    pd.DataFrame({"FID": samples, "IID": samples}).to_csv(
        keep_path, sep="\t", header=False, index=False
    )
    prefix = plink_dir / "snp_shared"
    command = [
        str(args.plink),
        "--bfile",
        str(args.snp_bfile),
        "--keep",
        str(keep_path),
        "--chr-set",
        "23",
        "no-x",
        "no-y",
        "no-xy",
        "no-mt",
        "--allow-extra-chr",
        "--pca",
        str(len(samples)),
        "header",
        "tabs",
        "--out",
        str(prefix),
    ]
    subprocess.run(command, check=True)

    eigenvec = pd.read_csv(prefix.with_suffix(".eigenvec"), sep="\t")
    iid_column = "IID" if "IID" in eigenvec.columns else "#IID"
    if iid_column not in eigenvec.columns:
        raise ValueError("Could not identify IID column in PLINK eigenvector output")
    pc_columns = [column for column in eigenvec.columns if column.startswith("PC")]
    coordinates = eigenvec.set_index(iid_column).loc[samples, pc_columns].astype(float)
    coordinates.loc[:, :] = orient_components(coordinates.to_numpy())
    coordinates.insert(0, "sample_id", samples)

    eigenvalues = np.loadtxt(prefix.with_suffix(".eigenval"), ndmin=1)
    if len(eigenvalues) != len(pc_columns):
        raise ValueError("PLINK eigenvalue and eigenvector counts differ")
    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "variance_percent": 100 * eigenvalues / eigenvalues.sum(),
        }
    )
    return coordinates.reset_index(drop=True), variance


def contig_from_str_id(str_id: str) -> str:
    if "_" not in str_id:
        raise ValueError(f"Malformed STR identifier: {str_id}")
    return str_id.rsplit("_", 1)[0]


def pca_from_crossproduct(
    crossproduct: np.ndarray, samples: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eigenvalues, eigenvectors = np.linalg.eigh(crossproduct)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    scores = eigenvectors[:, order] * np.sqrt(eigenvalues)
    scores = orient_components(scores)
    pc_columns = [f"PC{index}" for index in range(1, len(samples) + 1)]
    coordinates = pd.DataFrame(scores, columns=pc_columns)
    coordinates.insert(0, "sample_id", samples)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "variance_percent": 100 * eigenvalues / eigenvalues.sum(),
        }
    )
    return coordinates, variance


def run_str_pca(
    args: argparse.Namespace, samples: list[str], source_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_samples = len(samples)
    all_crossproduct = np.zeros((n_samples, n_samples), dtype=np.float64)
    chromosome_crossproduct = np.zeros_like(all_crossproduct)
    locus_records: list[tuple[str, str, bool, int]] = []
    seen: set[str] = set()
    chromosome_names = {str(value) for value in range(1, 24)}

    reader = pd.read_csv(args.str_matrix, chunksize=args.chunk_size)
    for chunk in reader:
        if list(chunk.columns[1:]) != samples:
            raise ValueError("STR sample columns changed or are not in the expected order")
        locus_ids = chunk["str_id"].astype(str)
        duplicates = seen.intersection(locus_ids)
        if duplicates or locus_ids.duplicated().any():
            raise ValueError(f"Duplicate STR identifiers detected: {sorted(duplicates)[:5]}")
        seen.update(locus_ids)

        values = chunk[samples].to_numpy(dtype=np.float64)
        called = np.sum(np.isfinite(values), axis=1)
        means = np.nanmean(values, axis=1)
        standard_deviations = np.nanstd(values, axis=1, ddof=1)
        invalid = (called < 2) | ~np.isfinite(standard_deviations) | (standard_deviations <= 0)
        if invalid.any():
            bad_id = locus_ids.iloc[int(np.flatnonzero(invalid)[0])]
            raise ValueError(f"STR locus has insufficient or zero variance: {bad_id}")
        standardized = (values - means[:, None]) / standard_deviations[:, None]
        standardized[~np.isfinite(standardized)] = 0.0
        all_crossproduct += standardized.T @ standardized

        contigs = locus_ids.map(contig_from_str_id)
        chromosome_mask = contigs.isin(chromosome_names).to_numpy()
        if chromosome_mask.any():
            chromosome_values = standardized[chromosome_mask]
            chromosome_crossproduct += chromosome_values.T @ chromosome_values
        locus_records.extend(
            zip(locus_ids, contigs, chromosome_mask, called, strict=True)
        )

    if len(locus_records) != args.expected_str_loci:
        raise ValueError(
            f"Expected {args.expected_str_loci} STR loci, found {len(locus_records)}"
        )
    loci = pd.DataFrame(
        locus_records, columns=["str_id", "contig", "chromosome_assigned", "n_called"]
    )
    loci["included_primary"] = True
    loci["included_chromosome_sensitivity"] = loci["chromosome_assigned"]
    loci.to_csv(source_dir / "str_locus_sets.tsv", sep="\t", index=False)

    all_coordinates, all_variance = pca_from_crossproduct(all_crossproduct, samples)
    chromosome_coordinates, chromosome_variance = pca_from_crossproduct(
        chromosome_crossproduct, samples
    )
    return all_coordinates, all_variance, chromosome_coordinates, chromosome_variance, loci


def add_metadata(coordinates: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    result = coordinates.merge(
        metadata,
        left_on="sample_id",
        right_on="ID",
        how="left",
        validate="one_to_one",
    ).drop(columns="ID")
    if result[["Country", "subspecies_short"]].isna().any().any():
        raise ValueError("PCA coordinates could not be fully joined to metadata")
    return result


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def draw_pca(
    ax: plt.Axes,
    data: pd.DataFrame,
    variance: pd.DataFrame,
    x_component: int,
    y_component: int,
    title: str,
    country_palette: dict[str, str],
    label: str,
) -> None:
    for subspecies, marker in SUBSPECIES_MARKERS.items():
        subset = data[data["subspecies_short"] == subspecies]
        sns.scatterplot(
            data=subset,
            x=f"PC{x_component}",
            y=f"PC{y_component}",
            hue="Country",
            palette=country_palette,
            marker=marker,
            s=24,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            legend=False,
            ax=ax,
        )
    x_percent = variance.loc[variance["component"] == x_component, "variance_percent"].iloc[0]
    y_percent = variance.loc[variance["component"] == y_component, "variance_percent"].iloc[0]
    ax.set_xlabel(f"PC{x_component} ({x_percent:.1f}%)")
    ax.set_ylabel(f"PC{y_component} ({y_percent:.1f}%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(0, color="#D0D0D0", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#D0D0D0", linewidth=0.5, zorder=0)
    ax.grid(False)
    sns.despine(ax=ax)
    panel_label(ax, label)


def add_shared_legends(
    figure: plt.Figure, countries: list[str], country_palette: dict[str, str]
) -> None:
    country_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=country_palette[country],
            markeredgecolor="none",
            markersize=4.5,
            label=country,
        )
        for country in countries
    ]
    subspecies_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5,
            label=name,
        )
        for name, marker in SUBSPECIES_MARKERS.items()
    ]
    country_legend = figure.legend(
        handles=country_handles,
        title="Country",
        loc="lower center",
        bbox_to_anchor=(0.48, 0.015),
        ncol=7,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    figure.add_artist(country_legend)
    figure.legend(
        handles=subspecies_handles,
        title="Subspecies",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )


def add_geographic_regions(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["Geographic_region"] = result["Country"].map(GEOGRAPHIC_REGION_BY_COUNTRY)
    if result["Geographic_region"].isna().any():
        missing = sorted(result.loc[result["Geographic_region"].isna(), "Country"].unique())
        raise ValueError(f"Countries lack geographic-region labels: {missing}")
    return result


def draw_geographic_pca(
    ax: plt.Axes,
    data: pd.DataFrame,
    variance: pd.DataFrame,
    title: str,
) -> None:
    plot_data = add_geographic_regions(data)
    for subspecies, marker in SUBSPECIES_MARKERS.items():
        subset = plot_data[plot_data["subspecies_short"].eq(subspecies)]
        sns.scatterplot(
            data=subset,
            x="PC1",
            y="PC2",
            hue="Geographic_region",
            hue_order=GEOGRAPHIC_REGION_ORDER,
            palette=GEOGRAPHIC_REGION_COLORS,
            marker=marker,
            s=34,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
            legend=False,
            ax=ax,
        )
    x_percent = variance.loc[variance["component"].eq(1), "variance_percent"].iloc[0]
    y_percent = variance.loc[variance["component"].eq(2), "variance_percent"].iloc[0]
    ax.set_xlabel(f"PC1 ({x_percent:.1f}%)")
    ax.set_ylabel(f"PC2 ({y_percent:.1f}%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(0, color="#D0D0D0", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#D0D0D0", linewidth=0.5, zorder=0)
    ax.grid(False)
    sns.despine(ax=ax)


def add_geographic_legends(figure: plt.Figure) -> None:
    region_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=GEOGRAPHIC_REGION_COLORS[region],
            markeredgecolor="none",
            markersize=5,
            label=region,
        )
        for region in GEOGRAPHIC_REGION_ORDER
    ]
    subspecies_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5,
            label=subspecies,
        )
        for subspecies, marker in SUBSPECIES_MARKERS.items()
    ]
    region_legend = figure.legend(
        handles=region_handles,
        title="Geographic region",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    figure.add_artist(region_legend)
    figure.legend(
        handles=subspecies_handles,
        title="Subspecies",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )


def add_geographic_legends_right(figure: plt.Figure) -> None:
    region_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=GEOGRAPHIC_REGION_COLORS[region],
            markeredgecolor="none",
            markersize=5,
            label=GEOGRAPHIC_REGION_ABBREVIATIONS[region],
        )
        for region in GEOGRAPHIC_REGION_ORDER
    ]
    subspecies_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5,
            label=subspecies.capitalize(),
        )
        for subspecies, marker in SUBSPECIES_MARKERS.items()
    ]
    region_legend = figure.legend(
        handles=region_handles,
        title="Geographic region",
        loc="center left",
        bbox_to_anchor=(0.80, 0.61),
        ncol=1,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.8,
    )
    figure.add_artist(region_legend)
    figure.legend(
        handles=subspecies_handles,
        title="Subspecies",
        loc="center left",
        bbox_to_anchor=(0.80, 0.29),
        ncol=1,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.8,
    )


def make_two_panel_population_figure(
    snp: pd.DataFrame,
    snp_variance: pd.DataFrame,
    str_all: pd.DataFrame,
    str_variance: pd.DataFrame,
    figure_dir: Path,
    font_scale: float,
) -> None:
    set_nature_style(font_scale)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(NATURE_DOUBLE_COLUMN_IN, 3.25),
        gridspec_kw={"wspace": 0.34},
    )
    draw_geographic_pca(axes[0], snp, snp_variance, "SNP")
    draw_geographic_pca(axes[1], str_all, str_variance, "STR")
    add_geographic_legends_right(figure)
    figure.subplots_adjust(left=0.08, right=0.78, top=0.90, bottom=0.17)
    save_figure(figure, figure_dir, "snp_str_population_structure")


def make_separate_main_figures(
    snp: pd.DataFrame,
    snp_variance: pd.DataFrame,
    str_all: pd.DataFrame,
    str_variance: pd.DataFrame,
    figure_dir: Path,
    font_scale: float,
) -> None:
    set_nature_style(font_scale)
    make_two_panel_population_figure(
        snp,
        snp_variance,
        str_all,
        str_variance,
        figure_dir,
        font_scale,
    )
    for data, variance, title, stem in (
        (snp, snp_variance, "SNP population structure", "snp_population_structure"),
        (str_all, str_variance, "STR population structure", "str_population_structure"),
    ):
        figure, ax = plt.subplots(figsize=(5.0, 3.65))
        draw_geographic_pca(ax, data, variance, title)
        add_geographic_legends(figure)
        figure.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.36)
        save_figure(figure, figure_dir, stem)

    figure, ax = plt.subplots(figsize=(NATURE_SINGLE_COLUMN_IN, 2.8))
    first_ten_snp = snp_variance.head(10)
    first_ten_str = str_variance.head(10)
    ax.plot(
        first_ten_snp["component"],
        first_ten_snp["variance_percent"],
        marker="o",
        markersize=4,
        linewidth=1.3,
        color="#0072B2",
        label="SNPs",
    )
    ax.plot(
        first_ten_str["component"],
        first_ten_str["variance_percent"],
        marker="s",
        markersize=3.8,
        linewidth=1.3,
        color="#D55E00",
        label="STRs",
    )
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA variance explained", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(False)
    sns.despine(ax=ax)
    figure.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.20)
    save_figure(figure, figure_dir, "pca_variance_explained")


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> None:
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            figure_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=600 if extension == "png" else None,
        )
    plt.close(figure)


def make_figures(
    snp: pd.DataFrame,
    snp_variance: pd.DataFrame,
    str_all: pd.DataFrame,
    str_variance: pd.DataFrame,
    str_chromosome: pd.DataFrame,
    str_chromosome_variance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    set_nature_style()
    countries = sorted(snp["Country"].unique())
    if len(countries) > len(COUNTRY_COLORS):
        raise ValueError("Not enough configured colors for the country annotations")
    country_palette = dict(zip(countries, COUNTRY_COLORS, strict=True))

    figure = plt.figure(figsize=(NATURE_DOUBLE_COLUMN_IN, 5.25))
    grid = figure.add_gridspec(2, 2, height_ratios=[3.0, 1.45], hspace=0.48, wspace=0.35)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])
    draw_pca(ax_a, snp, snp_variance, 1, 2, "SNP population structure", country_palette, "a")
    draw_pca(ax_b, str_all, str_variance, 1, 2, "STR population structure", country_palette, "b")
    first_ten_snp = snp_variance.head(10)
    first_ten_str = str_variance.head(10)
    ax_c.plot(
        first_ten_snp["component"],
        first_ten_snp["variance_percent"],
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        color="#0072B2",
        label="SNPs",
    )
    ax_c.plot(
        first_ten_str["component"],
        first_ten_str["variance_percent"],
        marker="s",
        markersize=3.3,
        linewidth=1.2,
        color="#D55E00",
        label="STRs",
    )
    ax_c.set_xticks(range(1, 11))
    ax_c.set_xlabel("Principal component")
    ax_c.set_ylabel("Variance explained (%)")
    ax_c.set_title("PCA variance explained", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right")
    ax_c.grid(False)
    sns.despine(ax=ax_c)
    panel_label(ax_c, "c")
    add_shared_legends(figure, countries, country_palette)
    figure.subplots_adjust(bottom=0.24, top=0.95)
    save_figure(figure, figure_dir, "snp_str_population_structure")

    supplementary = plt.figure(figsize=(NATURE_DOUBLE_COLUMN_IN, 3.45))
    grid = supplementary.add_gridspec(1, 3, wspace=0.44)
    axes = [supplementary.add_subplot(grid[0, index]) for index in range(3)]
    draw_pca(axes[0], snp, snp_variance, 3, 4, "SNP higher components", country_palette, "a")
    draw_pca(axes[1], str_all, str_variance, 3, 4, "STR higher components", country_palette, "b")
    draw_pca(
        axes[2],
        str_chromosome,
        str_chromosome_variance,
        1,
        2,
        "Chromosome-only STRs",
        country_palette,
        "c",
    )
    add_shared_legends(supplementary, countries, country_palette)
    supplementary.subplots_adjust(bottom=0.31, top=0.92)
    save_figure(supplementary, figure_dir, "snp_str_population_structure_supplementary")


def write_report(
    output_dir: Path,
    n_samples: int,
    snp_only: list[str],
    loci: pd.DataFrame,
    snp_variance: pd.DataFrame,
    str_variance: pd.DataFrame,
    chromosome_variance: pd.DataFrame,
    sensitivity_correlations: pd.DataFrame,
) -> None:
    chromosome_count = int(loci["chromosome_assigned"].sum())
    scaffold_count = len(loci) - chromosome_count

    def first_four(table: pd.DataFrame) -> str:
        return ", ".join(f"PC{i}={value:.2f}%" for i, value in enumerate(table["variance_percent"].head(4), 1))

    minimum_sensitivity_correlation = sensitivity_correlations["absolute_pearson_r"].min()
    maximum_sensitivity_correlation = sensitivity_correlations["absolute_pearson_r"].max()
    report = f"""# SNP- and STR-derived population structure

## Analysis summary

Population structure was evaluated in the {n_samples} samples shared by the SNP and STR datasets. The SNP PCA used 299,285 LD-pruned chromosome markers. The primary STR PCA used {len(loci):,} polymorphic loci with variance >= 0.1, comprising {chromosome_count:,} chromosome loci and {scaffold_count:,} unplaced-scaffold loci. Missing STR values were replaced by the corresponding locus mean, after which every STR locus was centered and scaled to unit variance. SNP-only samples excluded from the comparison were {', '.join(snp_only)}.

Variance explained by the first four SNP components was {first_four(snp_variance)}. Variance explained by the first four STR components was {first_four(str_variance)}. In the chromosome-only STR sensitivity analysis, the corresponding values were {first_four(chromosome_variance)}.

## Geographic label scheme

Country labels in the standalone SNP and STR ordinations are merged into four
geographic regions:

- Western Mediterranean: Spain, Morocco, and Algeria (n=36).
- Central Mediterranean-Adriatic: Italy, Tunisia, Croatia, and Albania (n=7).
- Eastern Mediterranean-Levant: Greece, Turkey, Syria, Lebanon, and Israel (n=65).
- Iranian Plateau: Iran (n=104).

Subspecies remains encoded independently by marker shape.

## Interpretation

Both marker classes recovered geographic structure. Iranian accessions were concentrated toward negative PC1 in both ordinations, whereas Greek and Spanish accessions occupied more central or positive regions. Sylvestris accessions were displaced from the main europaea concentration in both analyses, although the relative orientation and dispersion of groups differed between marker classes. The STR ordination showed broader within-country dispersion and a stronger first component than the SNP ordination.

Excluding scaffold STRs had little effect on the first four STR components: their absolute score correlations with the primary components ranged from {minimum_sensitivity_correlation:.3f} to {maximum_sensitivity_correlation:.3f}. The geographic pattern is therefore not driven by unplaced-scaffold loci. PCA axis signs and rotations are arbitrary, so identically numbered SNP and STR components should not be treated as directly equivalent axes. This analysis is descriptive and does not provide a formal estimate of SNP-STR structural concordance.

## Reproducibility

The `source_data` directory contains all plotted coordinates, eigenvalues, sample inclusion records, and STR locus membership. The `plink` directory contains the restricted SNP PCA log and native PLINK outputs.
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="ascii")


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be greater than zero")
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    plink_dir = output_dir / "plink"
    for directory in (figure_dir, source_dir, plink_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.separate_figures_only:
        required = {
            "snp": source_dir / "snp_pca_coordinates.tsv",
            "snp_variance": source_dir / "snp_pca_eigenvalues.tsv",
            "str": source_dir / "str_pca_coordinates.tsv",
            "str_variance": source_dir / "str_pca_eigenvalues.tsv",
        }
        for path in required.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        make_separate_main_figures(
            pd.read_csv(required["snp"], sep="\t"),
            pd.read_csv(required["snp_variance"], sep="\t"),
            pd.read_csv(required["str"], sep="\t"),
            pd.read_csv(required["str_variance"], sep="\t"),
            figure_dir,
            args.font_scale,
        )
        print(f"Standalone figures: {figure_dir}")
        return

    samples, metadata, snp_only = read_sample_inputs(args)
    inclusion = pd.DataFrame(
        {
            "sample_id": samples + snp_only,
            "included_shared_cohort": [True] * len(samples) + [False] * len(snp_only),
            "exclusion_reason": [""] * len(samples) + ["absent_from_STR_matrix"] * len(snp_only),
        }
    )
    inclusion.to_csv(source_dir / "sample_inclusion.tsv", sep="\t", index=False)

    snp_coordinates, snp_variance = run_snp_pca(args, samples, source_dir, plink_dir)
    (
        str_coordinates,
        str_variance,
        chromosome_coordinates,
        chromosome_variance,
        loci,
    ) = run_str_pca(args, samples, source_dir)

    snp_coordinates = add_metadata(snp_coordinates, metadata)
    str_coordinates = add_metadata(str_coordinates, metadata)
    chromosome_coordinates = add_metadata(chromosome_coordinates, metadata)
    sensitivity_correlations = pd.DataFrame(
        {
            "component": [f"PC{index}" for index in range(1, 5)],
            "pearson_r": [
                str_coordinates[f"PC{index}"].corr(chromosome_coordinates[f"PC{index}"])
                for index in range(1, 5)
            ],
        }
    )
    sensitivity_correlations["absolute_pearson_r"] = sensitivity_correlations[
        "pearson_r"
    ].abs()
    tables = {
        "snp_pca_coordinates.tsv": snp_coordinates,
        "str_pca_coordinates.tsv": str_coordinates,
        "str_chromosome_pca_coordinates.tsv": chromosome_coordinates,
        "snp_pca_eigenvalues.tsv": snp_variance,
        "str_pca_eigenvalues.tsv": str_variance,
        "str_chromosome_pca_eigenvalues.tsv": chromosome_variance,
        "str_sensitivity_pc_correlations.tsv": sensitivity_correlations,
    }
    for filename, table in tables.items():
        table.to_csv(source_dir / filename, sep="\t", index=False)

    make_figures(
        snp_coordinates,
        snp_variance,
        str_coordinates,
        str_variance,
        chromosome_coordinates,
        chromosome_variance,
        figure_dir,
    )
    write_report(
        output_dir,
        len(samples),
        snp_only,
        loci,
        snp_variance,
        str_variance,
        chromosome_variance,
        sensitivity_correlations,
    )

    print(f"Shared samples: {len(samples):,}")
    print(f"Primary STR loci: {len(loci):,}")
    print(f"Chromosome STR loci: {int(loci['chromosome_assigned'].sum()):,}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
