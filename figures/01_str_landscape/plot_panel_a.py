#!/usr/bin/env python3
"""Plot the chromosome-scale density of polymorphic olive STRs."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/01_str_landscape"
INPUT_DIR = ANALYSIS_DIR / "inputs"
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=INPUT_DIR / "O_europeana_strs.functional.annotated.bed",
    )
    parser.add_argument(
        "--gff3",
        type=Path,
        default=INPUT_DIR / "Olea_europaea.OLEA9.60.gff3.gz",
    )
    parser.add_argument(
        "--variance",
        type=Path,
        default=INPUT_DIR / "str_length_variance.csv",
    )
    parser.add_argument("--window-size", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--font-scale", type=float, default=1.4)
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="Regenerate Panel A from its existing source table.",
    )
    return parser.parse_args()


def chromosome_lengths(gff3: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with gzip.open(gff3, "rt") as handle:
        for line in handle:
            if line.startswith("##sequence-region"):
                _, chrom, start, end = line.split()
                if chrom.isdigit() and 1 <= int(chrom) <= 23:
                    if int(start) != 1:
                        raise ValueError(f"Unexpected sequence start for chromosome {chrom}: {start}")
                    lengths[chrom] = int(end)
            elif not line.startswith("#"):
                break
    expected = {str(number) for number in range(1, 24)}
    if set(lengths) != expected:
        missing = sorted(expected - set(lengths), key=int)
        raise ValueError(f"Missing chromosome lengths in GFF3: {missing}")
    return dict(sorted(lengths.items(), key=lambda item: int(item[0])))


def load_loci(args: argparse.Namespace) -> tuple[pd.DataFrame, int, int]:
    annotation = pd.read_csv(
        args.annotation,
        sep="\t",
        usecols=["#chrom", "start", "end"],
        dtype={"#chrom": "string", "start": "int64", "end": "int64"},
    ).rename(columns={"#chrom": "chrom"})
    annotation["str_id"] = annotation["chrom"] + "_" + annotation["start"].astype(str)
    if annotation["str_id"].duplicated().any():
        duplicate = annotation.loc[annotation["str_id"].duplicated(), "str_id"].iloc[0]
        raise ValueError(f"Duplicate STR identifier in annotation: {duplicate}")

    evaluable = pd.read_csv(args.variance, usecols=["str_id"], dtype="string")
    if evaluable["str_id"].duplicated().any():
        raise ValueError("Duplicate STR identifier in variance input")

    polymorphic_ids = set(evaluable["str_id"])
    annotation_ids = set(annotation["str_id"])
    missing = polymorphic_ids - annotation_ids
    if missing:
        raise ValueError(f"Unmatched polymorphic STR IDs: {len(missing)}")

    annotation["polymorphic"] = annotation["str_id"].isin(polymorphic_ids)
    scaffold_polymorphic = int(
        annotation.loc[~annotation["chrom"].isin([str(i) for i in range(1, 24)]), "polymorphic"].sum()
    )
    return annotation, len(polymorphic_ids), scaffold_polymorphic


def make_windows(
    loci: pd.DataFrame, lengths: dict[str, int], window_size: int
) -> pd.DataFrame:
    chromosome_loci = loci[loci["chrom"].isin(lengths)].copy()
    chromosome_loci["window_index"] = chromosome_loci["start"] // window_size
    counts = (
        chromosome_loci.groupby(["chrom", "window_index"], observed=True)[
            ["polymorphic"]
        ]
        .sum()
        .reset_index()
        .rename(columns={"polymorphic": "polymorphic_count"})
    )

    rows = []
    for chrom, chrom_length in lengths.items():
        for index, start in enumerate(range(0, chrom_length, window_size)):
            end = min(start + window_size, chrom_length)
            rows.append((chrom, index, start, end, chrom_length))
    windows = pd.DataFrame(
        rows, columns=["chrom", "window_index", "window_start", "window_end", "chrom_length"]
    ).merge(counts, how="left", on=["chrom", "window_index"])
    windows[["polymorphic_count"]] = windows[["polymorphic_count"]].fillna(0).astype(int)

    windows["window_span_mb"] = (windows["window_end"] - windows["window_start"]) / 1_000_000
    windows["polymorphic_density_per_mb"] = windows["polymorphic_count"] / windows["window_span_mb"]
    return windows


def heatmap_matrix(windows: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = windows.pivot(index="chrom", columns="window_index", values=value)
    return matrix.reindex(index=[str(i) for i in range(1, 24)])


def plot_panel(windows: pd.DataFrame, output_dir: Path, font_scale: float) -> None:
    density = heatmap_matrix(windows, "polymorphic_density_per_mb")
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6 * font_scale,
        "axes.titlesize": 7 * font_scale,
        "axes.labelsize": 6 * font_scale,
        "xtick.labelsize": 5.5 * font_scale,
        "ytick.labelsize": 5.5 * font_scale,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COLUMN_IN, 3.6))

    sns.heatmap(
        density,
        ax=ax,
        cmap="viridis",
        mask=density.isna(),
        linewidths=0,
        cbar_kws={"label": "pSTRs per Mb", "shrink": 0.75, "pad": 0.018},
    )

    max_windows = density.shape[1]
    tick_positions = np.arange(0, max_windows + 1, 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(position) for position in tick_positions], rotation=0)
    ax.set_title("Distribution of polymorphic STRs across chromosomes 1-23", loc="left", fontweight="bold")
    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Chromosome")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(length=0)
    ax.grid(False)
    fig.subplots_adjust(left=0.085, right=0.94, top=0.90, bottom=0.14)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "panel_a_str_density"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("Window size must be positive")
    if args.font_scale <= 0:
        raise ValueError("Font scale must be positive")
    if args.figure_only:
        source = args.output_dir / "source_data/panel_a_1mb_windows.tsv"
        if not source.is_file():
            raise FileNotFoundError(source)
        windows = pd.read_csv(source, sep="\t", dtype={"chrom": "string"})
        plot_panel(windows, args.output_dir, args.font_scale)
        print("Regenerated Panel A from its existing source table")
        return

    for path in (args.annotation, args.gff3, args.variance):
        if not path.is_file():
            raise FileNotFoundError(path)

    lengths = chromosome_lengths(args.gff3)
    loci, total_polymorphic, scaffold_polymorphic = load_loci(args)
    windows = make_windows(loci, lengths, args.window_size)
    chromosome_polymorphic = int(windows["polymorphic_count"].sum())
    if chromosome_polymorphic + scaffold_polymorphic != total_polymorphic:
        raise ValueError("Chromosome and scaffold counts do not sum to the polymorphic total")

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(source_dir / "panel_a_1mb_windows.tsv", sep="\t", index=False, na_rep="NA")
    summary = pd.DataFrame(
        [
            ("polymorphic_loci_total", total_polymorphic),
            ("polymorphic_loci_chromosomes_1_23", chromosome_polymorphic),
            ("polymorphic_loci_unplaced_scaffolds", scaffold_polymorphic),
            ("windows_total", len(windows)),
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(source_dir / "panel_a_summary.tsv", sep="\t", index=False)
    plot_panel(windows, args.output_dir, args.font_scale)
    print(f"Plotted {chromosome_polymorphic:,} chromosome STRs in {len(windows)} windows")
    print(f"Reported {scaffold_polymorphic:,} polymorphic STRs on unplaced scaffolds")


if __name__ == "__main__":
    main()
