#!/usr/bin/env python3
"""Test motif-specific pSTR enrichment across olive genomic contexts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "inputs"
MOTIF_ORDER = range(1, 7)
CONTEXT_ORDER = [
    "CDS",
    "five_prime_UTR",
    "three_prime_UTR",
    "intron",
    "promoter",
    "intergenic",
]
CONTEXT_LABELS = {
    "CDS": "CDS",
    "five_prime_UTR": "5' UTR",
    "three_prime_UTR": "3' UTR",
    "intron": "Intron",
    "promoter": "Promoter",
    "intergenic": "Intergenic",
}
NATURE_SINGLE_COLUMN_IN = 89 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=INPUT_DIR / "O_europeana_strs.functional.promoter2kb.annotated.bed",
    )
    parser.add_argument(
        "--pstrs",
        type=Path,
        default=INPUT_DIR / "str_length_variance.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--font-scale", type=float, default=1.3)
    return parser.parse_args()


def load_targets(annotation_path: Path, pstr_path: Path) -> pd.DataFrame:
    targets = pd.read_csv(
        annotation_path,
        sep="\t",
        usecols=["#chrom", "start", "motif_length", "functional_region"],
        dtype={"#chrom": "string", "start": "int64", "motif_length": "int64"},
    ).rename(columns={"#chrom": "chrom"})
    targets["str_id"] = targets["chrom"] + "_" + targets["start"].astype(str)
    if targets["str_id"].duplicated().any():
        raise ValueError("Duplicate STR identifiers in annotation")
    pstrs = pd.read_csv(pstr_path, usecols=["str_id"], dtype="string")
    if pstrs["str_id"].duplicated().any():
        raise ValueError("Duplicate pSTR identifiers")
    pstr_ids = set(pstrs["str_id"])
    missing = pstr_ids - set(targets["str_id"])
    if missing:
        raise ValueError(f"{len(missing):,} pSTRs lack annotation")
    targets["is_pstr"] = targets["str_id"].isin(pstr_ids).astype(int)
    return targets


def adjust_bh(pvalues: pd.Series) -> np.ndarray:
    values = pvalues.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def calculate_enrichment(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for motif_length in MOTIF_ORDER:
        motif_data = targets[targets["motif_length"].eq(motif_length)]
        motif_total = len(motif_data)
        motif_pstr = int(motif_data["is_pstr"].sum())
        motif_baseline = motif_pstr / motif_total
        for context in CONTEXT_ORDER:
            in_context = motif_data["functional_region"].eq(context)
            context_total = int(in_context.sum())
            context_pstr = int(motif_data.loc[in_context, "is_pstr"].sum())
            outside_total = motif_total - context_total
            outside_pstr = motif_pstr - context_pstr
            table = [
                [context_pstr, context_total - context_pstr],
                [outside_pstr, outside_total - outside_pstr],
            ]
            fisher = fisher_exact(table)
            context_frequency = context_pstr / context_total
            enrichment = context_frequency / motif_baseline
            rows.append((
                motif_length, context, context_total, context_pstr,
                100 * context_frequency, 100 * motif_baseline, enrichment,
                np.log2(enrichment), fisher.statistic, fisher.pvalue,
            ))
    result = pd.DataFrame(rows, columns=[
        "motif_length", "functional_region", "all_str_count", "pstr_count",
        "pstr_percent", "motif_baseline_percent", "fold_enrichment",
        "log2_fold_enrichment", "odds_ratio_vs_other_contexts", "fisher_pvalue",
    ])
    result["fdr_bh"] = adjust_bh(result["fisher_pvalue"])
    return result


def set_style(font_scale: float) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6 * font_scale,
        "axes.titlesize": 7 * font_scale,
        "axes.labelsize": 6 * font_scale,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 5.5 * font_scale,
        "ytick.labelsize": 5.5 * font_scale,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def plot_heatmap(enrichment: pd.DataFrame, output_dir: Path, font_scale: float) -> None:
    set_style(font_scale)
    values = enrichment.pivot(index="motif_length", columns="functional_region", values="log2_fold_enrichment")
    values = values.reindex(index=MOTIF_ORDER, columns=CONTEXT_ORDER)
    fold = enrichment.pivot(index="motif_length", columns="functional_region", values="fold_enrichment").reindex_like(values)
    fdr = enrichment.pivot(index="motif_length", columns="functional_region", values="fdr_bh").reindex_like(values)
    annotations = fold.copy().astype(object)
    for motif_length in MOTIF_ORDER:
        for context in CONTEXT_ORDER:
            star = "*" if fdr.loc[motif_length, context] < 0.05 else ""
            annotations.loc[motif_length, context] = f"{fold.loc[motif_length, context]:.2f}{star}"

    limit = float(np.nanmax(np.abs(values.to_numpy())))
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COLUMN_IN, 3.15))
    sns.heatmap(
        values,
        ax=ax,
        cmap=sns.diverging_palette(240, 15, s=80, l=50, as_cmap=True),
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 5.2 * font_scale},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "log2 fold enrichment", "shrink": 0.72, "pad": 0.03},
    )
    ax.set_xticklabels([CONTEXT_LABELS[value] for value in CONTEXT_ORDER], rotation=35, ha="right")
    ax.set_yticklabels([str(value) for value in MOTIF_ORDER], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("Motif length (bp)")
    ax.set_title("Motif-specific pSTR enrichment", loc="left", fontweight="bold")
    ax.text(
        0,
        -0.31,
        "*Benjamini-Hochberg FDR < 0.05",
        transform=ax.transAxes,
        fontsize=5.2 * font_scale,
        ha="left",
    )
    fig.subplots_adjust(left=0.15, right=0.91, top=0.90, bottom=0.29)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "motif_by_genomic_context_enrichment"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def write_report(output: Path, targets: pd.DataFrame, enrichment: pd.DataFrame) -> None:
    strongest = enrichment.loc[enrichment["fold_enrichment"].idxmax()]
    weakest = enrichment.loc[enrichment["fold_enrichment"].idxmin()]
    output.write_text(f"""# Motif-by-Genomic-Context pSTR Enrichment

This analysis compares {int(targets['is_pstr'].sum()):,} pSTRs with all
{len(targets):,} STR targets. Within each motif class and genomic context, pSTR
frequency is divided by the genome-wide pSTR frequency for that motif length.
Values above 1 indicate enrichment and values below 1 indicate depletion.

The strongest enrichment is {strongest['fold_enrichment']:.2f}x for
{int(strongest['motif_length'])}-bp motifs in
{CONTEXT_LABELS[strongest['functional_region']]}. The strongest depletion is
{weakest['fold_enrichment']:.2f}x for {int(weakest['motif_length'])}-bp motifs in
{CONTEXT_LABELS[weakest['functional_region']]}. Fisher tests compare each cell
with all other contexts within the same motif class, followed by
Benjamini-Hochberg correction across {len(enrichment)} tests.
""")


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be greater than zero")
    for path in (args.annotation, args.pstrs):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.annotation, args.pstrs)
    enrichment = calculate_enrichment(targets)
    enrichment.to_csv(source_dir / "motif_by_genomic_context_enrichment.tsv", sep="\t", index=False)
    plot_heatmap(enrichment, args.output_dir, args.font_scale)
    write_report(args.output_dir / "analysis_report.md", targets, enrichment)
    print(f"Tested {len(enrichment)} motif-context combinations")


if __name__ == "__main__":
    main()
