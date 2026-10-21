#!/usr/bin/env python3
"""Plot the correlation between oleic- and linoleic-acid percentages by source."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, t


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/06_trait_association"
OLEIC = "OleicAcid_Percentage"
LINOLEIC = "LinoleicAcid_Percentage"
SOURCE_ORDER = [
    "Iranian cohort",
    "PRJNA556567",
    "PRJEB35540",
    "PRJNA782823",
]
PALETTE = {
    "Iranian cohort": "#3B6FB6",
    "PRJNA556567": "#D55E00",
    "PRJEB35540": "#009E73",
    "PRJNA782823": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotypes",
        type=Path,
        default=ANALYSIS_DIR / "inputs/phenotypes.tsv",
    )
    parser.add_argument(
        "--sample-sources",
        type=Path,
        default=ANALYSIS_DIR / "inputs/Supplementary_Data_S1.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    return parser.parse_args()


def format_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def center_adjusted_correlation(data: pd.DataFrame) -> tuple[float, float]:
    center = pd.get_dummies(data["center"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(data)), center])
    x = data[OLEIC].to_numpy(dtype=float)
    y = data[LINOLEIC].to_numpy(dtype=float)
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = float(np.corrcoef(x_resid, y_resid)[0, 1])
    n_adjusters = design.shape[1] - 1
    degrees_freedom = len(data) - n_adjusters - 2
    statistic = r * np.sqrt(degrees_freedom / (1 - r**2))
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
    return r, p_value


def calculate_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pearson = pearsonr(data[OLEIC], data[LINOLEIC])
    spearman = spearmanr(data[OLEIC], data[LINOLEIC])
    adjusted_r, adjusted_p = center_adjusted_correlation(data)
    rows.extend(
        [
            {
                "scope": "all_samples",
                "method": "Pearson",
                "n": len(data),
                "correlation": pearson.statistic,
                "p_value": pearson.pvalue,
            },
            {
                "scope": "all_samples",
                "method": "Spearman",
                "n": len(data),
                "correlation": spearman.statistic,
                "p_value": spearman.pvalue,
            },
            {
                "scope": "center_adjusted",
                "method": "partial_Pearson",
                "n": len(data),
                "correlation": adjusted_r,
                "p_value": adjusted_p,
            },
        ]
    )
    for center, group in data.groupby("center", sort=True):
        for method, result in (
            ("Pearson", pearsonr(group[OLEIC], group[LINOLEIC])),
            ("Spearman", spearmanr(group[OLEIC], group[LINOLEIC])),
        ):
            rows.append(
                {
                    "scope": center,
                    "method": method,
                    "n": len(group),
                    "correlation": result.statistic,
                    "p_value": result.pvalue,
                }
            )
    return pd.DataFrame(rows)


def add_sample_sources(data: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    sources = pd.read_csv(source_path, dtype=str)
    required = {"analysis_sample_id", "BioProject"}
    missing = required - set(sources.columns)
    if missing:
        raise ValueError(f"Sample-source table lacks columns: {sorted(missing)}")
    if sources["analysis_sample_id"].duplicated().any():
        duplicates = sources.loc[
            sources["analysis_sample_id"].duplicated(keep=False), "analysis_sample_id"
        ].tolist()
        raise ValueError(f"Duplicate sample IDs in source table: {duplicates}")

    source_map = sources.set_index("analysis_sample_id")["BioProject"]
    data = data.copy()
    data["source"] = data["Sample_ID"].map(source_map)
    data["source"] = data["source"].where(
        data["source"].isin(SOURCE_ORDER[1:]), "Iranian cohort"
    )
    observed = set(data["source"])
    unexpected = observed - set(SOURCE_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected sample sources: {sorted(unexpected)}")
    return data


def make_figure(data: pd.DataFrame, statistics: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_theme(style="ticks", context="notebook", font_scale=1.4)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )
    fig, ax = plt.subplots(figsize=(7.3, 5.8))
    sns.scatterplot(
        data=data,
        x=OLEIC,
        y=LINOLEIC,
        hue="source",
        hue_order=SOURCE_ORDER,
        palette=PALETTE,
        s=55,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    sns.regplot(
        data=data,
        x=OLEIC,
        y=LINOLEIC,
        scatter=False,
        ci=95,
        color="#333333",
        line_kws={"linewidth": 1.8},
        ax=ax,
    )

    overall = statistics.query("scope == 'all_samples' and method == 'Pearson'").iloc[0]
    annotation = f"Pearson $r$ = {overall.correlation:.3f}, $P$ = {format_p(overall.p_value)}"
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    ax.set(xlabel="Oleic acid (%)", ylabel="Linoleic acid (%)")
    ax.legend(
        title="Source",
        loc="upper right",
        frameon=False,
        fontsize=10.5,
        title_fontsize=12,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"oleic_linoleic_correlation.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.phenotypes, sep="\t")
    required = {"Sample_ID", OLEIC, LINOLEIC, "center"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Phenotype table lacks columns: {sorted(missing)}")
    data = data.dropna(subset=[OLEIC, LINOLEIC, "center"]).copy()
    if not len(data):
        raise ValueError("No samples have both phenotypes")
    data = add_sample_sources(data, args.sample_sources)

    figure_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    statistics = calculate_statistics(data)
    statistics.to_csv(
        source_dir / "oleic_linoleic_correlation.tsv", sep="\t", index=False
    )
    make_figure(data, statistics, figure_dir)
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
