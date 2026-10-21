#!/usr/bin/env python3
"""Select the best ADMIXTURE K and plot ancestry for the 212-sample cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PCA_DIR = SCRIPT_DIR.parent
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"
ANCESTRY_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E6AB02", "#6A3D9A"]
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
        "--adm-results", type=Path, default=PROJECT_ROOT / "adm_results"
    )
    parser.add_argument(
        "--fam", type=Path, default=PROJECT_ROOT / "raw_snp/olive.admix.fam"
    )
    parser.add_argument(
        "--sample-keep", type=Path, default=PCA_DIR / "source_data/shared_samples.keep"
    )
    parser.add_argument(
        "--metadata", type=Path, default=PROJECT_ROOT / "olive_str/metadata/Geo_loc.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
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


def normalize_subspecies(value: str) -> str:
    value = value.strip().lower()
    if value.endswith("sylvestris"):
        return "sylvestris"
    if value.endswith("europaea"):
        return "europaea"
    raise ValueError(f"Unexpected subspecies label in selected cohort: {value!r}")


def select_best_k(path: Path) -> tuple[pd.DataFrame, int]:
    cv = pd.read_csv(path, sep="\t")
    if not {"K", "cv_error"}.issubset(cv.columns) or cv.empty:
        raise ValueError("CV table must contain K and cv_error columns")
    cv["K"] = pd.to_numeric(cv["K"], errors="raise").astype(int)
    cv["cv_error"] = pd.to_numeric(cv["cv_error"], errors="raise")
    if cv["K"].duplicated().any() or cv["cv_error"].isna().any():
        raise ValueError("CV table contains duplicate K or missing errors")
    best_k = int(cv.loc[cv["cv_error"].idxmin(), "K"])
    return cv.sort_values("K").reset_index(drop=True), best_k


def load_ancestry(args: argparse.Namespace, best_k: int) -> pd.DataFrame:
    fam = pd.read_csv(
        args.fam, sep=r"\s+", header=None,
        names=["FID", "sample_id", "father", "mother", "sex", "phenotype"], dtype=str,
    )
    q_path = args.adm_results / f"olive.admix.{best_k}.Q"
    q = pd.read_csv(q_path, sep=r"\s+", header=None)
    if len(q) != len(fam) or q.shape[1] != best_k:
        raise ValueError(
            f"K={best_k} Q dimensions {q.shape} do not match {len(fam)} FAM samples"
        )
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Q ancestry proportions do not sum to one")
    q.columns = [f"ancestry_{index}" for index in range(1, best_k + 1)]
    ancestry = pd.concat([fam[["FID", "sample_id"]], q], axis=1)

    keep = pd.read_csv(
        args.sample_keep, sep=r"\s+", header=None, names=["keep_fid", "sample_id"], dtype=str
    )
    if len(keep) != args.expected_samples or keep["sample_id"].duplicated().any():
        raise ValueError(f"Expected {args.expected_samples} unique selected samples")
    ancestry = keep.merge(ancestry, on="sample_id", how="left", validate="one_to_one")
    if ancestry.filter(like="ancestry_").isna().any().any():
        raise ValueError("At least one selected sample is absent from the ADMIXTURE Q matrix")

    metadata = pd.read_csv(args.metadata, encoding="latin-1", dtype=str).fillna("")
    required = {"ID", "Country", "Subspecies"}
    if not required.issubset(metadata.columns) or metadata["ID"].duplicated().any():
        raise ValueError("Metadata lacks unique ID, Country, and Subspecies columns")
    ancestry = ancestry.merge(
        metadata[["ID", "Country", "Subspecies"]],
        left_on="sample_id", right_on="ID", how="left", validate="one_to_one",
    )
    if ancestry[["Country", "Subspecies"]].isna().any().any():
        raise ValueError("Selected samples lack metadata")
    ancestry["subspecies"] = ancestry["Subspecies"].map(normalize_subspecies)
    return ancestry


def order_components_and_samples(data: pd.DataFrame, best_k: int) -> tuple[pd.DataFrame, list[str]]:
    components = [f"ancestry_{index}" for index in range(1, best_k + 1)]
    group_means = data.groupby(["subspecies", "Country"])[components].mean()
    component_order = sorted(
        components,
        key=lambda component: (
            int(group_means[component].idxmax()[0] == "sylvestris"),
            group_means[component].idxmax()[1],
            component,
        ),
    )

    ordered_blocks = []
    for subspecies in ("europaea", "sylvestris"):
        subset = data[data["subspecies"] == subspecies]
        for country in sorted(subset["Country"].unique()):
            block = subset[subset["Country"] == country].copy()
            values = block[component_order].to_numpy()
            block["dominant_component"] = values.argmax(axis=1)
            block["dominant_proportion"] = values.max(axis=1)
            block = block.sort_values(
                ["dominant_component", "dominant_proportion", "sample_id"],
                ascending=[True, False, True], kind="stable",
            )
            ordered_blocks.append(block)
    ordered = pd.concat(ordered_blocks, ignore_index=True)
    ordered["plot_position"] = np.arange(len(ordered))
    return ordered, component_order


def plot_cv(cv: pd.DataFrame, best_k: int, figure_dir: Path) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.plot(cv["K"], cv["cv_error"], color="#4D4D4D", lw=1.1, marker="o", ms=3.5)
    best = cv[cv["K"] == best_k].iloc[0]
    ax.scatter(best["K"], best["cv_error"], s=34, color="#D55E00", zorder=3)
    ax.annotate(
        f"Best K = {best_k}\nCV error = {best['cv_error']:.5f}",
        xy=(best["K"], best["cv_error"]), xytext=(best["K"] + 1.0, best["cv_error"] + 0.006),
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7}, va="bottom",
    )
    ax.set(xlabel="Number of ancestral populations (K)", ylabel="Cross-validation error")
    ax.set_xticks(cv["K"])
    sns.despine(ax=ax)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"admixture_cross_validation.{extension}", bbox_inches="tight")
    plt.close(fig)


def contiguous_blocks(data: pd.DataFrame, column: str) -> list[tuple[str, int, int]]:
    values = data[column].tolist()
    blocks = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            blocks.append((values[start], start, index))
            start = index
    return blocks


def plot_admixture(
    data: pd.DataFrame, component_order: list[str], best_k: int, figure_dir: Path
) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COLUMN_IN, 3.35))
    x = data["plot_position"].to_numpy()
    bottom = np.zeros(len(data))
    for color, component in zip(ANCESTRY_COLORS, component_order):
        values = data[component].to_numpy()
        ax.bar(x, values, bottom=bottom, width=1.0, color=color, linewidth=0)
        bottom += values

    country_blocks = contiguous_blocks(data, "Country")
    for country, start, end in country_blocks:
        if start > 0:
            ax.axvline(start - 0.5, color="white", lw=0.6, zorder=5)
        ax.text(
            (start + end - 1) / 2, -0.025, COUNTRY_CODES.get(country, country), ha="right", va="center",
            rotation=90, rotation_mode="anchor", fontsize=5.5,
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
    ax.set_ylabel("Ancestry proportion")
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(
        handles=[Patch(facecolor=color, label=f"Cluster {index}")
                 for index, color in enumerate(ANCESTRY_COLORS[:best_k], start=1)],
        ncol=best_k, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23),
        columnspacing=1.0, handlelength=1.2,
    )
    sns.despine(ax=ax, bottom=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.83, bottom=0.31)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"admixture_best_k{best_k}.{extension}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    args: argparse.Namespace, cv: pd.DataFrame, best_k: int, data: pd.DataFrame,
    component_order: list[str], excluded: list[str],
) -> None:
    best_error = cv.loc[cv["K"] == best_k, "cv_error"].iloc[0]
    counts = data["subspecies"].value_counts()
    text = f"""# SNP ADMIXTURE analysis

## Design

The optimal number of ancestral populations was selected as the tested K with the minimum ADMIXTURE cross-validation error. The Q matrix was matched to `raw_snp/olive.admix.fam` before sample filtering, preserving the original ADMIXTURE row order. The analysis then retained the same 212 accessions used for the SNP-STR population-structure analyses: {counts['europaea']} *Olea europaea* subsp. *europaea* and {counts['sylvestris']} subsp. *sylvestris*. Samples present in the SNP ADMIXTURE run but absent from this cohort were excluded: {', '.join(excluded)}.

## Best K

The lowest cross-validation error was observed at **K = {best_k}** (CV error = {best_error:.5f}). The complete K = {int(cv['K'].min())}-{int(cv['K'].max())} curve is shown in `figures/admixture_cross_validation.*` and retained in `source_data/cv_errors.tsv`.

## Plot organization

The K = {best_k} ancestry plot is grouped first by subspecies and then alphabetically by country. ISO 3166-1 alpha-3 country codes are printed below the bars to keep small groups legible. Within each country, accessions are ordered by their dominant ancestry component and component proportion. ADMIXTURE component labels are arbitrary; the plotted cluster order ({', '.join(component_order)}) was chosen only to provide a stable visual sequence and does not imply evolutionary chronology.

## Outputs

- `figures/admixture_best_k{best_k}.*`: 212-sample ancestry plot
- `figures/admixture_cross_validation.*`: K-selection curve
- `source_data/admixture_k{best_k}_plot_data.tsv`: plotted Q proportions, metadata, and order
- `source_data/sample_inclusion.tsv`: ADMIXTURE sample inclusion status
"""
    (args.output_dir / "analysis_report.md").write_text(text)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    for directory in (args.output_dir, source_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cv, best_k = select_best_k(args.adm_results / "logs/cv_errors.tsv")
    cv.to_csv(source_dir / "cv_errors.tsv", sep="\t", index=False)
    ancestry = load_ancestry(args, best_k)
    ordered, component_order = order_components_and_samples(ancestry, best_k)

    fam = pd.read_csv(args.fam, sep=r"\s+", header=None, usecols=[1], names=["sample_id"])
    selected = set(ancestry["sample_id"])
    inclusion = fam.copy()
    inclusion["included_212_cohort"] = inclusion["sample_id"].isin(selected)
    inclusion.to_csv(source_dir / "sample_inclusion.tsv", sep="\t", index=False)
    excluded = inclusion.loc[~inclusion["included_212_cohort"], "sample_id"].tolist()

    output_columns = [
        "plot_position", "sample_id", "Country", "Subspecies", "subspecies",
        *component_order, "dominant_component", "dominant_proportion",
    ]
    ordered[output_columns].to_csv(
        source_dir / f"admixture_k{best_k}_plot_data.tsv", sep="\t", index=False
    )
    plot_cv(cv, best_k, figure_dir)
    plot_admixture(ordered, component_order, best_k, figure_dir)
    write_report(args, cv, best_k, ordered, component_order, excluded)
    print(f"Best K: {best_k} (CV error {cv.loc[cv['K'] == best_k, 'cv_error'].iloc[0]:.5f})")
    print(f"Plotted samples: {len(ordered)}; excluded from Q matrix: {', '.join(excluded)}")


if __name__ == "__main__":
    main()
