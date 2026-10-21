#!/usr/bin/env python3
"""Pool matched original components and correlate SNP with STR ancestry proportions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[1] / "scripts/02_population_structure"
FONT_FAMILY = "Nimbus Sans"
SUBSPECIES_COLORS = {"europaea": "#4D4D4D", "sylvestris": "#56B4E9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-components", type=Path,
        default=ANALYSIS_DIR / "source_data/sample_component_proportions.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--resamples", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20_260_714)
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


def fast_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denominator == 0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denominator)


def load_arrays(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    data = pd.read_csv(args.sample_components, sep="\t")
    required = {
        "sample_id", "subspecies", "Country", "component_pair",
        "snp_ancestry_proportion", "str_ancestry_proportion",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"Input lacks columns: {sorted(required - set(data.columns))}")
    if data["sample_id"].nunique() != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} unique samples")
    pairs = data["component_pair"].drop_duplicates().tolist()
    if len(pairs) != 6 or len(data) != args.expected_samples * len(pairs):
        raise ValueError("Expected six observations per sample")
    snp = data.pivot(index="sample_id", columns="component_pair", values="snp_ancestry_proportion")
    str_q = data.pivot(index="sample_id", columns="component_pair", values="str_ancestry_proportion")
    sample_order = sorted(data["sample_id"].unique())
    snp = snp.loc[sample_order, pairs]
    str_q = str_q.loc[sample_order, pairs]
    if snp.isna().any().any() or str_q.isna().any().any():
        raise ValueError("Incomplete sample-component matrix")
    return data, snp.to_numpy(), str_q.to_numpy(), sample_order


def block_resampling(
    snp: np.ndarray, str_q: np.ndarray, resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = len(snp)
    bootstrap = np.empty(resamples)
    permutation = np.empty(resamples)
    for index in range(resamples):
        sampled = rng.integers(0, n_samples, size=n_samples)
        bootstrap[index] = fast_pearson(snp[sampled].ravel(), str_q[sampled].ravel())
        permutation[index] = fast_pearson(snp.ravel(), str_q[rng.permutation(n_samples)].ravel())
    return bootstrap, permutation


def summarize(
    data: pd.DataFrame,
    snp: np.ndarray,
    str_q: np.ndarray,
    bootstrap: np.ndarray,
    permutation: np.ndarray,
) -> pd.DataFrame:
    observed_r = fast_pearson(snp.ravel(), str_q.ravel())
    observed_rho = spearmanr(snp.ravel(), str_q.ravel()).statistic
    p_value = (int((permutation >= observed_r).sum()) + 1) / (len(permutation) + 1)
    rows = [
        {
            "cohort": "all_samples",
            "n_samples": data["sample_id"].nunique(),
            "n_pooled_observations": len(data),
            "pearson_r": observed_r,
            "pearson_block_bootstrap_ci_low": np.quantile(bootstrap, 0.025),
            "pearson_block_bootstrap_ci_high": np.quantile(bootstrap, 0.975),
            "spearman_rho": observed_rho,
            "sample_block_permutation_p": p_value,
        }
    ]
    for subspecies, subset in data.groupby("subspecies", sort=False):
        rows.append(
            {
                "cohort": subspecies,
                "n_samples": subset["sample_id"].nunique(),
                "n_pooled_observations": len(subset),
                "pearson_r": pearsonr(
                    subset["snp_ancestry_proportion"], subset["str_ancestry_proportion"]
                ).statistic,
                "pearson_block_bootstrap_ci_low": np.nan,
                "pearson_block_bootstrap_ci_high": np.nan,
                "spearman_rho": spearmanr(
                    subset["snp_ancestry_proportion"], subset["str_ancestry_proportion"]
                ).statistic,
                "sample_block_permutation_p": np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot(data: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.8, 3.55))
    for subspecies in ("europaea", "sylvestris"):
        subset = data[data["subspecies"] == subspecies]
        ax.scatter(
            subset["snp_ancestry_proportion"], subset["str_ancestry_proportion"],
            s=11 if subspecies == "europaea" else 20,
            color=SUBSPECIES_COLORS[subspecies],
            alpha=0.42 if subspecies == "europaea" else 0.85,
            edgecolor="white", linewidth=0.2, zorder=2,
        )
    sns.regplot(
        data=data, x="snp_ancestry_proportion", y="str_ancestry_proportion",
        scatter=False, ci=None, ax=ax, line_kws={"color": "#222222", "lw": 1.0},
    )
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--", zorder=1)
    all_result = summary.set_index("cohort").loc["all_samples"]
    europaea_result = summary.set_index("cohort").loc["europaea"]
    ax.text(
        0.04, 0.96,
        f"Pearson $r$ = {all_result['pearson_r']:.3f}\n"
        f"95% CI {all_result['pearson_block_bootstrap_ci_low']:.3f}–"
        f"{all_result['pearson_block_bootstrap_ci_high']:.3f}\n"
        f"Spearman $\\rho$ = {all_result['spearman_rho']:.3f}\n"
        f"europaea $r$ = {europaea_result['pearson_r']:.3f}",
        transform=ax.transAxes, ha="left", va="top",
    )
    ax.set(
        xlabel="SNP ancestry proportion",
        ylabel="STR ancestry proportion",
        xlim=(-0.025, 1.025), ylim=(-0.025, 1.025),
    )
    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=SUBSPECIES_COLORS[label],
               label=label, markersize=4.5)
        for label in ("europaea", "sylvestris")
    ]
    ax.legend(
        handles=handles, title="Subspecies", frameon=False, ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"pooled_snp_str_ancestry_correlation.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data, snp, str_q, _ = load_arrays(args)
    bootstrap, permutation = block_resampling(snp, str_q, args.resamples, args.seed)
    summary = summarize(data, snp, str_q, bootstrap, permutation)
    summary.to_csv(source_dir / "pooled_ancestry_correlation_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"bootstrap_pearson_r": bootstrap, "permuted_pearson_r": permutation}
    ).to_csv(
        source_dir / "pooled_ancestry_correlation_resampling.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )
    plot(data, summary, figure_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
