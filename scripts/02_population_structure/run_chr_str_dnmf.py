#!/usr/bin/env python3
"""Select dNMF K using chromosome STRs with variance strictly above 0.1."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PCA_DIR = SCRIPT_DIR.parent
NATURE_DOUBLE_COLUMN_IN = 183 / 25.4
FONT_FAMILY = "Nimbus Sans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--str-matrix", type=Path,
        default=PROJECT_ROOT / "olive_str/str_data/str_var_0.1.csv",
    )
    parser.add_argument(
        "--locus-set", type=Path, default=PCA_DIR / "source_data/str_locus_sets.tsv"
    )
    parser.add_argument(
        "--sample-keep", type=Path, default=PCA_DIR / "source_data/shared_samples.keep"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "chr_str_var_gt_0.1"
    )
    parser.add_argument("--minimum-k", type=int, default=2)
    parser.add_argument("--maximum-k", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--matching-threshold", type=float, default=0.9)
    parser.add_argument("--variance-threshold", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    parser.add_argument("--expected-samples", type=int, default=212)
    parser.add_argument("--expected-chromosome-loci-before-variance", type=int, default=45_666)
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def prepare_input(args: argparse.Namespace, source_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    keep = pd.read_csv(
        args.sample_keep, sep=r"\s+", header=None, names=["FID", "sample_id"], dtype=str
    )
    if len(keep) != args.expected_samples or keep["sample_id"].duplicated().any():
        raise ValueError(f"Expected {args.expected_samples} unique focal samples")

    locus_set = pd.read_csv(args.locus_set, sep="\t", dtype={"str_id": str})
    required = {"str_id", "contig", "included_chromosome_sensitivity"}
    if not required.issubset(locus_set.columns):
        raise ValueError(f"Locus set lacks columns: {sorted(required - set(locus_set.columns))}")
    chromosome_loci = locus_set.loc[
        locus_set["included_chromosome_sensitivity"], "str_id"
    ].tolist()
    if len(chromosome_loci) != args.expected_chromosome_loci_before_variance:
        raise ValueError(
            f"Expected {args.expected_chromosome_loci_before_variance} chromosome loci, "
            f"found {len(chromosome_loci)}"
        )

    data = pd.read_csv(args.str_matrix, index_col=0, low_memory=False)
    data.index = data.index.astype(str)
    if data.index.duplicated().any():
        raise ValueError("STR matrix contains duplicate locus IDs")
    missing_loci = sorted(set(chromosome_loci) - set(data.index))
    missing_samples = sorted(set(keep["sample_id"]) - set(data.columns))
    if missing_loci or missing_samples:
        raise ValueError(
            f"Input mismatch: {len(missing_loci)} missing loci, {len(missing_samples)} missing samples"
        )
    data = data.loc[chromosome_loci, keep["sample_id"]]

    # The source variance filter used population variance (ddof=0); apply the requested strict cutoff.
    variances = data.var(axis=1, skipna=True, ddof=0)
    retained = variances > args.variance_threshold
    excluded = pd.DataFrame(
        {"str_id": data.index[~retained], "variance": variances.loc[~retained].to_numpy()}
    )
    excluded.to_csv(source_dir / "excluded_variance_boundary_loci.tsv", sep="\t", index=False)
    data = data.loc[retained]
    if data.empty:
        raise ValueError("No STR loci remain after variance filtering")

    data.to_csv(source_dir / "chr_str_var_gt_0.1.csv.gz", compression="gzip")
    pd.DataFrame(
        {
            "str_id": data.index,
            "variance": variances.loc[data.index].to_numpy(),
            "n_called": data.notna().sum(axis=1).to_numpy(),
        }
    ).to_csv(source_dir / "input_locus_qc.tsv.gz", sep="\t", index=False, compression="gzip")

    imputed = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(data.T)
    standardized = StandardScaler().fit_transform(imputed)
    positive = np.clip(standardized, a_min=0, a_max=None)
    negative = np.clip(-standardized, a_min=0, a_max=None)
    return data, positive, negative


def fit_direction(matrix: np.ndarray, k: int, seed: int, max_iterations: int) -> tuple[np.ndarray, float, int]:
    model = NMF(
        n_components=k,
        init="nndsvdar",
        solver="cd",
        max_iter=max_iterations,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        q = model.fit_transform(matrix)
    row_sums = q.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"K={k} produced a zero-sum ancestry row")
    q /= row_sums
    return q, float(model.reconstruction_err_), int(model.n_iter_)


def run_models(args: argparse.Namespace, positive: np.ndarray, negative: np.ndarray) -> pd.DataFrame:
    rows = []
    seed_sequence = np.random.SeedSequence(args.seed)
    repeat_sequences = seed_sequence.spawn(args.repeats)
    for repeat, repeat_sequence in enumerate(repeat_sequences, start=1):
        direction_sequences = repeat_sequence.spawn(2 * (args.maximum_k - args.minimum_k + 1))
        for offset, k in enumerate(range(args.minimum_k, args.maximum_k + 1)):
            pos_seed = int(direction_sequences[2 * offset].generate_state(1)[0])
            neg_seed = int(direction_sequences[2 * offset + 1].generate_state(1)[0])
            q_pos, pos_error, pos_iterations = fit_direction(
                positive, k, pos_seed, args.max_iterations
            )
            q_neg, neg_error, neg_iterations = fit_direction(
                negative, k, neg_seed, args.max_iterations
            )
            correlations = np.corrcoef(q_pos.T, q_neg.T)[:k, k:]
            if not np.isfinite(correlations).all():
                raise ValueError(f"Non-finite component correlation at repeat={repeat}, K={k}")
            row_index, column_index = linear_sum_assignment(-np.abs(correlations))
            matched = correlations[row_index, column_index]
            absolute = np.abs(matched)
            rows.append(
                {
                    "repeat": repeat,
                    "K": k,
                    "positive_seed": pos_seed,
                    "negative_seed": neg_seed,
                    "mean_correlation": matched.mean(),
                    "mean_absolute_correlation": absolute.mean(),
                    "minimum_absolute_correlation": absolute.min(),
                    "matched_components": int((absolute > args.matching_threshold).sum()),
                    "matched_fraction": float((absolute > args.matching_threshold).mean()),
                    "positive_reconstruction_error": pos_error,
                    "negative_reconstruction_error": neg_error,
                    "positive_iterations": pos_iterations,
                    "negative_iterations": neg_iterations,
                    "all_components_converged": (
                        pos_iterations < args.max_iterations and neg_iterations < args.max_iterations
                    ),
                }
            )
            print(
                f"repeat={repeat}/{args.repeats} K={k}: "
                f"matched={rows[-1]['matched_components']}/{k}, "
                f"mean|r|={rows[-1]['mean_absolute_correlation']:.4f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_and_select(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    summary = raw.groupby("K", as_index=False).agg(
        median_matched_components=("matched_components", "median"),
        minimum_matched_components=("matched_components", "min"),
        median_matched_fraction=("matched_fraction", "median"),
        median_mean_absolute_correlation=("mean_absolute_correlation", "median"),
        minimum_mean_absolute_correlation=("mean_absolute_correlation", "min"),
        all_fits_converged=("all_components_converged", "all"),
    )
    summary["all_repeats_fully_matched"] = (
        summary["minimum_matched_components"] == summary["K"]
    )
    stable = summary[summary["all_repeats_fully_matched"]]
    if not stable.empty:
        best_k = int(stable["K"].max())
    else:
        ranked = summary.sort_values(
            ["median_matched_fraction", "minimum_mean_absolute_correlation", "K"],
            ascending=[False, False, False], kind="stable",
        )
        best_k = int(ranked.iloc[0]["K"])
    return summary, best_k


def refit_best_k(
    args: argparse.Namespace, raw: pd.DataFrame, best_k: int,
    positive: np.ndarray, negative: np.ndarray, samples: list[str], source_dir: Path,
) -> None:
    candidates = raw[raw["K"] == best_k].sort_values(
        ["matched_components", "mean_absolute_correlation"], ascending=False
    )
    chosen = candidates.iloc[0]
    q_pos, _, _ = fit_direction(
        positive, best_k, int(chosen["positive_seed"]), args.max_iterations
    )
    q_neg, _, _ = fit_direction(
        negative, best_k, int(chosen["negative_seed"]), args.max_iterations
    )
    correlations = np.corrcoef(q_pos.T, q_neg.T)[:best_k, best_k:]
    rows, columns = linear_sum_assignment(-np.abs(correlations))
    matched_columns = columns[np.argsort(rows)]
    q_neg = q_neg[:, matched_columns]
    q_columns = [f"Q{index}" for index in range(1, best_k + 1)]
    for direction, coefficients in (("pos", q_pos), ("neg", q_neg)):
        output = pd.DataFrame(coefficients, columns=q_columns)
        output.insert(0, "sample_id", samples)
        output.to_csv(
            source_dir / f"best_k{best_k}_{direction}.tsv", sep="\t", index=False
        )
    pd.DataFrame(
        {
            "positive_component": q_columns,
            "negative_original_component": [
                f"Q{index + 1}" for index in matched_columns
            ],
            "pearson_r": correlations[np.arange(best_k), matched_columns],
            "absolute_pearson_r": np.abs(
                correlations[np.arange(best_k), matched_columns]
            ),
        }
    ).to_csv(
        source_dir / f"best_k{best_k}_component_matching.tsv",
        sep="\t",
        index=False,
    )


def plot_selection(raw: pd.DataFrame, summary: pd.DataFrame, best_k: int, figure_dir: Path) -> None:
    set_nature_style()
    fig, axes = plt.subplots(2, 1, figsize=(NATURE_DOUBLE_COLUMN_IN * 0.62, 4.2), sharex=True)
    sns.lineplot(
        data=raw, x="K", y="matched_components", units="repeat", estimator=None,
        color="#B3B3B3", lw=0.65, alpha=0.65, marker="o", ms=2.5, ax=axes[0],
    )
    axes[0].plot(
        summary["K"], summary["median_matched_components"], color="#0072B2",
        marker="o", ms=4, lw=1.2, label="Median across repeats",
    )
    axes[0].plot(summary["K"], summary["K"], ls="--", color="#777777", lw=0.7,
                 label="All K components matched")
    axes[0].axvline(best_k, color="#D55E00", ls=":", lw=1.0)
    axes[0].set_ylabel("Matched components")
    axes[0].legend(frameon=False, loc="upper left")

    sns.lineplot(
        data=raw, x="K", y="mean_absolute_correlation", units="repeat", estimator=None,
        color="#B3B3B3", lw=0.65, alpha=0.65, marker="o", ms=2.5, ax=axes[1],
    )
    axes[1].plot(
        summary["K"], summary["median_mean_absolute_correlation"],
        color="#0072B2", marker="o", ms=4, lw=1.2,
    )
    axes[1].axvline(best_k, color="#D55E00", ls=":", lw=1.0)
    axes[1].text(
        best_k + 0.2, axes[1].get_ylim()[0] + 0.03, f"Best K = {best_k}",
        color="#D55E00", va="bottom",
    )
    axes[1].set(xlabel="Number of ancestral components (K)", ylabel="Mean matched |correlation|")
    axes[1].set_xticks(summary["K"])
    for label, axis in zip("ab", axes):
        axis.text(-0.13, 1.05, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
        sns.despine(ax=axis)
    fig.tight_layout(h_pad=1.0)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"dnmf_k_selection.{extension}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    args: argparse.Namespace, data: pd.DataFrame, excluded: pd.DataFrame,
    raw: pd.DataFrame, summary: pd.DataFrame, best_k: int,
) -> None:
    best = summary[summary["K"] == best_k].iloc[0]
    nonconverged = int((~raw["all_components_converged"]).sum())
    text = f"""# dNMF K selection from chromosome STRs

## Input

The analysis used {len(data):,} chromosome-assigned STRs with population variance strictly greater than {args.variance_threshold} across the same {data.shape[1]} samples used in the SNP-STR population-structure analyses. {len(excluded):,} chromosome locus at the exact variance boundary was excluded. Missing genotypes were imputed with the locus mean, loci were standardized, and standardized values were separated into positive and negative matrices following the supplied `dNMF_K.py` method.

## Model selection

K = {args.minimum_k}-{args.maximum_k} was evaluated using {args.repeats} genuinely independent `nndsvdar` initializations per K. Positive and negative components were paired by maximum absolute Pearson correlation using the Hungarian algorithm. A component pair was considered matched when |r| > {args.matching_threshold}. The selected rank is the largest K for which every repeat matched all K components. This selects **K = {best_k}**, with {best['minimum_matched_components']:.0f}/{best_k} components matched in the least concordant repeat, median matched fraction {best['median_matched_fraction']:.3f}, and median mean |r| {best['median_mean_absolute_correlation']:.4f}.

The original script used deterministic `nndsvd` initialization while varying a nominal random seed and ran five internal repeats for every repeat-index invocation. This analysis uses `nndsvdar`, which makes the specified seeds effective, and records every fit explicitly. Fits reaching the iteration limit: {nonconverged}/{len(raw)}.

## Outputs

- `figures/dnmf_k_selection.*`: model-selection diagnostics
- `source_data/dnmf_repeat_metrics.tsv`: every repeat-by-K result
- `source_data/dnmf_k_summary.tsv`: aggregated selection metrics
- `source_data/best_k{best_k}_pos.tsv`: positive-direction ancestry coefficients from the strongest best-K repeat
- `source_data/best_k{best_k}_neg.tsv`: matched negative-direction ancestry coefficients from the strongest best-K repeat
- `source_data/best_k{best_k}_component_matching.tsv`: positive-to-negative component matching
- `source_data/chr_str_var_gt_0.1.csv.gz`: exact dNMF input matrix
"""
    (args.output_dir / "analysis_report.md").write_text(text)


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    figure_dir = args.output_dir / "figures"
    for directory in (args.output_dir, source_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data, positive, negative = prepare_input(args, source_dir)
    raw = run_models(args, positive, negative)
    raw.to_csv(source_dir / "dnmf_repeat_metrics.tsv", sep="\t", index=False)
    summary, best_k = summarize_and_select(raw)
    summary.to_csv(source_dir / "dnmf_k_summary.tsv", sep="\t", index=False)
    refit_best_k(args, raw, best_k, positive, negative, data.columns.tolist(), source_dir)
    plot_selection(raw, summary, best_k, figure_dir)
    excluded = pd.read_csv(source_dir / "excluded_variance_boundary_loci.tsv", sep="\t")
    write_report(args, data, excluded, raw, summary, best_k)
    print(f"Best K: {best_k}")


if __name__ == "__main__":
    main()
