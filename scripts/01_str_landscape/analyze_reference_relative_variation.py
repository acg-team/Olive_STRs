#!/usr/bin/env python3
"""Compare absolute reference-relative pSTR deviation between olive subspecies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "inputs"
GROUP_ORDER = ["europaea", "sylvestris"]
GROUP_LABELS = {"europaea": "subsp. europaea", "sylvestris": "subsp. sylvestris"}
RAW_GROUP_MAP = {
    "Olea europaea subsp. europaea": "europaea",
    "Olea europaea subsp. Sylvestris": "sylvestris",
}
PALETTE = {"europaea": "#287271", "sylvestris": "#D55E00"}
NATURE_SINGLE_COLUMN_IN = 89 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=INPUT_DIR / "merged_str_length.csv",
    )
    parser.add_argument(
        "--str-bed",
        type=Path,
        default=INPUT_DIR / "O_europeana_strs.bed",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=INPUT_DIR / "Geo_loc.csv",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_711)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def load_metadata(path: Path, samples: list[str]) -> pd.DataFrame:
    metadata = pd.read_csv(path, encoding="latin-1", dtype={"ID": "string"})
    if metadata["ID"].duplicated().any():
        raise ValueError("Geo_loc.csv contains duplicate sample IDs")
    metadata = metadata.set_index("ID")
    missing = sorted(set(samples) - set(metadata.index))
    if missing:
        raise ValueError(f"Main-cohort samples missing from Geo_loc.csv: {missing}")
    result = metadata.loc[samples].reset_index().rename(columns={"ID": "sample"})
    result["subspecies"] = result["Subspecies"].map(RAW_GROUP_MAP)
    if result["subspecies"].isna().any():
        unknown = result.loc[result["subspecies"].isna(), "Subspecies"].unique().tolist()
        raise ValueError(f"Unexpected main-cohort subspecies labels: {unknown}")
    return result


def reference_repeat_numbers(path: Path) -> pd.DataFrame:
    bed = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "motif_length", "motif"],
        dtype={"chrom": "string", "start": "int64", "end": "int64", "motif_length": "int64"},
    )
    bed["str_id"] = bed["chrom"] + "_" + bed["start"].astype(str)
    bed["reference_repeat_number"] = (bed["end"] - bed["start"]) / bed["motif_length"]
    if bed["str_id"].duplicated().any():
        raise ValueError("STR BED contains duplicate identifiers")
    if not np.allclose(bed["reference_repeat_number"], np.round(bed["reference_repeat_number"])):
        raise ValueError("BED tract lengths are not exact motif multiples")
    bed["reference_repeat_number"] = bed["reference_repeat_number"].round().astype(int)
    return bed[["str_id", "chrom", "start", "end", "motif_length", "motif", "reference_repeat_number"]]


def prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(args.matrix, dtype={"str_id": "string"})
    if matrix["str_id"].duplicated().any():
        raise ValueError("pSTR matrix contains duplicate identifiers")
    samples = matrix.columns[1:].tolist()
    if not samples:
        raise ValueError("pSTR matrix contains no samples")
    metadata = load_metadata(args.metadata, samples)
    values = matrix[samples].apply(pd.to_numeric, errors="raise")

    loci = matrix[["str_id"]].merge(
        reference_repeat_numbers(args.str_bed), on="str_id", how="left", validate="one_to_one"
    )
    if loci["reference_repeat_number"].isna().any():
        raise ValueError("Some pSTR loci do not map to the STR BED")

    absolute = values.sub(loci["reference_repeat_number"], axis=0).abs()
    absolute = absolute.div(loci["reference_repeat_number"], axis=0) * 100
    absolute.insert(0, "str_id", matrix["str_id"])
    loci["n_called"] = values.notna().sum(axis=1)
    return absolute, loci, metadata


def summarize_samples(
    absolute: pd.DataFrame, metadata: pd.DataFrame, n_loci: int
) -> pd.DataFrame:
    values = absolute.set_index("str_id")
    metadata_by_sample = metadata.set_index("sample")
    rows = []
    for sample in values.columns:
        observed = values[sample].dropna()
        rows.append({
            "sample": sample,
            "subspecies": metadata_by_sample.loc[sample, "subspecies"],
            "country": metadata_by_sample.loc[sample, "Country"],
            "called_loci": len(observed),
            "call_rate_percent": 100 * len(observed) / n_loci,
            "mean_absolute_deviation_percent": observed.mean(),
        })
    return pd.DataFrame(rows)


def bootstrap_summary(samples: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in GROUP_ORDER:
        values = samples.loc[
            samples["subspecies"].eq(group), "mean_absolute_deviation_percent"
        ].to_numpy()
        draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        rows.append({
            "subspecies": group,
            "n_samples": len(values),
            "mean_of_sample_means": values.mean(),
            "median_of_sample_means": np.median(values),
            "bootstrap_mean_ci95_low": np.quantile(draws, 0.025),
            "bootstrap_mean_ci95_high": np.quantile(draws, 0.975),
        })
    return pd.DataFrame(rows)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    comparisons = np.greater.outer(x, y).sum() - np.less.outer(x, y).sum()
    return float(comparisons / (len(x) * len(y)))


def statistical_test(samples: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    x = samples.loc[
        samples["subspecies"].eq("europaea"), "mean_absolute_deviation_percent"
    ].to_numpy()
    y = samples.loc[
        samples["subspecies"].eq("sylvestris"), "mean_absolute_deviation_percent"
    ].to_numpy()
    test = mannwhitneyu(x, y, alternative="two-sided")
    rng = np.random.default_rng(seed)
    bootstrap_delta = np.empty(iterations)
    for index in range(iterations):
        bootstrap_delta[index] = cliffs_delta(
            rng.choice(x, len(x), replace=True), rng.choice(y, len(y), replace=True)
        )
    return pd.DataFrame([{
        "metric": "mean_absolute_deviation_percent",
        "test": "Mann-Whitney U",
        "comparison": "europaea vs sylvestris",
        "statistic": test.statistic,
        "pvalue": test.pvalue,
        "cliffs_delta_europaea_vs_sylvestris": cliffs_delta(x, y),
        "effect_ci95_low": np.quantile(bootstrap_delta, 0.025),
        "effect_ci95_high": np.quantile(bootstrap_delta, 0.975),
    }])


def set_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def plot_deviation(samples: pd.DataFrame, output_dir: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COLUMN_IN, 2.75))
    sns.boxplot(
        data=samples,
        x="subspecies",
        y="mean_absolute_deviation_percent",
        order=GROUP_ORDER,
        hue="subspecies",
        palette=PALETTE,
        legend=False,
        width=0.52,
        linewidth=0.7,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=samples,
        x="subspecies",
        y="mean_absolute_deviation_percent",
        order=GROUP_ORDER,
        hue="subspecies",
        palette=PALETTE,
        legend=False,
        jitter=0.20,
        alpha=0.55,
        size=2.0,
        linewidth=0,
        ax=ax,
    )
    ax.set_xticks(range(len(GROUP_ORDER)), [GROUP_LABELS[group] for group in GROUP_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("Mean absolute deviation\nfrom reference (%)")
    ax.set_title("Reference-relative pSTR deviation", loc="left", fontweight="bold")
    ax.grid(False)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.23, right=0.97, top=0.88, bottom=0.19)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "reference_relative_variation"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def write_report(
    output: Path,
    loci: pd.DataFrame,
    samples: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    counts = samples["subspecies"].value_counts()
    group_means = samples.groupby("subspecies")["mean_absolute_deviation_percent"].mean()
    test = tests.iloc[0]
    output.write_text(f"""# Absolute Reference-relative pSTR Deviation

The analysis included all {len(loci):,} pSTR loci in the main cohort: 
{counts['europaea']} *Olea europaea* subsp. *europaea* samples and
{counts['sylvestris']} subsp. *sylvestris* samples. Cuspidata samples were not
included. No variance threshold or additional subspecies-specific call-rate
filter was applied.

ConSTRain `REPLEN` values and reference length were expressed in repeat units.
For each called sample-locus genotype, absolute deviation was calculated as
`100 * abs(REPLEN - reference repeat number) / reference repeat number`, where
`reference repeat number = (BED end - BED start) / motif length`. The plotted
value is the mean across all called pSTR loci for each sample.

| Subspecies | Samples | Mean of sample mean deviations (%) |
|---|---:|---:|
| europaea | {counts['europaea']} | {group_means['europaea']:.4f} |
| sylvestris | {counts['sylvestris']} | {group_means['sylvestris']:.4f} |

The two-sided Mann-Whitney test gave P={test['pvalue']:.3g}, with Cliff's delta
(europaea relative to sylvestris) of
{test['cliffs_delta_europaea_vs_sylvestris']:.3f}. Individuals are the
replicates, and inference for sylvestris remains limited by its sample size of
{counts['sylvestris']}.

Deviation is relative to the single reference assembly and should not be
interpreted as a directly observed mutation rate.
""")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    for path in (args.matrix, args.str_bed, args.metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    absolute, loci, metadata = prepare_data(args)
    samples = summarize_samples(absolute, metadata, len(loci))
    group_summary = bootstrap_summary(samples, args.bootstrap, args.seed)
    tests = statistical_test(samples, args.bootstrap, args.seed)

    loci.to_csv(source_dir / "comparison_loci.tsv", sep="\t", index=False)
    absolute.to_csv(
        source_dir / "absolute_reference_deviation_percent.csv", index=False, na_rep=""
    )
    samples.to_csv(source_dir / "sample_summaries.tsv", sep="\t", index=False)
    group_summary.to_csv(source_dir / "subspecies_summaries.tsv", sep="\t", index=False)
    tests.to_csv(source_dir / "statistical_tests.tsv", sep="\t", index=False)
    plot_deviation(samples, args.output_dir)
    write_report(args.output_dir / "reference_relative_variation_report.md", loci, samples, tests)
    print(f"Compared {len(samples)} samples across all {len(loci):,} pSTR loci")


if __name__ == "__main__":
    main()
