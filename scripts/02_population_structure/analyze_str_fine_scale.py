#!/usr/bin/env python3
"""Test whether split STR components add fine-scale geographic information."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
STRUCTURE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STRUCTURE_DIR.parents[1]
SNP_COLUMNS = [
    "ancestry_1", "ancestry_2", "ancestry_3",
    "ancestry_4", "ancestry_5", "ancestry_6",
]
SPLITS = {
    "C1: Q3/Q8": ("Q3", "Q8"),
    "C2: Q4/Q7": ("Q4", "Q7"),
    "C4: Q1/Q5": ("Q1", "Q5"),
}
PARENT_SNP_COLUMNS = {
    "C1: Q3/Q8": "ancestry_1",
    "C2: Q4/Q7": "ancestry_2",
    "C4: Q1/Q5": "ancestry_4",
}
SPLIT_DISPLAY_LABELS = {
    "C1: Q3/Q8": "C1-(Q3+Q8)",
    "C2: Q4/Q7": "C2-(Q4+Q7)",
    "C4: Q1/Q5": "C4-(Q1+Q5)",
}
MINIMUM_GROUP_SIZE = 5
PSEUDOCOUNT = 1e-5
FONT_FAMILY = "Nimbus Sans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snp-q",
        type=Path,
        default=STRUCTURE_DIR / "admixture/source_data/admixture_k6.csv",
    )
    parser.add_argument(
        "--str-q",
        type=Path,
        default=STRUCTURE_DIR
        / "dNMF/chr_str_var_gt_0.1/source_data/dnmf_pos9.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "olive_str/metadata/Geo_loc.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def ilr_transform(values: np.ndarray) -> np.ndarray:
    adjusted = values + PSEUDOCOUNT
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return np.log(adjusted) @ helmert(adjusted.shape[1], full=False).T


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    snp = pd.read_csv(
        args.snp_q,
        sep="," if args.snp_q.suffix.lower() == ".csv" else "\t",
    )
    if "sample" in snp.columns and "sample_id" not in snp.columns:
        snp = snp.rename(columns={"sample": "sample_id"})
    numbered_snp = [str(index) for index in range(6)]
    if all(column in snp.columns for column in numbered_snp):
        snp = snp.rename(
            columns={
                str(index): f"ancestry_{index + 1}"
                for index in range(6)
            }
        )

    str_q = pd.read_csv(
        args.str_q,
        sep="," if args.str_q.suffix.lower() == ".csv" else "\t",
    )
    if "sample" in str_q.columns and "sample_id" not in str_q.columns:
        str_q = str_q.rename(columns={"sample": "sample_id"})
    numbered_str = [str(index) for index in range(9)]
    if all(column in str_q.columns for column in numbered_str):
        str_q = str_q.rename(
            columns={
                str(index): f"Q{index + 1}"
                for index in range(9)
            }
        )

    metadata = pd.read_csv(args.metadata, encoding="latin-1", dtype=str).fillna("")
    required_snp = {"sample_id", *SNP_COLUMNS}
    if not required_snp.issubset(snp.columns):
        raise ValueError(f"SNP input lacks {sorted(required_snp - set(snp.columns))}")
    q_columns = [f"Q{index}" for index in range(1, 10)]
    if not {"sample_id", *q_columns}.issubset(str_q.columns):
        raise ValueError("STR input must contain sample_id and Q1-Q9")
    if not {"ID", "Country", "Provience"}.issubset(metadata.columns):
        raise ValueError("Metadata must contain ID, Country, and Provience")
    for frame, sample_column in ((snp, "sample_id"), (str_q, "sample_id"), (metadata, "ID")):
        if frame[sample_column].duplicated().any():
            raise ValueError(f"Duplicate sample IDs in {sample_column}")

    metadata["subspecies"] = np.where(
        metadata["Subspecies"].str.lower().str.endswith("sylvestris"),
        "sylvestris",
        "europaea",
    )
    data = (
        snp[["sample_id", *SNP_COLUMNS]]
        .merge(str_q[["sample_id", *q_columns]], on="sample_id", validate="one_to_one")
        .merge(
            metadata[["ID", "Country", "Provience", "subspecies"]],
            left_on="sample_id",
            right_on="ID",
            validate="one_to_one",
        )
        .drop(columns="ID")
    )
    if len(data) != 212:
        raise ValueError(f"Expected 212 shared samples, found {len(data)}")
    if not np.allclose(data[SNP_COLUMNS].sum(axis=1), 1, atol=1e-4):
        raise ValueError("SNP ancestry proportions do not sum to one")
    if not np.allclose(data[q_columns].sum(axis=1), 1, atol=1e-8):
        raise ValueError("STR ancestry proportions do not sum to one")

    data["fine_geography"] = np.where(
        data["Country"].eq("Iran"),
        "Iran: " + data["Provience"].str.strip(),
        data["Country"],
    )
    data.loc[
        data["Country"].eq("Iran")
        & data["Provience"].str.strip().isin(["", "-"]),
        "fine_geography",
    ] = ""
    data["included_subspecies"] = data["subspecies"].eq("europaea")
    candidate = data[data["included_subspecies"] & data["fine_geography"].ne("")]
    counts = candidate["fine_geography"].value_counts()
    retained_groups = counts[counts >= MINIMUM_GROUP_SIZE].index.tolist()
    data["included_group_size"] = data["fine_geography"].isin(retained_groups)
    analysis = data[data["included_subspecies"] & data["included_group_size"]].copy()

    group_order = [
        group
        for group in ("Greece", "Spain", "Syria")
        if group in retained_groups
    ] + sorted(group for group in retained_groups if group.startswith("Iran: "))
    analysis["fine_geography"] = pd.Categorical(
        analysis["fine_geography"], categories=group_order, ordered=True
    )
    analysis = analysis.sort_values(["fine_geography", "sample_id"]).reset_index(drop=True)
    return data, analysis


def design_matrices(
    snp_ilr: np.ndarray, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    group_dummies = pd.get_dummies(groups, drop_first=True, dtype=float).to_numpy()
    reduced = np.column_stack([np.ones(len(snp_ilr)), snp_ilr])
    full = np.column_stack([reduced, group_dummies])
    return reduced, full


def residualize(values: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, _ = np.linalg.qr(design, mode="reduced")
    q = q[:, :np.linalg.matrix_rank(design)]
    fitted = q @ (q.T @ values)
    return values - fitted, q


def incremental_geography_test(
    geography: pd.Series,
    reduced: np.ndarray,
    full: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    categories = list(geography.cat.categories)
    codes = geography.cat.codes.to_numpy()
    one_hot = np.eye(len(categories))[codes]
    # Orthonormal contrasts preserve the Euclidean separation among geographic
    # categories without choosing one arbitrary reference group.
    response = one_hot @ helmert(len(categories), full=False).T
    reduced_residual, q_reduced = residualize(response, reduced)
    full_residual, q_full = residualize(response, full)
    sse_reduced = float(np.square(reduced_residual).sum())
    sse_full = float(np.square(full_residual).sum())
    observed = (sse_reduced - sse_full) / sse_reduced
    reduced_fitted = response - reduced_residual

    exceedances = 0
    for _ in range(permutations):
        permuted = reduced_fitted + reduced_residual[
            rng.permutation(len(reduced_residual))
        ]
        permuted_reduced = permuted - q_reduced @ (q_reduced.T @ permuted)
        permuted_full = permuted - q_full @ (q_full.T @ permuted)
        permuted_sse_reduced = float(np.square(permuted_reduced).sum())
        permuted_sse_full = float(np.square(permuted_full).sum())
        statistic = (
            permuted_sse_reduced - permuted_sse_full
        ) / permuted_sse_reduced
        exceedances += statistic >= observed
    return {
        "partial_r_squared": observed,
        "permutation_p": (exceedances + 1) / (permutations + 1),
        "sse_matched_snp_only": sse_reduced,
        "sse_matched_snp_plus_str_pair": sse_full,
    }


def add_split_balances(
    data: pd.DataFrame, snp_ilr: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    reduced = np.column_stack([np.ones(len(snp_ilr)), snp_ilr])
    rows = []
    for split_label, (first, second) in SPLITS.items():
        balance = np.log(
            (result[first].to_numpy() + PSEUDOCOUNT)
            / (result[second].to_numpy() + PSEUDOCOUNT)
        ) / np.sqrt(2)
        residual, _ = residualize(balance, reduced)
        result[f"{split_label} balance"] = balance
        result[f"{split_label} SNP-adjusted residual"] = residual
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": result["sample_id"],
                    "fine_geography": result["fine_geography"].astype(str),
                    "split": split_label,
                    "first_str_component": first,
                    "second_str_component": second,
                    "str_ilr_balance": balance,
                    "snp_adjusted_balance_residual": residual,
                }
            )
        )
    return result, pd.concat(rows, ignore_index=True)


def cross_validated_prediction(
    data: pd.DataFrame,
    snp_ilr: np.ndarray,
    cv_repeats: int,
    seed: int,
) -> pd.DataFrame:
    labels = data["fine_geography"].astype(str).to_numpy()
    classes = np.unique(labels)
    balances = {
        split_label: np.log(
            (data[first].to_numpy() + PSEUDOCOUNT)
            / (data[second].to_numpy() + PSEUDOCOUNT)
        )[:, None]
        / np.sqrt(2)
        for split_label, (first, second) in SPLITS.items()
    }
    features = {"SNP only": snp_ilr}
    features.update(
        {
            f"SNP + {split_label}": np.column_stack([snp_ilr, balance])
            for split_label, balance in balances.items()
        }
    )
    features["SNP + all STR splits"] = np.column_stack(
        [snp_ilr, *balances.values()]
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=cv_repeats, random_state=seed
    )
    rows = []
    for split_index, (train, test) in enumerate(splitter.split(snp_ilr, labels)):
        repeat = split_index // 5 + 1
        fold = split_index % 5 + 1
        for model_name, matrix in features.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=5_000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            )
            model.fit(matrix[train], labels[train])
            probabilities = model.predict_proba(matrix[test])
            predicted = model.predict(matrix[test])
            model_classes = model.named_steps["logisticregression"].classes_
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "log_loss": log_loss(
                        labels[test], probabilities, labels=model_classes
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        labels[test], predicted
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_prediction(cv: pd.DataFrame) -> pd.DataFrame:
    repeat_means = (
        cv.groupby(["repeat", "model"], as_index=False)
        [["log_loss", "balanced_accuracy"]]
        .mean()
    )
    baseline = repeat_means[repeat_means["model"].eq("SNP only")].set_index("repeat")
    rows = []
    for model_name, frame in repeat_means.groupby("model", sort=False):
        if model_name == "SNP only":
            continue
        frame = frame.set_index("repeat").loc[baseline.index]
        log_delta = baseline["log_loss"] - frame["log_loss"]
        accuracy_delta = frame["balanced_accuracy"] - baseline["balanced_accuracy"]
        rows.append(
            {
                "model": model_name,
                "mean_log_loss_improvement": log_delta.mean(),
                "log_loss_improvement_ci_low": np.quantile(log_delta, 0.025),
                "log_loss_improvement_ci_high": np.quantile(log_delta, 0.975),
                "mean_balanced_accuracy_improvement": accuracy_delta.mean(),
                "balanced_accuracy_improvement_ci_low": np.quantile(
                    accuracy_delta, 0.025
                ),
                "balanced_accuracy_improvement_ci_high": np.quantile(
                    accuracy_delta, 0.975
                ),
                "n_cv_repeats": len(frame),
            }
        )
    return pd.DataFrame(rows)


def plot_balances(long_data: pd.DataFrame, figure_dir: Path) -> None:
    set_style()
    group_order = list(dict.fromkeys(long_data["fine_geography"]))
    palette = sns.color_palette("tab20", n_colors=len(group_order))
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.4), sharex=True)
    for axis, (split_label, frame) in zip(
        axes, long_data.groupby("split", sort=False)
    ):
        sns.boxplot(
            data=frame,
            x="fine_geography",
            y="snp_adjusted_balance_residual",
            order=group_order,
            palette=palette,
            hue="fine_geography",
            legend=False,
            showfliers=False,
            linewidth=0.8,
            ax=axis,
        )
        sns.stripplot(
            data=frame,
            x="fine_geography",
            y="snp_adjusted_balance_residual",
            order=group_order,
            color="#222222",
            size=2.2,
            alpha=0.45,
            jitter=0.18,
            ax=axis,
        )
        axis.axhline(0, color="#777777", lw=0.7, ls="--")
        axis.set(
            title=split_label,
            xlabel="",
            ylabel="STR balance residual\nafter SNP adjustment",
        )
        sns.despine(ax=axis)
    axes[-1].tick_params(axis="x", rotation=42)
    for label in axes[-1].get_xticklabels():
        label.set_ha("right")
    figure.tight_layout(h_pad=1.0)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            figure_dir / f"str_split_balances_by_fine_geography.{extension}",
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_summary(
    tests: pd.DataFrame, prediction: pd.DataFrame, figure_dir: Path
) -> None:
    set_style()
    split_labels = [
        SPLIT_DISPLAY_LABELS[label]
        for label in tests["split"]
    ]
    colors = sns.color_palette("colorblind", len(split_labels))

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    y = np.arange(len(split_labels))
    axis.barh(y, tests["partial_r_squared"], color=colors, alpha=0.8)
    for index, row in tests.iterrows():
        axis.text(
            row["partial_r_squared"] + 0.003,
            index,
            "****" if row["permutation_p"] <= 0.0001
            else f"P={row['permutation_p']:.4f}",
            ha="left",
            va="center",
            fontsize=7,
        )
    axis.set(
        yticks=y,
        yticklabels=split_labels,
        xlabel="Partial $R^2$",
        ylabel="",
        title="Additional geography explained by STR pairs\nafter matched SNP component",
        xlim=(0, tests["partial_r_squared"].max() + 0.025),
    )
    axis.invert_yaxis()
    sns.despine(ax=axis)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            figure_dir / f"str_incremental_geographic_information.{extension}",
            bbox_inches="tight",
        )
    plt.close(figure)

    plot_prediction = prediction.copy()
    plot_prediction["label"] = (
        plot_prediction["model"]
        .str.replace("SNP + ", "", regex=False)
        .str.replace("all STR splits", "All splits", regex=False)
    )
    plot_prediction["label"] = plot_prediction["label"].replace(
        SPLIT_DISPLAY_LABELS
    )
    x = np.arange(len(plot_prediction))
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    axes[0].bar(
        x,
        plot_prediction["mean_log_loss_improvement"],
        color=sns.color_palette("Set2", len(x)),
    )
    axes[0].errorbar(
        x,
        plot_prediction["mean_log_loss_improvement"],
        yerr=np.vstack(
            [
                plot_prediction["mean_log_loss_improvement"]
                - plot_prediction["log_loss_improvement_ci_low"],
                plot_prediction["log_loss_improvement_ci_high"]
                - plot_prediction["mean_log_loss_improvement"],
            ]
        ),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=0.8,
    )
    axes[0].axhline(0, color="#555555", lw=0.7)
    axes[0].set(
        xticks=x,
        xticklabels=plot_prediction["label"],
        ylabel="Reduction in log loss",
        title="Held-out geographic prediction",
    )

    axes[1].bar(
        x,
        plot_prediction["mean_balanced_accuracy_improvement"],
        color=sns.color_palette("Set2", len(x)),
    )
    axes[1].errorbar(
        x,
        plot_prediction["mean_balanced_accuracy_improvement"],
        yerr=np.vstack(
            [
                plot_prediction["mean_balanced_accuracy_improvement"]
                - plot_prediction["balanced_accuracy_improvement_ci_low"],
                plot_prediction["balanced_accuracy_improvement_ci_high"]
                - plot_prediction["mean_balanced_accuracy_improvement"],
            ]
        ),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=0.8,
    )
    axes[1].axhline(0, color="#555555", lw=0.7)
    axes[1].set(
        xticks=x,
        xticklabels=plot_prediction["label"],
        ylabel="Δ balanced accuracy",
        title="Held-out geographic prediction",
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        sns.despine(ax=axis)
    figure.tight_layout(w_pad=1.2)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            figure_dir / f"str_held_out_geographic_prediction.{extension}",
            bbox_inches="tight",
        )
    plt.close(figure)


def write_report(
    args: argparse.Namespace,
    all_samples: pd.DataFrame,
    analysis: pd.DataFrame,
    tests: pd.DataFrame,
    prediction: pd.DataFrame,
) -> None:
    retained_counts = analysis["fine_geography"].value_counts(sort=False)
    group_lines = "\n".join(
        f"- {group}: n = {count}" for group, count in retained_counts.items()
    )
    test_lines = "\n".join(
        f"- {row.split}: partial R² = {row.partial_r_squared:.3f}, "
        f"permutation P = {row.permutation_p:.4g}"
        for row in tests.itertuples(index=False)
    )
    prediction_lines = "\n".join(
        f"- {row.model}: mean log-loss improvement = "
        f"{row.mean_log_loss_improvement:.3f}; mean balanced-accuracy "
        f"improvement = {row.mean_balanced_accuracy_improvement:.3f}"
        for row in prediction.itertuples(index=False)
    )
    excluded = all_samples[
        all_samples["included_subspecies"]
        & ~all_samples["included_group_size"]
    ]["fine_geography"].value_counts()
    excluded_lines = "\n".join(
        f"- {group or '(missing Iranian province)'}: n = {count}"
        for group, count in excluded.items()
    )
    text = f"""# Additional fine-scale structure in split STR components

## Design

The analysis used the native SNP component order in `admixture_k6.csv` and
the STR dNMF K=9 coefficients in `dnmf_pos9.csv`. It examined C1→Q3/Q8,
C2→Q4/Q7, and C4→Q1/Q5. For each alignment, fine geography was encoded with
orthonormal contrasts. The reduced model used the matched SNP component as
the sole ancestry predictor; the full model added both assigned STR
components. For example, the Q3/Q8 test compared `geography ~ C1` with
`geography ~ C1 + Q3 + Q8`.

A Freedman–Lane residual permutation test ({args.permutations:,}
permutations) evaluated the increase in multivariate explained variance.
The reported partial R² is therefore the additional geographic variation
explained by the STR pair after removing its matched SNP component.

The held-out analysis used {args.cv_repeats} repeats of stratified five-fold
cross-validation. Multinomial logistic models based on SNP ILRs alone were
compared with models adding each STR balance or all three balances.

## Geographic groups

The primary analysis retained {len(analysis)} *europaea* accessions. Other
countries were labeled by country; Iranian accessions were labeled by province.
Groups with fewer than {MINIMUM_GROUP_SIZE} samples were excluded.

{group_lines}

Excluded small or missing groups among *europaea*:

{excluded_lines or "- None"}

## Conditional association results

{test_lines}

`****` in the partial-R² figure denotes permutation P ≤ 0.0001, the minimum
attainable value with 9,999 permutations plus the observed arrangement.

## Cross-validated incremental prediction

Positive log-loss and balanced-accuracy improvements favor adding STRs.

{prediction_lines}

These results support finer-scale STR structure only where the STR pair adds
significant geographic explanatory power beyond its matched SNP component and
the out-of-sample prediction improvement is positive. Visual separation alone
is not treated as sufficient evidence.

## Outputs

- `figures/str_split_balances_by_fine_geography.*`
- `figures/str_incremental_geographic_information.*`
- `figures/str_held_out_geographic_prediction.*`
- `source_data/sample_str_split_balances.tsv`
- `source_data/conditional_geography_tests.tsv`
- `source_data/cross_validated_geography_prediction.tsv`
- `source_data/prediction_summary.tsv`
- `source_data/sample_inclusion.tsv`
"""
    (args.output_dir / "analysis_report.md").write_text(text)


def main() -> None:
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    all_samples, analysis = load_data(args)
    snp_ilr = ilr_transform(analysis[SNP_COLUMNS].to_numpy(dtype=float))
    analysis, long_balances = add_split_balances(analysis, snp_ilr)

    rng = np.random.default_rng(args.seed)
    test_rows = []
    for split_label, (first, second) in SPLITS.items():
        parent_column = PARENT_SNP_COLUMNS[split_label]
        reduced = np.column_stack(
            [np.ones(len(analysis)), analysis[parent_column].to_numpy()]
        )
        full = np.column_stack(
            [
                reduced,
                analysis[first].to_numpy(),
                analysis[second].to_numpy(),
            ]
        )
        result = incremental_geography_test(
            analysis["fine_geography"],
            reduced,
            full,
            args.permutations,
            rng,
        )
        test_rows.append(
            {
                "split": split_label,
                "matched_snp_component": split_label.split(":")[0],
                "matched_snp_source_column": parent_column,
                "str_components_added": f"{first},{second}",
                "n_samples": len(analysis),
                "n_geographic_groups": analysis["fine_geography"].nunique(),
                "permutations": args.permutations,
                **result,
            }
        )
    tests = pd.DataFrame(test_rows)

    cv = cross_validated_prediction(
        analysis, snp_ilr, args.cv_repeats, args.seed + 1
    )
    prediction = summarize_prediction(cv)

    all_samples[
        [
            "sample_id", "subspecies", "Country", "Provience",
            "fine_geography", "included_subspecies", "included_group_size",
        ]
    ].to_csv(source_dir / "sample_inclusion.tsv", sep="\t", index=False)
    long_balances.to_csv(
        source_dir / "sample_str_split_balances.tsv", sep="\t", index=False
    )
    tests.to_csv(
        source_dir / "conditional_geography_tests.tsv", sep="\t", index=False
    )
    cv.to_csv(
        source_dir / "cross_validated_geography_prediction.tsv",
        sep="\t",
        index=False,
    )
    prediction.to_csv(
        source_dir / "prediction_summary.tsv", sep="\t", index=False
    )

    plot_balances(long_balances, figure_dir)
    plot_summary(tests, prediction, figure_dir)
    write_report(args, all_samples, analysis, tests, prediction)
    print(tests.to_string(index=False))
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
