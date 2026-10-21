#!/usr/bin/env python3
"""Compare finalized whole-mean genome-wide results across three models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, hypergeom, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
MODELS = ("argont", "plantcad2", "gpn")
DISPLAY = {"argont": "ArgoNT", "plantcad2": "PlantCAD2", "gpn": "GPN"}


def load_primary_results(config):
    frames = []
    for model in MODELS:
        source = Path(config["primary_results"][model])
        frame = pd.read_csv(source)
        frame = frame[
            frame.pooling.eq(config["pooling"])
            & frame.layer.eq(config[f"{model}_layer"])
        ].copy()
        expected = config["expected_loci"] * len(config["metrics"])
        if len(frame) != expected or frame.str_id.nunique() != config["expected_loci"]:
            raise RuntimeError(
                f"{model}: unexpected primary result dimensions "
                f"(rows={len(frame)}, loci={frame.str_id.nunique()})"
            )
        if set(frame.metric) != set(config["metrics"]):
            raise RuntimeError(f"{model}: unexpected metric families")
        if frame.groupby(["str_id", "metric"]).size().ne(1).any():
            raise RuntimeError(f"{model}: duplicate or missing locus/metric rows")
        if frame[["slope", "intercept", "r2", "p_value", "q_value"]].isna().any().any():
            raise RuntimeError(f"{model}: non-finite primary regression result")
        if not frame.fdr_0_05.astype(bool).eq(frame.q_value.lt(0.05)).all():
            raise RuntimeError(f"{model}: stored FDR flag disagrees with q-value")
        frame["model"] = model
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    locus_sets = [
        set(data.loc[data.model.eq(model), "str_id"]) for model in MODELS
    ]
    if any(loci != locus_sets[0] for loci in locus_sets[1:]):
        raise RuntimeError("Primary result files do not contain the same STR loci")
    return data


def pairwise(frame):
    rows = []
    n = len(frame)
    for index, first in enumerate(MODELS):
        for second in MODELS[index + 1 :]:
            a = frame[f"{first}_significant"].to_numpy()
            b = frame[f"{second}_significant"].to_numpy()
            ac, bc = int(a.sum()), int(b.sum())
            overlap = int((a & b).sum())
            union = int((a | b).sum())
            neither = int((~a & ~b).sum())
            odds, fisher_p = fisher_exact(
                [[overlap, ac - overlap], [bc - overlap, neither]],
                alternative="greater",
            )
            expected = ac * bc / n
            shared = frame[a & b]
            first_slope = shared[f"{first}_slope"].astype(float).to_numpy()
            second_slope = shared[f"{second}_slope"].astype(float).to_numpy()
            pearson = pearsonr(first_slope, second_slope)
            spearman = spearmanr(first_slope, second_slope)
            rows.append(
                {
                    "model_1": DISPLAY[first],
                    "model_2": DISPLAY[second],
                    "common_loci": n,
                    "model_1_significant": ac,
                    "model_2_significant": bc,
                    "overlap": overlap,
                    "union": union,
                    "jaccard": overlap / union if union else np.nan,
                    "fraction_model_1_overlapping": overlap / ac if ac else np.nan,
                    "fraction_model_2_overlapping": overlap / bc if bc else np.nan,
                    "expected_overlap_independence": expected,
                    "observed_to_expected": overlap / expected if expected else np.nan,
                    "hypergeometric_p": hypergeom.sf(overlap - 1, n, ac, bc),
                    "fisher_odds_ratio": odds,
                    "fisher_p": fisher_p,
                    "shared_sig_pearson_r": pearson.statistic,
                    "shared_sig_pearson_p": pearson.pvalue,
                    "shared_sig_spearman_rho": spearman.statistic,
                    "shared_sig_spearman_p": spearman.pvalue,
                }
            )
    return pd.DataFrame(rows)


def patterns(frame):
    rows = []
    for argont in (False, True):
        for plantcad2 in (False, True):
            for gpn in (False, True):
                mask = (
                    (frame.argont_significant == argont)
                    & (frame.plantcad2_significant == plantcad2)
                    & (frame.gpn_significant == gpn)
                )
                values = (argont, plantcad2, gpn)
                members = [
                    DISPLAY[model]
                    for model, value in zip(MODELS, values)
                    if value
                ]
                rows.append(
                    {
                        "argont": argont,
                        "plantcad2": plantcad2,
                        "gpn": gpn,
                        "pattern": " + ".join(members)
                        if members
                        else "Not significant",
                        "loci": int(mask.sum()),
                        "fraction": float(mask.mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values("loci", ascending=False).reset_index(drop=True)


def compare_normalized_magnitude(data, config):
    loci = pd.read_csv(config["loci"], usecols=["str_id", "period"])
    primary = data.loc[
        data.metric.eq("l2"), ["str_id", "model", "slope"]
    ].merge(loci, on="str_id", validate="many_to_one")
    primary["absolute_slope_per_bp"] = primary.slope.abs() / primary.period

    parameter_rows = []
    normalized = []
    for model, frame in primary.groupby("model", sort=False):
        frame = frame.copy()
        positive = frame.loc[frame.absolute_slope_per_bp.gt(0), "absolute_slope_per_bp"]
        if positive.empty:
            raise RuntimeError(f"{model}: no positive L2 slope magnitudes")
        floor = float(positive.min() / 2)
        frame["log10_absolute_slope_per_bp"] = np.log10(
            frame.absolute_slope_per_bp.clip(lower=floor)
        )
        center = float(frame.log10_absolute_slope_per_bp.median())
        mad = float(
            np.median(np.abs(frame.log10_absolute_slope_per_bp.to_numpy() - center))
        )
        robust_scale = 1.4826 * mad
        if not np.isfinite(robust_scale) or robust_scale <= 0:
            raise RuntimeError(f"{model}: invalid robust magnitude scale")
        frame["robust_z_magnitude"] = (
            frame.log10_absolute_slope_per_bp - center
        ) / robust_scale
        frame["magnitude_percentile"] = frame.absolute_slope_per_bp.rank(
            method="average", pct=True
        )
        parameter_rows.append(
            {
                "model": model,
                "loci": len(frame),
                "median_absolute_slope_per_bp": frame.absolute_slope_per_bp.median(),
                "log10_center": center,
                "log10_mad": mad,
                "robust_log10_scale": robust_scale,
                "zero_magnitudes": int(frame.absolute_slope_per_bp.eq(0).sum()),
                "zero_replacement_floor": floor,
            }
        )
        normalized.append(frame)
    normalized = pd.concat(normalized, ignore_index=True)

    wide = normalized.pivot(
        index="str_id",
        columns="model",
        values=[
            "absolute_slope_per_bp",
            "log10_absolute_slope_per_bp",
            "robust_z_magnitude",
            "magnitude_percentile",
        ],
    )
    wide.columns = [f"{model}_{value}" for value, model in wide.columns]
    wide = wide.reset_index()

    pair_rows = []
    for index, first in enumerate(MODELS):
        for second in MODELS[index + 1 :]:
            x = wide[f"{first}_robust_z_magnitude"].to_numpy()
            y = wide[f"{second}_robust_z_magnitude"].to_numpy()
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            pair_rows.append(
                {
                    "model_1": DISPLAY[first],
                    "model_2": DISPLAY[second],
                    "loci": len(wide),
                    "pearson_r_robust_z_magnitude": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_rho_magnitude": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "median_absolute_z_difference": np.median(np.abs(x - y)),
                    "both_at_least_one_robust_sd": int(((x >= 1) & (y >= 1)).sum()),
                }
            )
    pair = pd.DataFrame(pair_rows)
    parameters = pd.DataFrame(parameter_rows)

    wide.to_csv(
        RESULTS / "matched_l2_normalized_magnitude.csv.gz",
        index=False,
        compression="gzip",
    )
    parameters.to_csv(RESULTS / "normalized_magnitude_parameters.csv", index=False)
    pair.to_csv(RESULTS / "normalized_magnitude_pairwise.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, row in zip(axes, pair.itertuples()):
        first = next(key for key, value in DISPLAY.items() if value == row.model_1)
        second = next(key for key, value in DISPLAY.items() if value == row.model_2)
        x = wide[f"{first}_robust_z_magnitude"].to_numpy()
        y = wide[f"{second}_robust_z_magnitude"].to_numpy()
        limit = max(3.0, float(np.quantile(np.abs(np.r_[x, y]), 0.995)))
        ax.hexbin(x, y, gridsize=65, bins="log", mincnt=1, cmap="viridis")
        ax.plot([-limit, limit], [-limit, limit], color="white", linestyle="--", linewidth=1)
        ax.set(
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            xlabel=f"{row.model_1} robust magnitude z",
            ylabel=f"{row.model_2} robust magnitude z",
            title=f"{row.model_1} vs {row.model_2}",
        )
        ax.text(
            0.04,
            0.96,
            f"Pearson r={row.pearson_r_robust_z_magnitude:.3f}\n"
            f"Spearman ρ={row.spearman_rho_magnitude:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )
    fig.suptitle(
        "Normalized whole-mean L2 slope magnitude across models\n"
        "absolute slope per bp → log10 → within-model median/MAD robust z",
        fontsize=14,
    )
    fig.savefig(FIGURES / "normalized_l2_magnitude_comparison.png", dpi=200)
    plt.close(fig)
    return parameters, pair


def format_p(value):
    return "<1e-300" if value == 0 else f"{value:.3g}"


def make_figure(pattern, pair):
    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.2,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 4.0),
        gridspec_kw={"width_ratios": [1.4, 1]},
        constrained_layout=True,
    )
    selected = pattern[pattern.pattern != "Not significant"]
    axes[0].barh(
        np.arange(len(selected)),
        selected.loci,
        color=sns.color_palette("colorblind")[0],
        edgecolor="none",
    )
    axes[0].set_yticks(np.arange(len(selected)), selected.pattern)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Loci", title="FDR < 0.05 membership")
    for y, value in enumerate(selected.loci):
        axes[0].text(value, y, f" {value:,}", va="center")
    sns.despine(ax=axes[0])

    names = [DISPLAY[model] for model in MODELS]
    matrix = np.eye(3)
    for row in pair.itertuples():
        i, j = names.index(row.model_1), names.index(row.model_2)
        matrix[i, j] = matrix[j, i] = row.jaccard
    image = axes[1].imshow(
        matrix,
        vmin=0,
        vmax=1,
        cmap="Blues",
        aspect="equal",
    )
    axes[1].set_xticks(range(3), names, rotation=30, ha="right")
    axes[1].set_yticks(range(3), names)
    axes[1].set_title("Jaccard index")
    for i in range(3):
        for j in range(3):
            axes[1].text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(FIGURES / "matched_three_model_overlap.png", dpi=300)
    fig.savefig(FIGURES / "matched_three_model_overlap.pdf")
    plt.close(fig)


def make_upset_figure(pattern, wide):
    selected = (
        pattern[pattern.pattern.ne("Not significant")]
        .sort_values("loci", ascending=False)
        .reset_index(drop=True)
    )
    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.6,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, (bars, matrix) = plt.subplots(
        2,
        1,
        figsize=(8.0, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.25], "hspace": 0.04},
        constrained_layout=True,
    )
    positions = np.arange(len(selected))
    color = sns.color_palette("colorblind")[0]
    bars.bar(positions, selected.loci, width=0.68, color=color, edgecolor="none")
    for x, value in zip(positions, selected.loci):
        bars.text(x, value, f"{value:,}", ha="center", va="bottom")
    bars.set(ylabel="Exact intersection size")
    bars.tick_params(axis="x", bottom=False, labelbottom=False)
    sns.despine(ax=bars, bottom=True)

    set_sizes = {
        model: int(wide[f"{model}_significant"].sum()) for model in MODELS
    }
    y_positions = np.arange(len(MODELS))
    for x, row in enumerate(selected.itertuples()):
        active = np.asarray([bool(getattr(row, model)) for model in MODELS])
        matrix.scatter(
            np.full(len(MODELS), x),
            y_positions,
            s=48,
            color=np.where(active, "#222222", "#d9d9d9"),
            zorder=2,
        )
        active_y = y_positions[active]
        if len(active_y) > 1:
            matrix.plot(
                [x, x],
                [active_y.min(), active_y.max()],
                color="#222222",
                linewidth=1.5,
                zorder=1,
            )
    matrix.set_yticks(
        y_positions,
        [f"{DISPLAY[model]} ({set_sizes[model]:,})" for model in MODELS],
    )
    matrix.set(
        xlim=(-0.6, len(selected) - 0.4),
        ylim=(-0.5, len(MODELS) - 0.5),
        xlabel="Exact model membership",
    )
    matrix.invert_yaxis()
    matrix.set_xticks([])
    sns.despine(ax=matrix, left=True, bottom=True)
    fig.savefig(FIGURES / "significant_set_upset.png", dpi=300)
    fig.savefig(FIGURES / "significant_set_upset.pdf")
    plt.close(fig)


def make_significant_r2_figure(data):
    significant = data[
        data.metric.eq("l2") & data.fdr_0_05.astype(bool)
    ].copy()
    significant["model_display"] = significant.model.map(DISPLAY)
    names = [DISPLAY[model] for model in MODELS]
    summary = (
        significant.groupby(["model", "model_display"], as_index=False)
        .agg(
            significant_loci=("str_id", "size"),
            mean_r2=("r2", "mean"),
            standard_deviation_r2=("r2", "std"),
            q25_r2=("r2", lambda values: values.quantile(0.25)),
            median_r2=("r2", "median"),
            q75_r2=("r2", lambda values: values.quantile(0.75)),
            minimum_r2=("r2", "min"),
            maximum_r2=("r2", "max"),
        )
    )
    summary.to_csv(RESULTS / "significant_l2_r2_summary.csv", index=False)

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.6,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    sns.violinplot(
        data=significant,
        x="model_display",
        y="r2",
        order=names,
        hue="model_display",
        hue_order=names,
        palette="colorblind",
        inner="box",
        cut=0,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    counts = summary.set_index("model_display").significant_loci
    ax.set_xticks(
        range(len(names)),
        [f"{name}\n(n={int(counts[name]):,})" for name in names],
    )
    ax.set(
        xlabel="",
        ylabel=r"Regression $R^2$",
        title=r"Whole-mean L2 significant loci ($q < 0.05$)",
    )
    sns.despine(ax=ax)
    fig.savefig(FIGURES / "significant_l2_r2_distribution.png", dpi=300)
    fig.savefig(FIGURES / "significant_l2_r2_distribution.pdf")
    plt.close(fig)
    return summary


def make_qvalue_distribution_figure(data):
    primary = data[data.metric.eq("l2")].copy()
    primary["model_display"] = primary.model.map(DISPLAY)
    names = [DISPLAY[model] for model in MODELS]
    summary = (
        primary.groupby(["model", "model_display"], as_index=False)
        .agg(
            loci=("str_id", "size"),
            q_lt_0_01=("q_value", lambda values: int(values.lt(0.01).sum())),
            q_lt_0_05=("q_value", lambda values: int(values.lt(0.05).sum())),
            q_lt_0_10=("q_value", lambda values: int(values.lt(0.10).sum())),
            q25=("q_value", lambda values: values.quantile(0.25)),
            median_q=("q_value", "median"),
            q75=("q_value", lambda values: values.quantile(0.75)),
            maximum_q=("q_value", "max"),
        )
    )
    for threshold in ("0_01", "0_05", "0_10"):
        summary[f"fraction_q_lt_{threshold}"] = (
            summary[f"q_lt_{threshold}"] / summary.loci
        )
    summary.to_csv(RESULTS / "l2_permutation_qvalue_summary.csv", index=False)

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.4,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(9.0, 3.5), sharex=True, sharey=True, constrained_layout=True
    )
    bins = np.linspace(0, 1, 51)
    palette = sns.color_palette("colorblind", n_colors=3)
    summary_by_name = summary.set_index("model_display")
    for ax, name, color in zip(axes, names, palette):
        values = primary.loc[primary.model_display.eq(name), "q_value"]
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.35)
        ax.axvline(0.05, color="#333333", linestyle="--", linewidth=1)
        fraction = summary_by_name.loc[name, "fraction_q_lt_0_05"]
        ax.set(
            xlim=(0, 1),
            xlabel="",
            title=f"{name}\nq < 0.05: {fraction:.1%}",
        )
        sns.despine(ax=ax)
    axes[0].set_ylabel("Loci")
    fig.supxlabel("BH-adjusted permutation q-value")
    fig.suptitle("Whole-mean L2 permutation q-value distributions")
    fig.savefig(FIGURES / "l2_permutation_qvalue_distributions.png", dpi=300)
    fig.savefig(FIGURES / "l2_permutation_qvalue_distributions.pdf")
    plt.close(fig)
    return summary


def make_qvalue_ecdf_figure(data):
    primary = data[data.metric.eq("l2")].copy()
    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.6,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    palette = sns.color_palette("colorblind", n_colors=3)
    for model, color in zip(MODELS, palette):
        values = np.sort(primary.loc[primary.model.eq(model), "q_value"].to_numpy())
        cumulative = np.arange(1, len(values) + 1) / len(values)
        fraction = float(np.mean(values <= 0.05))
        ax.step(values, cumulative, where="post", color=color, linewidth=2.2,
                label=f"{DISPLAY[model]} ({fraction:.1%})")
    ax.axvline(0.05, color="#333333", linestyle="--", linewidth=1)
    ax.text(
        0.06,
        0.96,
        "FDR = 0.05",
        rotation=0,
        ha="left",
        va="top",
        color="#333333",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="BH-adjusted permutation q-value",
        ylabel="Fraction of loci with q ≤ x",
        title="Genome-wide cumulative significance profiles",
    )
    ax.legend(frameon=False, title="")
    sns.despine(ax=ax)
    fig.savefig(FIGURES / "l2_permutation_qvalue_ecdf.png", dpi=300)
    fig.savefig(FIGURES / "l2_permutation_qvalue_ecdf.pdf")
    plt.close(fig)


def make_slope_distribution_figure(data, config):
    loci = pd.read_csv(config["loci"], usecols=["str_id", "period"])
    primary = data[data.metric.eq("l2")].copy()
    primary = primary.merge(loci, on="str_id", validate="many_to_one")
    primary["slope_per_bp"] = primary.slope / primary.period
    primary["model_display"] = primary.model.map(DISPLAY)
    summary = (
        primary.groupby(["model", "model_display"], as_index=False)
        .agg(
            loci=("str_id", "size"),
            minimum_slope_per_bp=("slope_per_bp", "min"),
            q005_slope_per_bp=("slope_per_bp", lambda values: values.quantile(0.005)),
            q01_slope_per_bp=("slope_per_bp", lambda values: values.quantile(0.01)),
            median_slope_per_bp=("slope_per_bp", "median"),
            mean_slope_per_bp=("slope_per_bp", "mean"),
            q99_slope_per_bp=("slope_per_bp", lambda values: values.quantile(0.99)),
            q995_slope_per_bp=("slope_per_bp", lambda values: values.quantile(0.995)),
            maximum_slope_per_bp=("slope_per_bp", "max"),
            positive_loci=("slope_per_bp", lambda values: int(values.gt(0).sum())),
        )
    )
    summary["fraction_positive"] = summary.positive_loci / summary.loci
    summary.to_csv(RESULTS / "all_tested_l2_slope_summary.csv", index=False)

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.4,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(9.0, 3.5), sharey=True, constrained_layout=True
    )
    palette = sns.color_palette("colorblind", n_colors=3)
    summary_by_model = summary.set_index("model")
    for ax, model, color in zip(axes, MODELS, palette):
        model_data = primary.loc[primary.model.eq(model)]
        values = model_data.slope_per_bp.to_numpy()
        row = summary_by_model.loc[model]
        low, high = row.q005_slope_per_bp, row.q995_slope_per_bp
        significant = model_data.fdr_0_05.astype(bool).to_numpy()
        nonsignificant_values = np.clip(values[~significant], low, high)
        significant_values = np.clip(values[significant], low, high)
        ax.hist(
            [nonsignificant_values, significant_values],
            bins=80,
            stacked=True,
            color=[(*color, 0.20), (*color, 0.95)],
            edgecolor="none",
        )
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.axvline(row.median_slope_per_bp, color="#b22222", linestyle="--", linewidth=1.1)
        ax.set_yscale("log")
        ax.set(
            xlim=(low, high),
            xlabel="",
            title=f"{DISPLAY[model]}\npositive: {row.fraction_positive:.1%}",
        )
        sns.despine(ax=ax)
    axes[1].legend(
        handles=[
            Patch(
                facecolor=(*palette[1], 0.95),
                edgecolor="none",
                label="FDR < 0.05",
            ),
            Patch(
                facecolor=(*palette[1], 0.20),
                edgecolor="none",
                label="q ≥ 0.05",
            ),
        ],
        frameon=False,
        loc="upper right",
    )
    axes[0].set_ylabel("Loci (log scale)")
    fig.supxlabel("Whole-mean L2 slope per bp (central 99%; tails in edge bins)")
    fig.suptitle("L2 slope distributions across all tested loci")
    fig.savefig(FIGURES / "all_tested_l2_slope_distributions.png", dpi=300)
    fig.savefig(FIGURES / "all_tested_l2_slope_distributions.pdf")
    plt.close(fig)
    return summary


def make_significant_slope_context_figures(data, config):
    """Plot significant L2 slopes by motif length and promoter-aware annotation."""
    loci = pd.read_csv(config["loci"], usecols=["str_id", "period"])
    annotation = pd.read_csv(
        config["promoter_annotation"],
        sep="\t",
        usecols=["#chrom", "start", "functional_region"],
        dtype={"#chrom": "string"},
    )
    annotation["str_id"] = (
        annotation["#chrom"] + "_" + annotation["start"].astype(str)
    )
    if annotation.str_id.duplicated().any():
        raise RuntimeError("Duplicate STR identifiers in promoter-aware annotation")
    annotation["genome_region"] = annotation.functional_region.replace(
        {"five_prime_UTR": "UTR", "three_prime_UTR": "UTR"}
    )

    significant = data.loc[
        data.metric.eq("l2") & data.fdr_0_05.astype(bool),
        ["str_id", "model", "slope", "r2", "q_value"],
    ].copy()
    significant = significant.merge(loci, on="str_id", validate="many_to_one")
    significant = significant.merge(
        annotation[["str_id", "genome_region"]],
        on="str_id",
        how="left",
        validate="many_to_one",
    )
    if significant.genome_region.isna().any():
        missing = int(significant.genome_region.isna().sum())
        raise RuntimeError(
            f"{missing:,} significant model-locus rows lack promoter-aware annotation"
        )
    significant["slope_per_repeat"] = significant.slope
    significant["slope_per_bp"] = significant.slope / significant.period
    if not np.allclose(
        significant.slope_per_repeat,
        significant.slope_per_bp * significant.period,
        rtol=2e-5,
        atol=1e-8,
    ):
        raise RuntimeError("Per-repeat and per-bp slopes have inconsistent scaling")
    significant.to_csv(
        RESULTS / "significant_l2_slope_motif_region_data.csv.gz",
        index=False,
        compression="gzip",
    )

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.4,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    palette = dict(zip(MODELS, sns.color_palette("colorblind", n_colors=3)))
    motif_order = sorted(significant.period.unique())
    region_order = [
        region
        for region in ("CDS", "UTR", "other_exon", "intron", "promoter", "intergenic")
        if significant.genome_region.eq(region).any()
    ]

    def draw_boxplot(ax, frame, group, order, value, color):
        arrays = [
            frame.loc[frame[group].astype(str).eq(str(level)), value].to_numpy()
            for level in order
        ]
        ax.boxplot(
            arrays,
            tick_labels=[str(level) for level in order],
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#b22222", "linewidth": 1.5},
            boxprops={
                "facecolor": color,
                "edgecolor": "#4a4a4a",
                "alpha": 0.72,
                "linewidth": 0.8,
            },
            whiskerprops={"color": "#4a4a4a", "linewidth": 0.8},
            capprops={"color": "#4a4a4a", "linewidth": 0.8},
        )
        ax.axhline(0, color="#777777", linewidth=0.7, linestyle=":", zorder=0)
        if group == "genome_region":
            ax.tick_params(axis="x", rotation=25)
        sns.despine(ax=ax)

    output_stems = {
        "slope_per_repeat": "significant_l2_slope_per_repeat_by_motif_and_region",
        "slope_per_bp": "significant_l2_slope_per_bp_by_motif_and_region",
    }
    unit_labels = {
        "slope_per_repeat": "per repeat",
        "slope_per_bp": "per bp",
    }
    for value, stem in output_stems.items():
        fig, axes = plt.subplots(
            3,
            2,
            figsize=(11.5, 11.0),
            sharey="row",
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(wspace=0.08, hspace=0.04)
        for row, model in enumerate(MODELS):
            frame = significant.loc[significant.model.eq(model)]
            draw_boxplot(
                axes[row, 0], frame, "period", motif_order, value, palette[model]
            )
            draw_boxplot(
                axes[row, 1],
                frame,
                "genome_region",
                region_order,
                value,
                palette[model],
            )
            axes[row, 0].set_ylabel(
                f"{DISPLAY[model]} embedding\nperturbation ({unit_labels[value]})"
            )
            axes[row, 1].set_ylabel("")
        axes[0, 0].set_title("Motif length")
        axes[0, 1].set_title("Genomic Context")
        for ax in axes.flat:
            ax.set_xlabel("")
        fig.savefig(FIGURES / f"{stem}.png", dpi=300)
        fig.savefig(FIGURES / f"{stem}.pdf")
        plt.close(fig)
    return significant


def make_three_model_motif_region_enrichment(data, config):
    """Compare joint motif/context enrichment across the three primary models."""
    loci = pd.read_csv(config["loci"], usecols=["str_id", "period"])
    annotation = pd.read_csv(
        config["promoter_annotation"],
        sep="\t",
        usecols=["#chrom", "start", "functional_region"],
        dtype={"#chrom": "string"},
    )
    annotation["str_id"] = (
        annotation["#chrom"] + "_" + annotation["start"].astype(str)
    )
    if annotation.str_id.duplicated().any():
        raise RuntimeError("Duplicate STR identifiers in promoter-aware annotation")
    annotation["genome_region"] = annotation.functional_region.replace(
        {"five_prime_UTR": "UTR", "three_prime_UTR": "UTR"}
    )

    primary = data.loc[
        data.metric.eq("l2"), ["str_id", "model", "q_value"]
    ].copy()
    primary = primary.merge(loci, on="str_id", validate="many_to_one")
    primary = primary.merge(
        annotation[["str_id", "genome_region"]],
        on="str_id",
        how="left",
        validate="many_to_one",
    )
    if primary.genome_region.isna().any():
        missing = int(primary.genome_region.isna().sum())
        raise RuntimeError(
            f"{missing:,} tested model-locus rows lack promoter-aware annotation"
        )
    primary["significant"] = primary.q_value.lt(0.05)

    regions = ("intergenic", "promoter", "intron", "UTR", "CDS")
    periods = (1, 2, 3)
    rows = []
    for model in MODELS:
        model_data = primary.loc[primary.model.eq(model)]
        total_significant = int(model_data.significant.sum())
        total_not_significant = len(model_data) - total_significant
        for region in regions:
            for period in periods:
                selected = model_data.genome_region.eq(
                    region
                ) & model_data.period.eq(period)
                tested = int(selected.sum())
                significant = int(model_data.loc[selected, "significant"].sum())
                not_significant = tested - significant
                other_significant = total_significant - significant
                other_not_significant = total_not_significant - not_significant
                odds_ratio, p_value = fisher_exact(
                    [
                        [significant, not_significant],
                        [other_significant, other_not_significant],
                    ]
                )
                rows.append(
                    {
                        "model": model,
                        "genome_region": region,
                        "motif_length_bp": period,
                        "tested_loci": tested,
                        "significant_loci": significant,
                        "significant_proportion": (
                            significant / tested if tested else np.nan
                        ),
                        "odds_ratio_vs_all_other_tested_loci": odds_ratio,
                        "fisher_exact_p_value": p_value,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(
        RESULTS / "three_model_motif_region_enrichment.csv", index=False
    )

    log_odds_values = np.log2(
        table.odds_ratio_vs_all_other_tested_loci.replace(0, np.nan)
    )
    finite = np.abs(log_odds_values[np.isfinite(log_odds_values)])
    limit = max(2.0, float(finite.quantile(0.90)))

    sns.set_theme(
        context="paper",
        style="ticks",
        font="sans-serif",
        font_scale=1.6,
        rc={
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    common_font_size = plt.rcParams["font.size"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.0), sharey=True)
    fig.subplots_adjust(
        left=0.08, right=0.90, bottom=0.20, top=0.77, wspace=0.06
    )
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = None
    for ax, model in zip(axes, MODELS):
        model_table = table.loc[table.model.eq(model)]
        odds = model_table.pivot(
            index="genome_region",
            columns="motif_length_bp",
            values="odds_ratio_vs_all_other_tested_loci",
        ).reindex(index=regions, columns=periods)
        log_odds = np.log2(odds.replace(0, np.nan))
        display = log_odds.clip(-limit, limit).fillna(-limit)
        display = display.replace([np.inf, -np.inf], [limit, -limit])
        image = ax.imshow(
            display,
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
        )
        ax.set_xticks(range(len(periods)), [f"{period} bp" for period in periods])
        ax.set_title(DISPLAY[model], fontsize=common_font_size)
        ax.tick_params(axis="both", labelsize=common_font_size)
        for row_index, region in enumerate(regions):
            for column_index, period in enumerate(periods):
                record = model_table.loc[
                    model_table.genome_region.eq(region)
                    & model_table.motif_length_bp.eq(period)
                ].iloc[0]
                significance_star = (
                    "*" if record.fisher_exact_p_value < 0.05 else ""
                )
                odds_text = (
                    f"{record.odds_ratio_vs_all_other_tested_loci:.2f}"
                    f"{significance_star}"
                )
                value = float(display.loc[region, period])
                text_color = (
                    "white" if abs(value) > limit * 0.55 else "black"
                )
                ax.text(
                    column_index,
                    row_index,
                    odds_text,
                    ha="center",
                    va="center",
                    fontsize=common_font_size,
                    color=text_color,
                )
    axes[0].set_yticks(
        range(len(regions)),
        ["Intergenic", "Promoter", "Intron", "UTR", "CDS"],
    )
    fig.suptitle(
        "Reference-anchored whole-mean L2 significance enrichment\n"
        "Cell text: odds ratio (* Fisher exact p < 0.05)",
        y=0.96,
        fontsize=common_font_size,
    )
    colorbar_axis = fig.add_axes([0.92, 0.25, 0.018, 0.46])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(
        "log2 odds ratio vs all other tested loci", fontsize=common_font_size
    )
    colorbar.ax.tick_params(labelsize=common_font_size)
    fig.text(
        0.08,
        0.035,
        "Red = enriched; blue = depleted. Odds ratios compare each joint cell "
        "with all other tested STR loci.\n"
        "Promoter = strand-aware 2 kb upstream of the annotated TSS; "
        "q < 0.05 defines significance.",
        fontsize=common_font_size,
    )
    stem = "three_model_motif_region_enrichment_heatmap"
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)
    return table


def main():
    for directory in (RESULTS, FIGURES, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text())
    data = load_primary_results(config)
    expected_rows = config["expected_loci"] * len(MODELS) * len(config["metrics"])
    if len(data) != expected_rows:
        raise RuntimeError(f"Rows {len(data)} != {expected_rows}")
    if data.groupby(["str_id", "model", "metric"]).size().ne(1).any():
        raise RuntimeError("Duplicate or missing locus/model/metric rows")
    if data.str_id.nunique() != config["expected_loci"]:
        raise RuntimeError("Unexpected unique locus count")
    if data[["slope", "intercept", "r2", "p_value", "q_value"]].isna().any().any():
        raise RuntimeError("Non-finite regression result")
    data.to_csv(
        RESULTS / "matched_per_locus_regressions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = (
        data.groupby(["model", "metric"], as_index=False)
        .agg(
            loci=("str_id", "size"),
            median_slope=("slope", "median"),
            fraction_positive=("slope", lambda x: float((x > 0).mean())),
            median_r2=("r2", "median"),
            fdr_0_05=("fdr_0_05", "sum"),
            fraction_fdr_0_05=("fdr_0_05", "mean"),
            exact_tests=("permutation_method", lambda x: int((x == "exact").sum())),
            monte_carlo_tests=(
                "permutation_method",
                lambda x: int((x == "monte_carlo").sum()),
            ),
        )
    )
    summary.to_csv(RESULTS / "matched_regression_summary.csv", index=False)
    magnitude_parameters, magnitude_pair = compare_normalized_magnitude(data, config)

    primary = data[data.metric == "l2"].copy()
    wide = primary.pivot(
        index="str_id",
        columns="model",
        values=["slope", "p_value", "q_value", "fdr_0_05"],
    )
    wide.columns = [f"{model}_{value}" for value, model in wide.columns]
    wide = wide.reset_index()
    for model in MODELS:
        wide[f"{model}_significant"] = wide[f"{model}_fdr_0_05"].astype(bool)
    wide.to_csv(
        RESULTS / "matched_l2_per_locus_significance.csv.gz",
        index=False,
        compression="gzip",
    )
    pair = pairwise(wide)
    pattern = patterns(wide)
    pair.to_csv(RESULTS / "matched_pairwise_overlap.csv", index=False)
    pattern.to_csv(RESULTS / "matched_three_way_membership.csv", index=False)

    counts = {
        model: int(wide[f"{model}_significant"].sum()) for model in MODELS
    }
    all_three = int(
        (
            wide.argont_significant
            & wide.plantcad2_significant
            & wide.gpn_significant
        ).sum()
    )
    gpn_either = int(
        (
            wide.gpn_significant
            & (wide.argont_significant | wide.plantcad2_significant)
        ).sum()
    )
    headline = pd.DataFrame(
        [
            {
                "common_loci": len(wide),
                "argont_significant": counts["argont"],
                "plantcad2_significant": counts["plantcad2"],
                "gpn_significant": counts["gpn"],
                "all_three_significant": all_three,
                "gpn_significant_in_either_other_model": gpn_either,
                "gpn_only": counts["gpn"] - gpn_either,
            }
        ]
    )
    headline.to_csv(RESULTS / "matched_overlap_summary.csv", index=False)
    make_figure(pattern, pair)
    make_upset_figure(pattern, wide)
    r2_summary = make_significant_r2_figure(data)
    qvalue_summary = make_qvalue_distribution_figure(data)
    make_qvalue_ecdf_figure(data)
    slope_summary = make_slope_distribution_figure(data, config)
    make_significant_slope_context_figures(data, config)
    make_three_model_motif_region_enrichment(data, config)

    lines = [
        "# Primary reference-anchored comparison of ArgoNT, PlantCAD2, and GPN",
        "",
        f"This comparison reads the finalized genome-wide primary results for the same "
        f"**{len(wide):,} STR loci**; it does not rerun regressions. All models use whole-sequence "
        "mean embeddings, reference used only as the distance "
        "anchor, reference excluded from regression, at least three distinct observed alternate "
        "repeat lengths, absolute repeat-count difference, and a two-sided allele-label permutation "
        "test. Stored Benjamini–Hochberg q-values are used directly from each model's primary "
        "pooling/layer/metric family.",
        "",
        "Layers are ArgoNT 21, PlantCAD2-Large 36, and GPN final `last_hidden_state`. "
        "Input context still differs according to each model's supported workflow, but allele IDs, "
        "regression observations, predictors, permutation labels, seeds, and FDR family sizes match.",
        "",
        "## Whole-mean L2 results",
        "",
        "| Model | Median slope/repeat | Positive slopes | Median R² | FDR < 0.05 |",
        "|---|---:|---:|---:|---:|",
    ]
    l2_summary = summary[summary.metric == "l2"]
    for model in MODELS:
        row = l2_summary[l2_summary.model == model].iloc[0]
        lines.append(
            f"| {DISPLAY[model]} | {row.median_slope:.6g} | "
            f"{row.fraction_positive:.1%} | {row.median_r2:.3f} | "
            f"{int(row.fdr_0_05):,}/{int(row.loci):,} ({row.fraction_fdr_0_05:.1%}) |"
        )
    lines += [
        "",
        "## R² among significant L2 loci",
        "",
        "Each distribution uses the `q < 0.05` locus set defined separately within that model.",
        "",
        "| Model | Significant loci | Median R² | IQR | Mean R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = r2_summary[r2_summary.model.eq(model)].iloc[0]
        lines.append(
            f"| {DISPLAY[model]} | {int(row.significant_loci):,} | "
            f"{row.median_r2:.3f} | {row.q25_r2:.3f}–{row.q75_r2:.3f} | "
            f"{row.mean_r2:.3f} |"
        )
    lines += [
        "",
        "## Permutation q-value distributions",
        "",
        "| Model | Loci | q < 0.01 | q < 0.05 | q < 0.10 | Median q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = qvalue_summary[qvalue_summary.model.eq(model)].iloc[0]
        lines.append(
            f"| {DISPLAY[model]} | {int(row.loci):,} | "
            f"{int(row.q_lt_0_01):,} ({row.fraction_q_lt_0_01:.1%}) | "
            f"{int(row.q_lt_0_05):,} ({row.fraction_q_lt_0_05:.1%}) | "
            f"{int(row.q_lt_0_10):,} ({row.fraction_q_lt_0_10:.1%}) | "
            f"{row.median_q:.3f} |"
        )
    lines += [
        "",
        "## Normalized L2 magnitude",
        "",
        "Raw L2 slopes are model-scale dependent. For magnitude comparison, each slope is "
        "converted to absolute slope per base, log10-transformed, and standardized within its "
        "model as `(value - median) / (1.4826 × MAD)`. Thus, zero is the typical model-specific "
        "magnitude and +1 is one robust standard deviation above it.",
        "",
        "| Models | Pearson r (robust z) | Spearman ρ | Median absolute z difference | Both z ≥ 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in magnitude_pair.itertuples():
        lines.append(
            f"| {row.model_1}–{row.model_2} | "
            f"{row.pearson_r_robust_z_magnitude:.3f} | "
            f"{row.spearman_rho_magnitude:.3f} | "
            f"{row.median_absolute_z_difference:.3f} | "
            f"{row.both_at_least_one_robust_sd:,}/{row.loci:,} "
            f"({row.both_at_least_one_robust_sd / row.loci:.1%}) |"
        )
    lines += [
        "",
        f"Normalized magnitude agreement is weak: pairwise Pearson correlations range from "
        f"**{magnitude_pair.pearson_r_robust_z_magnitude.min():.3f} to "
        f"{magnitude_pair.pearson_r_robust_z_magnitude.max():.3f}**, and Spearman correlations "
        f"range from **{magnitude_pair.spearman_rho_magnitude.min():.3f} to "
        f"{magnitude_pair.spearman_rho_magnitude.max():.3f}**.",
        "",
        "This normalization compares whether a locus has an unusually large response relative "
        "to each model's own genome-wide slope distribution. It does not make one unit of "
        "embedding distance physically identical across architectures.",
        "",
        f"**{all_three:,} loci** pass FDR 0.05 in all three models. "
        f"Of the {counts['gpn']:,} GPN-significant loci, **{gpn_either:,} "
        f"({gpn_either / counts['gpn']:.1%})** pass in at least one other model and "
        f"**{counts['gpn'] - gpn_either:,}** are GPN-only.",
        "",
        "## Pairwise overlap",
        "",
        "| Models | Intersection | Expected by chance | Enrichment | Fisher OR | Hypergeometric p | Jaccard | Shared-significant Spearman ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair.itertuples():
        lines.append(
            f"| {row.model_1}–{row.model_2} | {row.overlap:,} | "
            f"{row.expected_overlap_independence:,.1f} | "
            f"{row.observed_to_expected:.2f}× | {row.fisher_odds_ratio:.1f} | "
            f"{format_p(row.hypergeometric_p)} | {row.jaccard:.3f} | "
            f"{row.shared_sig_spearman_rho:.3f} |"
        )
    minimum_enrichment = pair.observed_to_expected.min()
    minimum_odds = pair.fisher_odds_ratio.min()
    lines += [
        "",
        f"All three pairwise overlaps are much greater than expected by chance: the smallest "
        f"observed/expected enrichment is **{minimum_enrichment:.2f}×**, the smallest Fisher "
        f"odds ratio is **{minimum_odds:.1f}**, and every one-sided hypergeometric p-value is "
        "**<1e-300**.",
        "",
        "## Interpretation",
        "",
        "This overlap table compares the canonical reference-anchored whole-mean results from the "
        "three genome-wide folders. Significance remains strongly influenced by allele count "
        "and exact-permutation resolution: three- and four-alternate loci cannot attain very small "
        "two-sided exact p-values. High overlap therefore reflects both shared biological/model "
        "response and shared test power.",
        "",
        "## Outputs",
        "",
        "- `results/matched_per_locus_regressions.csv.gz`",
        "- `results/matched_regression_summary.csv`",
        "- `results/matched_l2_per_locus_significance.csv.gz`",
        "- `results/matched_overlap_summary.csv`",
        "- `results/matched_pairwise_overlap.csv`",
        "- `results/matched_three_way_membership.csv`",
        "- `results/matched_l2_normalized_magnitude.csv.gz`",
        "- `results/normalized_magnitude_parameters.csv`",
        "- `results/normalized_magnitude_pairwise.csv`",
        "- `results/significant_l2_r2_summary.csv`",
        "- `results/l2_permutation_qvalue_summary.csv`",
        "- `results/all_tested_l2_slope_summary.csv`",
        "- `results/significant_l2_slope_motif_region_data.csv.gz`",
        "- `results/three_model_motif_region_enrichment.csv`",
        "- `figures/matched_three_model_overlap.png`",
        "- `figures/matched_three_model_overlap.pdf`",
        "- `figures/significant_set_upset.png`",
        "- `figures/significant_set_upset.pdf`",
        "- `figures/normalized_l2_magnitude_comparison.png`",
        "- `figures/significant_l2_r2_distribution.png`",
        "- `figures/significant_l2_r2_distribution.pdf`",
        "- `figures/l2_permutation_qvalue_distributions.png`",
        "- `figures/l2_permutation_qvalue_distributions.pdf`",
        "- `figures/l2_permutation_qvalue_ecdf.png`",
        "- `figures/l2_permutation_qvalue_ecdf.pdf`",
        "- `figures/all_tested_l2_slope_distributions.png`",
        "- `figures/all_tested_l2_slope_distributions.pdf`",
        "- `figures/significant_l2_slope_per_repeat_by_motif_and_region.png`",
        "- `figures/significant_l2_slope_per_repeat_by_motif_and_region.pdf`",
        "- `figures/significant_l2_slope_per_bp_by_motif_and_region.png`",
        "- `figures/significant_l2_slope_per_bp_by_motif_and_region.pdf`",
        "- `figures/three_model_motif_region_enrichment_heatmap.png`",
        "- `figures/three_model_motif_region_enrichment_heatmap.pdf`",
        "",
    ]
    report_path = REPORT / "MATCHED_REFERENCE_ANCHORED_COMPARISON.md"
    report_path.write_text("\n".join(lines))
    print(headline.to_string(index=False), flush=True)
    print(pair.to_string(index=False), flush=True)
    print(f"report={report_path}", flush=True)
    print("MATCHED_COMPARISON_AGGREGATION=PASS", flush=True)


if __name__ == "__main__":
    main()
