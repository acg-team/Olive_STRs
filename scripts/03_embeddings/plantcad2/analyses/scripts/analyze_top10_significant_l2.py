#!/usr/bin/env python3
"""Characterize top whole-mean L2 slopes among significant PlantCAD2 layer-36 loci."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
ANALYSES = ROOT / "analyses"
CATALOG = ROOT.parent / "agront"
ANNOTATED = ANALYSES / "results/layer36_whole_mean_l2_fdr05_annotated.csv.gz"
REGRESSIONS = ROOT / "results/per_locus_regressions.csv.gz"
LOCI = CATALOG / "inputs/loci.csv.gz"
ALLELES = CATALOG / "inputs/alleles.csv.gz"
SEED = 20260724


def bh(values):
    values = np.asarray(values, float); order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(len(values)); output[order] = np.minimum(adjusted, 1)
    return output


def canonical_motif(motif):
    motif = str(motif).upper()
    complement = str.maketrans("ACGTN", "TGCAN")
    reverse = motif.translate(complement)[::-1]
    candidates = [motif[i:] + motif[:i] for i in range(len(motif))]
    candidates += [reverse[i:] + reverse[:i] for i in range(len(reverse))]
    return min(candidates)


def entropy(sequence, k=1):
    sequence = str(sequence).upper()
    words = [sequence[i:i+k] for i in range(len(sequence) - k + 1)
             if set(sequence[i:i+k]) <= set("ACGT")]
    if not words:
        return np.nan
    counts = np.asarray(list(Counter(words).values()), float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / (2 * k))


def sequence_features(row):
    left, right, motif = str(row.left_flank).upper(), str(row.right_flank).upper(), str(row.motif).upper()
    flank = left + right
    valid = sum(flank.count(base) for base in "ACGT")
    return pd.Series({
        "motif_gc": (motif.count("G") + motif.count("C")) / len(motif),
        "flank_gc": (flank.count("G") + flank.count("C")) / valid if valid else np.nan,
        "flank_n_fraction": 1 - valid / len(flank),
        "flank_base_entropy": entropy(flank, 1),
        "flank_dinucleotide_entropy": entropy(flank, 2),
    })


def continuous_comparisons(data, features):
    rows = []
    for feature in features:
        top = data.loc[data.top10, feature].dropna().to_numpy()
        other = data.loc[~data.top10, feature].dropna().to_numpy()
        test = mannwhitneyu(top, other, alternative="two-sided")
        effect = 2 * test.statistic / (len(top) * len(other)) - 1
        rho, rho_p = spearmanr(data[feature], data.slope_per_bp, nan_policy="omit")
        rows.append({"feature": feature, "n_top": len(top), "n_other": len(other),
                     "top_median": np.median(top), "other_median": np.median(other),
                     "median_difference": np.median(top) - np.median(other),
                     "rank_biserial_top_vs_other": effect, "p_value": test.pvalue,
                     "spearman_with_slope": rho, "spearman_p_value": rho_p})
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result.p_value)
    result["spearman_q_value"] = bh(result.spearman_p_value)
    return result.sort_values("rank_biserial_top_vs_other", key=abs, ascending=False)


def categorical_comparisons(data, factors):
    omnibus, levels = [], []
    overall = data.top10.mean()
    for factor in factors:
        table = pd.crosstab(data[factor].astype(str), data.top10)
        for value in (False, True):
            if value not in table: table[value] = 0
        table = table.reindex(columns=[False, True], fill_value=0)
        chi2, p, _, _ = chi2_contingency(table.to_numpy())
        n = table.values.sum(); k = min(table.shape) - 1
        omnibus.append({"factor": factor, "levels": len(table), "chi_square": chi2,
                        "p_value": p, "cramers_v": np.sqrt(chi2 / (n * k)) if k else np.nan})
        for level, counts in table.iterrows():
            top, other = int(counts[True]), int(counts[False])
            top_else = int(data.top10.sum()) - top
            other_else = int((~data.top10).sum()) - other
            odds, fp = fisher_exact([[top, other], [top_else, other_else]])
            total = top + other
            levels.append({"factor": factor, "level": level, "n_loci": total,
                           "top10_loci": top, "top10_fraction": top / total,
                           "enrichment_overall": (top / total) / overall,
                           "odds_ratio": odds, "p_value": fp})
    omnibus = pd.DataFrame(omnibus); omnibus["q_value"] = bh(omnibus.p_value)
    levels = pd.DataFrame(levels)
    levels["q_value_within_factor"] = levels.groupby("factor").p_value.transform(lambda x: bh(x.to_numpy()))
    return omnibus, levels


def build_model(data, numeric, categorical):
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([("features", transformer),
                         ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))])
    x, y = data[numeric + categorical], data.top10.astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, x, y, scoring="roc_auc", cv=cv, n_jobs=1)
    pipeline.fit(x, y)
    names = pipeline.named_steps["features"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef = pd.DataFrame({"feature": names, "coefficient": coefficients,
                         "odds_ratio": np.exp(coefficients)}).sort_values("coefficient", key=abs, ascending=False)
    return scores, coef


def load_context_slopes(significant_ids):
    frames = []
    for chunk in pd.read_csv(REGRESSIONS, chunksize=250_000):
        keep = (chunk.str_id.isin(significant_ids) & chunk.metric.eq("l2") &
                chunk.predictor.eq("absolute_repeat_difference_from_reference"))
        keep &= ((chunk.pooling.eq("whole_mean")) |
                 (chunk.pooling.eq("str_mean") & chunk.layer.eq(36)))
        frames.append(chunk.loc[keep, ["str_id", "pooling", "layer", "slope", "r2", "q_value"]])
    context = pd.concat(frames, ignore_index=True)
    periods = pd.read_csv(LOCI, usecols=["str_id", "period"])
    context = context.merge(periods, on="str_id", validate="many_to_one")
    context["slope"] = context.slope / context.period
    return context.drop(columns="period")


def context_summary(context, membership, primary):
    context = context.merge(membership, on="str_id", validate="many_to_one")
    rows = []
    primary_map = primary.set_index("str_id").slope_per_bp
    for (pooling, layer), group in context.groupby(["pooling", "layer"]):
        top = group.loc[group.top10, "slope"].to_numpy(); other = group.loc[~group.top10, "slope"].to_numpy()
        test = mannwhitneyu(top, other, alternative="two-sided")
        aligned = group.set_index("str_id").slope.reindex(primary_map.index)
        rho, p = spearmanr(primary_map, aligned, nan_policy="omit")
        rows.append({"pooling": pooling, "layer": layer, "n_loci": len(group),
                     "top_median_slope_per_bp": np.median(top), "other_median_slope_per_bp": np.median(other),
                     "top_to_other_median_ratio": np.median(top) / np.median(other),
                     "rank_biserial_top_vs_other": 2 * test.statistic / (len(top) * len(other)) - 1,
                     "p_value": test.pvalue, "spearman_with_layer36_whole_mean": rho,
                     "spearman_p_value": p})
    result = pd.DataFrame(rows).sort_values(["pooling", "layer"])
    result["q_value"] = bh(result.p_value)
    return result, context


def format_p(value):
    return "<1e-300" if value == 0 else f"{value:.3g}"


def main():
    result_dir = ANALYSES / "results"; figure_dir = ANALYSES / "figures"; report_dir = ANALYSES / "report"
    for directory in (result_dir, figure_dir, report_dir): directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(ANNOTATED)
    n_top = int(np.ceil(len(data) * .10))
    order = data.slope_per_bp.sort_values(ascending=False).index
    data["top10"] = False; data.loc[order[:n_top], "top10"] = True
    threshold = data.loc[data.top10, "slope_per_bp"].min()

    loci = pd.read_csv(LOCI)
    extra = loci[["str_id", "ref_repeats", "left_flank", "right_flank", "n_observed_lengths",
                  "n_embedded_lengths", "max_tokens"]]
    data = data.merge(extra, on="str_id", validate="one_to_one")
    seq = data[["left_flank", "right_flank", "motif"]].apply(sequence_features, axis=1)
    data = pd.concat([data, seq], axis=1)
    data["canonical_motif"] = data.motif.map(canonical_motif)
    data["reference_str_bp"] = data.ref_repeats * data.period

    alleles = pd.read_csv(ALLELES, usecols=["str_id", "allele_repeats", "observed", "is_reference"])
    alleles = alleles[(alleles.observed.eq(1)) & (alleles.is_reference.eq(0)) &
                      alleles.str_id.isin(set(data.str_id))]
    allele_summary = (alleles.groupby("str_id").allele_repeats
                      .agg(min_allele_repeats="min", max_allele_repeats="max", median_allele_repeats="median",
                           allele_repeat_sd="std").reset_index())
    data = data.merge(allele_summary, on="str_id", validate="one_to_one")
    data["allele_range_repeats"] = data.max_allele_repeats - data.min_allele_repeats
    data["allele_range_bp"] = data.allele_range_repeats * data.period
    data["min_allele_bp"] = data.min_allele_repeats * data.period
    data["median_allele_bp"] = data.median_allele_repeats * data.period
    data["max_allele_bp"] = data.max_allele_repeats * data.period
    data["reference_relative_position"] = ((data.ref_repeats - data.min_allele_repeats) /
                                           data.allele_range_repeats.replace(0, np.nan))

    continuous = ["period", "ref_repeats", "reference_str_bp", "min_allele_repeats", "median_allele_repeats",
                  "max_allele_repeats", "min_allele_bp", "median_allele_bp", "max_allele_bp", "n_alternate_alleles",
                  "allele_range_repeats", "allele_range_bp", "allele_repeat_sd", "reference_relative_position", "motif_gc",
                  "flank_gc", "flank_n_fraction", "flank_base_entropy", "flank_dinucleotide_entropy",
                  "r2", "q_value"]
    comparisons = continuous_comparisons(data, continuous)
    canonical_counts = data.canonical_motif.value_counts()
    data["canonical_motif_test"] = data.canonical_motif.where(data.canonical_motif.map(canonical_counts) >= 50, "other_rare")
    omnibus, levels = categorical_comparisons(data, ["period", "genome_region", "canonical_motif_test", "contig"])

    biological_numeric = ["median_allele_bp", "allele_range_bp", "n_alternate_alleles", "allele_repeat_sd",
                          "reference_relative_position", "motif_gc",
                          "flank_gc", "flank_n_fraction", "flank_base_entropy", "flank_dinucleotide_entropy"]
    categorical = ["period", "genome_region"]
    bio_scores, bio_coef = build_model(data, biological_numeric, categorical)
    diagnostic_scores, diagnostic_coef = build_model(data, biological_numeric + ["r2"], categorical)

    membership = data[["str_id", "top10"]]
    context = load_context_slopes(set(data.str_id))
    context_stats, context_rows = context_summary(context, membership, data[["str_id", "slope_per_bp"]])

    output_columns = [x for x in data.columns if x not in {"left_flank", "right_flank"}]
    data[output_columns].sort_values("slope_per_bp", ascending=False).to_csv(
        result_dir / "top10_significant_l2_loci_characteristics.csv.gz", index=False, compression="gzip")
    comparisons.to_csv(result_dir / "top10_continuous_feature_comparisons.csv", index=False)
    omnibus.to_csv(result_dir / "top10_categorical_omnibus_tests.csv", index=False)
    levels.to_csv(result_dir / "top10_categorical_enrichment.csv", index=False)
    bio_coef.to_csv(result_dir / "top10_biological_logistic_coefficients.csv", index=False)
    diagnostic_coef.to_csv(result_dir / "top10_diagnostic_logistic_coefficients.csv", index=False)
    context_stats.to_csv(result_dir / "top10_cross_layer_pooling_summary.csv", index=False)

    # Main descriptive figure.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].hist(data.slope_per_bp, bins=100, color="#4682a9")
    axes[0, 0].axvline(threshold, color="#b22222", linestyle="--", label=f"top 10% cutoff={threshold:.3f}")
    axes[0, 0].set(xlabel="Layer 36 whole-mean L2 slope per bp", ylabel="Loci", title="Top-decile definition"); axes[0, 0].legend()
    for ax, factor, title in [(axes[0, 1], "period", "Top-decile rate by motif length"),
                              (axes[0, 2], "genome_region", "Top-decile rate by genome region")]:
        rates = data.groupby(factor, observed=True).top10.agg(["mean", "size"])
        ax.bar(rates.index.astype(str), rates["mean"] * 100, color="#76a88c")
        ax.axhline(data.top10.mean() * 100, color="#555555", linestyle="--")
        ax.set(ylabel="Top-decile loci (%)", title=title); ax.tick_params(axis="x", rotation=25)
    for ax, feature, label in [(axes[1, 0], "min_allele_bp", "Minimum observed allele length (bp)"),
                               (axes[1, 1], "r2", "Regression $R^2$"),
                               (axes[1, 2], "flank_dinucleotide_entropy", "Flank dinucleotide entropy")]:
        arrays = [data.loc[~data.top10, feature].dropna(), data.loc[data.top10, feature].dropna()]
        ax.boxplot(arrays, tick_labels=["Other significant", "Top 10%"], showfliers=False, patch_artist=True,
                   medianprops={"color": "#b22222"}, boxprops={"facecolor": "#8fb7cc"})
        ax.set(ylabel=label, title=label); ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Characteristics of highest Layer 36 whole-mean L2 slopes", fontsize=16)
    fig.savefig(figure_dir / "top10_significant_l2_characteristics.png", dpi=180); plt.close(fig)

    # Cross-layer persistence figure.
    wm = context_stats[context_stats.pooling.eq("whole_mean")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(wm.layer, wm.top_median_slope_per_bp, marker="o", label="Top 10% at layer 36")
    axes[0].plot(wm.layer, wm.other_median_slope_per_bp, marker="o", label="Other significant")
    axes[0].set(xlabel="PlantCAD2-Large layer", ylabel="Median L2 slope per bp", title="Persistence across layers"); axes[0].legend()
    axes[1].plot(wm.layer, wm.spearman_with_layer36_whole_mean, marker="o", color="#7b5ea7")
    axes[1].set(xlabel="PlantCAD2-Large layer", ylabel="Spearman correlation", ylim=(0, 1.03), title="Correlation with Layer 36 slope")
    fig.savefig(figure_dir / "top10_l2_cross_layer_persistence.png", dpi=180); plt.close(fig)

    strongest = comparisons[~comparisons.feature.isin(["r2", "q_value", "period"])].iloc[0]
    motif_omni = omnibus[omnibus.factor.eq("period")].iloc[0]
    region_omni = omnibus[omnibus.factor.eq("genome_region")].iloc[0]
    str_mean = context_stats[(context_stats.pooling == "str_mean") & (context_stats.layer == 36)].iloc[0]
    layer12 = context_stats[(context_stats.pooling == "whole_mean") & (context_stats.layer == 12)].iloc[0]
    layer24 = context_stats[(context_stats.pooling == "whole_mean") & (context_stats.layer == 24)].iloc[0]
    layer48 = context_stats[(context_stats.pooling == "whole_mean") & (context_stats.layer == 48)].iloc[0]
    top_median = data.loc[data.top10, "slope_per_bp"].median(); other_median = data.loc[~data.top10, "slope_per_bp"].median()
    top_r2 = data.loc[data.top10, "r2"].median(); other_r2 = data.loc[~data.top10, "r2"].median()
    mono_fraction = data.loc[data.top10, "period"].eq(1).mean()
    motif_level = (levels[levels.factor == "canonical_motif_test"]
                   .sort_values("enrichment_overall", ascending=False).iloc[0])
    str_mean_direction = "higher" if str_mean.top_to_other_median_ratio > 1 else "lower"
    r2_direction = "more" if top_r2 > other_r2 else "less"

    lines = [
        "# Why do the highest significant Layer 36 whole-mean slopes differ?", "",
        f"The target group is the **{n_top:,} loci** with the largest reference-anchored Layer 36 whole-sequence mean L2 per-bp slopes among **{len(data):,} loci with q < 0.05**. The reference is the embedding-distance anchor and is excluded from regression. Exactly 10% rounded up were selected; the cutoff is **{threshold:.4f}**, the top-group median is **{top_median:.4f}**, and the other-significant median is **{other_median:.4f}**.", "",
        "## Main findings", "",
        f"- Motif length is associated with top-decile membership (Cramer's V={motif_omni.cramers_v:.3f}, p={format_p(motif_omni.p_value)}), while genome annotation is weaker (V={region_omni.cramers_v:.3f}, p={format_p(region_omni.p_value)}).",
        f"- The strongest non-test continuous association is `{strongest.feature}`: median {strongest.top_median:.4g} in the top group versus {strongest.other_median:.4g} in the others (rank-biserial={strongest.rank_biserial_top_vs_other:.3f}).",
        f"- **{mono_fraction:.1%}** of the top group are mononucleotide loci. The most enriched canonical motif class is `{motif_level.level}`: {int(motif_level.top10_loci)}/{int(motif_level.n_loci)} ({motif_level.top10_fraction:.1%}) enter the top group, or {motif_level.enrichment_overall:.2f}× the overall rate.",
        f"- The top group is slightly {r2_direction} linear across allele differences: median regression R² is **{top_r2:.3f}** versus **{other_r2:.3f}** in the other significant loci. The small difference should not be treated as an independent biological explanation.",
        f"- A logistic model using locus, allele, motif, flank, and region features has 5-fold CV ROC-AUC **{bio_scores.mean():.3f} ± {bio_scores.std():.3f}**. Adding regression R² changes it to **{diagnostic_scores.mean():.3f} ± {diagnostic_scores.std():.3f}**. This measures separability, not causality.",
        f"- Whole-mean slopes persist across layers: Spearman correlations with Layer 36 are {layer12.spearman_with_layer36_whole_mean:.3f} (layer 12), {layer24.spearman_with_layer36_whole_mean:.3f} (layer 24), and {layer48.spearman_with_layer36_whole_mean:.3f} (layer 48).",
        f"- Under STR-mean pooling at layer 36, the top group has a {str_mean.top_to_other_median_ratio:.2f}× median slope ({str_mean_direction} than the other significant loci), with rank-biserial effect {str_mean.rank_biserial_top_vs_other:.3f}.", "",
        "## Interpretation", "",
        "The results distinguish properties associated with high slopes from proof of mechanism. The cross-validated biological-feature model quantifies how well sequence and allele architecture separate the group. Regression R² is reported separately because it is a property of the same fitted response and is not an independent biological explanation.", "",
        "The cross-layer whole-mean results test whether the primary sequence-wide response is stable across model depth. The layer-36 STR-mean comparison is retained as a sensitivity analysis: agreement supports a tract-local component, whereas divergence indicates that pooling choice materially changes locus ranking.", "",
        "All comparisons are conditional on q < 0.05. Discovery depends on the number and spacing of non-reference allele lengths and permutation resolution, so conclusions cannot be generalized to every prepared STR locus.", "",
        "## Outputs", "",
        "- `results/top10_significant_l2_loci_characteristics.csv.gz`", "- `results/top10_continuous_feature_comparisons.csv`",
        "- `results/top10_categorical_omnibus_tests.csv`", "- `results/top10_categorical_enrichment.csv`",
        "- `results/top10_biological_logistic_coefficients.csv`", "- `results/top10_diagnostic_logistic_coefficients.csv`",
        "- `results/top10_cross_layer_pooling_summary.csv`", "- `figures/top10_significant_l2_characteristics.png`",
        "- `figures/top10_l2_cross_layer_persistence.png`", ""]
    report = report_dir / "TOP10_SIGNIFICANT_L2_CHARACTERIZATION.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report}")
    print("TOP10_SIGNIFICANT_L2_ANALYSIS=PASS")


if __name__ == "__main__":
    main()
