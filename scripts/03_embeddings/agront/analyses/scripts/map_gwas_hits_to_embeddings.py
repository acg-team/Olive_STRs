#!/usr/bin/env python3
"""Map oil-quality STR-GWAS discoveries into the ArgoNT embedding analysis."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
ANALYSES = ROOT / "analyses"
GWAS_ROOT = Path(os.environ.get("OLIVE_STR_ASSOCIATION_ROOT", "../../06_trait_association"))
GWAS = GWAS_ROOT / "results/fdr_0.05_followup_loci.tsv"
VARIANCE = Path(os.environ.get("OLIVE_STR_VARIANCE", "inputs/str_length_variance.csv"))
REGRESSIONS = ANALYSES / "results/layer21_whole_mean_regressions.csv.gz"
LOCI = ROOT / "inputs/loci.csv.gz"


def status(group):
    if group.passes_study_bonferroni.any(): return "study-wide Bonferroni"
    if group.passes_trait_bonferroni.any(): return "trait-wise Bonferroni"
    return "GWAS FDR only"


def effect_string(group):
    order = ["oleic_acid", "linoleic_acid", "log_oleic_linoleic_ratio"]
    rows = []
    for trait in order:
        part = group[group.trait.eq(trait)]
        if len(part):
            row = part.sort_values("pooled_p").iloc[0]
            rows.append(f"{trait}:{row.pooled_beta:.4g} (p={row.pooled_p:.3g}, FDR={row.pooled_fdr:.3g})")
    return "; ".join(rows)


def main():
    result_dir = ANALYSES / "results"; figure_dir = ANALYSES / "figures"; report_dir = ANALYSES / "report"
    for directory in (result_dir, figure_dir, report_dir): directory.mkdir(parents=True, exist_ok=True)

    gwas = pd.read_csv(GWAS, sep="\t")
    rows = []
    for locus, group in gwas.groupby("str_id", sort=False):
        best = group.sort_values("pooled_p").iloc[0]
        rows.append({
            "str_id": locus, "gwas_status": status(group), "traits": ",".join(sorted(group.trait.unique())),
            "trait_associations": len(group), "best_trait": best.trait, "best_pooled_beta": best.pooled_beta,
            "best_pooled_p": best.pooled_p, "best_pooled_fdr": best.pooled_fdr,
            "center_robust_all": bool(group.center_robust.all()), "chrom": str(best.chrom),
            "start": int(best.start), "end": int(best.end), "motif_length_gwas": int(best.motif_length),
            "motif_gwas": best.motif, "gwas_functional_region": best.functional_region,
            "nearest_gene_id": best.gene_id, "nearest_gene_name": best.gene_name,
            "distance_to_gene": best.distance_to_gene, "gwas_effects": effect_string(group),
        })
    hits = pd.DataFrame(rows)

    regression = pd.read_csv(REGRESSIONS)
    regression = regression[(regression.metric == "l2") &
                            (regression.predictor == "absolute_repeat_difference_from_reference")].copy()
    regression = regression.rename(columns={"slope": "embedding_l2_slope_per_repeat", "r2": "embedding_r2",
                                             "p_value": "embedding_p", "q_value": "embedding_q"})
    keep = ["str_id", "n_alternate_alleles", "n_unique_absolute_differences",
            "embedding_l2_slope_per_repeat", "embedding_r2",
            "embedding_p", "embedding_q", "permutation_method", "n_permutations"]
    regression = regression[keep]
    locus_periods = pd.read_csv(LOCI, usecols=["str_id", "period"])
    regression = regression.merge(locus_periods, on="str_id", validate="one_to_one")
    regression["embedding_l2_slope_per_bp"] = regression.embedding_l2_slope_per_repeat / regression.period
    regression["embedding_slope_percentile"] = regression.embedding_l2_slope_per_bp.rank(pct=True)
    embedding_q05_slopes = regression.loc[regression.embedding_q < .05, "embedding_l2_slope_per_bp"]
    embedding_q05_top_cutoff = embedding_q05_slopes.quantile(.9, interpolation="lower")

    loci = pd.read_csv(LOCI, usecols=["str_id", "period", "motif", "n_observed_lengths"])
    variance = pd.read_csv(VARIANCE).rename(columns={"variance": "full_cohort_length_variance",
                                                     "n_called": "full_cohort_n_called"})
    mapped = hits.merge(regression, on="str_id", how="left", validate="one_to_one")
    mapped = mapped.merge(loci, on="str_id", how="left", validate="one_to_one")
    mapped = mapped.merge(variance, on="str_id", how="left", validate="one_to_one")
    mapped["in_embedding_catalog"] = mapped.n_observed_lengths.notna()
    mapped["eligible_for_reference_regression"] = mapped.embedding_l2_slope_per_bp.notna()
    mapped["embedding_filter_reason_if_absent"] = np.where(
        mapped.eligible_for_reference_regression, "included",
        np.where(mapped.n_observed_lengths.notna(),
                 "prepared but fewer than 3 distinct observed non-reference alleles",
                 np.where(mapped.full_cohort_length_variance < .1, "full-cohort length variance < 0.1",
                          "not retained in prepared locus catalog")))
    mapped["embedding_fdr05"] = mapped.embedding_q < .05
    mapped["top10_among_embedding_fdr05"] = (mapped.embedding_fdr05 &
        (mapped.embedding_l2_slope_per_bp >= embedding_q05_top_cutoff))
    mapped = mapped.sort_values(["gwas_status", "best_pooled_p"], ascending=[True, True])
    mapped.to_csv(result_dir / "gwas_fdr_str_embedding_map.tsv", sep="\t", index=False)

    present = mapped[mapped.eligible_for_reference_regression].copy()
    hit_ids = set(present.str_id)
    background = regression[~regression.str_id.isin(hit_ids)]
    mw = mannwhitneyu(present.embedding_l2_slope_per_bp, background.embedding_l2_slope_per_bp,
                      alternative="two-sided")
    rank_biserial = 2 * mw.statistic / (len(present) * len(background)) - 1
    overall_top_cutoff = regression.embedding_l2_slope_per_bp.quantile(.9, interpolation="lower")
    hit_top = int((present.embedding_l2_slope_per_bp >= overall_top_cutoff).sum())
    background_top = int((background.embedding_l2_slope_per_bp >= overall_top_cutoff).sum())
    top_fisher = fisher_exact([[hit_top, len(present)-hit_top],
                               [background_top, len(background)-background_top]])

    # Slope distribution and GWAS-hit placement.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    lower, upper = regression.embedding_l2_slope_per_bp.quantile([.005, .995])
    axes[0].hist(regression.embedding_l2_slope_per_bp.clip(lower=lower, upper=upper), bins=100, color="#4b86a6")
    f = present[present.gwas_status.str.contains("Bonferroni")]
    e = present[~present.gwas_status.str.contains("Bonferroni")]
    axes[0].scatter(e.embedding_l2_slope_per_bp.clip(lower=lower, upper=upper), np.full(len(e), 20), marker="|", s=220,
                    color="#e69f00", label="GWAS FDR only")
    axes[0].scatter(f.embedding_l2_slope_per_bp.clip(lower=lower, upper=upper), np.full(len(f), 60), marker="|", s=260,
                    color="#b22222", label="Bonferroni GWAS")
    axes[0].set(xlabel="Layer 21 whole-mean L2 slope per bp (central 99%)", ylabel="Loci",
                title="GWAS loci in the embedding-slope distribution"); axes[0].legend()

    colors = present.gwas_status.map({"GWAS FDR only": "#e69f00", "trait-wise Bonferroni": "#d55e00",
                                      "study-wide Bonferroni": "#8b1a1a"})
    axes[1].scatter(present.embedding_slope_percentile * 100, -np.log10(present.best_pooled_p), c=colors, s=52)
    formal = present[present.gwas_status.str.contains("Bonferroni")]
    for row in formal.itertuples():
        axes[1].annotate(row.str_id, (row.embedding_slope_percentile*100, -np.log10(row.best_pooled_p)),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[1].axvline(90, color="#666666", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Embedding-slope percentile among all embedded loci", ylabel="−log10(best GWAS p)",
                title="GWAS evidence versus embedding slope")
    fig.savefig(figure_dir / "gwas_fdr_strs_in_embedding_results.png", dpi=180); plt.close(fig)

    # Direct comparison of GWAS evidence with embedding FDR. Marker size shows
    # allele count because exact-permutation resolution is allele-count dependent.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    color_map = {"GWAS FDR only": "#e69f00", "trait-wise Bonferroni": "#d55e00",
                 "study-wide Bonferroni": "#8b1a1a"}
    y = -np.log10(present.embedding_q)
    marker_sizes = 32 + 9 * present.n_alternate_alleles
    for ax, column, xlabel in [
        (axes[0], "best_pooled_p", "−log10(best pooled GWAS p)"),
        (axes[1], "best_pooled_fdr", "−log10(best pooled GWAS FDR)"),
    ]:
        x = -np.log10(present[column])
        for gwas_status, group in present.groupby("gwas_status"):
            idx = group.index
            significant = group.embedding_fdr05.to_numpy()
            ax.scatter(x.loc[idx], y.loc[idx], s=marker_sizes.loc[idx],
                       c=color_map[gwas_status], label=gwas_status,
                       edgecolors=np.where(significant, "black", "none"), linewidths=1.2, alpha=.9)
        ax.axhline(-np.log10(.05), color="#555555", linestyle="--", linewidth=1,
                   label="embedding q=0.05")
        if column == "best_pooled_fdr":
            ax.axvline(-np.log10(.05), color="#888888", linestyle=":", linewidth=1,
                       label="GWAS FDR=0.05")
        ax.set(xlabel=xlabel, ylabel="−log10(embedding q)", title="GWAS evidence versus embedding FDR")
    short_labels = {
        "CACTIH010001827.1_963000": "...1827.1:963000",
        "CACTIH010001808.1_930235": "...1808.1:930235",
        "CACTIH010001808.1_854553": "...1808.1:854553",
        "10_3687767": "10:3687767",
    }
    offsets = {
        "CACTIH010001827.1_963000": (-100, 8),
        "CACTIH010001808.1_930235": (-20, -18),
        "CACTIH010001808.1_854553": (5, 5),
        "10_3687767": (5, 5),
    }
    for row in present[present.gwas_status.str.contains("Bonferroni") | present.top10_among_embedding_fdr05].itertuples():
        for ax, column in [(axes[0], "best_pooled_p"), (axes[1], "best_pooled_fdr")]:
            ax.annotate(short_labels.get(row.str_id, row.str_id),
                        (-np.log10(getattr(row, column)), -np.log10(row.embedding_q)),
                        xytext=offsets.get(row.str_id, (5, 5)), textcoords="offset points", fontsize=7)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=8, loc="best")
    fig.suptitle("Mapped GWAS loci: marker size = allele count; black outline = embedding q < 0.05", fontsize=13)
    fig.savefig(figure_dir / "gwas_evidence_vs_embedding_fdr.png", dpi=180); plt.close(fig)

    formal_all = mapped[mapped.gwas_status.str.contains("Bonferroni")]
    formal_present = formal_all[formal_all.eligible_for_reference_regression]
    region_counts = mapped.gwas_functional_region.fillna("unannotated").value_counts()
    evidence_rho, evidence_rho_p = spearmanr(-np.log10(present.best_pooled_p),
                                             -np.log10(present.embedding_q))
    lines = [
        "# Oil-quality STR-GWAS loci in the ArgoNT embedding analysis", "",
        f"The primary SNP-PC-adjusted STR-GWAS reports **{len(gwas):,} FDR < 0.05 trait–locus associations at {len(hits):,} unique STR loci**. This report maps those loci to the reference-anchored Layer 21 whole-sequence mean L2 regressions. The reference is the distance anchor and is excluded as a regression observation.", "",
        "## Mapping", "",
        f"- **{int(mapped.in_embedding_catalog.sum())}/{len(hits)}** GWAS loci are in the prepared embedding catalog.",
        f"- **{len(present)}/{len(hits)}** meet the reference-anchored regression requirement of at least three distinct observed non-reference alleles.",
        f"- **{len(formal_present)}/{len(formal_all)}** formal Bonferroni GWAS loci are eligible for the reference-anchored regression.",
        f"- **{int(present.embedding_fdr05.sum())}/{len(present)}** eligible GWAS loci pass embedding q < 0.05; **{int(present.top10_among_embedding_fdr05.sum())}** are in the highest-slope decile among embedding-significant loci.", "",
        "GWAS FDR and embedding FDR test different hypotheses. The former tests phenotype association; the latter tests whether reference-relative embedding distance changes systematically with absolute repeat-count difference. They should not be expected to agree.", "",
        "## Are GWAS loci unusually high-slope?", "",
        f"The {len(present)} eligible GWAS loci have median slope **{present.embedding_l2_slope_per_bp.median():.4f}** and median percentile **{present.embedding_slope_percentile.median()*100:.1f}**. Other eligible loci have median slope **{background.embedding_l2_slope_per_bp.median():.4f}**. The Mann–Whitney comparison gives p={mw.pvalue:.3g} and rank-biserial effect {rank_biserial:.3f}.", "",
        f"{hit_top}/{len(present)} eligible GWAS loci fall in the top 10% of slopes among all eligible loci (Fisher exact p={top_fisher.pvalue:.3g}, odds ratio={top_fisher.statistic:.3g}).", "",
        "## GWAS evidence versus embedding FDR", "",
        f"Across the {len(present)} eligible GWAS loci, GWAS strength (`-log10(p)`) and embedding significance (`-log10(q)`) have Spearman rho **{evidence_rho:.3f}** (p={evidence_rho_p:.3g}). Marker size shows the number of observed non-reference alleles and a black outline indicates embedding q < 0.05.", "",
        "The strongest GWAS signals are not the most embedding-significant loci. This is expected because GWAS p-values test association with oil traits, whereas embedding q-values test whether embedding distance increases systematically with allele-length difference.", "",
        "## Genomic placement", "",
        "Unique GWAS FDR loci by the GWAS functional annotation: " + ", ".join(f"{key}={value}" for key, value in region_counts.items()) + ".", "",
        "## Formal GWAS loci", "",
        "| STR | GWAS status | Best trait | GWAS p | Embedding | L2 slope/bp | Slope percentile | Embedding q |", "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in formal_all.sort_values("best_pooled_p").itertuples():
        emb = "eligible" if row.eligible_for_reference_regression else row.embedding_filter_reason_if_absent
        slope = f"{row.embedding_l2_slope_per_bp:.4f}" if row.eligible_for_reference_regression else "NA"
        pct = f"{row.embedding_slope_percentile*100:.1f}" if row.eligible_for_reference_regression else "NA"
        q = f"{row.embedding_q:.4g}" if row.eligible_for_reference_regression else "NA"
        lines.append(f"| {row.str_id} | {row.gwas_status} | {row.best_trait} | {row.best_pooled_p:.3g} | {emb} | {slope} | {pct} | {q} |")
    lines += ["", "## Outputs", "", "- `results/gwas_fdr_str_embedding_map.tsv`", "- `figures/gwas_fdr_strs_in_embedding_results.png`", "- `figures/gwas_evidence_vs_embedding_fdr.png`", ""]
    report = report_dir / "GWAS_SIGNIFICANT_STR_EMBEDDING_MAP.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report}")
    print("GWAS_STR_EMBEDDING_MAP=PASS")


if __name__ == "__main__":
    main()
