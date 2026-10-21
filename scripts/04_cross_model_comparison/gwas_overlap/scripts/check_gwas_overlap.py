#!/usr/bin/env python3
"""Check three-model consensus STR loci against STR-GWAS follow-up loci."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
MATCHED = (
    WORKSPACE
    / "04_cross_model_comparison/results/matched_l2_per_locus_significance.csv.gz"
)
GWAS = Path(os.environ.get(
    "OLIVE_STR_GWAS_FOLLOWUP",
    WORKSPACE / "06_trait_association/results/fdr_0.05_followup_loci.tsv",
))
PRIOR_MAP = (
    WORKSPACE
    / "03_embeddings/agront/analyses/results/gwas_fdr_str_embedding_map.tsv"
)
MODELS = ("argont", "plantcad2", "gpn")
DISPLAY = {"argont": "ArgoNT", "plantcad2": "PlantCAD2", "gpn": "GPN"}


def gwas_status(group):
    if group.passes_study_bonferroni.any():
        return "study-wide Bonferroni"
    if group.passes_trait_bonferroni.any():
        return "trait-wise Bonferroni"
    return "GWAS FDR only"


def aggregate_gwas(frame):
    rows = []
    for str_id, group in frame.groupby("str_id", sort=False):
        best = group.sort_values("pooled_p").iloc[0]
        rows.append(
            {
                "str_id": str_id,
                "gwas_status": gwas_status(group),
                "traits": ",".join(sorted(group.trait.unique())),
                "trait_associations": len(group),
                "best_trait": best.trait,
                "best_pooled_beta": best.pooled_beta,
                "best_pooled_p": best.pooled_p,
                "best_pooled_fdr": best.pooled_fdr,
                "center_robust_all": bool(group.center_robust.all()),
                "chrom": str(best.chrom),
                "start": int(best.start),
                "end": int(best.end),
                "motif_length": int(best.motif_length),
                "motif": best.motif,
                "functional_region": best.functional_region,
                "nearest_gene_id": best.gene_id,
                "nearest_gene_name": best.gene_name,
                "distance_to_gene": best.distance_to_gene,
            }
        )
    return pd.DataFrame(rows)


def main():
    for directory in (RESULTS, FIGURES, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCHED)
    if len(matched) != 62847 or matched.str_id.duplicated().any():
        raise RuntimeError("Unexpected matched comparison dimensions")
    matched["all_three_significant"] = (
        matched.argont_significant
        & matched.plantcad2_significant
        & matched.gpn_significant
    )
    consensus = matched[matched.all_three_significant]

    gwas_raw = pd.read_csv(GWAS, sep="\t")
    gwas = aggregate_gwas(gwas_raw)
    if gwas.str_id.duplicated().any():
        raise RuntimeError("GWAS aggregation produced duplicate STR loci")
    prior = pd.read_csv(PRIOR_MAP, sep="\t")
    prior = prior[
        [
            "str_id",
            "in_embedding_catalog",
            "embedding_filter_reason_if_absent",
            "n_observed_lengths",
            "full_cohort_length_variance",
        ]
    ]
    detail = gwas.merge(prior, on="str_id", how="left", validate="one_to_one")
    detail = detail.merge(matched, on="str_id", how="left", validate="one_to_one")
    detail["eligible_for_matched_test"] = detail.argont_q_value.notna()
    detail["all_three_significant"] = detail.all_three_significant.eq(True)
    detail["matched_test_status"] = np.select(
        [
            detail.eligible_for_matched_test,
            detail.in_embedding_catalog.fillna(False),
        ],
        [
            "eligible",
            "excluded: fewer than 3 distinct observed non-reference alleles",
        ],
        default="excluded from embedding catalogue",
    )
    detail = detail.sort_values(
        ["all_three_significant", "gwas_status", "best_pooled_p"],
        ascending=[False, True, True],
    )
    detail.to_csv(RESULTS / "gwas_followup_consensus_overlap.tsv", sep="\t", index=False)

    eligible = detail[detail.eligible_for_matched_test].copy()
    n = len(matched)
    selected = len(consensus)
    gwas_n = len(eligible)
    observed_overlap = int(eligible.all_three_significant.sum())
    expected = selected * gwas_n / n
    depletion_p = hypergeom.cdf(observed_overlap, n, selected, gwas_n)
    enrichment_p = hypergeom.sf(observed_overlap - 1, n, selected, gwas_n)
    table = [
        [observed_overlap, gwas_n - observed_overlap],
        [selected - observed_overlap, n - selected - gwas_n + observed_overlap],
    ]
    odds_ratio, fisher_two_sided = fisher_exact(table, alternative="two-sided")
    counts = {
        model: int(eligible[f"{model}_significant"].sum()) for model in MODELS
    }
    summary = pd.DataFrame(
        [
            {
                "gwas_trait_locus_associations": len(gwas_raw),
                "unique_gwas_followup_loci": len(gwas),
                "gwas_loci_in_embedding_catalog": int(
                    detail.in_embedding_catalog.fillna(False).sum()
                ),
                "gwas_loci_eligible_for_matched_test": gwas_n,
                "three_model_consensus_loci": selected,
                "observed_consensus_gwas_overlap": observed_overlap,
                "expected_overlap_if_random": expected,
                "hypergeometric_depletion_p": depletion_p,
                "hypergeometric_enrichment_p": enrichment_p,
                "fisher_two_sided_p": fisher_two_sided,
                "fisher_odds_ratio": odds_ratio,
                "eligible_gwas_argont_fdr05": counts["argont"],
                "eligible_gwas_plantcad2_fdr05": counts["plantcad2"],
                "eligible_gwas_gpn_fdr05": counts["gpn"],
            }
        ]
    )
    summary.to_csv(RESULTS / "gwas_consensus_overlap_summary.csv", index=False)

    plot_data = eligible.sort_values("best_pooled_p").copy()
    q = np.column_stack(
        [plot_data[f"{model}_q_value"].astype(float) for model in MODELS]
    )
    values = -np.log10(np.clip(q, 1e-300, 1))
    fig, ax = plt.subplots(
        figsize=(8.5, max(5.5, 0.36 * len(plot_data) + 2)),
        constrained_layout=True,
    )
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(3), [DISPLAY[model] for model in MODELS])
    ax.set_yticks(range(len(plot_data)), plot_data.str_id, fontsize=8)
    for row in range(len(plot_data)):
        for column in range(3):
            ax.text(
                column,
                row,
                f"q={q[row, column]:.3g}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] > 2 else "black",
            )
    ax.set_title(
        "Primary whole-mean L2 FDR for eligible STR-GWAS follow-up loci"
    )
    fig.colorbar(image, ax=ax, label="−log10(q)")
    fig.savefig(FIGURES / "gwas_followup_matched_model_qvalues.png", dpi=200)
    plt.close(fig)

    formal = detail[detail.gwas_status.str.contains("Bonferroni")]
    lines = [
        "# STR-GWAS follow-up loci versus three-model consensus loci",
        "",
        "The oil-quality STR-GWAS follow-up table contains "
        f"**{len(gwas_raw)} trait–locus associations at {len(gwas)} unique STR loci**. "
        f"The primary reference-anchored embedding comparison contains **{selected:,} loci** "
        "that pass whole-mean L2 FDR 0.05 in ArgoNT, PlantCAD2, and GPN.",
        "",
        "## Overlap",
        "",
        f"- **{int(detail.in_embedding_catalog.fillna(False).sum())}/{len(gwas)}** GWAS follow-up "
        "loci are in the original embedding catalogue.",
        f"- **{gwas_n}/{len(gwas)}** meet the matched test requirement of at least three distinct "
        "observed non-reference alleles.",
        f"- **{observed_overlap}/{gwas_n} eligible GWAS loci overlap the {selected:,}-locus "
        "three-model consensus set.**",
        f"- Random overlap at the observed set sizes is {expected:.2f} loci. The one-sided "
        f"hypergeometric depletion p-value is {depletion_p:.3g}; with only {gwas_n} eligible "
        "GWAS loci, the observed deviation from random expectation is imprecise.",
        "",
        "Among the eligible GWAS loci considered one model at a time, "
        f"{counts['argont']} pass ArgoNT FDR 0.05, {counts['plantcad2']} pass PlantCAD2, "
        f"and {counts['gpn']} pass GPN. Consensus requires all three.",
        "",
        "## Formal GWAS loci",
        "",
        "| STR | GWAS status | Best trait | GWAS p | Matched-test status | ArgoNT q | PlantCAD2 q | GPN q | Consensus |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in formal.sort_values("best_pooled_p").itertuples():
        q_values = []
        for model in MODELS:
            value = getattr(row, f"{model}_q_value")
            q_values.append(f"{value:.4g}" if pd.notna(value) else "NA")
        lines.append(
            f"| {row.str_id} | {row.gwas_status} | {row.best_trait} | "
            f"{row.best_pooled_p:.3g} | {row.matched_test_status} | "
            f"{q_values[0]} | {q_values[1]} | {q_values[2]} | "
            f"{'yes' if row.all_three_significant else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Limited consensus overlap does not mean the models fail to represent trait-associated STRs. "
        "The embedding test asks whether perturbation distance increases consistently with allele-length "
        "difference; GWAS asks whether allele length predicts an oil trait. A locus can have a strong "
        "trait association without an unusually monotonic or model-consistent embedding-distance response. "
        "Test-resolution limitations and the small eligible GWAS set further limit inference.",
        "",
        "## Outputs",
        "",
        "- `results/gwas_followup_consensus_overlap.tsv`",
        "- `results/gwas_consensus_overlap_summary.csv`",
        "- `figures/gwas_followup_matched_model_qvalues.png`",
        "",
    ]
    report_path = REPORT / "GWAS_FOLLOWUP_CONSENSUS_OVERLAP.md"
    report_path.write_text("\n".join(lines))
    print(summary.to_string(index=False))
    print(f"report={report_path}")
    print("GWAS_CONSENSUS_OVERLAP=PASS")


if __name__ == "__main__":
    main()
