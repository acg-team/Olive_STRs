#!/usr/bin/env python3
"""Compare STR-GWAS FDR with primary embedding FDR for three models."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
ROOT = SCRIPTS_ROOT / "07_association_embedding_integration"
SOURCE = (
    SCRIPTS_ROOT
    / "04_cross_model_comparison/gwas_overlap/results/"
    "gwas_followup_consensus_overlap.tsv"
)
MATCHED = (
    SCRIPTS_ROOT
    / "04_cross_model_comparison/results/"
    "matched_l2_per_locus_significance.csv.gz"
)
GWAS_ROOT = Path(os.environ.get(
    "OLIVE_STR_ASSOCIATION_ROOT", SCRIPTS_ROOT / "06_trait_association"
))
GWAS_TESTED = GWAS_ROOT / "source_data/tested_loci.tsv.gz"
GWAS_RESULTS = GWAS_TESTED.parents[1] / "results"
GWAS_FOLLOWUP = GWAS_RESULTS / "fdr_0.05_followup_loci.tsv"
GWAS_ASSOCIATIONS = {
    "oleic_acid": GWAS_RESULTS / "oleic_acid_association.tsv.gz",
    "linoleic_acid": GWAS_RESULTS / "linoleic_acid_association.tsv.gz",
    "log_oleic_linoleic_ratio": (
        GWAS_RESULTS / "log_oleic_linoleic_ratio_association.tsv.gz"
    ),
}
POPULATION_ROOT = Path(os.environ.get(
    "OLIVE_STR_POPULATION_ROOT", SCRIPTS_ROOT / "02_population_structure"
))
POPULATION_PANEL = POPULATION_ROOT / "source_data/population_informative_panel_798.tsv"
POPULATION_UNIVERSE = POPULATION_PANEL.parent / "input_locus_qc.tsv.gz"
ALLELE_SOURCE = (
    SCRIPTS_ROOT
    / "03_embeddings/agront/analyses/results/"
    "gwas_fdr_str_embedding_map.tsv"
)
RESULTS = ROOT / "results"
FIGURES = Path(__file__).resolve().parent / "output"
REPORT = ROOT / "report"

MODELS = [
    ("argont", "ArgoNT", "#0072B2"),
    ("plantcad2", "PlantCAD2", "#E69F00"),
    ("gpn", "GPN", "#009E73"),
]
def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE, sep="\t")
    allele = pd.read_csv(
        ALLELE_SOURCE,
        sep="\t",
        usecols=["str_id", "n_alternate_alleles"],
    )
    if len(source) != 68 or source.str_id.duplicated().any():
        raise RuntimeError("Expected 68 unique STR-GWAS follow-up loci")
    if allele.str_id.duplicated().any():
        raise RuntimeError("Duplicate STR identifiers in allele-count source")

    source = source.merge(allele, on="str_id", how="left", validate="one_to_one")
    rows = []
    for model, display, color in MODELS:
        part = source[
            [
                "str_id",
                "gwas_status",
                "best_trait",
                "best_pooled_p",
                "best_pooled_fdr",
                "n_alternate_alleles",
                f"{model}_q_value",
            ]
        ].copy()
        part = part.rename(columns={f"{model}_q_value": "embedding_q"})
        part["model"] = display
        part["model_color"] = color
        part["eligible"] = part.embedding_q.notna()
        part["embedding_fdr05"] = part.embedding_q < 0.05
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long["minus_log10_gwas_fdr"] = -np.log10(
        long.best_pooled_fdr.clip(lower=np.finfo(float).tiny)
    )
    long["minus_log10_embedding_q"] = -np.log10(
        long.embedding_q.clip(lower=np.finfo(float).tiny)
    )
    return source, long


def add_panel(ax: plt.Axes, data: pd.DataFrame, model: str) -> None:
    present = data[data.eligible].copy()
    sizes = 45 + 12 * present.n_alternate_alleles.fillna(1)
    model_color = present.model_color.iloc[0]
    edge = np.where(present.embedding_fdr05, "black", "none")
    ax.scatter(
        present.minus_log10_gwas_fdr,
        present.minus_log10_embedding_q,
        s=sizes,
        c=model_color,
        edgecolors=edge,
        linewidths=1.4,
        alpha=0.9,
        zorder=3,
    )

    threshold = -np.log10(0.05)
    ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1.5, zorder=1)
    ax.axvline(threshold, color="#888888", linestyle=":", linewidth=1.5, zorder=1)
    ax.set(
        title=model,
        xlabel="−log10(best pooled STR–phenotype FDR)",
        ylabel="−log10(embedding q)",
    )
    ax.set_box_aspect(1)
    sns.despine(ax=ax)


def legend_handles(selected_model: str | None = None) -> list[Line2D]:
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=display,
        )
        for _, display, color in MODELS
        if selected_model is None or display == selected_model
    ]
    return model_handles + [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="embedding q = 0.05",
        ),
        Line2D(
            [0],
            [0],
            color="#888888",
            linestyle=":",
            linewidth=1.5,
            label="STR–phenotype FDR = 0.05",
        ),
    ]


def test_enrichment(source: pd.DataFrame) -> pd.DataFrame:
    """Test whether GWAS follow-up loci are enriched among significant loci."""
    matched = pd.read_csv(MATCHED)
    if len(matched) != 62_847 or matched.str_id.duplicated().any():
        raise RuntimeError("Unexpected matched three-model testing universe")

    gwas_tested = pd.read_csv(
        GWAS_TESTED,
        sep="\t",
        usecols=["str_id", "included"],
    )
    gwas_tested = gwas_tested[gwas_tested.included].copy()
    if gwas_tested.str_id.duplicated().any():
        raise RuntimeError("Duplicate loci in STR-GWAS tested-locus universe")
    common_ids = set(matched.str_id) & set(gwas_tested.str_id)
    matched = matched[matched.str_id.isin(common_ids)].copy()
    if len(matched) != 62_739:
        raise RuntimeError("Expected 62,739 loci tested by GWAS and all models")

    gwas_ids = set(source.loc[source.eligible_for_matched_test, "str_id"])
    if len(gwas_ids) != 46:
        raise RuntimeError("Expected 46 eligible STR-GWAS follow-up loci")
    in_gwas = matched.str_id.isin(gwas_ids)
    rows = []
    for model, display, _ in MODELS:
        significant = matched[f"{model}_significant"].astype(bool)
        gwas_sig = int((in_gwas & significant).sum())
        gwas_not_sig = int((in_gwas & ~significant).sum())
        non_gwas_sig = int((~in_gwas & significant).sum())
        non_gwas_not_sig = int((~in_gwas & ~significant).sum())
        table = [
            [gwas_sig, gwas_not_sig],
            [non_gwas_sig, non_gwas_not_sig],
        ]
        odds_ratio, enrichment_p = fisher_exact(table, alternative="greater")
        _, two_sided_p = fisher_exact(table, alternative="two-sided")
        rows.append(
            {
                "model": display,
                "tested_loci": len(matched),
                "eligible_gwas_loci": int(in_gwas.sum()),
                "gwas_embedding_significant": gwas_sig,
                "gwas_embedding_not_significant": gwas_not_sig,
                "non_gwas_embedding_significant": non_gwas_sig,
                "non_gwas_embedding_not_significant": non_gwas_not_sig,
                "gwas_significant_fraction": gwas_sig / int(in_gwas.sum()),
                "non_gwas_significant_fraction": (
                    non_gwas_sig / int((~in_gwas).sum())
                ),
                "fisher_odds_ratio": odds_ratio,
                "fisher_enrichment_p": enrichment_p,
                "fisher_two_sided_p": two_sided_p,
            }
        )
    enrichment = pd.DataFrame(rows)
    enrichment.to_csv(
        RESULTS / "phenotype_str_enrichment_fisher.csv",
        index=False,
    )
    return enrichment


def test_top_one_percent() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test the top 1% of locus-level GWAS evidence in the common universe."""
    matched = pd.read_csv(MATCHED)
    gwas_tested = pd.read_csv(GWAS_TESTED, sep="\t", usecols=["str_id", "included"])
    common_ids = set(gwas_tested.loc[gwas_tested.included, "str_id"])
    matched = matched[matched.str_id.isin(common_ids)].copy()
    if len(matched) != 62_739:
        raise RuntimeError("Unexpected common GWAS/embedding universe")

    trait_p = []
    for trait, path in GWAS_ASSOCIATIONS.items():
        frame = pd.read_csv(path, sep="\t", usecols=["str_id", "pooled_p"])
        if frame.str_id.duplicated().any():
            raise RuntimeError(f"Duplicate STRs in {path}")
        trait_p.append(
            frame.set_index("str_id").pooled_p.rename(f"{trait}_pooled_p")
        )
    evidence = pd.concat(trait_p, axis=1)
    evidence["best_pooled_p"] = evidence.min(axis=1)
    evidence["best_trait"] = evidence[
        [f"{trait}_pooled_p" for trait in GWAS_ASSOCIATIONS]
    ].idxmin(axis=1).str.removesuffix("_pooled_p")
    evidence = evidence.reset_index()

    ranked = matched.merge(
        evidence,
        on="str_id",
        how="left",
        validate="one_to_one",
    )
    if ranked.best_pooled_p.isna().any():
        raise RuntimeError("Common loci missing GWAS P-values")
    ranked = ranked.sort_values(
        ["best_pooled_p", "str_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["gwas_best_p_rank"] = np.arange(1, len(ranked) + 1)
    top_n = int(np.ceil(0.01 * len(ranked)))
    ranked["top_1pct_gwas"] = ranked.gwas_best_p_rank <= top_n
    ranked["gwas_best_p_percentile"] = ranked.gwas_best_p_rank / len(ranked)
    ranked.to_csv(RESULTS / "top1pct_gwas_loci.tsv", sep="\t", index=False)

    rows = []
    selected = ranked.top_1pct_gwas
    for model, display, _ in MODELS:
        significant = ranked[f"{model}_significant"].astype(bool)
        selected_sig = int((selected & significant).sum())
        selected_not_sig = int((selected & ~significant).sum())
        other_sig = int((~selected & significant).sum())
        other_not_sig = int((~selected & ~significant).sum())
        table = [
            [selected_sig, selected_not_sig],
            [other_sig, other_not_sig],
        ]
        odds_ratio, enrichment_p = fisher_exact(table, alternative="greater")
        _, two_sided_p = fisher_exact(table, alternative="two-sided")
        rows.append(
            {
                "model": display,
                "tested_loci": len(ranked),
                "top_1pct_loci": top_n,
                "top_1pct_max_best_pooled_p": ranked.loc[
                    ranked.top_1pct_gwas, "best_pooled_p"
                ].max(),
                "top_1pct_embedding_significant": selected_sig,
                "top_1pct_embedding_not_significant": selected_not_sig,
                "other_embedding_significant": other_sig,
                "other_embedding_not_significant": other_not_sig,
                "top_1pct_significant_fraction": selected_sig / top_n,
                "other_significant_fraction": other_sig / (len(ranked) - top_n),
                "fisher_odds_ratio": odds_ratio,
                "fisher_enrichment_p": enrichment_p,
                "fisher_two_sided_p": two_sided_p,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "top1pct_gwas_enrichment_fisher.csv", index=False)
    return ranked, result


def test_population_informative_overlap() -> pd.DataFrame:
    """Test dNMF population-panel enrichment among embedding-significant loci."""
    panel = pd.read_csv(POPULATION_PANEL, sep="\t", usecols=["str_id"])
    universe = pd.read_csv(POPULATION_UNIVERSE, sep="\t", usecols=["str_id"])
    matched = pd.read_csv(MATCHED)
    if len(panel) != 798 or panel.str_id.duplicated().any():
        raise RuntimeError("Unexpected population-informative panel")
    if len(universe) != 45_665 or universe.str_id.duplicated().any():
        raise RuntimeError("Unexpected dNMF input locus universe")

    matched = matched[matched.str_id.isin(set(universe.str_id))].copy()
    if len(matched) != 35_331:
        raise RuntimeError("Expected 35,331 common dNMF/embedding loci")
    in_panel = matched.str_id.isin(set(panel.str_id))
    if int(in_panel.sum()) != 577:
        raise RuntimeError("Expected 577 matched population-panel loci")

    tests = [
        ("ArgoNT", matched.argont_significant.astype(bool)),
        ("PlantCAD2", matched.plantcad2_significant.astype(bool)),
        ("GPN", matched.gpn_significant.astype(bool)),
        (
            "Significant in all three",
            matched[
                [
                    "argont_significant",
                    "plantcad2_significant",
                    "gpn_significant",
                ]
            ].all(axis=1),
        ),
        (
            "Significant in ≥1 model",
            matched[
                [
                    "argont_significant",
                    "plantcad2_significant",
                    "gpn_significant",
                ]
            ].any(axis=1),
        ),
    ]
    rows = []
    for label, significant in tests:
        panel_sig = int((in_panel & significant).sum())
        panel_not_sig = int((in_panel & ~significant).sum())
        other_sig = int((~in_panel & significant).sum())
        other_not_sig = int((~in_panel & ~significant).sum())
        table = [
            [panel_sig, panel_not_sig],
            [other_sig, other_not_sig],
        ]
        odds_ratio, enrichment_p = fisher_exact(table, alternative="greater")
        _, two_sided_p = fisher_exact(table, alternative="two-sided")
        rows.append(
            {
                "embedding_set": label,
                "dnmf_candidate_loci": len(universe),
                "common_tested_loci": len(matched),
                "population_panel_loci": len(panel),
                "matched_population_panel_loci": int(in_panel.sum()),
                "unmatched_population_panel_loci": len(panel) - int(in_panel.sum()),
                "panel_embedding_significant": panel_sig,
                "panel_embedding_not_significant": panel_not_sig,
                "other_embedding_significant": other_sig,
                "other_embedding_not_significant": other_not_sig,
                "expected_overlap": int(in_panel.sum())
                * int(significant.sum())
                / len(matched),
                "fisher_odds_ratio": odds_ratio,
                "fisher_enrichment_p": enrichment_p,
                "fisher_two_sided_p": two_sided_p,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        RESULTS / "population_informative_embedding_overlap_fisher.csv",
        index=False,
    )
    return result


def write_report(
    source: pd.DataFrame,
    long: pd.DataFrame,
    enrichment: pd.DataFrame,
    top_one_percent: pd.DataFrame,
    population_overlap: pd.DataFrame,
) -> None:
    eligible = long[long.eligible]
    shared_ids = set(
        source.loc[source.all_three_significant.eq(True), "str_id"]
    )
    shared_traits = pd.read_csv(GWAS_FOLLOWUP, sep="\t")
    shared_traits = shared_traits[shared_traits.str_id.isin(shared_ids)].copy()
    shared_traits = shared_traits.sort_values(["str_id", "pooled_p"])
    if len(shared_ids) != 4 or shared_traits.str_id.nunique() != 4:
        raise RuntimeError("Unexpected three-model-supported phenotype loci")
    lines = [
        "# STR-GWAS follow-up loci versus embedding FDR",
        "",
        "This analysis uses the 68 unique loci in the STR-GWAS FDR 0.05 "
        "follow-up set. Embedding q-values are the current primary "
        "reference-anchored whole-mean L2 results: ArgoNT layer 21, "
        "PlantCAD2 layer 36, and GPN final layer.",
        "",
        "The panels plot `−log10(best pooled STR–phenotype FDR)` against "
        "`−log10(embedding q)`. Marker size represents the number of observed "
        "non-reference alleles, and a black outline denotes embedding q < 0.05. "
        "Points use the model-comparison palette: blue for ArgoNT, orange for "
        "PlantCAD2, and green for GPN.",
        "",
        "## Mapping summary",
        "",
        f"- Unique STR-GWAS follow-up loci: **{len(source)}**",
    ]
    for _, display, _ in MODELS:
        part = eligible[eligible.model.eq(display)]
        lines.append(
            f"- {display}: **{len(part)}/{len(source)}** loci have matched "
            f"embedding tests; **{int(part.embedding_fdr05.sum())}/{len(part)}** "
            "pass embedding FDR 0.05."
        )
    lines += [
        "",
        "## Phenotypes associated with loci supported by all three models",
        "",
        "Four STR–phenotype follow-up loci pass embedding q < 0.05 in ArgoNT, "
        "PlantCAD2, and GPN. All FDR-significant phenotype associations for "
        "these loci are listed below.",
        "",
        "| STR locus | Associated trait | Effect per repeat | STR–phenotype FDR | Center robust |",
        "|---|---|---:|---:|---:|",
    ]
    trait_labels = {
        "oleic_acid": "Oleic acid",
        "linoleic_acid": "Linoleic acid",
        "log_oleic_linoleic_ratio": "Log oleic/linoleic ratio",
    }
    for row in shared_traits.itertuples():
        lines.append(
            f"| {row.str_id} | {trait_labels.get(row.trait, row.trait)} | "
            f"{row.pooled_beta:+.3f} | {row.pooled_fdr:.4f} | "
            f"{'yes' if row.center_robust else 'no'} |"
        )
    lines += [
        "",
        "Three of the four loci are associated with linoleic acid, and two are "
        "associated with oleic acid. `CACTIH010007616.1_49534` is associated "
        "with both traits: increasing repeat length predicts higher oleic acid "
        "and lower linoleic acid. These are exploratory FDR-supported "
        "associations; none passes the trait-wise or study-wide Bonferroni "
        "threshold.",
        "",
        "## Enrichment among embedding-significant loci",
        "",
        "The enrichment universe is the **62,739 loci tested both by STR-GWAS "
        "and by all three embedding models**. Restricting to this intersection "
        "avoids incorrectly treating embedding loci that were never tested by "
        "GWAS as phenotype-negative. The phenotype-associated set contains the "
        "46 STR-GWAS follow-up loci eligible in this common universe. For each "
        "model, a one-sided Fisher exact test (`alternative = greater`) asks "
        "whether phenotype-associated STRs have higher odds of embedding "
        "significance than the other 62,693 commonly tested STRs.",
        "",
        "| Model | GWAS significant/tested (%) | Other significant/tested (%) | Odds ratio | Fisher enrichment P |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in enrichment.itertuples():
        gwas_pct = 100 * row.gwas_significant_fraction
        other_tested = (
            row.non_gwas_embedding_significant
            + row.non_gwas_embedding_not_significant
        )
        other_pct = 100 * row.non_gwas_significant_fraction
        lines.append(
            f"| {row.model} | {row.gwas_embedding_significant}/"
            f"{row.eligible_gwas_loci} ({gwas_pct:.1f}%) | "
            f"{row.non_gwas_embedding_significant:,}/{other_tested:,} "
            f"({other_pct:.1f}%) | **{row.fisher_odds_ratio:.3f}** | "
            f"**{row.fisher_enrichment_p:.3g}** |"
        )
    lines += [
        "",
        "An odds ratio above 1 indicates enrichment. The reported P-values "
        "are exact one-sided enrichment P-values; the results table also "
        "contains two-sided Fisher P-values.",
        "",
        "**Conclusion:** none of the three models shows evidence that "
        "phenotype-associated STRs are enriched among embedding-significant "
        "loci. All three odds ratios are at or below 1 and all enrichment "
        "P-values are greater than 0.05.",
        "",
        "## Exploratory top-1% GWAS analysis",
        "",
        "As a requested exploratory sensitivity analysis, loci were ranked by their "
        "smallest pooled GWAS P-value across oleic acid, linoleic acid, and the "
        "log oleic/linoleic ratio. The top **628/62,739 loci** (1%; maximum "
        f"best pooled P = {top_one_percent.top_1pct_max_best_pooled_p.iloc[0]:.4g}) "
        "were designated the phenotype-evidence set. Exact rank was used to "
        "keep the set size fixed, with STR identifier as a deterministic "
        "tie-breaker.",
        "",
        "| Model | Top 1% significant/tested (%) | Other significant/tested (%) | Odds ratio | Fisher enrichment P |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top_one_percent.itertuples():
        top_pct = 100 * row.top_1pct_significant_fraction
        other_n = row.other_embedding_significant + row.other_embedding_not_significant
        other_pct = 100 * row.other_significant_fraction
        lines.append(
            f"| {row.model} | {row.top_1pct_embedding_significant}/"
            f"{row.top_1pct_loci} ({top_pct:.1f}%) | "
            f"{row.other_embedding_significant:,}/{other_n:,} "
            f"({other_pct:.1f}%) | **{row.fisher_odds_ratio:.3f}** | "
            f"**{row.fisher_enrichment_p:.3g}** |"
        )
    lines += [
        "",
        "The top-1% sensitivity analysis also provides no evidence of "
        "enrichment: embedding-significant fractions are slightly lower in "
        "the top GWAS-evidence set for all three models.",
        "",
        "## dNMF population-informative panel overlap",
        "",
        "The retained dNMF panel contains 798 population-informative chromosome "
        "STRs selected from 45,665 variance-filtered candidate loci. Enrichment "
        "was tested in the **35,331 loci present in both the dNMF candidate "
        "universe and the matched embedding analysis**. Of the 798 panel loci, "
        "577 had matched embedding tests; the other 221 were excluded rather "
        "than classified as embedding-negative.",
        "",
        "| Embedding set | Panel overlap | Expected overlap | Odds ratio | Fisher enrichment P |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in population_overlap.itertuples():
        lines.append(
            f"| {row.embedding_set} | {row.panel_embedding_significant}/"
            f"{row.matched_population_panel_loci} | {row.expected_overlap:.1f} | "
            f"**{row.fisher_odds_ratio:.3f}** | "
            f"**{row.fisher_enrichment_p:.3g}** |"
        )
    lines += [
        "",
        "There is overlap—118 panel loci are embedding-significant in at least "
        "one model and 56 are significant in all three—but every observed "
        "overlap is below random expectation. The one-sided Fisher exact tests "
        "provide no evidence that population-informative loci are enriched "
        "among embedding-significant STRs.",
        "",
        "GWAS FDR and embedding FDR test different hypotheses. GWAS FDR "
        "measures evidence of association with an oil-quality trait, whereas "
        "embedding FDR measures evidence that reference-relative embedding "
        "distance changes with absolute STR repeat-count difference.",
        "",
        "## Outputs",
        "",
        "- `results/str_gwas_followup_three_model_embedding_fdr.tsv`",
        "- `results/phenotype_str_enrichment_fisher.csv`",
        "- `results/top1pct_gwas_enrichment_fisher.csv`",
        "- `results/top1pct_gwas_loci.tsv`",
        "- `results/population_informative_embedding_overlap_fisher.csv`",
        "- `figures/three_model_gwas_evidence_vs_embedding_fdr.png`",
        "- `figures/argont_gwas_evidence_vs_embedding_fdr.png`",
        "- `figures/plantcad2_gwas_evidence_vs_embedding_fdr.png`",
        "- `figures/gpn_gwas_evidence_vs_embedding_fdr.png`",
        "",
    ]
    (REPORT / "GWAS_EMBEDDING_FDR.md").write_text("\n".join(lines))


def main() -> None:
    for directory in (RESULTS, FIGURES, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="talk", style="ticks", font_scale=1.3)
    source, long = prepare_data()
    long.to_csv(
        RESULTS / "str_gwas_followup_three_model_embedding_fdr.tsv",
        sep="\t",
        index=False,
    )
    enrichment = test_enrichment(source)
    _, top_one_percent = test_top_one_percent()
    population_overlap = test_population_informative_overlap()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(26, 8),
        sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.25, wspace=0.12)
    for ax, (_, display, _) in zip(axes, MODELS):
        add_panel(ax, long[long.model.eq(display)], display)
        ax.set_xlabel("")
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    axes[-1].legend(
        handles=legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
    )
    fig.suptitle(
        "STR–phenotype evidence versus embedding FDR",
        y=1.10,
    )
    fig.supxlabel(
        "−log10(best pooled STR–phenotype FDR)",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    fig.savefig(
        FIGURES / "three_model_gwas_evidence_vs_embedding_fdr.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    for _, display, _ in MODELS:
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        add_panel(ax, long[long.model.eq(display)], display)
        ax.legend(
            handles=legend_handles(selected_model=display),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True,
        )
        fig.savefig(
            FIGURES / f"{display.lower()}_gwas_evidence_vs_embedding_fdr.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    write_report(
        source,
        long,
        enrichment,
        top_one_percent,
        population_overlap,
    )
    print("GWAS_EMBEDDING_FDR=PASS")


if __name__ == "__main__":
    main()
