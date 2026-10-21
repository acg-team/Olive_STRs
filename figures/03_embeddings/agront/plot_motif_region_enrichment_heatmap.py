#!/usr/bin/env python3
"""Plot joint motif-length/genome-region enrichment among significant STR loci."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/argont_motif_region_heatmap_mpl")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "scripts/03_embeddings/agront"
ANALYSES = ROOT / "analyses"
REGRESSIONS = ANALYSES / "results/layer21_whole_mean_regressions.csv.gz"
LOCI = ROOT / "inputs/loci.csv.gz"
PROMOTER_ANNOTATION = Path(os.environ.get(
    "OLIVE_STR_ANNOTATION", "inputs/O_europeana_strs.functional.promoter2kb.annotated.bed"
))
REGIONS = ["intergenic", "promoter", "intron", "UTR", "CDS"]
PERIODS = [1, 2, 3, 4, 5, 6]


def load_promoter_annotation() -> pd.DataFrame:
    annotation = pd.read_csv(
        PROMOTER_ANNOTATION,
        sep="\t",
        usecols=["#chrom", "start", "end", "motif_length", "functional_region"],
        dtype={"#chrom": "string"},
    )
    annotation["str_id"] = (
        annotation["#chrom"] + "_" + annotation["start"].astype(str)
    )
    if annotation["str_id"].duplicated().any():
        raise ValueError("Duplicate STR identifiers in promoter-aware annotation")
    annotation["genome_region"] = annotation["functional_region"].replace(
        {"five_prime_UTR": "UTR", "three_prime_UTR": "UTR"}
    )
    return annotation[["str_id", "genome_region"]]


def main():
    result_dir = ANALYSES / "results"
    figure_dir = Path(__file__).resolve().parent / "output"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    regression = pd.read_csv(REGRESSIONS)
    regression = regression[regression.metric.eq("l2")][["str_id", "q_value"]]
    loci = pd.read_csv(LOCI, usecols=["str_id", "contig", "start", "end", "period"])
    data = regression.merge(loci, on="str_id", validate="one_to_one").reset_index(drop=True)
    data = data.merge(
        load_promoter_annotation(),
        on="str_id",
        how="left",
        validate="one_to_one",
    )
    if data["genome_region"].isna().any():
        missing = int(data["genome_region"].isna().sum())
        raise ValueError(f"{missing:,} analyzed STR loci lack promoter-aware annotation")
    data["significant"] = data.q_value.lt(0.05)

    total_significant = int(data.significant.sum())
    total_not_significant = len(data) - total_significant
    rows = []
    for genome_region in REGIONS:
        for period in PERIODS:
            selected = data.genome_region.eq(genome_region) & data.period.eq(period)
            tested = int(selected.sum())
            significant = int(data.loc[selected, "significant"].sum())
            not_significant = tested - significant
            other_significant = total_significant - significant
            other_not_significant = total_not_significant - not_significant
            odds_ratio, p_value = fisher_exact(
                [[significant, not_significant],
                 [other_significant, other_not_significant]]
            )
            rows.append({
                "genome_region": genome_region,
                "motif_length_bp": period,
                "tested_loci": tested,
                "significant_loci": significant,
                "significant_proportion": significant / tested if tested else np.nan,
                "odds_ratio_vs_all_other_tested_loci": odds_ratio,
                "fisher_exact_p_value": p_value,
            })

    table = pd.DataFrame(rows)
    table.to_csv(result_dir / "motif_length_genome_region_enrichment.csv", index=False)

    odds = table.pivot(index="genome_region", columns="motif_length_bp",
                       values="odds_ratio_vs_all_other_tested_loci").reindex(
                           index=REGIONS, columns=PERIODS)
    log_odds = np.log2(odds.replace(0, np.nan))
    finite = np.abs(log_odds.to_numpy()[np.isfinite(log_odds.to_numpy())])
    limit = max(2.0, float(np.quantile(finite, 0.90)))
    display = log_odds.clip(-limit, limit).fillna(-limit)

    fig, ax = plt.subplots(figsize=(12.2, 6.4), constrained_layout=True)
    image = ax.imshow(display, cmap="RdBu_r",
                      norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                      aspect="auto")
    ax.set_xticks(range(len(PERIODS)), [f"{x} bp" for x in PERIODS])
    ax.set_yticks(
        range(len(REGIONS)), ["Intergenic", "Promoter", "Intron", "UTR", "CDS"]
    )
    ax.set_xlabel("STR motif length")
    ax.set_ylabel("Genome region")
    ax.set_title("Reference-anchored ArgoNT L2 significance enrichment\n"
                 "Cell text: odds ratio; significant/tested (%)")

    for row_index, genome_region in enumerate(REGIONS):
        for column_index, period in enumerate(PERIODS):
            record = table[(table.genome_region == genome_region) &
                           (table.motif_length_bp == period)].iloc[0]
            odds_text = f"OR {record.odds_ratio_vs_all_other_tested_loci:.2f}"
            proportion_text = (f"{int(record.significant_loci)}/{int(record.tested_loci)} "
                               f"({record.significant_proportion:.1%})")
            value = display.loc[genome_region, period]
            text_color = "white" if abs(value) > limit * 0.55 else "black"
            ax.text(column_index, row_index, odds_text + "\n" + proportion_text,
                    ha="center", va="center", fontsize=8.5, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.025)
    colorbar.set_label("log2 odds ratio vs all other tested loci")
    ax.text(0, -0.17,
            "Red = enriched; blue = depleted. Odds ratios compare each joint cell "
            "with all other tested STR loci.\n"
            "Promoter = strand-aware 2 kb upstream of the annotated TSS; "
            "q < 0.05 defines significance.",
            transform=ax.transAxes, fontsize=9)
    output = figure_dir / "motif_length_genome_region_enrichment_heatmap.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output}")
    print("MOTIF_REGION_HEATMAP=PASS")


if __name__ == "__main__":
    main()
