#!/usr/bin/env python3
"""Select the 798-locus dNMF population panel and 2,054-locus backup set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR / "chr_str_var_gt_0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--loading-percentile", type=float, default=0.995)
    parser.add_argument("--minimum-specificity", type=float, default=0.40)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=PROJECT_ROOT
        / "olive_str/annotation/O_europeana_strs.functional.promoter2kb.annotated.bed",
    )
    parser.add_argument(
        "--rst",
        type=Path,
        default=PROJECT_ROOT
        / "olive_str/population_structure_snp_str/subspecies_rst_distribution/"
        "source_data/locus_rst.tsv.gz",
    )
    return parser.parse_args()


def load_annotation(path: Path) -> pd.DataFrame:
    annotation = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "#chrom",
            "start",
            "end",
            "motif_length",
            "motif",
            "functional_region",
            "gene_id",
            "gene_name",
        ],
        dtype={"#chrom": str, "start": int, "end": int},
    )
    annotation["str_id"] = annotation["#chrom"] + "_" + annotation["start"].astype(str)
    if annotation["str_id"].duplicated().any():
        raise ValueError("Functional annotation contains duplicate STR IDs")
    return annotation.drop(columns=["#chrom"])


def build_table(
    ids: pd.Index,
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    importance: pd.DataFrame,
    specificity: pd.Series,
    passes: pd.DataFrame,
    qc: pd.DataFrame,
    annotation: pd.DataFrame,
    rst: pd.DataFrame,
) -> pd.DataFrame:
    selected = importance.loc[ids]
    primary = selected.idxmax(axis=1)
    component_index = np.array([int(value[1:]) - 1 for value in primary])
    row_index = np.arange(len(ids))
    positive_values = positive.loc[ids].to_numpy()[row_index, component_index]
    negative_values = negative.loc[ids].to_numpy()[row_index, component_index]
    importance_values = selected.to_numpy()[row_index, component_index]
    percentile = importance.rank(pct=True).loc[ids].to_numpy()[row_index, component_index]
    selected_components = passes.loc[ids].apply(
        lambda row: ",".join(row.index[row].tolist()), axis=1
    )
    table = pd.DataFrame(
        {
            "str_id": ids,
            "primary_component": primary.to_numpy(),
            "loading_direction": np.where(
                positive_values >= negative_values, "positive", "negative"
            ),
            "normalized_positive_weight": positive_values,
            "normalized_negative_weight": negative_values,
            "component_importance": importance_values,
            "component_importance_percentile": percentile,
            "component_specificity": specificity.loc[ids].to_numpy(),
            "n_components_passing_top_0.5pct": passes.loc[ids].sum(axis=1).to_numpy(),
            "selected_components": selected_components.to_numpy(),
        }
    )
    table = table.merge(qc, on="str_id", how="left", validate="one_to_one")
    table["call_rate"] = table["n_called"] / 212
    table = table.merge(annotation, on="str_id", how="left", validate="one_to_one")
    table = table.merge(rst, on="str_id", how="left", validate="one_to_one")
    table["top5_subspecies_rst"] = table["rst"].ge(0.403389)
    return table.sort_values(
        ["primary_component", "component_importance_percentile", "component_specificity"],
        ascending=[True, False, False],
        kind="stable",
    )


def main() -> None:
    args = parse_args()
    source_dir = args.analysis_dir / "source_data"
    positive_raw = pd.read_csv(
        source_dir / "best_k9_pos_weights.tsv.gz", sep="\t", index_col="str_id"
    )
    negative_raw = pd.read_csv(
        source_dir / "best_k9_neg_weights.tsv.gz", sep="\t", index_col="str_id"
    )
    if not positive_raw.index.equals(negative_raw.index):
        raise ValueError("Positive and negative weight matrices have different locus orders")
    if positive_raw.columns.tolist() != negative_raw.columns.tolist():
        raise ValueError("Positive and negative weight matrices have different components")

    positive = positive_raw / positive_raw.sum(axis=0)
    negative = negative_raw / negative_raw.sum(axis=0)
    importance = np.maximum(positive, negative)
    specificity = importance.max(axis=1) / importance.sum(axis=1)
    thresholds = importance.quantile(args.loading_percentile)
    passes = importance.ge(thresholds, axis=1)
    backup_ids = importance.index[passes.any(axis=1)]
    panel_ids = importance.index[passes.any(axis=1) & specificity.ge(args.minimum_specificity)]

    qc = pd.read_csv(
        source_dir / "input_locus_qc.tsv.gz", sep="\t", dtype={"str_id": str}
    )
    annotation = load_annotation(args.annotation)
    rst = pd.read_csv(args.rst, sep="\t", usecols=["str_id", "rst"], dtype={"str_id": str})
    backup = build_table(
        backup_ids, positive, negative, importance, specificity, passes, qc, annotation, rst
    )
    panel = build_table(
        panel_ids, positive, negative, importance, specificity, passes, qc, annotation, rst
    )
    if len(backup) != 2054 or len(panel) != 798:
        raise ValueError(
            f"Expected backup/panel sizes 2054/798, observed {len(backup)}/{len(panel)}"
        )
    backup.insert(1, "in_population_informative_panel", backup["str_id"].isin(panel_ids))
    panel.to_csv(
        source_dir / "population_informative_panel_798.tsv", sep="\t", index=False
    )
    backup.to_csv(
        source_dir / "population_informative_backup_2054.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "component": importance.columns,
            "top_0.5pct_threshold": thresholds.to_numpy(),
            "backup_loci_assigned": backup["primary_component"]
            .value_counts()
            .reindex(importance.columns, fill_value=0)
            .to_numpy(),
            "panel_loci_assigned": panel["primary_component"]
            .value_counts()
            .reindex(importance.columns, fill_value=0)
            .to_numpy(),
        }
    ).to_csv(
        source_dir / "population_informative_panel_component_summary.tsv",
        sep="\t",
        index=False,
    )
    print(f"Population-informative panel: {len(panel):,} loci")
    print(f"Backup set: {len(backup):,} loci")


if __name__ == "__main__":
    main()
