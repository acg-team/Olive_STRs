#!/usr/bin/env python3
"""Create the deterministic common 1,000-locus robustness input set."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "shared_inputs"
CATALOG = ROOT.parent / "03_embeddings/agront/inputs"
MATCHED = ROOT.parent / "04_cross_model_comparison/results/matched_per_locus_regressions.csv.gz"
SEED = 20260724
N_LOCI = 1000
N_SHARDS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eligible = pd.read_csv(MATCHED, usecols=["str_id"]).str_id.drop_duplicates().sort_values()
    if len(eligible) != 62847:
        raise RuntimeError(f"Expected 62,847 eligible loci, found {len(eligible):,}")
    rng = np.random.default_rng(SEED)
    selected_ids = sorted(rng.choice(eligible.to_numpy(), size=N_LOCI, replace=False).tolist())
    selected_set = set(selected_ids)

    loci = pd.read_csv(CATALOG / "loci.csv.gz")
    alleles = pd.read_csv(CATALOG / "alleles.csv.gz")
    loci = loci[loci.str_id.isin(selected_set)].copy()
    alleles = alleles[alleles.str_id.isin(selected_set)].copy()
    if loci.str_id.nunique() != N_LOCI:
        raise RuntimeError("Selected loci are missing from the catalog")

    checks = alleles.groupby("str_id").apply(
        lambda frame: pd.Series({
            "reference_count": int(frame.is_reference.astype(bool).sum()),
            "observed_nonreference_lengths": int(
                frame.loc[
                    frame.observed.astype(bool) & ~frame.is_reference.astype(bool),
                    "allele_repeats",
                ].nunique()
            ),
            "allele_count": len(frame),
        }),
        include_groups=False,
    )
    if (checks.reference_count != 1).any() or (checks.observed_nonreference_lengths < 3).any():
        raise RuntimeError("Sample contains a locus that is not regression eligible")

    loads = np.zeros(N_SHARDS, dtype=int)
    assignments: dict[str, int] = {}
    ordered = checks.reset_index().sort_values(
        ["allele_count", "str_id"], ascending=[False, True]
    )
    for row in ordered.itertuples(index=False):
        shard = int(np.argmin(loads))
        assignments[row.str_id] = shard
        loads[shard] += int(row.allele_count)

    loci["robustness_shard"] = loci.str_id.map(assignments).astype(int)
    alleles = alleles.rename(columns={"shard": "source_shard", "row_in_shard": "source_row_in_shard"})
    alleles["robustness_shard"] = alleles.str_id.map(assignments).astype(int)
    alleles = alleles.sort_values(["robustness_shard", "str_id", "allele_repeats", "allele_id"])
    loci = loci.sort_values(["robustness_shard", "str_id"])

    write_csv_gz(loci, OUT / "loci.csv.gz")
    write_csv_gz(alleles, OUT / "alleles.csv.gz")
    pd.DataFrame({"str_id": selected_ids}).to_csv(OUT / "selected_loci.csv", index=False)
    checks.reset_index().merge(
        loci[["str_id", "period", "motif", "robustness_shard"]], on="str_id", validate="one_to_one"
    ).to_csv(OUT / "locus_validation.csv", index=False)

    manifest = {
        "status": "PASS",
        "seed": SEED,
        "eligible_loci": int(len(eligible)),
        "selected_loci": int(loci.str_id.nunique()),
        "selected_alleles": int(len(alleles)),
        "robustness_shards": N_SHARDS,
        "alleles_per_shard": loads.tolist(),
        "source_sha256": {
            "cross_model/matched_per_locus_regressions.csv.gz": sha256(MATCHED),
            "catalog/loci.csv.gz": sha256(CATALOG / "loci.csv.gz"),
            "catalog/alleles.csv.gz": sha256(CATALOG / "alleles.csv.gz"),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
