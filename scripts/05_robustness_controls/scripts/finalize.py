#!/usr/bin/env python3
"""Validate completed robustness outputs and write checksums and index report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd


HERE = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    fixed_validation = json.loads(
        (HERE / "fixed_total_embedding_comparison/results/validation.json").read_text()
    )
    control_validation = json.loads(
        (HERE / "matched_indel_control/results/validation.json").read_text()
    )
    if fixed_validation["status"] != "PASS" or control_validation["status"] != "PASS":
        raise RuntimeError("A component validation did not pass")

    sample = pd.read_csv(HERE / "shared_inputs/selected_loci.csv")
    fixed = pd.read_csv(
        HERE / "fixed_total_embedding_comparison/results/current_vs_fixed_total_per_allele.csv.gz"
    )
    controls = pd.read_csv(
        HERE / "matched_indel_control/results/str_vs_control_per_locus.csv.gz"
    )
    if sample.str_id.nunique() != 1000:
        raise RuntimeError("Shared sample is not exactly 1,000 loci")
    if set(fixed.model) != {"ArgoNT", "PlantCAD2"}:
        raise RuntimeError("Fixed-total result has unexpected models")
    if set(controls.model) != {"ArgoNT", "PlantCAD2", "GPN"}:
        raise RuntimeError("Control result has unexpected models")

    embedding_files = sorted(HERE.glob("*/results/embeddings/*.h5"))
    if len(embedding_files) != 110:
        raise RuntimeError(f"Expected 110 embedding shards, found {len(embedding_files)}")
    for path in embedding_files:
        with h5py.File(path) as handle:
            if not len(handle["allele_id"]) or "whole_mean" not in handle:
                raise RuntimeError(f"Incomplete embedding file: {path}")

    included = []
    for pattern in (
        "README.md", "shared_inputs/*", "scripts/*.py", "slurm/*.slurm",
        "fixed_total_embedding_comparison/results/*",
        "fixed_total_embedding_comparison/figures/*",
        "fixed_total_embedding_comparison/report/*",
        "matched_indel_control/results/*",
        "matched_indel_control/figures/*",
        "matched_indel_control/report/*",
    ):
        included.extend(path for path in HERE.glob(pattern) if path.is_file())
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(HERE)}"
        for path in sorted(set(included), key=lambda x: str(x))
        if path.name != "SHA256SUMS"
    ]
    (HERE / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    report = """# Robustness results

Both prespecified robustness tests completed successfully on the same
deterministic 1,000-locus set.

- [Fixed-total direct embedding comparison](fixed_total_embedding_comparison/report/REPORT.md)
- [Current-context matched-indel control](matched_indel_control/report/REPORT.md)

The fixed-total test compares embedding vectors directly and does not perform
regression. The matched-indel test is separate and compares control slopes
against the existing current-setting STR slopes.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({
        "status": "PASS", "embedding_shards": len(embedding_files),
        "checksum_files": len(checksum_lines),
    }))


if __name__ == "__main__":
    main()
