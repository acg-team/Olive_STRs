#!/usr/bin/env python3
"""Validate smoke or production genome-wide GPN embedding outputs."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from common import ROOT, load_config


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite_in_chunks(dataset, chunk=512):
    return all(
        np.isfinite(dataset[start : start + chunk]).all()
        for start in range(0, len(dataset), chunk)
    )


def validate_file(path, expected_ids, config):
    with h5py.File(path) as handle:
        ids = handle["allele_id"].asstr()[:].tolist()
        n = len(ids)
        if ids != expected_ids:
            raise RuntimeError(f"{path.name}: allele IDs/order differ from catalog")
        for key in ("whole_mean", "str_mean"):
            expected = (n, config["hidden_size"])
            if handle[key].shape != expected:
                raise RuntimeError(f"{path.name}: {key} {handle[key].shape} != {expected}")
            if not finite_in_chunks(handle[key]):
                raise RuntimeError(f"{path.name}: {key} has non-finite values")
        if handle.attrs["model_revision"] != config["model_revision"]:
            raise RuntimeError(f"{path.name}: revision mismatch")
        if not str(handle.attrs["embedding_layer"]).startswith("last_hidden_state"):
            raise RuntimeError(f"{path.name}: wrong embedding layer")
        if "A100" not in handle.attrs["gpu_name"]:
            raise RuntimeError(f"{path.name}: not generated on A100")
        if (handle["right_flank_bp"][:] < 0).any():
            raise RuntimeError(f"{path.name}: negative right flank")
        return ids


def smoke(config):
    benchmark = json.loads((ROOT / "results/smoke/benchmark.json").read_text())
    if benchmark["status"] != "PASS" or "A100" not in benchmark["gpu_name"]:
        raise RuntimeError("Smoke benchmark did not pass on A100")
    path = ROOT / "results/smoke/smoke_embeddings.h5"
    with h5py.File(path) as handle:
        if len(handle["allele_id"]) != config["smoke_alleles"]:
            raise RuntimeError("Smoke allele count mismatch")
        if 0 not in handle["allele_repeats"][:]:
            raise RuntimeError("Smoke set lacks a zero-repeat allele")
        if handle["repeat_bp"][:].max() != 60:
            raise RuntimeError("Smoke set lacks the longest allele")
        for key in ("whole_mean", "str_mean"):
            if handle[key].shape != (config["smoke_alleles"], config["hidden_size"]):
                raise RuntimeError(f"Smoke {key} shape mismatch")
            if not finite_in_chunks(handle[key]):
                raise RuntimeError(f"Smoke {key} has non-finite values")
        norms = np.linalg.norm(handle["str_mean"][:], axis=1)
        if (norms <= 0).any():
            raise RuntimeError("Smoke STR embeddings include a zero-norm row")
    output = ROOT / "results/smoke/validation.json"
    output.write_text(
        json.dumps(
            {"status": "PASS", "alleles": config["smoke_alleles"], "benchmark": benchmark},
            indent=2,
        )
        + "\n"
    )
    print("GPN_GENOME_WIDE_SMOKE_VALIDATION=PASS")


def production(config):
    catalog = Path(config["catalog_root"])
    expected = defaultdict(list)
    with gzip.open(catalog / "inputs/alleles.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[int(row["shard"])].append(row["allele_id"])
    files = [
        ROOT / f"results/embeddings/shard_{index:04d}.h5"
        for index in range(config["expected_shards"])
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} shards; first {missing[0]}")
    partials = list((ROOT / "results/embeddings").glob("*.partial"))
    if partials:
        raise RuntimeError(f"Partial embedding files remain: {partials[:3]}")
    all_ids = []
    for shard, path in enumerate(files):
        all_ids.extend(validate_file(path, expected[shard], config))
    if len(all_ids) != config["expected_alleles"] or len(set(all_ids)) != len(all_ids):
        raise RuntimeError("Total or unique allele count mismatch")
    payload = {
        "status": "PASS",
        "shards": len(files),
        "alleles": len(all_ids),
        "unique_alleles": len(set(all_ids)),
        "model_revision": config["model_revision"],
        "input_checksums": {
            name: digest(catalog / "inputs" / name)
            for name in ("loci.csv.gz", "alleles.csv.gz", "shards.csv")
        },
    }
    output = ROOT / "results/embedding_validation.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    smoke(config) if args.smoke else production(config)


if __name__ == "__main__":
    main()
