#!/usr/bin/env python3
"""Validate smoke or production PlantCAD2 embedding outputs."""

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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def finite_in_chunks(dataset, chunk=256):
    return all(np.isfinite(dataset[start:start + chunk]).all() for start in range(0, len(dataset), chunk))


def validate_file(path, expected_ids, config):
    with h5py.File(path) as h5:
        ids = h5["allele_id"].asstr()[:].tolist(); n = len(ids)
        if ids != expected_ids: raise RuntimeError(f"{path.name}: allele IDs/order differ from catalog")
        for key in ("whole_mean", "str_mean"):
            expected = (n, len(config["layers"]), config["hidden_size"])
            if h5[key].shape != expected: raise RuntimeError(f"{path.name}: {key} {h5[key].shape} != {expected}")
            if not finite_in_chunks(h5[key]): raise RuntimeError(f"{path.name}: {key} has non-finite values")
        if list(h5.attrs["layers"]) != config["layers"]: raise RuntimeError(f"{path.name}: layer mismatch")
        if h5.attrs["model_revision"] != config["model_revision"]: raise RuntimeError(f"{path.name}: revision mismatch")
        if "A100" not in h5.attrs["gpu_name"]: raise RuntimeError(f"{path.name}: not generated on A100")
        return ids


def production(config):
    catalog = Path(config["catalog_root"]); expected = defaultdict(list)
    with gzip.open(catalog / "inputs/alleles.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle): expected[int(row["shard"])].append(row["allele_id"])
    files = [ROOT / f"results/embeddings/shard_{i:04d}.h5" for i in range(config["expected_shards"])]
    missing = [str(path) for path in files if not path.exists()]
    if missing: raise RuntimeError(f"Missing {len(missing)} shards; first {missing[0]}")
    partials = list((ROOT / "results/embeddings").glob("*.partial"))
    if partials: raise RuntimeError(f"Partial embedding files remain: {partials[:3]}")
    all_ids = []
    for shard, path in enumerate(files): all_ids.extend(validate_file(path, expected[shard], config))
    if len(all_ids) != config["expected_alleles"] or len(set(all_ids)) != len(all_ids):
        raise RuntimeError("Total or unique allele count mismatch")
    payload = {
        "status": "PASS", "shards": len(files), "alleles": len(all_ids), "unique_alleles": len(set(all_ids)),
        "model_revision": config["model_revision"],
        "input_checksums": {name: digest(catalog / "inputs" / name) for name in ("loci.csv.gz", "alleles.csv.gz", "shards.csv")},
    }
    output = ROOT / "results/embedding_validation.json"
    output.with_suffix(".json.partial").write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(output.with_suffix(".json.partial"), output); print(json.dumps(payload, indent=2))


def smoke(config):
    benchmark = json.loads((ROOT / "results/smoke/benchmark.json").read_text())
    if benchmark["status"] != "PASS" or "A100" not in benchmark["gpu_name"]:
        raise RuntimeError("Smoke benchmark did not pass on A100")
    path = ROOT / "results/smoke/smoke_embeddings.h5"
    with h5py.File(path) as h5:
        if len(h5["allele_id"]) != config["smoke_alleles"]: raise RuntimeError("Smoke allele count mismatch")
        if 0 not in h5["allele_repeats"][:]: raise RuntimeError("Smoke set lacks zero-repeat allele")
        if h5["sequence_length"][:].max() != 1060: raise RuntimeError("Smoke set lacks longest allele")
        for key in ("whole_mean", "str_mean"):
            if h5[key].shape != (config["smoke_alleles"], len(config["layers"]), config["hidden_size"]):
                raise RuntimeError(f"Smoke {key} shape mismatch")
            if not finite_in_chunks(h5[key]): raise RuntimeError(f"Smoke {key} non-finite")
    output = ROOT / "results/smoke/validation.json"
    output.write_text(json.dumps({"status": "PASS", "alleles": config["smoke_alleles"], "benchmark": benchmark}, indent=2) + "\n")
    print("PLANTCAD2_GENOME_WIDE_SMOKE_VALIDATION=PASS")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    smoke(config) if args.smoke else production(config)


if __name__ == "__main__": main()
