#!/usr/bin/env python3
"""Create one virtual HDF5 view over all completed embedding shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from common import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config"); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    config = load_config(args.config); root = Path(config["root"]); directory = root / "results/embeddings"
    manifest = json.loads((root / "results/preparation_manifest.json").read_text())
    paths = [directory / f"shard_{i:04d}.h5" for i in range(manifest["shards"])]
    missing = [str(path) for path in paths if not path.exists()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} embedding shards; first: {missing[0]}")
    sizes = []
    for path in paths:
        with h5py.File(path) as h5: sizes.append(len(h5["allele_repeats"]))
    total, layers, hidden = sum(sizes), len(config["layers"]), 1500
    layouts = {name: h5py.VirtualLayout((total, layers, hidden), dtype=np.float32) for name in ("whole_mean", "str_mean")}
    offset = 0
    for path, size in zip(paths, sizes):
        for name in layouts:
            source = h5py.VirtualSource(str(path), name, shape=(size, layers, hidden))
            layouts[name][offset:offset + size] = source
        offset += size
    output = directory / "master.h5"; partial = output.with_suffix(".h5.partial")
    if output.exists() and not args.force: raise FileExistsError(f"{output} exists; use --force")
    string = h5py.string_dtype("utf-8")
    allele_ids, loci, repeats, observed, references = [], [], [], [], []
    for path in paths:
        with h5py.File(path) as h5:
            allele_ids.extend(h5["allele_id"].asstr()[:]); loci.extend(h5["str_id"].asstr()[:])
            repeats.append(h5["allele_repeats"][:]); observed.append(h5["observed"][:]); references.append(h5["is_reference"][:])
    with h5py.File(partial, "w", libver="latest") as h5:
        for name, layout in layouts.items(): h5.create_virtual_dataset(name, layout)
        h5.create_dataset("allele_id", data=np.asarray(allele_ids, dtype=object), dtype=string)
        h5.create_dataset("str_id", data=np.asarray(loci, dtype=object), dtype=string)
        h5.create_dataset("allele_repeats", data=np.concatenate(repeats))
        h5.create_dataset("observed", data=np.concatenate(observed))
        h5.create_dataset("is_reference", data=np.concatenate(references))
        h5.attrs["layers"] = config["layers"]; h5.attrs["model_revision"] = config["model_revision"]
        h5.attrs["weights_revision"] = config.get("weights_revision", config["model_revision"])
    os.replace(partial, output); print(f"Wrote {output} with {total:,} alleles")


if __name__ == "__main__": main()
