#!/usr/bin/env python3
"""Create a virtual HDF5 view over all GPN embedding shards."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["root"])
    paths = [
        root / f"results/embeddings/shard_{index:04d}.h5"
        for index in range(config["expected_shards"])
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} shards; first: {missing[0]}")
    sizes = []
    for path in paths:
        with h5py.File(path) as handle:
            sizes.append(len(handle["allele_id"]))
    total = sum(sizes)
    shape = (total, config["hidden_size"])
    layouts = {
        name: h5py.VirtualLayout(shape, dtype=np.float32)
        for name in ("whole_mean", "str_mean")
    }
    offset = 0
    for path, size in zip(paths, sizes):
        for name in layouts:
            layouts[name][offset : offset + size] = h5py.VirtualSource(
                str(path), name, shape=(size, shape[1])
            )
        offset += size
    output = root / "results/embeddings/master.h5"
    partial = output.with_suffix(".h5.partial")
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; use --force")
    string = h5py.string_dtype("utf-8")
    values = {
        key: []
        for key in (
            "allele_id",
            "str_id",
            "allele_repeats",
            "observed",
            "is_reference",
            "repeat_bp",
            "right_flank_bp",
            "strand_str_mean_l2",
        )
    }
    for path in paths:
        with h5py.File(path) as handle:
            values["allele_id"].extend(handle["allele_id"].asstr()[:])
            values["str_id"].extend(handle["str_id"].asstr()[:])
            for key in values.keys() - {"allele_id", "str_id"}:
                values[key].append(handle[key][:])
    with h5py.File(partial, "w", libver="latest") as handle:
        for name, layout in layouts.items():
            handle.create_virtual_dataset(name, layout)
        handle.create_dataset(
            "allele_id", data=np.asarray(values["allele_id"], object), dtype=string
        )
        handle.create_dataset(
            "str_id", data=np.asarray(values["str_id"], object), dtype=string
        )
        for key in values.keys() - {"allele_id", "str_id"}:
            handle.create_dataset(key, data=np.concatenate(values[key]))
        handle.attrs["model_id"] = config["model_id"]
        handle.attrs["model_revision"] = config["model_revision"]
        handle.attrs["embedding_layer"] = config["embedding_layer"]
        handle.attrs["sequence_design"] = (
            "fixed 256-bp left flank; allele-specific STR; right flank trimmed to 512 bp"
        )
    os.replace(partial, output)
    print(f"Wrote {output} with {total:,} alleles")


if __name__ == "__main__":
    main()
