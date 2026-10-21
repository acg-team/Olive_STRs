#!/usr/bin/env python3
"""Create a virtual HDF5 view over all PlantCAD2 embedding shards."""

import argparse
import os
from pathlib import Path
import h5py
import numpy as np
from common import load_config


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); root = Path(config["root"])
    paths = [root / f"results/embeddings/shard_{i:04d}.h5" for i in range(config["expected_shards"])]
    missing = [path for path in paths if not path.exists()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} shards; first: {missing[0]}")
    sizes = []
    for path in paths:
        with h5py.File(path) as h5: sizes.append(len(h5["allele_id"]))
    total = sum(sizes); shape = (total, len(config["layers"]), config["hidden_size"])
    layouts = {name: h5py.VirtualLayout(shape, dtype=np.float32) for name in ("whole_mean", "str_mean")}
    offset = 0
    for path, size in zip(paths, sizes):
        for name in layouts:
            layouts[name][offset:offset + size] = h5py.VirtualSource(str(path), name, shape=(size, shape[1], shape[2]))
        offset += size
    output = root / "results/embeddings/master.h5"; partial = output.with_suffix(".h5.partial")
    if output.exists() and not args.force: raise FileExistsError(f"{output} exists; use --force")
    string = h5py.string_dtype("utf-8"); values = {k: [] for k in ("allele_id", "str_id", "allele_repeats", "observed", "is_reference", "sequence_length")}
    for path in paths:
        with h5py.File(path) as h5:
            values["allele_id"].extend(h5["allele_id"].asstr()[:]); values["str_id"].extend(h5["str_id"].asstr()[:])
            for key in ("allele_repeats", "observed", "is_reference", "sequence_length"): values[key].append(h5[key][:])
    with h5py.File(partial, "w", libver="latest") as h5:
        for name, layout in layouts.items(): h5.create_virtual_dataset(name, layout)
        h5.create_dataset("allele_id", data=np.asarray(values["allele_id"], object), dtype=string)
        h5.create_dataset("str_id", data=np.asarray(values["str_id"], object), dtype=string)
        for key in ("allele_repeats", "observed", "is_reference", "sequence_length"): h5.create_dataset(key, data=np.concatenate(values[key]))
        h5.attrs["layers"] = config["layers"]; h5.attrs["model_id"] = config["model_id"]; h5.attrs["model_revision"] = config["model_revision"]
    os.replace(partial, output); print(f"Wrote {output} with {total:,} alleles")


if __name__ == "__main__": main()
