#!/usr/bin/env python3
"""Compute reference-anchored STR embedding regressions for one ArgoNT shard."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import load_config


POOLINGS = ("whole_mean", "str_mean")
METRICS = ("l1", "l2", "cosine")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--max-loci", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def boundaries(values):
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    if len(np.unique(values)) != len(starts):
        raise RuntimeError("Loci are not contiguous within the embedding shard")
    return starts, stops


def distance_columns(handle, start, stop, reference, alternates, layers):
    labels, columns = [], []
    for pooling in POOLINGS:
        vectors = handle[pooling][start:stop]
        for layer_index, layer in enumerate(layers):
            matrix = vectors[:, layer_index].astype(np.float64)
            ref = matrix[reference]
            alt = matrix[alternates]
            delta = alt - ref
            l1 = np.abs(delta).sum(axis=1)
            l2 = np.linalg.norm(delta, axis=1)
            denominator = np.maximum(
                np.linalg.norm(alt, axis=1) * np.linalg.norm(ref), 1e-12
            )
            cosine = np.maximum(1 - (alt @ ref) / denominator, 0)
            for metric, values in (("l1", l1), ("l2", l2), ("cosine", cosine)):
                labels.append((pooling, int(layer), metric))
                columns.append(values)
    return labels, np.column_stack(columns)


def regression(x, y):
    xc = x - x.mean()
    yc = y - y.mean(axis=0)
    ssx = xc @ xc
    if ssx <= 0:
        raise RuntimeError("Absolute repeat difference has zero variance")
    slope = (xc @ yc) / ssx
    intercept = y.mean(axis=0) - slope * x.mean()
    fitted = x[:, None] * slope + intercept
    residual = ((y - fitted) ** 2).sum(axis=0)
    total = (yc**2).sum(axis=0)
    r2 = np.where(total > 0, 1 - residual / total, np.nan)
    return slope, intercept, r2, yc, ssx


def permutation_pvalues(x, y_centered, ssx, observed_slope, maximum, seed):
    n = len(x)
    factorial = math.factorial(n)
    if factorial <= maximum:
        permuted = np.asarray(list(itertools.permutations(x.tolist())), dtype=np.float64)
        correction = 0
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        order = np.argsort(rng.random((maximum, n)), axis=1)
        permuted = x[order]
        correction = 1
        method = "monte_carlo"
    permuted -= permuted.mean(axis=1, keepdims=True)
    slopes = (permuted @ y_centered) / ssx
    exceed = np.sum(
        np.abs(slopes) >= np.abs(observed_slope)[None, :] - 1e-12, axis=0
    )
    used = len(permuted)
    return (exceed + correction) / (used + correction), used, method


def main():
    args = arguments()
    config = load_config(args.config)
    root = Path(config["root"])
    embedding = root / f"results/embeddings/shard_{args.shard:04d}.h5"
    output_dir = root / (
        "results/reference_anchored_smoke"
        if args.max_loci is not None
        else "results/reference_anchored_shards"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"regression_{args.shard:04d}.csv.gz"
    if output.exists() and not args.force:
        print(f"Complete output already exists: {output}")
        return

    rows = []
    with h5py.File(embedding) as handle:
        layers = [int(value) for value in handle.attrs["layers"]]
        if layers != config["layers"]:
            raise RuntimeError(f"Embedding layers {layers} != config {config['layers']}")
        str_ids = handle["str_id"].asstr()[:]
        repeats = handle["allele_repeats"][:].astype(np.int32)
        observed = handle["observed"][:].astype(bool)
        is_reference = handle["is_reference"][:].astype(bool)
        starts, stops = boundaries(str_ids)
        eligible_count = 0
        for start, stop in zip(starts, stops):
            reference = np.flatnonzero(is_reference[start:stop])
            if len(reference) != 1:
                raise RuntimeError(f"{str_ids[start]}: reference count={len(reference)}")
            reference = int(reference[0])
            alternates = np.flatnonzero(
                observed[start:stop] & ~is_reference[start:stop]
            )
            alternate_repeats = repeats[start:stop][alternates]
            if len(np.unique(alternate_repeats)) < 3:
                continue
            x = np.abs(
                alternate_repeats - repeats[start:stop][reference]
            ).astype(np.float64)
            labels, y = distance_columns(
                handle, start, stop, reference, alternates, layers
            )
            slope, intercept, r2, y_centered, ssx = regression(x, y)
            seed = config.get("reference_analysis_random_seed", config["random_seed"]) ^ int.from_bytes(
                hashlib.sha256(str_ids[start].encode()).digest()[:8], "little"
            )
            p_value, used, method = permutation_pvalues(
                x,
                y_centered,
                ssx,
                slope,
                config["permutations"],
                seed,
            )
            for column, (pooling, layer, metric) in enumerate(labels):
                rows.append(
                    {
                        "str_id": str_ids[start],
                        "pooling": pooling,
                        "layer": layer,
                        "metric": metric,
                        "predictor": "absolute_repeat_difference_from_reference",
                        "reference_included": False,
                        "n_alternate_alleles": len(alternates),
                        "n_unique_absolute_differences": len(np.unique(x)),
                        "reference_repeats": int(repeats[start:stop][reference]),
                        "reference_observed": bool(observed[start:stop][reference]),
                        "slope": slope[column],
                        "intercept": intercept[column],
                        "r2": r2[column],
                        "p_value": p_value[column],
                        "n_permutations": used,
                        "permutation_method": method,
                        "source_shard": args.shard,
                    }
                )
            eligible_count += 1
            if args.max_loci is not None and eligible_count >= args.max_loci:
                break

    frame = pd.DataFrame(rows)
    if len(frame) != eligible_count * len(POOLINGS) * len(layers) * len(METRICS):
        raise RuntimeError("Unexpected regression row count")
    temporary = output.with_suffix(".csv.gz.partial")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(output)
    print(
        f"output={output} loci={eligible_count} rows={len(frame)} "
        "ARGONT_REFERENCE_ANCHORED_SHARD=PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
