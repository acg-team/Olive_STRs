#!/usr/bin/env python3
"""Run matched reference-anchored permutation regressions for three models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("argont", "plantcad2", "gpn")
METRICS = ("l1", "l2", "cosine")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--max-loci", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def boundaries(values):
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    if len(np.unique(values)) != len(starts):
        raise RuntimeError("Loci are not contiguous within shard")
    return starts, stops


def distances(matrix, reference_index, alternate_indexes):
    reference = matrix[reference_index].astype(np.float64)
    alternate = matrix[alternate_indexes].astype(np.float64)
    delta = alternate - reference
    l1 = np.abs(delta).sum(axis=1)
    l2 = np.linalg.norm(delta, axis=1)
    denominator = np.maximum(
        np.linalg.norm(alternate, axis=1) * np.linalg.norm(reference), 1e-12
    )
    cosine = np.maximum(1 - (alternate @ reference) / denominator, 0)
    return np.column_stack([l1, l2, cosine])


def regression(x, y):
    xc = x - x.mean()
    yc = y - y.mean(axis=0)
    ssx = xc @ xc
    if ssx <= 0:
        raise RuntimeError("Predictor has zero variance")
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
        keys = rng.random((maximum, n))
        order = np.argsort(keys, axis=1)
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


def model_matrix(handle, model, start, stop, config):
    dataset = handle["str_mean"]
    if model == "gpn":
        if dataset.ndim != 2 or dataset.shape[1] != 512:
            raise RuntimeError("Unexpected GPN STR-mean shape")
        return dataset[start:stop]
    layer = config[f"{model}_layer"]
    layers = list(handle.attrs["layers"])
    if layer not in layers:
        raise RuntimeError(f"{model}: layer {layer} not found in {layers}")
    return dataset[start:stop, layers.index(layer), :]


def process_shard(shard, config, remaining):
    paths = {
        model: Path(config[f"{model}_root"])
        / f"results/embeddings/shard_{shard:04d}.h5"
        for model in MODELS
    }
    rows = []
    with (
        h5py.File(paths["argont"]) as argont,
        h5py.File(paths["plantcad2"]) as plantcad2,
        h5py.File(paths["gpn"]) as gpn,
    ):
        handles = {"argont": argont, "plantcad2": plantcad2, "gpn": gpn}
        allele_ids = argont["allele_id"].asstr()[:]
        str_ids = argont["str_id"].asstr()[:]
        repeats = argont["allele_repeats"][:].astype(np.int32)
        observed = argont["observed"][:].astype(bool)
        is_reference = argont["is_reference"][:].astype(bool)
        for model, handle in handles.items():
            if handle["allele_id"].asstr()[:].tolist() != allele_ids.tolist():
                raise RuntimeError(f"Shard {shard}: {model} allele IDs/order differ")
            if not np.array_equal(handle["allele_repeats"][:], repeats):
                raise RuntimeError(f"Shard {shard}: {model} repeat counts differ")
            if not np.array_equal(handle["observed"][:].astype(bool), observed):
                raise RuntimeError(f"Shard {shard}: {model} observed flags differ")
            if not np.array_equal(handle["is_reference"][:].astype(bool), is_reference):
                raise RuntimeError(f"Shard {shard}: {model} reference flags differ")

        starts, stops = boundaries(str_ids)
        for start, stop in zip(starts, stops):
            reference = np.flatnonzero(is_reference[start:stop])
            if len(reference) != 1:
                raise RuntimeError(f"{str_ids[start]}: reference count={len(reference)}")
            reference_index = int(reference[0])
            alternate_indexes = np.flatnonzero(
                observed[start:stop] & ~is_reference[start:stop]
            )
            alternate_repeats = repeats[start:stop][alternate_indexes]
            if (
                len(np.unique(alternate_repeats))
                < config["minimum_distinct_observed_alternate_alleles"]
            ):
                continue
            x = np.abs(
                alternate_repeats - repeats[start:stop][reference_index]
            ).astype(np.float64)
            columns = []
            for model in MODELS:
                matrix = model_matrix(
                    handles[model], model, start, stop, config
                )
                columns.append(distances(matrix, reference_index, alternate_indexes))
            y = np.concatenate(columns, axis=1)
            slope, intercept, r2, y_centered, ssx = regression(x, y)
            seed = config["random_seed"] ^ int.from_bytes(
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
            column = 0
            for model in MODELS:
                for metric in METRICS:
                    rows.append(
                        {
                            "str_id": str_ids[start],
                            "model": model,
                            "layer": config[f"{model}_layer"],
                            "pooling": "str_mean",
                            "metric": metric,
                            "predictor": "absolute_repeat_difference_from_reference",
                            "reference_included": False,
                            "n_alternate_alleles": len(alternate_indexes),
                            "n_unique_absolute_differences": len(np.unique(x)),
                            "reference_repeats": int(
                                repeats[start:stop][reference_index]
                            ),
                            "reference_observed": bool(
                                observed[start:stop][reference_index]
                            ),
                            "slope": slope[column],
                            "intercept": intercept[column],
                            "r2": r2[column],
                            "p_value": p_value[column],
                            "n_permutations": used,
                            "permutation_method": method,
                            "source_shard": shard,
                        }
                    )
                    column += 1
            if remaining is not None:
                remaining -= 1
                if remaining == 0:
                    return rows, remaining
    return rows, remaining


def main():
    args = arguments()
    config = load_config()
    output_dir = ROOT / ("results/smoke" if args.max_loci is not None else "results/shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"regressions_task_{args.task:02d}.csv.gz"
    if output.exists() and not args.force:
        print(f"Complete output exists: {output}")
        return
    all_rows = []
    remaining = args.max_loci
    for shard in range(args.task, config["expected_shards"], args.tasks):
        rows, remaining = process_shard(shard, config, remaining)
        all_rows.extend(rows)
        print(
            f"task={args.task} shard={shard} eligible_loci={len(all_rows) // 9}",
            flush=True,
        )
        if remaining == 0:
            break
    frame = pd.DataFrame(all_rows)
    expected_columns = 18
    if len(frame.columns) != expected_columns:
        raise RuntimeError(f"Unexpected output columns: {frame.columns.tolist()}")
    temporary = output.with_suffix(".csv.gz.partial")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(output)
    print(
        f"output={output} rows={len(frame)} loci={frame.str_id.nunique()} "
        "MATCHED_REFERENCE_PERMUTATION=PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
