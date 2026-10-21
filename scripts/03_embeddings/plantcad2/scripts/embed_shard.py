#!/usr/bin/env python3
"""Generate four-layer PlantCAD2 embeddings for one prepared ArgoNT catalog shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from common import ROOT, load_config, normalize_dna


@dataclass
class Allele:
    allele_id: str
    locus: str
    repeats: int
    observed: bool
    is_reference: bool
    motif: str
    left: str
    right: str

    @property
    def sequence(self):
        return normalize_dna(self.left + self.motif * self.repeats + self.right)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-benchmark", action="store_true")
    return parser.parse_args()


def fix_and_register_ties(model, forward_pattern="fwd", reverse_pattern="rev"):
    tied = list(getattr(model, "_tied_weights_keys", None) or [])
    modules = dict(model.named_modules())
    for name, forward in modules.items():
        if forward_pattern not in name or "proj" not in name:
            continue
        reverse_name = name.replace(forward_pattern, reverse_pattern)
        if reverse_name not in modules:
            continue
        reverse = modules[reverse_name]
        if hasattr(forward, "weight"):
            reverse.weight = forward.weight; tied.append(f"{reverse_name}.weight")
        if hasattr(forward, "bias") and forward.bias is not None:
            reverse.bias = forward.bias; tied.append(f"{reverse_name}.bias")
    model._tied_weights_keys = sorted(set(tied))
    return model


def combine_rcps(raw, hidden_size):
    if raw.shape[-1] != hidden_size * 2:
        raise RuntimeError(f"Unexpected RCPS hidden width {raw.shape[-1]}")
    return (raw[..., :hidden_size] + raw[..., hidden_size:].flip(dims=[1])) / 2


def region_slice(left_flank, repeat_bases):
    return slice(left_flank, left_flank + repeat_bases) if repeat_bases else slice(left_flank - 1, left_flank + 1)


def read_catalog(config):
    root = Path(config["catalog_root"]); loci = {}
    with gzip.open(root / "inputs/loci.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle): loci[row["str_id"]] = row
    alleles = []
    with gzip.open(root / "inputs/alleles.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle): alleles.append(row)
    return loci, alleles


def make_alleles(loci, rows):
    output = []
    for row in rows:
        locus = loci[row["str_id"]]
        output.append(Allele(
            row["allele_id"], row["str_id"], int(row["allele_repeats"]),
            bool(int(row["observed"])), bool(int(row["is_reference"])),
            locus["motif"], locus["left_flank"], locus["right_flank"],
        ))
    return output


def load_production_alleles(config, shard):
    loci, rows = read_catalog(config)
    selected = [row for row in rows if int(row["shard"]) == shard]
    if not selected: raise ValueError(f"Shard {shard} contains no alleles")
    return make_alleles(loci, selected)


def load_smoke_alleles(config):
    loci, rows = read_catalog(config); count = int(config["smoke_alleles"])
    selected, seen = [], set()
    def add(row):
        if row["allele_id"] not in seen:
            selected.append(row); seen.add(row["allele_id"])
    add(rows[0])
    add(next(row for row in rows if int(row["allele_repeats"]) == 0))
    add(max(rows, key=lambda row: int(row["sequence_length"])))
    add(next(row for row in rows if "N" in (loci[row["str_id"]]["left_flank"] + loci[row["str_id"]]["right_flank"]).upper()))
    for index in np.linspace(0, len(rows) - 1, count, dtype=int):
        add(rows[index])
        if len(selected) == count: break
    for row in rows:
        if len(selected) == count: break
        add(row)
    if len(selected) != count: raise RuntimeError("Could not assemble smoke allele set")
    return make_alleles(loci, selected)


def load_model(config):
    source = Path(config["local_model_path"]); dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        source, trust_remote_code=True, local_files_only=True, torch_dtype=dtype,
    )
    model = fix_and_register_ties(model)
    if model.num_parameters() != 694385664:
        raise RuntimeError(f"Incorrect tied parameter count: {model.num_parameters()}")
    return tokenizer, model.eval().cuda()


def embed(alleles, tokenizer, model, config, batch_size):
    layers = tuple(config["layers"]); hidden_size = int(config["hidden_size"])
    left_flank = int(config["left_flank_bp"])
    sequences = [allele.sequence for allele in alleles]
    groups = defaultdict(list)
    for index, sequence in enumerate(sequences): groups[len(sequence)].append(index)
    whole = np.empty((len(alleles), len(layers), hidden_size), np.float32)
    region = np.empty_like(whole)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    completed = 0; next_report = 500
    for length in sorted(groups):
        indexes = groups[length]
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start:start + batch_size]
            batch_sequences = [sequences[index] for index in batch_indexes]
            encoded = tokenizer(
                batch_sequences, add_special_tokens=True, return_tensors="pt",
                return_attention_mask=False, return_token_type_ids=False, padding=False,
            )["input_ids"]
            if encoded.shape != (len(batch_indexes), length):
                raise RuntimeError(f"Tokenizer is not base-for-token: {encoded.shape} versus {(len(batch_indexes), length)}")
            with torch.inference_mode():
                output = model(input_ids=encoded.cuda(), output_hidden_states=True)
            for layer_column, layer in enumerate(layers):
                hidden = combine_rcps(output.hidden_states[layer].float(), hidden_size)
                for row, allele_index in enumerate(batch_indexes):
                    repeat_bases = len(alleles[allele_index].motif) * alleles[allele_index].repeats
                    whole[allele_index, layer_column] = hidden[row].mean(0).cpu().numpy()
                    region[allele_index, layer_column] = hidden[row, region_slice(left_flank, repeat_bases)].mean(0).cpu().numpy()
            completed += len(batch_indexes)
            if completed >= next_report or completed == len(alleles):
                print(f"embedded={completed}/{len(alleles)} batch_size={batch_size}", flush=True)
                while next_report <= completed: next_report += 500
            del output, encoded
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if not np.isfinite(whole).all() or not np.isfinite(region).all():
        raise RuntimeError("Non-finite embeddings")
    return whole, region, {
        "batch_size": batch_size, "seconds": elapsed,
        "alleles_per_second": len(alleles) / elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def write_h5(path, alleles, whole, region, config, benchmark=None):
    partial = path.with_suffix(path.suffix + ".partial"); path.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists(): partial.unlink()
    string = h5py.string_dtype("utf-8")
    with h5py.File(partial, "w") as h5:
        h5.create_dataset("allele_id", data=np.asarray([x.allele_id for x in alleles], object), dtype=string)
        h5.create_dataset("str_id", data=np.asarray([x.locus for x in alleles], object), dtype=string)
        h5.create_dataset("allele_repeats", data=np.asarray([x.repeats for x in alleles], np.int32))
        h5.create_dataset("observed", data=np.asarray([x.observed for x in alleles], np.bool_))
        h5.create_dataset("is_reference", data=np.asarray([x.is_reference for x in alleles], np.bool_))
        h5.create_dataset("sequence_length", data=np.asarray([len(x.sequence) for x in alleles], np.int32))
        h5.create_dataset("whole_mean", data=whole, compression="gzip", shuffle=True)
        h5.create_dataset("str_mean", data=region, compression="gzip", shuffle=True)
        h5.attrs["layers"] = config["layers"]
        h5.attrs["model_id"] = config["model_id"]; h5.attrs["model_revision"] = config["model_revision"]
        h5.attrs["dtype"] = config["dtype"]; h5.attrs["gpu_name"] = torch.cuda.get_device_name(0)
        h5.attrs["sequence_design"] = "500bp_left+variable_STR+500bp_right"
        h5.attrs["pooling"] = "whole mean and actual STR-region mean"
        h5.attrs["zero_repeat_str_pooling"] = "two positions adjacent to deletion junction"
        h5.attrs["n_token_policy"] = "N maps to tokenizer [UNK]"
        if benchmark:
            for key, value in benchmark.items(): h5.attrs[key] = value
    os.replace(partial, path)


def smoke_benchmark(config, tokenizer, model):
    alleles = load_smoke_alleles(config); results = []; reference = None; candidates = []
    output_dir = ROOT / "results/smoke"; output_dir.mkdir(parents=True, exist_ok=True)
    for batch_size in config["benchmark_batch_sizes"]:
        try:
            whole, region, timing = embed(alleles, tokenizer, model, config, int(batch_size))
            if reference is None:
                reference = (whole.copy(), region.copy())
                cosine_min = 1.0; relative_l2 = 0.0
            else:
                current = np.concatenate([whole.reshape(len(alleles), -1), region.reshape(len(alleles), -1)], axis=1)
                baseline = np.concatenate([reference[0].reshape(len(alleles), -1), reference[1].reshape(len(alleles), -1)], axis=1)
                cosine = np.sum(current * baseline, 1) / np.maximum(np.linalg.norm(current, axis=1) * np.linalg.norm(baseline, axis=1), 1e-12)
                cosine_min = float(cosine.min())
                relative_l2 = float(np.linalg.norm(current - baseline) / np.maximum(np.linalg.norm(baseline), 1e-12))
            record = timing | {"status": "PASS", "minimum_cosine_to_batch1": cosine_min, "relative_l2_to_batch1": relative_l2}
            results.append(record)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if timing["peak_gpu_memory_gib"] < 0.8 * total and cosine_min >= 0.9999 and relative_l2 <= 0.01:
                candidates.append((timing["alleles_per_second"], int(batch_size), whole, region, timing))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); results.append({"batch_size": int(batch_size), "status": "OOM"})
    if not candidates: raise RuntimeError("No smoke benchmark batch size passed")
    _, selected, whole, region, timing = max(candidates, key=lambda item: item[0])
    payload = {
        "status": "PASS", "recommended_batch_size": selected,
        "gpu_name": torch.cuda.get_device_name(0), "results": results,
    }
    (output_dir / "benchmark.json.partial").write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(output_dir / "benchmark.json.partial", output_dir / "benchmark.json")
    write_h5(output_dir / "smoke_embeddings.h5", alleles, whole, region, config, timing)
    print(json.dumps(payload, indent=2), flush=True)


def main():
    args = arguments(); config = load_config(args.config)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    tokenizer, model = load_model(config)
    if args.smoke_benchmark:
        smoke_benchmark(config, tokenizer, model); return
    if args.shard is None: raise ValueError("--shard is required outside smoke mode")
    output = ROOT / f"results/embeddings/shard_{args.shard:04d}.h5"
    if output.exists() and not args.force:
        print(f"Complete output already exists: {output}"); return
    benchmark_path = ROOT / "results/smoke/benchmark.json"
    batch_size = args.batch_size
    if batch_size is None and benchmark_path.exists():
        batch_size = json.loads(benchmark_path.read_text())["recommended_batch_size"]
    batch_size = int(batch_size or config["batch_size_fallback"])
    alleles = load_production_alleles(config, args.shard)
    whole, region, timing = embed(alleles, tokenizer, model, config, batch_size)
    write_h5(output, alleles, whole, region, config, timing)
    print(f"Wrote {output}: alleles={len(alleles)} shape={whole.shape} timing={timing}", flush=True)


if __name__ == "__main__":
    main()
