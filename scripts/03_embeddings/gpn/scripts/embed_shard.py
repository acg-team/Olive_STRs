#!/usr/bin/env python3
"""Generate final-layer GPN embeddings for one prepared STR catalog shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import gpn.model  # noqa: F401 -- registers the GPN model with Transformers
import h5py
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from common import ROOT, load_config, normalize_dna, reverse_complement


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
    def repeat_bp(self):
        return len(self.motif) * self.repeats

    def sequence(self, context, left_flank):
        right_length = context - left_flank - self.repeat_bp
        if right_length < 0:
            raise ValueError(f"{self.allele_id}: {self.repeat_bp}-bp STR exceeds context")
        sequence = (
            normalize_dna(self.left)[-left_flank:]
            + normalize_dna(self.motif) * self.repeats
            + normalize_dna(self.right)[:right_length]
        )
        if len(sequence) != context:
            raise ValueError(f"{self.allele_id}: reconstructed length {len(sequence)} != {context}")
        return sequence


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-benchmark", action="store_true")
    return parser.parse_args()


def read_catalog(config):
    root = Path(config["catalog_root"])
    loci = {}
    with gzip.open(root / "inputs/loci.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            loci[row["str_id"]] = row
    rows = []
    with gzip.open(root / "inputs/alleles.csv.gz", "rt", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    return loci, rows


def make_alleles(loci, rows):
    result = []
    for row in rows:
        locus = loci[row["str_id"]]
        result.append(
            Allele(
                allele_id=row["allele_id"],
                locus=row["str_id"],
                repeats=int(row["allele_repeats"]),
                observed=bool(int(row["observed"])),
                is_reference=bool(int(row["is_reference"])),
                motif=normalize_dna(locus["motif"]),
                left=locus["left_flank"],
                right=locus["right_flank"],
            )
        )
    return result


def load_production_alleles(config, shard):
    loci, rows = read_catalog(config)
    selected = [row for row in rows if int(row["shard"]) == shard]
    if not selected:
        raise ValueError(f"Shard {shard} contains no alleles")
    return make_alleles(loci, selected)


def load_smoke_alleles(config):
    loci, rows = read_catalog(config)
    count = int(config["smoke_alleles"])
    selected, seen = [], set()

    def add(row):
        if row["allele_id"] not in seen:
            selected.append(row)
            seen.add(row["allele_id"])

    add(rows[0])
    add(next(row for row in rows if int(row["allele_repeats"]) == 0))
    add(max(rows, key=lambda row: int(row["sequence_length"])))
    add(
        next(
            row
            for row in rows
            if "N" in (loci[row["str_id"]]["left_flank"] + loci[row["str_id"]]["right_flank"]).upper()
        )
    )
    for index in np.linspace(0, len(rows) - 1, count, dtype=int):
        add(rows[index])
        if len(selected) == count:
            break
    for row in rows:
        if len(selected) == count:
            break
        add(row)
    if len(selected) != count:
        raise RuntimeError("Could not assemble the requested smoke set")
    return make_alleles(loci, selected)


def load_model(config):
    source = Path(config["local_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(source, local_files_only=True)
    if model.config.n_layers != 25 or model.config.hidden_size != config["hidden_size"]:
        raise RuntimeError(
            f"Unexpected architecture: layers={model.config.n_layers}, "
            f"hidden={model.config.hidden_size}"
        )
    return tokenizer, model.eval().cuda()


def str_slice(left_flank, repeat_bp):
    return (
        slice(left_flank, left_flank + repeat_bp)
        if repeat_bp
        else slice(left_flank - 1, left_flank + 1)
    )


def run_model(tokens, model, labels):
    with torch.inference_mode():
        hidden = model.model(input_ids=tokens.cuda()).last_hidden_state.float()
    finite = torch.isfinite(hidden).all(dim=(1, 2))
    if finite.all():
        return hidden
    affected = [labels[index] for index in torch.where(~finite)[0].cpu().tolist()]
    print(f"nonfinite_batched={affected}; retrying batch rows individually", flush=True)
    recovered = []
    for index, label in enumerate(labels):
        with torch.inference_mode():
            one = model.model(input_ids=tokens[index : index + 1].cuda()).last_hidden_state.float()
        if not torch.isfinite(one).all():
            raise RuntimeError(f"Persistent non-finite embedding for {label}")
        recovered.append(one)
    return torch.cat(recovered)


def embed(alleles, tokenizer, model, config, batch_size):
    context = int(config["context_bp"])
    left_flank = int(config["left_flank_bp"])
    hidden_size = int(config["hidden_size"])
    sequences = [allele.sequence(context, left_flank) for allele in alleles]
    whole = np.empty((len(alleles), hidden_size), dtype=np.float32)
    region = np.empty_like(whole)
    strand_region_l2 = np.empty(len(alleles), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    next_report = 1000

    for start in range(0, len(alleles), batch_size):
        stop = min(start + batch_size, len(alleles))
        forward_sequences = sequences[start:stop]
        reverse_sequences = [reverse_complement(sequence) for sequence in forward_sequences]
        encoded = tokenizer(
            forward_sequences + reverse_sequences,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        expected = (2 * len(forward_sequences), context)
        if tuple(encoded.shape) != expected:
            raise RuntimeError(f"Tokenizer shape {tuple(encoded.shape)} != {expected}")
        labels = [allele.allele_id for allele in alleles[start:stop]]
        labels += [f"{label}:reverse" for label in labels]
        hidden = run_model(encoded, model, labels)
        count = len(forward_sequences)
        forward = hidden[:count]
        reverse_aligned = hidden[count:].flip(dims=[1])
        combined = (forward + reverse_aligned) / 2
        for local_index, allele in enumerate(alleles[start:stop]):
            output_index = start + local_index
            region_slice = str_slice(left_flank, allele.repeat_bp)
            forward_region = forward[local_index, region_slice].mean(dim=0)
            reverse_region = reverse_aligned[local_index, region_slice].mean(dim=0)
            whole[output_index] = combined[local_index].mean(dim=0).cpu().numpy()
            region[output_index] = combined[local_index, region_slice].mean(dim=0).cpu().numpy()
            strand_region_l2[output_index] = torch.linalg.vector_norm(
                forward_region - reverse_region
            ).item()
        if stop >= next_report or stop == len(alleles):
            print(f"embedded={stop}/{len(alleles)} batch_size={batch_size}", flush=True)
            while next_report <= stop:
                next_report += 1000
        del encoded, hidden, forward, reverse_aligned, combined

    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    if not np.isfinite(whole).all() or not np.isfinite(region).all():
        raise RuntimeError("Non-finite pooled embeddings")
    return whole, region, strand_region_l2, {
        "batch_size": batch_size,
        "seconds": seconds,
        "alleles_per_second": len(alleles) / seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def write_h5(path, alleles, whole, region, strand_region_l2, config, benchmark=None):
    partial = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    string = h5py.string_dtype("utf-8")
    repeat_bp = np.asarray([allele.repeat_bp for allele in alleles], dtype=np.int32)
    with h5py.File(partial, "w") as handle:
        handle.create_dataset(
            "allele_id", data=np.asarray([x.allele_id for x in alleles], object), dtype=string
        )
        handle.create_dataset(
            "str_id", data=np.asarray([x.locus for x in alleles], object), dtype=string
        )
        handle.create_dataset(
            "allele_repeats", data=np.asarray([x.repeats for x in alleles], dtype=np.int32)
        )
        handle.create_dataset(
            "observed", data=np.asarray([x.observed for x in alleles], dtype=np.bool_)
        )
        handle.create_dataset(
            "is_reference", data=np.asarray([x.is_reference for x in alleles], dtype=np.bool_)
        )
        handle.create_dataset("repeat_bp", data=repeat_bp)
        handle.create_dataset(
            "right_flank_bp", data=int(config["context_bp"]) - int(config["left_flank_bp"]) - repeat_bp
        )
        handle.create_dataset("whole_mean", data=whole, compression="gzip", shuffle=True)
        handle.create_dataset("str_mean", data=region, compression="gzip", shuffle=True)
        handle.create_dataset("strand_str_mean_l2", data=strand_region_l2)
        handle.attrs["model_id"] = config["model_id"]
        handle.attrs["model_revision"] = config["model_revision"]
        handle.attrs["embedding_layer"] = "last_hidden_state (output of convolutional block 25)"
        handle.attrs["context_bp"] = config["context_bp"]
        handle.attrs["left_flank_bp"] = config["left_flank_bp"]
        handle.attrs["sequence_design"] = (
            "fixed 256-bp left flank; allele-specific STR; right flank trimmed to 512 bp"
        )
        handle.attrs["pooling"] = "whole-sequence mean and actual STR-region mean"
        handle.attrs["strand_handling"] = (
            "mean of forward and reverse-complement embeddings after coordinate alignment"
        )
        handle.attrs["zero_repeat_str_pooling"] = "two positions adjacent to deletion junction"
        handle.attrs["gpu_name"] = torch.cuda.get_device_name(0)
        if benchmark:
            for key, value in benchmark.items():
                handle.attrs[key] = value
    os.replace(partial, path)


def smoke_benchmark(config, tokenizer, model):
    alleles = load_smoke_alleles(config)
    output_dir = ROOT / "results/smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = None
    results, candidates = [], []
    selected_output = None
    for batch_size in config["benchmark_batch_sizes"]:
        try:
            whole, region, strand_l2, timing = embed(
                alleles, tokenizer, model, config, int(batch_size)
            )
            if reference is None:
                reference = np.concatenate([whole, region], axis=1)
                minimum_cosine, relative_l2 = 1.0, 0.0
            else:
                current = np.concatenate([whole, region], axis=1)
                denominator = np.maximum(
                    np.linalg.norm(current, axis=1) * np.linalg.norm(reference, axis=1), 1e-12
                )
                minimum_cosine = float(
                    np.min(np.sum(current * reference, axis=1) / denominator)
                )
                relative_l2 = float(
                    np.linalg.norm(current - reference)
                    / np.maximum(np.linalg.norm(reference), 1e-12)
                )
            record = timing | {
                "status": "PASS",
                "minimum_cosine_to_batch8": minimum_cosine,
                "relative_l2_to_batch8": relative_l2,
            }
            results.append(record)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if (
                timing["peak_gpu_memory_gib"] < 0.8 * total_memory
                and minimum_cosine >= 0.99999
                and relative_l2 <= 0.001
            ):
                candidates.append((timing["alleles_per_second"], int(batch_size)))
                selected_output = (whole, region, strand_l2, timing)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"batch_size": int(batch_size), "status": "OOM"})
    if not candidates:
        raise RuntimeError("No smoke benchmark batch size passed")
    _, selected = max(candidates)
    whole, region, strand_l2, timing = embed(alleles, tokenizer, model, config, selected)
    payload = {
        "status": "PASS",
        "recommended_batch_size": selected,
        "gpu_name": torch.cuda.get_device_name(0),
        "results": results,
    }
    temporary = output_dir / "benchmark.json.partial"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output_dir / "benchmark.json")
    write_h5(
        output_dir / "smoke_embeddings.h5",
        alleles,
        whole,
        region,
        strand_l2,
        config,
        timing,
    )
    print(json.dumps(payload, indent=2), flush=True)


def main():
    args = arguments()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer, model = load_model(config)
    if args.smoke_benchmark:
        smoke_benchmark(config, tokenizer, model)
        return
    if args.shard is None:
        raise ValueError("--shard is required outside smoke mode")
    output = ROOT / f"results/embeddings/shard_{args.shard:04d}.h5"
    if output.exists() and not args.force:
        print(f"Complete output already exists: {output}")
        return
    benchmark_path = ROOT / "results/smoke/benchmark.json"
    batch_size = args.batch_size
    if batch_size is None and benchmark_path.exists():
        batch_size = json.loads(benchmark_path.read_text())["recommended_batch_size"]
    batch_size = int(batch_size or config["batch_size_fallback"])
    alleles = load_production_alleles(config, args.shard)
    whole, region, strand_l2, timing = embed(
        alleles, tokenizer, model, config, batch_size
    )
    write_h5(output, alleles, whole, region, strand_l2, config, timing)
    print(f"Wrote {output}: alleles={len(alleles)} timing={timing}", flush=True)


if __name__ == "__main__":
    main()
