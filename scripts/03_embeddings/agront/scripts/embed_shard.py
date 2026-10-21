#!/usr/bin/env python3
"""Generate whole-sequence and actual-STR mean ArgoNT embeddings for one shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from common import load_config, normalize_dna


COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


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
        return self.left + self.motif * self.repeats + self.right

    @property
    def repeat_start(self):
        return len(self.left)

    @property
    def repeat_end(self):
        return len(self.left) + len(self.motif) * self.repeats


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-alleles", type=int, help="Smoke-test only; does not create a production shard")
    return parser.parse_args()


def load_alleles(root: Path, shard: int, limit: int | None) -> list[Allele]:
    loci = {}
    with gzip.open(root / "inputs/loci.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            loci[row["str_id"]] = row
    alleles = []
    with gzip.open(root / "inputs/alleles.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["shard"]) != shard:
                continue
            locus = loci[row["str_id"]]
            alleles.append(Allele(
                row["allele_id"], row["str_id"], int(row["allele_repeats"]), bool(int(row["observed"])),
                bool(int(row["is_reference"])), locus["motif"], locus["left_flank"], locus["right_flank"],
            ))
            if limit is not None and len(alleles) >= limit:
                break
    if not alleles:
        raise ValueError(f"Shard {shard} contains no alleles")
    return alleles


def views(allele: Allele, frames: int):
    sequence = allele.sequence
    orientations = [("forward", sequence, allele.repeat_start, allele.repeat_end)]
    reverse = sequence.translate(COMPLEMENT)[::-1]
    orientations.append(("reverse", reverse, len(sequence) - allele.repeat_end, len(sequence) - allele.repeat_start))
    for strand, oriented, repeat_start, repeat_end in orientations:
        for offset in range(frames):
            yield f"{strand}:{offset}", oriented[offset:], repeat_start - offset, repeat_end - offset


def pooled_token_indexes(spans, repeat_start: int, repeat_end: int):
    if repeat_start < repeat_end:
        return [i for i, (start, end) in enumerate(spans) if start < repeat_end and end > repeat_start]
    # A zero-repeat allele has no STR bases. Represent its deletion junction by
    # the token spanning the junction, or both neighboring tokens if the
    # junction falls exactly on a token boundary.
    spanning = [i for i, (start, end) in enumerate(spans) if start < repeat_start < end]
    if spanning:
        return spanning
    return [i for i, (start, end) in enumerate(spans) if end == repeat_start or start == repeat_start]


def encode(tokenizer, label: str, sequence: str, repeat_start: int, repeat_end: int):
    ids = torch.tensor(tokenizer(sequence, add_special_tokens=True)["input_ids"], dtype=torch.long)
    if len(ids) > 1026:
        raise ValueError(f"{label}: {len(ids)} tokens exceed ArgoNT's 1026-position context")
    if ids[0].item() != tokenizer.cls_token_id:
        raise RuntimeError(f"{label}: first token is not CLS")
    tokens = tokenizer.convert_ids_to_tokens(ids[1:].tolist())
    if "".join(tokens) != sequence:
        raise RuntimeError(f"{label}: token strings do not reconstruct the input")
    spans, cursor = [], 0
    for token in tokens:
        spans.append((cursor, cursor + len(token)))
        cursor += len(token)
    targets = pooled_token_indexes(spans, repeat_start, repeat_end)
    if not targets:
        raise RuntimeError(f"{label}: no token overlaps the STR")
    return ids, targets


def run_model_batch(items, tokenizer, model, device, layers, whole_sum, str_sum, view_count):
    max_length = max(len(item[2]) for item in items)
    ids = torch.full((len(items), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for row, (_, _, sequence_ids, _) in enumerate(items):
        ids[row, : len(sequence_ids)] = sequence_ids
        attention[row, : len(sequence_ids)] = 1
    ids, attention = ids.to(device), attention.to(device)
    with torch.inference_mode():
        outputs = model(input_ids=ids, attention_mask=attention, output_hidden_states=True)
    for row, (allele_index, _, sequence_ids, targets) in enumerate(items):
        positions = torch.tensor(targets, device=device)
        for layer_index, layer in enumerate(layers):
            hidden = outputs.hidden_states[layer][row, 1:len(sequence_ids)].float()
            whole_sum[allele_index, layer_index] += hidden.mean(0).cpu().numpy()
            str_sum[allele_index, layer_index] += hidden.index_select(0, positions).mean(0).cpu().numpy()
        view_count[allele_index] += 1


def main():
    args = arguments()
    config = load_config(args.config)
    root = Path(config["root"])
    output_dir = root / "results/embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".smoke" if args.limit_alleles else ""
    output = output_dir / f"shard_{args.shard:04d}{suffix}.h5"
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() and not args.force:
        print(f"Complete output already exists: {output}")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    alleles = load_alleles(root, args.shard, args.limit_alleles)
    layers = tuple(config["layers"])
    batch_size = args.batch_size or config["batch_size"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]
    device = torch.device(args.device)
    model_source = config.get("local_model_path", config["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=config["model_revision"], cache_dir=config["hf_cache"],
        trust_remote_code=True, local_files_only=args.local_files_only,
    )
    model_config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    weights_source = config.get("local_safetensors_path", model_source)
    model = AutoModelForMaskedLM.from_pretrained(
        weights_source, config=model_config, revision=config["model_revision"], cache_dir=config["hf_cache"],
        trust_remote_code=True, local_files_only=args.local_files_only, use_safetensors=True, dtype=dtype,
    ).to(device).eval()
    hidden_size = int(model.config.hidden_size)
    whole_sum = np.zeros((len(alleles), len(layers), hidden_size), dtype=np.float32)
    str_sum = np.zeros_like(whole_sum)
    view_count = np.zeros(len(alleles), dtype=np.int16)
    batch = []
    for allele_index, allele in enumerate(alleles):
        expected = len(allele.left) + len(allele.motif) * allele.repeats + len(allele.right)
        if len(allele.sequence) != expected:
            raise RuntimeError(f"{allele.allele_id}: reconstruction length mismatch")
        for label, sequence, start, end in views(allele, config["frames"]):
            sequence = normalize_dna(sequence)
            ids, targets = encode(tokenizer, f"{allele.allele_id}/{label}", sequence, start, end)
            batch.append((allele_index, label, ids, targets))
            if len(batch) == batch_size:
                run_model_batch(batch, tokenizer, model, device, layers, whole_sum, str_sum, view_count)
                batch = []
    if batch:
        run_model_batch(batch, tokenizer, model, device, layers, whole_sum, str_sum, view_count)
    expected_views = config["frames"] * 2
    if np.any(view_count != expected_views):
        raise RuntimeError(f"Expected {expected_views} views per allele; found {np.unique(view_count)}")
    whole = whole_sum / view_count[:, None, None]
    str_mean = str_sum / view_count[:, None, None]
    if not np.isfinite(whole).all() or not np.isfinite(str_mean).all():
        raise RuntimeError("Non-finite embeddings generated")
    string = h5py.string_dtype("utf-8")
    if partial.exists():
        partial.unlink()
    with h5py.File(partial, "w") as h5:
        h5.create_dataset("allele_id", data=np.asarray([x.allele_id for x in alleles], dtype=object), dtype=string)
        h5.create_dataset("str_id", data=np.asarray([x.locus for x in alleles], dtype=object), dtype=string)
        h5.create_dataset("allele_repeats", data=np.asarray([x.repeats for x in alleles], dtype=np.int32))
        h5.create_dataset("observed", data=np.asarray([x.observed for x in alleles], dtype=np.bool_))
        h5.create_dataset("is_reference", data=np.asarray([x.is_reference for x in alleles], dtype=np.bool_))
        h5.create_dataset("whole_mean", data=whole, compression="gzip", shuffle=True)
        h5.create_dataset("str_mean", data=str_mean, compression="gzip", shuffle=True)
        h5.attrs["layers"] = layers
        h5.attrs["model_id"] = config["model_id"]
        h5.attrs["model_revision"] = config["model_revision"]
        h5.attrs["weights_revision"] = config.get("weights_revision", config["model_revision"])
        h5.attrs["frames"] = config["frames"]
        h5.attrs["strands"] = "both"
        h5.attrs["sequence_design"] = "500bp_left+variable_STR+500bp_right"
        h5.attrs["zero_repeat_str_pooling"] = "deletion-junction token, or two adjacent tokens at an exact token boundary"
        h5.attrs["likelihood_score"] = "not_generated"
    os.replace(partial, output)
    print(f"Wrote {output}: {len(alleles)} alleles, shape {whole.shape}")


if __name__ == "__main__":
    main()
