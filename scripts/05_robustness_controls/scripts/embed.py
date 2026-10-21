#!/usr/bin/env python3
"""Embed fixed-total STR alleles or current-context matched-indel controls."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from robustness_common import (
    LAYERS,
    ROOT,
    build_gpn_control_sequences,
    build_sequences,
    load_shard,
    output_path,
    region_slice,
    write_embeddings,
)


COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("argont", "plantcad2", "gpn"), required=True)
    parser.add_argument("--mode", choices=("fixed_total", "control"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path):
    import json
    return json.loads(Path(path).read_text())


def embed_argont(alleles, sequences, starts, ends, batch_size):
    source_scripts = ROOT.parent / "03_embeddings/agront/scripts"
    sys.path.insert(0, str(source_scripts))
    source = importlib.import_module("embed_shard")
    config = load_json(ROOT.parent / "03_embeddings/agront/config.json")
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["local_model_path"], revision=config["model_revision"],
        cache_dir=config["hf_cache"], trust_remote_code=True, local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(config["local_model_path"], local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        config["local_safetensors_path"], config=model_config, revision=config["model_revision"],
        cache_dir=config["hf_cache"], trust_remote_code=True, local_files_only=True,
        use_safetensors=True, dtype=torch.float16,
    ).cuda().eval()
    layers = LAYERS["argont"]
    hidden_size = int(model.config.hidden_size)
    whole_sum = np.zeros((len(alleles), len(layers), hidden_size), np.float32)
    region_sum = np.zeros_like(whole_sum)
    counts = np.zeros(len(alleles), np.int16)
    batch = []
    started = time.monotonic()
    for index, sequence in enumerate(sequences):
        orientations = [
            (sequence, int(starts[index]), int(ends[index])),
            (sequence.translate(COMPLEMENT)[::-1], len(sequence) - int(ends[index]), len(sequence) - int(starts[index])),
        ]
        for oriented, repeat_start, repeat_end in orientations:
            for offset in range(6):
                ids, targets = source.encode(
                    tokenizer, f"{alleles[index].allele_id}/{offset}",
                    oriented[offset:], repeat_start - offset, repeat_end - offset,
                )
                batch.append((index, str(offset), ids, targets))
                if len(batch) == batch_size:
                    source.run_model_batch(batch, tokenizer, model, torch.device("cuda"), layers,
                                           whole_sum, region_sum, counts)
                    batch = []
    if batch:
        source.run_model_batch(batch, tokenizer, model, torch.device("cuda"), layers,
                               whole_sum, region_sum, counts)
    if np.any(counts != 12):
        raise RuntimeError(f"ArgoNT view counts: {np.unique(counts)}")
    return whole_sum / counts[:, None, None], region_sum / counts[:, None, None], {
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "weights_revision": config["weights_revision"], "frames": 6, "strands": "both",
        "seconds": time.monotonic() - started,
    }


def plantcad2_combine(raw, hidden_size):
    if raw.shape[-1] != hidden_size * 2:
        raise RuntimeError(f"Unexpected PlantCAD2 RCPS width {raw.shape[-1]}")
    return (raw[..., :hidden_size] + raw[..., hidden_size:].flip(dims=[1])) / 2


def fix_ties(model):
    tied = list(getattr(model, "_tied_weights_keys", None) or [])
    modules = dict(model.named_modules())
    for name, forward in modules.items():
        if "fwd" not in name or "proj" not in name:
            continue
        reverse_name = name.replace("fwd", "rev")
        if reverse_name not in modules:
            continue
        reverse = modules[reverse_name]
        if hasattr(forward, "weight"):
            reverse.weight = forward.weight
            tied.append(f"{reverse_name}.weight")
        if hasattr(forward, "bias") and forward.bias is not None:
            reverse.bias = forward.bias
            tied.append(f"{reverse_name}.bias")
    model._tied_weights_keys = sorted(set(tied))
    return model


def embed_plantcad2(alleles, sequences, starts, ends, batch_size):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    config = load_json(ROOT.parent / "03_embeddings/plantcad2/config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        config["local_model_path"], trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        config["local_model_path"], trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model = fix_ties(model).eval().cuda()
    if model.num_parameters() != 694385664:
        raise RuntimeError(f"Unexpected PlantCAD2 parameter count {model.num_parameters()}")
    layers = LAYERS["plantcad2"]
    hidden_size = 1536
    whole = np.empty((len(alleles), len(layers), hidden_size), np.float32)
    region = np.empty_like(whole)
    groups = defaultdict(list)
    for index, sequence in enumerate(sequences):
        groups[len(sequence)].append(index)
    started = time.monotonic()
    for length in sorted(groups):
        indexes = groups[length]
        for offset in range(0, len(indexes), batch_size):
            selected = indexes[offset:offset + batch_size]
            encoded = tokenizer(
                [sequences[i] for i in selected], add_special_tokens=True, return_tensors="pt",
                return_attention_mask=False, return_token_type_ids=False, padding=False,
            )["input_ids"]
            if tuple(encoded.shape) != (len(selected), length):
                raise RuntimeError(f"PlantCAD2 tokenizer shape {tuple(encoded.shape)}")
            with torch.inference_mode():
                output = model(input_ids=encoded.cuda(), output_hidden_states=True)
            for layer_column, layer in enumerate(layers):
                hidden = plantcad2_combine(output.hidden_states[layer].float(), hidden_size)
                for row, allele_index in enumerate(selected):
                    whole[allele_index, layer_column] = hidden[row].mean(0).cpu().numpy()
                    slc = region_slice(int(starts[allele_index]), int(ends[allele_index]))
                    region[allele_index, layer_column] = hidden[row, slc].mean(0).cpu().numpy()
    return whole, region, {
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "seconds": time.monotonic() - started,
    }


def embed_gpn(alleles, sequences, starts, ends, batch_size):
    import gpn.model  # noqa: F401
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    config = load_json(ROOT.parent / "03_embeddings/gpn/config.json")
    tokenizer = AutoTokenizer.from_pretrained(config["local_model_path"], local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(config["local_model_path"], local_files_only=True).eval().cuda()
    whole = np.empty((len(alleles), 512), np.float32)
    region = np.empty_like(whole)
    started = time.monotonic()
    for offset in range(0, len(alleles), batch_size):
        stop = min(offset + batch_size, len(alleles))
        forward_sequences = sequences[offset:stop]
        reverse_sequences = [x.translate(COMPLEMENT)[::-1] for x in forward_sequences]
        encoded = tokenizer(
            forward_sequences + reverse_sequences, add_special_tokens=False, padding=False,
            truncation=False, return_tensors="pt", return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        expected = (2 * len(forward_sequences), 512)
        if tuple(encoded.shape) != expected:
            raise RuntimeError(f"GPN tokenizer shape {tuple(encoded.shape)} != {expected}")
        with torch.inference_mode():
            hidden = model.model(input_ids=encoded.cuda()).last_hidden_state.float()
        if not torch.isfinite(hidden).all():
            recovered = []
            labels = [x.allele_id for x in alleles[offset:stop]]
            labels += [f"{x}:reverse" for x in labels]
            for row, label in enumerate(labels):
                with torch.inference_mode():
                    one = model.model(input_ids=encoded[row:row + 1].cuda()).last_hidden_state.float()
                if not torch.isfinite(one).all():
                    raise RuntimeError(f"Persistent non-finite GPN embedding for {label}")
                recovered.append(one)
            hidden = torch.cat(recovered)
        count = stop - offset
        combined = (hidden[:count] + hidden[count:].flip(dims=[1])) / 2
        for row, allele_index in enumerate(range(offset, stop)):
            whole[allele_index] = combined[row].mean(0).cpu().numpy()
            slc = region_slice(int(starts[allele_index]), int(ends[allele_index]))
            region[allele_index] = combined[row, slc].mean(0).cpu().numpy()
    return whole, region, {
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "seconds": time.monotonic() - started,
    }


def main():
    args = arguments()
    if args.model == "gpn" and args.mode != "control":
        raise ValueError("GPN has no fixed-total alternative test")
    if args.mode == "control" and args.replicate not in (1, 2, 3):
        raise ValueError("Control mode requires --replicate 1, 2, or 3")
    if args.mode == "fixed_total" and args.replicate is not None:
        raise ValueError("Fixed-total mode does not use replicates")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    path = output_path(args.model, args.mode, args.shard, args.replicate)
    if path.exists() and not args.force:
        print(f"Complete output exists: {path}")
        return
    alleles = load_shard(args.shard)
    if args.model == "gpn":
        sequences, starts, ends, tapes = build_gpn_control_sequences(alleles, args.replicate)
    else:
        sequences, starts, ends, tapes = build_sequences(alleles, args.mode, args.replicate)
    defaults = {"argont": 8, "plantcad2": 8, "gpn": 32}
    batch_size = int(args.batch_size or defaults[args.model])
    if args.model == "argont":
        whole, region, metadata = embed_argont(alleles, sequences, starts, ends, batch_size)
    elif args.model == "plantcad2":
        whole, region, metadata = embed_plantcad2(alleles, sequences, starts, ends, batch_size)
    else:
        whole, region, metadata = embed_gpn(alleles, sequences, starts, ends, batch_size)
    if not np.isfinite(whole).all() or not np.isfinite(region).all():
        raise RuntimeError("Non-finite embeddings")
    metadata["batch_size"] = batch_size
    metadata["alleles"] = len(alleles)
    write_embeddings(path, alleles, whole, region, args.model, args.mode, args.replicate, tapes, metadata)
    print(f"output={path} alleles={len(alleles)} whole_shape={whole.shape} ROBUST_EMBEDDING=PASS")


if __name__ == "__main__":
    main()
