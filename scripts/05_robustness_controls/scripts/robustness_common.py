#!/usr/bin/env python3
"""Shared sequence construction and HDF5 helpers for robustness analyses."""

from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "shared_inputs"
N_SHARDS = 10
LAYERS = {"argont": (11, 21, 31, 40), "plantcad2": (12, 24, 36, 48), "gpn": (25,)}


def normalize_dna(sequence: str) -> str:
    return "".join(base if base in "ACGT" else "N" for base in sequence.upper())


@dataclass
class Allele:
    allele_id: str
    str_id: str
    repeats: int
    observed: bool
    is_reference: bool
    motif: str
    period: int
    left: str
    right: str
    source_shard: int

    @property
    def repeat_bp(self) -> int:
        return self.period * self.repeats


def load_shard(shard: int) -> list[Allele]:
    loci = {}
    with gzip.open(INPUT / "loci.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["robustness_shard"]) == shard:
                loci[row["str_id"]] = row
    result = []
    with gzip.open(INPUT / "alleles.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["robustness_shard"]) != shard:
                continue
            locus = loci[row["str_id"]]
            result.append(Allele(
                allele_id=row["allele_id"],
                str_id=row["str_id"],
                repeats=int(row["allele_repeats"]),
                observed=bool(int(row["observed"])),
                is_reference=bool(int(row["is_reference"])),
                motif=normalize_dna(locus["motif"]),
                period=int(locus["period"]),
                left=normalize_dna(locus["left_flank"]),
                right=normalize_dna(locus["right_flank"]),
                source_shard=int(row["source_shard"]),
            ))
    if not result:
        raise ValueError(f"Robustness shard {shard} is empty")
    return result


def has_tandem_repeat(sequence: str) -> bool:
    for period in range(1, 7):
        width = period * 4
        for start in range(0, len(sequence) - width + 1):
            unit = sequence[start:start + period]
            if sequence[start:start + width] == unit * 4:
                return True
    return False


def control_tape(allele: Allele, replicate: int, length: int) -> str:
    source = np.asarray([base for base in allele.left + allele.right if base in "ACGT"], dtype="U1")
    if len(source) < length:
        raise RuntimeError(f"{allele.str_id}: only {len(source)} unambiguous flank bases for {length}-bp tape")
    token = f"{allele.str_id}|{replicate}|shuffled-local-flanks-v1".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    for _ in range(10000):
        candidate = "".join(rng.permutation(source)[:length].tolist())
        if not has_tandem_repeat(candidate):
            return candidate
    raise RuntimeError(f"{allele.str_id}: failed to create tandem-repeat-free control tape")


def build_sequences(alleles: list[Allele], mode: str, replicate: int | None = None):
    tapes = {}
    if mode == "control":
        if replicate not in (1, 2, 3):
            raise ValueError("Control mode requires replicate 1, 2, or 3")
        by_locus = {}
        for allele in alleles:
            by_locus.setdefault(allele.str_id, []).append(allele)
        for locus, group in by_locus.items():
            representative = group[0]
            tapes[locus] = control_tape(
                representative, int(replicate), max(x.repeat_bp for x in group)
            )

    sequences, starts, ends = [], [], []
    for allele in alleles:
        insertion = (
            allele.motif * allele.repeats
            if mode == "fixed_total"
            else tapes[allele.str_id][:allele.repeat_bp]
        )
        if mode == "fixed_total":
            right_bp = 1000 - 500 - allele.repeat_bp
            if right_bp < 0:
                raise ValueError(f"{allele.allele_id}: STR exceeds fixed total context")
            sequence = allele.left[-500:] + insertion + allele.right[:right_bp]
            if len(sequence) != 1000:
                raise RuntimeError(f"{allele.allele_id}: fixed-total sequence length={len(sequence)}")
        elif mode == "control":
            sequence = allele.left[-500:] + insertion + allele.right[:500]
            if len(sequence) != 1000 + allele.repeat_bp:
                raise RuntimeError(f"{allele.allele_id}: current-context control length mismatch")
        else:
            raise ValueError(mode)
        sequences.append(normalize_dna(sequence))
        starts.append(500)
        ends.append(500 + allele.repeat_bp)
    return sequences, np.asarray(starts), np.asarray(ends), tapes


def build_gpn_control_sequences(alleles: list[Allele], replicate: int):
    by_locus = {}
    for allele in alleles:
        by_locus.setdefault(allele.str_id, []).append(allele)
    tapes = {
        locus: control_tape(group[0], replicate, max(x.repeat_bp for x in group))
        for locus, group in by_locus.items()
    }
    sequences, starts, ends = [], [], []
    for allele in alleles:
        right_bp = 512 - 256 - allele.repeat_bp
        if right_bp < 0:
            raise ValueError(f"{allele.allele_id}: tract exceeds GPN context")
        insertion = tapes[allele.str_id][:allele.repeat_bp]
        sequence = allele.left[-256:] + insertion + allele.right[:right_bp]
        if len(sequence) != 512:
            raise RuntimeError(f"{allele.allele_id}: GPN control length={len(sequence)}")
        sequences.append(normalize_dna(sequence))
        starts.append(256)
        ends.append(256 + allele.repeat_bp)
    return sequences, np.asarray(starts), np.asarray(ends), tapes


def region_slice(start: int, end: int):
    return slice(start, end) if start < end else slice(start - 1, start + 1)


def output_path(model: str, mode: str, shard: int, replicate: int | None) -> Path:
    base = ROOT
    if mode == "fixed_total":
        return base / "fixed_total_embedding_comparison/results/embeddings" / (
            f"{model}_fixed_total_shard_{shard:04d}.h5"
        )
    return base / "matched_indel_control/results/embeddings" / (
        f"{model}_control_r{replicate}_shard_{shard:04d}.h5"
    )


def write_embeddings(
    path: Path,
    alleles: list[Allele],
    whole: np.ndarray,
    region: np.ndarray,
    model: str,
    mode: str,
    replicate: int | None,
    tapes: dict[str, str],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("allele_id", data=np.asarray([x.allele_id for x in alleles], object), dtype=string)
        handle.create_dataset("str_id", data=np.asarray([x.str_id for x in alleles], object), dtype=string)
        handle.create_dataset("allele_repeats", data=np.asarray([x.repeats for x in alleles], np.int32))
        handle.create_dataset("repeat_bp", data=np.asarray([x.repeat_bp for x in alleles], np.int32))
        handle.create_dataset("observed", data=np.asarray([x.observed for x in alleles], bool))
        handle.create_dataset("is_reference", data=np.asarray([x.is_reference for x in alleles], bool))
        handle.create_dataset("whole_mean", data=whole, compression="gzip", shuffle=True)
        handle.create_dataset("str_mean", data=region, compression="gzip", shuffle=True)
        handle.attrs["model"] = model
        handle.attrs["layers"] = LAYERS[model]
        handle.attrs["mode"] = mode
        handle.attrs["replicate"] = -1 if replicate is None else replicate
        handle.attrs["sequence_design"] = (
            "fixed 500-bp left and fixed 1000-bp total"
            if mode == "fixed_total"
            else ("fixed 256-bp left and fixed 512-bp total" if model == "gpn"
                  else "fixed 500-bp left and fixed 500-bp right")
        )
        for key, value in metadata.items():
            handle.attrs[key] = value
        if tapes:
            group = handle.create_group("control_tapes")
            for locus, tape in tapes.items():
                group.attrs[locus] = tape
    temporary.replace(path)
