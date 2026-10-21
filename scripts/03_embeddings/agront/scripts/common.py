#!/usr/bin/env python3
"""Shared I/O and sequence helpers for the genome-wide STR workflow."""

from __future__ import annotations

import csv
import gzip
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    config["root"] = str(ROOT)
    return config


def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    return gzip.open(path, mode, newline="") if path.suffix == ".gz" else path.open(mode, newline="")


def read_dicts(path: str | Path):
    with open_text(path) as handle:
        yield from csv.DictReader(handle)


def normalize_dna(sequence: str) -> str:
    return "".join(base if base in "ACGT" else "N" for base in sequence.upper())


def estimated_token_count(sequence: str) -> int:
    """Conservative exact count for AgroNT's 6-mer/N segmentation plus CLS."""
    count = 1
    run = 0
    for base in sequence:
        if base in "ACGT":
            run += 1
        else:
            count += math.ceil(run / 6) + 1
            run = 0
    return count + math.ceil(run / 6)


class IndexedFasta:
    def __init__(self, fasta: str | Path, index: str | Path, samtools: str | Path):
        self.fasta = Path(fasta)
        self.index = Path(index)
        if not self.index.exists():
            self.index.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [str(samtools), "faidx", "--fai-idx", str(self.index), str(self.fasta)],
                check=True,
            )
        self.records = {}
        with self.index.open() as handle:
            for line in handle:
                name, length, offset, line_bases, line_width, *_ = line.rstrip().split("\t")
                self.records[name] = tuple(map(int, (length, offset, line_bases, line_width)))
        self.handle = self.fasta.open("rb")

    def close(self):
        self.handle.close()

    def fetch(self, contig: str, start: int, end: int) -> str:
        if contig not in self.records:
            raise KeyError(f"Contig {contig!r} is absent from the reference FASTA")
        length, offset, line_bases, line_width = self.records[contig]
        if start < 0 or end < start or end > length:
            raise ValueError(f"Reference interval {contig}:{start}-{end} is out of bounds (length {length})")
        byte_start = offset + (start // line_bases) * line_width + start % line_bases
        byte_end = offset + (end // line_bases) * line_width + end % line_bases
        self.handle.seek(byte_start)
        raw = self.handle.read(byte_end - byte_start + 2)
        sequence = raw.replace(b"\n", b"").replace(b"\r", b"")[: end - start]
        if len(sequence) != end - start:
            raise RuntimeError(f"Short FASTA read at {contig}:{start}-{end}")
        return normalize_dna(sequence.decode("ascii"))
