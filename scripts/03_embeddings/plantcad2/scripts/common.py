#!/usr/bin/env python3
"""Shared helpers for the genome-wide PlantCAD2 workflow."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    config_path = Path(path) if path else ROOT / "config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    config["root"] = str(ROOT)
    return config


def normalize_dna(sequence):
    return "".join(base if base in "ACGT" else "N" for base in sequence.upper())
