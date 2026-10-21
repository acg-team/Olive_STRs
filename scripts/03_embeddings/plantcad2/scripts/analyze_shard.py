#!/usr/bin/env python3
"""Run the shared reference-anchored shard analysis against PlantCAD2 embeddings."""
import runpy
from pathlib import Path
import common  # noqa: F401 -- ensures the PlantCAD2 ROOT is used by the shared script

SHARED = Path(__file__).resolve().parents[2] / "agront/scripts/analyze_shard.py"
runpy.run_path(str(SHARED), run_name="__main__")
