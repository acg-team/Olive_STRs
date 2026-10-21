#!/usr/bin/env python3
"""Write checksums for completed PlantCAD2 outputs."""
import hashlib
from pathlib import Path
from common import ROOT


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def main():
    output = ROOT / "results/SHA256SUMS"; paths = []
    for directory in (ROOT / "results", ROOT / "figures", ROOT / "report"):
        paths.extend(path for path in directory.rglob("*") if path.is_file() and path != output and not path.name.endswith(".partial"))
    with output.open("w") as handle:
        for path in sorted(paths): handle.write(f"{digest(path)}  {path.relative_to(ROOT)}\n")
    print(f"Wrote {output} for {len(paths)} files")


if __name__ == "__main__": main()
