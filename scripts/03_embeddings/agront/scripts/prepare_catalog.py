#!/usr/bin/env python3
"""Build the >=3-observed-length STR allele catalog and deterministic shards."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from common import IndexedFasta, estimated_token_count, load_config, normalize_dna


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_genotype(raw: str) -> tuple[int, ...]:
    value = ast.literal_eval(raw)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Invalid genotype {raw!r}")
    alleles = []
    for item in value:
        number = float(item)
        if not number.is_integer() or number < 0:
            raise ValueError(f"Non-integer or negative STR allele {item!r}")
        alleles.append(int(number))
    return tuple(alleles)


def sample_files(config: dict) -> list[tuple[str, Path]]:
    pass_dir = Path(config["pass_only_dir"])
    found = {p.name.removesuffix("_ConSTRain_CNVs.csv"): p for p in pass_dir.glob("*_ConSTRain_CNVs.csv")}
    ordered = []
    with Path(config["samples_tsv"]).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample = row["sample"]
            if sample not in found:
                raise FileNotFoundError(f"Missing PASS-only CSV for sample {sample}")
            ordered.append((sample, found[sample]))
    return ordered


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = arguments()
    config = load_config(args.config)
    root = Path(config["root"])
    inputs, results = root / "inputs", root / "results"
    inputs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    outputs = [inputs / "loci.csv.gz", inputs / "alleles.csv.gz", inputs / "shards.csv", results / "genotype_lookup.csv.gz"]
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("Catalog outputs already exist; use --force to replace them")

    samples = sample_files(config)
    required_calls = math.ceil(config["minimum_call_rate"] * len(samples))
    qc = set()
    with Path(config["variance_csv"]).open() as handle:
        for row in csv.DictReader(handle):
            if int(row["n_called"]) >= required_calls and float(row["variance"]) >= config["minimum_variance"]:
                qc.add(row["str_id"])

    bed = {}
    with Path(config["str_bed"]).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 5:
                raise ValueError(f"BED line {line_number} has fewer than five columns")
            contig, start, end, period, motif = fields[:5]
            locus = f"{contig}_{start}"
            if locus in qc:
                bed[locus] = (contig, int(start), int(end), int(period), normalize_dna(motif))
    missing_bed = qc - bed.keys()
    if missing_bed:
        raise ValueError(f"{len(missing_bed)} QC loci are absent from the STR BED")

    observed = defaultdict(set)
    malformed = []
    for sample, path in samples:
        with path.open() as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 2):
                locus = row["str_id"]
                if locus not in qc:
                    continue
                try:
                    observed[locus].update(parse_genotype(row["genotype"]))
                except Exception as error:
                    malformed.append((sample, row_number, locus, str(error)))
    candidate = {locus for locus, lengths in observed.items() if len(lengths) >= config["minimum_observed_allele_lengths"]}

    fasta = IndexedFasta(config["reference_fasta"], inputs / "reference.fa.fai", config["samtools"])
    valid_loci, excluded = {}, []
    try:
        for locus in sorted(candidate, key=lambda x: (bed[x][0], bed[x][1])):
            contig, start, end, period, motif = bed[locus]
            reason = None
            if period <= 0 or len(motif) != period:
                reason = "period_or_motif_length_invalid"
            elif (end - start) % period:
                reason = "reference_interval_not_divisible_by_period"
            elif start < config["left_flank_bp"]:
                reason = "left_flank_out_of_bounds"
            else:
                ref_repeats = (end - start) // period
                try:
                    left = fasta.fetch(contig, start - config["left_flank_bp"], start)
                    reference_repeat = fasta.fetch(contig, start, end)
                    right = fasta.fetch(contig, end, end + config["right_flank_bp"])
                except Exception as error:
                    reason = f"reference_fetch_failed:{error}"
                else:
                    if reference_repeat != motif * ref_repeats:
                        reason = "bed_motif_does_not_match_reference"
                    else:
                        all_lengths = sorted(observed[locus] | {ref_repeats})
                        max_tokens = max(estimated_token_count(left + motif * repeats + right) for repeats in all_lengths)
                        if max_tokens > 1026:
                            reason = f"model_context_exceeded:{max_tokens}"
                        else:
                            valid_loci[locus] = {
                                "str_id": locus, "contig": contig, "start": start, "end": end,
                                "period": period, "motif": motif, "ref_repeats": ref_repeats,
                                "left_flank": left, "right_flank": right,
                                "observed_lengths": sorted(observed[locus]), "all_lengths": all_lengths,
                                "max_tokens": max_tokens,
                            }
            if reason:
                excluded.append((locus, reason))
    finally:
        fasta.close()

    shard, shard_size = 0, 0
    allele_rows, shard_rows = [], []
    for locus, item in valid_loci.items():
        locus_size = len(item["all_lengths"])
        if shard_size and shard_size + locus_size > config["maximum_alleles_per_shard"]:
            shard += 1
            shard_size = 0
        shard_rows.append((locus, shard, locus_size))
        for repeats in item["all_lengths"]:
            allele_rows.append({
                "allele_id": f"{locus}|r{repeats}", "str_id": locus, "allele_repeats": repeats,
                "observed": int(repeats in observed[locus]), "is_reference": int(repeats == item["ref_repeats"]),
                "sequence_length": len(item["left_flank"]) + repeats * item["period"] + len(item["right_flank"]),
                "shard": shard, "row_in_shard": shard_size,
            })
            shard_size += 1

    with gzip.open(inputs / "loci.csv.gz", "wt", newline="") as handle:
        fields = ["str_id", "contig", "start", "end", "period", "motif", "ref_repeats", "left_flank", "right_flank", "n_observed_lengths", "n_embedded_lengths", "max_tokens"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in valid_loci.values():
            writer.writerow({key: item[key] for key in fields if key in item} | {
                "n_observed_lengths": len(item["observed_lengths"]), "n_embedded_lengths": len(item["all_lengths"])
            })
    with gzip.open(inputs / "alleles.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(allele_rows[0]) if allele_rows else ["allele_id", "str_id", "allele_repeats", "observed", "is_reference", "sequence_length", "shard", "row_in_shard"])
        writer.writeheader(); writer.writerows(allele_rows)
    with (inputs / "shards.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["str_id", "shard", "n_alleles"]); writer.writerows(shard_rows)
    with (results / "excluded_loci.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["str_id", "reason"]); writer.writerows(excluded)
    with (results / "malformed_genotypes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["sample", "row", "str_id", "reason"]); writer.writerows(malformed)

    valid = set(valid_loci)
    allele_id = {(locus, repeats): f"{locus}|r{repeats}" for locus in valid for repeats in observed[locus]}
    call_count = 0
    with gzip.open(results / "genotype_lookup.csv.gz", "wt", newline="") as output:
        writer = csv.writer(output); writer.writerow(["sample", "str_id", "copy_number", "allele_ids"])
        for sample, path in samples:
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    locus = row["str_id"]
                    if locus not in valid:
                        continue
                    genotype = parse_genotype(row["genotype"])
                    writer.writerow([sample, locus, len(genotype), ";".join(allele_id[(locus, x)] for x in genotype)])
                    call_count += 1

    manifest = {
        "samples": len(samples), "minimum_required_calls": required_calls,
        "qc_loci": len(qc), "candidate_loci_with_three_observed_lengths": len(candidate),
        "valid_loci": len(valid_loci), "excluded_after_sequence_validation": len(excluded),
        "embedded_alleles": len(allele_rows), "observed_alleles": sum(int(x["observed"]) for x in allele_rows),
        "zero_repeat_alleles": sum(int(x["allele_repeats"]) == 0 for x in allele_rows),
        "genotype_calls": call_count, "shards": shard + 1 if allele_rows else 0,
        "inputs": {key: {"path": config[key], "sha256": sha256(Path(config[key]))} for key in ("reference_fasta", "str_bed", "variance_csv", "samples_tsv")},
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "weights_revision": config.get("weights_revision", config["model_revision"]),
        "filter": "at least three distinct allele repeat counts observed in PASS-only sample genotypes",
        "likelihood_score": "not_generated",
    }
    with (results / "preparation_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2); handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
