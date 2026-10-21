#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STR_BED="${1:-${SCRIPT_DIR}/../../olive_str_genotyping/ref/O_europeana_strs.bed}"
GFF3="${2:-${SCRIPT_DIR}/Olea_europaea.OLEA9.60.gff3.gz}"
OUTPUT="${3:-${SCRIPT_DIR}/O_europeana_strs.annotated.bed}"

if ! command -v bedtools >/dev/null 2>&1; then
    echo "ERROR: bedtools is not available on PATH" >&2
    exit 1
fi

for input in "$STR_BED" "$GFF3"; do
    if [[ ! -s "$input" ]]; then
        echo "ERROR: input file is missing or empty: $input" >&2
        exit 1
    fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# GFF3 is 1-based and inclusive; BED is 0-based and half-open.
gzip -cd "$GFF3" |
    awk -F '\t' 'BEGIN { OFS="\t" }
        $0 !~ /^#/ && $3 == "gene" {
            id=name=biotype="."
            n=split($9, attributes, ";")
            for (i=1; i<=n; i++) {
                split(attributes[i], pair, "=")
                if (pair[1] == "gene_id") id=pair[2]
                else if (pair[1] == "Name") name=pair[2]
                else if (pair[1] == "biotype") biotype=pair[2]
            }
            print $1, $4-1, $5, id, name, biotype, $7
        }' |
    LC_ALL=C sort -k1,1 -k2,2n -k3,3n > "$tmpdir/genes.bed"

LC_ALL=C sort -k1,1 -k2,2n -k3,3n "$STR_BED" > "$tmpdir/strs.bed"

{
    printf '#chrom\tstart\tend\tmotif_length\tmotif\tgene_start\tgene_end\tgene_id\tgene_name\tgene_biotype\tgene_strand\tlocation\tdistance_to_gene\n'
    bedtools closest \
        -a "$tmpdir/strs.bed" \
        -b "$tmpdir/genes.bed" \
        -d -t first -nonamecheck |
        awk -F '\t' 'BEGIN { OFS="\t" }
            {
                location=($13 == 0 ? "overlapping_gene" : "intergenic")
                print $1, $2, $3, $4, $5, $7, $8, $9, $10, $11, $12, location, $13
            }'
} > "$OUTPUT"

expected=$(wc -l < "$STR_BED")
observed=$(awk 'END { print NR-1 }' "$OUTPUT")
if [[ "$observed" -ne "$expected" ]]; then
    echo "ERROR: expected $expected annotations but produced $observed" >&2
    exit 1
fi

echo "Annotated $observed STR loci: $OUTPUT"
