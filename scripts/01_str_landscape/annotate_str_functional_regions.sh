#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${SCRIPT_DIR}/../../olive_str_genotyping/.conda-env"
ANNOTATED_BED="${1:-${SCRIPT_DIR}/O_europeana_strs.annotated.bed}"
GFF3="${2:-${SCRIPT_DIR}/Olea_europaea.OLEA9.60.gff3.gz}"
FILTERED_CSV="${3:-${SCRIPT_DIR}/../../olive_str_genotyping/results_all_bam_files/str_var_0.1.csv}"
OUTPUT_BED="${4:-${SCRIPT_DIR}/O_europeana_strs.functional.annotated.bed}"
SUMMARY_TSV="${5:-${SCRIPT_DIR}/str_functional_annotation_summary.tsv}"
PROMOTER_BP="${6:-0}"

if [[ ! "$PROMOTER_BP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: promoter size must be a non-negative integer: $PROMOTER_BP" >&2
    exit 1
fi

if [[ -x "${CONDA_ENV}/bin/bedtools" ]]; then
    BEDTOOLS="${CONDA_ENV}/bin/bedtools"
elif command -v bedtools >/dev/null 2>&1; then
    BEDTOOLS="$(command -v bedtools)"
else
    echo "ERROR: bedtools is not available" >&2
    exit 1
fi

for input in "$ANNOTATED_BED" "$GFF3" "$FILTERED_CSV"; do
    if [[ ! -s "$input" ]]; then
        echo "ERROR: input file is missing or empty: $input" >&2
        exit 1
    fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Extract GFF3 features, converting 1-based inclusive coordinates to BED.
gzip -cd "$GFF3" |
    awk -F '\t' -v out="$tmpdir" 'BEGIN { OFS="\t" }
        $0 !~ /^#/ && ($3 == "gene" || $3 == "exon" || $3 == "CDS" ||
                       $3 == "five_prime_UTR" || $3 == "three_prime_UTR") {
            print $1, $4-1, $5 > (out "/" $3 ".raw.bed")
            if ($3 == "gene")
                print $1, $4-1, $5, $7 > (out "/gene_with_strand.raw.bed")
        }'

for feature in gene exon CDS five_prime_UTR three_prime_UTR; do
    LC_ALL=C sort -k1,1 -k2,2n -k3,3n "$tmpdir/${feature}.raw.bed" |
        "$BEDTOOLS" merge -i - > "$tmpdir/${feature}.bed"
done

# Introns are genic bases outside the union of all exons. Other exons exclude
# bases already annotated as CDS or either UTR class.
"$BEDTOOLS" subtract -a "$tmpdir/gene.bed" -b "$tmpdir/exon.bed" > "$tmpdir/intron.bed"
cat "$tmpdir/CDS.bed" "$tmpdir/five_prime_UTR.bed" "$tmpdir/three_prime_UTR.bed" |
    LC_ALL=C sort -k1,1 -k2,2n -k3,3n |
    "$BEDTOOLS" merge -i - > "$tmpdir/coding_and_utr.bed"
"$BEDTOOLS" subtract -a "$tmpdir/exon.bed" -b "$tmpdir/coding_and_utr.bed" > "$tmpdir/other_exon.bed"

# Define putative promoters as the strand-aware region immediately upstream of
# the annotated gene boundary/TSS. Genic features retain priority where a
# promoter overlaps a neighboring gene.
if [[ "$PROMOTER_BP" -gt 0 ]]; then
    awk -v window="$PROMOTER_BP" 'BEGIN { OFS="\t" }
        $4 == "+" {
            start=$2-window
            if (start < 0) start=0
            if (start < $2) print $1, start, $2
        }
        $4 == "-" { print $1, $3, $3+window }' \
        "$tmpdir/gene_with_strand.raw.bed" |
        LC_ALL=C sort -k1,1 -k2,2n -k3,3n |
        "$BEDTOOLS" merge -i - > "$tmpdir/promoter.bed"
fi

# Lower numeric priority wins when an STR touches multiple feature classes.
awk 'BEGIN { OFS="\t" } { print $0, 1, "CDS" }' "$tmpdir/CDS.bed" > "$tmpdir/features.bed"
awk 'BEGIN { OFS="\t" } { print $0, 2, "five_prime_UTR" }' "$tmpdir/five_prime_UTR.bed" >> "$tmpdir/features.bed"
awk 'BEGIN { OFS="\t" } { print $0, 3, "three_prime_UTR" }' "$tmpdir/three_prime_UTR.bed" >> "$tmpdir/features.bed"
awk 'BEGIN { OFS="\t" } { print $0, 4, "other_exon" }' "$tmpdir/other_exon.bed" >> "$tmpdir/features.bed"
awk 'BEGIN { OFS="\t" } { print $0, 5, "intron" }' "$tmpdir/intron.bed" >> "$tmpdir/features.bed"
if [[ "$PROMOTER_BP" -gt 0 ]]; then
    awk 'BEGIN { OFS="\t" } { print $0, 6, "promoter" }' "$tmpdir/promoter.bed" >> "$tmpdir/features.bed"
fi
LC_ALL=C sort -k1,1 -k2,2n -k3,3n "$tmpdir/features.bed" -o "$tmpdir/features.bed"

# Add a stable row number so assignments remain one-to-one even if coordinates
# are ever duplicated in a future STR catalogue.
awk -F '\t' 'BEGIN { OFS="\t" } NR > 1 { print $1, $2, $3, NR-1 }' \
    "$ANNOTATED_BED" > "$tmpdir/str_rows.bed"

"$BEDTOOLS" intersect -a "$tmpdir/str_rows.bed" -b "$tmpdir/features.bed" -wa -wb |
    awk -F '\t' 'BEGIN { OFS="\t" }
        !($4 in priority) || $8 < priority[$4] {
            priority[$4]=$8
            category[$4]=$9
        }
        END { for (row in category) print row, category[row] }' |
    sort -k1,1n > "$tmpdir/assignments.tsv"

awk -F '\t' 'BEGIN { OFS="\t" }
    NR == FNR { category[$1]=$2; next }
    FNR == 1 { print $0, "functional_region"; next }
    { row=FNR-1; print $0, (row in category ? category[row] : "intergenic") }' \
    "$tmpdir/assignments.tsv" "$ANNOTATED_BED" > "$OUTPUT_BED"

input_count=$(awk 'END { print NR-1 }' "$ANNOTATED_BED")
output_count=$(awk 'END { print NR-1 }' "$OUTPUT_BED")
if [[ "$input_count" -ne "$output_count" ]]; then
    echo "ERROR: expected $input_count output loci but produced $output_count" >&2
    exit 1
fi

awk -F '\t' 'BEGIN { OFS="\t" }
    NR == FNR {
        if (FNR > 1) {
            split($0, fields, ",")
            filtered[fields[1]]=1
        }
        next
    }
    FNR > 1 {
        full[$14]++
        id=$1 "_" $2
        if (id in filtered) {
            subset[$14]++
            matched[id]=1
        }
    }
    END {
        order[1]="CDS"
        order[2]="five_prime_UTR"
        order[3]="three_prime_UTR"
        order[4]="other_exon"
        order[5]="intron"
        n=5
        if (promoter_bp > 0) order[++n]="promoter"
        order[++n]="intergenic"
        for (key in full) full_total+=full[key]
        for (key in subset) subset_total+=subset[key]
        print "dataset", "functional_region", "count", "percentage"
        for (i=1; i<=n; i++) {
            key=order[i]
            printf "complete_catalogue\t%s\t%d\t%.4f\n", key, full[key]+0, 100*full[key]/full_total
        }
        for (i=1; i<=n; i++) {
            key=order[i]
            printf "variance_ge_0.1\t%s\t%d\t%.4f\n", key, subset[key]+0, 100*subset[key]/subset_total
        }
        if (length(filtered) != length(matched)) {
            printf "ERROR: matched %d of %d filtered loci\n", length(matched), length(filtered) > "/dev/stderr"
            exit 2
        }
    }' promoter_bp="$PROMOTER_BP" "$FILTERED_CSV" "$OUTPUT_BED" > "$SUMMARY_TSV"

echo "Functionally annotated $output_count STR loci: $OUTPUT_BED"
if [[ "$PROMOTER_BP" -gt 0 ]]; then
    echo "Promoter definition: ${PROMOTER_BP} bp upstream of the strand-aware annotated gene TSS"
fi
echo "Wrote functional annotation summary: $SUMMARY_TSV"
