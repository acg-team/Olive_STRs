#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/reference_model_comparison_mpl"

"${PYTHON}" "${HERE}/scripts/aggregate_and_report.py"
"${PYTHON}" "${REPO}/figures/04_cross_model_comparison/plot_locus_property_distributions.py"
"${PYTHON}" "${HERE}/gwas_overlap/scripts/check_gwas_overlap.py"

(
  cd "${HERE}"
  find results report figures \
       gwas_overlap/results gwas_overlap/report gwas_overlap/figures \
       -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
)
