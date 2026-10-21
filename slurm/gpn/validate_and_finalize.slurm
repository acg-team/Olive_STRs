#!/usr/bin/env bash
#SBATCH --job-name=gpn_finalize
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpn_finalize_%j.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/gpn"
mkdir -p logs
"$PY" scripts/validate_embeddings.py --config config.json
"$PY" scripts/build_master_h5.py --config config.json
"$PY" scripts/aggregate_results.py --config config.json
"$PY" "$REPO/figures/03_embeddings/gpn/plot_results.py" --config config.json

