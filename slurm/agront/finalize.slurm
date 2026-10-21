#!/usr/bin/env bash
#SBATCH --job-name=agront_finalize
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/agront_finalize_%j.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/agront"
mkdir -p logs
"$PY" scripts/aggregate_results.py --config config.json --force
"$PY" "$REPO/figures/03_embeddings/agront/plot_results.py" --config config.json

