#!/usr/bin/env bash
#SBATCH --job-name=agront_prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/agront_prepare_%j.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/agront"
mkdir -p logs
"$PY" scripts/prepare_catalog.py --config config.json

