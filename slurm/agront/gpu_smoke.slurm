#!/usr/bin/env bash
#SBATCH --job-name=agront_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agront_smoke_%j.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/agront"
mkdir -p logs results/embeddings
"$PY" scripts/embed_shard.py --config config.json --shard 0 --limit-alleles 8 --force

