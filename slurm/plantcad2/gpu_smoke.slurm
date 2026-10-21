#!/usr/bin/env bash
#SBATCH --job-name=plantcad2_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plantcad2_smoke_%j.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/plantcad2"
mkdir -p logs results/smoke
"$PY" scripts/embed_shard.py --config config.json --smoke-benchmark
"$PY" scripts/validate_embeddings.py --config config.json --smoke

