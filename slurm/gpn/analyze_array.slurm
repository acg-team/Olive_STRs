#!/usr/bin/env bash
#SBATCH --job-name=gpn_analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gpn_analyze_%A_%a.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/gpn"
mkdir -p logs results/reference_anchored_shards
for ((SHARD=SLURM_ARRAY_TASK_ID; SHARD<50; SHARD+=SLURM_ARRAY_TASK_COUNT)); do
  "$PY" scripts/analyze_shard.py --config config.json --shard "$SHARD"
done

