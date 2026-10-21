#!/usr/bin/env bash
#SBATCH --job-name=agront_analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/agront_analyze_%A_%a.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/agront"
mkdir -p logs results/reference_anchored_shards
SHARDS=$("$PY" -c 'import json; print(json.load(open("results/preparation_manifest.json"))["shards"])')
for ((SHARD=SLURM_ARRAY_TASK_ID; SHARD<SHARDS; SHARD+=SLURM_ARRAY_TASK_COUNT)); do
  "$PY" scripts/analyze_shard.py --config config.json --shard "$SHARD"
done

