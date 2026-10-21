#!/usr/bin/env bash
#SBATCH --job-name=agront_embed
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/agront_embed_%A_%a.out
set -euo pipefail
REPO="${OLIVE_STRS_REPO:?Set OLIVE_STRS_REPO to the repository root}"
PY="${PYTHON:-python}"
cd "$REPO/scripts/03_embeddings/agront"
mkdir -p logs results/embeddings
SHARDS=$("$PY" -c 'import json; print(json.load(open("results/preparation_manifest.json"))["shards"])')
for ((SHARD=SLURM_ARRAY_TASK_ID; SHARD<SHARDS; SHARD+=SLURM_ARRAY_TASK_COUNT)); do
  "$PY" scripts/embed_shard.py --config config.json --shard "$SHARD"
done

