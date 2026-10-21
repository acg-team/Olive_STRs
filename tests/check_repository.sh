#!/usr/bin/env bash
set -euo pipefail

private_path_pattern='/cfs/earth''/scratch/|/net''/home/'
if grep -R -n -E \
  --include='*.py' --include='*.R' --include='*.sh' --include='*.slurm' \
  --include='*.sbatch' --include='*.json' \
  "$private_path_pattern" .; then
  echo "Private absolute path found" >&2
  exit 1
fi

if find . -type f \( -name '*.h5' -o -name '*.safetensors' -o -name '*.vcf.gz' \
  -o -path '*/Slurm-*/*' \) -print -quit | grep -q .; then
  echo "Generated or bulky artifact found" >&2
  exit 1
fi

