# HPC execution

The `slurm/` directory contains portable examples for the computationally
expensive embedding and association stages. Edit `#SBATCH` resource requests
for the local cluster, then export the repository and Python locations:

```bash
export OLIVE_STRS_REPO=/path/to/Olive_STRs
export PYTHON=/path/to/model/environment/bin/python
export HF_HOME=/path/to/huggingface/cache
mkdir -p logs
sbatch --array=0-3 slurm/agront/embed_array.slurm
```

Submit jobs from the repository root so relative Slurm log paths resolve under
`logs/`. Run each model's smoke test before its production array. Embedding
shards use partial files and atomic renaming, so failed array tasks can be
resubmitted safely.

The manuscript production design used four A100 workers. CPU regression arrays
used four workers, and each worker processed shard numbers congruent to its
array index. Full embeddings require hundreds of gigabytes of transient disk
space and are intentionally excluded from Git and Zenodo.

