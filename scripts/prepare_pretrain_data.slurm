#!/bin/bash
# Download BabyLM Strict, prep, pack, and pretokenize for EBDLM pretraining.
# CPU-only; uses .venv in repo on scratch (not uv). ~22 min on a Core Ultra 7 (8 workers); 48 CPUs on Hopper.
# Keep walltime tight for scheduler backfill — bump only if HF download + tokenizer train on a cold start.
# Output: pretokenized/strict/ (HF dataset on disk, input_ids @ 512); tokenizers/tokenizer_strict.json if missing.
# Submit from repo root: sbatch scripts/prepare_pretrain_data.slurm
#SBATCH --partition=normal
#SBATCH --job-name=prep-strict
#SBATCH --output=prep-strict-%j.out
#SBATCH --error=prep-strict-%j.err
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2GB

# Prerequisite: repo at ${SCRATCH}/ebdlm-babylm, venv via setup_hopper_env.slurm.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
REPO="${SCRATCH}/ebdlm-babylm"
cd "${REPO}"

# Load modules (py311 only; CPU pretokenize does not need CUDA toolkit).
module load hosts/hopper
module load gnu12/12.3.0
module load python/3.11.7-cx

PYTHON="${REPO}/.venv/bin/python"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-48}"

# BPE training (tokenizers crate / Rayon) when tokenizer JSON is missing.
export RAYON_RS_NUM_THREADS="${NUM_WORKERS}"
# Pretokenize uses datasets.map(num_proc=...); avoid per-worker thread oversubscription.
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" -u prepare_pretrain_data.py \
  num_workers="${NUM_WORKERS}"
