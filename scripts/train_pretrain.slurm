#!/bin/bash
# BabyLM Strict LLaDA pretraining (4× A100 80GB, Lightning DDP).
# Prerequisite: pretokenized/strict/, tokenizers/tokenizer_strict.json (prepare_pretrain_data.slurm),
# and .venv (setup_hopper_env.slurm).
# One SLURM task with --gres=gpu:4; Lightning spawns DDP locally (see trainer_runtime.py).
# Submit from repo root: sbatch scripts/train_pretrain.slurm
#SBATCH --job-name=ebdlm-pretrain
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:4

#SBATCH --output=ebdlm-pretrain-%j.out
#SBATCH --error=ebdlm-pretrain-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

# Prerequisite: repo at ${SCRATCH}/ebdlm-babylm, venv via setup_hopper_env.slurm.
set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
REPO="${SCRATCH}/ebdlm-babylm"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

cd "${REPO}"

# Load modules (match setup_hopper_env.slurm; do not load ohpc).
module load hosts/hopper gnu12/12.3.0 cuda/12.6.2-ou python/3.11.7-cx

PYTHON="${REPO}/.venv/bin/python"

# W&B: --export=NONE clears the submit env; load key from file (not a wandb default path).
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

export TOKENIZERS_PARALLELISM=false
# One thread per process; DDP ranks + dataloader workers share 32 Slurm CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Defaults: dataset=strict, model=base, trainer.max_epochs=10, logging=wandb.
srun --export=ALL "${PYTHON}" -u train.py
