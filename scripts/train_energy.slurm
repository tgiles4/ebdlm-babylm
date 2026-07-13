#!/bin/bash
# BabyLM Strict EDLM energy-head training (NCE) on a frozen LLaDAMDLM.
# Prerequisite: trained LLaDA HF export under runs/<run>/hf/, pretokenized/strict/,
# tokenizer, and .venv (setup_hopper_env.slurm).
# One SLURM task with --gres=gpu:2; Lightning spawns DDP locally (see trainer_runtime.py).
#
# BabyLM Strict word budget is 1000M total across LLaDA + energy. Leave
# headroom (e.g. chck_700M); do not use chck_1000M.
#
# Submit from repo root:
#   sbatch scripts/train_energy.slurm 2026-06-28_213248_strict chck_700M
#
#SBATCH --job-name=ebdlm-energy
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=ebdlm-energy-%j.out
#SBATCH --error=ebdlm-energy-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Prerequisite: repo at ${SCRATCH}/ebdlm-babylm, venv via setup_hopper_env.slurm.
set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
REPO="${SCRATCH}/ebdlm-babylm"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

cd "${REPO}"

if [ "$#" -ne 2 ]; then
  echo "Usage: sbatch scripts/train_energy.slurm <run_name> <chck_name>" >&2
  echo "Example: sbatch scripts/train_energy.slurm 2026-06-28_213248_strict chck_700M" >&2
  exit 1
fi

LLADA_RUN="$1"
CHCK="$2"
LLADA_CHECKPOINT="${REPO}/runs/${LLADA_RUN}/hf/${CHCK}"
if [ ! -d "${LLADA_CHECKPOINT}" ]; then
  echo "Missing LLaDA HF export: ${LLADA_CHECKPOINT}" >&2
  exit 1
fi

# Load modules (match setup_hopper_env.slurm; do not load ohpc).
module load hosts/hopper gnu12/12.3.0 cuda/12.6.2-ou python/3.11.7-cx

PYTHON="${REPO}/.venv/bin/python"

# W&B: --export=NONE clears the submit env; load key from file (not a wandb default path).
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

export TOKENIZERS_PARALLELISM=false
# One thread per process; DDP ranks + dataloader workers share 16 Slurm CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Defaults: dataset=strict, model=base, trainer.max_epochs=10, logging=wandb.
srun --export=ALL "${PYTHON}" -u train.py \
  train_energy=true \
  "paths.llada_checkpoint=${LLADA_CHECKPOINT}"
