#!/bin/bash
# BabyLM Strict LLaDA pretraining (2× A100 80GB, Lightning DDP).
# Prerequisite: pretokenized/strict/, tokenizers/tokenizer_strict.json (prepare_pretrain_data.slurm),
# and .venv (setup_hopper_env.slurm).
# One SLURM task with --gres=gpu:2; Lightning spawns DDP locally (see trainer_runtime.py).
# Submit from repo root (Hydra overrides after the script name):
#   sbatch -J ebdlm-thin-A scripts/train_diffusion.slurm \
#     run.name=backbone-thin-196k model.intermediate_size=1152 \
#     trainer.per_device_batch_size=192 trainer.accumulate_grad_batches=1 \
#     trainer.learning_rate=0.007
#SBATCH --job-name=ebdlm-diffusion
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=ebdlm-diffusion-%j.out
#SBATCH --error=ebdlm-diffusion-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --mem=128G
#SBATCH --time=02:30:00

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
# One thread per process; DDP ranks + dataloader workers share 16 Slurm CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Defaults: dataset=strict, model=base, trainer.max_epochs=10, logging=wandb.
# "$@" forwards Hydra overrides from: sbatch ... scripts/train_diffusion.slurm key=val ...
srun --export=ALL "${PYTHON}" -u train.py "$@"
