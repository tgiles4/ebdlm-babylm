#!/bin/bash
# BabyLM Strict LLaDA pretraining (4× A100 80GB, Lightning DDP).
# Prerequisite: pretokenized/strict/, tokenizers/tokenizer_strict.json (prepare_pretrain_data.slurm),
# and .venv (setup_hopper_env.slurm).
# Lightning auto-selects DDP when SLURM allocates multiple GPUs.
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
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
REPO="${SCRATCH}/ebdlm-babylm"
cd "${REPO}"

# Load modules (match setup_hopper_env.slurm; do not load ohpc).
module load hosts/hopper
module load gnu12/12.3.0
module load cuda/12.6.2-ou
module load python/3.11.7-cx

PYTHON="${REPO}/.venv/bin/python"

# W&B: load API key from a file (avoids putting secrets in the script or sbatch env).
# Create once: echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

export TOKENIZERS_PARALLELISM=false

# Defaults: dataset=strict, model=base, trainer.max_epochs=10, logging=wandb.
"${PYTHON}" -u train.py
