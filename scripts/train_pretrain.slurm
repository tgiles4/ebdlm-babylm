#!/bin/bash
# BabyLM Strict LLaDA pretraining (4× A100 80GB, Lightning DDP).
# Prerequisite: pretokenized/strict/ and tokenizers/tokenizer_strict.json (see prepare_pretrain_data.slurm).
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

# Clone repo and set up environment (run on compute node). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path="${SCRATCH}/ebdlm"
mkdir -p "${path}"
cd "${path}"

if [ ! -d "ebdlm-babylm" ]; then
  git clone git@github.com:tgiles4/ebdlm-babylm.git
fi
cd ebdlm-babylm

# Load modules (match prepare_pretrain_data.slurm + requirements-hopper.txt).
module load hosts/hopper
module load gnu12/12.3.0
module load cuda/12.4.0
module load python/3.11.7-cx

ENV="${SCRATCH}/ebdlm/venv"

# W&B: load API key from a file (avoids putting secrets in the script or sbatch env).
# Create once: echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(cat "${HOME}/.wandb_api_key")"
fi

export TOKENIZERS_PARALLELISM=false

# Defaults: dataset=strict, model=base, trainer.max_epochs=10, logging=wandb.
"$ENV/bin/python" -u train.py
