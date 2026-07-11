#!/bin/bash
# Hopper venv via uv: torch 2.7.0+cu126 + flash-attn source build.
# sbatch scripts/setup_hopper_env.slurm
# Modules: hosts/hopper gnu12/12.3.0 cuda/12.6.2-ou python/3.11.7-cx  (no ohpc)
#SBATCH --job-name=setup-ebdlm-env
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --constraint=amd
#SBATCH --output=setup-ebdlm-env-%j.out
#SBATCH --error=setup-ebdlm-env-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
REPO="${SCRATCH}/ebdlm-babylm"

module load hosts/hopper gnu12/12.3.0 cuda/12.6.2-ou python/3.11.7-cx

export UV_CACHE_DIR="${SCRATCH}/uv_cache"
export XDG_CACHE_HOME="${SCRATCH}/xdg_cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

cd "${REPO}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

export FLASH_ATTN_CUDA_ARCHS="80"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=$(( SLURM_CPUS_PER_TASK * 3 / 4 ))
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TOKENIZERS_PARALLELISM=false

rm -rf .venv
uv sync --all-groups --python 3.11

"${REPO}/.venv/bin/python" -c "
import torch, flash_attn
from models.factory import _attn_implementation
assert '+cu126' in torch.__version__
assert torch.cuda.is_available()
assert _attn_implementation() == 'flash_attention_2'
print('OK', torch.__version__, flash_attn.__version__)
"
