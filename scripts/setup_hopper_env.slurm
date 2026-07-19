#!/bin/bash
# Hopper venv: torch 2.7.0+cu126 + flash-attn (cached Hopper wheel, else source build).
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

export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export XDG_CACHE_HOME="${SCRATCH}/xdg_cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

cd "${REPO}"
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

pip install -U pip wheel setuptools
pip install torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple

# Skip flash-attn lines: Dao manylinux wheels break on Hopper; install below from cache/source.
grep -viE '^[[:space:]]*flash-attn' requirements-hopper.txt > /tmp/requirements-hopper-nofa.txt
pip install -r /tmp/requirements-hopper-nofa.txt
pip install ninja packaging psutil einops

# Prefer the Hopper-built wheel already in pip_cache; otherwise compile.
FLASH_WHEEL="$(find "${PIP_CACHE_DIR}" -iname 'flash_attn-*.whl' 2>/dev/null | head -n 1 || true)"
if [[ -n "${FLASH_WHEEL}" ]]; then
  echo "Installing flash-attn from cache: ${FLASH_WHEEL}"
  pip install "${FLASH_WHEEL}"
else
  echo "No cached flash-attn wheel; building from source (slow)."
  export FLASH_ATTN_CUDA_ARCHS="80"
  export TORCH_CUDA_ARCH_LIST="8.0"
  export MAX_JOBS=$(( SLURM_CPUS_PER_TASK * 3 / 4 ))
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  pip install flash-attn==2.8.3 --no-build-isolation --no-binary flash-attn
fi

python -c "
import torch, flash_attn
from models.factory import _attn_implementation
assert '+cu126' in torch.__version__
assert torch.cuda.is_available()
assert _attn_implementation() == 'flash_attention_2'
print('OK', torch.__version__, flash_attn.__version__)
"
