#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --account=def-craffel_gpu
#SBATCH --job-name=create_env_fir
#SBATCH --output=logs/%j_create_env_fir.out
#SBATCH --error=logs/%j_create_env_fir.out

# Built as a GPU job so torch/bitsandbytes resolve against the node's CUDA.
# Usage: mkdir -p logs && sbatch setup/create_env_fir_uv.sh

set -e

mkdir -p logs

echo "============================================"
echo "Creating Proteus environment with uv at $(date)"
echo "============================================"

# fir is an Alliance cluster: load the software stack before cuda/python.
module load StdEnv/2023
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

uv sync

source .venv/bin/activate

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "import proteus; print('proteus', proteus.__version__)"

echo "============================================"
echo "Environment created. Sanity check the menu with:"
echo "  python scripts/inspect_menu.py --menu dev"
echo "============================================"
