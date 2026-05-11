#!/bin/bash
# 00_setup_env.sh
# Run this ONCE, on the login node:
#     bash 00_setup_env.sh
#
# What it does:
#   1. Loads the cluster's miniconda module
#   2. Creates a conda env called 'psbert' in your home directory
#   3. Installs PyTorch (CUDA 12.1 build) + transformers + the usual ML stack
#
# Time:  ~5–10 minutes (downloads ~3 GB of pip wheels)
# Disk:  ~5 GB in $HOME/.conda/envs/psbert

set -euo pipefail

ENV_NAME="psbert"
PYTHON_VER="3.11"

echo ">>> Loading miniconda module..."
module purge                              # start clean
module load miniconda3                    # cluster-provided Miniconda

# Initialize conda for this shell session (no permanent ~/.bashrc edit)
eval "$(conda shell.bash hook)"

echo ">>> Conda is now: $(which conda)"
conda --version

# ---------- Create the env if it doesn't exist ----------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Env '$ENV_NAME' already exists — skipping creation."
else
    echo ">>> Creating env '$ENV_NAME' with Python $PYTHON_VER..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VER" pip
fi

conda activate "$ENV_NAME"
echo ">>> Active env: $CONDA_DEFAULT_ENV"
echo ">>> Python:     $(which python) ($(python --version))"

# ---------- Install PyTorch with CUDA 12.1 ----------
# The A30 driver (CUDA 13.0 reported by nvidia-smi) is backward-compatible
# with CUDA 12.1 builds. Using cu121 because the wheels are mature.
echo ">>> Installing PyTorch (CUDA 12.1 build)..."
pip install --upgrade pip wheel
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# ---------- Install the rest ----------
echo ">>> Installing transformers + ML stack..."
pip install \
    transformers==4.42.4 \
    accelerate==0.32.1 \
    datasets==2.20.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    pyarrow==16.1.0 \
    scikit-learn==1.5.1 \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    tensorboard==2.17.0

echo ""
echo "=========================================================="
echo "  Setup complete."
echo ""
echo "  In any future shell, activate the env with:"
echo "      module load miniconda3"
echo "      conda activate psbert"
echo ""
echo "  In SLURM scripts these two lines are already included."
echo "=========================================================="
