#!/bin/bash
#SBATCH --job-name=psbert-weighted
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=h12429918@wu.ac.at

# 03_train_weighted.sh
# ---------------------------------------------------------------------------
# Runs BERT_training_weighted.py on one A30 GPU, sequentially over all
# quarters found in data/. Any arguments passed to sbatch after the script
# name are forwarded straight to the Python script, e.g.
#
#   mkdir -p logs
#   sbatch 03_train_weighted.sh --dry-run
#   sbatch 03_train_weighted.sh --only 2005-03-31
#   sbatch 03_train_weighted.sh                              # full sweep
#   sbatch 03_train_weighted.sh --pooling mean --coverage first-chunk
#   sbatch 03_train_weighted.sh --coverage full-sequence --context-window 256
#
# The Python script skips quarters whose embedding parquet already exists,
# so if the job hits the walltime you can just resubmit and it picks up
# where it stopped.
# ---------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs embeddings_weighted models_weighted

# ---------- Environment ----------
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

export PYTHONUNBUFFERED=1                       # stream prints into the .out file
export TOKENIZERS_PARALLELISM=false             # silences the HF fork warning
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---------- Diagnostics ----------
echo "=========================================================="
echo "job id     : ${SLURM_JOB_ID:-none}"
echo "node       : $(hostname)"
echo "started    : $(date)"
echo "workdir    : $PWD"
echo "python     : $(which python)"
echo "extra args : $*"
echo "----------------------------------------------------------"
nvidia-smi || echo "[warn] nvidia-smi not available"
python - <<'PY'
import torch, transformers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    print("bf16        :", torch.cuda.is_bf16_supported())
PY
echo "=========================================================="
echo ""

# ---------- Sanity check on the input data ----------
N_Q=$(ls -1 data/q_*.parquet 2>/dev/null | wc -l)
if [ "$N_Q" -eq 0 ]; then
    echo "ERROR: no q_*.parquet files under data/ — check the symlink."
    exit 1
fi
echo "found $N_Q quarterly parquet files in data/"
echo ""

# ---------- Train ----------
START=$SECONDS
python BERT_training_weighted.py "$@"
STATUS=$?

echo ""
echo "=========================================================="
echo "exit status : $STATUS"
echo "elapsed     : $(( (SECONDS - START) / 3600 ))h $(( ((SECONDS - START) % 3600) / 60 ))m"
echo "finished    : $(date)"
echo "embeddings  : $(ls -1 embeddings_weighted/*.parquet 2>/dev/null | wc -l) files"
echo "=========================================================="

exit $STATUS
