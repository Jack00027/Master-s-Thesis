#!/bin/bash
#SBATCH --job-name=psbert-2025
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=h12429918@wu.ac.at

# 03_train_recent.sh
# Train PS-BERT on the five most recent quarters of the REGENERATED data
# (2025-Q1 .. 2026-Q1), which carry corrected quarter labels and the new
# weights column.
#
# Submit with:
#   sbatch 03_train_recent.sh              # train
#   bash   03_train_recent.sh --dry-run    # list quarters, no GPU needed
#
# Watch:
#   squeue -u $USER
#   tail -f logs/psbert-2025-<jobid>.out
#
# WHY --no-skip IS SET
#   embeddings/ still contains output from the previous run on the OLD token
#   files. Two of the five target quarters (2025-03-31 and 2025-12-31) were
#   correctly named back then, so their files already exist and would be
#   SILENTLY SKIPPED under the default skip_existing. Those stale embeddings
#   do not correspond to the regenerated sequences, so they must be rebuilt.
#
# Resource notes vs. the previous sweep:
#   --mem=48G   these are the largest quarters in the panel (up to ~32k
#               investors / ~93k sequences), roughly 60% larger than the
#               2019 quarters, and the whole parquet is read into memory.
#   --time=8h   five quarters; the earlier full sweep of 84 fitted in 12h,
#               so this is generous. Raise it if the log shows otherwise.

set -euo pipefail

QUARTERS="2025-03-31 2025-06-30 2025-09-30 2025-12-31 2026-03-31"
PROJECT_DIR="$HOME/Master-s-Thesis"

# --dry-run works on the login node without SLURM
if [[ "${1:-}" == "--dry-run" ]]; then
    module load miniconda3
    eval "$(conda shell.bash hook)"
    conda activate psbert
    cd "$PROJECT_DIR"
    python BERT_training.py --only $QUARTERS --no-skip --dry-run
    exit 0
fi

mkdir -p logs

echo "═══════════════════════════════════════════════════════════════"
echo "PS-BERT training (recent quarters, regenerated data)"
echo "Job ID:    ${SLURM_JOB_ID:-unknown}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Quarters:  $QUARTERS"
echo "═══════════════════════════════════════════════════════════════"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

cd "$PROJECT_DIR"

# Fail early and loudly if the regenerated files are not actually present,
# rather than discovering it after the GPU allocation has been consumed.
echo ""
echo ">>> Checking input files"
for q in $QUARTERS; do
    f="data/q_${q}.parquet"
    if [[ ! -f "$f" ]]; then
        echo "MISSING: $f — regenerate with Data_Cleaning.r before submitting"
        exit 1
    fi
    echo "    found $f"
done

# Confirm the sequences carry the new schema. A file without chunk_id is from
# the old pipeline and would train on the wrong thing.
echo ""
echo ">>> Checking schema of the first target quarter"
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/q_2025-03-31.parquet")
cols = list(df.columns)
print("    columns:", cols)
missing = [c for c in ("chunk_id", "weights", "tokens", "n_tokens") if c not in cols]
if missing:
    raise SystemExit(f"    ERROR: missing column(s) {missing}; this looks like "
                     f"an OLD sequence file")
n_inv = df.investor_id.nunique()
print(f"    {len(df):,} chunks from {n_inv:,} investors")
print(f"    first chunks: {(df.chunk_id == 1).sum():,} (should equal investors)")
PY

echo ""
echo ">>> Training"
# --no-skip: rebuild even where an embedding from the old run exists (see above)
python BERT_training.py --only $QUARTERS --no-skip

echo ""
echo ">>> Resulting embedding files"
for q in $QUARTERS; do
    ls -lh "embeddings/q_${q}.parquet" 2>/dev/null || echo "    MISSING q_${q}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
