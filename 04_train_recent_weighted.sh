#!/bin/bash
#SBATCH --job-name=psbert-w2025
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=h12429918@wu.ac.at

# 04_train_recent_weighted.sh
# WEIGHTED investor embeddings for the five most recent quarters
# (2025-Q1 .. 2026-Q1), the extension arm of the thesis.
#
# Submit with:
#   sbatch 04_train_recent_weighted.sh              # train
#   bash   04_train_recent_weighted.sh --dry-run    # list quarters, no GPU
#
# WHAT DIFFERS FROM THE BASELINE RUN (03_train_recent.sh)
#   --pooling weighted   each position enters the pooled embedding in
#                        proportion to its portfolio weight, instead of
#                        every position counting equally.
#   --coverage first-chunk
#                        embeds the top-62 positions, exactly as the
#                        baseline does. This is deliberate: it isolates the
#                        POOLING OPERATOR as the single difference between
#                        the two runs, so any change in the resulting
#                        clusters is attributable to weighting alone.
#                        Wider coverage (all-chunks / full-sequence) is a
#                        separate experiment; run it afterwards, not now.
#   Everything else - data, seed, architecture, epochs - is identical.
#
# Weights come from the `weights` column written by Data_Cleaning.r, so the
# pooling is exact rather than the rank-decay approximation. The script
# checks this below and REFUSES to run on the proxy, because a silent
# fallback would make the comparison meaningless.

set -euo pipefail

QUARTERS="2025-03-31 2025-06-30 2025-09-30 2025-12-31 2026-03-31"
PROJECT_DIR="$HOME/Master-s-Thesis"
EMB_DIR="embeddings_weighted"
MODEL_DIR="models_weighted"

if [[ "${1:-}" == "--dry-run" ]]; then
    module load miniconda3
    eval "$(conda shell.bash hook)"
    conda activate psbert
    cd "$PROJECT_DIR"
    python BERT_training_weighted.py --only $QUARTERS --no-skip --dry-run \
        --pooling weighted --coverage first-chunk \
        --emb-dir "$EMB_DIR" --model-dir "$MODEL_DIR"
    exit 0
fi

mkdir -p logs

echo "═══════════════════════════════════════════════════════════════"
echo "PS-BERT WEIGHTED training (extension arm)"
echo "Job ID:    ${SLURM_JOB_ID:-unknown}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Quarters:  $QUARTERS"
echo "Output:    $EMB_DIR/ and $MODEL_DIR/"
echo "═══════════════════════════════════════════════════════════════"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

cd "$PROJECT_DIR"

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

# The weighted run is only meaningful with REAL weights. Verify they are
# present, aligned with the tokens, and sum to one per investor before
# spending the GPU allocation.
echo ""
echo ">>> Verifying weights in the first target quarter"
python - <<'PY'
import numpy as np, pandas as pd
df = pd.read_parquet("data/q_2025-03-31.parquet")
for c in ("chunk_id", "weights", "tokens", "n_tokens"):
    if c not in df.columns:
        raise SystemExit(f"    ERROR: missing column '{c}'. This weighted run "
                         f"needs sequence files from the updated "
                         f"Data_Cleaning.r; without them the script would "
                         f"silently fall back to rank-decay weights.")
bad_len = int(sum(len(w) != len(t) for w, t in zip(df.weights, df.tokens)))
if bad_len:
    raise SystemExit(f"    ERROR: {bad_len} row(s) where weights and tokens "
                     f"differ in length")
sums = df.groupby("investor_id").weights.apply(lambda s: sum(np.sum(w) for w in s))
off = float(np.max(np.abs(sums - 1.0)))
print(f"    {len(df):,} chunks from {df.investor_id.nunique():,} investors")
print(f"    weights aligned with tokens: yes")
print(f"    max |sum(weights) - 1| per investor: {off:.2e}")
if off > 1e-6:
    raise SystemExit("    ERROR: weights do not sum to 1 per investor")
w0 = np.asarray(df[df.chunk_id == 1].weights.iloc[0])
print(f"    example top-chunk weights: {np.round(w0[:5], 4)} (descending)")
PY

echo ""
echo ">>> Training (weighted pooling, top-62 coverage)"
python BERT_training_weighted.py --only $QUARTERS --no-skip \
    --pooling weighted --coverage first-chunk \
    --emb-dir "$EMB_DIR" --model-dir "$MODEL_DIR"

echo ""
echo ">>> Resulting embedding files"
ls -lh "$EMB_DIR"/q_*.parquet 2>/dev/null || echo "    none found"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
