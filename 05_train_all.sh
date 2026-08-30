#!/bin/bash
#SBATCH --job-name=psbert-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-84%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGE_ME
#
# Train one quarter per array task, across the full 85-quarter panel.
# Replaces the hardcoded $QUARTERS list in 03_/04_train_recent*.sh.
#
#   mkdir -p logs
#   python preflight_panel.py                             # gate the run
#   sbatch --export=ALL,ARM=base     05_train_all.sh      # replication arm
#   sbatch --export=ALL,ARM=weighted 05_train_all.sh      # weighted arm
#
# Both arms can queue at once -- they write to different directories and
# never touch the same file. The %4 throttle caps concurrent tasks so the
# GPU partition stays usable for everyone else.
#
# Re-running is safe and cheap: tasks whose output already exists exit
# immediately, so a partial sweep is resumed by resubmitting the same line.
#
# Watch:   squeue -u $USER
#          sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
# Failures only, after the fact:
#          sacct -j <jobid> --state=FAILED,TIMEOUT --format=JobID,State

set -euo pipefail

ARM="${ARM:-base}"

case "$ARM" in
  base)
    SCRIPT="BERT_training.py"
    EMB_DIR="embeddings_v2"
    MODEL_DIR="models_v2"
    EXTRA=""
    SUFFIX=""
    ;;
  weighted)
    SCRIPT="BERT_training_weighted.py"
    EMB_DIR="embeddings_weighted"
    MODEL_DIR="models_weighted"
    EXTRA="--pooling weighted --coverage first-chunk"
    SUFFIX="__weighted_first-chunk"
    ;;
  *)
    echo "ARM must be 'base' or 'weighted', got '$ARM'" >&2; exit 2 ;;
esac

cd ~/Master-s-Thesis

mapfile -t FILES < <(ls -1 data/q_*.parquet | sort)
N=${#FILES[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}

if (( IDX >= N )); then
  echo "Task $IDX beyond panel size $N -- nothing to do."
  echo "Set --array=0-$((N-1))%4 to match the panel exactly."
  exit 0
fi

FILE="${FILES[$IDX]}"
QUARTER=$(basename "$FILE" .parquet); QUARTER="${QUARTER#q_}"
OUT="${EMB_DIR}/q_${QUARTER}${SUFFIX}.parquet"

echo "═══════════════════════════════════════════════════════════════"
echo "arm=$ARM  quarter=$QUARTER  task=$IDX/$((N-1))  host=$(hostname)"
echo "started $(date)"
echo "═══════════════════════════════════════════════════════════════"

if [[ -f "$OUT" ]]; then
  echo "$OUT already exists -- skipping."
  exit 0
fi

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$EMB_DIR" "$MODEL_DIR"

python "$SCRIPT" --only "$QUARTER" --no-skip $EXTRA \
    --emb-dir "$EMB_DIR" --model-dir "$MODEL_DIR"

if [[ ! -f "$OUT" ]]; then
  echo "ERROR: expected $OUT was not written" >&2
  exit 1
fi

echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "finished $(date)"
