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
#
#   mkdir -p logs
#   python preflight_panel.py                             # gate the run
#   sbatch --export=ALL,ARM=weighted 05_train_all.sh      # weighted arm
#   sbatch --export=ALL,ARM=base     05_train_all.sh      # replication arm
#
# ARM defaults to weighted, so a bare `sbatch 05_train_all.sh` runs the
# weighted arm. Both arms can queue at once -- they write to different
# directories and never touch the same file. The %4 throttle caps
# concurrent tasks so the GPU partition stays usable for everyone else.
#
# Re-running is safe and cheap: tasks whose output already exists exit
# immediately, so a partial sweep is resumed by resubmitting the same line.
#
# ---------------------------------------------------------------------
# COVERAGE = paper  (changed from first-chunk)
#
# Data_Cleaning.r chunks a portfolio into k = ceil(n/62) EQUAL-size chunks
# of ceil(n/k) positions, so chunk 1 is usually SHORTER than 62:
#
#     n = 63  -> 32, 31          chunk 1 holds 32
#     n = 125 -> 42, 42, 41      chunk 1 holds 42
#     n = 187 -> 47, 47, 47, 46  chunk 1 holds 47
#
# --coverage first-chunk read chunks[0] and therefore pooled over fewer
# positions than Gabaix et al., who specify "the largest 62 positions".
# --coverage paper joins the chunks and takes the top 62, which is the
# paper's slice regardless of where the splits fell.
#
# Both arms now run through BERT_training_weighted.py so that a single
# code path serves them and they differ in POOLING ALONE. Routing the
# baseline through BERT_training.py would reintroduce the old extraction
# slice on one side only, and the arm comparison would no longer isolate
# the pooling effect.
#
# The output filenames carry the new variant tags, so nothing collides
# with the superseded __weighted_first-chunk / unsuffixed files. Those
# stay on disk; update the paths in Clusters.r and Cluster_Dynamics_v2.r
# or they will keep reading the old ones.
# ---------------------------------------------------------------------
#
# Watch:   squeue -u $USER
#          sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
# Failures only, after the fact:
#          sacct -j <jobid> --state=FAILED,TIMEOUT --format=JobID,State

set -euo pipefail

ARM="${ARM:-weighted}"

# One script for both arms; --pooling is the only difference.
SCRIPT="BERT_training_weighted.py"

case "$ARM" in
  weighted)
    EMB_DIR="embeddings_weighted"
    MODEL_DIR="models_weighted"
    EXTRA="--pooling weighted --coverage paper"
    SUFFIX="__weighted_paper"
    ;;
  base)
    EMB_DIR="embeddings_v2"
    MODEL_DIR="models_v2"
    EXTRA="--pooling mean --coverage paper"
    SUFFIX="__mean_paper"
    ;;
  *)
    echo "ARM must be 'weighted' or 'base', got '$ARM'" >&2; exit 2 ;;
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
echo "coverage=paper  out=$OUT"
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

# Guard against the variant tag drifting away from $SUFFIX. If the weights
# column were ever missing from the sequence parquets, variant_tag() would
# append _power0.75 and the file below would land under a different name --
# the run would "succeed" while the existence check at the end failed.
python - "$FILE" <<'PY'
import sys, pyarrow.parquet as pq
names = pq.read_schema(sys.argv[1]).names
if "weights" not in names:
    sys.exit("ERROR: no 'weights' column in %s; the variant tag would gain a "
             "_power<alpha> suffix and not match SUFFIX." % sys.argv[1])
PY

python "$SCRIPT" --only "$QUARTER" --no-skip $EXTRA \
    --emb-dir "$EMB_DIR" --model-dir "$MODEL_DIR"

if [[ ! -f "$OUT" ]]; then
  echo "ERROR: expected $OUT was not written" >&2
  ls -1 "$EMB_DIR" | grep -- "$QUARTER" || echo "  (no file for $QUARTER at all)"
  exit 1
fi

echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "finished $(date)"