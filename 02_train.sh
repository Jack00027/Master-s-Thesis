#!/bin/bash
#SBATCH --job-name=psbert-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=h12429918@wu.ac.at

# Full PS-BERT training sweep across all quarters in data/.
#
# Before submitting, make sure logs/ exists:
#   mkdir -p logs
#
# Submit with:
#   sbatch 02_train.sh                         # full sweep
#   sbatch 02_train.sh --first-only            # just the first quarter (dress rehearsal)
#
# Watch progress:
#   squeue -u $USER
#   tail -f logs/psbert-train-<jobid>.out

set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "PS-BERT training on $(hostname)"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "═══════════════════════════════════════════════════════════════"

# Show what GPU we got
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Load env
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate psbert

cd ~/Master-s-Thesis

# Optional: if first arg is --first-only, hack the script to stop after quarter 1.
# Otherwise run the full sweep.
if [[ "${1:-}" == "--first-only" ]]; then
    echo ""
    echo "→ DRESS REHEARSAL MODE: stopping after first quarter"
    echo ""
    # Process just the first quarter by temporarily renaming everything else.
    # (Simpler: rely on skip_existing once quarter 1 is done, then resubmit for full.)
    python -c "
import sys
from pathlib import Path
files = sorted(Path('data').glob('q_*.parquet'))
if not files:
    print('No data found', file=sys.stderr); sys.exit(1)
print(f'First quarter: {files[0].name}')
print(f'Total quarters available: {len(files)}')
"
    # Run training but limit to first quarter via a small Python wrapper
    python <<'PYEOF'
import sys
sys.argv = ['BERT_training.py']
from pathlib import Path
import BERT_training as bt

cfg = bt.default_config()
import random, numpy as np, torch
random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])

print(f"device     : {cfg['device']}")
print(f"bf16       : {cfg['device'] == 'cuda'}")
print(f"seq_dir    : {cfg['seq_dir']}")

files = sorted(Path(cfg["seq_dir"]).glob("q_*.parquet"))
print(f"\nProcessing only the FIRST of {len(files)} quarters\n")
bt.process_quarter(files[0], cfg)
print("\nDress rehearsal complete.")
PYEOF
else
    echo ""
    echo "→ FULL SWEEP MODE: processing all quarters"
    echo ""
    python BERT_training.py
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
