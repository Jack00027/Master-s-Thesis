"""
Verify that an embedding parquet from BERT_training.py looks sane.

Runs a battery of automated checks:
  1. File presence       - parquet, model checkpoints, history.json
  2. Shape & validity    - no NaN/Inf, correct columns, no all-zero rows
  3. Distribution        - L2 norms, per-dimension variance
  4. Similarity structure - cosine sim range, diversity (not collapsed)
  5. Training health     - loss curves decreased over training
  6. Investor-type structure - intra-type vs inter-type similarity

Each check prints PASS / WARN / FAIL. Exit code:
  0 = all PASS
  1 = some WARN, no FAIL
  2 = at least one FAIL

Usage:
  python verify_embeddings.py                                    # first quarter in embeddings/
  python verify_embeddings.py embeddings/q_2005-03-31.parquet    # specific file
  python verify_embeddings.py --all embeddings/                  # all quarters in a dir
  python verify_embeddings.py path --model-dir models/q_XXXX/    # explicit model dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Output styling ───────────────────────────────────────────────────────
if sys.stdout.isatty():
    GREEN, YELLOW, RED, BOLD, RESET = (
        "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m",
    )
else:
    GREEN = YELLOW = RED = BOLD = RESET = ""

PASS_TAG = f"{GREEN}✓ PASS{RESET}"
WARN_TAG = f"{YELLOW}⚠ WARN{RESET}"
FAIL_TAG = f"{RED}✗ FAIL{RESET}"


class Reporter:
    """Tracks check results and produces a summary at the end."""

    def __init__(self):
        self.n_pass = 0
        self.n_warn = 0
        self.n_fail = 0
        self.failures = []

    def check(self, status, name, detail=""):
        if status == "pass":
            tag = PASS_TAG
            self.n_pass += 1
        elif status == "warn":
            tag = WARN_TAG
            self.n_warn += 1
        else:
            tag = FAIL_TAG
            self.n_fail += 1
            self.failures.append(name)
        print(f"  {tag}  {name:<42} {detail}")

    def section(self, title):
        print(f"\n{BOLD}[{title}]{RESET}")

    def summary(self):
        total = self.n_pass + self.n_warn + self.n_fail
        print("\n" + "═" * 70)
        if self.n_fail > 0:
            print(f"{RED}{BOLD}RESULT: {self.n_fail} FAILED, "
                  f"{self.n_warn} WARNED, {self.n_pass} PASSED "
                  f"({total} total){RESET}")
            print("Failing checks:")
            for f in self.failures:
                print(f"  • {f}")
            return 2
        if self.n_warn > 0:
            print(f"{YELLOW}{BOLD}RESULT: PASSED WITH {self.n_warn} "
                  f"WARNINGS ({self.n_pass} passed, {total} total){RESET}")
            return 1
        print(f"{GREEN}{BOLD}RESULT: ALL {total} CHECKS PASSED{RESET}")
        return 0


# ── Check groups ─────────────────────────────────────────────────────────
def check_files(parquet_path, model_dir, r):
    r.section("FILE PRESENCE")
    if not parquet_path.exists():
        r.check("fail", "Embedding parquet", f"NOT FOUND: {parquet_path}")
        return False
    r.check("pass", "Embedding parquet", str(parquet_path))

    if not model_dir.exists():
        r.check("warn", "Model directory", f"missing: {model_dir}")
        return True   # not fatal — can still check the parquet

    r.check("pass", "Model directory", str(model_dir))
    expected = [
        ("bert_pretrain.pt", "Pretrain checkpoint"),
        ("bert.pt",          "Final checkpoint"),
        ("history.json",     "History JSON"),
    ]
    for filename, label in expected:
        path = model_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            r.check("pass", label, f"{filename} ({size_mb:.1f} MB)")
        else:
            r.check("warn", label, f"missing: {filename}")
    if (model_dir / "tokenizer").exists():
        r.check("pass", "Tokenizer", "tokenizer/")
    else:
        r.check("warn", "Tokenizer", "missing tokenizer/")
    return True


def check_shape_and_validity(df, r):
    r.section("SHAPE & VALIDITY")

    dim_cols = sorted(c for c in df.columns if c.startswith("dim_"))
    if not dim_cols:
        r.check("fail", "Dimension columns", "no dim_* columns found")
        return None
    n_investors, n_dims = len(df), len(dim_cols)
    r.check("pass", "Shape", f"{n_investors:,} investors × {n_dims} dims")

    for col in ["investor_id", "quarter_end"]:
        if col in df.columns:
            r.check("pass", f"Has '{col}' column", "")
        else:
            r.check("fail", f"Has '{col}' column", "missing")

    if "investor_type" in df.columns:
        n_types = df["investor_type"].nunique()
        r.check("pass", "Has 'investor_type' column", f"{n_types} unique types")
    else:
        r.check("warn", "Has 'investor_type' column", "missing")

    embeddings = df[dim_cols].values.astype(np.float64)

    n_nan = int(np.isnan(embeddings).sum())
    if n_nan == 0:
        r.check("pass", "No NaN values", "0 found")
    else:
        r.check("fail", "No NaN values", f"{n_nan} NaNs found")

    n_inf = int(np.isinf(embeddings).sum())
    if n_inf == 0:
        r.check("pass", "No Inf values", "0 found")
    else:
        r.check("fail", "No Inf values", f"{n_inf} Infs found")

    norms = np.linalg.norm(embeddings, axis=1)
    n_zero = int((norms < 1e-8).sum())
    if n_zero == 0:
        r.check("pass", "No all-zero embeddings", "0 found")
    elif n_zero < max(1, n_investors // 100):
        r.check("warn", "No all-zero embeddings",
                f"{n_zero} found ({n_zero / n_investors * 100:.1f}%)")
    else:
        r.check("fail", "No all-zero embeddings",
                f"{n_zero} found ({n_zero / n_investors * 100:.1f}%) — model may have failed")

    return embeddings


def check_distribution(embeddings, r):
    r.section("DISTRIBUTION")

    norms = np.linalg.norm(embeddings, axis=1)
    nmean, nstd = norms.mean(), norms.std()
    nmin, nmax = norms.min(), norms.max()

    if nmean < 0.01:
        r.check("fail", "L2 norms not zero",
                f"mean={nmean:.4f} — embeddings collapsed to zero")
    else:
        r.check("pass", "L2 norms",
                f"mean={nmean:.2f}, std={nstd:.2f}, range=[{nmin:.2f}, {nmax:.2f}]")

    cv = nstd / max(nmean, 1e-8)
    if cv < 1.0:
        r.check("pass", "L2 norm consistency", f"CV={cv:.2f}")
    elif cv < 2.0:
        r.check("warn", "L2 norm consistency",
                f"CV={cv:.2f} (high variability across investors)")
    else:
        r.check("fail", "L2 norm consistency",
                f"CV={cv:.2f} (extreme magnitude variation)")

    dim_stds = embeddings.std(axis=0)
    n_dead = int((dim_stds < 0.01).sum())
    if n_dead == 0:
        r.check("pass", "Per-dim variance",
                f"min_std={dim_stds.min():.3f}, max_std={dim_stds.max():.3f}")
    elif n_dead < max(1, len(dim_stds) // 10):
        r.check("warn", "Per-dim variance",
                f"{n_dead}/{len(dim_stds)} dims collapsed (~zero variance)")
    else:
        r.check("fail", "Per-dim variance",
                f"{n_dead}/{len(dim_stds)} dims collapsed — model may not have learned")


def check_similarity_structure(embeddings, r, n_pairs=5000):
    r.section("SIMILARITY STRUCTURE")

    n = len(embeddings)
    if n < 2:
        r.check("warn", "Sample size", "need ≥2 investors for similarity")
        return

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    normalized = embeddings / norms

    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    collisions = idx_a == idx_b
    while collisions.any():
        idx_b[collisions] = rng.integers(0, n, size=int(collisions.sum()))
        collisions = idx_a == idx_b

    cos_sims = (normalized[idx_a] * normalized[idx_b]).sum(axis=1)
    cmin, cmax = cos_sims.min(), cos_sims.max()
    cstd = cos_sims.std()
    mean_abs = np.abs(cos_sims).mean()

    r.check("pass", "Cosine sim range",
            f"[{cmin:.3f}, {cmax:.3f}] ({n_pairs:,} random pairs)")

    if mean_abs > 0.95:
        r.check("fail", "Not collapsed",
                f"mean |cos|={mean_abs:.3f} — embeddings collapsed to a line")
    elif mean_abs > 0.7:
        r.check("warn", "Not collapsed",
                f"mean |cos|={mean_abs:.3f} (suspiciously high)")
    else:
        r.check("pass", "Not collapsed", f"mean |cos|={mean_abs:.3f}")

    if cstd < 0.02:
        r.check("fail", "Has structure",
                f"std of cos={cstd:.3f} — embeddings nearly uniform")
    elif cstd < 0.05:
        r.check("warn", "Has structure",
                f"std of cos={cstd:.3f} (weak)")
    else:
        r.check("pass", "Has structure", f"std of cos={cstd:.3f}")


def check_training_health(history_path, r):
    r.section("TRAINING HEALTH")
    if not history_path.exists():
        r.check("warn", "history.json", "not found, skipping training-health checks")
        return
    try:
        history = json.loads(history_path.read_text())
    except Exception as exc:
        r.check("warn", "history.json", f"could not parse: {exc}")
        return

    pretrain_log = history.get("pretrain_log", [])
    train_losses = [e["loss"] for e in pretrain_log
                    if "loss" in e and "eval_loss" not in e]
    val_losses = [e["eval_loss"] for e in pretrain_log if "eval_loss" in e]

    if len(train_losses) >= 2:
        first, last = train_losses[0], train_losses[-1]
        if last < first * 0.95:
            r.check("pass", "Pretrain train loss decreased",
                    f"{first:.3f} → {last:.3f}")
        elif last < first:
            r.check("warn", "Pretrain train loss decreased",
                    f"{first:.3f} → {last:.3f} (small drop)")
        else:
            r.check("fail", "Pretrain train loss decreased",
                    f"{first:.3f} → {last:.3f} (didn't decrease)")
    else:
        r.check("warn", "Pretrain train loss", "fewer than 2 epochs logged")

    if len(val_losses) >= 2:
        first, last = val_losses[0], val_losses[-1]
        if last < first * 1.1:
            r.check("pass", "Pretrain val loss stable/decreased",
                    f"{first:.3f} → {last:.3f}")
        else:
            r.check("warn", "Pretrain val loss stable/decreased",
                    f"{first:.3f} → {last:.3f} (grew >10%, possible overfit)")

    ft_losses = history.get("finetune", {}).get("loss", [])
    if len(ft_losses) >= 2:
        first, last = ft_losses[0], ft_losses[-1]
        if last < first * 0.9:
            r.check("pass", "Finetune loss decreased",
                    f"{first:.3f} → {last:.3f}")
        elif last < first:
            r.check("warn", "Finetune loss decreased",
                    f"{first:.3f} → {last:.3f} (small drop)")
        else:
            r.check("warn", "Finetune loss decreased",
                    f"{first:.3f} → {last:.3f} (didn't decrease)")
    elif len(ft_losses) == 0:
        r.check("warn", "Finetune loss",
                "no finetune epochs logged (may have been skipped)")
    else:
        r.check("warn", "Finetune loss", "only one epoch logged")


def check_investor_type_structure(df, embeddings, r, n_pairs=4000):
    r.section("INVESTOR TYPE STRUCTURE")
    if "investor_type" not in df.columns:
        r.check("warn", "Investor type analysis", "no investor_type column")
        return

    types = df["investor_type"].values
    type_counts = pd.Series(types).value_counts()
    if len(type_counts) < 2 or (type_counts > 1).sum() < 2:
        r.check("warn", "Investor type analysis",
                "need ≥2 types each with ≥2 investors")
        return

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    normalized = embeddings / norms

    rng = np.random.default_rng(43)
    intra_sims, inter_sims = [], []
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(df), size=2)
        if i == j:
            continue
        cos = float(normalized[i] @ normalized[j])
        if types[i] == types[j]:
            intra_sims.append(cos)
        else:
            inter_sims.append(cos)

    if len(intra_sims) < 10 or len(inter_sims) < 10:
        r.check("warn", "Investor type analysis",
                "too few sampled pairs in groups")
        return

    intra_mean = float(np.mean(intra_sims))
    inter_mean = float(np.mean(inter_sims))
    diff = intra_mean - inter_mean

    if diff > 0.05:
        r.check("pass", "Intra-type sim > inter-type",
                f"intra={intra_mean:.3f}, inter={inter_mean:.3f}, Δ={diff:+.3f}")
    elif diff > 0.0:
        r.check("warn", "Intra-type sim > inter-type",
                f"intra={intra_mean:.3f}, inter={inter_mean:.3f}, Δ={diff:+.3f} (weak)")
    else:
        r.check("warn", "Intra-type sim > inter-type",
                f"intra={intra_mean:.3f}, inter={inter_mean:.3f}, Δ={diff:+.3f} "
                "(no type structure — concerning if expected)")


# ── Orchestration ────────────────────────────────────────────────────────
def infer_model_dir(parquet_path):
    """embeddings/q_2005-03-31.parquet -> models/q_2005-03-31/"""
    parent_str = str(parquet_path.parent)
    candidate = Path(parent_str.replace("embeddings", "models")) / parquet_path.stem
    return candidate


def verify_one(parquet_path, model_dir=None):
    parquet_path = Path(parquet_path)
    model_dir = Path(model_dir) if model_dir else infer_model_dir(parquet_path)

    print("═" * 70)
    print(f"{BOLD}Verifying: {parquet_path}{RESET}")
    print("═" * 70)

    r = Reporter()

    if not check_files(parquet_path, model_dir, r):
        return r.summary()

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        r.check("fail", "Parquet readable", f"could not parse: {exc}")
        return r.summary()

    embeddings = check_shape_and_validity(df, r)
    if embeddings is None:
        return r.summary()

    check_distribution(embeddings, r)
    check_similarity_structure(embeddings, r)

    if model_dir.exists():
        check_training_health(model_dir / "history.json", r)

    check_investor_type_structure(df, embeddings, r)

    return r.summary()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", nargs="?", default="embeddings",
        help="Single q_*.parquet file or a directory (default: embeddings/)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="If path is a directory, verify EVERY q_*.parquet in it",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Override the model directory (otherwise auto-inferred)",
    )
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        return verify_one(path, args.model_dir)

    if path.is_dir():
        files = sorted(path.glob("q_*.parquet"))
        if not files:
            print(f"No q_*.parquet files in {path}", file=sys.stderr)
            return 2
        if args.all:
            worst = 0
            for f in files:
                code = verify_one(f, args.model_dir)
                worst = max(worst, code)
                print()
            return worst
        return verify_one(files[0], args.model_dir)

    print(f"Path not found: {path}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())