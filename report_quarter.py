#!/usr/bin/env python3
"""
Generate a summary report of one quarter's embeddings.

Produces four visual/textual artifacts in a chosen output directory:
  1. type_distribution.png       - bar chart of investor type counts
  2. embedding_2d.png            - 2D UMAP/PCA projection, colored by type
  3. loss_curves.png             - pretrain train+val loss + finetune loss
  4. nearest_neighbors.txt       - top-5 most similar investors for a few seeds

Usage:
  python report_quarter.py embeddings/q_2005-03-31.parquet
  python report_quarter.py embeddings/q_2005-03-31.parquet --out report/
  python report_quarter.py embeddings/q_2005-03-31.parquet --seed-investor "Vanguard"

Notes:
  - Requires matplotlib and either umap-learn or sklearn (for PCA fallback)
  - Auto-finds models/<quarter>/history.json for loss curves
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Utilities ────────────────────────────────────────────────────────────
def load_embeddings(parquet_path):
    df = pd.read_parquet(parquet_path)
    dim_cols = sorted(c for c in df.columns if c.startswith("dim_"))
    embeddings = df[dim_cols].values.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    normalized = embeddings / norms
    return df, embeddings, normalized


def infer_model_dir(parquet_path):
    parent = str(parquet_path.parent).replace("embeddings", "models")
    return Path(parent) / parquet_path.stem


# ── 1. Type distribution ─────────────────────────────────────────────────
def plot_type_distribution(df, out_path):
    if "investor_type" not in df.columns:
        print("  (no investor_type column, skipping type distribution)")
        return
    counts = df["investor_type"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(counts))))
    bars = ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Number of investors")
    ax.set_title(f"Investor type composition ({len(df):,} total)")
    for bar, n in zip(bars, counts.values):
        ax.text(n, bar.get_y() + bar.get_height() / 2,
                f" {n:,}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


# ── 2. 2D projection ─────────────────────────────────────────────────────
def plot_embedding_2d(df, normalized, out_path):
    # Try UMAP first (better at preserving local structure), fall back to PCA
    try:
        import umap
        print("  Running UMAP (this takes ~30s for ~10k investors)...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            metric="cosine", random_state=42)
        coords = reducer.fit_transform(normalized)
        method = "UMAP"
    except ImportError:
        print("  umap-learn not installed, using PCA instead...")
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(normalized)
        method = "PCA"

    fig, ax = plt.subplots(figsize=(10, 8))
    if "investor_type" in df.columns:
        types = df["investor_type"].values
        unique = sorted(pd.Series(types).unique())
        cmap = plt.get_cmap("tab10")
        for i, t in enumerate(unique):
            mask = types == t
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=4, alpha=0.5, color=cmap(i % 10), label=f"{t} (n={mask.sum():,})")
        ax.legend(loc="best", fontsize=9, markerscale=3, framealpha=0.9)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.5, color="#4C72B0")

    quarter = df["quarter_end"].iloc[0] if "quarter_end" in df.columns else ""
    ax.set_title(f"{method} projection of investor embeddings — {quarter}")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


# ── 3. Loss curves ───────────────────────────────────────────────────────
def plot_loss_curves(history_path, out_path):
    if not history_path.exists():
        print(f"  (no history.json at {history_path}, skipping loss curves)")
        return
    history = json.loads(history_path.read_text())

    pretrain_log = history.get("pretrain_log", [])
    train_losses = [(e["epoch"], e["loss"]) for e in pretrain_log
                    if "loss" in e and "eval_loss" not in e]
    val_losses = [(e["epoch"], e["eval_loss"]) for e in pretrain_log
                  if "eval_loss" in e]
    ft_losses = history.get("finetune", {}).get("loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Pretrain
    if train_losses:
        x, y = zip(*train_losses)
        axes[0].plot(x, y, "o-", label="train", color="#4C72B0")
    if val_losses:
        x, y = zip(*val_losses)
        axes[0].plot(x, y, "s-", label="val", color="#DD8452")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MLM loss (cross-entropy)")
    axes[0].set_title("Phase 1: Masked-language pretraining")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Finetune
    if ft_losses:
        axes[1].plot(range(1, len(ft_losses) + 1), ft_losses,
                     "o-", color="#55A868")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Contrastive loss (InfoNCE)")
    axes[1].set_title("Phase 2: Sentence-transformer fine-tuning")
    axes[1].grid(alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


# ── 4. Nearest neighbors ─────────────────────────────────────────────────
def find_nearest_neighbors(df, normalized, seed_substrings, k=10, out_path=None):
    """For each seed substring, find the closest k investors by cosine similarity."""
    lines = []
    lines.append("Nearest-neighbor examples")
    lines.append("=" * 72)
    quarter = df["quarter_end"].iloc[0] if "quarter_end" in df.columns else "unknown"
    lines.append(f"Quarter: {quarter}")
    lines.append(f"Total investors: {len(df):,}")
    lines.append("")
    lines.append("For each seed investor, the top-{} most similar investors by cosine".format(k))
    lines.append("similarity in the embedding space. The model has never seen any")
    lines.append("investor names — only their portfolio compositions.")
    lines.append("")

    if "investor_id" not in df.columns:
        lines.append("(no investor_id column, cannot show neighbors)")
        output = "\n".join(lines)
        if out_path:
            out_path.write_text(output)
            print(f"  → {out_path}")
        return output

    investor_ids = df["investor_id"].astype(str).values
    investor_types = (df["investor_type"].astype(str).values
                      if "investor_type" in df.columns else None)

    found_any = False
    for seed in seed_substrings:
        # Find investors whose id contains the seed substring (case-insensitive)
        matches = [i for i, name in enumerate(investor_ids)
                   if seed.lower() in name.lower()]
        if not matches:
            lines.append(f"── Seed '{seed}': no match in this quarter ──")
            lines.append("")
            continue
        found_any = True
        # Use the first match as the seed
        seed_idx = matches[0]
        seed_name = investor_ids[seed_idx]
        seed_type = investor_types[seed_idx] if investor_types is not None else ""

        # Compute cosine sim to all others
        sims = normalized @ normalized[seed_idx]
        sims[seed_idx] = -np.inf   # exclude self
        top_k = np.argsort(sims)[::-1][:k]

        lines.append(f"── Seed: {seed_name} [{seed_type}] ──")
        for rank, j in enumerate(top_k, 1):
            t = f"[{investor_types[j]}]" if investor_types is not None else ""
            lines.append(f"  {rank:2d}. cos={sims[j]:+.3f}  {investor_ids[j]}  {t}")
        lines.append("")

    if not found_any:
        lines.append("None of the seed substrings matched any investor in this quarter.")
        lines.append("Try different seeds with --seed-investor.")

    output = "\n".join(lines)
    if out_path:
        out_path.write_text(output)
        print(f"  → {out_path}")
    return output


# ── Main ─────────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [
    "Vanguard", "Fidelity", "BlackRock", "State Street",
    "Berkshire", "Bridgewater", "Renaissance",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", help="Path to embeddings parquet file")
    parser.add_argument("--out", default="report",
                        help="Output directory (default: report/)")
    parser.add_argument("--model-dir", default=None,
                        help="Model directory (otherwise auto-inferred)")
    parser.add_argument("--seed-investor", action="append", default=None,
                        help="Add a seed investor name (repeatable). "
                             "Default: a list of well-known fund families.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many neighbors to list per seed (default: 10)")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir) if args.model_dir else infer_model_dir(parquet_path)
    seeds = args.seed_investor if args.seed_investor else DEFAULT_SEEDS

    print(f"Loading {parquet_path}...")
    df, embeddings, normalized = load_embeddings(parquet_path)
    print(f"  {len(df):,} investors, {embeddings.shape[1]} dimensions")
    print(f"  Output dir: {out_dir}/")
    print()

    print("[1/4] Type distribution")
    plot_type_distribution(df, out_dir / "type_distribution.png")
    print()

    print("[2/4] 2D projection")
    plot_embedding_2d(df, normalized, out_dir / "embedding_2d.png")
    print()

    print("[3/4] Loss curves")
    plot_loss_curves(model_dir / "history.json", out_dir / "loss_curves.png")
    print()

    print("[4/4] Nearest neighbors")
    find_nearest_neighbors(df, normalized, seeds,
                           k=args.top_k,
                           out_path=out_dir / "nearest_neighbors.txt")

    print()
    print(f"Report ready in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
