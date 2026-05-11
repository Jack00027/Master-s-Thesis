#!/usr/bin/env python3
"""
Produce two UMAP visualizations of investor embeddings for a single quarter:

  1. embedding_2d_zorder.png   - single panel, rare types drawn on top of OEF
  2. embedding_2d_faceted.png  - small multiples, one panel per type highlighted

Both plots use the SAME UMAP projection (computed once) so they're directly
comparable. Saves to the same output directory as the input parquet by default.

Usage:
  python plot_embeddings.py embeddings/test/q_2019-09-30.parquet
  python plot_embeddings.py embeddings/test/q_2019-09-30.parquet --out report/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_embeddings(parquet_path):
    df = pd.read_parquet(parquet_path)
    dim_cols = sorted(c for c in df.columns if c.startswith("dim_"))
    embeddings = df[dim_cols].values.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    normalized = embeddings / norms
    return df, normalized


def compute_projection(normalized, random_state=42):
    """Project to 2D. UMAP if available, otherwise PCA. Returns (coords, name)."""
    try:
        import umap
        print("  Running UMAP (~30s for ~10k investors)...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(normalized), "UMAP"
    except ImportError:
        print("  umap-learn not installed, falling back to PCA...")
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=random_state).fit_transform(normalized)
        return coords, "PCA"


def _type_color_map(types):
    """Stable color mapping: largest type gets color 0, then in descending order.
    Returns (color_map_dict, ordered_types_largest_first)."""
    order_desc = (
        pd.Series(types).value_counts().sort_values(ascending=False).index.tolist()
    )
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(order_desc)}, order_desc


def plot_zorder(df, coords, method, out_path):
    """Single panel. Largest type drawn at bottom; rarest on top."""
    fig, ax = plt.subplots(figsize=(10, 8))
    types = df["investor_type"].values
    color_map, order_desc = _type_color_map(types)

    # Iterate largest -> rarest. zorder ascends so rare types end up on top.
    for i, t in enumerate(order_desc):
        mask = types == t
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=4, alpha=0.5, color=color_map[t],
            label=f"{t} (n={mask.sum():,})",
            zorder=10 + i,
        )

    ax.legend(loc="best", fontsize=9, markerscale=3, framealpha=0.9)
    quarter = df["quarter_end"].iloc[0] if "quarter_end" in df.columns else ""
    ax.set_title(f"{method} projection of investor embeddings — {quarter}")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  → {out_path}")


def plot_faceted(df, coords, method, out_path):
    """Small multiples: each type highlighted against gray, plus one combined panel."""
    types = df["investor_type"].values
    color_map, order_desc = _type_color_map(types)

    # Show rarest types first (most informative) for the per-type panels.
    panel_order = order_desc[::-1]
    n_types = len(panel_order)
    n_panels = n_types + 1                    # + one combined overlay panel
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.6 * n_rows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    # Per-type highlight panels
    for j, t in enumerate(panel_order):
        ax = axes[j]
        mask = types == t
        ax.scatter(coords[~mask, 0], coords[~mask, 1],
                   s=2, alpha=0.12, color="lightgray", zorder=1)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.75, color=color_map[t], zorder=2)
        pct = 100 * mask.sum() / len(df)
        ax.set_title(f"{t}  (n={mask.sum():,}, {pct:.1f}%)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    # Combined overlay panel (with correct z-order: rare on top)
    ax = axes[n_types]
    for i, t in enumerate(order_desc):
        mask = types == t
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=3, alpha=0.5, color=color_map[t], label=t, zorder=10 + i)
    ax.set_title("All types overlaid", fontsize=10)
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    # Hide any unused panels
    for k in range(n_types + 1, len(axes)):
        axes[k].set_visible(False)

    quarter = df["quarter_end"].iloc[0] if "quarter_end" in df.columns else ""
    fig.suptitle(f"{method} projection by investor type — {quarter}",
                 fontsize=13, y=1.00)
    fig.supxlabel(f"{method}-1")
    fig.supylabel(f"{method}-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", help="Path to a quarter's embedding parquet file")
    parser.add_argument("--out", default="report",
                        help="Output directory (default: report/)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="UMAP random seed (default: 42)")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {parquet_path}...")
    df, normalized = load_embeddings(parquet_path)
    print(f"  {len(df):,} investors, {normalized.shape[1]} dimensions")
    if "investor_type" not in df.columns:
        print("ERROR: no investor_type column in parquet — can't color by type.",
              file=sys.stderr)
        return 1
    print()

    print("Computing 2D projection (used by both plots)...")
    coords, method = compute_projection(normalized, random_state=args.random_state)
    print()

    print("[1/2] Single panel (z-order fix)")
    plot_zorder(df, coords, method, out_dir / "embedding_2d_zorder.png")
    print()

    print("[2/2] Faceted small-multiples")
    plot_faceted(df, coords, method, out_dir / "embedding_2d_faceted.png")

    print()
    print(f"Done. Plots in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
