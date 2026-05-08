"""
PS-BERT training on quarterly portfolio sequences.

For each per-quarter parquet file produced by build_portfolio_sequences.R,
this script:
  1. Builds a per-quarter vocabulary of issuer tokens.
  2. Pre-trains a small BERT (4 layers, 2 heads, ctx=62) with masked-token
     prediction (15% masking, 80/10/10 split).
  3. Fine-tunes with a sentence-transformer step on even/odd portfolio splits
     so cosine similarity is a meaningful distance.
  4. Computes the investor embedding as the average contextualized embedding
     over the top-62 positions, and writes one row per investor.

Outputs:
  embeddings/q_YYYY-MM-DD.parquet   # (investor_id, investor_type, dim_0..dim_d)
  models/q_YYYY-MM-DD/              # tokenizer + model checkpoints

Usage:
  python train_ps_bert.py --seq-dir data --emb-dir embeddings --model-dir models
  python train_ps_bert.py --test                     # run on data/test only
  python train_ps_bert.py --hidden-size 32           # 32-dim embeddings
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertForMaskedLM


# ── Config ────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # I/O
    seq_dir:   str = "data"
    emb_dir:   str = "embeddings"
    model_dir: str = "models"

    # Architecture (per the paper)
    hidden_size:       int = 64    # embedding dimension d
    num_layers:        int = 4
    num_heads:         int = 2
    intermediate_size: int = 256   # FFN width, conventional 4 * hidden
    context_window:    int = 62    # tokens per portfolio
    max_position:      int = 64    # 62 + [CLS] + [SEP]

    # MLM
    mask_prob:        float = 0.15
    mask_token_prob:  float = 0.80   # within masked: 80% [MASK]
    random_token_prob: float = 0.10  # 10% random token
    keep_token_prob:  float = 0.10   # 10% kept as-is

    # Pre-training
    batch_size:    int = 64
    pretrain_lr:  float = 5e-4
    pretrain_epochs: int = 10
    weight_decay: float = 0.01
    warmup_frac:  float = 0.1
    train_split:  float = 0.9

    # Sentence-transformer fine-tuning
    finetune_lr:    float = 2e-4
    finetune_epochs: int = 3

    # Misc
    seed:    int = 42
    skip_existing: bool = True
    device:  str = field(default_factory=lambda:
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"))


# ── Vocabulary ───────────────────────────────────────────────────────────
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]


class Vocab:
    def __init__(self, tokens: List[str]):
        unique = sorted(set(tokens))
        self.itos = SPECIAL_TOKENS + unique
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def size(self):     return len(self.itos)
    @property
    def pad_id(self):   return self.stoi["[PAD]"]
    @property
    def cls_id(self):   return self.stoi["[CLS]"]
    @property
    def sep_id(self):   return self.stoi["[SEP]"]
    @property
    def mask_id(self):  return self.stoi["[MASK]"]
    @property
    def special_ids(self): return {self.pad_id, self.cls_id,
                                   self.sep_id, self.mask_id}

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def save(self, path: Path):
        path.write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        v = cls.__new__(cls)
        v.itos = json.loads(path.read_text())
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


# ── Datasets ─────────────────────────────────────────────────────────────
class MLMDataset(Dataset):
    """Yields (input_ids, attention_mask, labels) for masked-LM training."""

    def __init__(self, sequences: List[List[int]], vocab: Vocab, cfg: Config):
        self.sequences = sequences
        self.vocab = vocab
        self.cfg = cfg
        self.max_len = cfg.context_window + 2  # [CLS] + 62 + [SEP]
        # Token IDs eligible for the "10% random" replacement
        self.replaceable_ids = [
            i for i in range(vocab.size) if i not in vocab.special_ids
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        body = self.sequences[idx][: self.cfg.context_window]
        ids = [self.vocab.cls_id] + body + [self.vocab.sep_id]
        attn = [1] * len(ids)
        labels = [-100] * len(ids)  # -100 = ignore in CE loss

        # Mask 15% of body tokens
        for pos in range(1, len(ids) - 1):
            if random.random() >= self.cfg.mask_prob:
                continue
            labels[pos] = ids[pos]  # remember original
            r = random.random()
            if r < self.cfg.mask_token_prob:
                ids[pos] = self.vocab.mask_id
            elif r < self.cfg.mask_token_prob + self.cfg.random_token_prob:
                ids[pos] = random.choice(self.replaceable_ids)
            # else: keep as-is (10%)

        # Pad to max_len
        pad_len = self.max_len - len(ids)
        ids    += [self.vocab.pad_id] * pad_len
        attn   += [0] * pad_len
        labels += [-100] * pad_len

        return (torch.tensor(ids,    dtype=torch.long),
                torch.tensor(attn,   dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


class PairDataset(Dataset):
    """For sentence-transformer fine-tuning: even/odd portfolio splits.

    Each item returns two halves of one investor's portfolio (positive pair).
    Negatives are drawn implicitly via in-batch contrastive loss.
    """

    def __init__(self, sequences: List[List[int]], vocab: Vocab, cfg: Config):
        # Only investors with >= 2 tokens can be split
        self.sequences = [s for s in sequences if len(s) >= 2]
        self.vocab = vocab
        self.cfg = cfg
        self.max_len = cfg.context_window + 2

    def __len__(self):
        return len(self.sequences)

    def _build(self, body):
        body = body[: self.cfg.context_window]
        ids = [self.vocab.cls_id] + body + [self.vocab.sep_id]
        attn = [1] * len(ids)
        pad_len = self.max_len - len(ids)
        ids  += [self.vocab.pad_id] * pad_len
        attn += [0] * pad_len
        return (torch.tensor(ids,  dtype=torch.long),
                torch.tensor(attn, dtype=torch.long))

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        evens = seq[0::2]  # rank 1, 3, 5, ...
        odds  = seq[1::2]  # rank 2, 4, 6, ...
        return self._build(evens), self._build(odds)


# ── Models ───────────────────────────────────────────────────────────────
def build_bert_config(vocab_size: int, cfg: Config) -> BertConfig:
    return BertConfig(
        vocab_size            = vocab_size,
        hidden_size           = cfg.hidden_size,
        num_hidden_layers     = cfg.num_layers,
        num_attention_heads   = cfg.num_heads,
        intermediate_size     = cfg.intermediate_size,
        max_position_embeddings = cfg.max_position,
        type_vocab_size       = 1,         # single segment
        pad_token_id          = 0,
        hidden_act            = "gelu",
    )


# ── Training loops ───────────────────────────────────────────────────────
def cosine_lr(step: int, total: int, base_lr: float, warmup: int):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def pretrain_mlm(model: BertForMaskedLM,
                 train_ds: MLMDataset,
                 val_ds: MLMDataset,
                 cfg: Config) -> dict:
    device = cfg.device
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr,
                              weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.pretrain_epochs)
    warmup      = int(total_steps * cfg.warmup_frac)

    history = {"train_loss": [], "val_loss": []}
    step = 0
    for epoch in range(cfg.pretrain_epochs):
        model.train()
        running = 0.0
        for ids, attn, labels in train_loader:
            ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
            for g in optim.param_groups:
                g["lr"] = cosine_lr(step, total_steps, cfg.pretrain_lr, warmup)
            optim.zero_grad()
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += out.loss.item() * ids.size(0)
            step += 1
        history["train_loss"].append(running / len(train_ds))

        # Validation
        model.eval()
        running = 0.0
        with torch.no_grad():
            for ids, attn, labels in val_loader:
                ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                running += out.loss.item() * ids.size(0)
        history["val_loss"].append(running / max(1, len(val_ds)))

        print(f"    pretrain ep{epoch+1}/{cfg.pretrain_epochs}: "
              f"train={history['train_loss'][-1]:.4f} "
              f"val={history['val_loss'][-1]:.4f}")

    return history


def finetune_sentence_transformer(model: BertModel,
                                  pair_ds: PairDataset,
                                  cfg: Config) -> dict:
    """Siamese contrastive fine-tuning with in-batch negatives.

    For each batch of pairs (a, b), compute investor embeddings
    (mean-pooled over non-pad tokens), then maximize the cosine sim
    between matched pairs while minimizing it for in-batch negatives
    using a cross-entropy loss over the similarity matrix.
    """
    device = cfg.device
    model.to(device)
    loader = DataLoader(pair_ds, batch_size=cfg.batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr,
                              weight_decay=cfg.weight_decay)
    total_steps = max(1, len(loader) * cfg.finetune_epochs)
    warmup      = int(total_steps * cfg.warmup_frac)

    def encode(ids, attn):
        out = model(input_ids=ids, attention_mask=attn)
        h = out.last_hidden_state                    # (B, L, d)
        mask = attn.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, dim=-1)

    history = {"loss": []}
    step = 0
    for epoch in range(cfg.finetune_epochs):
        model.train()
        running = 0.0
        n = 0
        for (ids_a, attn_a), (ids_b, attn_b) in loader:
            ids_a, attn_a = ids_a.to(device), attn_a.to(device)
            ids_b, attn_b = ids_b.to(device), attn_b.to(device)
            for g in optim.param_groups:
                g["lr"] = cosine_lr(step, total_steps, cfg.finetune_lr, warmup)
            optim.zero_grad()

            ea = encode(ids_a, attn_a)               # (B, d)
            eb = encode(ids_b, attn_b)               # (B, d)
            logits = ea @ eb.t() * 20.0              # temperature 1/20
            target = torch.arange(ea.size(0), device=device)
            loss = (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.t(), target)) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * ea.size(0)
            n += ea.size(0)
            step += 1
        history["loss"].append(running / max(1, n))
        print(f"    finetune ep{epoch+1}/{cfg.finetune_epochs}: "
              f"loss={history['loss'][-1]:.4f}")

    return history


# ── Embedding extraction ─────────────────────────────────────────────────
@torch.no_grad()
def compute_investor_embeddings(model: BertModel,
                                investors: pd.DataFrame,
                                vocab: Vocab,
                                cfg: Config) -> np.ndarray:
    """For each investor's top-62 sequence, return mean-pooled contextualized
    embedding.  investors must have one row per investor with tokens already
    truncated to <= 62 (the `tokens` column from your parquet)."""
    device = cfg.device
    model.eval()
    max_len = cfg.context_window + 2

    out = np.zeros((len(investors), cfg.hidden_size), dtype=np.float32)
    for start in range(0, len(investors), cfg.batch_size):
        end = min(start + cfg.batch_size, len(investors))
        batch_ids, batch_attn = [], []
        for j in range(start, end):
            body = vocab.encode(list(investors.iloc[j]["tokens"]))[:cfg.context_window]
            ids  = [vocab.cls_id] + body + [vocab.sep_id]
            attn = [1] * len(ids)
            pad  = max_len - len(ids)
            ids  += [vocab.pad_id] * pad
            attn += [0] * pad
            batch_ids.append(ids); batch_attn.append(attn)
        ids  = torch.tensor(batch_ids,  dtype=torch.long, device=device)
        attn = torch.tensor(batch_attn, dtype=torch.long, device=device)
        h = model(input_ids=ids, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out[start:end] = pooled.cpu().numpy()
    return out


# ── Per-quarter pipeline ─────────────────────────────────────────────────
def process_quarter(parquet_path: Path, cfg: Config):
    q_label = parquet_path.stem.replace("q_", "")  # "2019-09-30"
    emb_path  = Path(cfg.emb_dir)   / f"q_{q_label}.parquet"
    model_dir = Path(cfg.model_dir) / f"q_{q_label}"

    if cfg.skip_existing and emb_path.exists():
        print(f"  [skip] q_{q_label} (embedding exists)")
        return

    print(f"\n── q_{q_label} ──")

    # 1. Load
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        print("  (empty quarter, skipping)")
        return
    # Each row already has tokens (list[str]) and n_tokens
    df["tokens"] = df["tokens"].apply(list)

    # Each investor may have multiple chunks (if portfolio > 62 assets).
    # For pre-training, use ALL chunks. For final embedding, use the FIRST
    # chunk per investor (the top-62 positions).
    sequences_all = df["tokens"].tolist()
    first_chunks  = (df.sort_values(["investor_id", "n_assets_full"],
                                    ascending=[True, False])
                       .drop_duplicates("investor_id", keep="first")
                       .reset_index(drop=True))

    print(f"  {len(df):,} sequences from {first_chunks.shape[0]:,} investors")

    # 2. Vocab
    flat = [t for seq in sequences_all for t in seq]
    vocab = Vocab(flat)
    print(f"  vocab: {vocab.size:,} tokens (incl. {len(SPECIAL_TOKENS)} special)")

    # 3. Encode all sequences
    encoded = [vocab.encode(s) for s in sequences_all]

    # 4. Train/val split (by investor for cleaner held-out)
    rng = random.Random(cfg.seed)
    inv_ids = df["investor_id"].unique().tolist()
    rng.shuffle(inv_ids)
    n_train = int(len(inv_ids) * cfg.train_split)
    train_inv = set(inv_ids[:n_train])
    train_idx = [i for i, inv in enumerate(df["investor_id"]) if inv in train_inv]
    val_idx   = [i for i, inv in enumerate(df["investor_id"]) if inv not in train_inv]
    train_seqs = [encoded[i] for i in train_idx]
    val_seqs   = [encoded[i] for i in val_idx]

    train_ds = MLMDataset(train_seqs, vocab, cfg)
    val_ds   = MLMDataset(val_seqs,   vocab, cfg)

    # 5. Pre-train BERT (MLM)
    config = build_bert_config(vocab.size, cfg)
    mlm = BertForMaskedLM(config)
    print(f"  model: {sum(p.numel() for p in mlm.parameters()):,} params")

    pre_hist = pretrain_mlm(mlm, train_ds, val_ds, cfg)
    print(f"  pretrain final: train={pre_hist['train_loss'][-1]:.4f} "
          f"val={pre_hist['val_loss'][-1]:.4f}")

    # 6. Sentence-transformer fine-tuning
    bert = mlm.bert  # shed the MLM head
    pair_ds = PairDataset(train_seqs, vocab, cfg)
    if len(pair_ds) >= cfg.batch_size:
        ft_hist = finetune_sentence_transformer(bert, pair_ds, cfg)
        print(f"  finetune final loss: {ft_hist['loss'][-1]:.4f}")
    else:
        ft_hist = {"loss": []}
        print("  (skipping fine-tuning: too few pairs for one batch)")

    # 7. Investor embeddings (top-62 positions per investor)
    embs = compute_investor_embeddings(bert, first_chunks, vocab, cfg)

    # 8. Save
    Path(cfg.emb_dir).mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "investor_id":   first_chunks["investor_id"].values,
        "investor_type": first_chunks["investor_type"].values,
        "quarter_end":   first_chunks["quarter_end"].values,
    })
    for d in range(cfg.hidden_size):
        out_df[f"dim_{d:03d}"] = embs[:, d]
    out_df.to_parquet(emb_path, index=False)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bert.state_dict(), model_dir / "bert.pt")
    vocab.save(model_dir / "vocab.json")
    (model_dir / "history.json").write_text(json.dumps(
        {"pretrain": pre_hist, "finetune": ft_hist}, indent=2))

    print(f"  saved → {emb_path}")

    # 9. Free memory
    del mlm, bert, train_ds, val_ds, pair_ds
    if cfg.device == "cuda":
        torch.cuda.empty_cache()


# ── Entry point ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir",   default="data")
    p.add_argument("--emb-dir",   default="embeddings")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--hidden-size",     type=int, default=64)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--batch-size",      type=int, default=64)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--no-skip", action="store_true",
                   help="re-train quarters even if embedding exists")
    p.add_argument("--test", action="store_true",
                   help="use data/test as input dir")
    args = p.parse_args()

    cfg = Config()
    cfg.seq_dir         = "data/test" if args.test else args.seq_dir
    cfg.emb_dir         = "embeddings/test" if args.test else args.emb_dir
    cfg.model_dir       = "models/test"     if args.test else args.model_dir
    cfg.hidden_size     = args.hidden_size
    cfg.pretrain_epochs = args.pretrain_epochs
    cfg.finetune_epochs = args.finetune_epochs
    cfg.batch_size      = args.batch_size
    cfg.seed            = args.seed
    cfg.skip_existing   = not args.no_skip

    # Reproducibility
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    print(f"device         : {cfg.device}")
    print(f"hidden_size    : {cfg.hidden_size}")
    print(f"pretrain epochs: {cfg.pretrain_epochs}")
    print(f"finetune epochs: {cfg.finetune_epochs}")
    print(f"batch_size     : {cfg.batch_size}")
    print(f"seq_dir        : {cfg.seq_dir}")

    files = sorted(Path(cfg.seq_dir).glob("q_*.parquet"))
    if not files:
        print(f"No q_*.parquet files found in {cfg.seq_dir}")
        return
    print(f"\n{len(files)} quarters to process\n")

    for path in files:
        process_quarter(path, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()