"""
PS-BERT training with WEIGHTED pooling and FULL-PORTFOLIO coverage.

Variant of BERT_training.py with two changes, both switchable so the
original behaviour stays reproducible as a baseline:

  1. POOLING (--pooling)
       mean      : unweighted mean over tokens          (original)
       weighted  : weighted average over tokens         (NEW)

     Where do the weights come from?
       - If the sequence parquet has a per-token weight column
         (--weight-col, default "weights"), those weights are used
         directly. This is the correct option and requires the R
         tokenizer to carry adj_mv/weight through to the parquet.
       - Otherwise weights are DERIVED FROM RANK via a decay function
         (--weight-scheme). Tokens are stored sorted by descending
         position size, so rank is a monotone proxy for weight -- but
         only a proxy. Say so in the thesis.

  2. COVERAGE (--coverage)
       paper         : embed the LARGEST context_window positions, joined
                       across chunks. This is what Gabaix et al. specify:
                       "the average contextualized embedding for a given
                       investor for the largest 62 positions".  (DEFAULT)
       first-chunk   : embed chunk 1 only. LEGACY -- see the note below;
                       chunk 1 is NOT the largest 62 positions.
       all-chunks    : embed EVERY chunk of a portfolio and aggregate
                       them into one investor vector, weighted by the
                       chunk's share of total portfolio weight
       full-sequence : re-join the chunks into the original ranked
                       portfolio and train/embed with a LONGER context
                       window (--context-window), so attention actually
                       spans the whole book

WHY "paper" AND "first-chunk" DIFFER
    Data_Cleaning.r chunks a portfolio into k = ceil(n / 62) chunks of
    ceil(n / k) positions each -- EQUAL-size chunks, matching the paper.
    Equal-size means chunk 1 is usually SHORTER than the context window:

        n = 63  -> 2 chunks: 32, 31        chunk 1 holds 32 positions
        n = 100 -> 2 chunks: 50, 50        chunk 1 holds 50 positions
        n = 124 -> 2 chunks: 62, 62        chunk 1 holds 62 positions
        n = 125 -> 3 chunks: 42, 42, 41    chunk 1 holds 42 positions
        n = 187 -> 4 chunks: 47, 47, 47, 46

    Chunk 1 reaches a full 62 only just below each multiple of 62, and for
    very large books where ceil(n/k) happens to land on 62. So reading
    chunks[0] pools over fewer positions than the paper for most
    multi-chunk investors, worst immediately above each boundary.
    "paper" joins the chunks and takes the top context_window, which is
    the largest-62 sequence regardless of where the splits fell.

    "first-chunk" is kept only so previously trained
    q_*__weighted_first-chunk.parquet files remain reproducible. Do not
    use it for new runs.

CONTRASTIVE PAIRS
    The paper splits the even and odd positions of a portfolio into two
    halves "where each half contains up to 62 assets". 62 is the context
    window, so the SOURCE being split is up to 2 * context_window
    positions and each strided half lands at up to context_window.
    pooling_sequences() implements that. The previous version sliced the
    source at context_window, producing ~31-asset halves: half the
    paper's, and a length mismatch with extraction, which pools a full
    62-token sequence.

Outputs go to SEPARATE folders from the baseline run, so the original
embeddings/ and models/ are never touched and the two sets can be compared
side by side. Variant is also encoded in the filename:

  embeddings_weighted/q_YYYY-MM-DD__<variant>.parquet
  models_weighted/q_YYYY-MM-DD__<variant>/

Reproducing Gabaix et al. exactly (unweighted):
  python BERT_training_weighted.py --pooling mean --coverage paper

The extension (weighted pooling is then the ONLY deviation):
  python BERT_training_weighted.py --pooling weighted --coverage paper

Suggested comparison set (same seed, same data):
  --pooling mean     --coverage paper            # replication baseline
  --pooling weighted --coverage paper            # pooling effect alone
  --pooling weighted --coverage all-chunks       # + tail of the book
  --pooling weighted --coverage full-sequence --context-window 256

NOTE: pooling_sequences() and training_sequences() no longer depend on
coverage (outside full-sequence), so "paper", "first-chunk" and
"all-chunks" train an IDENTICAL model for a given quarter and seed, and
differ only inside compute_investor_embeddings(). Several readouts can
therefore be extracted from one training run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


# =========================================================================
# Hyperparameters
# =========================================================================
def default_config():
    """Every hyperparameter lives here in one dict. Edit values to change defaults."""
    return {
        # I/O  (separate folders so the original embeddings are never touched)
        "seq_dir":           "data",
        "emb_dir":           "embeddings_weighted",
        "model_dir":         "models_weighted",
        # Architecture (per the paper)
        "hidden_size":       64,
        "num_layers":        4,
        "num_heads":         2,
        "intermediate_size": 256,
        "context_window":    62,
        # max_position is derived from context_window (+2 for [CLS]/[SEP])
        # Pretraining
        "mask_prob":         0.15,
        "batch_size":        64,
        "pretrain_lr":       5e-4,
        "pretrain_epochs":   10,
        "weight_decay":      0.01,
        "warmup_frac":       0.1,
        "train_split":       0.9,
        # Sentence-transformer fine-tuning
        "finetune_lr":       2e-4,
        "finetune_epochs":   3,
        # NEW: pooling and coverage
        "pooling":           "weighted",     # "mean" | "weighted"
        "coverage":          "paper",        # "paper"|"first-chunk"|"all-chunks"|"full-sequence"
        "weight_col":        "weights",      # real weights, if the parquet has them
        "weight_scheme":     "power",        # "power" | "zipf" | "exp" | "uniform"
        "weight_alpha":      0.75,           # decay strength (see rank_weights)
        # Misc
        "seed":              42,
        "skip_existing":     True,
        "device": ("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu")),
    }


def variant_tag(cfg):
    """Short string identifying this configuration, used in output paths."""
    tag = f"{cfg['pooling']}_{cfg['coverage']}"
    if cfg["pooling"] == "weighted" and not cfg["_real_weights"]:
        tag += f"_{cfg['weight_scheme']}{cfg['weight_alpha']:g}"
    if cfg["coverage"] == "full-sequence":
        tag += f"_ctx{cfg['context_window']}"
    return tag


# =========================================================================
# Vocabulary and tokenizer
# =========================================================================
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4


def build_vocab(tokens):
    """Build per-quarter integer<->string mappings from a flat list of tokens."""
    unique_tokens = sorted(set(tokens))
    itos = SPECIAL_TOKENS + unique_tokens
    stoi = {tok: i for i, tok in enumerate(itos)}
    return itos, stoi


def build_tokenizer(itos, model_max_length):
    """Wrap our vocab as a HuggingFace tokenizer (needed by Trainer + collator).
    WordLevel = no subword splitting — asset IDs are atomic units."""
    vocab_dict = {tok: i for i, tok in enumerate(itos)}
    backend = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object  = backend,
        pad_token         = "[PAD]",
        unk_token         = "[UNK]",
        cls_token         = "[CLS]",
        sep_token         = "[SEP]",
        mask_token        = "[MASK]",
        model_max_length  = model_max_length,
    )


def encode_tokens(tokens, stoi):
    """Convert issuer-id strings to integer ids (unknown -> [UNK])."""
    return [stoi.get(t, UNK_ID) for t in tokens]


def wrap_and_pad(body, max_len, body_weights=None):
    """Wrap a token-id list with [CLS]/[SEP] and pad to max_len.

    body:         list of integer token ids (assets only, no special tokens)
    body_weights: optional list of per-token pooling weights, same length

    Returns (input_ids, attention_mask, pool_weights). Special tokens and
    padding get pool_weight 0 so they never enter the weighted average --
    this is the key difference from attention_mask, which keeps [CLS]/[SEP]
    at 1 so the encoder still attends to them.
    """
    body = body[: max_len - 2]                          # room for [CLS], [SEP]
    input_ids      = [CLS_ID] + body + [SEP_ID]
    attention_mask = [1] * len(input_ids)
    if body_weights is None:
        pool_weights = [0.0] + [1.0] * len(body) + [0.0]
    else:
        pool_weights = [0.0] + list(body_weights[: len(body)]) + [0.0]

    pad_len = max_len - len(input_ids)
    input_ids      += [PAD_ID] * pad_len
    attention_mask += [0] * pad_len
    pool_weights   += [0.0] * pad_len
    return input_ids, attention_mask, pool_weights


# =========================================================================
# Weights: real if available, else derived from rank
# =========================================================================
def rank_weights(n, scheme, alpha):
    """Weights for positions of rank 1..n, normalized to sum to 1.

    Tokens are stored sorted by descending position size, so rank is a
    monotone (but only ordinal) proxy for portfolio weight.

      uniform : w_r = 1                -> identical to mean pooling
      zipf    : w_r = 1/r              -> classic heavy head
      power   : w_r = r^(-alpha)       -> alpha tunes head concentration
                                          (alpha=0 uniform, 1 = zipf)
      exp     : w_r = exp(-alpha(r-1)) -> very sharp decay, top few dominate

    'power' with alpha around 0.5-1.0 is a reasonable default: real
    portfolio weights typically follow an approximate power law, and a
    milder alpha avoids collapsing the embedding onto the top 2-3 names.
    """
    r = np.arange(1, n + 1, dtype=np.float64)
    if scheme == "uniform":
        w = np.ones(n)
    elif scheme == "zipf":
        w = 1.0 / r
    elif scheme == "power":
        w = r ** (-float(alpha))
    elif scheme == "exp":
        w = np.exp(-float(alpha) * (r - 1.0))
    else:
        raise ValueError(f"unknown weight scheme: {scheme}")
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / max(n, 1))


# =========================================================================
# Portfolio assembly: chunks -> per-investor sequences
# =========================================================================
def assemble_portfolios(df, cfg):
    """Turn the chunked sequence table into one record per investor.

    Data_Cleaning.r splits portfolios longer than context_window into
    EQUAL-size chunks (k = ceil(n / 62) chunks of ceil(n / k) each) and
    writes an explicit `chunk_id` (1 = top of the book), so ordering is
    read rather than inferred. Older files without the column fall back
    to file row order, which is what the pipeline produced before
    `chunk_id` existed.

    Because the chunks are equal-size rather than filled to 62, chunk 1 is
    generally NOT the largest 62 positions -- see the module docstring.
    Anything that wants the top 62 must join the chunks first.

    Chunk order matters here in a way it does not for the unweighted
    baseline: weights are normalised across the WHOLE portfolio, so a
    mis-ordered chunk would receive the wrong slice of the weight vector
    and the tail of the book would be weighted as if it were the head.

    Returns a DataFrame with one row per investor:
      investor_id, investor_type, quarter_end, n_assets_full,
      chunks  : list of token lists, in rank order
      weights : list of weight arrays aligned with `chunks`, normalized
                to sum to 1 ACROSS THE WHOLE PORTFOLIO
    """
    df = df.reset_index(drop=True).copy()

    if "chunk_id" in df.columns:
        order_col = "chunk_id"
    else:
        print("  [warn] no chunk_id column; falling back to file row order")
        df["_row"] = np.arange(len(df))
        order_col = "_row"

    real_w_col = cfg["weight_col"] if cfg["weight_col"] in df.columns else None
    cfg["_real_weights"] = real_w_col is not None

    records = {}
    for row in df.sort_values(["investor_id", order_col],
                              kind="mergesort").itertuples(index=False):
        rec = records.setdefault(row.investor_id, {
            "investor_id":   row.investor_id,
            "investor_type": getattr(row, "investor_type", None),
            "quarter_end":   getattr(row, "quarter_end", None),
            "n_assets_full": getattr(row, "n_assets_full", None),
            "chunks":        [],
            "raw_weights":   [],
        })
        rec["chunks"].append(list(row.tokens))
        if real_w_col is not None:
            rec["raw_weights"].append(np.asarray(getattr(row, real_w_col),
                                                 dtype=np.float64))

    out = []
    for rec in records.values():
        lengths = [len(c) for c in rec["chunks"]]
        n_total = int(sum(lengths))

        if real_w_col is not None:
            flat = np.concatenate(rec["raw_weights"]) if rec["raw_weights"] \
                   else np.ones(n_total)
            flat = np.clip(flat, 0.0, None)
            if len(flat) != n_total:            # weights/tokens out of step
                flat = np.ones(n_total)
            s = flat.sum()
            flat = flat / s if s > 0 else np.full(n_total, 1.0 / max(n_total, 1))
        else:
            # derive from GLOBAL rank across the whole portfolio, so chunk 2
            # continues at rank 63 rather than restarting at 1
            flat = rank_weights(n_total, cfg["weight_scheme"], cfg["weight_alpha"])

        split_points = np.cumsum(lengths)[:-1]
        rec["weights"] = [np.asarray(w) for w in np.split(flat, split_points)]
        rec.pop("raw_weights", None)
        out.append(rec)

    return pd.DataFrame(out)


def training_sequences(portfolios, cfg):
    """Sequences fed to MLM pretraining, per coverage mode.

    paper / first-chunk / all-chunks
        Every chunk is a training sequence, exactly as in the paper --
        coverage only changes what gets EMBEDDED, not what gets trained on.
    full-sequence
        Chunks are re-joined and truncated to the (larger) context window,
        so attention spans the whole book.
    """
    seqs = []
    if cfg["coverage"] == "full-sequence":
        limit = cfg["context_window"]
        for rec in portfolios.itertuples(index=False):
            joined = [t for c in rec.chunks for t in c][:limit]
            if joined:
                seqs.append(joined)
    else:
        for rec in portfolios.itertuples(index=False):
            for c in rec.chunks:
                if c:
                    seqs.append(list(c))
    return seqs


def pooling_sequences(portfolios, cfg, stoi):
    """(token_ids, weights) pairs used for contrastive fine-tuning.

    Gabaix et al.: "We split the even and odd positions of a portfolio to
    construct portfolio pairs, where each half contains up to 62 assets."
    62 is the context window, so the SOURCE being split is up to
    2 * context_window positions; a strided even/odd split then puts each
    half at up to context_window -- exactly one forward pass.

    The source is joined across chunks. chunks[0] is NOT the top 62:
    Data_Cleaning.r uses equal-size chunks, so a 100-asset book gives two
    chunks of 50 and chunks[0] holds only the largest 50. Weights were
    normalised across the whole portfolio in assemble_portfolios, so any
    prefix of the joined weight vector is correctly scaled relative to the
    book; pool_hidden renormalises whatever it receives.

    This is deliberately coverage-independent: the contrastive step is the
    same for paper / first-chunk / all-chunks / full-sequence, which is why
    those variants share one trained model and differ only at extraction.
    """
    limit  = cfg["context_window"]
    source = 2 * limit                    # so each strided half is <= limit
    items = []
    for rec in portfolios.itertuples(index=False):
        body = [t for c in rec.chunks for t in c][:source]
        if len(rec.weights):
            w = np.concatenate(rec.weights)[:source]
        else:
            w = np.ones(len(body), dtype=np.float64)
        if len(body) < 2:
            continue
        if len(w) != len(body):           # defensive: keep weights in step
            w = np.ones(len(body), dtype=np.float64)
        w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
        items.append((encode_tokens(body, stoi), w))
    return items


# =========================================================================
# Datasets
# =========================================================================
class MLMDataset(Dataset):
    """Yields one tokenized portfolio at a time. The data collator handles
    the actual masking later, at batching time. No weights here: the MLM
    objective is token-level and does not pool."""

    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids, attention_mask, _ = wrap_and_pad(self.sequences[idx], self.max_len)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PairDataset(Dataset):
    """Splits each portfolio into even/odd ranks for sentence-transformer
    training, carrying each half's pooling weights along so the contrastive
    views are pooled exactly the way the final embedding will be.

    pooling_sequences supplies bodies of up to 2 * context_window tokens,
    so each strided half is at most context_window and fits one forward
    pass without truncation."""

    def __init__(self, items, max_len):
        self.items = [(ids, w) for ids, w in items if len(ids) >= 2]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def _to_tensors(self, body, body_weights):
        input_ids, attention_mask, pool_weights = wrap_and_pad(
            body, self.max_len, body_weights)
        return (torch.tensor(input_ids,      dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(pool_weights,   dtype=torch.float))

    def __getitem__(self, idx):
        ids, w = self.items[idx]
        w = np.asarray(w, dtype=np.float64)
        if len(w) != len(ids):                     # defensive: length mismatch
            w = np.ones(len(ids), dtype=np.float64)
        even_ids, even_w = ids[0::2], w[0::2]      # ranks 1, 3, 5, ...
        odd_ids,  odd_w  = ids[1::2], w[1::2]      # ranks 2, 4, 6, ...
        return (self._to_tensors(even_ids, even_w),
                self._to_tensors(odd_ids,  odd_w))


# =========================================================================
# Model setup
# =========================================================================
def build_bert_config(vocab_size, cfg):
    return BertConfig(
        vocab_size              = vocab_size,
        hidden_size             = cfg["hidden_size"],
        num_hidden_layers       = cfg["num_layers"],
        num_attention_heads     = cfg["num_heads"],
        intermediate_size       = cfg["intermediate_size"],
        max_position_embeddings = cfg["context_window"] + 2,
        type_vocab_size         = 1,
        pad_token_id            = PAD_ID,
        hidden_act              = "gelu",
    )


# =========================================================================
# Small helpers used in the training loops
# =========================================================================
def cosine_lr(step, total_steps, base_lr, warmup_steps):
    """Linear warmup, then cosine decay to zero."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr):
    """Override the learning rate in every optimizer parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def pool_hidden(hidden_states, attention_mask, pool_weights, pooling):
    """Pool token vectors into one vector per sequence.

    mean     : average over non-pad positions            (original)
    weighted : weighted average with pool_weights, which are zero on
               [CLS]/[SEP]/padding, so only real holdings contribute

    hidden_states: (batch, seq_len, hidden)
    attention_mask/pool_weights: (batch, seq_len)
    Returns (batch, hidden). NOT normalized -- callers decide.
    """
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    w = pool_weights.float()
    w = torch.clamp(w, min=0.0)
    denom = w.sum(dim=1, keepdim=True)
    # fall back to mean if a row has no usable weight (degenerate portfolio)
    fallback = attention_mask.float()
    w = torch.where(denom > 0, w, fallback)
    denom = w.sum(dim=1, keepdim=True).clamp(min=1e-9)
    w = (w / denom).unsqueeze(-1)
    return (hidden_states * w).sum(dim=1)


def contrastive_loss(emb_a, emb_b, temperature=20.0):
    """Symmetric InfoNCE loss with in-batch negatives.

    emb_a, emb_b: each (batch, hidden), already L2-normalized.
    """
    similarity = emb_a @ emb_b.t() * temperature
    targets = torch.arange(emb_a.size(0), device=emb_a.device)
    loss_a_to_b = F.cross_entropy(similarity,     targets)
    loss_b_to_a = F.cross_entropy(similarity.t(), targets)
    return (loss_a_to_b + loss_b_to_a) / 2


def make_amp_context(device):
    """Return an autocast context manager: bf16 on CUDA, no-op elsewhere."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# =========================================================================
# Phase 1 — Pre-training (via HuggingFace Trainer)
# =========================================================================
def pretrain_mlm(model, train_dataset, val_dataset, tokenizer, cfg, output_dir):
    """Pre-train BERT with masked-language-modeling (unchanged from the
    original script: the MLM objective is token-level, so neither pooling
    nor weighting applies here)."""
    use_bf16 = (cfg["device"] == "cuda")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer       = tokenizer,
        mlm             = True,
        mlm_probability = cfg["mask_prob"],
    )

    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "trainer_logs"),
        overwrite_output_dir        = True,
        num_train_epochs            = cfg["pretrain_epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"],
        learning_rate               = cfg["pretrain_lr"],
        weight_decay                = cfg["weight_decay"],
        warmup_ratio                = cfg["warmup_frac"],
        eval_strategy               = "epoch",
        logging_strategy            = "epoch",
        save_strategy               = "no",
        report_to                   = "none",
        bf16                        = use_bf16,
        seed                        = cfg["seed"],
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = data_collator,
    )
    trainer.train()
    return trainer.state.log_history


# =========================================================================
# Phase 2 — Sentence-transformer fine-tuning (manual loop, bf16 autocast)
# =========================================================================
def encode_for_pooling(model, input_ids, attention_mask, pool_weights, cfg):
    """Run BERT and return L2-normalized pooled embeddings, one per sequence."""
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = pool_hidden(output.last_hidden_state, attention_mask,
                         pool_weights, cfg["pooling"])
    return F.normalize(pooled, dim=-1)


def finetune_sentence_transformer(model, pair_dataset, cfg):
    """Siamese contrastive fine-tuning with in-batch negatives, using the
    configured pooling so training and extraction share one geometry."""
    device       = cfg["device"]
    epochs       = cfg["finetune_epochs"]
    batch_size   = cfg["batch_size"]
    base_lr      = cfg["finetune_lr"]
    weight_decay = cfg["weight_decay"]
    warmup_frac  = cfg["warmup_frac"]

    model.to(device)
    loader = DataLoader(pair_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=weight_decay)
    total_steps  = max(1, len(loader) * epochs)
    warmup_steps = int(total_steps * warmup_frac)
    amp_ctx      = make_amp_context(device)

    history = {"loss": []}
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_examples = 0

        for (ids_a, attn_a, w_a), (ids_b, attn_b, w_b) in loader:
            ids_a, attn_a, w_a = ids_a.to(device), attn_a.to(device), w_a.to(device)
            ids_b, attn_b, w_b = ids_b.to(device), attn_b.to(device), w_b.to(device)

            set_optimizer_lr(optimizer,
                cosine_lr(step, total_steps, base_lr, warmup_steps))
            optimizer.zero_grad()

            with amp_ctx:
                emb_a = encode_for_pooling(model, ids_a, attn_a, w_a, cfg)
                emb_b = encode_for_pooling(model, ids_b, attn_b, w_b, cfg)
                loss = contrastive_loss(emb_a, emb_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_n = emb_a.size(0)
            epoch_loss_sum += loss.item() * batch_n
            n_examples += batch_n
            step += 1

        avg_loss = epoch_loss_sum / max(1, n_examples)
        history["loss"].append(avg_loss)
        print(f"    finetune ep{epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return history


# =========================================================================
# Embedding extraction
# =========================================================================
@torch.no_grad()
def compute_investor_embeddings(model, portfolios, stoi, cfg):
    """One pooled vector per investor, honouring --pooling and --coverage.

    paper
        The largest context_window positions, JOINED across chunks. This
        is the paper's "average contextualized embedding for a given
        investor for the largest 62 positions". Joining is required
        because Data_Cleaning.r chunks equally, so chunk 1 is generally
        shorter than the context window (see the module docstring).

    first-chunk
        LEGACY. Reads chunks[0] only, which under equal-size chunking is
        NOT the largest 62 positions for most multi-chunk investors. Kept
        so previously trained files remain reproducible; prefer "paper".

    full-sequence
        Same slice as "paper", but with a larger context window and with
        MLM pretraining also run on the joined sequence, so attention
        actually spans the book.

    all-chunks
        Every chunk is encoded separately and the results are SUMMED using
        weights that were normalized across the whole portfolio. Because
        each chunk contributes  sum_{i in chunk} w_i * h_i  with globally
        normalized w, the total equals the weighted average over all
        positions -- i.e. the tail of the book is represented, at its own
        (small) weight, instead of being discarded. Normalization to unit
        length happens once, at the end.

        Caveat worth stating in the thesis: attention still operates
        within a chunk, so cross-chunk interactions are not modelled.
        This is a bag-of-chunks approximation; --coverage full-sequence
        with a larger context window is the stronger (costlier) fix.
    """
    device      = cfg["device"]
    batch_size  = cfg["batch_size"]
    hidden_size = cfg["hidden_size"]
    max_len     = cfg["context_window"] + 2
    amp_ctx     = make_amp_context(device)

    model.eval()
    n_inv = len(portfolios)
    output = np.zeros((n_inv, hidden_size), dtype=np.float32)

    # Flatten to a list of (investor_index, token_ids, weights) work items
    work = []
    for i, rec in enumerate(portfolios.itertuples(index=False)):
        if cfg["coverage"] in ("paper", "full-sequence"):
            body = [t for c in rec.chunks for t in c][: cfg["context_window"]]
            w    = np.concatenate(rec.weights)[: cfg["context_window"]]
            work.append((i, encode_tokens(body, stoi), w))
        elif cfg["coverage"] == "first-chunk":
            body = list(rec.chunks[0])[: cfg["context_window"]]
            w    = np.asarray(rec.weights[0])[: cfg["context_window"]]
            work.append((i, encode_tokens(body, stoi), w))
        else:  # all-chunks
            for chunk, w_chunk in zip(rec.chunks, rec.weights):
                body = list(chunk)[: cfg["context_window"]]
                w    = np.asarray(w_chunk)[: cfg["context_window"]]
                work.append((i, encode_tokens(body, stoi), w))

    for start in range(0, len(work), batch_size):
        batch = work[start:start + batch_size]
        batch_ids, batch_attn, batch_w, batch_scale = [], [], [], []

        for _, ids, w in batch:
            w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
            # weight mass of this sequence within the whole portfolio;
            # pool_hidden renormalizes internally, so multiply it back in
            scale = float(w.sum())
            if scale <= 0:
                w, scale = np.ones(len(ids)), 1.0 / max(len(ids), 1)
            ids_p, attn_p, w_p = wrap_and_pad(ids, max_len, w)
            batch_ids.append(ids_p)
            batch_attn.append(attn_p)
            batch_w.append(w_p)
            batch_scale.append(scale if cfg["coverage"] == "all-chunks" else 1.0)

        ids_tensor  = torch.tensor(batch_ids,  dtype=torch.long,  device=device)
        attn_tensor = torch.tensor(batch_attn, dtype=torch.long,  device=device)
        w_tensor    = torch.tensor(batch_w,    dtype=torch.float, device=device)

        with amp_ctx:
            hidden = model(input_ids=ids_tensor,
                           attention_mask=attn_tensor).last_hidden_state
            pooled = pool_hidden(hidden, attn_tensor, w_tensor, cfg["pooling"])

        pooled = pooled.float().cpu().numpy()
        for (inv_idx, _, _), vec, scale in zip(batch, pooled, batch_scale):
            output[inv_idx] += vec * scale        # accumulate across chunks

    # single L2 normalization at the end (clustering uses direction only)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return output / norms


# =========================================================================
# Small pipeline helpers
# =========================================================================
def split_sequences(sequences, train_split, seed):
    """Shuffle and split training sequences into train / validation."""
    rng = random.Random(seed)
    idx = list(range(len(sequences)))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_split)
    train = [sequences[i] for i in idx[:n_train]]
    val   = [sequences[i] for i in idx[n_train:]]
    return train, val


def save_embeddings_parquet(embeddings, portfolios, hidden_size, out_path):
    """Write one row per investor with metadata columns + dim_000..dim_(d-1)."""
    df = pd.DataFrame({
        "investor_id":   portfolios["investor_id"].values,
        "investor_type": portfolios["investor_type"].values,
        "quarter_end":   portfolios["quarter_end"].values,
    })
    for d in range(hidden_size):
        df[f"dim_{d:03d}"] = embeddings[:, d]
    df.to_parquet(out_path, index=False)


# =========================================================================
# Per-quarter pipeline
# =========================================================================
def process_quarter(parquet_path, cfg):
    q_label = parquet_path.stem.replace("q_", "")
    tag     = variant_tag(cfg)
    emb_path  = Path(cfg["emb_dir"])   / f"q_{q_label}__{tag}.parquet"
    model_dir = Path(cfg["model_dir"]) / f"q_{q_label}__{tag}"

    if cfg["skip_existing"] and emb_path.exists():
        print(f"  [skip] q_{q_label} ({tag} embedding exists)")
        return

    print(f"\n── q_{q_label} [{tag}] ──")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and assemble portfolios from chunks
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        print("  (empty quarter, skipping)")
        return
    df["tokens"] = df["tokens"].apply(list)

    portfolios = assemble_portfolios(df, cfg)
    n_chunks   = int(sum(len(c) for c in portfolios["chunks"]))
    multi      = int((portfolios["chunks"].apply(len) > 1).sum())
    print(f"  {len(df):,} rows -> {len(portfolios):,} investors "
          f"({multi:,} with >1 chunk)")
    if cfg["_real_weights"]:
        print(f"  weights: REAL, from column '{cfg['weight_col']}'")
    else:
        print(f"  weights: derived from rank "
              f"({cfg['weight_scheme']}, alpha={cfg['weight_alpha']})")

    # How much the equal-size chunking costs a chunks[0] readout. Printed so
    # the "paper" vs "first-chunk" difference is visible per quarter rather
    # than assumed.
    first_len = portfolios["chunks"].apply(lambda cs: len(cs[0]))
    full_len  = portfolios["chunks"].apply(lambda cs: sum(len(c) for c in cs))
    top_len   = np.minimum(full_len, cfg["context_window"])
    short = int((first_len < top_len).sum())
    if short:
        print(f"  chunk 1 is shorter than the top-{cfg['context_window']} "
              f"slice for {short:,} investors "
              f"(median {int(first_len[first_len < top_len].median())} "
              f"vs {cfg['context_window']} positions)")

    # 2. Training sequences (coverage-dependent) + vocab
    sequences = training_sequences(portfolios, cfg)
    all_tokens = [t for seq in sequences for t in seq]
    itos, stoi = build_vocab(all_tokens)
    vocab_size = len(itos)
    max_len    = cfg["context_window"] + 2

    tokenizer = build_tokenizer(itos, model_max_length=max_len)
    tokenizer.save_pretrained(model_dir / "tokenizer")
    print(f"  vocab: {vocab_size:,} tokens | {len(sequences):,} train sequences "
          f"| context {cfg['context_window']}")

    # 3. Encode and split
    encoded = [encode_tokens(seq, stoi) for seq in sequences]
    train_seqs, val_seqs = split_sequences(encoded, cfg["train_split"], cfg["seed"])
    train_dataset = MLMDataset(train_seqs, max_len)
    val_dataset   = MLMDataset(val_seqs,   max_len)

    # 4. Pre-train BERT (MLM)
    bert_config = build_bert_config(vocab_size, cfg)
    model = BertForMaskedLM(bert_config)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")
    pretrain_log = pretrain_mlm(
        model, train_dataset, val_dataset, tokenizer, cfg, model_dir)

    # 5. Save the post-pretrain checkpoint (for ablation studies)
    bert = model.bert
    torch.save(bert.state_dict(), model_dir / "bert_pretrain.pt")

    # 6. Contrastive fine-tuning, pooled the same way as extraction
    pair_items   = pooling_sequences(portfolios, cfg, stoi)
    pair_dataset = PairDataset(pair_items, max_len)
    if pair_items:
        half_lens = [(len(ids) + 1) // 2 for ids, _ in pair_items]
        print(f"  pair halves: median {int(np.median(half_lens))}, "
              f"max {max(half_lens)} (context window {cfg['context_window']})")
        assert max(half_lens) <= cfg["context_window"], \
            "a contrastive half exceeds the context window"
    if len(pair_dataset) >= cfg["batch_size"]:
        finetune_history = finetune_sentence_transformer(bert, pair_dataset, cfg)
    else:
        finetune_history = {"loss": []}
        print("  (skipping fine-tuning: too few pairs for one batch)")

    # 7. Investor embeddings
    embeddings = compute_investor_embeddings(bert, portfolios, stoi, cfg)

    # 8. Save
    Path(cfg["emb_dir"]).mkdir(parents=True, exist_ok=True)
    save_embeddings_parquet(embeddings, portfolios, cfg["hidden_size"], emb_path)
    torch.save(bert.state_dict(), model_dir / "bert.pt")
    (model_dir / "history.json").write_text(json.dumps(
        {"config": {k: v for k, v in cfg.items() if not k.startswith("_")},
         "pretrain_log": pretrain_log,
         "finetune": finetune_history,
         "n_investors": int(len(portfolios)),
         "n_chunks": n_chunks},
        indent=2, default=str))
    print(f"  saved -> {emb_path}")

    # 9. Free GPU memory before next quarter
    del model, bert, train_dataset, val_dataset, pair_dataset
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()


# =========================================================================
# Entry point
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir",         default="data")
    parser.add_argument("--emb-dir",         default="embeddings_weighted",
                        help="kept separate from the baseline 'embeddings/' dir")
    parser.add_argument("--model-dir",       default="models_weighted")
    parser.add_argument("--hidden-size",     type=int, default=64)
    parser.add_argument("--context-window",  type=int, default=62,
                        help="tokens per sequence; raise it with "
                             "--coverage full-sequence to span whole books")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size",      type=int, default=64)
    parser.add_argument("--seed",            type=int, default=42)
    # NEW
    parser.add_argument("--pooling", choices=["mean", "weighted"],
                        default="weighted")
    parser.add_argument("--coverage",
                        choices=["paper", "first-chunk", "all-chunks",
                                 "full-sequence"],
                        default="paper",
                        help="'paper' = largest context_window positions "
                             "joined across chunks (Gabaix et al.); "
                             "'first-chunk' is legacy and reads chunks[0], "
                             "which is NOT the top 62 under equal-size "
                             "chunking")
    parser.add_argument("--weight-col", default="weights",
                        help="column with real per-token weights, if present")
    parser.add_argument("--weight-scheme",
                        choices=["power", "zipf", "exp", "uniform"],
                        default="power",
                        help="rank-decay used when real weights are absent")
    parser.add_argument("--weight-alpha", type=float, default=0.75)
    parser.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                        help="process only quarters on or after this date")
    parser.add_argument("--end",   default=None, metavar="YYYY-MM-DD",
                        help="process only quarters on or before this date")
    parser.add_argument("--only", nargs="+", default=None, metavar="YYYY-MM-DD",
                        help="process exactly these quarters")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the quarters that would be processed, then exit")
    parser.add_argument("--no-skip", action="store_true",
                        help="re-train quarters even if embedding exists")
    parser.add_argument("--test", action="store_true",
                        help="use data/test as input dir")
    args = parser.parse_args()

    cfg = default_config()
    cfg["seq_dir"]         = "data/test" if args.test else args.seq_dir
    cfg["emb_dir"]         = ("embeddings_weighted/test" if args.test
                              else args.emb_dir)
    cfg["model_dir"]       = ("models_weighted/test" if args.test
                              else args.model_dir)
    cfg["hidden_size"]     = args.hidden_size
    cfg["context_window"]  = args.context_window
    cfg["pretrain_epochs"] = args.pretrain_epochs
    cfg["finetune_epochs"] = args.finetune_epochs
    cfg["batch_size"]      = args.batch_size
    cfg["seed"]            = args.seed
    cfg["pooling"]         = args.pooling
    cfg["coverage"]        = args.coverage
    cfg["weight_col"]      = args.weight_col
    cfg["weight_scheme"]   = args.weight_scheme
    cfg["weight_alpha"]    = args.weight_alpha
    cfg["skip_existing"]   = not args.no_skip
    cfg["_real_weights"]   = False        # set for real in assemble_portfolios

    if cfg["coverage"] == "full-sequence" and cfg["context_window"] <= 62:
        print("[warn] --coverage full-sequence with context_window <= 62 "
              "still truncates long books; consider --context-window 128 or 256.")
    if cfg["coverage"] == "first-chunk":
        print("[warn] --coverage first-chunk reads chunks[0], which under "
              "equal-size chunking is NOT the largest "
              f"{cfg['context_window']} positions. Use --coverage paper "
              "unless you are reproducing an older run.")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    print(f"device         : {cfg['device']}")
    print(f"hidden_size    : {cfg['hidden_size']}")
    print(f"context_window : {cfg['context_window']}")
    print(f"pooling        : {cfg['pooling']}")
    print(f"coverage       : {cfg['coverage']}")
    print(f"weight scheme  : {cfg['weight_scheme']} (alpha={cfg['weight_alpha']})"
          f"  [used only if no real weight column]")
    print(f"pretrain epochs: {cfg['pretrain_epochs']}")
    print(f"finetune epochs: {cfg['finetune_epochs']}")
    print(f"batch_size     : {cfg['batch_size']}")
    print(f"bf16           : {cfg['device'] == 'cuda'}")
    print(f"seq_dir        : {cfg['seq_dir']}")

    files = sorted(Path(cfg["seq_dir"]).glob("q_*.parquet"))
    if not files:
        print(f"No q_*.parquet files found in {cfg['seq_dir']}")
        return

    # detect real weights once, up front, so the variant tag is stable
    import pyarrow.parquet as pq
    cfg["_real_weights"] = cfg["weight_col"] in pq.read_schema(files[0]).names

    # ---- restrict to the requested quarters ------------------------------
    def label(p):
        return p.stem.replace("q_", "")

    n_all = len(files)
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if label(f) in wanted]
        missing = wanted - {label(f) for f in files}
        if missing:
            print(f"ERROR: requested quarter(s) not found in {cfg['seq_dir']}: "
                  f"{', '.join(sorted(missing))}")
            return
    else:
        if args.start:
            files = [f for f in files if label(f) >= args.start]
        if args.end:
            files = [f for f in files if label(f) <= args.end]

    if not files:
        print("No quarters left after filtering.")
        return

    # the variant tag is only final once a file has been read (it depends on
    # whether real weights are present), so report the tag as configured
    tag_preview = variant_tag(cfg)
    print(f"\nSelected {len(files)} of {n_all} quarters: "
          f"{label(files[0])} .. {label(files[-1])}")
    print(f"Output name pattern: q_<quarter>__{tag_preview}.parquet")
    for f in files:
        emb = Path(cfg["emb_dir"]) / f"q_{label(f)}__{tag_preview}.parquet"
        state = "EXISTS -> would skip" if (emb.exists() and cfg["skip_existing"]) \
                else ("EXISTS -> will overwrite" if emb.exists() else "new")
        print(f"    {label(f):12s}  {state}")

    if args.dry_run:
        print("\nDry run: nothing was trained.")
        return
    print()

    for path in files:
        process_quarter(path, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()