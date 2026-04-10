import os
import math
import json
import random
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


class PortfoliosMLMDataset(Dataset):
    def __init__(self, token_lists: List[List[str]], tokenizer: BertTokenizerFast, max_len: int):
        self.token_lists = token_lists
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx: int):
        toks = self.token_lists[idx][: self.max_len]
        text = " ".join(toks)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len + 2,   # CLS/SEP
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def even_odd_split(tokens: List[str], max_len: int) -> Tuple[str, str]:
    toks = tokens[:max_len]
    a = toks[::2]
    b = toks[1::2]
    if len(a) == 0: a = toks[:1]
    if len(b) == 0: b = toks[:1]
    return " ".join(a), " ".join(b)


def build_vocab(token_lists: List[List[str]], out_dir: str) -> str:
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab_assets = sorted({t for toks in token_lists for t in toks})
    vocab = special + vocab_assets
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for t in vocab:
            f.write(t + "\n")
    return vocab_path


def build_mlm_model(vocab_size: int, max_len: int, hidden: int = 192, layers: int = 3, heads: int = 3) -> BertForMaskedLM:
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_len + 2 + 16,
        type_vocab_size=2,
    )
    return BertForMaskedLM(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences_parquet", type=str, default="data-r/portfolio_sequences_small.parquet")
    ap.add_argument("--out_dir", type=str, default="outputs_small")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--epochs_mlm", type=int, default=3)
    ap.add_argument("--bs_mlm", type=int, default=32)
    ap.add_argument("--lr_mlm", type=float, default=5e-4)
    ap.add_argument("--epochs_st", type=int, default=1)
    ap.add_argument("--bs_st", type=int, default=64)
    ap.add_argument("--lr_st", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sequences
    df = pd.read_parquet(args.sequences_parquet)
    # Expect columns: quarter, investor_id, tokens (list), n_assets
    token_lists = df["tokens"].tolist()

    # Build tokenizer
    tok_dir = os.path.join(args.out_dir, "tokenizer")
    vocab_path = build_vocab(token_lists, tok_dir)
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, do_lower_case=False)
    tokenizer.save_pretrained(tok_dir)

    # MLM pretrain
    mlm_dir = os.path.join(args.out_dir, "mlm_bert")
    os.makedirs(mlm_dir, exist_ok=True)

    ds_mlm = PortfoliosMLMDataset(token_lists, tokenizer, args.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    mlm_model = build_mlm_model(tokenizer.vocab_size, args.max_len)
    train_args = TrainingArguments(
        output_dir=mlm_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs_mlm,
        per_device_train_batch_size=args.bs_mlm,
        learning_rate=args.lr_mlm,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=mlm_model, args=train_args, train_dataset=ds_mlm, data_collator=collator)
    trainer.train()
    trainer.save_model(mlm_dir)
    tokenizer.save_pretrained(mlm_dir)

    # SentenceTransformer fine-tune (portfolio matching)
    pairs = []
    for toks in token_lists:
        a, b = even_odd_split(toks, args.max_len)
        pairs.append(InputExample(texts=[a, b]))

    word_emb = models.Transformer(mlm_dir, max_seq_length=args.max_len + 2)
    pooling = models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[word_emb, pooling], device=device)

    train_dl = DataLoader(pairs, shuffle=True, batch_size=args.bs_st, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(st_model)
    warmup_steps = math.ceil(len(train_dl) * args.epochs_st * 0.06)

    st_dir = os.path.join(args.out_dir, "sentence_transformer")
    st_model.fit(
        train_objectives=[(train_dl, train_loss)],
        epochs=args.epochs_st,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr_st},
        show_progress_bar=True,
        output_path=st_dir,
    )

    # Encode investor embeddings (quarter, investor_id)
    texts = [" ".join(t[:args.max_len]) for t in token_lists]
    embs = st_model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    embs = l2_normalize(embs)

    out = df[["quarter", "investor_id"]].copy()
    for j in range(embs.shape[1]):
        out[f"emb_{j}"] = embs[:, j].astype(np.float32)

    out_path = os.path.join(args.out_dir, "investor_embeddings_small.parquet")
    out.to_parquet(out_path, index=False)

    meta = {
        "device": device,
        "n_portfolios": len(df),
        "max_len": args.max_len,
        "mlm_dir": mlm_dir,
        "st_dir": st_dir,
        "out_path": out_path,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()