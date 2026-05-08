# Detecting Crowded Trades Using Investor Embeddings from Institutional Holdings

Master's thesis project. The goal is to learn dense vector representations of
institutional investors from their quarterly portfolio holdings, and use those
embeddings to detect *crowded trades* — stocks held by clusters of investors
with similar strategies.

The approach adapts **PS-BERT** (Portfolio-Sequence BERT): each investor's
portfolio is treated as a "sentence" of issuer tokens ordered by weight, and a
small BERT is trained per quarter with masked-token prediction plus a
sentence-transformer fine-tuning step. The mean-pooled contextualized output
is the investor embedding.

## Pipeline

```
WRDS / FactSet Ownership
        │
        ▼
  Data_Cleaning.r        ── per-quarter portfolio sequences
        │                   data/q_YYYY-MM-DD.parquet
        ▼
  BERT_training.py       ── per-quarter PS-BERT + embeddings
        │                   embeddings/q_YYYY-MM-DD.parquet
        │                   models/q_YYYY-MM-DD/
        ▼
  Master_Thesis.ipynb    ── analysis (crowded-trade detection, TBD)
```

## What's done so far

### 1. Data extraction & cleaning — [Data_Cleaning.r](Data_Cleaning.r)

Connects to the WRDS Postgres mirror of FactSet Ownership and builds one
portfolio-sequence parquet per quarter from 2005-Q1 to 2025-Q4.

For each quarter-end:

- **13F holdings** (`wrds_own_13f`, hedge funds): reports within ±7 days of
  quarter-end, joined to issuer entity IDs via `own_sec_entity_eq`.
- **Fund holdings** (`wrds_own_fund` + `own_ent_funds`): OEF / ETF / CEF / VAR
  funds, reports within ±5 days of quarter-end, deduplicated to the latest
  report per investor-issuer.
- **Combined** holdings are aggregated by `(investor_id, issuer_id)` adjusted
  market value.
- **Filtering**: drop investors whose top holding exceeds 75% of portfolio
  weight; iteratively prune until every investor has ≥20 stocks and every
  stock has ≥20 investors (this implicitly removes micro/nano caps without a
  separate market-cap filter).
- **Tokenization**: each investor's holdings become a sequence of issuer IDs
  sorted by weight descending. Long portfolios (>62 assets) are split into
  equal-size chunks so every chunk fits the BERT context window.

Output: `data/q_YYYY-MM-DD.parquet` with columns
`quarter_end, investor_id, investor_type, tokens, n_tokens, n_assets_full`.
**85 quarterly files have been generated** in [data/](data/).

### 2. PS-BERT training — [BERT_training.py](BERT_training.py)

Per-quarter training pipeline. For each `data/q_*.parquet`:

1. **Vocabulary** — build `[PAD] [CLS] [SEP] [MASK]` + the unique issuer IDs
   appearing that quarter.
2. **MLM pre-training** — small BERT (4 layers, 2 heads, hidden=64,
   ctx=62, ~configurable). Standard 15% masking (80/10/10 split between
   `[MASK]`, random token, kept-as-is). 10 epochs, AdamW, cosine LR with
   10% warmup, batch 64, 90/10 train/val split *by investor*.
3. **Sentence-transformer fine-tuning** — siamese contrastive learning.
   Each investor's portfolio is split into even-rank vs odd-rank halves
   (positive pair); negatives are in-batch. Cosine-similarity logits with
   temperature 1/20, symmetric cross-entropy. 3 epochs.
4. **Embedding extraction** — mean-pooling over the top-62 tokens of each
   investor's portfolio, written as one row per investor.

Outputs:
- `embeddings/q_YYYY-MM-DD.parquet` — `investor_id, investor_type, quarter_end, dim_000…dim_d`
- `models/q_YYYY-MM-DD/bert.pt`, `vocab.json`, `history.json`

Resumable: existing embedding files are skipped unless `--no-skip` is set.
A `--test` flag runs against `data/test/`.

### 3. Analysis notebook — [Master_Thesis.ipynb](Master_Thesis.ipynb)

Skeleton only at this point — title cell and a hello-world. The downstream
analysis (clustering investors by embedding, scoring stocks by cluster
concentration, validating that crowded trades predict drawdowns) is the
next step.

## Repository layout

```
.
├── Data_Cleaning.r         # WRDS → quarterly portfolio sequences
├── BERT_training.py        # PS-BERT training + investor embeddings
├── Master_Thesis.ipynb     # analysis notebook (WIP)
├── data/                   # q_YYYY-MM-DD.parquet (one per quarter)
├── embeddings/             # q_YYYY-MM-DD.parquet (created by training)
├── models/                 # per-quarter checkpoints (created by training)
├── .Renviron               # WRDS_USER / WRDS_PASSWORD (gitignored)
└── README.md
```

## Reproducing

### Requirements

- **R**: `tidyverse`, `lubridate`, `dbplyr`, `RPostgres`, `arrow`
- **Python ≥ 3.10**: `torch`, `transformers`, `pandas`, `pyarrow`, `numpy`
- WRDS credentials in `.Renviron`:
  ```
  WRDS_USER=...
  WRDS_PASSWORD=...
  ```

### Run

```bash
# 1. Build quarterly sequences (slow — pulls from WRDS)
Rscript Data_Cleaning.r

# 2. Train PS-BERT and extract embeddings (one model per quarter)
python BERT_training.py
# or, for a quick smoke test:
python BERT_training.py --test
```

## Key parameters

| Parameter | Value | Where |
|---|---|---|
| Quarter range | 2005-Q1 … 2025-Q4 | `Data_Cleaning.r` |
| Min stocks per investor-quarter | 20 | `Data_Cleaning.r` |
| Min investors per stock-quarter | 20 | `Data_Cleaning.r` |
| Max single-holding weight | 75% | `Data_Cleaning.r` |
| Context window | 62 tokens | both files |
| BERT hidden / layers / heads | 64 / 4 / 2 | `BERT_training.py` |
| MLM masking rate | 15% (80/10/10) | `BERT_training.py` |
| Pre-train / fine-tune epochs | 10 / 3 | `BERT_training.py` |
