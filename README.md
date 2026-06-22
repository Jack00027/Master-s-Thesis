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
  Clustering.r           ── spherical k-means, choose-k, style validation
        │                   elbow.pdf, k_criteria.pdf
        ▼
  crowding_metric.R      ── crowding time series (HHI + tightness)
        │                   crowding/crowding_timeseries.csv
        │                   crowding/crowding_hhi.pdf
        ▼
  Master_Thesis.ipynb    ── downstream analysis (predictive tests, TBD)
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

### 3. Investor clustering — [Clustering.r](Clustering.r)

Clusters investors *within a quarter* from their embeddings. Embeddings are
L2-normalized so they live on the unit sphere, and clustering is done with
**spherical k-means** (`skmeans`, `pclust` method, cosine geometry).

- **Choose-k sweep** over `k = 2…12`. For each k it fits spherical k-means and
  records the within-cluster cosine distortion (elbow), cosine silhouette,
  spherical Davies–Bouldin, and Calinski–Harabasz, then reports which k each
  criterion votes for. Two plots are written: [elbow.pdf](elbow.pdf) (avg
  within-cluster cosine dissimilarity vs k) and
  [k_criteria.pdf](k_criteria.pdf) (silhouette / DB / CH faceted across k).
  Silhouette is evaluated on a 5 000-investor subsample during the sweep (the
  full `n × n` cosine matrix is multi-GB).
- **Internal metrics** are bundled in a reusable `cluster_metrics_sphere()`
  helper: cosine silhouette (primary), spherical + Euclidean Davies–Bouldin,
  Calinski–Harabasz, and a cosine-dispersion pseudo-CH.
- **Final clustering** at the chosen **k = 6** (seed 42) is validated against
  external FactSet investor **style** labels (`factset_styles.rds`):
  style-composition-within-cluster and cluster-distribution-within-style
  cross-tabs, plus a stacked style-by-cluster bar chart.

### 4. Crowding metric — [crowding_metric.R](crowding_metric.R)

Builds the **crowding time series** across all quarters. For each quarter it
loads the normalized embeddings, runs spherical k-means at a **fixed k = 6**
(held constant so the metric is comparable across time — the HHI floor is
`1/K`), and computes two crowding measures:

- **Cluster-concentration HHI** `HHI_t = Σ_c s_{c,t}²` of cluster shares (plus
  a `[0,1]`-rescaled version and the effective number of clusters `1/HHI`).
- **Tightness** — average within-cluster cosine similarity, computed via the
  exact `O(n·d)` identity `(n·‖mean‖² − 1)/(n − 1)` instead of the `O(n²·d)`
  pairwise matrix; reported as a size-weighted mean and for the largest cluster.

Per-quarter cluster assignments are cached to `crowding/assignments/` (so a
crashed run resumes), and "crowded" quarters are flagged when HHI exceeds the
historical 90th / 95th percentile (with a commented expanding-window variant
for the look-ahead-free predictive test). Outputs:
`crowding/crowding_timeseries.csv` and `crowding/crowding_hhi.pdf`.

### 5. Analysis notebook — [Master_Thesis.ipynb](Master_Thesis.ipynb)

Skeleton only at this point. The remaining downstream analysis — cluster
*transition* dynamics across quarters and validating that crowded trades
predict subsequent drawdowns — is the next step.

## Repository layout

```
.
├── Data_Cleaning.r         # WRDS → quarterly portfolio sequences
├── BERT_training.py        # PS-BERT training + investor embeddings
├── Clustering.r            # spherical k-means + choose-k + style validation
├── crowding_metric.R       # crowding time series (HHI + tightness)
├── Master_Thesis.ipynb     # analysis notebook (WIP)
├── data/                   # q_YYYY-MM-DD.parquet (one per quarter)
├── embeddings/             # q_YYYY-MM-DD.parquet (created by training)
├── models/                 # per-quarter checkpoints (created by training)
├── crowding/               # crowding outputs + per-quarter assignments
├── factset_styles.rds      # external FactSet investor style labels
├── elbow.pdf, k_criteria.pdf  # choose-k diagnostics
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

# 3. Cluster investors for a single quarter + run the choose-k sweep
#    (writes elbow.pdf, k_criteria.pdf; validates against FactSet styles)
Rscript Clustering.r

# 4. Build the crowding time series across all quarters
#    (writes crowding/crowding_timeseries.csv and crowding/crowding_hhi.pdf)
Rscript crowding_metric.R
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
| Clustering algorithm | spherical k-means (`skmeans`, pclust) | `Clustering.r`, `crowding_metric.R` |
| Number of clusters k | 6 (fixed across quarters) | `Clustering.r`, `crowding_metric.R` |
| Clustering seed | 42 | `Clustering.r`, `crowding_metric.R` |
| Choose-k sweep range | 2 … 12 | `Clustering.r` |
| Crowded-quarter thresholds | 90th / 95th HHI percentile | `crowding_metric.R` |
