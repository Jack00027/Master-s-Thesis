# Investor Embeddings from Institutional Holdings — Progress Report

## 1. Project Objective

Replicate and extend the PS-BERT architecture introduced in Gabaix and Koijen (2025) to learn dense vector representations of institutional investors from their quarterly portfolio holdings. Each investor is treated as a sequence of issuer tokens ordered by portfolio weight. A small BERT is trained per quarter with a masked-token prediction objective followed by sentence-transformer fine-tuning. The mean-pooled contextualised output is the investor embedding. The downstream goal is to use these embeddings to detect crowded trades and test whether crowding predicts subsequent drawdowns.

---

## 2. Data Pipeline (`Data_Cleaning.r`)

### 2.1 Source data

Tables queried from the WRDS Postgres mirror of FactSet Ownership Data:

| Table | Content |
|---|---|
| `factset_own.wrds_own_13f` | Quarterly 13F filings (hedge funds) |
| `factset_own.wrds_own_fund` | Fund-level holdings (registered funds) |
| `factset_own.own_ent_funds` | Fund-type classification (OEF, ETF, CEF, VAR) |
| `factset_own.own_sec_entity_eq` | CUSIP-to-issuer entity mapping |

Positions are rolled up to the issuer-entity level rather than CUSIP, so multiple share classes of one firm collapse to a single token.

### 2.2 Coverage

2005-Q1 through 2025-Q4, **85 quarterly files** at `data/q_YYYY-MM-DD.parquet`.

R `seq.Date(by = "quarter")` starting from `2005-03-31` rolls invalid date arithmetic forward by one day, so Q2 and Q3 files are labelled `07-01` and `10-01` instead of `06-30` and `09-30`. Data inside the files is correct; only the `quarter_end` column and filenames are off by one day in those quarters.

### 2.3 Extraction

Two streams are pulled per quarter-end and stacked.

**13F holdings (hedge funds only)**: `entity_sub_type = "HF"`, report dates within ±7 days of quarter-end, `adj_mv > 0`. Aggregated to `(investor_id, issuer_id)` by summing dollar value.

**Fund holdings** (OEF, ETF, CEF, VAR): report dates within ±5 days of quarter-end. Duplicates resolved by keeping the latest report per (investor, issuer) sorted descending by `(report_date, filing_date, transfer_date, form_type)`.

### 2.4 Filtering

1. **Concentration filter**: drop (investor, quarter) where the largest position exceeds 75% of the portfolio (`MAX_TOP1_PCT = 0.75`).
2. **Bipartite pruning**: iteratively enforce ≥20 stocks per investor-quarter and ≥20 investors per stock-quarter until convergence.
3. **No survivorship correction** applied.

### 2.5 Tokenization

Positions within each (investor, quarter) are sorted descending by portfolio weight

$$
w_i \;=\; \frac{\text{adj\_mv}_i}{\sum_{j} \text{adj\_mv}_j}.
$$

Sequences longer than 62 tokens (the BERT context window) are split into

$$
k \;=\; \left\lceil \frac{N}{62} \right\rceil
$$

equal-sized chunks. All chunks for one investor share `investor_id` and `quarter_end`; their `tokens` lists are disjoint slices that together reconstruct the full portfolio.

### 2.6 Output schema

`data/q_YYYY-MM-DD.parquet`:

| Column | Type | Description |
|---|---|---|
| `quarter_end` | date | Quarter-end date |
| `investor_id` | string | FactSet entity or fund identifier |
| `investor_type` | string | One of HF, OEF, ETF, CEF, VAR |
| `tokens` | list&lt;string&gt; | Issuer IDs, weight-ordered, length ≤ 62 |
| `n_tokens` | int | Length of `tokens` |
| `n_assets_full` | int | Full portfolio size before chunking |

Representative row (single-chunk investor):

```json
{
  "quarter_end":    "2019-10-01",
  "investor_id":    "000C65-E",
  "investor_type":  "HF",
  "tokens":         ["06QJY3-E", "000LHL-E", "002X70-E", "0FM7LR-E", "..."],
  "n_tokens":       27,
  "n_assets_full":  27
}
```

---

## 3. Model Pipeline (`BERT_training.py`)

### 3.1 Per-quarter independent training

Each quarter is trained from scratch with its own vocabulary. No weight sharing across quarters.

### 3.2 Architecture

Hugging Face `BertConfig`:

| Parameter | Value |
|---|---|
| Hidden size $d$ | 64 |
| Hidden layers | 4 |
| Attention heads | 2 |
| Intermediate (FFN) size | 256 |
| Context window | 62 tokens |
| Max position embeddings | 64 (`[CLS]` + 62 + `[SEP]`) |
| Type vocabulary size | 1 |
| Checkpoint size on disk | ≈ 2.2 MB |

### 3.3 Vocabulary

Per quarter: `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]` plus every unique issuer ID appearing in that quarter's tokenized data. 2019-Q3 vocabulary: ≈ 11,900 tokens.

### 3.4 Masked-language-model pre-training

- 15% of body tokens are sampled for masking.
- Of those: 80% replaced with `[MASK]`, 10% with a random non-special token, 10% kept unchanged.
- Cross-entropy loss applied to masked positions only:

$$
\mathcal{L}_{\text{MLM}} \;=\; -\frac{1}{|M|} \sum_{i \in M} \log \Pr\!\left( t_i \;\middle|\; \mathbf{t}_{\setminus M} \right).
$$

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Optimiser | AdamW |
| Learning rate | $5 \times 10^{-4}$ |
| Weight decay | 0.01 |
| LR schedule | Cosine with 10% warmup |
| Batch size | 64 |
| Train / validation split | 90 / 10, by `investor_id` |
| Gradient clipping | Max norm 1.0 |

### 3.5 Sentence-transformer fine-tuning

Siamese contrastive learning. Tokens are split into even-rank / odd-rank halves to form a positive pair; negatives are drawn in-batch.

Let $(\mathbf{a}_i, \mathbf{b}_i)$ be the mean-pooled embeddings of investor $i$'s two halves in a batch of size $N$. The symmetric InfoNCE loss is

$$
\mathcal{L}_{\text{NCE}} \;=\; -\frac{1}{2N}\sum_{i=1}^{N} \!\left[
\log\!\frac{\exp\!\big(\tau\,\cos(\mathbf{a}_i, \mathbf{b}_i)\big)}{\sum_{j=1}^{N}\exp\!\big(\tau\,\cos(\mathbf{a}_i, \mathbf{b}_j)\big)}
\;+\;
\log\!\frac{\exp\!\big(\tau\,\cos(\mathbf{a}_i, \mathbf{b}_i)\big)}{\sum_{j=1}^{N}\exp\!\big(\tau\,\cos(\mathbf{a}_j, \mathbf{b}_i)\big)}
\right].
$$

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Optimiser | AdamW |
| Learning rate | $2 \times 10^{-4}$ |
| Temperature $\tau$ | 20 |

### 3.6 Embedding extraction

Each investor's first chunk (top 62 positions by weight) is passed through the model. The final-layer output $\mathbf{h}_{i,j} \in \mathbb{R}^{64}$ is mean-pooled over non-pad positions:

$$
\mathbf{e}_i \;=\; \frac{1}{|\mathcal{P}_i|} \sum_{j \in \mathcal{P}_i} \mathbf{h}_{i,j},
\qquad \mathcal{P}_i = \{j : \text{token}_{i,j} \neq \texttt{[PAD]}\}.
$$

Multi-chunk investors discard chunks 2 onwards at extraction.

Outputs:

- `embeddings/q_YYYY-MM-DD.parquet`: `investor_id, investor_type, quarter_end, dim_000, ..., dim_063`.
- `models/q_YYYY-MM-DD/`: tokenizer and model checkpoints.

---

## 4. Results for 2019-Q3

All numbers come from `embeddings/q_2019-10-01.parquet` and the corresponding training run. The filename uses `2019-10-01` rather than `2019-09-30` for the reason given in §2.2.

### 4.1 Investor distribution

**19,682 investors** after filtering.

| Investor type | Count | Share |
|---|---:|---:|
| OEF (open-end fund) | 14,676 | 74.6% |
| ETF | 3,081 | 15.7% |
| VAR (variable annuity) | 1,015 | 5.2% |
| HF (hedge fund) | 573 | 2.9% |
| CEF (closed-end fund) | 337 | 1.7% |
| **Total** | **19,682** | **100.0%** |

![Investor type distribution, 2019-Q3](type_distribution.png)

### 4.2 Sequence statistics

19,682 investors generate **57,391 training sequences**. Investors with more than 62 holdings (**8,366, or 42.5%**) are split into multiple chunks.

| Statistic | Holdings per investor |
|---|---:|
| Mean | 152.4 |
| Median | 53 |
| Min | 20 |
| Max | 8,929 |
| 75th percentile | 107 |
| 90th percentile | 349 |
| 99th percentile | 1,626 |

Per-type breakdown:

| Investor type | Mean | Median | Max |
|---|---:|---:|---:|
| VAR | 281.0 | 95 | 8,929 |
| ETF | 242.1 | 90 | 7,450 |
| HF | 184.5 | 46 | 3,165 |
| OEF | 125.4 | 50 | 6,058 |
| CEF | 67.4 | 50 | 504 |

### 4.3 Pre-training metrics

| Epoch | Train loss | Validation loss |
|:---:|:---:|:---:|
| 1 | 8.78 | 8.13 |
| 5 | 6.67 | 6.48 |
| 10 | 6.27 | 6.16 |

![Phase 1 (masked-language pre-training) and Phase 2 (sentence-transformer fine-tuning) loss curves, 2019-Q3](loss_curves.png)

Random-prediction baseline on a vocabulary of $|V| \approx 11{,}900$:

$$
\mathcal{L}_{\text{MLM}}^{\text{rand}} \;=\; \ln |V| \;\approx\; 9.4.
$$

Throughput: 3,737 samples/sec on NVIDIA A30 with bf16 mixed precision. Full 10-epoch pre-training: **139 seconds**.

### 4.4 Fine-tuning metrics

| Epoch | Loss |
|:---:|:---:|
| 1 | 0.38 |
| 2 | 0.19 |
| 3 | 0.16 |

Random baseline for in-batch contrastive learning with $N = 64$:

$$
\mathcal{L}_{\text{NCE}}^{\text{rand}} \;=\; \ln N \;\approx\; 4.16.
$$

### 4.5 Embedding distribution

19,682 rows × 64 dimensions.

- **L2 norms**: mean $\approx 4.9$, std $\approx 0.5$, range $[2.8,\, 6.8]$. No zero-norm vectors.
- **Per-dimension standard deviation**: every dimension has $\sigma \geq 0.33$; max $\sigma = 0.84$.
- **Pairwise cosine similarity** (5,000 random investor pairs): mean $|\!\cos\!| = 0.35$, std $= 0.30$, range $[-0.61,\, 1.00]$.
- **Investor-type structure** (20,000 random pairs):
    - Intra-type: $\overline{\cos} = 0.319$
    - Inter-type: $\overline{\cos} = 0.255$
    - $\Delta = +0.064$, where

$$
\Delta \;=\;
\overline{\cos(\mathbf{e}_a, \mathbf{e}_b)}_{\;\text{type}(a)=\text{type}(b)}
\;-\;
\overline{\cos(\mathbf{e}_a, \mathbf{e}_b)}_{\;\text{type}(a)\neq\text{type}(b)}.
$$

For comparison, $\Delta \approx -0.012$ on 2005-Q1.