# Investor Embeddings from Institutional Holdings

### Spherical Cauchy Mixtures of Learned Portfolio Similarity, and Their Dynamics

Code for the master's thesis of the same title (Jacopo Mei, WU Vienna, September
2026; supervisor Univ.Prof. Dipl.-Ing. Dr. Kurt Hornik). The full
text is in [`docs/MasterThesis_Jacopo_Mei.pdf`](docs/MasterThesis_Jacopo_Mei.pdf).

The thesis replicates the investor side of *Asset Embeddings* (Gabaix, Koijen,
Richmond and Yogo, 2025) and extends it. Each institutional portfolio is read as
a "sentence" of issuer tokens ordered by weight, a small BERT encoder is trained
on every quarter of holdings, and the resulting 64-dimensional investor vectors
are then:

1. **clustered on the unit sphere** with a finite mixture of spherical Cauchy
   distributions, and characterised with structured FactSet attributes instead
   of LLM narration (Chapter 5);
2. **aligned across years** with orthogonal Procrustes, anchored on the investors
   whose holdings changed least, to measure how far investors move and whether
   the clusters persist (Chapter 6, after Hamilton, Leskovec and Jurafsky, 2016).

---

## Contents

- [Repository layout](#repository-layout)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Requirements and setup](#requirements-and-setup)
- [Stage 1 — Preprocessing](#stage-1--preprocessing-01_preprocessing)
- [Stage 2 — Embeddings](#stage-2--embeddings-02_embeddings)
- [Stage 3 — Clustering](#stage-3--clustering-03_clustering)
- [Stage 4 — Dynamics](#stage-4--dynamics-04_dynamics)
- [From scripts to the thesis](#from-scripts-to-the-thesis)
- [Reproducibility](#reproducibility)
- [Data access and licensing](#data-access-and-licensing)
- [References](#references)

---

## Repository layout

```
.
├── 01_preprocessing/          WRDS extraction, token sequences, reference tables
│   ├── Data_Cleaning.r            holdings -> per-quarter token sequences
│   ├── Investor_data.r            investor and security reference tables
│   ├── preflight_panel.R          validates every sequence file before training
│   ├── make_panel_figure.R        Figure 1 (panel coverage)
│   ├── reporting_cycle_diagnostic.R   semi-annual reporters (Section 3.5)
│   └── panel_coverage_check.r     legacy: investor overlap across Q4s
├── 02_embeddings/             PS-BERT training (Python, GPU)
│   ├── BERT_training_weighted.py  pretraining, fine-tuning, embedding extraction
│   ├── train_all.sh               SLURM array job over all 85 quarters
│   └── losses.py                  collects training losses (Table 4)
├── 03_clustering/             spherical Cauchy mixture and interpretation
│   ├── choosing_k.r               stability-based k diagnostics
│   ├── Clusters.r                 fits the mixture, lenses A–D, association ranking
│   ├── structure_diagnostics.R    neighbourhood homogeneity (Section 5.2)
│   └── bic_curve.R                BIC against k (figures/bic_curve.pdf)
├── 04_dynamics/               cross-year alignment and persistence (Chapter 6)
│   ├── ch6_00_setup.R             configuration and shared helpers
│   ├── ch6_01_align.R             holdings stability, anchor rules, rotations
│   ├── ch6_02_stability_profile.R who is stable (one year pair)
│   ├── ch6_03_displacement.R      displacement and its link to turnover
│   ├── ch6_04_components.R        component persistence and switching
│   └── ch6_loco_check.R           leave-one-component-out check of the rules
├── docs/                      the thesis PDF
└── figures/                   figures produced by the scripts
```

The repository contains code and configuration only. Everything derived from
FactSet is created locally by the pipeline and is not part of the repository:

| Directory | Written by | Contents |
|---|---|---|
| `data/` | `Data_Cleaning.r` | `q_YYYY-MM-DD.parquet`, one token-sequence file per quarter |
| `reference/` | `Investor_data.r` (and `Data_Cleaning.r` for `mktcap/`) | investor and security reference tables, WRDS pull cache (`_cache/`), optional per-quarter market caps |
| `embeddings_weighted/` | `BERT_training_weighted.py` | weighted-arm investor embeddings |
| `embeddings_v3/` | `BERT_training_weighted.py` | baseline-arm investor embeddings |
| `models_weighted/`, `models_v3/` | `BERT_training_weighted.py` | trained encoders, tokenizer, `history.json` |
| `results/` | preprocessing diagnostics | cached counts and cross-tabs |
| `dynamics_hamilton/` | `04_dynamics/` | Chapter 6 outputs and caches |
| `logs/` | `train_all.sh` | SLURM logs |

**All scripts are run from the repository root**; every path inside them is
relative to it.

---

## Pipeline at a glance

```
WRDS / FactSet Ownership (13F hedge funds + registered funds)
   │
   ├─► Data_Cleaning.r ───────► data/q_<date>.parquet               85 quarters, 2005-Q1–2026-Q1
   │                              token sequences + weights + chunk_id
   │
   └─► Investor_data.r ───────► reference/investors_master.parquet
                                  reference/securities_master.parquet
   │
preflight_panel.R  (gate: fails loudly if any sequence file is malformed)
   │
BERT_training_weighted.py  ×85 quarters × 2 arms   (train_all.sh, SLURM)
   │   masked-token pretraining ─► contrastive fine-tuning ─► pooled embedding
   │
   ├─► embeddings_v3/q_<date>__mean_paper.parquet         baseline (replication)
   └─► embeddings_weighted/q_<date>__weighted_paper.parquet   weighted (extension)
   │
choosing_k.r ─► Clusters.r  (reference quarter 2025-Q4, k = 8)
   │   mixture_fit_<arm>_<date>_k8.rds, cluster_assignment_<arm>_<date>.rds
   │
04_dynamics/ch6_0*.R  (Q4 of 2005–2025)
       Procrustes rotations ─► displacement ─► persistence and switching
```

---

## Requirements and setup

**Data access.** A WRDS account with FactSet Ownership. Credentials are read
from environment variables, never from the source files. Put them in
`~/.Renviron` and restart R (the file is read only at startup):

```
WRDS_USER=your_wrds_username
WRDS_PASS=your_password
```

Both extraction scripts, `Data_Cleaning.r` and `Investor_data.r`, read these
two variables.

**R** 4.5.1. Packages: `tidyverse`, `dplyr`, `tidyr`, `purrr`, `readr`,
`stringr`, `tibble`, `lubridate`, `dbplyr`, `RPostgres`, `arrow`,
`data.table`, `flexmix`, `circlus`, `skmeans`, `mclust`, `clue`, `Matrix`,
`ggplot2`. The versions used for the thesis are in Appendix A, Table 22
(`flexmix` 2.3-20, `circlus` 0.0.2, `arrow` 21.0.0.1).

**Python** 3 with `torch`, `transformers`, `tokenizers`, `numpy`, `pandas`,
`pyarrow`. Training was run on a single NVIDIA A30 on the WU research cluster
in a conda environment named `psbert`; the script also runs on CUDA, Apple MPS
or CPU for single test quarters.

---

## Stage 1 — Preprocessing (`01_preprocessing/`)

### `Data_Cleaning.r` — token sequences

Builds one sequence file per quarter from 2005-Q1 to 2026-Q1, following
Appendix D of Gabaix et al. (2025):

- **Sources.** 13F holdings restricted to hedge funds (entity sub-type `HF`,
  reports within ±7 days of quarter end) and fund-level holdings of open-end,
  exchange-traded, closed-end and variable-annuity funds (±5 days; latest report
  per fund).
- **No CRSP merge.** Unlike the source paper, securities are kept at the FactSet
  issuer-entity level with no geographic restriction: the universe is global.
- **Filters.** Investors whose largest position exceeds 75% are dropped; then
  investors with fewer than 20 positions and issuers with fewer than 20 holders
  are pruned iteratively to a fixed point (this also removes micro- and
  nano-caps without a market-cap filter).
- **Tokenisation.** Holdings are sorted by market value and mapped to issuer
  tokens. Portfolios longer than 62 positions are split into `ceil(n / 62)`
  equal-size chunks.
- **Additions to the source pipeline.** Each token carries its portfolio weight
  (`weights`, normalised over the whole portfolio), and each chunk an explicit
  `chunk_id`, so the weighted pooling arm is estimable from identical inputs.

Output columns: `investor_id`, `investor_type`, `quarter_end`, `chunk_id`,
`tokens`, `weights`, `n_tokens`.

Switches at the top of the file: `START_QUARTER`, `END_QUARTER`, `TEST_MODE`
(two quarters into `data/test/`), `OVERWRITE`, `SAVE_MKTCAP`.

```bash
Rscript 01_preprocessing/Data_Cleaning.r
```

### `Investor_data.r` — reference tables

Builds the attribute tables used to characterise clusters (Section 3.4), as of
`AS_OF = 2025-12-31`:

- `reference/investors_master.parquet` — name, domicile, entity and fund type,
  declared style and turnover class, investment objective, portfolio
  characteristics (beta, P/E, P/B, dividend yield, sales growth, momentum,
  relative strength), managing institution and ultimate parent, AUM.
- `reference/securities_master.parquet` — issuer name, domicile, sector and
  industry, primary listing, capitalisation group, market cap, number of holders.
- `reference/securities_detail.parquet`, `reference/unresolved_issuers.csv` —
  listing-level detail and the issuers the entity master does not cover, with
  provenance columns so imputations are reportable.

The script caches each WRDS pull on disk and retries dropped connections, so an
interrupted run resumes.

### Checks and diagnostics

| Script | Purpose |
|---|---|
| `preflight_panel.R` | Verifies every sequence file: required columns, weights aligned with tokens and summing to one, contiguous `chunk_id`, valid quarter-end dates. Exits non-zero on failure, so it can gate a training run. |
| `make_panel_figure.R` | Investors and issuers per quarter with the semi-annual reporting cycle made visible → `figures/panel_by_quarter.pdf` (Figure 1). Add `--rebuild` to refresh the cache. |
| `reporting_cycle_diagnostic.R` | Classifies investors by which quarters they report in, confirming the Q2/Q4 sawtooth is a disclosure-calendar effect (Section 3.5). |
| `panel_coverage_check.r` | Legacy overlap diagnostic from an earlier layout; its embedding paths point to superseded directories. |

```bash
Rscript 01_preprocessing/preflight_panel.R          # --dir data --quiet
Rscript 01_preprocessing/make_panel_figure.R
```

---

## Stage 2 — Embeddings (`02_embeddings/`)

### `BERT_training_weighted.py`

One script serves both arms; they differ **only** in the pooling flag.

**Architecture** (per the paper where specified): BERT encoder with 4 layers, 2
attention heads, hidden size 64, intermediate size 256, context window of 62
holding tokens plus `[CLS]`/`[SEP]`. The vocabulary is the set of issuer tokens
in the quarter.

**Training**, one model per quarter:

1. *Masked-token pretraining*: 15% of tokens masked (80/10/10), 10 epochs,
   AdamW, learning rate 5e-4, 10% warm-up, 90/10 train/validation split by
   investor.
2. *Contrastive fine-tuning*: the even- and odd-ranked halves of each portfolio
   form a positive pair; InfoNCE with in-batch negatives and similarity scale 20
   (the SimCSE default, τ = 0.05); 3 epochs, learning rate 2e-4.
3. *Embedding extraction*: contextual token vectors of the largest 62 positions
   are pooled and L2-normalised.

**Arms.**

| Arm | Flags | Output |
|---|---|---|
| Baseline (replication) | `--pooling mean --coverage paper` | `embeddings_v3/q_<date>__mean_paper.parquet` |
| Weighted (extension) | `--pooling weighted --coverage paper` | `embeddings_weighted/q_<date>__weighted_paper.parquet` |

Weighted pooling averages token vectors by portfolio share, using the `weights`
column written by `Data_Cleaning.r`. If that column is missing, the script
falls back to rank-based weights and tags the output differently; `train_all.sh`
refuses to run in that case.

`--coverage paper` joins the chunks and embeds the largest 62 positions, as the
paper specifies. The other coverage options (`first-chunk`, `all-chunks`,
`full-sequence`) are kept for comparison; `first-chunk` is legacy and should not
be used, because under equal-size chunking chunk 1 is usually shorter than 62.

Useful flags: `--only <date> …`, `--start/--end`, `--dry-run`, `--no-skip`,
`--test` (reads `data/test/`), `--seed` (default 42).

Each embedding file has `investor_id`, `investor_type`, `quarter_end` and
`dim_000`–`dim_063`. Each model folder holds the encoder, the tokenizer and
`history.json` with the training losses.

```bash
# one quarter, locally
python 02_embeddings/BERT_training_weighted.py --only 2025-12-31 \
       --pooling weighted --coverage paper
```

### `train_all.sh` — the full panel on SLURM

Array job over all 85 quarters (`--array=0-84%4`, at most four tasks at once).
Completed quarters are skipped, so a partial sweep is resumed by resubmitting.
The job changes into `~/Master-s-Thesis` before running, so clone the
repository there or edit the `cd` line; it expects a conda environment named
`psbert`.

```bash
mkdir -p logs
Rscript 01_preprocessing/preflight_panel.R && \
  sbatch --export=ALL,ARM=weighted 02_embeddings/train_all.sh
sbatch --export=ALL,ARM=base 02_embeddings/train_all.sh
```

### `losses.py`

Collects the masked-token and contrastive losses from every `history.json` of
both arms into one table (source of Table 4).

---

## Stage 3 — Clustering (`03_clustering/`)

All results use the reference quarter **2025-Q4**, which is also the end point of
the Chapter 6 panel.

### `choosing_k.r` — choosing the number of components

BIC cannot choose k here: the per-component penalty is more than an order of
magnitude smaller than the likelihood gain, so BIC falls at every k on the grid.
This script provides the evidence used instead. On a fixed subsample it runs
spherical k-means and the spherical Cauchy mixture over a grid of k and reports
subsample stability (pairwise ARI), restart retention, the concentration profile
and the smallest component.

Output: `choosing_k_<arm>_<date>.csv` and `.rds`.

### `Clusters.r` — fitting and interpreting the mixture

Fits a spherical Cauchy mixture (`circlus::FLXMCspcauchy` in `flexmix`) at a
pre-chosen k and interprets every component through four lenses:

- **A — who:** investor type, fund type, declared style, turnover class, as lift
  over the population.
- **B — how:** breadth, AUM and the portfolio characteristics.
- **C — what:** country, region, sector and capitalisation composition of the
  holdings, and the most distinctive issuers by name.
- **D — administrative structure:** manager concentration, ultimate parent,
  declared mandate.

Every attribute is then scored against component membership (bias-corrected
Cramér's V for categorical, rank ε² for continuous) and grouped into families,
which answers whether the components reflect strategy or segmentation.

Main switches:

| Variable | Default | Meaning |
|---|---|---|
| `QUARTER` | `"2025-12-31"` | cross-section to cluster |
| `ARM` | `"weighted"` | `"weighted"` or `"baseline"` |
| `K_FIXED` | `8` | number of components (4 and 12 also reported) |
| `US_ONLY`, `US_THRESHOLD` | `FALSE`, `0.90` | Section 5.7: investors with ≥90% of positions in US issuers |
| `MINPRIOR` | `0.01` | smallest admissible component share (0.03 for the US-only run) |
| `N_RESTART`, `SEED` | `2`, `42` | EM restarts and seed |

Outputs: `mixture_fit_<arm>_<date>_k<k>[_us90].rds` (the flexmix fit, reused by
Chapter 6), `cluster_assignment_<arm>_<date>[_us90].rds` and two CSV tables.

```r
source("03_clustering/choosing_k.r")
source("03_clustering/Clusters.r")
source("03_clustering/structure_diagnostics.R")   # same session, reuses objects
source("03_clustering/bic_curve.R")               # figures/bic_curve.pdf (not in the final thesis)
```

`structure_diagnostics.R` computes the neighbourhood homogeneity of Section 5.2
(share of the 10 and 50 nearest neighbours sharing style, domicile and fund
type, against the base rate) and must run in the same session as `Clusters.r`.

---

## Stage 4 — Dynamics (`04_dynamics/`)

Chapter 6 asks whether the Chapter 5 components persist and when investors move
between them. Each year is trained separately, so embedding spaces are not
comparable; consecutive fourth quarters (2005–2025) are aligned by orthogonal
Procrustes. Comparing Q4 to Q4 keeps the semi-annual reporters in both end
points of every step.

**Anchors.** A rotation is estimated from investors whose holdings did not
change. Stability is the weighted overlap of the top-62 slice,
`wt_overlap = Σ min(w_t, w_{t+1})`, computed from holdings only. Three rules are
compared:

| Rule | Anchors | Role |
|---|---|---|
| `stable_thr` | `wt_overlap ≥ 0.85` | default |
| `stable_strat` | top quarter by `wt_overlap` within each component | robustness |
| `all` | every investor present in both years | benchmark (Hamilton et al.) |

The threshold is 0.85 rather than 0.90 because price drift alone caps
buy-and-hold portfolios just below 0.90; at 0.90 almost all index trackers are
excluded (Section 6.3).

### Scripts, in order

Run from the root. Source the setup file first: the other scripts only load it
themselves when it is not already in the session.

```r
source("04_dynamics/ch6_00_setup.R")
source("04_dynamics/ch6_01_align.R")
source("04_dynamics/ch6_02_stability_profile.R")   # PY0 <- 2023; PY1 <- 2024 for another pair
source("04_dynamics/ch6_03_displacement.R")
source("04_dynamics/ch6_04_components.R")
source("04_dynamics/ch6_loco_check.R")             # after 02, same session
```

| Script | What it does | Main outputs (`dynamics_hamilton/weighted/`) |
|---|---|---|
| `ch6_00_setup.R` | Configuration and shared functions: Procrustes, chaining, spherical Cauchy posterior, Hungarian matching, holdings loaders. | — |
| `ch6_01_align.R` | For each year pair: `wt_overlap`, the three anchor rules, a 20% held-out evaluation (stable and overall held-out cosine, shuffled-anchor null, effective rank, median displacement), final rotations and chains to 2025. | `anchor_rules_by_pair.csv`, `cache/rotations.rds`, `fig_alignment_by_pair.pdf` |
| `ch6_02_stability_profile.R` | Stability for one pair (default 2024→2025) by style, fund type, turnover class and component; extremes; displacement by component under each rule. | `stability_*.csv`, `stability_by_group_*.csv`, `displacement_by_component_*.csv`, four figures |
| `ch6_03_displacement.R` | Displacement `1 − cos(x_t R, x_{t+1})` under each rule, and its Spearman correlation with turnover per year, pooled, without the anchors, and by turnover decile. | `displacement_by_year.csv`, `turnover_correlation_pooled.csv`, `displacement_by_turnover_decile.csv`, `investor_displacement.rds`, three figures |
| `ch6_04_components.R` | Applies the stored Chapter 5 mixture to every aligned year (fixed classifier); refits the mixture per year and matches it to 2025; retention, persistence over horizons, switching rates and their timing. | `fixed_classifier_by_year.csv`, `refit_persistence.csv`, `retention_by_component.csv`, `persistence_decay.csv`, `switch_rates.csv`, `switch_step_effects.csv`, figures |
| `ch6_loco_check.R` | Leave-one-component-out comparison of the rules on stable investors. | `loco_check_*.csv` |

### Main settings (`ch6_00_setup.R`)

| Variable | Default | Meaning |
|---|---|---|
| `YEARS`, `REF_YEAR` | `2005:2025`, `2025` | annual Q4 panel and reference frame |
| `ANCHOR_RULE` | `"stable_thr"` | default rule; the other two are always computed too |
| `STAB_THRESHOLD` | `0.85` | stable-investor threshold |
| `STAB_TOP_SHARE` | `0.25` | share kept per stratum by `stable_strat` |
| `MIN_ANCHORS` | `200` | fallback size if too few investors meet the threshold |
| `HOLDOUT`, `N_PERM_NULL` | `0.20`, `20` | held-out share and null permutations |
| `REF_FIT_FILE` | `mixture_fit_weighted_2025-12-31_k8.rds` | the Chapter 5 fit, verified on load |
| `MINPRIOR` | `0` | for the per-year refits, so all eight components are kept |
| `UNIVERSE` | `"global"` | set the environment variable to `us90` for the Section 5.7 universe |
| `NOISE_EMB_FILE` | `NULL` | optional same-quarter, different-seed embeddings for a noise floor |

Caches go to `dynamics_hamilton/cache/`. If a setting that affects the rotations
changes, delete `rotations.rds`, `disp_step_*.rds`, `refit_*.rds` and
`ref_mixture_*.rds` before rerunning; the holdings and `wt_overlap_*` caches can
stay.

---

## From scripts to the thesis

| Thesis | Produced by |
|---|---|
| Figure 1, Table 2, Section 3.5 (panel coverage) | `make_panel_figure.R`, `reporting_cycle_diagnostic.R` |
| Table 4 (training diagnostics) | `losses.py` |
| Section 5.2 (structure) | `structure_diagnostics.R` |
| Section 5.3 (choice of k) | `choosing_k.r` |
| Tables 5–11 (Chapter 5) and Appendix B, Tables 23–43 | `Clusters.r` (baseline and weighted arms; k = 8 and 12; US-only) |
| Tables 12–13, Figure 2 (stability, threshold) | `ch6_02_stability_profile.R` |
| Tables 14–15 (anchor rules) | `ch6_01_align.R` |
| Table 16 (leave one component out) | `ch6_loco_check.R` |
| Tables 17–18, Figure 3 (displacement) | `ch6_03_displacement.R` |
| Tables 19–21 (persistence, switching) | `ch6_04_components.R` |

---

## Reproducibility

- Every stage uses seed 42: encoder training, the train/validation split, the
  EM initialisation and the subsampling for the validity indices. The one
  exception is the BIC scan, whose 5,000-investor subsample is drawn with seed
  4525.
- The masked-token stage never uses the pooling operator, so with a common seed
  the two arms run the same computation there: their masked-token losses are
  identical to every recorded digit in all ten epochs of all 85 quarters. The
  arms diverge only at fine-tuning.
- The spherical Cauchy mixture at k = 8 is sensitive to its starting point: a
  refit of the 2025-Q4 cross-section with different initialisations agrees with
  the Chapter 5 partition at an ARI of 0.51. Chapter 6 therefore loads the
  stored Chapter 5 fit (`REF_FIT_FILE`) and verifies it on load, instead of
  refitting it.
- No same-quarter, different-seed retrain was run for the thesis. Setting
  `NOISE_EMB_FILE` to such a model adds a retraining noise floor to `ch6_03`
  and `ch6_04`.

---

## Data access and licensing

The holdings and attribute data are licensed from FactSet through WRDS and
cannot be redistributed, so this repository contains code and configuration
only. Reproducing the pipeline requires your own WRDS subscription with access
to FactSet Ownership; `Data_Cleaning.r` and `Investor_data.r` rebuild every
data file from it.

---

## References

- Gabaix, X., Koijen, R. S. J., Richmond, R. J. and Yogo, M. (2025). Asset
  embeddings. Working paper.
- Hamilton, W. L., Leskovec, J. and Jurafsky, D. (2016). Diachronic word
  embeddings reveal statistical laws of semantic change. *Proceedings of ACL
  2016*.
- Kato, S. and McCullagh, P. (2020). Some properties of a Cauchy family on the
  sphere derived from the Möbius transformations. *Bernoulli*, 26(4).
- Sablica, L., Hornik, K. and Grün, B. (2025). circlus: An R package for circular
  and spherical clustering using Poisson kernel-based and spherical Cauchy
  distributions. *Austrian Journal of Statistics*, 54.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes
  problem. *Psychometrika*, 31(1).