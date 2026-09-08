# =============================================================================
# Clusters.r — model-based clustering of investor embeddings + interpretation
#
# Method: spherical Cauchy mixture (circlus / flexmix), fitted by EM.
#
#   1. Load embeddings, sweep k by BIC, fit the final mixture
#   2. Attach the reference tables built by Investor_data.r
#   3. Interpret each cluster through four lenses:
#        A  WHO      investor type, fund type, style, turnover
#        B  HOW      breadth, AUM, portfolio characteristics (beta, P/E, ...)
#        C  WHAT     holdings by country, sector, cap group, named issuers
#        D  ARTIFACT manager concentration, ultimate parent, declared mandate
#   4. Rank every attribute by how strongly it explains cluster membership,
#      which answers the central question quantitatively:
#      are these clusters STRATEGY or SEGMENTATION?
#
# Inputs
#   embeddings_weighted/q_<date>__weighted_paper.parquet   (weighted arm)
#   embeddings_v3/q_<date>__mean_paper.parquet             (baseline arm)
#   data/q_<date>.parquet                 token sequences
#   reference/investors_master.parquet    investor attributes  (Investor_data.r)
#   reference/securities_master.parquet   issuer attributes    (Investor_data.r)
#
# NOTE ON GEOGRAPHY COLUMNS
#   Both reference tables carry columns named country_desc / iso_country /
#   region_code, meaning the ISSUER's domicile in securities_master and the
#   INVESTOR's own domicile in investors_master. The investor-side columns are
#   renamed inv_* on read so the two can never be confused. The per-investor
#   counterparts of issuer geography are computed explicitly from holdings:
#   pct_us (continuous) and modal_issuer_country (categorical).
#
# NOTE ON THE HOLDINGS SLICE  (changed)
#   The holdings view must be the SAME slice of the book the encoder saw,
#   or every characterisation below describes a different portfolio from
#   the one that produced the embedding.
#
#   Data_Cleaning.r chunks a portfolio into k = ceil(n/62) EQUAL-size chunks
#   of ceil(n/k) positions, so chunk 1 is usually SHORTER than 62:
#       n = 63  -> 32, 31           n = 125 -> 42, 42, 41
#       n = 100 -> 50, 50           n = 187 -> 47, 47, 47, 46
#   Filtering chunk_id == 1 therefore does NOT give the top 62; it gives a
#   variable-length prefix, median ~47 positions on recent quarters.
#
#   The training script's --coverage paper joins the chunks and takes the
#   largest CONTEXT_WINDOW positions, per Gabaix et al. This script now does
#   the same, which affects the US filter, all of Lens C, and inv_feat.
# =============================================================================

library(arrow)
library(dplyr)
library(tidyr)
library(purrr)
library(flexmix)
library(circlus)

# ------------------------------ CONFIG ---------------------------------------
QUARTER   <- "2025-12-31"      # reference quarter: the terminal point of the
                               # Chapter 6 annual panel (YEARS ends 2025), so
                               # the cross-section analysed here is the same
                               # one the dynamics chain ends on

# Must match --context-window in the training run that produced EMB_FILE.
CONTEXT_WINDOW <- 62

# Which embedding arm to interpret:
#   "baseline" = unweighted mean pooling, the replication of Gabaix et al.
#   "weighted" = the extension, positions weighted by portfolio share
# Both arms now come from BERT_training_weighted.py with --coverage paper and
# differ in --pooling alone, so the comparison isolates the pooling effect.
ARM <- "weighted"

EMB_FILE <- switch(
  ARM,
  baseline = sprintf("embeddings_v3/q_%s__mean_paper.parquet", QUARTER),
  weighted = sprintf("embeddings_weighted/q_%s__weighted_paper.parquet",
                     QUARTER),
  stop("ARM must be 'baseline' or 'weighted'"))

# Superseded files from the pre-correction runs. They are still on disk and
# would load without complaint, so name them explicitly rather than letting a
# stale path go unnoticed.
LEGACY_FILE <- switch(
  ARM,
  baseline = sprintf("embeddings_v2/q_%s.parquet", QUARTER),
  weighted = sprintf("embeddings_weighted/q_%s__weighted_first-chunk.parquet",
                     QUARTER))

DATA_FILE <- sprintf("data/q_%s.parquet", QUARTER)
INV_REF   <- "reference/investors_master.parquet"
SEC_REF   <- "reference/securities_master.parquet"

# ---- Geography restriction ------------------------------------------------
# The asset-embeddings paper restricts its universe to US equities (via a
# merge with CRSP) BEFORE training, so non-US securities are never tokenised
# and the encoder never sees them. That design cannot be reproduced here
# without regenerating the sequences and retraining: these embeddings were
# learned on a global universe.
#
# US_ONLY does something related but weaker, and post-hoc. Investors whose
# books are predominantly US are selected, and the mixture is re-fitted on
# that subset alone. Cross-investor geographic variation is thereby removed
# from the CLUSTERING even though it remains present in the REPRESENTATION.
# The question it answers is: once geography is held roughly fixed, what
# organises the remaining investors?
#
# The share is computed over positions whose issuer has a known domicile, so
# it is a share of IDENTIFIED holdings; issuers missing from the security
# reference table are excluded from both numerator and denominator. It is
# computed on the top-CONTEXT_WINDOW slice, i.e. the same positions the
# encoder saw, not on the whole book.
US_ONLY      <- TRUE   # TRUE = cluster only predominantly-US investors
US_THRESHOLD <- 0.90    # min share of an investor's positions in US issuers

# Weight the Lens C / inv_feat tables by portfolio share instead of counting
# each position once. This FOLLOWS THE ARM: mean pooling treats every position
# in the window equally, so equal-count tables are its coherent counterpart;
# weighted pooling does not, so counting a 0.01% position the same as a 5% one
# would describe a portfolio the encoder never read. Override only to run the
# cross-check.
WEIGHT_HOLDINGS <- (ARM == "weighted")

# The US filter deliberately stays UNWEIGHTED in both arms. It defines the
# clustering universe, and the two arms must cluster the same investors for
# the between-arm ARI to mean anything. Weighting it would retain a different
# subset per arm and confound the comparison.

# Holdings weighting affects the descriptive tables only, never the mixture,
# so it stays out of FIT_CACHE and OUT_TAG: the fit is reusable across modes
# and downstream scripts keep finding cluster_assignment_<arm>_<quarter>_us90.
SUFFIX    <- if (US_ONLY) sprintf("_us%02d", round(100 * US_THRESHOLD)) else ""
FIT_CACHE <- sprintf("mixture_fits_%s_%s%s.rds", ARM, QUARTER, SUFFIX)
OUT_TAG   <- sprintf("%s_%s%s", ARM, QUARTER, SUFFIX)
TAB_TAG   <- paste0(OUT_TAG, if (WEIGHT_HOLDINGS) "_wh" else "")

K_GRID    <- 2:12       # k values tried in the BIC sweep
K_FINAL   <- NA         # NA = take the BIC winner; or set a number yourself.
                        # Reset explicitly below, because re-sourcing in the
                        # same session would otherwise leave it numeric from
                        # the previous run and silently ignore a new best_k.
SEED      <- 42
MINPRIOR  <- 0.01       # smallest allowed cluster share (~1% of investors)
MIN_COMP  <- 150        # ...but never fewer than this many investors,
                        # which keeps components from collapsing on
                        # small subsets such as the US-only run
MIN_SHARE <- 0.10       # issuer must be held by >=10% of a cluster to be shown

# ========================= 1. LOAD + CLUSTER =================================
cat(sprintf("Quarter: %s | arm: %s | universe: %s\nEmbeddings: %s\n",
            QUARTER, ARM,
            if (US_ONLY) sprintf("US-only (>=%.0f%% US positions)", 100*US_THRESHOLD)
            else "global", EMB_FILE))

if (!file.exists(EMB_FILE)) {
  msg <- sprintf("embedding file not found: %s", EMB_FILE)
  if (file.exists(LEGACY_FILE))
    msg <- paste0(msg, "\n  The pre-correction file ", LEGACY_FILE,
                  " does exist. It was built with --coverage first-chunk and",
                  "\n  is NOT the paper's top-", CONTEXT_WINDOW,
                  " slice. Re-run 05_train_all.sh rather than pointing here.")
  stop(msg)
}
stopifnot(file.exists(DATA_FILE))

df <- read_parquet(EMB_FILE)
X  <- as.matrix(df[, grepl("^dim_", names(df))])
if (!ncol(X)) stop("no dim_* columns in ", EMB_FILE)
if (anyDuplicated(df$investor_id))
  stop(sprintf("%s has %d duplicated investor_id rows; X and meta would be ",
               EMB_FILE, sum(duplicated(df$investor_id))),
       "misaligned with the cluster labels")
nrm <- sqrt(rowSums(X^2))
if (any(!is.finite(nrm)) || any(nrm == 0))
  stop(sprintf("%d embedding row(s) in %s have zero or non-finite norm",
               sum(!is.finite(nrm) | nrm == 0), EMB_FILE))
X  <- X / nrm                                # unit length (cosine geometry)
meta <- df |> select(investor_id, quarter_end, investor_type)

# ---- holdings, loaded here because the US filter needs them before the fit --
# Geography columns here are the ISSUER's and keep their original names.
sec_ref <- read_parquet(SEC_REF) |>
  select(issuer_id, any_of(c(
    "entity_proper_name", "iso_country", "country_desc", "region_code",
    "sector_code", "factset_industry_desc", "factset_sector_code",
    "cap_group", "market_cap", "l2_id", "n_holders_total")))

# sec_ref must hold at most ONE row per issuer. It is left-joined onto the
# position table AFTER the top-CONTEXT_WINDOW cap, so a fan-out would give an
# investor more than CONTEXT_WINDOW rows: the wt renormalisation would divide
# by an inflated sum, every share in Lens C would be wrong, and nothing would
# error. Deduplicate rather than fail, but say how many rows went.
n_sec_raw <- nrow(sec_ref)
sec_ref <- sec_ref |> distinct(issuer_id, .keep_all = TRUE)
if (nrow(sec_ref) < n_sec_raw)
  cat(sprintf("[warn] %s duplicate issuer_id row(s) in %s; kept the first of each\n",
              format(n_sec_raw - nrow(sec_ref), big.mark = ","), SEC_REF))

seqs_all <- read_parquet(DATA_FILE) |>
  semi_join(meta, by = "investor_id")

if (!"chunk_id" %in% names(seqs_all))
  stop("no chunk_id column in ", DATA_FILE,
       "; regenerate it with the current Data_Cleaning.r, since chunk order ",
       "cannot be recovered from row order reliably")

has_weights <- "weights" %in% names(seqs_all)
if (WEIGHT_HOLDINGS && !has_weights)
  stop("WEIGHT_HOLDINGS is TRUE but ", DATA_FILE, " has no `weights` column")

# ---- top-CONTEXT_WINDOW positions, joined ACROSS chunks ---------------------
# This is the slice --coverage paper feeds the encoder. Chunks are unnested
# in chunk_id order and ranked globally, so rank 63 continues from chunk 1
# into chunk 2 rather than restarting.
# Only the first two chunks can contain a top-CONTEXT_WINDOW position. Under
# the equal-size chunker the smallest chunk 1 is 32 (at n = 63), so chunks
# 1+2 always supply at least 63 positions. Filtering before the unnest avoids
# materialising every position of every 9,000-asset index fund -- roughly a
# 60% reduction in intermediate rows on a full quarter.
long_base <- seqs_all |>
  filter(chunk_id <= 2) |>
  arrange(investor_id, chunk_id) |>
  mutate(pos = if (has_weights)
                 map2(tokens, weights,
                      ~ tibble(issuer_id = as.character(.x), w = as.numeric(.y)))
               else
                 map(tokens, ~ tibble(issuer_id = as.character(.x), w = NA_real_))) |>
  select(investor_id, n_assets_full, pos) |>
  unnest(pos) |>
  group_by(investor_id) |>
  mutate(rank = row_number()) |>
  filter(rank <= CONTEXT_WINDOW) |>
  ungroup() |>
  left_join(sec_ref, by = "issuer_id")

# How much this differs from the old chunk_id == 1 slice, printed so the
# change is visible rather than assumed.
chunk1_len <- seqs_all |> filter(chunk_id == 1) |>
  transmute(investor_id, n1 = map_int(tokens, length))
top_len <- long_base |> count(investor_id, name = "ntop")
cmp <- inner_join(chunk1_len, top_len, by = "investor_id")
n_short <- sum(cmp$n1 < cmp$ntop)
cat(sprintf(paste0(
  "\nHoldings slice: top %d positions joined across chunks.\n",
  "  chunk 1 alone would have been shorter for %s of %s investors ",
  "(median %d vs %d)\n"),
  CONTEXT_WINDOW,
  format(n_short, big.mark = ","), format(nrow(cmp), big.mark = ","),
  if (n_short) as.integer(median(cmp$n1[cmp$n1 < cmp$ntop])) else 0L,
  CONTEXT_WINDOW))

# Position weight used by the holdings tables; 1 = count each position once.
#
# RENORMALISED WITHIN INVESTOR. `w` is a position's share of the WHOLE book,
# but only the top CONTEXT_WINDOW positions are kept, so the retained weights
# sum to well under 1 for a diversified fund and to ~1 for a concentrated one.
# Left raw, an investor whose top 62 covers 95% of its book would contribute
# ~4x the mass of one whose top 62 covers 25%, and the cluster tables would
# describe the concentrated members far more than the diversified ones.
# Dividing by the retained sum makes every investor contribute exactly 1, so
# the tables read as the AVERAGE PORTFOLIO COMPOSITION across cluster members.
# TWO weight columns, because two different objects need different things:
#
#   pw  portfolio weight, renormalised within the retained slice. Computed
#       ALWAYS, identically in both arms. Used by inv_feat, i.e. by the
#       association ranking.
#   wt  what the Lens C tables count by. Follows the arm: pw for weighted
#       pooling, 1 for mean pooling.
#
# Why they differ. Lens C is a PORTRAIT of what a cluster holds, so it should
# match the representation it describes -- following the arm is right there.
# The association ranking is a MEASURING INSTRUMENT, and it is read across
# arms. If the instrument changed with the arm, a shift in the family ranking
# could come from the clustering or from the yardstick and there would be no
# way to tell which. So the features are fixed.
long_base <- long_base |>
  group_by(investor_id) |>
  mutate(pw = if (has_weights) {
                s <- sum(w, na.rm = TRUE)
                if (is.finite(s) && s > 0) w / s else 1 / n()
              } else 1 / n(),
         wt = if (WEIGHT_HOLDINGS) pw else 1) |>
  ungroup()

# ---- optional restriction to predominantly-US investors -------------------
us_share <- long_base |>
  group_by(investor_id) |>
  summarise(n_known = sum(!is.na(iso_country)),
            pct_us  = mean(iso_country == "US", na.rm = TRUE),
            .groups = "drop") |>
  # an investor whose holdings all have unknown domicile yields NaN from the
  # mean of an empty set; such investors cannot be shown to be US-focused, so
  # they are treated as 0 and counted separately rather than silently dropped
  mutate(pct_us = if_else(n_known == 0, 0, pct_us))

n_nocountry <- sum(us_share$n_known == 0)
if (n_nocountry > 0)
  cat(sprintf("\n[note] %s investor(s) have no issuer with a known domicile; ",
              format(n_nocountry, big.mark = ",")),
      "treated as non-US.\n", sep = "")

# how many investors would survive at various thresholds - printed always,
# so the choice of US_THRESHOLD can be made from the data rather than guessed
cat("\nInvestors by US share of positions:\n")
for (thr in c(0.50, 0.75, 0.90, 0.95, 0.99)) {
  cat(sprintf("  >= %3.0f%% US : %6s investors (%4.1f%%)\n", 100 * thr,
              format(sum(us_share$pct_us >= thr, na.rm = TRUE), big.mark = ","),
              100 * mean(us_share$pct_us >= thr, na.rm = TRUE)))
}

if (US_ONLY) {
  keep <- us_share |> filter(pct_us >= US_THRESHOLD) |> pull(investor_id)
  cat(sprintf(paste0(
    "\nUS-ONLY restriction at %.0f%%: %s of %s investors retained (%.1f%%)\n",
    "  median US share among retained: %.3f | among dropped: %.3f\n"),
    100 * US_THRESHOLD,
    format(length(keep), big.mark = ","),
    format(nrow(meta), big.mark = ","),
    100 * length(keep) / nrow(meta),
    median(us_share$pct_us[us_share$investor_id %in% keep], na.rm = TRUE),
    median(us_share$pct_us[!us_share$investor_id %in% keep], na.rm = TRUE)))
  if (length(keep) < 500)
    stop("too few investors survive the US filter; lower US_THRESHOLD")

  idx  <- meta$investor_id %in% keep
  X    <- X[idx, , drop = FALSE]
  meta <- meta[idx, ]
  long_base <- long_base |> filter(investor_id %in% keep)
  cat(sprintf("  clustering %s investors, %s positions\n",
              format(nrow(meta), big.mark = ","),
              format(nrow(long_base), big.mark = ",")))
}

# The cache key is ARM + QUARTER + SUFFIX, none of which changed when the
# embeddings were rebuilt with --coverage paper. A cache from the superseded
# first-chunk run would therefore load silently and every result below would
# describe the old representation. The provenance recorded inside the file is
# checked instead of trusting the name.
cache_ok <- FALSE
if (file.exists(FIT_CACHE)) {
  cached <- readRDS(FIT_CACHE)
  stale <- c(
    if (is.null(cached$emb_file)) "written before provenance was recorded"
    else if (!identical(cached$emb_file, EMB_FILE))
      sprintf("fitted on %s, not %s", cached$emb_file, EMB_FILE),
    if (!is.null(cached$n_obs) && !identical(cached$n_obs, nrow(X)))
      sprintf("fitted on %s investors, now %s",
              format(cached$n_obs, big.mark = ","),
              format(nrow(X), big.mark = ",")),
    if (!is.null(cached$context_window) &&
        !identical(cached$context_window, CONTEXT_WINDOW))
      sprintf("context window %d, now %d", cached$context_window, CONTEXT_WINDOW))
  if (length(stale)) {
    cat(sprintf("[warn] ignoring stale fit cache %s\n         (%s)\n         Refitting.\n",
                FIT_CACHE, paste(stale, collapse = "; ")))
  } else {
    cache_ok <- TRUE
  }
}

if (cache_ok) {
  cat(sprintf("Loading cached mixture fits from %s\n", FIT_CACHE))
  fits <- cached$fits; sweep <- cached$sweep
  print(sweep)
} else {
  # minprior is a SHARE, so on a smaller sample it permits smaller components
  # in absolute terms. Components of a few dozen investors readily collapse
  # (rho -> 1) and return a non-finite likelihood, which aborts the sweep.
  # The floor is therefore raised until it admits at least MIN_COMP investors.
  eff_minprior <- max(MINPRIOR, MIN_COMP / nrow(X))
  if (eff_minprior > MINPRIOR)
    cat(sprintf("\n[note] minprior raised from %.3f to %.3f so that the ",
                MINPRIOR, eff_minprior),
        sprintf("smallest admissible component holds >= %d investors\n", MIN_COMP),
        sep = "")

  sweep <- data.frame(); fits <- list()
  for (k in K_GRID) {
    set.seed(SEED)
    # a degenerate component makes the likelihood non-finite and throws; that
    # k is skipped rather than aborting the whole sweep
    fit <- try(flexmix(X ~ 1, k = k, model = FLXMCspcauchy(),
                       control = list(minprior = eff_minprior)),
               silent = TRUE)
    if (inherits(fit, "try-error") || !is.finite(as.numeric(logLik(fit)))) {
      cat(sprintf("k=%2d -> FAILED (degenerate component); skipped\n", k))
      next
    }
    fits[[as.character(k)]] <- fit
    sweep <- rbind(sweep, data.frame(k_asked = k,
                                     k_kept  = length(unique(clusters(fit))),
                                     logLik  = as.numeric(logLik(fit)),
                                     BIC     = BIC(fit)))
    cat(sprintf("k=%2d -> kept %d components, BIC = %.0f\n",
                k, tail(sweep$k_kept, 1), tail(sweep$BIC, 1)))
  }
  if (!nrow(sweep)) stop("every k failed; raise MIN_COMP or shorten K_GRID")
  saveRDS(list(fits = fits, sweep = sweep, emb_file = EMB_FILE,
               n_obs = nrow(X), context_window = CONTEXT_WINDOW,
               us_threshold = if (US_ONLY) US_THRESHOLD else NA_real_,
               seed = SEED), FIT_CACHE)
  cat(sprintf("Cached mixture fits -> %s\n", FIT_CACHE))
}
# Only fits that actually retained k components are eligible. EM can return
# k-1 with a materially worse likelihood; selecting such a k would report one
# number and deliver another.
ok_fit <- sweep$k_kept == sweep$k_asked
if (any(!ok_fit))
  cat(sprintf("\n[note] k = %s returned fewer components than asked and are\n",
              paste(sweep$k_asked[!ok_fit], collapse = ", ")),
      "       excluded from selection (EM initialisation sensitivity)\n", sep = "")
if (!any(ok_fit)) stop("no k retained the requested number of components")
best_k <- sweep$k_asked[ok_fit][which.min(sweep$BIC[ok_fit])]
cat(sprintf("\nBIC selects k = %d%s\n", best_k,
            if (best_k == max(K_GRID)) "  (GRID BOUNDARY - not an interior optimum)" else ""))

# Re-sourcing safety. K_FINAL is overwritten below with a number, so a second
# source() in the same session would find it already numeric and never consult
# the new best_k. The user's choice is therefore captured UNCONDITIONALLY from
# whatever the CONFIG block currently says: an earlier `if (!exists(...))`
# guard here was worse than the bug, because it froze the first run's NA and
# silently discarded any manual K_FINAL set afterwards.
K_FINAL_USER <- K_FINAL
K_FINAL <- if (is.na(K_FINAL_USER)) best_k else K_FINAL_USER

# --- 3. a manually chosen k may not have been fitted at all ---
if (is.null(fits[[as.character(K_FINAL)]]))
  stop(sprintf(paste0("no fit stored at k = %d.\n",
                      "  Fitted k: %s\n",
                      "  (k values that failed or collapsed are absent.)"),
               K_FINAL, paste(names(fits), collapse = ", ")))
if (!is.na(K_FINAL_USER) && K_FINAL_USER != best_k)
  cat(sprintf("[note] K_FINAL set manually to %d; BIC would have chosen %d\n",
              K_FINAL_USER, best_k))
spc    <- fits[[as.character(K_FINAL)]]
labels <- clusters(spc)
cl_df  <- tibble(investor_id = meta$investor_id, cluster = labels)

rho <- as.numeric(parameters(spc)["rho", ])
sizes <- cl_df |> count(cluster, name = "size")
# rho comes from parameters() in component order; sizes from count() in label
# order. If the fit dropped a component these differ in length and the mutate
# below would recycle silently.
if (length(rho) != nrow(sizes))
  stop(sprintf("fit at k = %d has %d rho values but %d non-empty clusters",
               K_FINAL, length(rho), nrow(sizes)))
sizes <- sizes |> mutate(rho = round(rho, 3))
cat("\n=== Cluster sizes and concentration ===\n"); print(sizes, n = Inf)

# ========================= 2. ATTACH REFERENCE DATA ==========================
# country_desc / iso_country / region_code exist in BOTH reference tables with
# different meanings. Renaming the investor-side ones to inv_* prevents them
# from being read as issuer geography further down, in particular in the
# association ranking of section 4, where inv_feat draws its categorical
# columns from `inv` rather than from the holdings.
inv_ref <- read_parquet(INV_REF) |>
  select(investor_id, any_of(c(
    "fund_type", "fund_type_desc", "style_any", "turnover_any", "aum_any",
    "iso_country", "country_desc", "region_code",
    "mgr_entity_proper_name", "fs_ultimate_parent_entity_id",
    "entity_proper_name", "fund_family",
    "pe_ratio", "pb_ratio", "dividend_yield", "sales_growth",
    "price_momentum", "relative_strength", "beta",
    "invt_obj_code", "invt_obj_region_code", "invt_obj_country_code",
    "invt_obj_specialization_code", "invt_obj_asset_type_code"))) |>
  rename(any_of(c(inv_iso_country  = "iso_country",
                  inv_country_desc = "country_desc",
                  inv_region_code  = "region_code")))

inv <- cl_df |> left_join(meta, by = "investor_id") |>
                left_join(inv_ref, by = "investor_id")

# The reference tables were built from an earlier panel. Investors and
# issuers that appear only in newer quarters will not be in them, which
# shows up as reduced coverage rather than an error. Re-run Investor_data.r
# if coverage here is materially below the ~99% seen for 2019-Q3.
cat(sprintf("\nReference join: %.1f%% of investors matched, %.1f%% have a style\n",
            100 * mean(!is.na(inv$fund_type) | !is.na(inv$style_any)),
            100 * mean(!is.na(inv$style_any))))

# holdings with cluster labels attached
long <- long_base |> inner_join(cl_df, by = "investor_id")

cat(sprintf("Holdings: %s positions | %.1f%% of issuers named | weighting: %s\n",
            format(nrow(long), big.mark = ","),
            100 * mean(!is.na(long$entity_proper_name)),
            if (WEIGHT_HOLDINGS) "portfolio share" else "equal per position"))

# ---- helpers ----------------------------------------------------------------
# lift = share within cluster / share overall. >1 over-represented.
# `wt` is 1 unless WEIGHT_HOLDINGS, in which case shares are of portfolio
# weight rather than of position count. min_n stays a COUNT either way, so the
# support threshold means the same thing in both modes.
lift_table <- function(data, var, min_n = 20, top = 3) {
  var <- rlang::ensym(var)
  d <- data |> filter(!is.na(!!var))
  overall <- d |> group_by(!!var) |>
    summarise(w_all = sum(wt), .groups = "drop") |>
    mutate(p_all = w_all / sum(w_all)) |> select(-w_all)
  d |> group_by(cluster, !!var) |>
    summarise(n = n(), w_cl = sum(wt), .groups = "drop") |>
    group_by(cluster) |> mutate(p_cl = w_cl / sum(w_cl)) |> ungroup() |>
    left_join(overall, by = rlang::as_string(var)) |>
    mutate(lift = round(p_cl / p_all, 2),
           pct  = round(100 * p_cl, 1)) |>
    filter(n >= min_n) |>
    group_by(cluster) |> slice_max(lift, n = top, with_ties = FALSE) |> ungroup() |>
    select(cluster, !!var, n, pct, lift)
}

# share of each cluster falling in the most common category (concentration)
top_share <- function(data, var, label) {
  var <- rlang::ensym(var)
  data |> filter(!is.na(!!var)) |>
    group_by(cluster, !!var) |>
    summarise(w = sum(wt), .groups = "drop") |>
    group_by(cluster) |>
    mutate(pct = 100 * w / sum(w)) |>
    slice_max(pct, n = 1, with_ties = FALSE) |>
    ungroup() |>
    transmute(cluster, !!label := as.character(!!var), pct = round(pct, 1))
}

# `inv` is one row per investor and has no wt column; give it one so the
# helpers above work unchanged on investor-level tables.
inv$wt <- 1

# ========================= 3A. LENS A — WHO ==================================
cat("\n\n############ LENS A — WHO IS IN EACH CLUSTER ############\n")

cat("\n=== Investor type (lift) ===\n")
print(lift_table(inv, investor_type, top = 2), n = Inf)

if ("fund_type_desc" %in% names(inv)) {
  cat("\n=== Fund type (lift) ===\n")
  print(lift_table(inv, fund_type_desc, top = 2), n = Inf)
}

# Investor domicile: where the FUND is registered, not where it invests.
# Reported here so it is visibly distinct from the issuer geography of Lens C.
if ("inv_country_desc" %in% names(inv)) {
  cat("\n=== Investor domicile: largest per cluster ===\n")
  print(top_share(inv, inv_country_desc, "top_inv_domicile"), n = Inf)
  cat("\n=== Investor domicile (lift) ===\n")
  print(lift_table(inv, inv_country_desc, top = 3), n = Inf)
}

# Style is now ~85% covered (own_ent_funds.style via Investor_data.r), so this
# is a genuine external validation rather than the 2% corroboration it was.
cat("\n=== Declared style: distribution within each cluster (rows = cluster) ===\n")
style_tab <- inv |> filter(!is.na(style_any)) |>
  count(cluster, style_any) |>
  group_by(cluster) |> mutate(pct = round(100 * n / sum(n), 1)) |> ungroup() |>
  select(-n) |>
  pivot_wider(names_from = style_any, values_from = pct, values_fill = 0)
print(style_tab, n = Inf, width = Inf)

cat("\n=== Style lift (which styles concentrate where) ===\n")
print(lift_table(inv, style_any, top = 3), n = Inf)

cat("\n=== Turnover label (lift) ===\n")
print(lift_table(inv, turnover_any, top = 2), n = Inf)

# ========================= 3B. LENS B — HOW ==================================
cat("\n\n############ LENS B — HOW THEY INVEST ############\n")

# Portfolio characteristics: the strategy fingerprint. ~96-99% covered for
# funds. Cluster medians tell you value vs growth vs income in numbers,
# independently of the declared style label.
char_vars <- intersect(c("beta", "pe_ratio", "pb_ratio", "dividend_yield",
                         "sales_growth", "price_momentum", "relative_strength"),
                       names(inv))
cat("\n=== Portfolio characteristics (cluster medians) ===\n")
print(inv |>
        group_by(cluster) |>
        summarise(n = n(),
                  across(all_of(char_vars), ~ round(median(.x, na.rm = TRUE), 2)),
                  .groups = "drop"), n = Inf, width = Inf)

cat("\n=== Size and breadth ===\n")
# n_assets_full is the WHOLE book, not the top-CONTEXT_WINDOW slice, so
# pct_over_62 reports how many investors have a book the encoder truncated.
breadth <- seqs_all |> distinct(investor_id, n_assets_full) |>
  inner_join(cl_df, by = "investor_id")
print(inv |>
        group_by(cluster) |>
        summarise(med_aum_musd = round(median(aum_any, na.rm = TRUE) / 1e6, 1),
                  pct_aum_known = round(100 * mean(!is.na(aum_any)), 1),
                  .groups = "drop") |>
        left_join(breadth |> group_by(cluster) |>
                    summarise(med_assets = median(n_assets_full),
                              pct_truncated = round(
                                100 * mean(n_assets_full > CONTEXT_WINDOW), 1),
                              .groups = "drop"), by = "cluster"), n = Inf)

# ========================= 3C. LENS C — WHAT THEY HOLD =======================
# Everything below operates on `long`, which is the top-CONTEXT_WINDOW slice
# joined to sec_ref, so country_desc / iso_country / region_code here are the
# ISSUER's, and the positions are the ones the encoder actually read.
cat("\n\n############ LENS C — WHAT THEY HOLD (segmentation checks) ############\n")

cat("\n=== Issuer country: top country per cluster ===\n")
print(top_share(long, country_desc, "top_country"), n = Inf)

cat("\n=== Issuer country (lift) ===\n")
print(lift_table(long, country_desc, min_n = 50, top = 3), n = Inf)

cat("\n=== Issuer region mix (% of positions) ===\n")
print(long |> filter(!is.na(region_code)) |>
        group_by(cluster, region_code) |>
        summarise(w = sum(wt), .groups = "drop") |>
        group_by(cluster) |> mutate(pct = round(100 * w / sum(w), 1)) |>
        ungroup() |> select(-w) |>
        pivot_wider(names_from = region_code, values_from = pct, values_fill = 0),
      n = Inf, width = Inf)

cat("\n=== Sector (lift) ===\n")
print(lift_table(long, factset_industry_desc, min_n = 50, top = 3), n = Inf)

cat("\n=== Market-cap group mix (% of positions) ===\n")
print(long |> filter(!is.na(cap_group)) |>
        group_by(cluster, cap_group) |>
        summarise(w = sum(wt), .groups = "drop") |>
        group_by(cluster) |> mutate(pct = round(100 * w / sum(w), 1)) |>
        ungroup() |> select(-w) |>
        pivot_wider(names_from = cap_group, values_from = pct, values_fill = 0),
      n = Inf, width = Inf)

cat("\n=== Median market cap of holdings (USD m) ===\n")
print(long |> group_by(cluster) |>
        summarise(med_mktcap_musd = round(median(market_cap, na.rm = TRUE) / 1e6, 0),
                  pct_known = round(100 * mean(!is.na(market_cap)), 1),
                  .groups = "drop"), n = Inf)

# distinctive issuers, WITH NAMES. Counted per investor (does the investor
# hold it at all), so this one is deliberately unweighted in both modes.
n_by_cl  <- long |> distinct(investor_id, cluster) |> count(cluster, name = "n_cl")
held_all <- long |> distinct(investor_id, issuer_id) |>
  count(issuer_id, name = "n_all") |>
  mutate(p_all = n_all / n_distinct(long$investor_id))

distinctive <- long |>
  distinct(cluster, investor_id, issuer_id, entity_proper_name) |>
  count(cluster, issuer_id, entity_proper_name, name = "n_hold") |>
  left_join(n_by_cl, by = "cluster") |>
  mutate(p_cl = n_hold / n_cl) |>
  left_join(held_all |> select(issuer_id, p_all), by = "issuer_id") |>
  filter(p_cl >= MIN_SHARE) |>
  mutate(lift = round(p_cl / p_all, 1), p_cl = round(p_cl, 2)) |>
  group_by(cluster) |> slice_max(lift, n = 8, with_ties = FALSE) |> ungroup() |>
  select(cluster, name = entity_proper_name, p_cl, lift)

cat("\n=== Distinctive holdings (held by >=10% of the cluster) ===\n")
print(distinctive, n = Inf)

# ========================= 3D. LENS D — ARTIFACT CHECKS ======================
cat("\n\n############ LENS D — ADMINISTRATIVE ARTIFACTS ############\n")

# If one manager or one parent dominates a cluster, the cluster is a product
# lineup, not a strategy. mgr_entity_proper_name covers ~97-99% of funds,
# far better than fund_family (18-41%).
cat("\n=== Largest managing institution per cluster ===\n")
print(top_share(inv, mgr_entity_proper_name, "top_manager"), n = Inf)

cat("\n=== Manager concentration: share held by the top 3 managers ===\n")
print(inv |> filter(!is.na(mgr_entity_proper_name)) |>
        count(cluster, mgr_entity_proper_name) |>
        group_by(cluster) |>
        summarise(n_managers = n(),
                  top3_pct = round(100 * sum(sort(n, decreasing = TRUE)[1:min(3, n())]) / sum(n), 1),
                  .groups = "drop"), n = Inf)

# Declared mandate: a fund mandated to invest in a region explains a
# geographic cluster without any inference from holdings.
if ("invt_obj_region_code" %in% names(inv)) {
  cat("\n=== Declared mandate region (lift) ===\n")
  print(lift_table(inv, invt_obj_region_code, top = 2), n = Inf)
}
if ("invt_obj_specialization_code" %in% names(inv)) {
  cat("\n=== Declared specialisation (lift) ===\n")
  print(lift_table(inv, invt_obj_specialization_code, top = 2), n = Inf)
}

# ========================= 4. STRATEGY OR SEGMENTATION? ======================
# One number per attribute: how much of cluster membership does it explain?
#   categorical -> Cramer's V, bias-corrected (Bergsma 2013)
#   continuous  -> rank-based epsilon-squared
# Both are in [0,1]. They are different statistics, so read the RANKING
# rather than comparing values across the two families too literally.
#
# Both depart from the textbook versions for reasons that bear directly on
# the answer; see the function definitions below. n_levels is printed for the
# categorical attributes because cardinality is what the correction addresses.
cat("\n\n############ WHAT EXPLAINS THE CLUSTERS? ############\n")

# Cramer's V, BIAS-CORRECTED (Bergsma 2013).
#
# The uncorrected statistic normalises by min(r, c) - 1, which here is always
# k - 1 because the cluster label is the narrow margin. Nothing corrects for
# the ROW dimension, so a variable with thousands of levels
# (mgr_entity_proper_name, fs_ultimate_parent_entity_id) inflates chi-square
# through near-empty cells and scores higher than a variable with four levels
# for reasons of cardinality alone. Since the artifact family is exactly the
# high-cardinality one, the uncorrected statistic tilts the strategy-versus-
# segmentation ranking toward "artifact" by construction.
#
# The correction subtracts the expected chi-square under independence and
# shrinks both dimensions accordingly. n_levels is reported alongside so the
# reader can see which attributes are high-dimensional.
cramers_v <- function(x, y) {
  ok <- !is.na(x) & !is.na(y)
  if (sum(ok) < 50 || n_distinct(x[ok]) < 2) return(NA_real_)
  tab <- table(x[ok], y[ok])
  n <- sum(tab)
  if (n <= 1) return(NA_real_)
  chi  <- suppressWarnings(chisq.test(tab)$statistic)
  phi2 <- as.numeric(chi) / n
  r <- nrow(tab); c <- ncol(tab)
  if (min(r, c) < 2) return(NA_real_)
  phi2c <- max(0, phi2 - (r - 1) * (c - 1) / (n - 1))
  rc <- r - (r - 1)^2 / (n - 1)
  cc <- c - (c - 1)^2 / (n - 1)
  denom <- min(rc, cc) - 1
  if (denom <= 0) return(NA_real_)
  sqrt(phi2c / denom)
}

# Rank-based epsilon-squared, NOT eta-squared.
#
# Plain eta-squared on the raw values returned ~0 for beta and momentum: a
# handful of extreme tail values dominate the total sum of squares, so the
# between-cluster share vanishes even when the clusters separate cleanly on
# the bulk of the distribution. Ranking first bounds every observation's
# influence and restores the signal. This matters directly: the affected
# variables are the STRATEGY characteristics, i.e. the side of the central
# question that the raw statistic was silently suppressing.
# Weighted median: the smallest x at which cumulative weight reaches half.
wmedian <- function(x, w) {
  ok <- !is.na(x) & !is.na(w) & w > 0
  if (!any(ok)) return(NA_real_)
  x <- x[ok]; w <- w[ok]
  o <- order(x); x <- x[o]; w <- w[o]
  x[which(cumsum(w) / sum(w) >= 0.5)[1]]
}

eps_sq <- function(x, g) {
  ok <- !is.na(x) & !is.na(g)
  if (sum(ok) < 50) return(NA_real_)
  r <- rank(x[ok]); g <- factor(g[ok])
  if (n_distinct(g) < 2) return(NA_real_)
  ss_tot <- sum((r - mean(r))^2)
  ss_b <- sum(tapply(r, g, function(v) length(v) * (mean(v) - mean(r))^2))
  if (ss_tot == 0) NA_real_ else ss_b / ss_tot
}

# Investor-level features. Everything computed in the summarise() below is a
# per-investor summary of the HOLDINGS (top-CONTEXT_WINDOW slice); everything
# arriving through the join with `inv` is an attribute of the INVESTOR. Note
# that summarise() keeps only the columns it names, so issuer geography
# reaches this table only via pct_us and modal_issuer_country - hence both
# are constructed explicitly.
# WEIGHTING, stated explicitly because it is mixed on purpose:
#   pct_us, modal_issuer_country  UNWEIGHTED. pct_us must reproduce the US
#     filter exactly, and modal_issuer_country is its categorical twin.
#   pct_top_sec, med_cap          WEIGHTED by pw, in BOTH arms. Portfolio
#     sector concentration and size exposure are weight concepts: a fund with
#     40 tech names at 0.5% and 22 utilities at 3% is a utilities fund, and
#     the count-based measure calls it a tech fund.
inv_feat <- long |>
  group_by(investor_id) |>
  summarise(pct_us      = mean(iso_country == "US", na.rm = TRUE),
            pct_top_sec = {
              d <- tapply(pw, factset_industry_desc, sum)
              if (length(d)) max(d, na.rm = TRUE) / sum(d, na.rm = TRUE)
              else NA_real_
            },
            med_cap     = wmedian(market_cap, pw),
            # count-based twins, kept only for the sensitivity print below
            pct_top_sec_uw = {t <- table(factset_industry_desc)
                              if (length(t)) max(t)/sum(t) else NA_real_},
            med_cap_uw     = median(market_cap, na.rm = TRUE),
            # categorical counterpart of pct_us: the single country in which
            # the investor holds most of its identified positions
            modal_issuer_country = {
              t <- table(country_desc)
              if (length(t)) names(t)[which.max(t)] else NA_character_
            },
            .groups = "drop") |>
  right_join(inv, by = "investor_id") |>
  left_join(breadth |> select(investor_id, n_assets_full), by = "investor_id")

# Does the weighting choice change the answer? Printed rather than buried, so
# the decision is auditable. The _uw twins are excluded from assoc below.
cat("\n=== Weighted vs count-based features (sensitivity) ===\n")
print(tibble(
  feature = c("pct_top_sec", "med_cap"),
  eps_sq_weighted = c(eps_sq(inv_feat$pct_top_sec,    inv_feat$cluster),
                      eps_sq(inv_feat$med_cap,        inv_feat$cluster)),
  eps_sq_counted  = c(eps_sq(inv_feat$pct_top_sec_uw, inv_feat$cluster),
                      eps_sq(inv_feat$med_cap_uw,     inv_feat$cluster))) |>
  mutate(across(where(is.numeric), ~round(.x, 3))))
cat("Reported below: the WEIGHTED versions, in both arms.\n")

cat_vars <- intersect(c("style_any", "turnover_any", "fund_type_desc",
                        "investor_type",
                        "inv_country_desc", "modal_issuer_country",
                        "mgr_entity_proper_name", "fs_ultimate_parent_entity_id",
                        "invt_obj_region_code", "invt_obj_specialization_code",
                        "invt_obj_asset_type_code"), names(inv_feat))
num_vars <- intersect(c("beta", "pe_ratio", "pb_ratio", "dividend_yield",
                        "price_momentum", "relative_strength", "sales_growth",
                        "aum_any", "n_assets_full", "pct_us", "pct_top_sec",
                        "med_cap"), names(inv_feat))

assoc <- bind_rows(
  tibble(attribute = cat_vars, kind = "categorical",
         statistic = "Cramer's V (bias-corrected)",
         strength = map_dbl(cat_vars, ~ cramers_v(inv_feat[[.x]], inv_feat$cluster)),
         n_levels = map_dbl(cat_vars, ~ n_distinct(inv_feat[[.x]], na.rm = TRUE)),
         coverage = map_dbl(cat_vars, ~ mean(!is.na(inv_feat[[.x]])))),
  tibble(attribute = num_vars, kind = "continuous",
         statistic = "rank epsilon-squared",
         strength = map_dbl(num_vars, ~ eps_sq(inv_feat[[.x]], inv_feat$cluster)),
         n_levels = NA_real_,
         coverage = map_dbl(num_vars, ~ mean(!is.na(inv_feat[[.x]]))))
) |>
  mutate(family = case_when(
    attribute %in% c("inv_country_desc", "modal_issuer_country", "pct_us",
                     "invt_obj_region_code",
                     "invt_obj_country_code")             ~ "geography",
    attribute %in% c("pct_top_sec", "invt_obj_specialization_code") ~ "sector",
    attribute %in% c("med_cap", "aum_any")                ~ "size",
    attribute %in% c("mgr_entity_proper_name",
                     "fs_ultimate_parent_entity_id")      ~ "artifact",
    attribute %in% c("investor_type", "fund_type_desc",
                     "invt_obj_asset_type_code")          ~ "vehicle",
    TRUE                                                  ~ "strategy"),
    strength = round(strength, 3), coverage = round(coverage, 3)) |>
  arrange(desc(strength))

cat("\n=== Attributes ranked by how strongly they explain cluster membership ===\n")
cat("(inv_country_desc = where the fund is registered;",
    "modal_issuer_country = where it invests)\n")
print(assoc, n = Inf)

if (US_ONLY) {
  cat("\nNote: under the US-only restriction the HOLDINGS-based geography\n",
      "attributes (pct_us, modal_issuer_country) have little remaining\n",
      "variance by construction, so their scores are not comparable to the\n",
      "unrestricted run. Investor domicile still varies, since non-US funds\n",
      "may run US books. Compare the ORDERING of the other families.\n", sep = "")
}

cat("\n=== Rolled up by family (max strength within family) ===\n")
# Attributes whose statistic is NA are dropped BEFORE the rollup. With
# na.rm = TRUE inside summarise(), a family in which every attribute is NA
# gives which.max() a zero-length result (which errors) and max() -Inf.
# That is reachable: under US_ONLY, modal_issuer_country can collapse to a
# single level and return NA.
roll <- assoc |> filter(!is.na(strength)) |> group_by(family) |>
  summarise(best_attribute = attribute[which.max(strength)],
            strength = max(strength),
            n_scored = n(), .groups = "drop") |>
  arrange(desc(strength))
print(roll, n = Inf)

dropped <- assoc |> filter(is.na(strength))
if (nrow(dropped))
  cat(sprintf("[note] not scored (too few observations or a single level): %s\n",
              paste(dropped$attribute, collapse = ", ")))
missing_fam <- setdiff(unique(assoc$family), roll$family)
if (length(missing_fam))
  cat(sprintf("[note] family absent from the rollup entirely: %s\n",
              paste(missing_fam, collapse = ", ")))
cat("\nIf geography / artifact / vehicle outrank strategy, the clusters are\n",
    "segmentation. If strategy attributes lead, they are strategy groups.\n")

# ========================= 5. SAVE ===========================================
saveRDS(list(cluster = cl_df, k = K_FINAL, rho = rho, sweep = sweep,
             posterior = posterior(spc), seed = SEED, quarter = QUARTER,
             arm = ARM, emb_file = EMB_FILE,
             context_window = CONTEXT_WINDOW, weight_holdings = WEIGHT_HOLDINGS,
             stats = c(categorical = "Cramer's V (Bergsma-corrected)",
                       continuous  = "rank epsilon-squared"),
             feature_weighting = c(
               pct_us = "unweighted (matches the US filter)",
               modal_issuer_country = "unweighted",
               pct_top_sec = "portfolio-weighted, both arms",
               med_cap = "portfolio-weighted, both arms",
               lens_c_tables = if (WEIGHT_HOLDINGS) "portfolio-weighted"
                               else "equal per position"),
             style_tab = style_tab, distinctive = distinctive, assoc = assoc),
        sprintf("cluster_assignment_%s.rds", OUT_TAG))
write.csv(assoc, sprintf("cluster_association_ranking_%s.csv", TAB_TAG),
          row.names = FALSE)
write.csv(distinctive, sprintf("cluster_distinctive_holdings_%s.csv", TAB_TAG),
          row.names = FALSE)
cat(sprintf("\nSaved: cluster_assignment_%s.rds + 2 csv files (%s)\n",
            OUT_TAG, TAB_TAG))