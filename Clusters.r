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
#   embeddings/q_<date>.parquet           investor embeddings
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
# =============================================================================

library(arrow)
library(dplyr)
library(tidyr)
library(purrr)
library(flexmix)
library(circlus)

# ------------------------------ CONFIG ---------------------------------------
QUARTER   <- "2026-03-31"      # most recent quarter in the regenerated panel

# Which embedding arm to interpret:
#   "baseline" = unweighted mean pooling, the replication of Gabaix et al.
#   "weighted" = the extension, positions weighted by portfolio share
# The two arms use different file-name conventions because the weighted
# training script encodes its variant in the name.
ARM <- "weighted"

EMB_FILE <- switch(
  ARM,
  baseline = sprintf("embeddings_v2/q_%s.parquet", QUARTER),
  weighted = sprintf("embeddings_weighted/q_%s__weighted_first-chunk.parquet",
                     QUARTER),
  stop("ARM must be 'baseline' or 'weighted'"))

DATA_FILE <- sprintf("data/q_%s.parquet", QUARTER)
INV_REF   <- "reference/investors_master.parquet"
SEC_REF   <- "reference/securities_master.parquet"

# The BIC sweep is the only slow step (11 EM fits). Cache it so that
# re-running to tweak a lens costs seconds rather than half an hour.
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
# reference table are excluded from both numerator and denominator.
US_ONLY      <- TRUE   # TRUE = cluster only predominantly-US investors
US_THRESHOLD <- 0.90    # min share of an investor's positions in US issuers

SUFFIX    <- if (US_ONLY) sprintf("_us%02d", round(100 * US_THRESHOLD)) else ""
FIT_CACHE <- sprintf("mixture_fits_%s_%s%s.rds", ARM, QUARTER, SUFFIX)
OUT_TAG   <- sprintf("%s_%s%s", ARM, QUARTER, SUFFIX)

K_GRID    <- 2:12       # k values tried in the BIC sweep
K_FINAL   <- NA         # NA = take the BIC winner; or set a number yourself
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
stopifnot(file.exists(EMB_FILE), file.exists(DATA_FILE))
df <- read_parquet(EMB_FILE)
X  <- as.matrix(df[, grepl("^dim_", names(df))])
X  <- X / sqrt(rowSums(X^2))                 # unit length (cosine geometry)
meta <- df |> select(investor_id, quarter_end, investor_type)

# ---- holdings, loaded here because the US filter needs them before the fit --
# Geography columns here are the ISSUER's and keep their original names.
sec_ref <- read_parquet(SEC_REF) |>
  select(issuer_id, any_of(c(
    "entity_proper_name", "iso_country", "country_desc", "region_code",
    "sector_code", "factset_industry_desc", "factset_sector_code",
    "cap_group", "market_cap", "l2_id", "n_holders_total")))

seqs_all <- read_parquet(DATA_FILE) |>
  semi_join(meta, by = "investor_id")

# Top-62 chunk per investor. Data_Cleaning.r writes an explicit chunk_id
# (1 = top of the book), so this is a filter rather than an inference;
# slice_head() would depend on file row order.
seqs <- if ("chunk_id" %in% names(seqs_all)) {
  seqs_all |> filter(chunk_id == 1)
} else {
  warning("no chunk_id column; falling back to file row order")
  seqs_all |> group_by(investor_id) |> slice_head(n = 1) |> ungroup()
}
stopifnot(nrow(seqs) == n_distinct(seqs_all$investor_id))

long_base <- seqs |>
  mutate(pos = map(tokens, ~ tibble(issuer_id = as.character(.x),
                                    rank = seq_along(.x)))) |>
  select(investor_id, n_assets_full, pos) |>
  unnest(pos) |>
  left_join(sec_ref, by = "issuer_id")

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

if (file.exists(FIT_CACHE)) {
  cat(sprintf("Loading cached mixture fits from %s\n", FIT_CACHE))
  cached <- readRDS(FIT_CACHE)
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
  saveRDS(list(fits = fits, sweep = sweep), FIT_CACHE)
  cat(sprintf("Cached mixture fits -> %s\n", FIT_CACHE))
}
best_k <- sweep$k_asked[which.min(sweep$BIC)]
cat(sprintf("\nBIC selects k = %d%s\n", best_k,
            if (best_k == max(K_GRID)) "  (GRID BOUNDARY - not an interior optimum)" else ""))

if (is.na(K_FINAL)) K_FINAL <- best_k
spc    <- fits[[as.character(K_FINAL)]]
labels <- clusters(spc)
cl_df  <- tibble(investor_id = meta$investor_id, cluster = labels)

rho <- as.numeric(parameters(spc)["rho", ])
sizes <- cl_df |> count(cluster, name = "size") |> mutate(rho = round(rho, 3))
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

cat(sprintf("Holdings: %s positions | %.1f%% of issuers named\n",
            format(nrow(long), big.mark = ","),
            100 * mean(!is.na(long$entity_proper_name))))

# ---- helpers ----------------------------------------------------------------
# lift = share within cluster / share overall. >1 over-represented.
lift_table <- function(data, var, min_n = 20, top = 3) {
  var <- rlang::ensym(var)
  overall <- data |> filter(!is.na(!!var)) |>
    count(!!var) |> mutate(p_all = n / sum(n)) |> select(-n)
  data |> filter(!is.na(!!var)) |>
    count(cluster, !!var) |>
    group_by(cluster) |> mutate(p_cl = n / sum(n)) |> ungroup() |>
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
    count(cluster, !!var) |>
    group_by(cluster) |>
    mutate(pct = 100 * n / sum(n)) |>
    slice_max(pct, n = 1, with_ties = FALSE) |>
    ungroup() |>
    transmute(cluster, !!label := as.character(!!var), pct = round(pct, 1))
}

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
breadth <- seqs |> distinct(investor_id, n_assets_full) |>
  inner_join(cl_df, by = "investor_id")
print(inv |>
        group_by(cluster) |>
        summarise(med_aum_musd = round(median(aum_any, na.rm = TRUE) / 1e6, 1),
                  pct_aum_known = round(100 * mean(!is.na(aum_any)), 1),
                  .groups = "drop") |>
        left_join(breadth |> group_by(cluster) |>
                    summarise(med_assets = median(n_assets_full),
                              pct_over_62 = round(100 * mean(n_assets_full > 62), 1),
                              .groups = "drop"), by = "cluster"), n = Inf)

# ========================= 3C. LENS C — WHAT THEY HOLD =======================
# Everything below operates on `long`, which is holdings joined to sec_ref, so
# country_desc / iso_country / region_code here are the ISSUER's.
cat("\n\n############ LENS C — WHAT THEY HOLD (segmentation checks) ############\n")

cat("\n=== Issuer country: top country per cluster ===\n")
print(top_share(long, country_desc, "top_country"), n = Inf)

cat("\n=== Issuer country (lift) ===\n")
print(lift_table(long, country_desc, min_n = 50, top = 3), n = Inf)

cat("\n=== Issuer region mix (% of positions) ===\n")
print(long |> filter(!is.na(region_code)) |>
        count(cluster, region_code) |>
        group_by(cluster) |> mutate(pct = round(100 * n / sum(n), 1)) |>
        ungroup() |> select(-n) |>
        pivot_wider(names_from = region_code, values_from = pct, values_fill = 0),
      n = Inf, width = Inf)

cat("\n=== Sector (lift) ===\n")
print(lift_table(long, factset_industry_desc, min_n = 50, top = 3), n = Inf)

cat("\n=== Market-cap group mix (% of positions) ===\n")
print(long |> filter(!is.na(cap_group)) |>
        count(cluster, cap_group) |>
        group_by(cluster) |> mutate(pct = round(100 * n / sum(n), 1)) |>
        ungroup() |> select(-n) |>
        pivot_wider(names_from = cap_group, values_from = pct, values_fill = 0),
      n = Inf, width = Inf)

cat("\n=== Median market cap of holdings (USD m) ===\n")
print(long |> group_by(cluster) |>
        summarise(med_mktcap_musd = round(median(market_cap, na.rm = TRUE) / 1e6, 0),
                  pct_known = round(100 * mean(!is.na(market_cap)), 1),
                  .groups = "drop"), n = Inf)

# distinctive issuers, WITH NAMES
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
#   categorical -> Cramer's V     (0 = independent, 1 = perfectly determined)
#   continuous  -> eta-squared    (share of variance between clusters)
# Both are in [0,1]. They are different statistics, so read the RANKING
# rather than comparing values across the two families too literally.
cat("\n\n############ WHAT EXPLAINS THE CLUSTERS? ############\n")

cramers_v <- function(x, y) {
  ok <- !is.na(x) & !is.na(y)
  if (sum(ok) < 50 || n_distinct(x[ok]) < 2) return(NA_real_)
  tab <- table(x[ok], y[ok])
  chi <- suppressWarnings(chisq.test(tab)$statistic)
  n <- sum(tab)
  sqrt(as.numeric(chi) / (n * (min(dim(tab)) - 1)))
}
eta_sq <- function(x, g) {
  ok <- !is.na(x) & !is.na(g)
  if (sum(ok) < 50) return(NA_real_)
  x <- x[ok]; g <- factor(g[ok])
  ss_tot <- sum((x - mean(x))^2)
  ss_b <- sum(tapply(x, g, function(v) length(v) * (mean(v) - mean(x))^2))
  if (ss_tot == 0) return(NA_real_) else ss_b / ss_tot
}

# Investor-level features. Everything computed in the summarise() below is a
# per-investor summary of the HOLDINGS; everything arriving through the join
# with `inv` is an attribute of the INVESTOR. Note that summarise() keeps only
# the columns it names, so issuer geography reaches this table only via
# pct_us and modal_issuer_country - hence both are constructed explicitly.
inv_feat <- long |>
  group_by(investor_id) |>
  summarise(pct_us      = mean(iso_country == "US", na.rm = TRUE),
            pct_top_sec = {t <- table(factset_industry_desc); if (length(t)) max(t)/sum(t) else NA_real_},
            med_cap     = median(market_cap, na.rm = TRUE),
            # categorical counterpart of pct_us: the single country in which
            # the investor holds most of its identified positions
            modal_issuer_country = {
              t <- table(country_desc)
              if (length(t)) names(t)[which.max(t)] else NA_character_
            },
            .groups = "drop") |>
  right_join(inv, by = "investor_id") |>
  left_join(breadth |> select(investor_id, n_assets_full), by = "investor_id")

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
         strength = map_dbl(cat_vars, ~ cramers_v(inv_feat[[.x]], inv_feat$cluster)),
         coverage = map_dbl(cat_vars, ~ mean(!is.na(inv_feat[[.x]])))),
  tibble(attribute = num_vars, kind = "continuous",
         strength = map_dbl(num_vars, ~ eta_sq(inv_feat[[.x]], inv_feat$cluster)),
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
print(assoc |> group_by(family) |>
        summarise(best_attribute = attribute[which.max(strength)],
                  strength = max(strength, na.rm = TRUE), .groups = "drop") |>
        arrange(desc(strength)), n = Inf)
cat("\nIf geography / artifact / vehicle outrank strategy, the clusters are\n",
    "segmentation. If strategy attributes lead, they are strategy groups.\n")

# ========================= 5. SAVE ===========================================
saveRDS(list(cluster = cl_df, k = K_FINAL, rho = rho, sweep = sweep,
             posterior = posterior(spc), seed = SEED, quarter = QUARTER,
             style_tab = style_tab, distinctive = distinctive, assoc = assoc),
        sprintf("cluster_assignment_%s.rds", OUT_TAG))
write.csv(assoc, sprintf("cluster_association_ranking_%s.csv", OUT_TAG),
          row.names = FALSE)
write.csv(distinctive, sprintf("cluster_distinctive_holdings_%s.csv", OUT_TAG),
          row.names = FALSE)
cat(sprintf("\nSaved: cluster_assignment_%s.rds + 2 csv files\n", OUT_TAG))