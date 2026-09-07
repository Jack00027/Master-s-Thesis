# =============================================================================
# Cluster_Dynamics_v2.r — how investors move between clusters across quarters
#
# Supersedes Cluster_Dynamics.r. Fixes four bugs and adds the validation the
# first version lacked.
#
# FIXED
#   * `if (FIT_HERE) library(flexmix); library(circlus)` parsed as two
#     statements, loading circlus unconditionally
#   * lag() move detection ignored year adjacency, so an investor absent for
#     several years registered a one-year "move" across the whole gap
#   * mclust used but never declared
#   * alignment quality reported as a raw share with no chance baseline
#
# ADDED
#   * portfolio-overlap diagnostic: tests the sparsity condition that lets
#     Procrustes be fit on all common investors (Hamilton et al. 2016)
#   * orthogonal Procrustes with held-out validation, optional stable-anchor
#     restriction, and a direction assertion
#   * dual label matching (investor overlap AND aligned centroids) with an
#     agreement statistic — two criteria sharing no inputs
#   * assignment margin: best vs. second-best permutation cost
#   * anchored as well as sequential chaining
#
# Outputs -> dynamics/
# =============================================================================

library(arrow)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(clue)      # solve_LSAP (ships with skmeans)
library(mclust)    # adjustedRandIndex
library(flexmix)
library(circlus)

# ------------------------------ CONFIG ---------------------------------------
ARM          <- "weighted"
YEARS        <- 2005:2025
K_FIXED      <- 7
US_ONLY      <- TRUE
US_THRESHOLD <- 0.90
SEED         <- 42

N_RESTART    <- 5
MINPRIOR     <- 0.01
MIN_COMP     <- 150

FIT_HERE     <- TRUE
ANCHOR_YEARS <- c(2007, 2011, 2015, 2019, 2023, 2025)
POST_CONF    <- 0.70

# ---- alignment ----
# "sequential": year t aligned to already-aligned t-1. Rotations compose, so
#               error accumulates across 20 links.
# "anchored"  : every year aligned directly to REF_YEAR. No accumulation, but
#               needs enough common investors with a distant reference.
CHAIN_MODE   <- "sequential"
REF_YEAR     <- 2015          # used when CHAIN_MODE == "anchored"

# Restrict the Procrustes fit to investors whose portfolios barely changed.
# NULL = use all common investors (valid if the sparsity diagnostic in
# section 2 shows overlap concentrated near 1). A number in (0,1] = minimum
# Jaccard overlap to qualify as a stable anchor.
ANCHOR_JACCARD <- NULL

PROC_HOLDOUT <- 0.20          # share of anchors held out to validate rotation

SUFFIX <- if (US_ONLY) sprintf("_us%02d", round(100 * US_THRESHOLD)) else ""
dir.create("dynamics", showWarnings = FALSE)
stopifnot(all(ANCHOR_YEARS %in% YEARS), REF_YEAR %in% YEARS)

quarter_of <- function(y) sprintf("%d-12-31", y)
emb_path <- function(q) switch(
  ARM,
  baseline = sprintf("embeddings_v2/q_%s.parquet", q),
  weighted = sprintf("embeddings_weighted/q_%s__weighted_first-chunk.parquet", q),
  stop("ARM must be 'baseline' or 'weighted'"))

# ========================= 1. PER-YEAR FITS ==================================
# Now also stores X (embeddings) and mu (component centroids); both are needed
# for Procrustes and for centroid matching. Roughly 100 MB across 21 years.

us_investors <- function(q) {
  # Mirrors the US filter in Clusters.r. If yours differs in any detail,
  # copy it verbatim: a different subset silently breaks comparability with
  # the Chapter 5 results.
  sec <- read_parquet("reference/securities_master.parquet") |>
    select(issuer_id, iso_country)
  read_parquet(sprintf("data/q_%s.parquet", q)) |>
    filter(chunk_id == 1) |>
    select(investor_id, tokens) |>
    unnest_longer(tokens, values_to = "issuer_id") |>
    left_join(sec, by = "issuer_id") |>
    group_by(investor_id) |>
    summarise(us_share = mean(iso_country == "US", na.rm = TRUE),
              .groups = "drop") |>
    filter(us_share >= US_THRESHOLD) |>
    pull(investor_id)
}

fit_mixture <- function(X, K) {
  eff_minprior <- max(MINPRIOR, MIN_COMP / nrow(X))
  best <- NULL
  for (s in seq_len(N_RESTART)) {
    set.seed(SEED + s - 1)
    fit <- try(flexmix(X ~ 1, k = K, model = FLXMCspcauchy(),
                       control = list(minprior = eff_minprior)), silent = TRUE)
    if (inherits(fit, "try-error")) next
    if (!is.finite(as.numeric(logLik(fit)))) next
    if (length(unique(clusters(fit))) != K) next          # k-1 collapse
    if (is.null(best) || BIC(fit) < BIC(best)) best <- fit
  }
  best
}

# component centroids: unit-normalised mean direction of each cluster
centroids <- function(X, cl, K) {
  mu <- t(sapply(1:K, function(k) colMeans(X[cl == k, , drop = FALSE])))
  mu / sqrt(rowSums(mu^2))
}

fit_one_year <- function(y) {
  q  <- quarter_of(y)
  df <- read_parquet(emb_path(q))
  X  <- as.matrix(df[, grepl("^dim_", names(df))])
  X  <- X / sqrt(rowSums(X^2))
  ids <- df$investor_id

  if (US_ONLY) {
    sel <- ids %in% us_investors(q)
    X <- X[sel, , drop = FALSE]; ids <- ids[sel]
  }

  best <- fit_mixture(X, K_FIXED)
  if (is.null(best))
    stop(sprintf("year %d: no restart retained %d components", y, K_FIXED))

  cl   <- clusters(best)
  post <- posterior(best)
  list(year = y, investor_id = ids, X = X, cluster = cl,
       mu = centroids(X, cl, K_FIXED),
       max_post = apply(post, 1, max),
       rho = as.numeric(parameters(best)["rho", ]), bic = BIC(best))
}

load_one_year <- function(y) {
  f <- sprintf("cluster_assignment_%s_%s%s.rds", ARM, quarter_of(y), SUFFIX)
  r <- readRDS(f)
  q <- quarter_of(y)
  df <- read_parquet(emb_path(q))
  X  <- as.matrix(df[, grepl("^dim_", names(df))])
  X  <- X / sqrt(rowSums(X^2))
  rownames(X) <- df$investor_id
  ids <- r$cluster$investor_id
  stopifnot(all(ids %in% rownames(X)))        # posterior row order must match
  X <- X[ids, , drop = FALSE]
  cl <- r$cluster$cluster
  list(year = y, investor_id = ids, X = X, cluster = cl,
       mu = centroids(X, cl, K_FIXED),
       max_post = if (!is.null(r$posterior)) apply(r$posterior, 1, max) else NA_real_,
       rho = r$rho, bic = NA_real_)
}

cache <- sprintf("dynamics/yearly_fits_%s%s_k%d.rds", ARM, SUFFIX, K_FIXED)
if (file.exists(cache)) {
  yearly <- readRDS(cache)
  cat(sprintf("Loaded cached yearly fits <- %s\n", cache))
} else {
  yearly <- lapply(YEARS, function(y) {
    cat(sprintf("year %d ... ", y)); flush.console()
    out <- if (FIT_HERE) fit_one_year(y) else load_one_year(y)
    cat(sprintf("n = %s\n", format(length(out$investor_id), big.mark = ",")))
    out
  })
  names(yearly) <- as.character(YEARS)
  saveRDS(yearly, cache)
}

# ========================= 2. SPARSITY DIAGNOSTIC ============================
# Hamilton et al. align on the full vocabulary without selecting anchors. The
# justification is that semantic change is SPARSE: most words don't move, so
# the rotation is pinned down by the stable majority regardless of the
# drifters. The analogue here is that most investors' portfolios barely
# change year to year.
#
# If overlap is concentrated near 1 with a thin tail, align on everyone and
# cite Hamilton. If the mass is spread out, the rotation is being fit largely
# on movers and ANCHOR_JACCARD should be set.

holdings_sets <- function(q) {
  read_parquet(sprintf("data/q_%s.parquet", q)) |>
    filter(chunk_id == 1) |>
    select(investor_id, tokens) |>
    deframe() |> lapply(unique)
}

ov_cache <- "dynamics/portfolio_overlap.rds"
if (file.exists(ov_cache)) {
  overlap <- readRDS(ov_cache)
} else {
  cat("\nComputing year-over-year portfolio overlap ...\n")
  prev <- holdings_sets(quarter_of(YEARS[1]))
  overlap <- map_dfr(2:length(YEARS), function(t) {
    curr <- holdings_sets(quarter_of(YEARS[t]))
    ids  <- intersect(names(prev), names(curr))
    out  <- tibble(year_from = YEARS[t-1], year_to = YEARS[t], investor_id = ids,
                   jaccard = vapply(ids, function(i) {
                     a <- prev[[i]]; b <- curr[[i]]
                     length(intersect(a, b)) / length(union(a, b))
                   }, numeric(1)))
    prev <<- curr
    out
  })
  saveRDS(overlap, ov_cache)
}

cat("\n=== Portfolio overlap across consecutive years (sparsity check) ===\n")
print(overlap |> summarise(
  n = n(), q10 = quantile(jaccard, .10), median = median(jaccard),
  q90 = quantile(jaccard, .90), share_above_80 = mean(jaccard >= 0.80)) |>
  mutate(across(where(is.numeric), ~round(.x, 3))))
cat("If share_above_80 is high, sparsity holds and aligning on all common\n",
    "investors is defensible. If not, set ANCHOR_JACCARD.\n", sep = "")

ggsave("dynamics/portfolio_overlap.pdf",
       ggplot(overlap, aes(jaccard)) +
         geom_histogram(bins = 50) +
         labs(x = "Jaccard overlap of holdings, consecutive years", y = NULL) +
         theme_minimal(base_size = 10),
       width = 7, height = 4)

# ========================= 3. PROCRUSTES =====================================
# min_{R'R = I} || B R - A ||_F   with unit-norm rows, so this maximises the
# summed cosine between each investor's aligned source position and their
# reference position — matching the geometry used everywhere else.
#
# Schonemann (1966): M = B'A, svd(M) = U D V', R = U V'.
#
# CONVENTION, locked: M = t(B) %*% A pairs with R applied to B from the RIGHT.
# Reversing either alone gives R' — a rotation that preserves all within-year
# geometry perfectly (so every within-year diagnostic still passes) while
# destroying the cross-year comparison. It fails silently and looks like a
# finding. The held-out check below is what catches it.

procrustes_align <- function(X_src, ids_src, X_ref, ids_ref,
                             anchor_ids = NULL, holdout = PROC_HOLDOUT) {
  common <- intersect(ids_src, ids_ref)
  if (!is.null(anchor_ids)) common <- intersect(common, anchor_ids)
  if (length(common) < 100)
    warning(sprintf("only %d anchors — rotation unreliable", length(common)))

  B <- X_src[match(common, ids_src), , drop = FALSE]
  A <- X_ref[match(common, ids_ref), , drop = FALSE]

  set.seed(SEED)
  tr <- sample(length(common), max(1, floor((1 - holdout) * length(common))))
  te <- setdiff(seq_along(common), tr)

  sv <- svd(crossprod(B[tr, , drop = FALSE], A[tr, , drop = FALSE]))
  R  <- sv$u %*% t(sv$v)
  stopifnot(max(abs(crossprod(R) - diag(ncol(R)))) < 1e-8)   # orthogonality

  cos_before <- if (length(te)) mean(rowSums(B[te, , drop = FALSE] *
                                             A[te, , drop = FALSE])) else NA
  cos_after  <- if (length(te)) mean(rowSums((B[te, , drop = FALSE] %*% R) *
                                             A[te, , drop = FALSE])) else NA

  # Direction check. If the transpose convention were inverted this would
  # fail rather than pass quietly.
  if (!is.na(cos_after) && cos_after <= cos_before)
    warning("held-out cosine did not improve — check direction, or the two ",
            "spaces may not be related by a rotation at all")

  Y <- X_src %*% R
  list(R = R, X_aligned = Y / sqrt(rowSums(Y^2)),
       n_anchor = length(common),
       cos_before = cos_before, cos_after = cos_after)
}

# stable anchors, if requested
anchor_lookup <- if (is.null(ANCHOR_JACCARD)) NULL else
  overlap |> filter(jaccard >= ANCHOR_JACCARD) |>
  select(year_to, investor_id) |> group_by(year_to) |>
  summarise(ids = list(investor_id), .groups = "drop") |> deframe()

# ========================= 4. DUAL LABEL MATCHING ============================
# Two cost matrices that share no inputs:
#
#   overlap  — counts of co-membership. Alignment-free: computed from raw
#              labels, invariant to whatever rotation was applied. This
#              independence is its main virtue.
#   centroid — cosine between aligned component means. Needs the rotation,
#              and carries a circularity risk if the rotation was fit on
#              everyone rather than on stable anchors.
#
# Agreement between them is convergent evidence. Disagreement localises the
# year and component where the structure actually broke.

# margin between the best and the second-best permutation. If all centroids
# sit close together (anisotropy), the optimum wins by almost nothing and the
# permutation is essentially arbitrary.
assignment_margin <- function(M) {
  p1   <- solve_LSAP(M, maximum = TRUE)
  best <- sum(M[cbind(seq_len(nrow(M)), p1)])
  second <- -Inf
  for (i in seq_len(nrow(M))) {
    M2 <- M; M2[i, p1[i]] <- min(M) - 1        # forbid one matched edge
    p2 <- solve_LSAP(M2, maximum = TRUE)
    second <- max(second, sum(M[cbind(seq_len(nrow(M)), p2)]))
  }
  list(perm = p1, best = best, second = second,
       margin = (best - second) / best)
}

match_pair <- function(prev, curr, K, use_anchors = NULL) {
  # --- overlap cost ---
  common <- intersect(prev$investor_id, curr$investor_id)
  cl_p <- prev$cluster[match(common, prev$investor_id)]
  cl_c <- curr$cluster[match(common, curr$investor_id)]
  M_ovl <- matrix(as.numeric(table(factor(cl_p, 1:K), factor(cl_c, 1:K))), K, K)

  zero_rows <- which(rowSums(M_ovl) == 0)
  if (length(zero_rows))
    warning(sprintf("components %s have no common investors — their match is arbitrary",
                    paste(zero_rows, collapse = ", ")))

  a_ovl <- assignment_margin(M_ovl)

  # chance baseline for the matched share: what the marginals give for free
  s_p <- prop.table(table(factor(cl_p, 1:K)))
  s_c <- prop.table(table(factor(cl_c, 1:K)))
  matched <- a_ovl$best / sum(M_ovl)
  expected <- sum(s_p * s_c[a_ovl$perm])

  # --- centroid cost, via Procrustes ---
  pr <- procrustes_align(curr$X, curr$investor_id, prev$X, prev$investor_id,
                         anchor_ids = use_anchors)
  mu_c_aligned <- centroids(pr$X_aligned, curr$cluster, K)
  M_cen <- prev$mu %*% t(mu_c_aligned)
  a_cen <- assignment_margin(M_cen)

  relab <- integer(K); relab[a_ovl$perm] <- 1:K       # overlap is primary

  list(relab = relab,
       diag = tibble(
         n_common       = length(common),
         matched_share  = matched,
         expected_share = expected,
         excess_match   = matched - expected,
         margin_ovl     = a_ovl$margin,
         margin_cen     = a_cen$margin,
         perm_agree     = mean(a_ovl$perm == a_cen$perm),
         n_anchor       = pr$n_anchor,
         cos_before     = pr$cos_before,
         cos_after      = pr$cos_after))
}

cat("\n=== Label matching, per year-pair ===\n")
match_log <- tibble()
if (CHAIN_MODE == "sequential") {
  for (t in 2:length(yearly)) {
    ank <- if (is.null(anchor_lookup)) NULL else anchor_lookup[[as.character(YEARS[t])]]
    m <- match_pair(yearly[[t-1]], yearly[[t]], K_FIXED, ank)
    yearly[[t]]$cluster <- m$relab[yearly[[t]]$cluster]
    yearly[[t]]$mu <- centroids(yearly[[t]]$X, yearly[[t]]$cluster, K_FIXED)
    match_log <- bind_rows(match_log,
                           mutate(m$diag, from = YEARS[t-1], to = YEARS[t]))
  }
} else {
  ref <- yearly[[as.character(REF_YEAR)]]
  for (t in seq_along(yearly)) {
    if (YEARS[t] == REF_YEAR) next
    m <- match_pair(ref, yearly[[t]], K_FIXED, NULL)
    yearly[[t]]$cluster <- m$relab[yearly[[t]]$cluster]
    yearly[[t]]$mu <- centroids(yearly[[t]]$X, yearly[[t]]$cluster, K_FIXED)
    match_log <- bind_rows(match_log,
                           mutate(m$diag, from = REF_YEAR, to = YEARS[t]))
  }
}
print(match_log |> select(from, to, n_common, matched_share, excess_match,
                          margin_ovl, margin_cen, perm_agree,
                          cos_before, cos_after) |>
        mutate(across(where(is.numeric), ~round(.x, 3))), n = Inf)
write.csv(match_log, "dynamics/match_quality.csv", row.names = FALSE)

cat("\nRead this table before anything downstream:\n",
    "  excess_match near 0 -> the chain broke at that link\n",
    "  margin_* near 0     -> the permutation is essentially arbitrary\n",
    "  perm_agree = 1      -> overlap and centroid matching concur\n",
    "  cos_after <= cos_before -> rotation direction wrong, or the spaces\n",
    "                             are not related by a rotation\n", sep = "")

panel <- map_dfr(yearly, ~ tibble(investor_id = .x$investor_id, year = .x$year,
                                  cluster = .x$cluster, max_post = .x$max_post))
saveRDS(panel, "dynamics/cluster_panel.rds")

# ========================= 5. TRANSITIONS ====================================

trans_pair <- function(y_from, y_to, confident_only = FALSE) {
  a <- filter(panel, year == y_from); b <- filter(panel, year == y_to)
  if (confident_only) {
    a <- filter(a, max_post >= POST_CONF); b <- filter(b, max_post >= POST_CONF)
  }
  inner_join(select(a, investor_id, from = cluster),
             select(b, investor_id, to = cluster), by = "investor_id") |>
    count(from, to) |>
    complete(from = 1:K_FIXED, to = 1:K_FIXED, fill = list(n = 0)) |>
    group_by(from) |> mutate(p = n / sum(n)) |> ungroup() |>
    mutate(year_from = y_from, year_to = y_to)
}
transitions <- map2_dfr(YEARS[-length(YEARS)], YEARS[-1], trans_pair)
write.csv(transitions, "dynamics/transitions.csv", row.names = FALSE)

persistence_row <- function(y_from, y_to, confident_only = FALSE) {
  a <- filter(panel, year == y_from); b <- filter(panel, year == y_to)
  if (confident_only) {
    a <- filter(a, max_post >= POST_CONF); b <- filter(b, max_post >= POST_CONF)
  }
  j <- inner_join(select(a, investor_id, from = cluster),
                  select(b, investor_id, to = cluster), by = "investor_id")
  if (!nrow(j)) return(NULL)
  s_from <- prop.table(table(factor(j$from, 1:K_FIXED)))
  s_to   <- prop.table(table(factor(j$to,   1:K_FIXED)))
  tibble(year_from = y_from, year_to = y_to, n = nrow(j),
         persistence = mean(j$from == j$to),
         expected_random = sum(s_from * s_to),
         excess = mean(j$from == j$to) - sum(s_from * s_to),
         ari = adjustedRandIndex(j$from, j$to),
         confident_only = confident_only)
}
persistence <- bind_rows(
  map2_dfr(YEARS[-length(YEARS)], YEARS[-1], persistence_row, confident_only = FALSE),
  map2_dfr(YEARS[-length(YEARS)], YEARS[-1], persistence_row, confident_only = TRUE))
write.csv(persistence, "dynamics/persistence.csv", row.names = FALSE)

cat("\n=== Persistence (all / confident) ===\n")
print(persistence |> mutate(across(where(is.numeric), ~round(.x, 3))), n = Inf)

ari_decay <- map_dfr(1:10, function(h) {
  map_dfr(YEARS[YEARS + h <= max(YEARS)], function(y0) {
    j <- inner_join(filter(panel, year == y0)   |> select(investor_id, a = cluster),
                    filter(panel, year == y0+h) |> select(investor_id, b = cluster),
                    by = "investor_id")
    if (nrow(j) < 100) return(NULL)
    tibble(h = h, year0 = y0, n = nrow(j),
           ari = adjustedRandIndex(j$a, j$b), persistence = mean(j$a == j$b))
  })
})
write.csv(ari_decay, "dynamics/ari_decay.csv", row.names = FALSE)
cat("\n=== ARI decay ===\n")
print(ari_decay |> group_by(h) |>
        summarise(mean_ari = round(mean(ari), 3),
                  mean_persistence = round(mean(persistence), 3),
                  n_pairs = n(), .groups = "drop"), n = Inf)

# ---- who moves (adjacency-corrected) ----------------------------------------
# v1 compared each row to the previous ROW rather than the previous YEAR, so
# an investor absent for several years registered a single one-year "move"
# spanning the whole gap. Only consecutive-year pairs count.

switchers <- panel |>
  arrange(investor_id, year) |>
  group_by(investor_id) |>
  mutate(prev_year = lag(year), prev_cluster = lag(cluster),
         prev_post = lag(max_post)) |>
  ungroup() |>
  filter(!is.na(prev_year), year - prev_year == 1) |>
  mutate(moved = cluster != prev_cluster)

cat(sprintf("\nConsecutive-year observations: %s (v1 also counted %s gap-spanning pairs)\n",
            format(nrow(switchers), big.mark = ","),
            format(sum(!is.na(lag(panel$year))) - nrow(switchers), big.mark = ",")))

cat("\n=== Move rate by posterior confidence of the origin assignment ===\n")
print(switchers |>
        mutate(post_bin = cut(prev_post, c(0, .5, .7, .9, 1),
                              labels = c("<.5", ".5-.7", ".7-.9", ">.9"))) |>
        filter(!is.na(post_bin)) |> group_by(post_bin) |>
        summarise(n = n(), move_rate = round(mean(moved), 3), .groups = "drop"),
      n = Inf)

inv_ref <- read_parquet("reference/investors_master.parquet") |>
  select(investor_id, any_of(c("fund_type_desc", "style_any", "turnover_any",
                               "aum_any", "inv_country_desc")))
cat("\n=== Move rate by fund type ===\n")
print(switchers |> left_join(inv_ref, by = "investor_id") |>
        filter(!is.na(fund_type_desc)) |> group_by(fund_type_desc) |>
        summarise(n = n(), move_rate = round(mean(moved), 3), .groups = "drop") |>
        filter(n >= 200) |> arrange(move_rate), n = Inf)

flows <- map2_dfr(YEARS[-length(YEARS)], YEARS[-1], function(y0, y1) {
  a <- filter(panel, year == y0); b <- filter(panel, year == y1)
  tibble(year_from = y0, year_to = y1, n_from = nrow(a), n_to = nrow(b),
         exit_rate = round(mean(!a$investor_id %in% b$investor_id), 3),
         entry_rate = round(mean(!b$investor_id %in% a$investor_id), 3))
})
cat("\n=== Entry / exit (attrition is NOT movement) ===\n")
print(flows, n = Inf)
write.csv(flows, "dynamics/entry_exit.csv", row.names = FALSE)

# ========================= 6. PLOTS ==========================================

ggsave("dynamics/transition_heatmaps.pdf",
       transitions |> mutate(pair = sprintf("%d\u2192%d", year_from, year_to)) |>
         ggplot(aes(factor(to), factor(from), fill = p)) +
         geom_tile() +
         scale_fill_gradient(low = "white", high = "grey15", limits = c(0, 1),
                             name = "P(to | from)") +
         scale_y_discrete(limits = rev) + facet_wrap(~ pair) +
         labs(x = "cluster in t+1", y = "cluster in t") +
         theme_minimal(base_size = 8) + theme(panel.grid = element_blank()),
       width = 11, height = 9)

ggsave("dynamics/cluster_shares.pdf",
       panel |> count(year, cluster) |> group_by(year) |>
         mutate(share = n / sum(n)) |> ungroup() |>
         ggplot(aes(year, share, fill = factor(cluster))) + geom_area() +
         labs(x = NULL, y = "share of investors", fill = "cluster") +
         theme_minimal(base_size = 10),
       width = 8, height = 4.5)

ggsave("dynamics/persistence.pdf",
       persistence |> filter(!confident_only) |>
         select(year_to, persistence, expected_random) |> pivot_longer(-year_to) |>
         ggplot(aes(year_to, value, linetype = name)) +
         geom_line() + geom_point(size = 1) +
         scale_linetype_manual(values = c(persistence = "solid",
                                          expected_random = "dashed"), name = NULL) +
         labs(x = NULL, y = "share in the same cluster") +
         theme_minimal(base_size = 10),
       width = 8, height = 4.5)

ggsave("dynamics/ari_decay.pdf",
       ari_decay |> group_by(h) |> summarise(ari = mean(ari), .groups = "drop") |>
         ggplot(aes(h, ari)) + geom_line() + geom_point() +
         scale_x_continuous(breaks = 1:10) +
         labs(x = "horizon (years)", y = "mean adjusted Rand index") +
         theme_minimal(base_size = 10),
       width = 7, height = 4)

# match quality over time — the plot that says whether to trust the rest
ggsave("dynamics/match_quality.pdf",
       match_log |> select(to, excess_match, margin_ovl, perm_agree) |>
         pivot_longer(-to) |>
         ggplot(aes(to, value)) + geom_line() + geom_point(size = 1) +
         facet_wrap(~ name, ncol = 1, scales = "free_y") +
         labs(x = NULL, y = NULL) + theme_minimal(base_size = 9),
       width = 8, height = 6)

if (requireNamespace("ggalluvial", quietly = TRUE)) {
  library(ggalluvial)
  bal <- panel |> filter(year %in% ANCHOR_YEARS) |> count(investor_id) |>
    filter(n == length(ANCHOR_YEARS)) |> pull(investor_id)
  cat(sprintf("\nAlluvial: %s investors present in all %d anchor years\n",
              format(length(bal), big.mark = ","), length(ANCHOR_YEARS)))
  ggsave("dynamics/alluvial.pdf",
         panel |> filter(year %in% ANCHOR_YEARS, investor_id %in% bal) |>
           ggplot(aes(x = factor(year), stratum = factor(cluster),
                      alluvium = investor_id, fill = factor(cluster))) +
           geom_flow(alpha = 0.45, width = 0.3) +
           geom_stratum(width = 0.3, colour = "white", linewidth = 0.3) +
           labs(x = NULL, y = "investors", fill = "cluster") +
           theme_minimal(base_size = 10),
         width = 9, height = 5.5)
}

cat("\nDone. Outputs in dynamics/\n")
