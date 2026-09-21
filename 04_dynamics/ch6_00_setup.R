# =============================================================================
# ch6_00_setup.R -- shared configuration and helpers for Chapter 6
#
# Chapter 6 applies the diachronic-embedding method of Hamilton, Leskovec &
# Jurafsky (2016) to the per-year PS-BERT models (weighted arm, Q4 of each
# year):
#
#   Hamilton et al.                         this thesis
#   ---------------------------------------  ----------------------------------
#   one embedding model per decade           one PS-BERT model per year
#   orthogonal Procrustes between periods    orthogonal Procrustes between
#                                            years, anchored on investors whose
#                                            holdings did not change
#   semantic displacement of a word          displacement of an investor,
#                                            validated against its holdings
#                                            turnover
#
# Scripts, in order (run from the repository root):
#   ch6_01_align.R              holdings stability per year pair, the three
#                               anchor rules, their evaluation, rotations
#   ch6_02_stability_profile.R  who is stable (one pair, default 2024 -> 2025)
#   ch6_03_displacement.R       displacement and its link to turnover
#   ch6_04_components.R         component persistence and reallocation timing
#
# Every script starts with
#   if (!exists("CH6_SETUP_LOADED")) source("ch6_00_setup.R")
# =============================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(clue)       # solve_LSAP (Hungarian method, Kuhn 1955)
})

# ------------------------------- CONFIG --------------------------------------
UNIVERSE     <- Sys.getenv("UNIVERSE", "global")  # "global" | "us90"
US_THRESHOLD <- 0.90
YEARS        <- 2005:2025                         # fourth quarter of each year
REF_YEAR     <- 2025                              # = Chapter 5 cross-section
K            <- 8L
CTX          <- 62L                               # encoder context window
SEED         <- 42L

# -- anchors -------------------------------------------------------------------
# Holdings stability of an investor between two quarters is the weighted
# overlap of its top-62 slices, sum_a min(w0_a, w1_a). Anchors are selected on
# holdings only, never on embeddings.
RULES          <- c("stable_thr", "stable_strat", "all")
ANCHOR_RULE    <- "stable_thr"   # default; the other two are robustness series
STAB_THRESHOLD <- 0.85           # stable_thr: wt_overlap >= this; also defines
                                 # the stable held-out set and "stayers"
STAB_TOP_SHARE <- 0.25           # stable_strat: top quarter within each stratum
MIN_ANCHORS    <- 200L           # stable_thr falls back to the MIN_ANCHORS most
                                 # stable investors if the threshold leaves fewer
                                 # (flagged in the diagnostics)
HOLDOUT        <- 0.20           # share of investors held out for evaluation
N_PERM_NULL    <- 20L            # permutations for the shuffled-anchor null

# -- displacement ----------------------------------------------------------------
N_TURNOVER_BINS <- 10L           # turnover deciles for the binned relationship

# -- components ---------------------------------------------------------------
MINPRIOR     <- if (UNIVERSE == "us90") 0.03 else 0   # as in Chapter 5
N_RESTART    <- 2L
POST_CONF    <- 0.70
CRISIS_STEPS <- c(2008, 2020)    # year_to of steps flagged in figures

# -- optional noise floor: the SAME quarter retrained with a DIFFERENT seed ---
# Without it, displacement and churn have no reference level
# (Dubossarsky et al. 2017).
NOISE_YEAR     <- REF_YEAR
NOISE_EMB_FILE <- NULL   # e.g. "embeddings_weighted_seed7/q_2025-12-31__weighted_paper.parquet"

# -- file layout ---------------------------------------------------------------
EMB_DIR    <- "embeddings_weighted"
TAG        <- "__weighted_paper"
SEQ_DIR    <- "data"
SEC_MASTER      <- "reference/securities_master.parquet"
SEC_ID_COL      <- "issuer_id"
SEC_COUNTRY_COL <- "iso_country"
INV_MASTER      <- "reference/investors_master.parquet"

SUFFIX <- if (UNIVERSE == "us90") sprintf("_us%02d", round(100 * US_THRESHOLD)) else ""
# Chapter 5 labels -- CHECK this name against what Clusters.r writes
CH5_ASSIGN_FILE <- sprintf("cluster_assignment_weighted_%d-12-31%s.rds", REF_YEAR, SUFFIX)
REF_FIT_FILE    <- sprintf("mixture_fit_weighted_%d-12-31_k%d%s.rds", REF_YEAR, K, SUFFIX)
                             # the Chapter 5 flexmix fit (Clusters.r)

OUT_DIR   <- file.path("dynamics_hamilton", paste0("weighted", SUFFIX))
CACHE_DIR <- file.path("dynamics_hamilton", "cache")
dir.create(OUT_DIR,   recursive = TRUE, showWarnings = FALSE)
dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)

stopifnot(REF_YEAR %in% YEARS, ANCHOR_RULE %in% RULES)

# ------------------------------- paths ---------------------------------------
qdate    <- function(y) sprintf("%d-12-31", y)
emb_path <- function(y) file.path(EMB_DIR, sprintf("q_%s%s.parquet", qdate(y), TAG))
seq_path <- function(y) file.path(SEQ_DIR, sprintf("q_%s.parquet", qdate(y)))
out_file <- function(...) file.path(OUT_DIR, paste0(...))

msg <- function(...) { cat(sprintf(...), "\n", sep = ""); flush.console() }

# ------------------------------- linear algebra -----------------------------
unit_rows <- function(X) X / sqrt(rowSums(X^2))

# Orthogonal Procrustes, row-vector convention:
#   R = argmin_{R'R = I} || B R - A ||_F ,  B = rows in the earlier year,
#   A = the same investors in the later year;  B'A = U D V'  ->  R = U V'
# X_earlier %*% R lands in the later year's frame. Reflections are allowed.
# With unit rows this maximises the summed cosine between anchor pairs.
procrustes <- function(B, A) {
  sv <- svd(crossprod(B, A))
  sv$u %*% t(sv$v)
}

# R_list[[t]] maps YEARS[t] -> YEARS[t+1]; the chain from index i to j > i is
# the product in time order
compose_chain <- function(R_list, from_idx, to_idx) {
  R <- diag(ncol(R_list[[1]]))
  if (from_idx < to_idx)
    for (t in from_idx:(to_idx - 1)) R <- R %*% R_list[[t]]
  R
}

# Effective number of directions spanned by a set of anchors: participation
# ratio of the squared singular values (64 = isotropic, 1 = a single point)
eff_rank <- function(B) {
  e <- svd(B, nu = 0, nv = 0)$d^2
  sum(e)^2 / sum(e^2)
}

# spherical k-means, used to stratify stable_strat anchors in year pairs that
# do not end in the Chapter 5 reference quarter
sph_kmeans <- function(X, k, iter = 50, seed = SEED) {
  set.seed(seed)
  mu <- X[sample(nrow(X), k), , drop = FALSE]
  cl <- rep(0L, nrow(X))
  for (it in seq_len(iter)) {
    new <- max.col(X %*% t(mu), "first")
    if (all(new == cl)) break
    cl <- new
    S  <- as.matrix(crossprod(sparseMatrix(i = seq_along(cl), j = cl, x = 1,
                                           dims = c(nrow(X), k)), X))
    empty <- rowSums(S^2) == 0
    if (any(empty)) S[empty, ] <- X[sample(nrow(X), sum(empty)), ]
    mu <- unit_rows(S)
  }
  cl
}

# ------------------------------- matching ------------------------------------
# Hungarian assignment with a margin: how much the optimum beats the best
# assignment that avoids at least one of its edges. solve_LSAP needs
# non-negative input, so the matrix is shifted.
hungarian <- function(M) {
  stopifnot(nrow(M) <= ncol(M))
  M0 <- M - min(M) + 1
  p  <- as.integer(solve_LSAP(M0, maximum = TRUE))
  val <- function(pp) sum(M[cbind(seq_len(nrow(M)), pp)])
  best <- val(p); second <- -Inf
  for (i in seq_len(nrow(M))) {
    M2 <- M0; M2[i, p[i]] <- 0
    second <- max(second, val(as.integer(solve_LSAP(M2, maximum = TRUE))))
  }
  list(perm = p, best = best, second = second, margin = best - second)
}

ari <- function(a, b) mclust::adjustedRandIndex(a, b)

wilson <- function(x, n, z = 1.96) {
  p <- x / n; den <- 1 + z^2 / n
  c(lo = (p + z^2 / (2 * n) - z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / den,
    hi = (p + z^2 / (2 * n) + z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / den)
}

# ------------------------------- spherical Cauchy ----------------------------
# Log density w.r.t. the uniform distribution on S^{d-1} (thesis eq. 3)
spcauchy_logdens <- function(X, mu, rho) {
  d <- ncol(X)
  (d - 1) * (log1p(-rho^2) - log1p(rho^2 - 2 * rho * drop(X %*% mu)))
}

# par = list(mu = K x d matrix (unit rows), rho = K vector, pi = K vector)
spcauchy_posterior <- function(X, par) {
  L <- sapply(seq_along(par$rho), function(j)
    log(par$pi[j]) + spcauchy_logdens(X, par$mu[j, ], par$rho[j]))
  L <- matrix(L, nrow(X))
  m <- apply(L, 1, max)
  lse <- m + log(rowSums(exp(L - m)))
  list(post = exp(L - lse), loglik = lse)
}

# ------------------------------- I/O -----------------------------------------
read_vectors <- function(path) {
  df   <- as.data.frame(arrow::read_parquet(path))
  dims <- grep("^dim_", names(df), value = TRUE)
  X <- as.matrix(df[, dims]); storage.mode(X) <- "double"
  ids <- as.character(df$investor_id)
  if (anyDuplicated(ids)) stop(path, ": duplicated investor_id")
  ok <- is.finite(rowSums(X)) & rowSums(X^2) > 0
  X <- unit_rows(X[ok, , drop = FALSE]); rownames(X) <- ids[ok]
  X
}
load_embeddings <- function(y) read_vectors(emb_path(y))

# Holdings of one year, in long form (cached).
#   top    : largest CTX positions per investor (chunks joined in chunk_id
#            order before slicing); weights renormalised within the slice
#   n_full : positions in the whole portfolio
load_holdings <- function(y) {
  cache <- file.path(CACHE_DIR, sprintf("holdings_%d.rds", y))
  if (file.exists(cache)) return(readRDS(cache))
  df <- as.data.frame(arrow::read_parquet(
    seq_path(y), col_select = c("investor_id", "chunk_id", "tokens", "weights")))
  out <- holdings_from_chunks(df)
  saveRDS(out, cache)
  out
}

holdings_from_chunks <- function(df) {
  len <- lengths(df$tokens)
  long <- data.frame(
    investor_id = rep(as.character(df$investor_id), len),
    chunk_id    = rep(as.integer(df$chunk_id), len),
    pos         = sequence(len),
    issuer_id   = as.character(unlist(lapply(df$tokens, as.character), use.names = FALSE)),
    weight      = as.numeric(unlist(lapply(df$weights, as.numeric), use.names = FALSE)),
    stringsAsFactors = FALSE)
  g <- match(long$investor_id, unique(long$investor_id))
  long <- long[order(g, long$chunk_id, long$pos), ]
  r <- rle(long$investor_id)
  long$rank <- sequence(r$lengths)

  n_full <- setNames(r$lengths, r$values)

  top <- long[long$rank <= CTX, c("investor_id", "issuer_id", "weight")]
  top$weight <- top$weight / ave(top$weight, top$investor_id, FUN = sum)

  list(top = top, n_full = n_full)
}

# investors whose top-62 slice is >= US_THRESHOLD US issuers (unweighted, over
# positions with a known domicile) -- the Section 5.7 definition
universe_ids <- function(y, H = load_holdings(y)) {
  ids <- unique(H$top$investor_id)
  if (UNIVERSE == "global") return(ids)
  sec <- as.data.frame(arrow::read_parquet(SEC_MASTER,
                         col_select = c(SEC_ID_COL, SEC_COUNTRY_COL)))
  ctry <- sec[[SEC_COUNTRY_COL]][match(H$top$issuer_id, as.character(sec[[SEC_ID_COL]]))]
  ok <- !is.na(ctry)
  s <- tapply(ctry[ok] == "US", H$top$investor_id[ok], mean)
  intersect(ids, names(s)[s >= US_THRESHOLD])
}

# ------------------------------- holdings stability --------------------------
# wt_overlap = sum_a min(w0_a, w1_a) over the two top-62 slices
#            = 1 - (share of the slice that would have to be traded)
# Duplicate issuers within a slice (several share classes) are summed.
slice_weights <- function(top) {
  k <- paste(top$investor_id, top$issuer_id, sep = "\r")
  u <- !duplicated(k)
  w <- rowsum(top$weight, k, reorder = FALSE)
  data.frame(key = k[u], investor_id = top$investor_id[u],
             w = as.numeric(w), stringsAsFactors = FALSE)
}
wt_overlap <- function(top0, top1, ids) {
  d0 <- slice_weights(top0[top0$investor_id %in% ids, ])
  d1 <- slice_weights(top1[top1$investor_id %in% ids, ])
  i1 <- match(d0$key, d1$key)
  m  <- ifelse(is.na(i1), 0, pmin(d0$w, d1$w[i1]))
  setNames(as.numeric(tapply(m, d0$investor_id, sum)[ids]), ids)
}

# cached per pair
pair_stability <- function(y0, y1, ids, H0 = load_holdings(y0), H1 = load_holdings(y1)) {
  cache <- file.path(CACHE_DIR, sprintf("wt_overlap_%d_%d.rds", y0, y1))
  if (file.exists(cache)) {
    s <- readRDS(cache)
    if (all(ids %in% names(s))) return(s[ids])
  }
  s <- wt_overlap(H0$top, H1$top, ids)
  saveRDS(s, cache)
  s
}

have_gg <- requireNamespace("ggplot2", quietly = TRUE)

CH6_SETUP_LOADED <- TRUE
msg("[ch6] weighted arm | universe = %s | years %d-%d | ref %d | anchors = %s",
    UNIVERSE, min(YEARS), max(YEARS), REF_YEAR, ANCHOR_RULE)