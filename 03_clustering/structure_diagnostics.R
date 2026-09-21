# =============================================================================
# Structure diagnostics for sec:clusters:structure
#   (1) Observed pairwise cosine-similarity distribution vs. the uniform null,
#       plus a one-component spherical Cauchy MLE on the full cross-section.
#   (2) Neighbourhood attribute homogeneity at m = 10 and m = 50 vs. base rate,
#       for style, investor domicile, and fund type.
#
# Written to be run in the SAME R session right after sourcing Clusters.r,
# so it can reuse whatever embedding matrix and joined investor table that
# script already built, rather than re-deriving the reference join from
# scratch and risking a mismatch with the 99.9%/99.0% figures Clusters.r
# already reports at the top of every run.
# =============================================================================

library(tidyverse)

# ---- Step 0: find what's already in this session --------------------------
# Run this first. We need two things: an n x 64 numeric matrix of unit-norm
# embeddings (n should be 32,389), and a data frame/tibble with investor_id
# plus style, domicile, and fund-type columns for the same investors.

cat("Objects in session, with class and dimensions:\n")
for (nm in ls(envir = .GlobalEnv)) {
  obj <- get(nm, envir = .GlobalEnv)
  dims <- tryCatch(paste(dim(obj), collapse = " x "), error = function(e) NA)
  cat(sprintf("  %-25s %-15s %s\n", nm, paste(class(obj), collapse = "/"), dims))
}

cat("\nLooking for a candidate embedding matrix (numeric, ~32389 rows, ~64 cols)...\n")
for (nm in ls(envir = .GlobalEnv)) {
  obj <- get(nm, envir = .GlobalEnv)
  if ((is.matrix(obj) || is.data.frame(obj)) && is.numeric(as.matrix(obj))) {
    d <- dim(obj)
    if (!is.null(d) && d[1] > 30000 && d[1] < 35000 && d[2] > 50 && d[2] < 100) {
      cat(sprintf("  candidate: %s (%d x %d)\n", nm, d[1], d[2]))
    }
  }
}

cat("\nLooking for a candidate investor-attribute table (has investor_id and a style/domicile/fund-type-like column)...\n")
for (nm in ls(envir = .GlobalEnv)) {
  obj <- get(nm, envir = .GlobalEnv)
  if (is.data.frame(obj)) {
    nms <- names(obj)
    hit <- any(grepl("investor_id", nms, ignore.case = TRUE)) &&
           any(grepl("style|domicile|country|fund_type", nms, ignore.case = TRUE))
    if (hit) cat(sprintf("  candidate: %s (%d rows) -- columns: %s\n",
                          nm, nrow(obj), paste(nms, collapse = ", ")))
  }
}

# -----------------------------------------------------------------------------
# STOP HERE the first time you run this. Paste back what the three blocks
# above print, then fill in EMB_OBJ / EMB_ID_COL / ATTR_OBJ / ATTR_ID_COL /
# STYLE_COL / DOMICILE_COL / FUND_TYPE_COL below to match what's actually
# there before running the rest.
# -----------------------------------------------------------------------------

ATTR_OBJ        <- "inv"          # already-derived, trusted attribute table
ATTR_ID_COL     <- "investor_id"
STYLE_COL       <- "style_any"
DOMICILE_COL    <- "inv_country_desc"
FUND_TYPE_COL   <- "fund_type_desc"

# =============================================================================
# FALLBACK: load from parquet directly if nothing suitable is already in the
# session (e.g. a fresh R restart). Adjust paths if these differ.
# =============================================================================

load_from_parquet_fallback <- function() {
  emb_path  <- "embeddings_weighted/q_2025-12-31__weighted_paper.parquet"
  attr_path <- "investors_master.parquet"

  emb_raw  <- arrow::read_parquet(emb_path)
  attr_raw <- arrow::read_parquet(attr_path)

  cat("\n--- embeddings parquet ---\n")
  cat("dim:", paste(dim(emb_raw), collapse = " x "), "\n")
  print(names(emb_raw)[1:min(10, ncol(emb_raw))])

  cat("\n--- investors_master parquet ---\n")
  cat("dim:", paste(dim(attr_raw), collapse = " x "), "\n")
  print(names(attr_raw))

  list(emb_raw = emb_raw, attr_raw = attr_raw)
}

# Uncomment if Step 0 found nothing usable:
# fb <- load_from_parquet_fallback()
# # then inspect fb$emb_raw / fb$attr_raw and adjust the column constants above

# =============================================================================
# Once EMB_OBJ / ATTR_OBJ / column names above are set correctly, run from here.
# =============================================================================

build_inputs <- function() {
  emb_path <- "embeddings_weighted/q_2025-12-31__weighted_paper.parquet"
  # ^ double underscore before "weighted_paper" -- matches the filename that
  # actually printed from a real Clusters.r run earlier in this session.
  # If your actual path uses a single underscore, fix this line before running.

  emb_raw <- arrow::read_parquet(emb_path)
  cat("Embeddings parquet columns:\n")
  print(names(emb_raw))
  cat("Embeddings parquet dim:", paste(dim(emb_raw), collapse = " x "), "\n")

  id_col_guess <- intersect(c("investor_id", "factset_entity_id", "id"), names(emb_raw))
  if (length(id_col_guess) == 0) {
    stop("Could not find an id column among ", paste(names(emb_raw), collapse = ", "),
         " -- inspect the printed column names above and tell me which one is the id.")
  }
  id_col <- id_col_guess[1]
  cat("Using id column:", id_col, "\n")

  num_cols <- names(emb_raw)[vapply(emb_raw, is.numeric, logical(1))]
  cat("Numeric columns found:", length(num_cols), "\n")
  if (length(num_cols) != 64) {
    cat("WARNING: expected 64 numeric embedding columns, found", length(num_cols),
        "-- STOP and inspect names(emb_raw) before trusting anything downstream.\n")
  }

  emb_ids <- emb_raw[[id_col]]
  X_raw <- as.matrix(emb_raw[, num_cols])
  storage.mode(X_raw) <- "double"

  # --- Align to the already-trusted `inv` object by explicit id match ------
  attr_obj <- get(ATTR_OBJ, envir = .GlobalEnv)
  ord <- match(attr_obj[[ATTR_ID_COL]], emb_ids)
  n_missing <- sum(is.na(ord))
  cat(sprintf("\n%d of %d %s investors matched to an embedding (%.2f%%)\n",
              sum(!is.na(ord)), nrow(attr_obj), ATTR_OBJ,
              100 * mean(!is.na(ord))))
  if (n_missing > 0) {
    cat(sprintf("Dropping %d unmatched rows before proceeding.\n", n_missing))
  }
  keep <- !is.na(ord)
  attrs <- attr_obj[keep, ]
  X <- X_raw[ord[keep], , drop = FALSE]

  stopifnot(nrow(X) == nrow(attrs))
  cat(sprintf("Final aligned set: %d investors, verified by id match (not assumed by position).\n",
              nrow(X)))

  # confirm unit-norm; renormalise defensively if not (should already be ~1)
  norms <- sqrt(rowSums(X^2))
  cat(sprintf("Row-norm check: median %.6f, range [%.6f, %.6f]\n",
              median(norms), min(norms), max(norms)))
  if (max(abs(norms - 1)) > 1e-6) {
    cat("Norms are not exactly 1 -- renormalising defensively.\n")
    X <- X / norms
  }

  list(X = X, attrs = attrs)
}

# =============================================================================
# Diagnostic 1: pairwise cosine similarity vs. uniform null on S^{63}
# =============================================================================
# Full n(n-1)/2 pairs (~524M for n=32,389) is unnecessary and memory-heavy for
# characterising a distribution — a large random sample of pairs gives
# effectively exact quantiles. Pairs are drawn from the full cross-section, so
# this still describes "the full cross-section," just not exhaustively.

diagnose_cosine_similarity <- function(X, n_pairs = 5e6, seed = 20260910) {
  set.seed(seed)
  n <- nrow(X)
  i1 <- sample.int(n, n_pairs, replace = TRUE)
  i2 <- sample.int(n, n_pairs, replace = TRUE)
  keep <- i1 != i2
  i1 <- i1[keep]; i2 <- i2[keep]

  cos_sim <- rowSums(X[i1, , drop = FALSE] * X[i2, , drop = FALSE])

  qs <- quantile(cos_sim, c(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))

  d <- ncol(X)
  null_sd <- 1 / sqrt(d)

  cat("\n=== Pairwise cosine similarity: observed vs. uniform null ===\n")
  cat(sprintf("Pairs sampled: %s (from n = %d)\n", format(length(cos_sim), big.mark = ","), n))
  cat(sprintf("Observed  mean = %.4f   sd = %.4f\n", mean(cos_sim), sd(cos_sim)))
  cat(sprintf("Null      mean = %.4f   sd = %.4f   (1/sqrt(d), d=%d)\n", 0, null_sd, d))
  cat("Quantiles (observed):\n")
  print(round(qs, 4))

  invisible(list(cos_sim = cos_sim, quantiles = qs, null_sd = null_sd))
}

# =============================================================================
# Diagnostic 2: one-component spherical Cauchy MLE (Kato & McCullagh, 2020)
# =============================================================================
# Density on S^{p-1}: f(x; gamma) = (1-||gamma||^2) / omega_{p-1} * ||x-gamma||^{-p}
# gamma = rho * mu lies in the open unit ball of R^p. Log-likelihood (dropping
# the additive constant -n*log(omega_{p-1}), irrelevant to the optimum):
#   ll(gamma) = n*log(1-||gamma||^2) - (p/2) * sum_i log(||x_i-gamma||^2)
# with analytic gradient
#   grad(gamma) = -2n*gamma/(1-||gamma||^2) + p * sum_i (x_i-gamma)/||x_i-gamma||^2
# Fit by gradient ascent with backtracking (Armijo) line search, starting from
# gamma = 0 (the uniform distribution).

.sc_loglik <- function(gamma, X, p) {
  norm_g2 <- sum(gamma^2)
  if (norm_g2 >= 1) return(-Inf)
  R  <- sweep(X, 2, gamma, "-")
  d2 <- rowSums(R^2)
  nrow(X) * log(1 - norm_g2) - (p / 2) * sum(log(d2))
}

.sc_grad <- function(gamma, X, p) {
  n <- nrow(X)
  norm_g2 <- sum(gamma^2)
  R  <- sweep(X, 2, gamma, "-")
  d2 <- rowSums(R^2)
  w  <- 1 / d2
  term2 <- colSums(R * w)
  -2 * n * gamma / (1 - norm_g2) + p * term2
}

fit_spherical_cauchy_1comp <- function(X, max_iter = 2000, tol = 1e-8,
                                        step0 = 1e-3, verbose = TRUE) {
  p <- ncol(X)
  gamma <- rep(0, p)
  ll_prev <- .sc_loglik(gamma, X, p)
  step <- step0

  for (iter in seq_len(max_iter)) {
    g <- .sc_grad(gamma, X, p)
    repeat {
      gamma_new <- gamma + step * g
      if (sum(gamma_new^2) < (1 - 1e-6)) {
        ll_new <- .sc_loglik(gamma_new, X, p)
        if (is.finite(ll_new) && ll_new > ll_prev) break
      }
      step <- step / 2
      if (step < 1e-12) break
    }
    if (step < 1e-12) {
      if (verbose) cat("Step size collapsed; stopping.\n")
      break
    }
    change <- ll_new - ll_prev
    gamma <- gamma_new
    ll_prev <- ll_new
    step <- step * 1.5

    if (verbose && iter %% 50 == 0) {
      cat(sprintf("iter %d: loglik=%.4f  rho=%.5f  step=%.2e\n",
                  iter, ll_prev, sqrt(sum(gamma^2)), step))
    }
    if (abs(change) < tol) {
      if (verbose) cat(sprintf("Converged at iter %d.\n", iter))
      break
    }
  }

  rho <- sqrt(sum(gamma^2))
  list(gamma = gamma, rho = rho, mu = gamma / rho, loglik = ll_prev, iterations = iter)
}

diagnose_one_component_rho <- function(X) {
  cat("\n=== One-component spherical Cauchy fit, full cross-section ===\n")
  fit <- fit_spherical_cauchy_1comp(X)
  cat(sprintf("rho = %.4f  (loglik = %.2f, %d iterations)\n",
              fit$rho, fit$loglik, fit$iterations))
  cat("Sanity check: this should be LOWER than the k=2 mixture's median\n")
  cat("component rho (0.192) — a single component over the whole\n")
  cat("heterogeneous population should look less concentrated, not more.\n")
  invisible(fit)
}

# =============================================================================
# Diagnostic 3: neighbourhood attribute homogeneity at m = 10, 50
# =============================================================================
# Cosine distance nearest-neighbours == Euclidean nearest-neighbours for
# unit-norm vectors (||x-y||^2 = 2 - 2*cos_sim(x,y)), so an exact Euclidean
# k-d tree search (RANN) gives exact cosine-nearest-neighbours, no
# approximation introduced.

diagnose_neighbourhood_homogeneity <- function(X, attrs, style_col, domicile_col,
                                                fund_type_col, m_values = c(10, 50)) {
  if (!requireNamespace("RANN", quietly = TRUE)) {
    stop("Install RANN first: install.packages('RANN')")
  }
  n <- nrow(X)
  max_m <- max(m_values)
  k_query <- max_m + 5  # buffer in case self isn't uniquely first

  nn <- RANN::nn2(data = X, query = X, k = k_query)
  nn_idx <- nn$nn.idx

  # strip the self-match from each row, keep the next max_m
  nn_idx_clean <- t(vapply(seq_len(n), function(i) {
    row <- nn_idx[i, ]
    row <- row[row != i]
    row[seq_len(max_m)]
  }, FUN.VALUE = integer(max_m)))

  base_rate <- function(v) {
    tab <- table(v, useNA = "no")
    p <- tab / sum(tab)
    sum(p^2)
  }

  homogeneity_at_m <- function(attr_vec, nn_mat, m) {
    neigh_idx  <- nn_mat[, seq_len(m), drop = FALSE]
    neigh_attr <- matrix(attr_vec[neigh_idx], nrow = nrow(neigh_idx), ncol = m)
    match_mat  <- neigh_attr == attr_vec  # NA where either side NA
    valid_mat  <- !is.na(neigh_attr)
    match_count <- rowSums(match_mat, na.rm = TRUE)
    valid_count <- rowSums(valid_mat)
    ratio <- ifelse(valid_count > 0, match_count / valid_count, NA_real_)
    keep <- !is.na(attr_vec) & !is.na(ratio)
    mean(ratio[keep])
  }

  cols <- list(style = style_col, domicile = domicile_col, fund_type = fund_type_col)

  cat("\n=== Neighbourhood attribute homogeneity vs. base rate ===\n")
  for (label in names(cols)) {
    col <- cols[[label]]
    v <- attrs[[col]]
    br <- base_rate(v)
    cat(sprintf("\n%s (column: %s, coverage %.1f%%, base rate %.4f)\n",
                label, col, 100 * mean(!is.na(v)), br))
    for (m in m_values) {
      h <- homogeneity_at_m(v, nn_idx_clean, m)
      cat(sprintf("  m=%-3d homogeneity = %.4f   (lift over base rate: %.2fx)\n",
                  m, h, h / br))
    }
  }

  invisible(nn_idx_clean)
}

# =============================================================================
# Run everything (once the constants above are filled in correctly)
# =============================================================================

inputs <- build_inputs()
X <- inputs$X
attrs <- inputs$attrs

diagnose_cosine_similarity(X)
diagnose_one_component_rho(X)
diagnose_neighbourhood_homogeneity(X, attrs, STYLE_COL, DOMICILE_COL, FUND_TYPE_COL)