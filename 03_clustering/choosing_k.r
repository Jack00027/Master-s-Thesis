# =============================================================================
# choosing_k.r — fast K selection for the investor-embedding clustering
#
# WHY THIS SCRIPT EXISTS
#   BIC cannot select k for a spherical Cauchy mixture on these embeddings:
#   the penalty (d+1)log n is ~675 against likelihood gains of ~60,000 per
#   component, so BIC is effectively pure log-likelihood, monotone in k, and
#   returns whatever the grid boundary happens to be. Retention (how many
#   restarts keep k components) also fails on the global sample, where every
#   restart keeps every k because minprior is a share and 1% of 32,389 is a
#   324-investor floor that nothing falls below.
#
#   That leaves interpretability, which is a judgement and slow to apply.
#   This script supplies quantitative evidence to constrain that judgement,
#   cheaply, by running the SAME criterion under TWO different models:
#
#     A. spherical k-means (skmeans) on the full sample
#     B. the spherical Cauchy mixture on a fixed random subsample
#
#   The criterion is subsample stability (mean pairwise adjusted Rand index),
#   which uses no likelihood penalty and is therefore unaffected by the
#   arithmetic that breaks BIC. Agreement between two model families is
#   stronger evidence than either alone; disagreement localises the problem.
#
# WHAT IT DOES NOT DO
#   It does not choose k for you and it does not fit the reported model.
#   Run Clusters.r with K_RULE <- "fixed" at the two or three k this script
#   shortlists, and read the Lens A-D tables there. Interpretability remains
#   the deciding criterion; this narrows the field it is applied to.
#
# WHY IT IS FAST
#   No holdings pipeline. Clusters.r rebuilds the top-CONTEXT_WINDOW position
#   table on every source() — ~65,000 list-column tibbles unnested — before
#   EM starts, and none of it is needed to choose k. This script touches the
#   embedding parquet only.
#
# OUTPUT
#   choosing_k_<arm>_<quarter>.csv   one row per k, all diagnostics
#   choosing_k_<arm>_<quarter>.rds   the same plus the fitted label vectors
# =============================================================================

library(arrow)
library(dplyr)
library(tibble)
library(purrr)
library(flexmix)
library(circlus)
library(skmeans)
library(mclust)      # adjustedRandIndex

# ------------------------------ CONFIG ---------------------------------------
QUARTER <- "2025-12-31"
ARM     <- "weighted"          # "baseline" or "weighted"

EMB_FILE <- switch(
  ARM,
  baseline = sprintf("embeddings_v3/q_%s__mean_paper.parquet", QUARTER),
  weighted = sprintf("embeddings_weighted/q_%s__weighted_paper.parquet", QUARTER),
  stop("ARM must be 'baseline' or 'weighted'"))

K_GRID <- 2:20                 # wide, because the point is to find where the
                               # evidence stops improving, not to confirm a guess

# ---- subsample for the mixture scan ----------------------------------------
# 5,000 of 32,389. EM cost is roughly linear in n, so this is a ~6.5x speedup
# per fit. The subsample is drawn ONCE and reused at every k, so differences
# across k are not confounded with differences in who was sampled.
SUB_N <- 5000

# ---- restarts --------------------------------------------------------------
# The weighted arm has a demonstrably rough likelihood surface: on the full
# sample the 4->5 BIC step gained 28k where 3->4 and 5->6 gained ~145k and
# ~141k, i.e. a bad local optimum surviving best-of-eight. Single-restart
# scanning would turn that from a visible anomaly at one k into an invisible
# coin flip at every k. Four is affordable on a 5,000-row subsample.
N_RESTART_SCAN <- 4

# ---- stability -------------------------------------------------------------
# B subsample fits per k, all pairs compared by ARI on shared members:
# B = 10 gives 45 pairs, enough to place the mean given within-k standard
# deviations of ~0.11-0.15 observed on the US-only panel.
#
# nruns/restarts inside the stability fits are deliberately 1. The curve is
# meant to reflect how reproducible the PROCEDURE is, and best-of-n restarting
# suppresses exactly the initialisation variance it should be measuring.
STAB_B    <- 10
STAB_FRAC <- 0.80

SEED <- 42

# ---- component floor -------------------------------------------------------
# minprior is a SHARE, and the share is the right invariant to hold fixed when
# subsampling: a component holding 1% of 5,000 corresponds to a component
# holding 1% of 32,389, so a k whose components clear the floor on the
# subsample will clear it at full n.
#
# The competing consideration is numerical: components of a few dozen points
# collapse (rho -> 1) and return a non-finite likelihood. At n = 5,000 a 1%
# floor admits 50 members, which is below the ~150 at which collapse was first
# observed. Rather than raise the floor and break comparability, the floor is
# left at the share and collapses are COUNTED and reported — a k that collapses
# often is unstable regardless of its stability score, which is itself a
# finding. Raise MINPRIOR_SCAN only if collapses dominate the low-k end.
MINPRIOR_SCAN <- 0.01

# ---- skmeans cross-check (optional) ----------------------------------------
# Off by default: the stability sweep at full n is 19 x STAB_B fits on 32,389
# rows and dominates the runtime. Turn it on when the cross-method comparison
# is wanted — it also discharges the spherical k-means \todo in Section 5.10.
#
# SKM_FULL controls where it runs. TRUE uses the full sample, so the two scans
# differ in model rather than in sample size and a disagreement is
# interpretable; FALSE uses the same subsample as the mixture, which is far
# cheaper but confounds model with n if the mixture scan is ever moved.
RUN_SKMEANS <- TRUE
SKM_FULL    <- FALSE
SKM_NRUNS   <- 3               # restarts per skmeans fit (main sweep)

OUT_CSV <- sprintf("choosing_k_%s_%s.csv", ARM, QUARTER)
OUT_RDS <- sprintf("choosing_k_%s_%s.rds", ARM, QUARTER)

# ========================= 1. LOAD EMBEDDINGS ================================
cat(sprintf("choosing_k | quarter %s | arm %s\n%s\n\n", QUARTER, ARM, EMB_FILE))
stopifnot(file.exists(EMB_FILE))

df <- read_parquet(EMB_FILE)
X  <- as.matrix(df[, grepl("^dim_", names(df))])
if (!ncol(X)) stop("no dim_* columns in ", EMB_FILE)
if (anyDuplicated(df$investor_id))
  stop(sprintf("%s has %d duplicated investor_id rows",
               EMB_FILE, sum(duplicated(df$investor_id))))
nrm <- sqrt(rowSums(X^2))
if (any(!is.finite(nrm)) || any(nrm == 0))
  stop(sprintf("%d embedding row(s) have zero or non-finite norm",
               sum(!is.finite(nrm) | nrm == 0)))
X <- X / nrm                                   # unit length (cosine geometry)

n_full <- nrow(X); d_emb <- ncol(X)
cat(sprintf("n = %s investors, d = %d\n", format(n_full, big.mark = ","), d_emb))

# Fixed subsample, drawn once, reused at every k.
set.seed(SEED)
sub_idx <- sort(sample(n_full, min(SUB_N, n_full)))
Xs <- X[sub_idx, , drop = FALSE]
n_sub <- nrow(Xs)
cat(sprintf("mixture scan on a fixed subsample of %s (%.1f%% of the quarter)\n",
            format(n_sub, big.mark = ","), 100 * n_sub / n_full))
cat(sprintf("minprior %.3f -> smallest admissible component: %d on the subsample, %d at full n\n\n",
            MINPRIOR_SCAN, ceiling(MINPRIOR_SCAN * n_sub),
            ceiling(MINPRIOR_SCAN * n_full)))

# ========================= 2. HELPERS ========================================

# Mean pairwise ARI across B fits on overlapping subsamples of `mat`.
# `fitter` takes a matrix and k, and returns an integer label vector or NULL.
stability <- function(mat, k, fitter, B = STAB_B, frac = STAB_FRAC, seed0) {
  parts <- vector("list", B)
  for (b in seq_len(B)) {
    set.seed(seed0 + 7919L * k + b)
    idx <- sample(nrow(mat), floor(frac * nrow(mat)))
    lab <- fitter(mat[idx, , drop = FALSE], k)
    if (!is.null(lab)) parts[[b]] <- setNames(lab, idx)
  }
  parts <- Filter(Negate(is.null), parts)
  if (length(parts) < 2) return(c(mean = NA_real_, sd = NA_real_, pairs = 0))
  aris <- c()
  for (i in seq_len(length(parts) - 1L)) for (j in (i + 1L):length(parts)) {
    shared <- intersect(names(parts[[i]]), names(parts[[j]]))
    if (length(shared) < 100) next
    aris <- c(aris, adjustedRandIndex(parts[[i]][shared], parts[[j]][shared]))
  }
  if (!length(aris)) return(c(mean = NA_real_, sd = NA_real_, pairs = 0))
  c(mean = mean(aris), sd = stats::sd(aris), pairs = length(aris))
}

skm_fit <- function(mat, k, nruns = 1) {
  f <- try(skmeans(mat, k, control = list(nruns = nruns)), silent = TRUE)
  if (inherits(f, "try-error")) NULL else as.integer(f$cluster)
}

# Best-of-n-restarts spherical Cauchy mixture. Returns the fit plus the
# restart bookkeeping, since collapses are a diagnostic in their own right.
mix_fit <- function(mat, k, n_restart, minprior, seed0) {
  best <- NULL; n_ok <- 0L; n_collapse <- 0L; n_fail <- 0L
  for (s in seq_len(n_restart)) {
    set.seed(seed0 + 1000L * k + s)
    f <- try(flexmix(mat ~ 1, k = k, model = FLXMCspcauchy(),
                     control = list(minprior = minprior)), silent = TRUE)
    if (inherits(f, "try-error") || !is.finite(as.numeric(logLik(f)))) {
      n_fail <- n_fail + 1L; next
    }
    if (length(unique(clusters(f))) != k) { n_collapse <- n_collapse + 1L; next }
    n_ok <- n_ok + 1L
    if (is.null(best) || BIC(f) < BIC(best)) best <- f
  }
  list(fit = best, n_ok = n_ok, n_collapse = n_collapse, n_fail = n_fail)
}

mix_labels_only <- function(mat, k) {
  set.seed(sample.int(.Machine$integer.max, 1))   # seeded by caller
  f <- try(flexmix(mat ~ 1, k = k, model = FLXMCspcauchy(),
                   control = list(minprior = MINPRIOR_SCAN)), silent = TRUE)
  if (inherits(f, "try-error") || !is.finite(as.numeric(logLik(f)))) NULL
  else as.integer(clusters(f))
}

# ========================= 3. SKMEANS SWEEP (FULL SAMPLE) ====================
# skmeans has no likelihood and therefore no BIC. Its own objective (total
# within-cluster cosine dissimilarity) is monotone decreasing in k for the same
# reason BIC is, so it is recorded but NOT used to select. Selection evidence
# comes from stability, which is the criterion shared with the mixture scan.
cat("=== A. skmeans sweep ===\n")
skm_rows <- tibble(k = integer())
if (!RUN_SKMEANS) {
  cat("  skipped (RUN_SKMEANS = FALSE)\n\n")
} else {
Xk <- if (SKM_FULL) X else Xs
n_skm <- nrow(Xk)
cat(sprintf("  on %s rows\n", format(n_skm, big.mark = ",")))
t0 <- Sys.time()
skm_rows <- map_dfr(K_GRID, function(k) {
  set.seed(SEED + k)
  f <- try(skmeans(Xk, k, control = list(nruns = SKM_NRUNS)), silent = TRUE)
  if (inherits(f, "try-error")) {
    cat(sprintf("  k=%2d  skmeans failed\n", k))
    return(tibble(k = k))
  }
  lab   <- as.integer(f$cluster)
  sizes <- as.integer(table(lab))
  st    <- stability(Xk, k, function(m, kk) skm_fit(m, kk, nruns = 1), seed0 = SEED)
  cat(sprintf("  k=%2d  ARI %.3f (sd %.3f)  min comp %5d (%.2f%%)  obj %.4f\n",
              k, st["mean"], st["sd"], min(sizes),
              100 * min(sizes) / n_skm, f$value))
  tibble(k = k, skm_ari = st["mean"], skm_ari_sd = st["sd"],
         skm_min_size = min(sizes),
         skm_min_share = min(sizes) / n_skm,
         skm_objective = f$value,
         skm_labels = list(lab))
})
cat(sprintf("  [%.1f min]\n\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}

# ========================= 4. MIXTURE SCAN (SUBSAMPLE) =======================
# The model actually reported, at reduced n. Records what the full-sample
# sweep records (BIC, retention) plus the two diagnostics that survive when
# those fail: the fitted concentration profile and the smallest component.
cat(sprintf("=== B. spherical Cauchy mixture (subsample n = %s) ===\n",
            format(n_sub, big.mark = ",")))
t0 <- Sys.time()
mix_rows <- map_dfr(K_GRID, function(k) {
  r <- mix_fit(Xs, k, N_RESTART_SCAN, MINPRIOR_SCAN, SEED)
  if (is.null(r$fit)) {
    cat(sprintf("  k=%2d  no restart retained %d components (%d collapsed, %d failed)\n",
                k, k, r$n_collapse, r$n_fail))
    return(tibble(k = k, mix_kept = 0L, mix_collapse = r$n_collapse,
                  mix_fail = r$n_fail))
  }
  lab   <- as.integer(clusters(r$fit))
  rho   <- as.numeric(parameters(r$fit)["rho", ])
  sizes <- as.integer(table(lab))
  st    <- stability(Xs, k, function(m, kk) {
                        set.seed(SEED + 104729L * kk + nrow(m))
                        mix_labels_only(m, kk)
                      }, seed0 = SEED)
  cat(sprintf(paste0("  k=%2d  ARI %.3f (sd %.3f)  BIC %10.0f  ",
                     "min comp %4d (%.2f%%)  rho>0.5: %2d/%2d  med rho %.3f  %d/%d ok\n"),
              k, st["mean"], st["sd"], BIC(r$fit), min(sizes),
              100 * min(sizes) / n_sub, sum(rho > 0.5), k, median(rho),
              r$n_ok, N_RESTART_SCAN))
  tibble(k = k, mix_ari = st["mean"], mix_ari_sd = st["sd"],
         mix_bic = BIC(r$fit), mix_loglik = as.numeric(logLik(r$fit)),
         mix_kept = r$n_ok, mix_collapse = r$n_collapse, mix_fail = r$n_fail,
         mix_min_size = min(sizes), mix_min_share = min(sizes) / n_sub,
         mix_rho_med = median(rho), mix_rho_max = max(rho),
         mix_rho_gt50 = sum(rho > 0.5),
         mix_labels = list(lab))
})
cat(sprintf("  [%.1f min]\n\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# ========================= 5. CROSS-METHOD AGREEMENT =========================
# skmeans restricted to the same subsample, compared to the mixture partition
# at matched k. High agreement means the structure is recoverable under two
# different geometries and is not an artifact of the Cauchy tails; low
# agreement at a k the stability curves both like is worth investigating
# before that k is reported.
#
# CAVEAT for the write-up: skmeans partitions hard and has no concentration
# parameter, and geometric criteria structurally favour hard clustering. This
# is corroboration, not a second opinion of equal standing.
cat("=== C. cross-method agreement (same subsample, matched k) ===\n")
cross <- tibble(k = integer())
if (!RUN_SKMEANS) {
  cat("  skipped (RUN_SKMEANS = FALSE)\n\n")
} else {
cross <- map_dfr(K_GRID, function(k) {
  mlab <- mix_rows$mix_labels[mix_rows$k == k]
  if (!length(mlab) || is.null(mlab[[1]])) return(tibble(k = k))
  set.seed(SEED + 31L + k)
  slab <- skm_fit(Xs, k, nruns = SKM_NRUNS)
  if (is.null(slab)) return(tibble(k = k))
  a <- adjustedRandIndex(mlab[[1]], slab)
  cat(sprintf("  k=%2d  ARI(mixture, skmeans) = %.3f\n", k, a))
  tibble(k = k, cross_ari = a)
})
cat("\n")
}

# ========================= 6. SUMMARY ========================================
res <- mix_rows |> select(-any_of("mix_labels"))
if (nrow(skm_rows))
  res <- res |> full_join(skm_rows |> select(-any_of("skm_labels")), by = "k")
if (nrow(cross))
  res <- res |> full_join(cross, by = "k")
res <- res |> arrange(k)

cat("=== SUMMARY ===\n")
print(res |> select(any_of(c("k", "skm_ari", "mix_ari", "cross_ari", "mix_bic",
                             "mix_rho_med", "mix_rho_gt50", "mix_min_share",
                             "mix_kept"))) |>
        mutate(across(where(is.numeric), ~ round(.x, 3))), n = Inf, width = Inf)

# ---- reading aids, stated rather than left to the eye ----------------------
# k = 2 and 3 are excluded from every argmax below. A coarse split of any point
# cloud reproduces almost perfectly across subsamples because the pieces are far
# apart relative to sampling noise, so ARI is near 1 at k = 2 whatever the data.
# An unrestricted argmax returns 2 mechanically; the criterion is rewarding
# coarseness, not structure. The full curve is printed above so the excluded
# values stay auditable.
KMIN <- 4
elig <- res |> filter(k >= KMIN)

pick <- function(col) {
  if (!col %in% names(elig)) return(NA_integer_)
  v <- elig[[col]]
  if (all(is.na(v))) return(NA_integer_)
  elig$k[which.max(v)]
}
cat(sprintf("\nArgmax over k >= %d:  mixture %s%s\n", KMIN, pick("mix_ari"),
            if ("skm_ari" %in% names(elig))
              sprintf(" | skmeans %s", pick("skm_ari")) else ""))

flat <- function(col, sdcol) {
  if (!all(c(col, sdcol) %in% names(elig))) return(NA)
  v <- elig[[col]]; s <- elig[[sdcol]]
  if (all(is.na(v))) return(NA)
  diff(range(v, na.rm = TRUE)) < median(s, na.rm = TRUE)
}
if (isTRUE(flat("skm_ari", "skm_ari_sd")))
  cat("[warn] skmeans: between-k differences are smaller than the within-k spread.\n",
      "       Read as 'no k in range is distinguishable', not as a selection.\n", sep = "")
if (isTRUE(flat("mix_ari", "mix_ari_sd")))
  cat("[warn] mixture: between-k differences are smaller than the within-k spread.\n",
      "       Read as 'no k in range is distinguishable', not as a selection.\n", sep = "")

# BIC is recorded to demonstrate that it cannot select, not to select with.
if (sum(!is.na(res$mix_bic)) > 1) {
  db  <- diff(res$mix_bic[!is.na(res$mix_bic)])
  pen <- (d_emb + 1) * log(n_sub)
  cat(sprintf("\nBIC on the subsample: penalty/component %.0f, median |gain| %.0f, ratio %.1fx%s\n",
              pen, median(abs(db)), median(abs(db)) / pen,
              if (all(db < 0)) " — falls at every step, no interior minimum" else ""))
}

# The floor is the constraint that actually binds as k grows: once the smallest
# component approaches minprior, further components are being limited by the
# floor rather than by the data, and any component sitting just above it is at
# risk of disappearing at the next k. The index-tracker component has run
# 401 -> 356 -> 352 against a 324 floor at full n, so this is not hypothetical.
near <- elig |> filter(!is.na(mix_min_share), mix_min_share < 1.5 * MINPRIOR_SCAN)
if (nrow(near))
  cat(sprintf("\n[note] smallest component within 1.5x the floor from k = %d upward\n",
              min(near$k)))

saveRDS(list(res = res, skm = skm_rows, mix = mix_rows, cross = cross,
             sub_idx = sub_idx, investor_id = df$investor_id,
             quarter = QUARTER, arm = ARM, emb_file = EMB_FILE,
             k_grid = K_GRID, sub_n = n_sub, n_full = n_full,
             n_restart = N_RESTART_SCAN, stab_b = STAB_B,
             minprior = MINPRIOR_SCAN, seed = SEED), OUT_RDS)
write.csv(res, OUT_CSV, row.names = FALSE)
cat(sprintf("\nSaved: %s and %s\n", OUT_CSV, OUT_RDS))
cat("\nNext: run Clusters.r with K_RULE <- \"fixed\" at the two or three k\n",
    "this shortlists, and decide on the Lens A-D tables.\n", sep = "")