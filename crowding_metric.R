# =============================================================================
# crowding_metric.R
# Builds the crowding-metric time series from PS-BERT investor embeddings.
#
# For every quarter it:
#   1. loads the L2-normalized investor embeddings,
#   2. runs spherical k-means (skmeans, pclust, fixed K, seed 42),
#   3. computes the two crowding measures from your proposal:
#        - cluster-concentration HHI_t = sum_c s_{c,t}^2
#        - tightness crowding         = avg within-cluster cosine similarity
#   4. saves the per-quarter cluster assignments (for the later transition
#      matrices / dynamics), then frees memory.
#
# After the loop it flags "crowded" quarters (HHI above the historical 90th /
# 95th percentile) and writes a tidy time series + a plot.
#
# NOTE on the ONE thing you must check: the load_embeddings() function. I don't
# know the exact column layout of your embedding files, so I made a documented
# assumption and an easy override. Everything else is format-independent.
# =============================================================================

suppressMessages({
  library(arrow)     # parquet I/O  (read embeddings)
  library(skmeans)   # spherical k-means
})

# ----------------------------- CONFIG ---------------------------------------
EMB_DIR   <- "~/Master-s-Thesis/embeddings"   # folder of per-quarter embeddings
OUT_DIR   <- "~/Master-s-Thesis/crowding"     # outputs go here
ASG_DIR   <- file.path(OUT_DIR, "assignments")# per-quarter cluster assignments

K         <- 6        # <-- HOLD CONSTANT across quarters (see note below).
SEED      <- 42       # matches your single-quarter run
PCTLS     <- c(0.90, 0.95)   # crowded-quarter thresholds

# Quick run for the supervisor meeting: set to e.g. 20 to use only the most
# recent 20 quarters; set to NA (or 0) to run the full panel.
TEST_LAST_N <- 20

# Why K is fixed: HHI ranges in [1/K, 1]. Its floor depends on K, so a time
# series built with different K per quarter is NOT comparable. Pick the single
# K from your k-sweep and apply it to every quarter. 6 is your current choice.
# ----------------------------------------------------------------------------

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(ASG_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------- EMBEDDING LOADER  (CHECK THIS) ----------------------
# Returns: list(ids = <character vector>, X = <matrix, rows = investors,
#               unit-L2 rows>). X must have one row per investor.
#
# ASSUMPTION: each file is a parquet with one ID column and the remaining
# numeric columns being the embedding dimensions. If your embeddings already
# live behind your single-quarter clustering loader, just call that here
# instead of this body.
load_embeddings <- function(path) {
  df <- as.data.frame(arrow::read_parquet(path))

  # --- identify the ID column. Override `id_col` if yours is named differently.
  id_candidates <- c("factset_entity_id", "entity_id", "investor_id",
                     "investor", "id", "fundno", "fsym_id")
  id_col <- intersect(id_candidates, names(df))[1]
  if (is.na(id_col)) {
    # fall back to the first non-numeric column
    nonnum <- names(df)[!vapply(df, is.numeric, logical(1))]
    id_col <- if (length(nonnum)) nonnum[1] else NA
  }

  if (!is.na(id_col)) {
    ids <- as.character(df[[id_col]])
    Xdf <- df[, setdiff(names(df), id_col), drop = FALSE]
  } else {
    # no ID column at all -> use row index, all columns are embedding dims
    ids <- as.character(seq_len(nrow(df)))
    Xdf <- df
  }

  # keep only numeric embedding columns
  Xdf <- Xdf[, vapply(Xdf, is.numeric, logical(1)), drop = FALSE]
  X   <- as.matrix(Xdf)
  storage.mode(X) <- "double"

  # defensive re-normalization (cheap insurance; should already be unit-norm)
  nr <- sqrt(rowSums(X * X))
  nr[nr == 0] <- 1
  X <- X / nr

  list(ids = ids, X = X)
}
# ----------------------------------------------------------------------------

# ------------------------- CROWDING COMPUTATION -----------------------------
# Per-cluster tightness uses the exact O(n*d) identity (verified):
#   avg_{i!=j} <x_i, x_j> = (n * ||mean_i x_i||^2 - 1) / (n - 1)
# for unit-norm x_i, instead of the O(n^2 d) pairwise matrix.
crowding_from_clusters <- function(X, cl, K) {
  N      <- nrow(X)
  counts <- tabulate(cl, nbins = K)
  shares <- counts / N

  HHI      <- sum(shares^2)
  HHI_norm <- (HHI - 1 / K) / (1 - 1 / K)   # rescaled to [0,1]
  eff_n    <- 1 / HHI                         # effective # of clusters
  max_share <- max(shares)

  tight <- rep(NA_real_, K)
  for (c in seq_len(K)) {
    idx <- which(cl == c)
    n   <- length(idx)
    if (n >= 2) {
      m <- colMeans(X[idx, , drop = FALSE])   # raw mean of unit vectors
      tight[c] <- (n * sum(m * m) - 1) / (n - 1)
    }
  }
  # quarter-level aggregates
  w <- counts
  ok <- !is.na(tight)
  tight_wmean <- if (any(ok)) sum(tight[ok] * w[ok]) / sum(w[ok]) else NA_real_
  tight_maxcl <- tight[which.max(shares)]     # tightness of the biggest cluster

  list(HHI = HHI, HHI_norm = HHI_norm, eff_n = eff_n, max_share = max_share,
       tight_wmean = tight_wmean, tight_maxcl = tight_maxcl,
       shares = shares, tight = tight)
}
# ----------------------------------------------------------------------------

# --------------------------------- MAIN -------------------------------------
files <- list.files(path.expand(EMB_DIR), pattern = "\\.parquet$",
                    full.names = TRUE)
stopifnot(length(files) > 0)

# parse a YYYY-MM-DD date out of each filename for chronological ordering.
# (Your filenames may carry the known +1-day quarter-end roll, e.g. 2019-10-01
#  for 2019-Q3 -- harmless here, it only affects the label, not the ordering.)
dates <- as.Date(sub(".*?(\\d{4}-\\d{2}-\\d{2}).*", "\\1", basename(files)))
ord   <- order(dates)
files <- files[ord]; dates <- dates[ord]

if (!is.na(TEST_LAST_N) && TEST_LAST_N > 0 && TEST_LAST_N < length(files)) {
  keep  <- seq(length(files) - TEST_LAST_N + 1, length(files))
  files <- files[keep]; dates <- dates[keep]
  message(sprintf("TEST mode: using last %d quarters (%s ... %s)",
                  length(files), min(dates), max(dates)))
}

rows <- vector("list", length(files))

for (i in seq_along(files)) {
  f <- files[i]; d <- dates[i]
  message(sprintf("[%d/%d] %s", i, length(files), basename(f)))

  emb <- load_embeddings(f)
  X   <- emb$X; ids <- emb$ids
  N   <- nrow(X)

  if (N < K) {
    warning(sprintf("  skipped %s: only %d investors (< K=%d)", d, N, K))
    next
  }

  asg_file <- file.path(ASG_DIR, sprintf("asg_%s.rds", format(d, "%Y-%m-%d")))
  if (file.exists(asg_file)) {                 # crash-recovery: reuse if present
    cl <- readRDS(asg_file)$cluster
  } else {
    set.seed(SEED)
    km <- skmeans(X, k = K, method = "pclust")
    cl <- km$cluster
    saveRDS(list(ids = ids, cluster = cl, date = d), asg_file)
  }

  m <- crowding_from_clusters(X, cl, K)
  rows[[i]] <- data.frame(
    date        = d,
    n_investors = N,
    K           = K,
    HHI         = m$HHI,
    HHI_norm    = m$HHI_norm,
    eff_n_clust = m$eff_n,
    max_share   = m$max_share,
    tight_wmean = m$tight_wmean,
    tight_maxcl = m$tight_maxcl
  )

  rm(emb, X, ids, cl, m); gc(verbose = FALSE)
}

ts <- do.call(rbind, rows)
ts <- ts[order(ts$date), ]

# ---- crowded flags from FULL-SAMPLE percentiles (descriptive) --------------
thr <- quantile(ts$HHI, probs = PCTLS, na.rm = TRUE)
ts$crowded_90 <- ts$HHI >= thr[1]
ts$crowded_95 <- ts$HHI >= thr[2]

# For the PREDICTIVE test, swap the line above for an expanding-window
# threshold to avoid look-ahead, e.g.:
#   ts$crowded_90 <- mapply(function(i) ts$HHI[i] >= quantile(ts$HHI[1:i], .90),
#                           seq_len(nrow(ts)))

# ------------------------------- OUTPUTS ------------------------------------
csv_path <- file.path(OUT_DIR, "crowding_timeseries.csv")
write.csv(ts, csv_path, row.names = FALSE)

pdf_path <- file.path(OUT_DIR, "crowding_hhi.pdf")
pdf(pdf_path, width = 9, height = 5)
plot(ts$date, ts$HHI, type = "l", lwd = 2, col = "steelblue",
     xlab = "Quarter", ylab = expression(HHI[t]),
     main = sprintf("Strategy-crowding (HHI of cluster shares), K=%d", K))
abline(h = thr, lty = c(2, 3), col = c("orange", "red"))
crowded <- ts[ts$crowded_90, ]
points(crowded$date, crowded$HHI, pch = 19, col = "red")
legend("topleft", bty = "n", lty = c(1, 2, 3), pch = c(NA, NA, NA),
       col = c("steelblue", "orange", "red"),
       legend = c("HHI", sprintf("90th pct (%.4f)", thr[1]),
                  sprintf("95th pct (%.4f)", thr[2])))
dev.off()

cat("\n================  CROWDING SUMMARY  ================\n")
cat(sprintf("Quarters: %d   (%s ... %s)\n", nrow(ts), min(ts$date), max(ts$date)))
cat(sprintf("HHI: mean %.4f  min %.4f  max %.4f  (floor 1/K = %.4f)\n",
            mean(ts$HHI), min(ts$HHI), max(ts$HHI), 1/K))
cat(sprintf("Crowded (>=90th pct): %d quarters\n", sum(ts$crowded_90)))
print(utils::tail(ts[, c("date","n_investors","HHI","HHI_norm",
                         "tight_wmean","tight_maxcl","crowded_90")], 8),
      row.names = FALSE)
cat(sprintf("\nWrote: %s\n       %s\n       %s/  (per-quarter assignments)\n",
            csv_path, pdf_path, ASG_DIR))
