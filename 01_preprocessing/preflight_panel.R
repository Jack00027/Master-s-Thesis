#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## preflight_panel.R -- verify every quarterly sequence file before a long sweep.
##
## Run from ~/Master-s-Thesis (login node is fine; CPU only):
##
##     Rscript preflight_panel.R
##     Rscript preflight_panel.R --dir data --quiet
##
## Checks, per file:
##   * required columns present. weights and chunk_id are what the weighted arm
##     needs; without them BERT_training_weighted.py falls back to rank-decay
##     weights silently, and the run looks fine but measures the wrong thing.
##   * weights aligned element-wise with tokens
##   * weights sum to 1 per investor (they are normalised across the whole
##     portfolio, then split across chunks)
##   * chunk_id contiguous from 1 per investor, so chunk order is recoverable
##     rather than inferred from row order
##   * filename date is a real quarter end, catching any residue of the
##     seq.Date roll-forward bug
##
## Exits 1 if anything fails, so it can gate a submission:
##
##     Rscript preflight_panel.R && sbatch --export=ALL,ARM=weighted 05_train_all.sh
## ---------------------------------------------------------------------------

suppressPackageStartupMessages(library(arrow))

args    <- commandArgs(trailingOnly = TRUE)
get_opt <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1L]
}
DATA_DIR <- get_opt("--dir", "data")
QUIET    <- "--quiet" %in% args

REQUIRED    <- c("investor_id", "tokens", "weights", "chunk_id", "n_tokens")
VALID_ENDS  <- c("03-31", "06-30", "09-30", "12-31")
TOL         <- 1e-6

files <- sort(list.files(DATA_DIR, pattern = "^q_.*\\.parquet$", full.names = TRUE))
if (length(files) == 0L)
  stop("No files matched ", file.path(DATA_DIR, "q_*.parquet"),
       " -- run from ~/Master-s-Thesis", call. = FALSE)

cat("Checking", length(files), "quarterly files in", DATA_DIR, "\n\n")
problems <- character(0)
note <- function(...) problems <<- c(problems, paste0(...))

for (path in files) {

  nm <- basename(path)

  ## --- filename encodes a genuine quarter end ------------------------------
  m <- regmatches(nm, regexec("^q_(\\d{4})-(\\d{2}-\\d{2})\\.parquet$", nm))[[1]]
  if (length(m) == 0L) {
    note(nm, ": filename does not parse")
    next
  }
  if (!m[3] %in% VALID_ENDS)
    note(nm, ": ", m[3], " is not a quarter end (date-labelling bug?)")

  ## --- readable ------------------------------------------------------------
  df <- tryCatch(read_parquet(path), error = function(e) e)
  if (inherits(df, "error")) {
    note(nm, ": unreadable (", conditionMessage(df), ")")
    next
  }

  missing <- setdiff(REQUIRED, names(df))
  if (length(missing)) {
    note(nm, ": missing column(s) ", paste(missing, collapse = ", "))
    next
  }

  ## --- weights aligned with tokens, element for element --------------------
  bad_len <- sum(lengths(df$weights) != lengths(df$tokens))
  if (bad_len > 0L)
    note(nm, ": ", bad_len, " row(s) where weights and tokens differ in length")

  ## --- weights sum to 1 per investor ---------------------------------------
  per_chunk <- vapply(df$weights, function(w) sum(as.numeric(w)), numeric(1))
  sums      <- tapply(per_chunk, df$investor_id, sum)
  off       <- max(abs(sums - 1))
  if (!is.finite(off)) {
    note(nm, ": non-finite weight sums")
  } else if (off > TOL) {
    worst <- names(sums)[which.max(abs(sums - 1))]
    note(nm, ": max |sum(weights)-1| = ", format(off, digits = 3),
         " (worst investor ", worst, ")")
  }

  ## --- chunk_id contiguous from 1 within each investor ---------------------
  ok_chunks <- tapply(df$chunk_id, df$investor_id,
                      function(z) identical(sort(as.integer(z)), seq_along(z)))
  n_bad <- sum(!unlist(ok_chunks))
  if (n_bad > 0L)
    note(nm, ": ", n_bad, " investor(s) with non-contiguous chunk_id")

  if (!QUIET)
    cat(sprintf("  %-22s %7s investors  %8s chunks  wmax_err=%.1e\n",
                nm,
                format(length(unique(df$investor_id)), big.mark = ","),
                format(nrow(df), big.mark = ","),
                if (is.finite(off)) off else NaN))
}

cat("\n")
if (length(problems)) {
  cat(length(problems), "PROBLEM(S):\n")
  cat(paste0("  - ", problems, collapse = "\n"), "\n")
  quit(status = 1L)
}
cat("All", length(files), "files OK.\n")
