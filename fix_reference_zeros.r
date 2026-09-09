# =============================================================================
# fix_reference_zeros.r
#
# FactSet writes absent records as 0 rather than NULL in two places:
#
#   securities_master.parquet   market_cap == 0 for share classes that have no
#                               market cap (mutual funds, trusts). These enter
#                               med_cap -- the top-ranked attribute in the
#                               association table -- as a solid block at the
#                               bottom of the rank distribution.
#
#   investors_master.parquet    the seven portfolio characteristics are all
#                               exactly 0 for funds with no characteristic
#                               record. Column-wise na_if() would be wrong here
#                               (dividend_yield == 0 is legitimate for a fund
#                               holding non-payers), so the test is row-wise:
#                               null the row only when EVERY characteristic is
#                               simultaneously 0.
#
# Rewrites both parquets in place after taking a one-time backup. Idempotent:
# a second run finds nothing to change and says so. Does NOT require
# Investor_data.r to be re-run.
#
# Usage:  DRY_RUN <- TRUE to inspect the counts without writing.
# =============================================================================

library(arrow)
library(dplyr)

DRY_RUN <- FALSE

INV_REF <- "reference/investors_master.parquet"
SEC_REF <- "reference/securities_master.parquet"

CHAR_VARS <- c("beta", "pe_ratio", "pb_ratio", "dividend_yield",
               "sales_growth", "price_momentum", "relative_strength")

# Back up once. If a backup already exists it is the pre-fix original and must
# not be overwritten by an already-fixed file on a second run.
backup_once <- function(path) {
  bak <- sub("\\.parquet$", "_prezero.bak.parquet", path)
  if (!file.exists(bak)) {
    file.copy(path, bak)
    cat(sprintf("  backup -> %s\n", bak))
  } else {
    cat(sprintf("  backup already exists, left alone (%s)\n", bak))
  }
  invisible(bak)
}

# ---- 1. securities_master: market_cap ---------------------------------------

cat("\n=== securities_master.parquet ===\n")
sec <- read_parquet(SEC_REF)

if (!"market_cap" %in% names(sec)) {
  stop("no market_cap column in ", SEC_REF)
}

n_zero <- sum(sec$market_cap == 0, na.rm = TRUE)
n_na0  <- sum(is.na(sec$market_cap))

cat(sprintf("  rows                     : %s\n", format(nrow(sec), big.mark = ",")))
cat(sprintf("  market_cap == 0          : %s (%.2f%%)\n",
            format(n_zero, big.mark = ","), 100 * n_zero / nrow(sec)))
cat(sprintf("  market_cap already NA    : %s (%.2f%%)\n",
            format(n_na0, big.mark = ","), 100 * n_na0 / nrow(sec)))

# Consistency check: cap_group should already be NA wherever market_cap is 0.
# If it is not, the two columns disagree and that needs looking at before the
# cap-group mix table can be trusted.
if ("cap_group" %in% names(sec) && n_zero > 0) {
  n_bad <- sum(sec$market_cap == 0 & !is.na(sec$cap_group), na.rm = TRUE)
  cat(sprintf("  of those, cap_group non-NA: %s%s\n",
              format(n_bad, big.mark = ","),
              if (n_bad > 0) "   <-- columns disagree, inspect" else ""))
}

if (n_zero > 0 && "factset_industry_desc" %in% names(sec)) {
  cat("\n  industries of the zero-market_cap issuers (top 5):\n")
  sec |>
    filter(market_cap == 0) |>
    count(factset_industry_desc, sort = TRUE) |>
    head(5) |>
    as.data.frame() |>
    print(row.names = FALSE)
}

if (n_zero > 0) {
  if (DRY_RUN) {
    cat("\n  [dry run] would set market_cap 0 -> NA and rewrite\n")
  } else {
    backup_once(SEC_REF)
    sec$market_cap[which(sec$market_cap == 0)] <- NA_real_
    write_parquet(sec, SEC_REF)
    cat("  rewritten\n")
  }
} else {
  cat("\n  nothing to do (already fixed)\n")
}

# ---- 2. investors_master: characteristic block ------------------------------

cat("\n=== investors_master.parquet ===\n")
inv <- read_parquet(INV_REF)

present <- intersect(CHAR_VARS, names(inv))
missing <- setdiff(CHAR_VARS, names(inv))
if (length(missing))
  cat(sprintf("  [note] not in this file, skipped: %s\n",
              paste(missing, collapse = ", ")))
if (!length(present)) stop("none of the characteristic columns are in ", INV_REF)

cat(sprintf("  rows                     : %s\n", format(nrow(inv), big.mark = ",")))

# Per-column zero counts, so the legitimate zeros that this fix deliberately
# LEAVES ALONE are visible rather than silently swept up.
cat("\n  zeros per column (only the all-zero ROWS are nulled):\n")
for (v in present)
  cat(sprintf("    %-18s %s\n", v,
              format(sum(inv[[v]] == 0, na.rm = TRUE), big.mark = ",")))

# A row qualifies only if every characteristic is non-NA and exactly 0.
M <- as.matrix(inv[present])
all_zero <- rowSums(M == 0, na.rm = TRUE) == length(present) &
            rowSums(is.na(M))            == 0
n_rows <- sum(all_zero)

cat(sprintf("\n  rows with ALL %d characteristics == 0 : %s (%.2f%%)\n",
            length(present), format(n_rows, big.mark = ","),
            100 * n_rows / nrow(inv)))

if (n_rows > 0 && "mgr_entity_proper_name" %in% names(inv)) {
  cat("\n  managers of those investors (top 5):\n")
  inv[all_zero, ] |>
    count(mgr_entity_proper_name, sort = TRUE) |>
    head(5) |>
    as.data.frame() |>
    print(row.names = FALSE)
}

if (n_rows > 0) {
  if (DRY_RUN) {
    cat("\n  [dry run] would null those rows across all characteristics\n")
  } else {
    backup_once(INV_REF)
    for (v in present) inv[[v]][all_zero] <- NA_real_
    write_parquet(inv, INV_REF)
    cat("  rewritten\n")
  }
} else {
  cat("\n  nothing to do (already fixed)\n")
}

cat("\nDone. Re-run Clusters.r; the mixture fit is unaffected, so the cached\n",
    ".rds is still valid and only the descriptive tables change.\n", sep = "")
