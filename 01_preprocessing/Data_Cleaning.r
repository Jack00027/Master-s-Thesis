## ================================================================
# PS-BERT portfolio sequences from FactSet Ownership (WRDS)
# One file per quarter: data/q_YYYY-MM-DD.parquet
#
# Changes in this version:
#   1. QUARTER LABELS FIXED. seq.Date(from = <month end>, by = "quarter")
#      rolls invalid dates forward (June 31 -> July 1, Sept 31 -> Oct 1),
#      so Q2/Q3 files were labelled with the first day of the NEXT quarter.
#      Quarter starts are generated instead, and the ends derived from them.
#   2. WEIGHTS RETAINED. w = adj_mv / sum(adj_mv) is now written out,
#      chunked in step with the tokens, enabling weighted pooling.
#   3. CHUNK ID. Chunk order was previously implicit in row order; it is
#      now an explicit column.
#   4. MARKET CAPS (optional, free). adj_price and adj_shares_outstanding
#      already travel with the fund holdings, so per-quarter issuer market
#      caps are captured at no extra query cost.
# ================================================================

library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# ── Parameters ─────────────────────────────────────────────────
START_QUARTER  <- ymd("2005-01-01")
END_QUARTER    <- ymd("2026-03-31")   # 2026-Q2 is still back-filling
MIN_STOCKS     <- 20     # per investor-quarter
MIN_INVESTORS  <- 20     # per stock-quarter
MAX_TOP1_PCT   <- 0.75   # max single-holding weight
REPORT_WINDOW  <- 5      # days around quarter-end for fund reports
CONTEXT_WINDOW <- 62     # PS-BERT max sequence length
SAVE_MKTCAP    <- FALSE  # also write reference/mktcap/q_<date>.parquet

out_dir    <- "data"
mktcap_dir <- "reference/mktcap"

# ── Test mode ─────────────────────────────────────────────────
TEST_MODE <- FALSE

if (TEST_MODE) {
  START_QUARTER <- ymd("2019-07-01")
  END_QUARTER   <- ymd("2019-12-31")   # 2 quarters only
  out_dir       <- "data/test"
  mktcap_dir    <- "reference/mktcap/test"
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
if (SAVE_MKTCAP) dir.create(mktcap_dir, showWarnings = FALSE, recursive = TRUE)


# ── Overwrite behaviour ────────────────────────────────────────
# OVERWRITE = TRUE regenerates every quarter even if its file already
# exists, and removes the mislabelled twin (q_2005-07-01.parquet for
# 2005-06-30) AFTER the replacement has been written successfully, so an
# interrupted run never leaves a quarter with no file at all.
# Set FALSE to resume an interrupted run instead (skips completed quarters).
OVERWRITE <- FALSE

stale <- list.files(out_dir, pattern = "^q_\\d{4}-\\d{2}-01\\.parquet$")
if (length(stale)) {
  message(sprintf("%d mislabelled file(s) present (e.g. %s).",
                  length(stale), stale[1]))
  message(if (OVERWRITE)
            "  They will be removed as each quarter is regenerated."
          else
            "  OVERWRITE is FALSE, so these will be left in place.")
}


# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASS"),
  connect_timeout = 30
)

tbl_13f      <- tbl(wrds, in_schema("factset_own", "wrds_own_13f"))
tbl_fund     <- tbl(wrds, in_schema("factset_own", "wrds_own_fund"))
tbl_ent_fund <- tbl(wrds, in_schema("factset_own", "own_ent_funds"))
tbl_sec_map  <- tbl(wrds, in_schema("factset_own", "own_sec_entity_eq"))

# Quarter ENDS, derived from quarter STARTS. The 1st of a month is always a
# valid date, so nothing rolls forward: 03-31, 06-30, 09-30, 12-31.
quarter_ends <- ceiling_date(
  seq.Date(floor_date(START_QUARTER, "quarter"),
           floor_date(END_QUARTER,   "quarter"), by = "quarter"),
  "quarter") - days(1)

stopifnot(all(format(quarter_ends, "%m-%d") %in%
              c("03-31", "06-30", "09-30", "12-31")))

# Lazy reference for issuer mapping (joined server-side, never downloaded)
sec_map_lazy <- tbl_sec_map |>
  filter(!is.na(factset_entity_id)) |>
  select(fsym_id, issuer_id = factset_entity_id)

fund_map <- tbl_ent_fund |> filter(!is.na(fund_type)) |>
                            select(factset_fund_id, fund_type)


# ── Sequence chunking helper ──────────────────────────────────
# Splits a rank-ordered vector into chunks of at most `ctx` elements.
# Applied identically to tokens and to weights, so the two stay aligned.
chunk_seq <- function(x, n, ctx = CONTEXT_WINDOW) {
  if (n <= ctx) return(list(x))
  k <- ceiling(n / ctx)
  unname(split(x, ceiling(seq_along(x) / ceiling(n / k))))
}


# ── Per-quarter loop ──────────────────────────────────────────
message(length(quarter_ends), " quarters from ",
        first(quarter_ends), " to ", last(quarter_ends))

for (i in seq_along(quarter_ends)) {
  qe       <- quarter_ends[i]
  q_label  <- as.character(qe)
  out_file <- file.path(out_dir, sprintf("q_%s.parquet", q_label))

  message("[", i, "/", length(quarter_ends), "] ", q_label)

  if (file.exists(out_file) && !OVERWRITE) {
    message("   already exists, skipping")
    next
  }

  t0 <- Sys.time()

  # Precompute date bounds in R (dbplyr can't push date - numeric to Postgres SQL)
  q_13f_lo  <- qe - 7
  q_13f_hi  <- qe + 7
  q_fund_lo <- qe - REPORT_WINDOW
  q_fund_hi <- qe + REPORT_WINDOW


  # ── 1. 13F holdings (hedge funds only) ──
  holdings_13f <- tbl_13f |>
    filter(entity_sub_type == "HF",
           report_date >= q_13f_lo,
           report_date <= q_13f_hi,
           adj_mv > 0) |>
    inner_join(sec_map_lazy, by = "fsym_id") |>
    select(investor_id = factset_entity_id, issuer_id, report_date, adj_mv) |>
    collect() |>
    mutate(report_date = as.Date(report_date), quarter_end = qe) |>
    group_by(investor_id, quarter_end, issuer_id) |>
    summarise(adj_mv = sum(as.numeric(adj_mv)), investor_type = "HF",
              .groups = "drop")


  # ── 2. Fund holdings ──
  # adj_price and adj_shares_outstanding are carried through only so that
  # issuer market caps can be captured in step 2b; they are dropped after.
  fund_raw <- tbl_fund |>
    inner_join(sec_map_lazy, by = "fsym_id") |>
    inner_join(fund_map, by = "factset_fund_id") |>
    filter(fund_type %in% c("OEF", "ETF", "CEF", "VAR"),
           report_date >= q_fund_lo,
           report_date <= q_fund_hi,
           adj_mv > 0) |>
    select(investor_id = factset_fund_id, report_date, adj_mv,
           investor_type = fund_type, issuer_id, fsym_id,
           adj_price, adj_shares_outstanding) |>
    collect() |>
    filter(!is.na(issuer_id)) |>
    mutate(report_date = as.Date(report_date), quarter_end = qe)

  # ── 2b. Issuer market caps for this quarter (free: already downloaded) ──
  if (SAVE_MKTCAP) {
    mktcap <- fund_raw |>
      filter(!is.na(adj_price), !is.na(adj_shares_outstanding),
             adj_price > 0, adj_shares_outstanding > 0) |>
      distinct(issuer_id, fsym_id, adj_price, adj_shares_outstanding) |>
      mutate(market_cap = adj_price * adj_shares_outstanding) |>
      group_by(issuer_id) |>
      slice_max(market_cap, n = 1, with_ties = FALSE) |>   # primary listing
      ungroup() |>
      mutate(quarter_end = qe) |>
      select(quarter_end, issuer_id, fsym_id, adj_price,
             adj_shares_outstanding, market_cap)
    write_parquet(mktcap, file.path(mktcap_dir, sprintf("q_%s.parquet", q_label)))
    rm(mktcap)
  }

  holdings_fund <- fund_raw |>
    select(-fsym_id, -adj_price, -adj_shares_outstanding) |>
    arrange(investor_id, issuer_id, desc(report_date)) |>
    group_by(investor_id, issuer_id) |>
    slice_head(n = 1) |>                       # most recent report in window
    ungroup() |>
    group_by(investor_id, investor_type, quarter_end, issuer_id) |>
    summarise(adj_mv = sum(as.numeric(adj_mv)), .groups = "drop")
  rm(fund_raw)


  # ── 3. Combine ──
  holdings <- bind_rows(holdings_13f, holdings_fund)
  rm(holdings_13f, holdings_fund)

  if (nrow(holdings) == 0) {
    message("   no holdings, skipping")
    next
  }


  # ── 4. Concentration filter + bipartite pruning ──
  #    The bipartite pruning (>=20 investors per stock) implicitly
  #    removes micro/nano caps since they have too few holders,
  #    making a separate market-cap filter unnecessary.
  holdings <- holdings |>
    group_by(investor_id, quarter_end) |>
    filter(max(adj_mv) / sum(adj_mv) <= MAX_TOP1_PCT) |>
    ungroup()

  repeat {
    n0 <- nrow(holdings)
    holdings <- holdings |>
      group_by(investor_id, quarter_end) |> filter(n() >= MIN_STOCKS) |> ungroup() |>
      group_by(issuer_id, quarter_end)   |> filter(n() >= MIN_INVESTORS) |> ungroup()
    if (nrow(holdings) == n0) break
  }

  if (nrow(holdings) == 0) {
    message("   nothing survived pruning, skipping")
    next
  }


  # ── 5. Portfolio → token sequences ──
  # Weights are computed on the FULL portfolio and chunked alongside the
  # tokens, so a chunk's weights are shares of the whole book (they sum to
  # 1 across all chunks of an investor, not within a chunk). The training
  # code relies on this to weight tail positions correctly.
  sequences <- holdings |>
    group_by(investor_id, quarter_end) |>
    mutate(w = adj_mv / sum(adj_mv)) |>
    arrange(desc(w), .by_group = TRUE) |>
    summarise(investor_type = first(investor_type),
              tokens   = list(as.character(issuer_id)),
              weights  = list(w),
              n_assets = n(),
              .groups  = "drop") |>
    mutate(chunks  = map2(tokens,  n_assets, chunk_seq),
           wchunks = map2(weights, n_assets, chunk_seq)) |>
    select(-tokens, -weights) |>
    mutate(chunk_id = map(chunks, seq_along)) |>
    unnest(c(chunks, wchunks, chunk_id)) |>
    mutate(tokens   = chunks,
           weights  = wchunks,
           n_tokens = map_int(tokens, length)) |>
    select(quarter_end, investor_id, investor_type, chunk_id,
           tokens, weights, n_tokens, n_assets_full = n_assets)

  # sanity: weights must align with tokens and sum to 1 per investor
  stopifnot(all(sequences$n_tokens == map_int(sequences$weights, length)))
  wsum <- sequences |> group_by(investor_id) |>
    summarise(s = sum(map_dbl(weights, sum)), .groups = "drop")
  stopifnot(max(abs(wsum$s - 1)) < 1e-6)


  # ── 6. Save and free memory ──
  write_parquet(sequences, out_file)

  # Remove the mislabelled twin now that the replacement exists. The old
  # naming rolled month ends forward by one day, so the stale name for
  # this quarter is qe + 1 (2005-06-30 -> q_2005-07-01.parquet). Q1 and Q4
  # were already labelled correctly and were overwritten in place above,
  # so no twin exists for them and nothing is removed.
  twin <- file.path(out_dir, sprintf("q_%s.parquet", qe + days(1)))
  if (twin != out_file && file.exists(twin)) {
    file.remove(twin)
    message("   removed mislabelled ", basename(twin))
  }

  elapsed_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  message(sprintf("   %s holdings, %s investors, %s sequences (%.1f min)",
                  format(nrow(holdings), big.mark = ","),
                  format(n_distinct(holdings$investor_id), big.mark = ","),
                  format(nrow(sequences), big.mark = ","),
                  elapsed_min))

  rm(holdings, sequences); gc(verbose = FALSE)
}

dbDisconnect(wrds)
message("Done.")