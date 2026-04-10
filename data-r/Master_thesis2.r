# ================================================================
# PS-BERT portfolio sequences from FactSet Ownership (WRDS)
# Output: data/portfolio_sequences.parquet
# ================================================================

library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# dake cakkzo dake





# ── Parameters ─────────────────────────────────────────────────
START_QUARTER  <- ymd("2005-01-01")
END_QUARTER    <- ymd("2022-12-31")
MIN_STOCKS     <- 20     # per investor-quarter
MIN_INVESTORS  <- 20     # per stock-quarter
MAX_TOP1_PCT   <- 0.75   # max single-holding weight
REPORT_WINDOW  <- 5      # days before quarter-end for fund reports
CONTEXT_WINDOW <- 62     # PS-BERT max sequence length

out_dir <- "data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

tbl_13f     <- tbl(wrds, in_schema("factset_own", "wrds_own_13f"))
tbl_fund    <- tbl(wrds, in_schema("factset_own", "wrds_own_fund"))
tbl_sec_map <- tbl(wrds, in_schema("factset_own", "own_sec_entity_eq"))

quarter_ends <- seq.Date(
  ceiling_date(START_QUARTER, "quarter") - days(1),
  ceiling_date(END_QUARTER,   "quarter") - days(1),
  by = "quarter"
)

snap_qe <- function(d) {
  qe <- ceiling_date(d, "quarter") - days(1)
  if_else(d > qe, ceiling_date(d + days(1), "quarter") - days(1), qe)
}

# Lazy reference for issuer mapping (joined server-side, never downloaded)
sec_map_lazy <- tbl_sec_map |>
  filter(!is.na(factset_entity_id)) |>
  select(fsym_id, issuer_id = factset_entity_id)


# ── 1. 13F holdings (hedge funds) ─────────────────────────────
message("1. 13F holdings")

holdings_13f <- map_dfr(year(START_QUARTER):year(END_QUARTER), \(yr) {
  message("   ", yr)
  tbl_13f |>
    filter(entity_sub_type == "HF",
           report_date >= as.Date(paste0(yr, "-01-01")),
           report_date <= as.Date(paste0(yr, "-12-31")),
           adj_mv > 0) |>
    inner_join(sec_map_lazy, by = "fsym_id") |>
    select(investor_id = factset_entity_id, issuer_id, report_date, adj_mv) |>
    collect()
}) |>
  mutate(report_date = as.Date(report_date),
         quarter_end = snap_qe(report_date)) |>
  filter(quarter_end %in% quarter_ends) |>
  group_by(investor_id, quarter_end, issuer_id) |>
  summarise(adj_mv = sum(as.numeric(adj_mv)), investor_type = "HF",
            .groups = "drop")

message("   ", format(nrow(holdings_13f), big.mark = ","), " rows")


# ── 2. Fund holdings (MF, ETF, CEF, VA) ───────────────────────
message("2. Fund holdings")

holdings_fund <- map_dfr(quarter_ends, \(qe) {
  message("   ", qe)
  q_start <- qe - REPORT_WINDOW
  tbl_fund |>
    filter(entity_sub_type %in% c("OEF", "ETF", "CEF", "VAR"),
           report_date >= q_start,
           report_date <= qe,
           adj_mv > 0) |>
    select(investor_id = factset_fund_id, report_date, adj_mv,
           investor_type = entity_sub_type,
           issuer_id = factset_sec_entity_id) |>
    collect() |>
    mutate(quarter_end = qe)
}) |>
  filter(!is.na(issuer_id)) |>
  mutate(report_date = as.Date(report_date)) |>
  arrange(investor_id, issuer_id, quarter_end, desc(report_date)) |>
  group_by(investor_id, issuer_id, quarter_end) |>
  slice_head(n = 1) |>
  ungroup() |>
  group_by(investor_id, investor_type, quarter_end, issuer_id) |>
  summarise(adj_mv = sum(as.numeric(adj_mv)), .groups = "drop")

message("   ", format(nrow(holdings_fund), big.mark = ","), " rows")


# ── 3. Combine ────────────────────────────────────────────────
message("3. Combine")

holdings <- bind_rows(holdings_13f, holdings_fund)
rm(holdings_13f, holdings_fund)
message("   ", format(nrow(holdings), big.mark = ","), " rows")


# ── 4. Concentration filter + bipartite pruning ───────────────
#    The bipartite pruning (>=20 investors per stock) implicitly
#    removes micro/nano caps since they have too few holders,
#    making a separate market-cap filter unnecessary.
message("4. Filtering")

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
  message("   ", format(nrow(holdings), big.mark = ","))
}

message("   ", format(nrow(holdings), big.mark = ","), " rows, ",
        n_distinct(holdings$investor_id), " investors, ",
        n_distinct(holdings$issuer_id), " firms")


# ── 5. Portfolio → token sequences ────────────────────────────
message("5. Building sequences")

sequences <- holdings |>
  group_by(investor_id, quarter_end) |>
  mutate(w = adj_mv / sum(adj_mv)) |>
  arrange(desc(w), .by_group = TRUE) |>
  summarise(investor_type = first(investor_type),
            tokens   = list(as.character(issuer_id)),
            n_assets = n(),
            .groups  = "drop")

chunk_seq <- function(tokens, n, ctx = CONTEXT_WINDOW) {
  if (n <= ctx) return(list(tokens))
  k <- ceiling(n / ctx)
  split(tokens, ceiling(seq_along(tokens) / ceiling(n / k)))
}

sequences <- sequences |>
  mutate(chunks = map2(tokens, n_assets, chunk_seq)) |>
  unnest(chunks) |>
  mutate(tokens   = chunks,
         n_tokens = map_int(tokens, length)) |>
  select(quarter_end, investor_id, investor_type,
         tokens, n_tokens, n_assets_full = n_assets)

message("   ", format(nrow(sequences), big.mark = ","), " sequences")


# ── 6. Save ───────────────────────────────────────────────────
write_parquet(sequences, file.path(out_dir, "portfolio_sequences.parquet"))

sequences |>
  summarise(investors  = n_distinct(investor_id),
            quarters   = n_distinct(quarter_end),
            median_len = median(n_tokens),
            mean_len   = round(mean(n_tokens), 1)) |>
  print()

sequences |> count(investor_type, sort = TRUE) |> print()

dbDisconnect(wrds)
message("Done.")
