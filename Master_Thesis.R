install.packages("pak")
pak::pkg_install(c("tidyverse", "tidyfinance", "dbplyr", "RPostgres"))

pak::pkg_install("arrow")

library(tidyverse)
library(tidyfinance)
library(arrow)
library(dbplyr)
library(RPostgres)

# -----------------------
# 0) Choose ONE quarter
# -----------------------
# If report_date is quarter-end (common for 13F), set both equal.
# Example: 2019Q1 -> 2019-03-31
report_date_target <- ymd("2019-03-31")

MIN_ASSETS    <- 20
MAX_INVESTORS <- 50  # toy sample; set to NULL to pull all funds for that quarter

out_dir <- "data-r"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------
# 1) Connect
# -----------------------
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu",
  dbname = "wrds",
  port = 9737,
  sslmode = "require",
  user = Sys.getenv("WRDS_USER"),
  password = Sys.getenv("WRDS_PASSWORD")
)

own_db <- tbl(wrds, I("factset_own.wrds_own_13f"))

# -----------------------
# 2) (Toy) pick a small set of investors in that quarter
# -----------------------
investors_tbl <- own_db |>
  filter(report_date == report_date_target) |>
  distinct(factset_entity_id)

if (!is.null(MAX_INVESTORS)) {
  investors_tbl <- investors_tbl |> head(MAX_INVESTORS)
}

investors <- investors_tbl |>
  collect() |>
  pull(factset_entity_id) |>
  as.character()

message("Investors in sample: ", length(investors))

# -----------------------
# 3) Pull full holdings for those investors and that report_date
# -----------------------
holdings_raw <- own_db |>
  filter(report_date == report_date_target) |>
  filter(factset_entity_id %in% investors) |>
  select(
    investor_id = factset_entity_id,
    report_date,
    cusip,
    adj_mv,
    entity_sub_type
  ) |>
  collect()

# -----------------------
# 4) Clean + dedup + weights + sequences (FULL portfolio)
# -----------------------
holdings <- holdings_raw |>
  mutate(
    investor_id = as.character(investor_id),
    report_date = as.Date(report_date),
    quarter     = paste0(year(report_date), "Q", quarter(report_date)),
    cusip       = str_replace_all(as.character(cusip), "\\s+", ""),
    cusip       = str_sub(cusip, 1, 8),      # keep 8-char CUSIP core
    adj_mv      = as.numeric(adj_mv)
  ) |>
  filter(
    !is.na(investor_id),
    !is.na(report_date),
    !is.na(cusip), cusip != "",
    !is.na(adj_mv), adj_mv > 0
  )

# Deduplicate within investor-security (summing repeated lines)
holdings <- holdings |>
  group_by(investor_id, quarter, cusip, entity_sub_type) |>
  summarise(adj_mv = sum(adj_mv, na.rm = TRUE), .groups = "drop")

# Portfolio weights
holdings <- holdings |>
  group_by(investor_id, quarter) |>
  mutate(w = adj_mv / sum(adj_mv, na.rm = TRUE)) |>
  ungroup()

# Full ordered token list per portfolio
portfolio_sequences <- holdings |>
  mutate(asset_token = paste0("CUSIP_", cusip)) |>
  arrange(investor_id, desc(w)) |>
  group_by(quarter, investor_id) |>
  summarise(
    tokens   = list(asset_token),
    weights  = list(w),
    n_assets = n(),
    .groups  = "drop"
  ) |>
  filter(n_assets >= MIN_ASSETS)

# -----------------------
# 5) Save
# -----------------------
write_parquet(holdings, file.path(out_dir, "factset13f_holdings_one_quarter.parquet"))
write_parquet(portfolio_sequences, file.path(out_dir, "portfolio_sequences_one_quarter.parquet"))

dbDisconnect(wrds)

message("Saved: ",
        file.path(out_dir, "factset13f_holdings_one_quarter.parquet"),
        " and ",
        file.path(out_dir, "portfolio_sequences_one_quarter.parquet"))
