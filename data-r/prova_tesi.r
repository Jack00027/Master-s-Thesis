library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)
library(glue)

# ── Parameters ──────────────────────────────────────────────
START_QUARTER <- ymd("2005-01-01")
END_QUARTER   <- ymd("2022-12-31")

MIN_STOCKS_PER_INVESTOR <- 20
MIN_INVESTORS_PER_STOCK <- 20
MAX_SINGLE_HOLDING_PCT  <- 0.75
REPORT_DATE_WINDOW_DAYS <- 5
SIZE_PCTL_CUTOFF        <- 0.20



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
tbl_ent_fund <- tbl(wrds, in_schema("factset_own", "own_ent_funds"))


# C. Sample rows where entity_type is NA
tbl_fund |>
  filter(is.na(entity_type),
         report_date >= as.Date("2025-01-01"),
         report_date <= as.Date("2025-12-31"),
         iso_country == "US") |>
  select(entity_proper_name, sec_entity_proper_name,
         entity_type, adj_mv, iso_country) |>
  head(20) |>
  collect() |>
  print()


tbl_ent_fund |> count(fund_type, sort = TRUE) |> collect() |> print()


