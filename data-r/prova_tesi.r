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
tbl1 <- tbl(wrds, in_schema("factset_own", "own_ent_institutions"))

fund_map <- tbl1 |> filter(!is.na(entity_sub_type)) |> 
                            select(factset_entity_id, entity_sub_type)


tbl_13f |> count(entity_sub_type, sort = TRUE) |> collect() |> print()

tbl_13f |> select(factset_entity_id) |>
           inner_join(fund_map, by = "factset_entity_id") |> 
           count(entity_sub_type, sort = TRUE) |> collect() |> print()
