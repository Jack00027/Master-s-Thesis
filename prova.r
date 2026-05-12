library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

tbl_13f      <- tbl(wrds, in_schema("factset_own", "wrds_own_13f"))
tbl_fund     <- tbl(wrds, in_schema("factset_own", "wrds_own_fund"))
tbl_ent_fund <- tbl(wrds, in_schema("factset_own", "own_ent_funds"))
tbl_sec_map  <- tbl(wrds, in_schema("factset_own", "own_sec_entity_eq"))



style_distribution <- tbl_13f |>
  group_by(manager_style) |>
  summarise(count = n()) |>
  arrange(desc(count)) |>
  collect() |>
  print()