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

out_dir <- "data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Connect ─────────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host     = "wrds-pgdata.wharton.upenn.edu",
  dbname   = "wrds",
  port     = 9737,
  sslmode  = "require",
  user     = Sys.getenv("WRDS_USER"),
  password = Sys.getenv("WRDS_PASSWORD")
)


# ================================================================
# 0. DISCOVER TABLES — run once, then set config below
# ================================================================

print(# What FactSet schemas do you have access to?
dbGetQuery(wrds, "
  SELECT DISTINCT table_schema
  FROM information_schema.tables
  WHERE table_schema LIKE 'factset%'
  ORDER BY 1
")
)

print(print(dbGetQuery(wrds, "
  SELECT table_name
  FROM information_schema.tables
  WHERE table_schema = 'factset_own'
  ORDER BY 1
"))
)

print( dbGetQuery(wrds, "
   SELECT DISTINCT entity_sub_type, COUNT(*)
   FROM factset_own.own_ent_institutions
   GROUP BY 1 ORDER BY 1
 "))

 print(dbGetQuery(wrds, "
   SELECT DISTINCT fund_type, COUNT(*)
   FROM factset_own.own_ent_funds
   GROUP BY 1 ORDER BY 1
 "))

