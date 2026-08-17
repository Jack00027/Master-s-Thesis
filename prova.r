library(tidyverse)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# ── WRDS connection ────────────────────────────────────────────


message("connecting..."); t <- Sys.time()
wrds <- dbConnect(Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD"),
  connect_timeout = 10)
message("connected in ", round(difftime(Sys.time(), t, units="secs"), 1), "s")

message("trivial query..."); print(dbGetQuery(wrds, "SELECT 1 AS ok"))
message("metadata..."); tbl_fund <- tbl(wrds, in_schema("factset_own","wrds_own_fund"))
message("done")

tbl_13f      <- tbl(wrds, in_schema("factset_own", "wrds_own_13f"))
tbl_fund     <- tbl(wrds, in_schema("factset_own", "wrds_own_fund"))
tbl_ent_fund <- tbl(wrds, in_schema("factset_own", "own_ent_funds"))
tbl_sec_map  <- tbl(wrds, in_schema("factset_own", "own_sec_coverage_eq"))

probe <- function(q_start, q_end) {
  t <- Sys.time()
  r <- dbGetQuery(wrds, sprintf("
    SELECT report_date FROM factset_own.own_fund_detail_eq
    WHERE report_date BETWEEN '%s' AND '%s' LIMIT 1", q_start, q_end))
  cat(sprintf("%s .. %s : %s  (%.1fs)\n", q_start, q_end,
              if (nrow(r)) "DATA EXISTS" else "empty",
              as.numeric(difftime(Sys.time(), t, units = "secs"))))
}

probe("2025-03-01","2025-03-31")
probe("2025-06-01","2025-06-30")
probe("2025-09-01","2025-09-30")
probe("2025-12-01","2025-12-31")
probe("2026-03-01","2026-03-31")
probe("2026-06-01","2026-06-30")

