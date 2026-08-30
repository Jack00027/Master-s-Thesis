## ---------------------------------------------------------------------------
## reporting_cycle_diagnostic.R
##
## Who actually skips March and September?
##
## For every investor, compare the quarters in which it appears against the
## quarters in which it *could* have appeared (i.e. those inside its own first
## -to-last span), separately for the high-reporting quarters (Q2, Q4) and the
## low ones (Q1, Q3). An investor that reports semi-annually shows near-full
## coverage of Q2/Q4 and near-zero coverage of Q1/Q3.
##
## Input : data/q_*.parquet          (quarter_end, investor_id, investor_type)
##         factset_investors.rds     (optional; for domicile / mandate / manager)
## Output: results/reporting_cycle_by_investor.csv
##         console cross-tabs
## ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(tidyr)
})

DATA_DIR  <- "data"
INV_TABLE <- "factset_investors.rds"   # set to NA to skip the attribute joins
OUT       <- "results/reporting_cycle_by_investor.csv"

## thresholds for the classification
COV_HI <- 0.80   # "reports in this season" if coverage at or above this
COV_LO <- 0.20   # "does not report in this season" if coverage at or below
MIN_ELIGIBLE <- 4  # need at least this many chances in each season to classify

dir.create("results", showWarnings = FALSE)

## --- 1. investor x quarter appearance panel --------------------------------
files <- list.files(DATA_DIR, pattern = "^q_.*\\.parquet$", full.names = TRUE)
stopifnot(length(files) > 0)
message("Reading ", length(files), " quarterly files ...")

panel <- lapply(files, function(f)
  read_parquet(f, col_select = c("quarter_end", "investor_id", "investor_type"))
) |> bind_rows() |> distinct(quarter_end, investor_id, .keep_all = TRUE)

panel <- panel |> mutate(quarter_end = as.Date(quarter_end))

qs <- sort(unique(panel$quarter_end))
season <- ifelse((as.integer(format(qs, "%m")) - 1) %/% 3 + 1 %in% c(2, 4),
                 "hi", "lo")
## (explicit, to avoid operator-precedence surprises)
season <- ifelse(((as.integer(format(qs, "%m")) - 1) %/% 3 + 1) %in% c(2, 4),
                 "hi", "lo")
qtab <- tibble(quarter_end = qs, season = season, idx = seq_along(qs))

## cumulative count of hi / lo quarters, so "chances inside a span" is O(1)
cum_hi <- cumsum(qtab$season == "hi")
cum_lo <- cumsum(qtab$season == "lo")

panel <- panel |> left_join(qtab, by = "quarter_end")

## --- 2. per-investor coverage within its own active span -------------------
inv <- panel |>
  group_by(investor_id) |>
  summarise(
    investor_type = dplyr::last(investor_type),
    first_idx = min(idx), last_idx = max(idx),
    obs_hi = sum(season == "hi"), obs_lo = sum(season == "lo"),
    n_obs  = n(), .groups = "drop"
  ) |>
  mutate(
    elig_hi = cum_hi[last_idx] - ifelse(first_idx > 1, cum_hi[first_idx - 1], 0),
    elig_lo = cum_lo[last_idx] - ifelse(first_idx > 1, cum_lo[first_idx - 1], 0),
    cov_hi  = obs_hi / pmax(elig_hi, 1),
    cov_lo  = obs_lo / pmax(elig_lo, 1),
    span    = last_idx - first_idx + 1
  )

inv <- inv |>
  mutate(cycle = case_when(
    elig_hi < MIN_ELIGIBLE | elig_lo < MIN_ELIGIBLE ~ "too short to classify",
    cov_hi >= COV_HI & cov_lo <= COV_LO             ~ "semi-annual (skips Q1/Q3)",
    cov_hi >= COV_HI & cov_lo >= COV_HI             ~ "quarterly",
    cov_lo >= COV_HI & cov_hi <= COV_LO             ~ "inverse (skips Q2/Q4)",
    TRUE                                            ~ "intermittent"
  ))

cat("\n=== Reporting cycle, all investors ever observed ===\n")
print(inv |> count(cycle, sort = TRUE) |> mutate(pct = sprintf("%.1f%%", 100*n/sum(n))))

cat("\n=== Cycle by investor type (row %) ===\n")
print(inv |> filter(cycle != "too short to classify") |>
        count(investor_type, cycle) |>
        group_by(investor_type) |> mutate(pct = round(100*n/sum(n), 1)) |>
        select(-n) |> pivot_wider(names_from = cycle, values_from = pct,
                                  values_fill = 0) |> ungroup())

## --- 3. how much of the seasonal gap do the skippers explain? --------------
semi <- inv |> filter(cycle == "semi-annual (skips Q1/Q3)") |> pull(investor_id)
byq <- panel |>
  mutate(grp = ifelse(investor_id %in% semi, "semi-annual", "other")) |>
  count(quarter_end, season, grp) |>
  group_by(season, grp) |> summarise(mean_n = mean(n), .groups = "drop") |>
  pivot_wider(names_from = season, values_from = mean_n)

cat("\n=== Mean investors per quarter, by group ===\n")
print(byq |> mutate(gap = hi - lo, gap_pct = sprintf("%+.1f%%", 100*(hi/lo - 1))))
cat(sprintf("\nSemi-annual reporters account for %.0f%% of the Q2/Q4 excess.\n",
            100 * (byq$hi[byq$grp=="semi-annual"] - byq$lo[byq$grp=="semi-annual"]) /
                  (sum(byq$hi) - sum(byq$lo))))

## --- 4. who are they? ------------------------------------------------------
if (!is.na(INV_TABLE) && file.exists(INV_TABLE)) {
  ref <- readRDS(INV_TABLE)
  key <- intersect(c("investor_id", "factset_entity_id", "fsym_id"), names(ref))[1]
  ref <- ref |> rename(investor_id = !!key)

  inv2 <- inv |> filter(cycle != "too short to classify") |>
    left_join(ref, by = "investor_id")

  show_by <- function(col, top = 12) {
    if (!col %in% names(inv2)) { message("(no column ", col, ")"); return(invisible()) }
    cat("\n=== Share semi-annual, by ", col, " (top ", top, " by count) ===\n", sep = "")
    out <- inv2 |> filter(!is.na(.data[[col]])) |>
      group_by(grp = .data[[col]]) |>
      summarise(n = n(),
                pct_semi = round(100*mean(cycle == "semi-annual (skips Q1/Q3)"), 1),
                .groups = "drop") |>
      filter(n >= 50) |> arrange(desc(n)) |> head(top)
    print(out)
  }
  for (v in c("investor_domicile", "domicile", "iso_country", "fund_type",
              "mandate_region", "manager_name", "declared_style"))
    show_by(v)
}

write.csv(inv, OUT, row.names = FALSE)
message("\nWrote ", OUT)

## ---------------------------------------------------------------------------
## Ground truth, if the local answer looks ambiguous.
## Absence from a parquet file means "did not survive the filters", which is
## not the same as "did not file". To separate the two, check raw report dates
## in WRDS for a sample of suspected skippers, with no filters applied:
##
##   con <- DBI::dbConnect(RPostgres::Postgres(),
##            host = "wrds-pgdata.wharton.upenn.edu", port = 9737,
##            dbname = "wrds", sslmode = "require",
##            user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASS"))
##   ids <- head(semi, 200)
##   raw <- DBI::dbGetQuery(con, "
##       SELECT fsym_id, report_date, COUNT(*) AS n_positions
##       FROM factset_own.wrds_own_fund
##       WHERE fsym_id = ANY($1) AND report_date BETWEEN '2015-01-01' AND '2025-12-31'
##       GROUP BY 1,2 ORDER BY 1,2", list(ids))
##   # then tabulate report_date by calendar quarter: if these funds have no
##   # March/September report dates at all, the cycle is disclosure, not filtering.
## ---------------------------------------------------------------------------
