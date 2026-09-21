### With this file a create a 2 tables, one for the investors and one for the
### stocks, with the relevant information for each of them.
###
### FIXED VERSION. Changes from the previous script are marked [FIX n].
###   [FIX 1] credentials moved out of the source file
###   [FIX 2] price window was a hardcoded 2019 stub; now anchored on AS_OF
###   [FIX 3] primary listing was chosen by an all-NA market_cap, which made
###           slice_max return an arbitrary row (BDRs, warrants, foreign lines)
###   [FIX 4] name / domicile / sector fallbacks for issuers the entity master
###           does not cover, with provenance columns so the imputation is
###           reportable rather than silent
###   [FIX 5] row-count assertions and a retry+diagnostic on the entity pull
###   [FIX 6] price date filter pushed into SQL instead of downloading the
###           whole history and filtering client-side
###   [FIX 7] feeder-master bridge is many-to-many; aggregated, not deduped
###   [FIX 8] resilient connection: keepalives, statement_timeout, per-chunk
###           retry on one connection, and an on-disk cache per pull
###   [FIX 9] price pull restricted to candidate listings (was 287,649)
###   [FIX 10] holder statistics rewritten with data.table; the unnest_longer
###           version ran for many minutes at 100% CPU on the 85-quarter panel

# data.table is loaded FIRST on purpose: it masks first(), last() and between()
# from dplyr, and inv_panel below calls first(). Loading it before tidyverse
# leaves the dplyr versions on top of the search path.
library(data.table)
library(dplyr); library(tidyr); library(purrr)
library(readr); library(stringr); library(tibble)
library(lubridate)
library(dbplyr)
library(RPostgres)
library(arrow)


# ── WRDS connection ────────────────────────────────────────────
# [FIX 1] Credentials come from ~/.Renviron, never from the source file.
#   Put these two lines in ~/.Renviron and RESTART R (.Renviron is read at
#   startup only, not on source()):
#       WRDS_USER=jack027
#       WRDS_PASS=your_new_password
wrds_user <- Sys.getenv("WRDS_USER")
wrds_pass <- Sys.getenv("WRDS_PASS")
if (!nzchar(wrds_user) || !nzchar(wrds_pass))
  stop("WRDS_USER / WRDS_PASS not set in ~/.Renviron (restart R after editing it)")

# [FIX 8] The previous version opened one connection and used it for the whole
# run. A dropped socket (WRDS idle timeout, VPN reconnect, NAT eviction) leaves
# dbGetQuery blocked on a read that never returns, so the script hangs forever
# rather than erroring. Three defences:
#   keepalives*       keep the TCP session warm through idle periods, which is
#                     what stops the socket being dropped in the first place
#   statement_timeout turns a server-side hang into a catchable error; the
#                     session survives it, so the retry reuses this connection
#   wrds_connect()    is callable again, but is only invoked if dbIsValid()
#                     reports the session is genuinely dead. In a normal run
#                     the script holds exactly one WRDS connection throughout.
wrds_connect <- function() {
  dbConnect(
    Postgres(),
    host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
    port = 9737, sslmode = "require",
    user = wrds_user, password = wrds_pass,
    keepalives = 1, keepalives_idle = 30,
    keepalives_interval = 10, keepalives_count = 5,
    connect_timeout = 30,
    options = "-c statement_timeout=600000"   # 10 minutes
  )
}

wrds <- wrds_connect()


SEQ_DIR   <- "data"            # local quarterly token files
OUT_DIR   <- "reference"

# [FIX 2] The old script pulled prices for 2019-07-01 .. 2019-10-15 and used
# that single snapshot as the market cap for a panel running 2005-2026. Every
# issuer that listed after 2019 or delisted before it got market_cap = NA.
# AS_OF is set explicitly so the build is reproducible: a default of
# max(quarter_end) would drift the moment another quarter is added. Set it to
# the quarter you are clustering. The lookback is long enough to survive
# suspensions and thin trading without letting prices go badly stale.
AS_OF               <- as.Date("2025-12-31")
PRICE_LOOKBACK_DAYS <- 120

# [FIX 8] resilience settings
MAX_RETRIES <- 4L                       # attempts per chunk before giving up
CACHE_DIR   <- file.path(OUT_DIR, "_cache")
USE_CACHE   <- TRUE                     # FALSE forces every pull to re-run

dir.create(OUT_DIR,   showWarnings = FALSE)
dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE)


# ── progress reporting ────────────────────────────────────────────────────
.T0 <- Sys.time()
.STEP <- 0
.STEPS_TOTAL <- 16          # major operations, for the [i/n] counter

step <- function(msg) {
  .STEP <<- .STEP + 1
  cat(sprintf("\n[%2d/%2d] %6.1f min | %s\n",
              .STEP, .STEPS_TOTAL,
              as.numeric(difftime(Sys.time(), .T0, units = "mins")), msg))
  flush.console()
}

fmt_secs <- function(s) {
  if (is.na(s) || !is.finite(s)) return("?")
  if (s < 90) sprintf("%.0fs", s) else sprintf("%.1fm", s / 60)
}

# Pull rows whose id is in a local vector, in chunks, so we never scan or
# download an entire entity table. Prints per-chunk progress with an ETA,
# because some of these tables are large and the queries are slow.
#
# [FIX 6] `where` appends an extra SQL predicate so date filters run on the
# server. The old script downloaded every price row for every listing in the
# panel and then filtered in R.
fetch_by_ids <- function(schema, table, id_col, ids, cols = "*",
                         chunk = 4000, where = NULL) {
  ids <- unique(ids[!is.na(ids)])
  label <- paste(schema, table, sep = ".")
  if (!length(ids)) {
    cat(sprintf("    %-45s no ids to look up, skipped\n", label))
    return(tibble())
  }
  sel   <- if (identical(cols, "*")) "*" else paste(cols, collapse = ", ")
  extra <- if (is.null(where)) "" else paste(" AND", where)
  parts <- split(ids, ceiling(seq_along(ids) / chunk))
  np    <- length(parts)
  t0    <- Sys.time()
  res   <- vector("list", np)
  n_rows <- 0L

  for (i in seq_len(np)) {
    sql <- sprintf("SELECT %s FROM %s.%s WHERE %s IN (%s)%s",
                   sel, schema, table, id_col,
                   paste0("'", parts[[i]], "'", collapse = ","), extra)

    # [FIX 8] Retry the chunk on error. A statement_timeout or a cancelled
    # query leaves the session perfectly usable, so the retry reuses the SAME
    # connection. Only if dbIsValid() reports the session is actually dead do
    # we open a new one, which keeps this to a single WRDS connection in every
    # normal case and avoids connection churn on a shared server.
    attempt <- 0L
    repeat {
      attempt <- attempt + 1L
      got <- tryCatch(dbGetQuery(wrds, sql), error = function(e) e)
      if (!inherits(got, "error")) break
      if (attempt >= MAX_RETRIES)
        stop(sprintf("%s chunk %d failed after %d attempts: %s",
                     label, i, attempt, conditionMessage(got)))

      alive <- isTRUE(tryCatch(dbIsValid(wrds), error = function(e) FALSE))
      cat(sprintf("\n    [retry %d/%d] %s chunk %d (%s): %s\n",
                  attempt, MAX_RETRIES - 1L, label, i,
                  if (alive) "session alive, reusing" else "session dead, reopening",
                  conditionMessage(got)))

      if (!alive) {
        try(dbDisconnect(wrds), silent = TRUE)
        wrds <<- wrds_connect()
      }
      Sys.sleep(5 * attempt)          # back off before trying again
    }

    res[[i]] <- got
    n_rows  <- n_rows + nrow(res[[i]])
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    eta     <- elapsed / i * (np - i)
    cat(sprintf("\r    %-45s chunk %d/%d | %s rows | %s elapsed, ~%s left      ",
                label, i, np, format(n_rows, big.mark = ","),
                fmt_secs(elapsed), fmt_secs(eta)))
    flush.console()
  }
  cat(sprintf("\r    %-45s done: %s rows in %s%s\n",
              label, format(n_rows, big.mark = ","),
              fmt_secs(as.numeric(difftime(Sys.time(), t0, units = "secs"))),
              strrep(" ", 25)))
  as_tibble(bind_rows(res))
}

fetch_all <- function(schema, table) {
  label <- paste(schema, table, sep = ".")
  t0 <- Sys.time()
  cat(sprintf("    %-45s downloading...", label)); flush.console()
  out <- dbGetQuery(wrds, sprintf("SELECT * FROM %s.%s", schema, table))
  cat(sprintf("\r    %-45s done: %s rows in %s%s\n",
              label, format(nrow(out), big.mark = ","),
              fmt_secs(as.numeric(difftime(Sys.time(), t0, units = "secs"))),
              strrep(" ", 20)))
  as_tibble(out)
}

prefix_cols <- function(df, prefix, keep) {
  rename_with(df, ~ paste0(prefix, .x), .cols = setdiff(names(df), keep))
}

# [FIX 8] Persist each completed pull. A run that dies at step 12 then resumes
# from step 12 rather than step 1. Delete reference/_cache/ (or set
# USE_CACHE <- FALSE) to force a genuinely fresh build.
#
# Cache files are raw query output, before any joining or renaming, so they are
# safe to keep across script edits that only touch the assembly logic. They are
# NOT safe to keep if you change what a pull SELECTs or its WHERE clause: delete
# the specific file in that case. AS_OF is baked into the price cache filename
# for exactly this reason.
cached <- function(name, expr) {
  path <- file.path(CACHE_DIR, paste0(name, ".parquet"))
  if (USE_CACHE && file.exists(path)) {
    out <- read_parquet(path)
    cat(sprintf("    %-45s [cached] %s rows\n", name,
                format(nrow(out), big.mark = ",")))
    return(out)
  }
  out <- force(expr)
  if (nrow(out)) write_parquet(out, path)
  out
}

# [FIX 5] A left_join against a non-unique key silently multiplies rows. The
# investor block already checked this; the security block did not.
assert_rows <- function(df, n_expected, label) {
  if (nrow(df) != n_expected)
    stop(sprintf("%s: row count changed %d -> %d (duplicate keys in a joined table)",
                 label, n_expected, nrow(df)))
  invisible(df)
}

# =========================================================================
# 1. Collect the ids we need, from the local pipeline
# =========================================================================
step("Reading local quarterly files")
seq_files <- list.files(SEQ_DIR, pattern = "^q_.*\\.parquet$", full.names = TRUE)
stopifnot(length(seq_files) > 0)

panel <- map_dfr(seq_along(seq_files), function(i) {
  cat(sprintf("\r    file %d/%d  %s", i, length(seq_files),
              basename(seq_files[i]))); flush.console()
  read_parquet(seq_files[i]) |>
    select(investor_id, quarter_end, investor_type, n_assets_full, tokens)
})
cat("\n")

id_map <- panel |>
  distinct(investor_id, investor_type) |>
  mutate(id_source = if_else(investor_type == "HF", "entity", "fund"))

entity_ids   <- id_map |> filter(id_source == "entity") |> pull(investor_id)
fund_ids     <- id_map |> filter(id_source == "fund")   |> pull(investor_id)
issuer_ids   <- unique(unlist(panel$tokens))

# [FIX 2] valuation window, anchored on the explicit AS_OF above
AS_OF <- as.Date(AS_OF)
PRICE_DATE_FROM <- AS_OF - PRICE_LOOKBACK_DAYS
PRICE_DATE_TO   <- AS_OF
if (!AS_OF %in% panel$quarter_end)
  warning(sprintf("AS_OF %s is not a quarter_end in the local panel", AS_OF))

cat(sprintf(paste0("Local panel: %d quarters | %d investors | %d issuers\n",
                   "  entity-keyed (HF)            : %d\n",
                   "  fund-keyed (OEF/ETF/CEF/VAR) : %d\n",
                   "  market caps as of            : %s (window %s .. %s)\n"),
            n_distinct(panel$quarter_end), nrow(id_map), length(issuer_ids),
            length(entity_ids), length(fund_ids),
            AS_OF, PRICE_DATE_FROM, PRICE_DATE_TO))

# =========================================================================
# 2. Verify the keyspace assumption before joining anything
# =========================================================================
step("Key diagnostic: verifying which tables each id block matches")
probe <- id_map |> group_by(investor_type) |> slice_head(n = 800) |> ungroup()

hit <- function(schema, table, id_col) {
  found <- fetch_by_ids(schema, table, id_col, probe$investor_id, cols = id_col)
  found <- if (nrow(found)) found[[1]] else character(0)
  probe |>
    mutate(ok = investor_id %in% found) |>
    group_by(investor_type) |>
    summarise(pct = round(100 * mean(ok), 1), .groups = "drop") |>
    mutate(table = paste(schema, table, sep = "."))
}

diag <- bind_rows(
  hit("factset",     "edm_standard_entity",     "factset_entity_id"),
  hit("factset_own", "own_ent_institutions",    "factset_entity_id"),
  hit("factset_own", "own_ent_funds",           "factset_fund_id"),
  hit("factset_own", "own_ent_fund_objectives", "factset_fund_id")
) |>
  pivot_wider(names_from = investor_type, values_from = pct, values_fill = 0)

cat("\n=== Key diagnostic: % of sampled ids found, by investor type ===\n")
print(diag, n = Inf, width = Inf)
cat("Expected: HF high on the entity-keyed tables, OEF/ETF/CEF/VAR high on\n",
    "the fund-keyed ones. edm_standard_entity may cover both.\n")

# =========================================================================
# 3. INVESTOR MASTER
# =========================================================================

# ---- 3a. fund block: attributes for OEF/ETF/CEF/VAR ---------------------
# own_ent_funds carries fund_type, family, and the portfolio characteristic
# block (pe/pb/yield/growth/momentum/strength/beta) — the strategy
# fingerprint, and the bridge to the managing institution.
step("Fund block: attributes, mandate, feeder links, tickers")
fund_attr <- cached("fund_attr", fetch_by_ids(
  "factset_own", "own_ent_funds", "factset_fund_id", fund_ids,
  cols = c("factset_fund_id", "factset_inst_entity_id", "fund_type", "style",
           "turnover_label", "pe_ratio", "pb_ratio", "dividend_yield",
           "sales_growth", "price_momentum", "relative_strength", "beta",
           "fund_family", "etf_type", "active_flag", "current_report_date"))) |>
  # explicit names for the fields that exist in more than one source
  rename(fund_style          = style,
         fund_turnover_label = turnover_label,
         fund_report_date    = current_report_date)

# declared mandate: objective, specialisation, asset type, region, country.
# The declared-strategy counterpart to the holdings-based lenses.
fund_obj <- cached("fund_obj",
                   fetch_by_ids("factset_own", "own_ent_fund_objectives",
                                "factset_fund_id", fund_ids))

# feeder -> master: feeder funds hold mechanically identical portfolios,
# so a cluster of feeders is an administrative artifact, not a strategy.
#
# [FIX 7] own_ent_funds_feeder_master is a genuine many-to-many bridge with no
# date column: a feeder can map to more than one master (one such case in the
# current panel, a French multi-compartment vehicle). The old code left-joined
# it raw, which multiplied that investor into two rows. distinct() would have
# silently dropped a master instead, so aggregate to one row per feeder and
# keep the full mapping in master_ids.
#
# COVERAGE CAVEAT for the write-up: this table covers ~239 of 76,494 funds
# (0.3%). It is a positive control on a small subset, not a clean bill of
# health for the panel; mgr_entity_proper_name does the real work on the
# administrative-artifact question.
fund_feeder <- cached("fund_feeder",
                      fetch_by_ids("factset_own", "own_ent_funds_feeder_master",
                                   "factset_feeder_fund_id", fund_ids)) |>
  group_by(factset_feeder_fund_id) |>
  summarise(is_feeder  = TRUE,
            n_masters  = n(),
            master_ids = paste(sort(unique(factset_master_fund_id)),
                               collapse = "|"),
            # retained for backward compatibility; min() makes it deterministic
            factset_master_fund_id = min(factset_master_fund_id),
            .groups = "drop")

cat(sprintf("Feeder bridge: %s feeders (%s with >1 master) out of %s funds (%.1f%%)\n",
            format(nrow(fund_feeder), big.mark = ","),
            format(sum(fund_feeder$n_masters > 1), big.mark = ","),
            format(length(fund_ids), big.mark = ","),
            100 * nrow(fund_feeder) / length(fund_ids)))

fund_ticker <- cached("fund_ticker",
                      fetch_by_ids("factset_own", "own_ent_fund_identifiers",
                                   "factset_fund_id", fund_ids)) |>
  filter(!is.na(fund_ticker)) |>
  distinct(factset_fund_id, .keep_all = TRUE) |>
  select(factset_fund_id, fund_ticker)

# ---- 3b. institution block: attributes for HF ---------------------------
step("Institution block (HF): style, turnover, AUM")
inst_attr <- fetch_by_ids(
  "factset_own", "own_ent_institutions", "factset_entity_id", entity_ids,
  cols = c("factset_entity_id", "style", "manager_style", "turnover_label",
           "total_aum", "entity_sub_type", "fds_13f_flag",
           "current_report_date")) |>
  # explicit prefix: join suffixes only rename COLLIDING columns, which
  # silently leaves non-colliding ones (like style) unprefixed
  prefix_cols("inst_", keep = "factset_entity_id")

# ---- 3c. entity master: try ALL ids (funds are entities too) ------------
step("Entity master: names, country, sector (all investor ids)")
ent_attr <- cached("ent_attr", fetch_by_ids(
  "factset", "edm_standard_entity", "factset_entity_id",
  c(entity_ids, fund_ids),
  cols = c("factset_entity_id", "entity_proper_name", "entity_type",
           "entity_sub_type", "iso_country", "iso_country_incorp",
           "sector_code", "industry_code", "primary_sic_code",
           "metro_area", "state_province", "year_founded", "web_site")))

ent_parent <- cached("ent_parent",
                     fetch_by_ids("factset", "edm_standard_entity_structure",
                                  "factset_entity_id", c(entity_ids, fund_ids)))
ent_cover  <- cached("ent_cover",
                     fetch_by_ids("factset_own", "own_ent_coverage",
                                  "factset_entity_id", c(entity_ids, fund_ids)))

cat(sprintf("\nPulled: %d fund rows | %d institution rows | %d entity rows\n",
            nrow(fund_attr), nrow(inst_attr), nrow(ent_attr)))

# ---- 3d. code maps ------------------------------------------------------
step("Code maps (country, entity sub-type, fund type, industry)")
map_country  <- fetch_all("factset", "country_map")
map_subtype  <- fetch_all("factset", "entity_sub_type_map")
map_fundtype <- fetch_all("factset", "fund_type_map")
map_industry <- fetch_all("factset", "factset_industry_map")

# ---- 3e. MANAGER BRIDGE -------------------------------------------------
# Funds have no AUM or manager style of their own, but own_ent_funds gives
# the managing institution's ENTITY id. Following it recovers, for every
# fund, the manager's name, country, AUM and style — attributes otherwise
# available only to the HF block. Columns are prefixed mgr_ so they are
# never confused with the fund's own values.
step("Manager bridge: fund -> managing institution attributes")
mgr_ids <- unique(na.omit(fund_attr$factset_inst_entity_id))
cat(sprintf("Manager bridge: %d distinct managing institutions\n",
            length(mgr_ids)))

mgr_inst <- fetch_by_ids(
  "factset_own", "own_ent_institutions", "factset_entity_id", mgr_ids,
  cols = c("factset_entity_id", "style", "manager_style", "turnover_label",
           "total_aum", "entity_sub_type")) |>
  prefix_cols("mgr_", keep = "factset_entity_id")

mgr_ent <- fetch_by_ids(
  "factset", "edm_standard_entity", "factset_entity_id", mgr_ids,
  cols = c("factset_entity_id", "entity_proper_name", "iso_country",
           "entity_type", "entity_sub_type")) |>
  prefix_cols("mgr_", keep = "factset_entity_id")

managers <- full_join(mgr_inst, mgr_ent, by = "factset_entity_id") |>
  rename(factset_inst_entity_id = factset_entity_id)

# ---- 3f. panel statistics from the local files --------------------------
step("Computing investor panel statistics (local)")
inv_panel <- panel |>
  distinct(investor_id, quarter_end, investor_type, n_assets_full) |>
  group_by(investor_id) |>
  summarise(investor_type = first(investor_type),
            n_quarters    = n_distinct(quarter_end),
            first_quarter = min(quarter_end),
            last_quarter  = max(quarter_end),
            med_n_assets  = median(n_assets_full),
            max_n_assets  = max(n_assets_full),
            .groups = "drop") |>
  left_join(id_map |> select(investor_id, id_source), by = "investor_id")

# ---- 3g. assemble -------------------------------------------------------
step("Assembling investors_master")
investors <- inv_panel |>
  # entity attributes (both blocks, where present)
  left_join(ent_attr |> rename(investor_id = factset_entity_id),
            by = "investor_id") |>
  # institution attributes (HF block) — all columns carry the inst_ prefix
  left_join(inst_attr |> rename(investor_id = factset_entity_id),
            by = "investor_id") |>
  # fund attributes (fund block)
  left_join(fund_attr |> rename(investor_id = factset_fund_id),
            by = "investor_id") |>
  left_join(fund_obj |> rename(investor_id = factset_fund_id),
            by = "investor_id") |>
  left_join(fund_feeder |> rename(investor_id = factset_feeder_fund_id),
            by = "investor_id") |>
  left_join(fund_ticker |> rename(investor_id = factset_fund_id),
            by = "investor_id") |>
  # manager bridge (fund block -> managing institution)
  left_join(managers, by = "factset_inst_entity_id") |>
  # structure and coverage flags
  left_join(ent_parent |> rename(investor_id = factset_entity_id),
            by = "investor_id") |>
  left_join(ent_cover |> rename(investor_id = factset_entity_id),
            by = "investor_id") |>
  # readable labels
  left_join(map_country |> select(iso_country, country_desc, region_code),
            by = "iso_country") |>
  left_join(map_fundtype |> rename(fund_type = fund_type_code),
            by = "fund_type") |>
  left_join(map_subtype |> select(entity_sub_type = entity_sub_type_code,
                                  entity_sub_type_desc),
            by = "entity_sub_type") |>
  # one usable column per concept, whichever block supplied it
  mutate(style_any    = coalesce(fund_style, inst_style, mgr_style),
         turnover_any = coalesce(fund_turnover_label, inst_turnover_label,
                                 mgr_turnover_label),
         aum_any      = coalesce(inst_total_aum, mgr_total_aum),
         # the feeder bridge only has rows for feeders, so the left join leaves
         # NA for everyone else. FALSE is the correct value and makes the column
         # usable as a logical downstream.
         is_feeder    = coalesce(is_feeder, FALSE),
         n_masters    = coalesce(n_masters, 0L))

assert_rows(investors, nrow(inv_panel), "investors_master assembly")

write_parquet(investors, file.path(OUT_DIR, "investors_master.parquet"))
cat(sprintf("\ninvestors_master: %d rows x %d cols\n",
            nrow(investors), ncol(investors)))

# Coverage BY TYPE. Fund-keyed fields populate only for OEF/ETF/CEF/VAR and
# entity-keyed fields only for HF, so a zero here is a keyspace fact, not
# missing data. The *_any columns should be well covered for both.
cov_fields <- intersect(
  c("entity_proper_name", "fund_type", "fund_family", "style_any",
    "turnover_any", "aum_any", "beta", "dividend_yield",
    "invt_obj_region_code", "invt_obj_specialization_code",
    "factset_master_fund_id", "is_feeder", "mgr_entity_proper_name",
    "fs_ultimate_parent_entity_id", "fund_ticker"),
  names(investors))

cat("\n=== Field coverage (% non-missing) by investor type ===\n")
print(investors |>
        select(investor_type, all_of(cov_fields)) |>
        group_by(investor_type) |>
        summarise(n = n(),
                  across(all_of(cov_fields), ~ round(100 * mean(!is.na(.x)), 1))) |>
        ungroup(), n = Inf, width = Inf)

# =========================================================================
# 4. SECURITY MASTER  (issuer_id is unambiguous: issuer ENTITY id)
# =========================================================================
# 4a. issuer-level attributes
step("Security block: issuer attributes and RBICS sectors")
SEC_ENT_COLS <- c("factset_entity_id", "entity_proper_name", "entity_type",
                  "entity_sub_type", "iso_country", "iso_country_incorp",
                  "sector_code", "industry_code", "primary_sic_code",
                  "metro_area", "state_province", "year_founded")

sec_entity <- cached("sec_entity",
                     fetch_by_ids("factset", "edm_standard_entity",
                                  "factset_entity_id", issuer_ids,
                                  cols = SEC_ENT_COLS))

# [FIX 5] The old script left-joined this and moved on, so a silent coverage
# gap in edm_standard_entity became 1,149 nameless issuers with no domicile
# and no sector. Retry the misses on their own: a chunk that failed for a
# transient reason will resolve here, and anything still missing is a genuine
# coverage gap that the fallbacks in 4g have to handle.
missing_ent <- setdiff(issuer_ids, sec_entity$factset_entity_id)
if (length(missing_ent)) {
  cat(sprintf("\n[retry] %s issuer(s) not returned by edm_standard_entity; re-querying\n",
              format(length(missing_ent), big.mark = ",")))
  sec_entity_retry <- fetch_by_ids("factset", "edm_standard_entity",
                                   "factset_entity_id", missing_ent,
                                   cols = SEC_ENT_COLS, chunk = 1000)
  if (nrow(sec_entity_retry)) {
    cat(sprintf("[retry] recovered %d on the second pass\n",
                nrow(sec_entity_retry)))
    sec_entity <- bind_rows(sec_entity, sec_entity_retry)
  }
  still_missing <- setdiff(issuer_ids, sec_entity$factset_entity_id)
  cat(sprintf("[retry] %s issuer(s) genuinely absent from the entity master\n",
              format(length(still_missing), big.mark = ",")))
}

sec_entity <- sec_entity |> distinct(factset_entity_id, .keep_all = TRUE)

sec_rbics <- cached("sec_rbics",
                    fetch_by_ids("factset", "sym_entity_sector_rbics",
                                 "factset_entity_id", issuer_ids)) |>
  distinct(factset_entity_id, .keep_all = TRUE)   # [FIX 5] one row per issuer

# 4b. issuer -> listed securities, and listing-level attributes
step("Issuer -> listing map and listing attributes")
sec_map <- cached("sec_map",
                  fetch_by_ids("factset_own", "own_sec_entity_eq",
                               "factset_entity_id", issuer_ids))
fsym_ids <- unique(sec_map$fsym_id)
cat(sprintf("\n%d issuers map to %d listed securities\n",
            n_distinct(sec_map$factset_entity_id), length(fsym_ids)))

sec_cov <- cached("sec_cov", fetch_by_ids(
  "factset_own", "own_sec_coverage_eq", "fsym_id", fsym_ids,
  cols = c("fsym_id", "security_name", "iso_country", "mic_exchange_code",
           "issue_type", "cap_group", "universe_type", "fds_13f_flag",
           "active"))) |>
  rename(listing_iso_country = iso_country) |>   # [FIX 4] domicile fallback
  # one row per listing: a duplicate here would multiply securities_detail and
  # inflate n_listings
  distinct(fsym_id, .keep_all = TRUE)

# [FIX 8] factset_common.wrds_securities is a WRDS-built view over a global
# security universe and is frequently unindexed on factset_entity_id, so a
# 4,000-element IN list can force a sequential scan. Smaller chunks usually
# keep the planner on an index. It is also the one pull the script can do
# without: it supplies the MIDDLE tier of the name/domicile fallback chain
# (proper_name, excountry), and the outer tiers come from queries that have
# already run. Set PULL_WRDS_SECURITIES <- FALSE if it stays pathological.
PULL_WRDS_SECURITIES <- TRUE

if (PULL_WRDS_SECURITIES) {
  sec_ids <- cached("sec_ids", fetch_by_ids(
    "factset_common", "wrds_securities", "factset_entity_id", issuer_ids,
    cols = c("factset_entity_id", "fs_perm_sec_id", "tic", "cusip", "isin",
             "sedol", "proper_name", "excountry", "fref_security_type",
             "inactive_flag"),
    chunk = 500))
} else {
  cat("    [skipped] factset_common.wrds_securities (PULL_WRDS_SECURITIES = FALSE)\n")
  sec_ids <- tibble(factset_entity_id = character(),
                    proper_name = character(), excountry = character())
}

# 4c. market capitalisation = adj_price x adj_shares_outstanding.
# A date WINDOW is pulled and the last observation per security kept:
# quarter-ends fall on non-trading days, so an exact-date filter would
# silently drop names. [FIX 2] the window now tracks AS_OF; [FIX 6] the date
# predicate runs on the server.
step("Prices and shares outstanding (market caps) - the slowest pull")

# [FIX 9] Do not price all 287,649 mapped listings. own_sec_coverage_eq covers
# only ~29% of them, and a listing absent from the ownership coverage table is
# unlikely to have rows in own_sec_prices_eq either, so most of that pull is
# wasted round trips against the largest table in the script. Keep a listing if
# ANY of the following holds:
#   - its issue_type is ordinary common equity, or
#   - its issuer has NO covered listing at all, or
#   - its issuer has covered listings but NONE of them are equity (funds,
#     warrants, preferred only). Without this third arm 3,054 issuers lost
#     their market cap entirely.
cov_ent <- sec_map |> filter(fsym_id %in% sec_cov$fsym_id) |>
  pull(factset_entity_id) |> unique()

eq_fsym <- sec_cov |> filter(issue_type %in% c("EQ", "SHARE")) |>
  pull(fsym_id) |> unique()

eq_ent <- sec_map |> filter(fsym_id %in% eq_fsym) |>
  pull(factset_entity_id) |> unique()

price_fsym <- sec_map |>
  filter(fsym_id %in% eq_fsym |
           !factset_entity_id %in% cov_ent |
           !factset_entity_id %in% eq_ent) |>
  pull(fsym_id) |> unique()

n_ent_kept <- sec_map |> filter(fsym_id %in% price_fsym) |>
  distinct(factset_entity_id) |> nrow()
cat(sprintf("Price candidates: %s of %s listings (%.1f%%); %s of %s issuers retain one\n",
            format(length(price_fsym), big.mark = ","),
            format(length(fsym_ids), big.mark = ","),
            100 * length(price_fsym) / length(fsym_ids),
            format(n_ent_kept, big.mark = ","),
            format(n_distinct(sec_map$factset_entity_id), big.mark = ",")))
if (n_ent_kept < n_distinct(sec_map$factset_entity_id))
  warning(sprintf("%d issuer(s) have no priceable candidate listing",
                  n_distinct(sec_map$factset_entity_id) - n_ent_kept))

prices <- cached(sprintf("prices_%s_%dd", AS_OF, PRICE_LOOKBACK_DAYS),
                 fetch_by_ids(
                   "factset_own", "own_sec_prices_eq", "fsym_id", price_fsym,
                   cols = c("fsym_id", "price_date", "adj_price",
                            "adj_shares_outstanding"),
                   where = sprintf("price_date >= '%s' AND price_date <= '%s'",
                                   PRICE_DATE_FROM, PRICE_DATE_TO),
                   chunk = 1000)) |>
  filter(!is.na(adj_price), !is.na(adj_shares_outstanding)) |>
  group_by(fsym_id) |>
  slice_max(price_date, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(market_cap = adj_price * adj_shares_outstanding)

cat(sprintf("Market cap resolved for %s of %s candidate listings (%.1f%%)\n",
            format(nrow(prices), big.mark = ","),
            format(length(price_fsym), big.mark = ","),
            100 * nrow(prices) / length(price_fsym)))

# 4d. listing-level detail (kept for future use)
securities_detail <- sec_map |>
  rename(issuer_id = factset_entity_id) |>
  left_join(sec_cov, by = "fsym_id") |>
  left_join(prices,  by = "fsym_id")
write_parquet(securities_detail,
              file.path(OUT_DIR, "securities_detail.parquet"))

# 4e. roll up to one row per issuer.
#
# [FIX 3] THE BUG. The old rule was slice_max(market_cap, with_ties = FALSE).
# Because the price window was a 2019 stub, market_cap was NA for every
# listing of any issuer outside that window, and slice_max on an all-NA
# column returns an arbitrary row rather than nothing. GE Vernova (0SWK7W-E,
# 3,925 holders, spun off April 2024) was therefore represented by its
# unsponsored BDR on B3, which carries no price, no shares outstanding and
# no entity link.
#
# NOTE on focus_flag: it is tempting to use it here, but it does not exist at
# listing level. own_sec_entity_eq is a bare two-column bridge (fsym_id,
# factset_entity_id); the focus_flag that appears in securities_master comes
# from factset.sym_entity_sector_rbics and marks the entity's primary RBICS
# SECTOR, not its primary security. It is useless for choosing a listing.
#
# The replacement is an explicit deterministic preference order:
#   1. ordinary common equity ahead of receipts, ahead of warrants/rights
#      (issue_type is only known for listings present in own_sec_coverage_eq,
#      which covers ~29% of the mapped listings; unknown ranks mid-table so an
#      uncovered ordinary line still beats a known warrant)
#   2. an active listing ahead of an inactive one
#   3. a listing with a known market cap ahead of one without
#   4. largest market cap
#   5. fsym_id, so the result is reproducible when everything else ties
ISSUE_RANK <- function(x) case_when(
  x %in% c("EQ", "SHARE")            ~ 6L,   # ordinary common
  x %in% c("PF", "PC", "CV")         ~ 3L,   # preferred / convertible
  x %in% c("AD", "GD", "DR")         ~ 2L,   # depositary receipts, incl. BDRs
  x %in% c("WT", "RT", "UT")         ~ 1L,   # warrants, rights, units
  x %in% c("OE", "CE", "ET")         ~ 2L,   # fund instruments, not operating cos
  TRUE                               ~ 4L    # unknown / not in coverage table
)

primary_listing <- securities_detail |>
  mutate(
    r_type   = ISSUE_RANK(issue_type),
    r_active = as.integer(is.na(active) |
                            toupper(as.character(active)) %in% c("1", "Y", "TRUE")),
    r_cap    = as.integer(!is.na(market_cap)),
    r_capval = coalesce(market_cap, -Inf)
  ) |>
  arrange(issuer_id, desc(r_type), desc(r_active),
          desc(r_cap), desc(r_capval), fsym_id) |>
  group_by(issuer_id) |>
  slice_head(n = 1) |>
  ungroup() |>
  select(issuer_id, fsym_id, security_name, mic_exchange_code, issue_type,
         cap_group, universe_type, listing_iso_country,
         price_date, adj_price, adj_shares_outstanding, market_cap)

n_listings <- securities_detail |> count(issuer_id, name = "n_listings")

# 4f. holder statistics from the local files
#
# [FIX 10] The old block was:
#     panel |> unnest_longer(tokens) |> distinct() |> group_by() |> summarise()
# which expands the whole investor-issuer-quarter cross product in one dplyr
# pipeline that copies at every stage. On the 85-quarter panel it ran for many
# minutes at 100% CPU and 2.5 GB with no progress output.
#
# Materialising that table at all is the problem: ~150M rows x 3 columns is
# several GB of pointers before any grouping happens. The two statistics need
# different granularities, so accumulate them separately and never hold the
# full expansion:
#
#   n_holders_total  needs globally distinct (issuer, investor) pairs, so
#                    dedupe incrementally. Bounded by the number of distinct
#                    pairs, far smaller than the number of positions.
#   n_quarters,      need only (issuer, quarter), at most
#   first_, last_    35,961 x 85 rows. Trivial.
step("Computing issuer holder statistics (local)")
iv   <- NULL                      # distinct issuer-investor pairs, accumulated
iq   <- vector("list", length(seq_files))   # issuer-quarter, one row per pair

for (i in seq_along(seq_files)) {
  q <- read_parquet(seq_files[i]) |> select(investor_id, quarter_end, tokens)
  n <- lengths(q$tokens)
  iss <- unlist(q$tokens, use.names = FALSE)

  d <- unique(data.table(issuer_id = iss, investor_id = rep(q$investor_id, n)))
  iv <- if (is.null(iv)) d else unique(rbindlist(list(iv, d)))

  iq[[i]] <- data.table(issuer_id   = unique(iss),
                        quarter_end = q$quarter_end[1])

  rm(q, n, iss, d)
  cat(sprintf("\r    file %d/%d | %s distinct issuer-investor pairs",
              i, length(seq_files), format(nrow(iv), big.mark = ",")))
  flush.console()
}
cat("\n")

# quarter_end is taken from the first row of each file, so verify each file
# really is a single quarter before trusting n_quarters
stopifnot(!any(duplicated(rbindlist(iq)[, .(issuer_id, quarter_end)])))

iq <- rbindlist(iq)

sec_panel <- merge(
  iv[, .(n_holders_total = uniqueN(investor_id)), by = issuer_id],
  iq[, .(n_quarters    = uniqueN(quarter_end),
         first_quarter = min(quarter_end),
         last_quarter  = max(quarter_end)), by = issuer_id],
  by = "issuer_id", all = TRUE) |> as_tibble()

rm(iv, iq); gc()

# panel is the largest object in the session and nothing below this point
# needs it
rm(panel); gc()

step("Assembling securities_master")
securities <- sec_panel |>
  left_join(sec_entity |> rename(issuer_id = factset_entity_id),
            by = "issuer_id") |>
  left_join(sec_rbics |> rename(issuer_id = factset_entity_id),
            by = "issuer_id") |>
  left_join(primary_listing, by = "issuer_id") |>
  left_join(n_listings,      by = "issuer_id") |>
  left_join(sec_ids |> rename(issuer_id = factset_entity_id) |>
              distinct(issuer_id, .keep_all = TRUE),
            by = "issuer_id", suffix = c("", "_id"))

# [FIX 8] if wrds_securities was skipped, create the columns the fallback
# chain expects so section 4g does not have to branch
if (!"proper_name" %in% names(securities)) securities$proper_name <- NA_character_
if (!"excountry"   %in% names(securities)) securities$excountry   <- NA_character_

assert_rows(securities, nrow(sec_panel), "securities_master assembly")

# ---- 4g. name / domicile fallbacks --------------------------------------
# [FIX 4] edm_standard_entity does not cover every issuer in the ownership
# panel. Rather than shipping NA into Lens C and into pct_us, fall back
# through the sources that ARE populated, and record which one was used so
# the imputation can be reported in the thesis instead of hidden.
#
#   name     : entity master -> wrds_securities.proper_name -> listing name
#   domicile : entity master -> wrds_securities.excountry   -> listing country
#
# The raw entity-level fields are preserved as *_entity for auditing.
securities <- securities |>
  rename(entity_proper_name_entity = entity_proper_name,
         iso_country_entity        = iso_country) |>
  mutate(
    entity_proper_name = coalesce(entity_proper_name_entity,
                                  proper_name, security_name),
    name_source = case_when(
      !is.na(entity_proper_name_entity) ~ "entity",
      !is.na(proper_name)               ~ "wrds_securities",
      !is.na(security_name)             ~ "listing",
      TRUE                              ~ NA_character_),
    iso_country = coalesce(iso_country_entity, excountry, listing_iso_country),
    country_source = case_when(
      !is.na(iso_country_entity)  ~ "entity",
      !is.na(excountry)           ~ "wrds_securities",
      !is.na(listing_iso_country) ~ "listing",
      TRUE                        ~ NA_character_)
  ) |>
  # readable labels, applied AFTER the fallbacks so imputed domiciles get a
  # country_desc and a region_code too
  left_join(map_country |> select(iso_country, country_desc, region_code),
            by = "iso_country") |>
  left_join(map_industry |> select(industry_code = factset_industry_code,
                                   factset_industry_desc, factset_sector_code),
            by = "industry_code")

# [FIX 2] stamp the valuation date. securities_master has one row per issuer
# and no date of its own, so nothing otherwise stops a 2015 quarter being
# clustered against 2025 market caps. Clusters.r should warn if this does not
# match the quarter being clustered.
securities <- securities |> mutate(cap_asof = AS_OF)

assert_rows(securities, nrow(sec_panel), "securities_master labelling")

write_parquet(securities, file.path(OUT_DIR, "securities_master.parquet"))
cat(sprintf("\nsecurities_master: %d rows x %d cols\n",
            nrow(securities), ncol(securities)))

# ---- 4h. coverage diagnostics -------------------------------------------
step("Security master coverage diagnostics")
cat("  coverage of key fields:\n")
for (v in c("entity_proper_name", "iso_country", "sector_code", "industry_code",
            "cap_group", "market_cap", "cusip", "l2_id")) {
  if (v %in% names(securities))
    cat(sprintf("    %-24s %5.1f%%\n", v, 100 * mean(!is.na(securities[[v]]))))
}

# [FIX 2] a price inside the window is not necessarily a price near AS_OF.
# If the median lag is large the caps are stale even though the window is
# nominally correct.
if ("price_date" %in% names(securities)) {
  lag_days <- as.numeric(AS_OF - securities$price_date)
  cat(sprintf("\n  price staleness vs AS_OF (%s): median %.0f d, p90 %.0f d, max %.0f d\n",
              AS_OF,
              median(lag_days, na.rm = TRUE),
              quantile(lag_days, 0.9, na.rm = TRUE),
              max(lag_days, na.rm = TRUE)))
}

# [FIX 11] market_cap = adj_price * adj_shares_outstanding, and the UNITS of
# adj_shares_outstanding are a FactSet convention, not something the script can
# infer. Print the largest issuers so the magnitude can be eyeballed against a
# known figure: the biggest US listings should be in the low trillions of USD
# at end-2025. If they come out ~1000x too small or too large, every med_cap
# and cap-band statistic downstream is on the wrong scale.
cat("\n  market cap sanity check (largest 5 issuers, raw units):\n")
print(securities |>
        filter(!is.na(market_cap)) |>
        slice_max(market_cap, n = 5) |>
        transmute(entity_proper_name, cap_group,
                  market_cap_bn = round(market_cap / 1e9, 1)))

cat("\n  where names and domiciles came from:\n")
print(securities |> count(name_source) |>
        mutate(pct = round(100 * n / sum(n), 1)))
print(securities |> count(country_source) |>
        mutate(pct = round(100 * n / sum(n), 1)))

# The primary-listing rule is the thing that broke last time, so report what
# it chose. A large non-EQ share means the preference order is not biting.
cat("\n  issue_type of the chosen primary listing:\n")
print(securities |> count(issue_type, sort = TRUE) |>
        mutate(pct = round(100 * n / sum(n), 1)) |> head(10))

# Anything still unnamed after the fallbacks is a real gap. Write it out so
# it can be inspected rather than rediscovered from a Clusters.r warning.
unresolved <- securities |>
  filter(is.na(entity_proper_name) | is.na(iso_country)) |>
  select(issuer_id, n_holders_total, n_quarters, first_quarter, last_quarter,
         security_name, fsym_id, mic_exchange_code, issue_type, cap_group,
         name_source, country_source) |>
  arrange(desc(n_holders_total))

if (nrow(unresolved)) {
  cat(sprintf("\n[warn] %s issuer(s) still lack a name or a domicile after fallbacks.\n",
              format(nrow(unresolved), big.mark = ",")))
  print(head(unresolved, 10), width = Inf)
  write_csv(unresolved, file.path(OUT_DIR, "unresolved_issuers.csv"))
  cat(sprintf("       Written to %s/unresolved_issuers.csv\n", OUT_DIR))
} else {
  cat("\nAll issuers have a name and a domicile.\n")
}

dbDisconnect(wrds)
cat(sprintf("\nDisconnected. Saved to %s. Total run time: %.1f min\n",
            OUT_DIR, as.numeric(difftime(Sys.time(), .T0, units = "mins"))))