### With this file a create a 2 tables, one for the investors and one for the stocks, with the relevant information for each of them.

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
  user = "jack027", password = "bLw3!SGrhzL88$g"
)


SEQ_DIR         <- "data"            # local quarterly token files
OUT_DIR         <- "reference"
PRICE_DATE_FROM <- "2019-07-01"      # window for market caps (see §4c)
PRICE_DATE_TO   <- "2019-10-15"
dir.create(OUT_DIR, showWarnings = FALSE)


# ── progress reporting ────────────────────────────────────────────────────
.T0 <- Sys.time()
.STEP <- 0
.STEPS_TOTAL <- 14          # major operations, for the [i/n] counter

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
fetch_by_ids <- function(schema, table, id_col, ids, cols = "*", chunk = 4000) {
  ids <- unique(ids[!is.na(ids)])
  label <- paste(schema, table, sep = ".")
  if (!length(ids)) {
    cat(sprintf("    %-45s no ids to look up, skipped\n", label))
    return(tibble())
  }
  sel   <- if (identical(cols, "*")) "*" else paste(cols, collapse = ", ")
  parts <- split(ids, ceiling(seq_along(ids) / chunk))
  np    <- length(parts)
  t0    <- Sys.time()
  res   <- vector("list", np)
  n_rows <- 0L

  for (i in seq_len(np)) {
    res[[i]] <- dbGetQuery(
      wrds, sprintf("SELECT %s FROM %s.%s WHERE %s IN (%s)",
                    sel, schema, table, id_col,
                    paste0("'", parts[[i]], "'", collapse = ",")))
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

cat(sprintf(paste0("Local panel: %d quarters | %d investors | %d issuers\n",
                   "  entity-keyed (HF)            : %d\n",
                   "  fund-keyed (OEF/ETF/CEF/VAR) : %d\n"),
            n_distinct(panel$quarter_end), nrow(id_map), length(issuer_ids),
            length(entity_ids), length(fund_ids)))

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
fund_attr <- fetch_by_ids(
  "factset_own", "own_ent_funds", "factset_fund_id", fund_ids,
  cols = c("factset_fund_id", "factset_inst_entity_id", "fund_type", "style",
           "turnover_label", "pe_ratio", "pb_ratio", "dividend_yield",
           "sales_growth", "price_momentum", "relative_strength", "beta",
           "fund_family", "etf_type", "active_flag", "current_report_date")) |>
  # explicit names for the fields that exist in more than one source
  rename(fund_style          = style,
         fund_turnover_label = turnover_label,
         fund_report_date    = current_report_date)

# declared mandate: objective, specialisation, asset type, region, country.
# The declared-strategy counterpart to the holdings-based lenses.
fund_obj <- fetch_by_ids("factset_own", "own_ent_fund_objectives",
                         "factset_fund_id", fund_ids)

# feeder -> master: feeder funds hold mechanically identical portfolios,
# so a cluster of feeders is an administrative artifact, not a strategy.
fund_feeder <- fetch_by_ids("factset_own", "own_ent_funds_feeder_master",
                            "factset_feeder_fund_id", fund_ids)

fund_ticker <- fetch_by_ids("factset_own", "own_ent_fund_identifiers",
                            "factset_fund_id", fund_ids) |>
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
ent_attr <- fetch_by_ids(
  "factset", "edm_standard_entity", "factset_entity_id",
  c(entity_ids, fund_ids),
  cols = c("factset_entity_id", "entity_proper_name", "entity_type",
           "entity_sub_type", "iso_country", "iso_country_incorp",
           "sector_code", "industry_code", "primary_sic_code",
           "metro_area", "state_province", "year_founded", "web_site"))

ent_parent <- fetch_by_ids("factset", "edm_standard_entity_structure",
                           "factset_entity_id", c(entity_ids, fund_ids))
ent_cover  <- fetch_by_ids("factset_own", "own_ent_coverage",
                           "factset_entity_id", c(entity_ids, fund_ids))

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
         aum_any      = coalesce(inst_total_aum, mgr_total_aum))

# a left_join against a non-unique key would silently duplicate investors
if (nrow(investors) != nrow(inv_panel)) {
  warning(sprintf("row count changed in assembly: %d -> %d (duplicate keys in a joined table)",
                  nrow(inv_panel), nrow(investors)))
}

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
    "factset_master_fund_id", "mgr_entity_proper_name",
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
sec_entity <- fetch_by_ids(
  "factset", "edm_standard_entity", "factset_entity_id", issuer_ids,
  cols = c("factset_entity_id", "entity_proper_name", "entity_type",
           "entity_sub_type", "iso_country", "iso_country_incorp",
           "sector_code", "industry_code", "primary_sic_code",
           "metro_area", "state_province", "year_founded"))

sec_rbics <- fetch_by_ids("factset", "sym_entity_sector_rbics",
                          "factset_entity_id", issuer_ids)

# 4b. issuer -> listed securities, and listing-level attributes
step("Issuer -> listing map and listing attributes")
sec_map <- fetch_by_ids("factset_own", "own_sec_entity_eq",
                        "factset_entity_id", issuer_ids)
fsym_ids <- unique(sec_map$fsym_id)
cat(sprintf("\n%d issuers map to %d listed securities\n",
            n_distinct(sec_map$factset_entity_id), length(fsym_ids)))

sec_cov <- fetch_by_ids(
  "factset_own", "own_sec_coverage_eq", "fsym_id", fsym_ids,
  cols = c("fsym_id", "security_name", "iso_country", "mic_exchange_code",
           "issue_type", "cap_group", "universe_type", "fds_13f_flag",
           "active"))

sec_ids <- fetch_by_ids(
  "factset_common", "wrds_securities", "factset_entity_id", issuer_ids,
  cols = c("factset_entity_id", "fs_perm_sec_id", "tic", "cusip", "isin",
           "sedol", "proper_name", "excountry", "fref_security_type",
           "inactive_flag"))

# 4c. market capitalisation = adj_price x adj_shares_outstanding.
# A date WINDOW is pulled and the last observation per security kept:
# quarter-ends fall on non-trading days, so an exact-date filter would
# silently drop names.
step("Prices and shares outstanding (market caps) - the slowest pull")
prices <- fetch_by_ids(
  "factset_own", "own_sec_prices_eq", "fsym_id", fsym_ids,
  cols = c("fsym_id", "price_date", "adj_price", "adj_shares_outstanding")) |>
  filter(price_date >= as.Date(PRICE_DATE_FROM),
         price_date <= as.Date(PRICE_DATE_TO)) |>
  group_by(fsym_id) |>
  slice_max(price_date, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(market_cap = adj_price * adj_shares_outstanding)

# 4d. listing-level detail (kept for future use)
securities_detail <- sec_map |>
  rename(issuer_id = factset_entity_id) |>
  left_join(sec_cov, by = "fsym_id") |>
  left_join(prices,  by = "fsym_id")
write_parquet(securities_detail,
              file.path(OUT_DIR, "securities_detail.parquet"))

# 4e. roll up to one row per issuer, largest listing treated as primary
primary_listing <- securities_detail |>
  group_by(issuer_id) |>
  slice_max(market_cap, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(issuer_id, fsym_id, security_name, mic_exchange_code, issue_type,
         cap_group, universe_type, adj_price, adj_shares_outstanding,
         market_cap)

n_listings <- securities_detail |> count(issuer_id, name = "n_listings")

# 4f. holder statistics from the local files
step("Computing issuer holder statistics (local)")
sec_panel <- panel |>
  select(investor_id, quarter_end, tokens) |>
  unnest_longer(tokens, values_to = "issuer_id") |>
  distinct(quarter_end, investor_id, issuer_id) |>
  group_by(issuer_id) |>
  summarise(n_holders_total = n_distinct(investor_id),
            n_quarters      = n_distinct(quarter_end),
            first_quarter   = min(quarter_end),
            last_quarter    = max(quarter_end),
            .groups = "drop")

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
            by = "issuer_id", suffix = c("", "_id")) |>
  left_join(map_country |> select(iso_country, country_desc, region_code),
            by = "iso_country") |>
  left_join(map_industry |> select(industry_code = factset_industry_code,
                                   factset_industry_desc, factset_sector_code),
            by = "industry_code")

write_parquet(securities, file.path(OUT_DIR, "securities_master.parquet"))
cat(sprintf("\nsecurities_master: %d rows x %d cols\n",
            nrow(securities), ncol(securities)))
cat("  coverage of key fields:\n")
for (v in c("entity_proper_name", "iso_country", "sector_code", "cap_group",
            "market_cap", "cusip", "l2_id")) {
  if (v %in% names(securities))
    cat(sprintf("    %-24s %5.1f%%\n", v, 100 * mean(!is.na(securities[[v]]))))
}

dbDisconnect(wrds)
cat(sprintf("\nDisconnected. Saved to %s. Total run time: %.1f min\n",
            OUT_DIR, as.numeric(difftime(Sys.time(), .T0, units = "mins"))))