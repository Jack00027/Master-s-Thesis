### =========================================================================
### Explore_step2.r  — second offline pass over reference/wrds_catalog.csv
###
### Step 1 identified the tables worth using. This prints their exact
### columns so the pulls can be written without guessing. Still NO database
### connection.
### =========================================================================

library(tidyverse)

catalog <- read_csv("reference/wrds_catalog.csv", show_col_types = FALSE) |>
  as_tibble() |>
  filter(!str_detect(table_schema, "_old$"))

show <- function(schema, tbl) {
  rows <- catalog |>
    filter(table_schema == schema, table_name == tbl) |>
    arrange(ordinal_position)
  if (!nrow(rows)) { cat(sprintf("\n--- %s.%s : NOT FOUND ---\n", schema, tbl)); return(invisible()) }
  cat(sprintf("\n=== %s.%s  (%d cols) ===\n", schema, tbl, nrow(rows)))
  print(rows |> select(ordinal_position, column_name, data_type), n = Inf)
}

# ---- INVESTOR side --------------------------------------------------------
show("factset_own", "own_ent_institutions")      # institution attributes
show("factset_own", "own_ent_coverage")          # coverage window per entity
show("factset_own", "own_ent_fund_objectives")   # declared investment objective
show("factset_own", "own_ent_fund_managers")     # manager names
show("factset_own", "own_ent_fund_identifiers")  # fund identifiers
show("factset_own", "own_ent_inst_identifiers")  # institution identifiers
show("factset_own", "own_ent_funds_feeder_master")  # feeder -> master links
show("factset_own", "own_ent_13f_combined_inst")    # 13F filer structure
show("factset_own", "own_ent_13f_subfiler_inst")
show("factset",     "edm_standard_entity_structure")  # parent / ultimate parent

# ---- SECURITY side --------------------------------------------------------
show("factset_own", "own_sec_coverage_eq")       # security coverage
show("factset_own", "own_sec_prices_eq")         # price / shares / mkt cap
show("factset_own", "own_sec_map_eq")
show("factset_own", "own_sec_entity_hist_eq")
show("factset",     "sym_entity_sector_rbics")   # RBICS sector classification
show("factset",     "sym_entity_sector")
show("factset",     "edm_standard_entity_identifiers")
show("factset_common", "wrds_securities")        # cusip / isin / ticker master

# ---- HOLDINGS tables (for reference: what the pipeline already reads) -----
show("factset_own", "wrds_own_13f")
show("factset_own", "wrds_own_fund")
show("factset_own", "own_inst_13f_detail_eq")
show("factset_own", "own_fund_detail_eq")

# ---- code maps (decode fund_type / entity_type / sector codes) ------------
show("factset", "fund_type_map")
show("factset", "entity_type_map")
show("factset", "entity_sub_type_map")
show("factset", "factset_industry_map")
show("factset", "country_map")

# ---- which of your investor_ids look like fund ids vs entity ids? ---------
# The two keying schemes matter: own_ent_funds keys on factset_fund_id,
# edm_standard_entity on factset_entity_id. Check what the pipeline uses.
if (file.exists("data")) {
  f <- list.files("data", pattern = "^q_.*\\.parquet$", full.names = TRUE)
  if (length(f)) {
    library(arrow)
    d <- read_parquet(f[1])
    cat("\n=== sample ids from the local pipeline ===\n")
    cat("investor_id :", head(unique(d$investor_id), 5), "\n")
    cat("token/issuer:", head(unlist(d$tokens[1]), 5), "\n")
  }
}
