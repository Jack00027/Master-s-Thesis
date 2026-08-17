### =========================================================================
### Explore_catalog.r
### Offline exploration of reference/wrds_catalog.csv (written by pass 1 of
### Investor_data.r). NO database connection needed — everything here reads
### the local CSV, so it costs nothing and cannot lock the account.
###
### Goal: find out exactly which tables/columns hold
###   (a) investor attributes   (names, type, country, family)
###   (b) issuer attributes     (names, country, sector, identifiers)
###   (c) anything extra worth keeping (prices, shares, classifications)
### =========================================================================

library(tidyverse)

catalog <- read_csv("reference/wrds_catalog.csv", show_col_types = FALSE) |>
  as_tibble() |>
  filter(!str_detect(table_schema, "_old$"))    # drop duplicated *_old schemas

cat(sprintf("Catalog: %d columns | %d tables | %d schemas\n\n",
            nrow(catalog),
            n_distinct(paste(catalog$table_schema, catalog$table_name)),
            n_distinct(catalog$table_schema)))

# =========================================================================
# 1. The ownership schema in full — only 34 tables, worth seeing all of them
# =========================================================================
own_tables <- catalog |>
  filter(table_schema == "factset_own") |>
  group_by(table_name) |>
  summarise(n_cols = n(),
            columns = paste(column_name, collapse = ", "),
            .groups = "drop") |>
  arrange(table_name)

cat("=== factset_own: tables and column counts ===\n")
print(own_tables |> select(table_name, n_cols), n = Inf)

# full column lists written to a readable text file
sink("reference/factset_own_columns.txt")
for (i in seq_len(nrow(own_tables))) {
  cat("\n===", own_tables$table_name[i],
      sprintf("(%d cols)\n", own_tables$n_cols[i]))
  cat(str_wrap(own_tables$columns[i], 100), "\n")
}
sink()
cat("\nFull factset_own column lists -> reference/factset_own_columns.txt\n")

# =========================================================================
# 2. Keyword hunt: which tables carry the attributes we want?
# =========================================================================
find_cols <- function(pattern, label, schemas = NULL, max_show = 25) {
  hits <- catalog |>
    filter(str_detect(column_name, regex(pattern, ignore_case = TRUE)))
  if (!is.null(schemas)) hits <- hits |> filter(table_schema %in% schemas)
  cat(sprintf("\n=== %s  [pattern: %s] — %d hits ===\n",
              label, pattern, nrow(hits)))
  print(hits |>
          count(table_schema, table_name, name = "n_matching_cols") |>
          arrange(desc(n_matching_cols)) |>
          head(max_show), n = max_show)
  invisible(hits)
}

key_schemas <- c("factset", "factset_own", "factset_common")

find_cols("entity_proper_name|entity_name",      "ENTITY NAMES",        key_schemas)
find_cols("iso_country|country",                 "COUNTRY",             key_schemas)
find_cols("sector|industry|rbics|sic",           "SECTOR / INDUSTRY",   key_schemas)
find_cols("entity_type|entity_sub_type|fund_type|institution_type",
                                                 "ENTITY / FUND TYPE",  key_schemas)
find_cols("cusip|isin|sedol|ticker",             "SECURITY IDENTIFIERS", key_schemas)
find_cols("price|mkt_val|market_value|market_cap|shares_out",
                                                 "PRICE / SIZE",        key_schemas)
find_cols("parent|family|ultimate",              "PARENT / FAMILY",     key_schemas)
find_cols("style|strategy|orientation|turnover", "STYLE / STRATEGY",    key_schemas)
find_cols("adj_holding|adj_mv|position|holding", "HOLDINGS / POSITIONS", key_schemas)

# =========================================================================
# 3. Candidate master tables (one row per entity / security)
# =========================================================================
cat("\n=== Candidate ENTITY master tables ===\n")
print(catalog |>
  filter(table_schema %in% key_schemas,
         str_detect(table_name, "ent_|entity|sym_coverage|sym_entity"),
         str_detect(column_name, "entity_id")) |>
  count(table_schema, table_name, name = "n_cols") |>
  arrange(table_schema, table_name), n = 40)

# =========================================================================
# 4. Inspect one table in detail — change the name and re-run
# =========================================================================
show_table <- function(schema, tbl) {
  cat(sprintf("\n=== %s.%s ===\n", schema, tbl))
  print(catalog |>
    filter(table_schema == schema, table_name == tbl) |>
    select(ordinal_position, column_name, data_type), n = Inf)
}

show_table("factset_own", "own_ent_funds")
show_table("factset_own", "own_sec_entity_eq")
show_table("factset",     "edm_standard_entity")
