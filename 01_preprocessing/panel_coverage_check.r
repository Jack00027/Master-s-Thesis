# =============================================================================
# panel_coverage_check.r — how many investors survive across Q4s?
#
# Reads embedding files only. No clustering, no EM. Run this before
# Cluster_Dynamics.r to see whether the panel supports the analysis.
#
# The three numbers that matter, and what each governs:
#   adjacent-year overlap  -> sequential chaining (the default). Needs to be
#                             large; this is the one that must hold.
#   overlap with REF_YEAR  -> CHAIN_MODE = "anchored". Usually thinner.
#   present in ALL years   -> the alluvial's balanced panel, and the
#                             survivorship composition of long-horizon ARI.
# =============================================================================

library(arrow); library(dplyr); library(purrr); library(tidyr); library(ggplot2)

ARM          <- "weighted"
YEARS        <- 2005:2025
REF_YEAR     <- 2015
ANCHOR_YEARS <- c(2007, 2011, 2015, 2019, 2023, 2025)
US_ONLY      <- FALSE
US_THRESHOLD <- 0.90

quarter_of <- function(y) sprintf("%d-12-31", y)
emb_path <- function(q) switch(
  ARM,
  baseline = sprintf("embeddings_v2/q_%s.parquet", q),
  weighted = sprintf("embeddings_weighted/q_%s__weighted_all-chunks.parquet", q),
  stop("ARM must be 'baseline' or 'weighted'"))

# ---- investor ids per year (applying the same US filter as the main script) --
ids_by_year <- map(YEARS, function(y) {
  q <- quarter_of(y)
  f <- emb_path(q)
  if (!file.exists(f)) { message(sprintf("missing: %s", f)); return(character(0)) }
  ids <- read_parquet(f, col_select = "investor_id")$investor_id

  if (US_ONLY) {
    sec <- read_parquet("reference/securities_master.parquet") |>
      select(issuer_id, iso_country)
    keep <- read_parquet(sprintf("data/q_%s.parquet", q)) |>
      filter(chunk_id == 1) |> select(investor_id, tokens) |>
      unnest_longer(tokens, values_to = "issuer_id") |>
      left_join(sec, by = "issuer_id") |>
      group_by(investor_id) |>
      summarise(us = mean(iso_country == "US", na.rm = TRUE), .groups = "drop") |>
      filter(us >= US_THRESHOLD) |> pull(investor_id)
    ids <- intersect(ids, keep)
  }
  ids
})
names(ids_by_year) <- as.character(YEARS)

cat("\n=== Investors per year ===\n")
print(tibble(year = YEARS, n = lengths(ids_by_year)), n = Inf)

# ---- 1. adjacent-year overlap: governs sequential chaining -------------------
adj <- map_dfr(2:length(YEARS), function(t) {
  a <- ids_by_year[[t-1]]; b <- ids_by_year[[t]]
  tibble(from = YEARS[t-1], to = YEARS[t],
         n_from = length(a), n_to = length(b),
         n_common = length(intersect(a, b)),
         share_of_from = length(intersect(a, b)) / max(1, length(a)))
})
cat("\n=== Adjacent-year overlap (sequential chaining) ===\n")
print(adj |> mutate(share_of_from = round(share_of_from, 3)), n = Inf)
cat(sprintf("\nSmallest adjacent common set: %s investors (%d -> %d)\n",
            format(min(adj$n_common), big.mark = ","),
            adj$from[which.min(adj$n_common)], adj$to[which.min(adj$n_common)]))

# ---- 2. overlap with the reference year: governs anchored chaining -----------
ref <- ids_by_year[[as.character(REF_YEAR)]]
anch <- tibble(year = YEARS,
               n_common_with_ref = map_int(ids_by_year, ~length(intersect(.x, ref))))
cat(sprintf("\n=== Overlap with REF_YEAR = %d (anchored chaining) ===\n", REF_YEAR))
print(anch, n = Inf)
cat("Use anchored mode only if these stay comfortably large at both ends.\n")

# ---- 3. survivorship: balanced panels ---------------------------------------
all_years  <- reduce(ids_by_year, intersect)
anchor_bal <- reduce(ids_by_year[as.character(ANCHOR_YEARS)], intersect)
cat(sprintf("\n=== Balanced panels ===\nPresent in all %d years: %s\nPresent in all %d anchor years: %s\n",
            length(YEARS), format(length(all_years), big.mark = ","),
            length(ANCHOR_YEARS), format(length(anchor_bal), big.mark = ",")))
if (length(anchor_bal) < 500)
  cat("  -> thin for an alluvial. Widen the spacing or use fewer anchor years.\n")

# ---- 4. horizon coverage: the survivorship bias in ARI decay ----------------
# At large h the sample tilts toward long-lived investors, who are exactly the
# most stable ones. The decay curve then flattens for reasons unrelated to
# structural persistence.
hz <- map_dfr(1:10, function(h) {
  pairs <- YEARS[YEARS + h <= max(YEARS)]
  n <- map_int(pairs, ~length(intersect(ids_by_year[[as.character(.x)]],
                                        ids_by_year[[as.character(.x + h)]])))
  tibble(h = h, n_pairs = length(pairs), median_common = median(n),
         min_common = min(n))
})
cat("\n=== Common investors by horizon (ARI decay sample) ===\n")
print(hz, n = Inf)
cat("Watch median_common shrink with h. Where it gets small, the decay curve\n",
    "is measuring survivors, not structure. Report h=1 persistence on the same\n",
    "surviving subsample to quantify the bias.\n", sep = "")

ggsave("dynamics/panel_coverage.pdf",
       adj |> select(to, n_common, share_of_from) |>
         pivot_longer(-to) |>
         ggplot(aes(to, value)) + geom_line() + geom_point(size = 1) +
         facet_wrap(~ name, ncol = 1, scales = "free_y") +
         labs(x = NULL, y = NULL, title = "Adjacent-year investor overlap") +
         theme_minimal(base_size = 9),
       width = 8, height = 5)

cat("\nDone.\n")
