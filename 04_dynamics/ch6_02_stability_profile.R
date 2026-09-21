# =============================================================================
# ch6_02_stability_profile.R -- who has stable holdings (one year pair)
#
# Descriptive companion to ch6_01_align.R for one pair, default Q4 2024 ->
# Q4 2025:
#   - distribution of wt_overlap
#   - wt_overlap by declared style, fund type, turnover class and Chapter 5
#     component (turnover is never used to compute wt_overlap, so its
#     ordering is an independent check of the measure)
#   - the most and least stable investors
#   - the three anchor rules for this pair, and displacement against
#     stability under the default rule
# Requires ch6_01_align.R to have been run.
# =============================================================================
if (!exists("CH6_SETUP_LOADED")) source("ch6_00_setup.R")

PY0 <- if (exists("PY0")) PY0 else 2024
PY1 <- if (exists("PY1")) PY1 else 2025
stopifnot(PY1 == PY0 + 1, PY0 %in% YEARS, PY1 %in% YEARS)
tag <- sprintf("%d_%d", PY0, PY1)

rot <- readRDS(file.path(CACHE_DIR, "rotations.rds"))
X0 <- load_embeddings(PY0); X1 <- load_embeddings(PY1)
H0 <- load_holdings(PY0);   H1 <- load_holdings(PY1)
ids <- Reduce(intersect, list(rownames(X0), rownames(X1),
                              names(H0$n_full), names(H1$n_full)))
s <- pair_stability(PY0, PY1, ids, H0, H1)

stab <- data.frame(investor_id = ids, wt_overlap = as.numeric(s), stringsAsFactors = FALSE)

if (file.exists(INV_MASTER)) {
  im <- as.data.frame(arrow::read_parquet(INV_MASTER))
  im$investor_id <- as.character(im$investor_id)
  keep <- intersect(c("investor_id", "entity_proper_name", "fund_type_desc",
                      "style_any", "turnover_any"), names(im))
  stab <- merge(stab, im[!duplicated(im$investor_id), keep],
                by = "investor_id", all.x = TRUE, sort = FALSE)
}
if (PY1 == REF_YEAR && file.exists(CH5_ASSIGN_FILE)) {
  a5 <- readRDS(CH5_ASSIGN_FILE)$cluster
  stab$component <- a5$cluster[match(stab$investor_id, as.character(a5$investor_id))]
}

# displacement under each rule's rotation for this pair
t_idx <- match(PY0, YEARS)
for (r in RULES) {
  R <- rot$seq[[r]][[t_idx]]
  stab[[paste0("disp_", r)]] <- 1 - rowSums((X0[stab$investor_id, , drop = FALSE] %*% R) *
                                            X1[stab$investor_id, , drop = FALSE])
}
write.csv(stab, out_file("stability_", tag, ".csv"), row.names = FALSE)

# ---- distribution ------------------------------------------------------------
msg("== %d -> %d: wt_overlap for %d investors ==", PY0, PY1, nrow(stab))
print(round(quantile(stab$wt_overlap, c(.1, .25, .5, .75, .9)), 3))
msg("share >= %.2f: %.3f (%d investors) | exactly 0: %d",
    STAB_THRESHOLD, mean(stab$wt_overlap >= STAB_THRESHOLD),
    sum(stab$wt_overlap >= STAB_THRESHOLD), sum(stab$wt_overlap == 0))

# ---- by group ----------------------------------------------------------------
by_group <- function(var, min_n = 50) {
  if (!var %in% names(stab)) return(NULL)
  g <- stab[!is.na(stab[[var]]), ]
  out <- do.call(rbind, lapply(split(g, g[[var]]), function(d) data.frame(
    variable = var, level = as.character(d[[var]][1]), n = nrow(d),
    median_wt_overlap = median(d$wt_overlap),
    share_stable = mean(d$wt_overlap >= STAB_THRESHOLD),
    n_stable = sum(d$wt_overlap >= STAB_THRESHOLD))))
  out <- out[out$n >= min_n, ]
  out[order(-out$median_wt_overlap), ]
}
groups <- do.call(rbind, lapply(c("style_any", "fund_type_desc", "turnover_any", "component"),
                                by_group))
write.csv(groups, out_file("stability_by_group_", tag, ".csv"), row.names = FALSE)
msg("\n== wt_overlap by group ==")
print(format(groups, digits = 3), row.names = FALSE)

# ---- extremes ------------------------------------------------------------------
show <- intersect(c("investor_id", "entity_proper_name", "style_any", "fund_type_desc",
                    "component", "wt_overlap"), names(stab))
o <- order(-stab$wt_overlap)
msg("\n== 20 most stable ==");  print(format(head(stab[o, show], 20), digits = 3), row.names = FALSE)
msg("\n== 20 least stable =="); print(format(tail(stab[o, show], 20), digits = 3), row.names = FALSE)

# ---- rules for this pair -------------------------------------------------------
d <- rot$diagnostics
d <- d[d$year_from == PY0, ]
msg("\n== anchor rules, %d -> %d ==", PY0, PY1)
print(format(d[, -(1:2)], digits = 3), row.names = FALSE)

disp_def <- stab[[paste0("disp_", ANCHOR_RULE)]]
msg("\nUnder %s: median displacement %.3f for stable investors, %.3f for the rest; Spearman(displacement, wt_overlap) = %.3f",
    ANCHOR_RULE,
    median(disp_def[stab$wt_overlap >= STAB_THRESHOLD]),
    median(disp_def[stab$wt_overlap <  STAB_THRESHOLD]),
    cor(disp_def, stab$wt_overlap, method = "spearman"))

if ("component" %in% names(stab)) {
  comp_disp <- do.call(rbind, lapply(split(stab, stab$component), function(x) data.frame(
    component = x$component[1], n = nrow(x),
    n_stable = sum(x$wt_overlap >= STAB_THRESHOLD),
    t(sapply(RULES, function(r) median(x[[paste0("disp_", r)]]))))))
  names(comp_disp)[-(1:3)] <- paste0("median_disp_", RULES)
  write.csv(comp_disp, out_file("displacement_by_component_", tag, ".csv"), row.names = FALSE)
  msg("\n== median displacement by component under each rule ==")
  print(format(comp_disp, digits = 3), row.names = FALSE)
}

# ---- figures -------------------------------------------------------------------
if (have_gg) {
  library(ggplot2)
  g1 <- ggplot(stab, aes(wt_overlap)) + geom_histogram(bins = 50) +
    geom_vline(xintercept = STAB_THRESHOLD, linetype = "dashed") +
    labs(x = sprintf("wt_overlap, Q4 %d vs Q4 %d", PY0, PY1), y = "investors") +
    theme_minimal(base_size = 10)
  ggsave(out_file("fig_stability_hist_", tag, ".pdf"), g1, width = 7, height = 4)

  if ("style_any" %in% names(stab)) {
    g2 <- ggplot(stab[!is.na(stab$style_any), ],
                 aes(wt_overlap, reorder(style_any, wt_overlap, median))) +
      geom_boxplot(outlier.size = .3) + labs(x = "wt_overlap", y = NULL) +
      theme_minimal(base_size = 10)
    ggsave(out_file("fig_stability_by_style_", tag, ".pdf"), g2, width = 7, height = 4)
  }
  if ("component" %in% names(stab)) {
    g3 <- ggplot(stab[!is.na(stab$component), ], aes(wt_overlap, factor(component))) +
      geom_boxplot(outlier.size = .3) + labs(x = "wt_overlap", y = "component") +
      theme_minimal(base_size = 10)
    ggsave(out_file("fig_stability_by_component_", tag, ".pdf"), g3, width = 7, height = 4)
  }
  g4 <- ggplot(stab, aes(wt_overlap, disp_def)) + geom_point(alpha = .1, size = .4) +
    geom_smooth(se = FALSE, method = "gam", formula = y ~ s(x, bs = "cs")) +
    geom_vline(xintercept = STAB_THRESHOLD, linetype = "dashed") +
    labs(x = "holdings stability (wt_overlap)",
         y = sprintf("displacement after alignment (%s)", ANCHOR_RULE)) +
    theme_minimal(base_size = 10)
  ggsave(out_file("fig_stability_vs_displacement_", tag, ".pdf"), g4, width = 7, height = 4.5)
}
msg("\nDone -> %s", OUT_DIR)
