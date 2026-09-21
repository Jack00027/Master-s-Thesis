# =============================================================================
# ch6_01_align.R -- investor-anchored Procrustes between consecutive years
#
# For every pair of consecutive years (Q4 t -> Q4 t+1):
#
#   1. Holdings stability of each investor present in both years:
#      wt_overlap = sum_a min(w0_a, w1_a) over the two top-62 slices.
#
#   2. Anchors, three rules:
#        stable_thr    wt_overlap >= STAB_THRESHOLD              (default)
#        stable_strat  top STAB_TOP_SHARE by wt_overlap within each stratum
#                      (Chapter 5 components when t+1 = REF_YEAR, otherwise a
#                      spherical k-means with K groups on the t+1 embeddings)
#        all           every investor present in both years     (benchmark)
#
#   3. Evaluation. A random HOLDOUT share of investors (E) is removed before
#      any rule selects anchors, so every rule is scored on the same unseen
#      investors:
#        cos_E_stable     mean cosine after rotation, members of E with
#                         wt_overlap >= STAB_THRESHOLD
#        cos_E_all        mean cosine after rotation, all members of E
#        null_E_all       cos_E_all when the anchor correspondence is
#                         shuffled before fitting (chance level)
#        eff_rank         effective number of directions the anchors span
#        median_disp_all  median 1 - cos over all common investors under the
#                         final rotation
#
#   4. Final rotation for each rule, fitted on the rule's anchors among ALL
#      common investors.
#
# Rotations into the reference frame are chains of consecutive rotations, so
# no pair of distant years ever needs anchors of its own.
#
# Output
#   cache/rotations.rds          seq[[rule]][[t]]  YEARS[t] -> YEARS[t+1]
#                                chain[[rule]][[y]] y -> REF_YEAR
#   anchor_rules_by_pair.csv     the five metrics per pair and rule
# =============================================================================
if (!exists("CH6_SETUP_LOADED")) source("ch6_00_setup.R")

nY   <- length(YEARS)
iref <- match(REF_YEAR, YEARS)

ch5_labels <- NULL
if (file.exists(CH5_ASSIGN_FILE)) {
  a5 <- readRDS(CH5_ASSIGN_FILE)$cluster
  ch5_labels <- setNames(a5$cluster, as.character(a5$investor_id))
} else {
  msg("[note] %s not found: stable_strat uses k-means strata in every pair", CH5_ASSIGN_FILE)
}

top_share <- function(v, share) v >= quantile(v, 1 - share, na.rm = TRUE)

# apply a rule to a candidate set; s = wt_overlap, st = strata (named)
select_rule <- function(rule, cand, s, st) {
  switch(rule,
    all = list(ids = cand, fallback = FALSE),
    stable_thr = {
      a <- cand[s[cand] >= STAB_THRESHOLD]
      if (length(a) >= MIN_ANCHORS) list(ids = a, fallback = FALSE)
      else list(ids = cand[order(-s[cand])][seq_len(min(MIN_ANCHORS, length(cand)))],
                fallback = TRUE)
    },
    stable_strat = list(ids = unlist(lapply(split(cand, st[cand]), function(v)
                          v[top_share(s[v], STAB_TOP_SHARE)]), use.names = FALSE),
                        fallback = FALSE),
    stop("unknown rule ", rule))
}

mean_cos <- function(X0, X1, R, ids)
  if (length(ids)) mean(rowSums((X0[ids, , drop = FALSE] %*% R) * X1[ids, , drop = FALSE])) else NA_real_

seqR <- setNames(lapply(RULES, function(r) vector("list", nY - 1)), RULES)
diag_rows <- list()
X1 <- load_embeddings(YEARS[1])

for (t in seq_len(nY - 1)) {
  y0 <- YEARS[t]; y1 <- YEARS[t + 1]
  X0 <- X1; X1 <- load_embeddings(y1)
  H0 <- load_holdings(y0); H1 <- load_holdings(y1)

  ids <- Reduce(intersect, list(rownames(X0), rownames(X1),
                                names(H0$n_full), names(H1$n_full)))
  s <- pair_stability(y0, y1, ids, H0, H1)

  if (y1 == REF_YEAR && !is.null(ch5_labels)) {
    st <- ch5_labels[ids]; st[is.na(st)] <- 0L
  } else {
    st <- sph_kmeans(X1[ids, , drop = FALSE], K)
  }
  names(st) <- ids

  set.seed(SEED + y0)
  E        <- sample(ids, round(HOLDOUT * length(ids)))
  E_stable <- E[s[E] >= STAB_THRESHOLD]
  pool     <- setdiff(ids, E)

  msg("\n== %d -> %d: %d common investors, %d stable (%.1f%%), %d held out (%d stable)",
      y0, y1, length(ids), sum(s >= STAB_THRESHOLD), 100 * mean(s >= STAB_THRESHOLD),
      length(E), length(E_stable))

  for (rule in RULES) {
    ev <- select_rule(rule, pool, s, st)
    a  <- ev$ids
    R_ev <- procrustes(X0[a, , drop = FALSE], X1[a, , drop = FALSE])
    null <- mean(replicate(N_PERM_NULL, {
      Rn <- procrustes(X0[a, , drop = FALSE], X1[sample(a), , drop = FALSE])
      mean_cos(X0, X1, Rn, E)
    }))

    fin <- select_rule(rule, ids, s, st)
    R   <- procrustes(X0[fin$ids, , drop = FALSE], X1[fin$ids, , drop = FALSE])
    stopifnot(max(abs(crossprod(R) - diag(ncol(R)))) < 1e-8)
    seqR[[rule]][[t]] <- R

    row <- data.frame(
      year_from = y0, year_to = y1, rule = rule,
      n_anchor = length(fin$ids), fallback = fin$fallback,
      eff_rank = eff_rank(X0[a, , drop = FALSE]),
      cos_E_stable = if (length(E_stable) >= 5) mean_cos(X0, X1, R_ev, E_stable) else NA_real_,
      cos_E_all = mean_cos(X0, X1, R_ev, E),
      null_E_all = null,
      median_disp_all = median(1 - rowSums((X0[ids, , drop = FALSE] %*% R) *
                                           X1[ids, , drop = FALSE])),
      stringsAsFactors = FALSE)
    diag_rows[[length(diag_rows) + 1]] <- row
    with(row, msg("  %-12s n=%6d%s  eff_rank %5.1f  cos stable %.3f  cos all %.3f  null %.3f  median disp %.3f",
                  rule, n_anchor, if (fallback) " (fallback)" else "", eff_rank,
                  cos_E_stable, cos_E_all, null_E_all, median_disp_all))
    if (fin$fallback)
      warning(sprintf("%d->%d: fewer than %d investors with wt_overlap >= %.2f; stable_thr used the %d most stable",
                      y0, y1, MIN_ANCHORS, STAB_THRESHOLD, MIN_ANCHORS))
  }
}

diag_tab <- do.call(rbind, diag_rows)
write.csv(diag_tab, out_file("anchor_rules_by_pair.csv"), row.names = FALSE)

# chains into the reference frame
chain <- setNames(lapply(RULES, function(r) {
  out <- setNames(vector("list", nY), YEARS)
  for (i in seq_len(nY))
    out[[i]] <- if (i <= iref) compose_chain(seqR[[r]], i, iref)
                else t(compose_chain(seqR[[r]], iref, i))
  out
}), RULES)

saveRDS(list(seq = seqR, chain = chain, years = YEARS, ref = REF_YEAR,
             threshold = STAB_THRESHOLD, diagnostics = diag_tab),
        file.path(CACHE_DIR, "rotations.rds"))

# ---- figure -------------------------------------------------------------------
if (have_gg) {
  library(ggplot2)
  long <- rbind(
    data.frame(diag_tab[, c("year_to", "rule")], metric = "cos_E_stable", v = diag_tab$cos_E_stable),
    data.frame(diag_tab[, c("year_to", "rule")], metric = "cos_E_all",    v = diag_tab$cos_E_all),
    data.frame(diag_tab[, c("year_to", "rule")], metric = "null_E_all",   v = diag_tab$null_E_all))
  g <- ggplot(long, aes(year_to, v, colour = rule, linetype = metric)) +
    geom_line() + geom_point(size = .7) +
    labs(x = "year t+1", y = "mean held-out cosine", colour = NULL, linetype = NULL) +
    theme_minimal(base_size = 10) + theme(legend.position = "bottom")
  ggsave(out_file("fig_alignment_by_pair.pdf"), g, width = 8, height = 4.5)
}
msg("\nRotations -> %s", file.path(CACHE_DIR, "rotations.rds"))
