# =============================================================================
# ch6_03_displacement.R -- displacement and its link to holdings turnover
#
# 1. Displacement between consecutive aligned years (Hamilton et al. 2016)
#      delta_i(t) = 1 - cos( x_i(t) R_{t->t+1}, x_i(t+1) )
#    under the default anchor rule and the two robustness rules.
#
# 2. Turnover of the top-62 slice
#      turnover_i(t) = 1 - wt_overlap_i(t, t+1)
#
# 3. Validation: displacement should rise with turnover. Checked with
#      - Spearman correlation per year pair, with a 95% interval
#        (Fisher z, se = sqrt(1.06 / (n - 3)), Fieller et al. 1957)
#      - the same correlation without the stayers (wt_overlap >=
#        STAB_THRESHOLD), who are the stable_thr anchors and therefore
#        close to the rotation by construction
#      - pooled correlation and median displacement by turnover decile
#    all under each of the three rules.
#
# 4. Reference levels: stayers' displacement and, if configured, a
#    same-quarter different-seed pair.
# Requires ch6_01_align.R.
# =============================================================================
if (!exists("CH6_SETUP_LOADED")) source("ch6_00_setup.R")

rot <- readRDS(file.path(CACHE_DIR, "rotations.rds"))
nY  <- length(YEARS)
dcol <- function(r) paste0("delta_", r)

# ---------------------------- displacement per pair ---------------------------
rows <- list()
X1 <- load_embeddings(YEARS[1])
for (t in seq_len(nY - 1)) {
  y0 <- YEARS[t]; y1 <- YEARS[t + 1]
  X0 <- X1; X1 <- load_embeddings(y1)
  cache_t <- file.path(CACHE_DIR, sprintf("disp_step_%d%s.rds", y0, SUFFIX))
  if (file.exists(cache_t)) {
    d <- readRDS(cache_t)
  } else {
    H0 <- load_holdings(y0); H1 <- load_holdings(y1)
    ids <- Reduce(intersect, list(rownames(X0), rownames(X1), names(H1$n_full),
                                  universe_ids(y0, H0)))
    A0 <- X0[ids, , drop = FALSE]; A1 <- X1[ids, , drop = FALSE]
    d <- data.frame(investor_id = ids, year_from = y0, year_to = y1,
                    turnover = 1 - as.numeric(pair_stability(y0, y1, ids, H0, H1)),
                    stringsAsFactors = FALSE)
    for (r in RULES) d[[dcol(r)]] <- 1 - rowSums((A0 %*% rot$seq[[r]][[t]]) * A1)
    saveRDS(d, cache_t)
  }
  rows[[t]] <- d
}
inv <- do.call(rbind, rows)
inv$stayer <- inv$turnover <= 1 - STAB_THRESHOLD
inv$delta  <- inv[[dcol(ANCHOR_RULE)]]
saveRDS(inv, out_file("investor_displacement.rds"))

# ---------------------------- correlation helper ------------------------------
spearman <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  n <- sum(ok)
  if (n < 10) return(c(rho = NA, lo = NA, hi = NA, p = NA, n = n))
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
  rho <- unname(ct$estimate)
  se <- sqrt(1.06 / (n - 3))
  z  <- atanh(rho)
  c(rho = rho, lo = tanh(z - 1.96 * se), hi = tanh(z + 1.96 * se), p = ct$p.value, n = n)
}

# ---------------------------- per year ----------------------------------------
by_year <- do.call(rbind, lapply(split(inv, inv$year_to), function(d) {
  out <- data.frame(
    year_to = d$year_to[1], n = nrow(d),
    median_turnover = median(d$turnover),
    median_delta = median(d$delta),
    q25 = quantile(d$delta, .25, names = FALSE),
    q75 = quantile(d$delta, .75, names = FALSE),
    stayers_n = sum(d$stayer),
    stayers_median_delta = if (any(d$stayer)) median(d$delta[d$stayer]) else NA_real_,
    movers_median_delta = median(d$delta[!d$stayer]))
  for (r in RULES) {
    a <- spearman(d$turnover, d[[dcol(r)]])
    b <- spearman(d$turnover[!d$stayer], d[[dcol(r)]][!d$stayer])
    out[[paste0("rho_", r)]]            <- a[["rho"]]
    out[[paste0("rho_", r, "_lo")]]     <- a[["lo"]]
    out[[paste0("rho_", r, "_hi")]]     <- a[["hi"]]
    out[[paste0("rho_", r, "_movers")]] <- b[["rho"]]
    if (r != ANCHOR_RULE) out[[paste0("median_delta_", r)]] <- median(d[[dcol(r)]])
  }
  out
}))
write.csv(by_year, out_file("displacement_by_year.csv"), row.names = FALSE)

msg("\n== displacement and turnover by year (%s) ==", ANCHOR_RULE)
print(format(by_year[, c("year_to", "n", "median_turnover", "median_delta",
                         "stayers_n", "stayers_median_delta", "movers_median_delta")],
             digits = 3), row.names = FALSE)
msg("\n== Spearman(turnover, displacement) by year: all investors | movers only ==")
print(format(by_year[, c("year_to", paste0("rho_", RULES), paste0("rho_", RULES, "_movers"))],
             digits = 3), row.names = FALSE)

# ---------------------------- pooled -------------------------------------------
pooled <- do.call(rbind, lapply(RULES, function(r) {
  a  <- spearman(inv$turnover, inv[[dcol(r)]])
  b  <- spearman(inv$turnover[!inv$stayer], inv[[dcol(r)]][!inv$stayer])
  yr <- by_year[[paste0("rho_", r)]]
  data.frame(rule = r, n = a[["n"]],
             rho_pooled = a[["rho"]], lo = a[["lo"]], hi = a[["hi"]], p = a[["p"]],
             rho_pooled_movers = b[["rho"]],
             years_positive = sum(yr > 0, na.rm = TRUE),
             years_ci_above_0 = sum(by_year[[paste0("rho_", r, "_lo")]] > 0, na.rm = TRUE),
             min_year_rho = min(yr, na.rm = TRUE), max_year_rho = max(yr, na.rm = TRUE))
}))
write.csv(pooled, out_file("turnover_correlation_pooled.csv"), row.names = FALSE)
msg("\n== pooled Spearman(turnover, displacement), %d year pairs ==", nrow(by_year))
print(format(pooled, digits = 3), row.names = FALSE)

# ---------------------------- binned relationship -------------------------------
brks <- unique(quantile(inv$turnover, seq(0, 1, length.out = N_TURNOVER_BINS + 1)))
inv$bin <- cut(inv$turnover, brks, include.lowest = TRUE, labels = FALSE)
binned <- do.call(rbind, lapply(split(inv, inv$bin), function(d) {
  out <- data.frame(bin = d$bin[1], n = nrow(d),
                    turnover_lo = min(d$turnover), turnover_hi = max(d$turnover),
                    turnover_median = median(d$turnover))
  for (r in RULES) out[[paste0("median_delta_", r)]] <- median(d[[dcol(r)]])
  out$q25_delta <- quantile(d$delta, .25, names = FALSE)
  out$q75_delta <- quantile(d$delta, .75, names = FALSE)
  out
}))
write.csv(binned, out_file("displacement_by_turnover_decile.csv"), row.names = FALSE)
msg("\n== median displacement by turnover decile ==")
print(format(binned[, c("bin", "n", "turnover_lo", "turnover_hi", paste0("median_delta_", RULES))],
             digits = 3), row.names = FALSE)
for (r in RULES) {
  v <- binned[[paste0("median_delta_", r)]]
  msg("  %-12s rises in %d of %d consecutive deciles", r, sum(diff(v) > 0), length(v) - 1)
}

# ---------------------------- noise floor ---------------------------------------
noise <- NULL
if (!is.null(NOISE_EMB_FILE) && file.exists(NOISE_EMB_FILE)) {
  Xa <- load_embeddings(NOISE_YEAR); Xb <- read_vectors(NOISE_EMB_FILE)
  ids <- intersect(rownames(Xa), rownames(Xb))
  # same quarter, same holdings: every investor is a valid anchor
  Rn <- procrustes(Xb[ids, ], Xa[ids, ])
  dn <- 1 - rowSums((Xb[ids, ] %*% Rn) * Xa[ids, ])
  noise <- data.frame(year = NOISE_YEAR, n = length(ids), median = median(dn),
                      q25 = quantile(dn, .25, names = FALSE),
                      q75 = quantile(dn, .75, names = FALSE),
                      p90 = quantile(dn, .90, names = FALSE))
  write.csv(noise, out_file("displacement_noise_floor.csv"), row.names = FALSE)
  msg("\n== noise floor (same holdings, different seed) ==")
  print(format(noise, digits = 3), row.names = FALSE)
} else {
  msg("\n[skip] no seed pair configured: displacement has no retraining noise floor.")
}

# ---------------------------- figures -------------------------------------------
if (have_gg) {
  library(ggplot2)

  g1 <- ggplot(by_year, aes(year_to)) +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = .15) +
    geom_line(aes(y = median_delta, linetype = "all investors")) +
    geom_line(aes(y = stayers_median_delta, linetype = "stayers")) +
    { if (!is.null(noise)) geom_hline(yintercept = noise$median, colour = "grey40") } +
    geom_vline(xintercept = CRISIS_STEPS, colour = "grey85") +
    labs(x = "year t+1", y = "displacement 1 - cos", linetype = NULL) +
    theme_minimal(base_size = 10) + theme(legend.position = "bottom")
  ggsave(out_file("fig_displacement_by_year.pdf"), g1, width = 8, height = 4)

  bl <- do.call(rbind, lapply(RULES, function(r)
    data.frame(turnover = binned$turnover_median, rule = r,
               delta = binned[[paste0("median_delta_", r)]])))
  g2 <- ggplot(bl, aes(turnover, delta, colour = rule)) +
    geom_ribbon(data = binned, inherit.aes = FALSE,
                aes(x = turnover_median, ymin = q25_delta, ymax = q75_delta), alpha = .12) +
    geom_line() + geom_point(size = 1.2) +
    labs(x = "turnover of the top-62 slice (decile median)",
         y = "median displacement", colour = NULL,
         caption = sprintf("band: interquartile range under %s", ANCHOR_RULE)) +
    theme_minimal(base_size = 10) + theme(legend.position = "bottom")
  ggsave(out_file("fig_displacement_vs_turnover.pdf"), g2, width = 7, height = 4.5)

  yl <- do.call(rbind, lapply(RULES, function(r)
    data.frame(year_to = by_year$year_to, rule = r,
               rho = by_year[[paste0("rho_", r)]],
               lo = by_year[[paste0("rho_", r, "_lo")]],
               hi = by_year[[paste0("rho_", r, "_hi")]])))
  g3 <- ggplot(yl, aes(year_to, rho, colour = rule)) +
    geom_hline(yintercept = 0, colour = "grey70") +
    geom_pointrange(aes(ymin = lo, ymax = hi), size = .2,
                    position = position_dodge(width = .5)) +
    labs(x = "year t+1", y = "Spearman(turnover, displacement)", colour = NULL) +
    theme_minimal(base_size = 10) + theme(legend.position = "bottom")
  ggsave(out_file("fig_turnover_correlation_by_year.pdf"), g3, width = 8, height = 4)
}
msg("\nDone -> %s", OUT_DIR)
