# =============================================================================
# ch6_04_components.R -- are the Chapter 5 components durable, and when do
# investors move between them?
#
# Every year is carried into the 2025-Q4 frame with the chained rotation of
# the default anchor rule (ch6_01_align.R). Two complementary readings follow.
#
# A. Fixed classifier. The Chapter 5 mixture (mu, rho, pi) is applied, as is,
#    to every aligned year: an investor's label in 2010 is its posterior-MAP
#    component under the 2025 model. Labels under the two robustness rules
#    are computed alongside, and their agreement is reported.
#
# B. Free refit. The mixture is re-estimated in every aligned year. If the
#    Chapter 5 components are durable, the refit recovers them: centroids
#    match the reference (Hungarian on cosine), and the partition agrees with
#    the fixed classifier (ARI). A membership-based matching of consecutive
#    refits, which uses no rotation, cross-checks the centroid matching.
#
# Timing: switching rates per step, a homogeneity test, and a logistic model
# with step effects and controls (origin posterior, turnover).
# Requires ch6_01_align.R; uses ch6_03_displacement.R output if present.
# =============================================================================
if (!exists("CH6_SETUP_LOADED")) source("ch6_00_setup.R")
suppressPackageStartupMessages(library(flexmix))   # circlus is called as circlus::

rot <- readRDS(file.path(CACHE_DIR, "rotations.rds"))
ROB <- setdiff(RULES, ANCHOR_RULE)
R_to_ref <- function(y, rule = ANCHOR_RULE) rot$chain[[rule]][[as.character(y)]]
nY <- length(YEARS)

# ---------------------------- mixture fitting --------------------------------
# Returns a standard list so the rest of the script never touches flexmix.
extract_spcauchy <- function(fit, d) {
  comps <- lapply(fit@components, function(cm) cm[[1]]@parameters)
  one <- function(p) {
    nm  <- names(p)
    len <- vapply(p, length, 1L)
    if (any(nm == "rho") && any(len == d)) {
      mu <- as.numeric(p[[which(len == d)[1]]]); rho <- as.numeric(p[["rho"]])
    } else if (sum(len == d) == 1 && !any(len == 1)) {     # psi = rho * mu
      psi <- as.numeric(p[[which(len == d)]]); rho <- sqrt(sum(psi^2)); mu <- psi / rho
    } else if (any(len == d) && any(len == 1)) {
      mu <- as.numeric(p[[which(len == d)[1]]]); rho <- as.numeric(p[[which(len == 1)[1]]])
    } else stop("cannot read component parameters; names: ", paste(nm, collapse = ", "))
    list(mu = mu / sqrt(sum(mu^2)), rho = rho)
  }
  cp <- lapply(comps, one)
  list(mu  = do.call(rbind, lapply(cp, `[[`, "mu")),
       rho = vapply(cp, `[[`, 0, "rho"),
       pi  = as.numeric(flexmix::prior(fit)))
}

fit_mixture <- function(X, k, seed = SEED) {
  best <- NULL; best_bic <- Inf
  for (s in seq_len(N_RESTART)) {
    set.seed(seed + s - 1)
    f <- try(flexmix::flexmix(X ~ 1, k = k, model = circlus::FLXMCspcauchy(),
                              control = list(minprior = MINPRIOR)), silent = TRUE)
    if (inherits(f, "try-error")) {
      msg("  restart %d failed: %s", s, conditionMessage(attr(f, "condition")))
      next
    }
    if (!is.finite(f@logLik)) next
    f_bic <- -2 * f@logLik + f@df * log(nrow(X))     # read from the fit's slots:
    if (is.null(best) || f_bic < best_bic) {         # no logLik()/BIC() dispatch
      best <- f; best_bic <- f_bic
    }
  }
  if (is.null(best)) stop("no restart converged")
  par <- extract_spcauchy(best, ncol(X))
  # the parameter reading is a guess about circlus internals -- verify it
  chk <- seq_len(min(2000, nrow(X)))
  pp  <- spcauchy_posterior(X[chk, , drop = FALSE], par)$post
  err <- max(abs(pp - flexmix::posterior(best)[chk, , drop = FALSE]))
  if (err > 1e-3) stop(sprintf("parameter extraction does not reproduce flexmix posteriors (max err %.3g)", err))
  pr <- spcauchy_posterior(X, par)
  list(par = par, cluster = max.col(pr$post, "first"), post = pr$post,
       loglik = mean(pr$loglik), k = length(par$rho))
}

relabel_par <- function(par, perm) {        # component j becomes label perm[j]
  o <- order(perm)
  list(mu = par$mu[o, , drop = FALSE], rho = par$rho[o], pi = par$pi[o])
}

# ---------------------------- reference mixture ------------------------------
Xr <- load_embeddings(REF_YEAR)
ids_r <- intersect(rownames(Xr), universe_ids(REF_YEAR))
Xr <- Xr[ids_r, , drop = FALSE]

ref_cache <- file.path(CACHE_DIR, sprintf("ref_mixture%s_k%d.rds", SUFFIX, K))
if (file.exists(ref_cache)) {
  ref <- readRDS(ref_cache)
} else {
  if (!is.null(REF_FIT_FILE) && !file.exists(REF_FIT_FILE))
    warning(REF_FIT_FILE, " not found: the reference mixture will be refitted")
  if (!is.null(REF_FIT_FILE) && file.exists(REF_FIT_FILE)) {
    obj <- readRDS(REF_FIT_FILE)
    # the file may hold the flexmix object itself or a list containing it
    find_fit <- function(o) {
      if (inherits(o, "flexmix")) return(o)
      if (is.list(o)) {
        nm <- intersect(c("fit", "best", "model", "mixture"), names(o))
        for (n in c(nm, setdiff(names(o), nm))) {
          if (inherits(o[[n]], "flexmix")) return(o[[n]])
        }
      }
      NULL
    }
    fit <- find_fit(obj)
    if (is.null(fit))
      stop("no flexmix object found in ", REF_FIT_FILE, " (class: ",
           paste(class(obj), collapse = "/"), "; elements: ",
           paste(names(obj), collapse = ", "), ")")
    msg("Loaded the Chapter 5 fit from %s (k = %d)", REF_FIT_FILE, fit@k)
    if (is.list(obj)) {                      # metadata written by Clusters.r
      if (!is.null(obj$emb_file) && normalizePath(obj$emb_file, mustWork = FALSE) !=
          normalizePath(emb_path(REF_YEAR), mustWork = FALSE))
        warning("the Chapter 5 fit was estimated on ", obj$emb_file,
                ", this run reads ", emb_path(REF_YEAR))
      if (!is.null(obj$minprior) && !isTRUE(all.equal(obj$minprior, MINPRIOR)))
        warning("Chapter 5 used minprior = ", obj$minprior, ", the setup has ", MINPRIOR)
      if (!is.null(obj$us_threshold) && is.na(obj$us_threshold) != (UNIVERSE == "global"))
        warning("the Chapter 5 fit and this run use different universes")
    }
    par <- extract_spcauchy(fit, ncol(Xr))

    # the fit's rows must be the reference investors in Clusters.r's order;
    # the Chapter 5 assignment file records that order
    n_fit <- nrow(flexmix::posterior(fit))
    if (file.exists(CH5_ASSIGN_FILE)) {
      ord <- as.character(readRDS(CH5_ASSIGN_FILE)$cluster$investor_id)
      if (length(ord) == n_fit && all(ord %in% rownames(Xr))) {
        Xr <- Xr[ord, , drop = FALSE]; ids_r <- ord
      }
    }
    if (n_fit != nrow(Xr))
      stop(REF_FIT_FILE, " was fitted on ", n_fit, " investors, the reference quarter has ",
           nrow(Xr), " in this universe")
    chk <- seq_len(min(2000, nrow(Xr)))
    err <- max(abs(spcauchy_posterior(Xr[chk, , drop = FALSE], par)$post -
                   flexmix::posterior(fit)[chk, , drop = FALSE]))
    if (err > 1e-3)
      stop(sprintf("%s: posteriors not reproduced (max err %.3g); the row order of the embeddings differs from Clusters.r",
                   REF_FIT_FILE, err))
    msg("Chapter 5 posteriors reproduced (max abs. difference %.2g)", err)
    pr  <- spcauchy_posterior(Xr, par)
    ref <- list(par = par, cluster = max.col(pr$post, "first"), post = pr$post,
                loglik = mean(pr$loglik), k = length(par$rho))
  } else {
    msg("Refitting the reference mixture at %d (k = %d) ...", REF_YEAR, K)
    ref <- fit_mixture(Xr, K)
  }
  ref$ari_ch5 <- NA_real_
  if (file.exists(CH5_ASSIGN_FILE)) {
    a5 <- readRDS(CH5_ASSIGN_FILE)$cluster
    lab5 <- a5$cluster[match(ids_r, as.character(a5$investor_id))]
    ok <- !is.na(lab5)
    tab <- unclass(table(factor(ref$cluster[ok], 1:ref$k), factor(lab5[ok], 1:K)))
    h <- hungarian(tab)
    ref$par <- relabel_par(ref$par, h$perm)
    ref$cluster <- h$perm[ref$cluster]
    ref$post <- ref$post[, order(h$perm), drop = FALSE]
    ref$ari_ch5 <- ari(ref$cluster[ok], lab5[ok])
    msg("Reference relabelled to Chapter 5 numbering; ARI with Chapter 5 = %.3f", ref$ari_ch5)
    if (ref$ari_ch5 < 0.9)
      warning("the reference mixture differs from the Chapter 5 partition; ",
              "set REF_FIT_FILE to the Chapter 5 fit so both chapters describe the same components")
  } else {
    warning("Chapter 5 assignment file not found (", CH5_ASSIGN_FILE,
            "): component numbers will not match the Chapter 5 tables")
  }
  ref$ids <- ids_r
  saveRDS(ref, ref_cache)
}
K_ref <- ref$k
write.csv(data.frame(component = seq_len(K_ref), rho = ref$par$rho, pi = ref$par$pi,
                     n = tabulate(ref$cluster, K_ref)),
          out_file("reference_components.csv"), row.names = FALSE)

# ---------------------------- A. fixed classifier ----------------------------
msg("\n== A. fixed classifier ==")
panel_rows <- list(); year_rows <- list(); aligned_cache <- list()
for (y in YEARS) {
  X <- load_embeddings(y)
  ids <- intersect(rownames(X), universe_ids(y))
  X <- X[ids, , drop = FALSE]
  Z   <- X %*% R_to_ref(y)
  pr  <- spcauchy_posterior(Z, ref$par)
  lab <- max.col(pr$post, "first")
  pr_rob <- lapply(ROB, function(r)
    max.col(spcauchy_posterior(X %*% R_to_ref(y, r), ref$par)$post, "first"))
  names(pr_rob) <- ROB
  pdf_ <- data.frame(investor_id = ids, year = y, label = lab,
                     max_post = apply(pr$post, 1, max), loglik = pr$loglik,
                     stringsAsFactors = FALSE)
  for (r in ROB) pdf_[[paste0("label_", r)]] <- pr_rob[[r]]
  panel_rows[[as.character(y)]] <- pdf_
  sh <- tabulate(lab, K_ref) / length(lab)
  yr_row <- data.frame(year = y, n = length(ids), loglik_ref = mean(pr$loglik))
  for (r in ROB) yr_row[[paste0("agree_", r)]] <- mean(lab == pr_rob[[r]])
  year_rows[[as.character(y)]] <- cbind(yr_row, t(setNames(sh, paste0("share_", seq_len(K_ref)))))
  aligned_cache[[as.character(y)]] <- Z
  msg("  %d  n=%6d  mean loglik %.2f  %s labels agree with %s",
      y, length(ids), mean(pr$loglik), ANCHOR_RULE,
      paste(sprintf("%s %.3f", ROB, sapply(ROB, function(r) mean(lab == pr_rob[[r]]))), collapse = ", "))
}
panel <- do.call(rbind, panel_rows)
yearly <- do.call(rbind, year_rows)
saveRDS(panel, out_file("fixed_classifier_panel.rds"))

# ---------------------------- B. free refit ----------------------------------
msg("\n== B. per-year refit ==")
refits <- list(); refit_rows <- list()
for (y in YEARS) {
  s <- as.character(y)
  cf <- file.path(CACHE_DIR, sprintf("refit_%d%s_k%d.rds", y, SUFFIX, K))
  if (file.exists(cf)) { rf <- readRDS(cf) } else {
    rf <- if (y == REF_YEAR) ref else fit_mixture(aligned_cache[[s]], K)
    rf$post <- NULL
    saveRDS(rf, cf)
  }
  # centroid matching to the reference (needs the rotation)
  hc <- hungarian(rf$par$mu %*% t(ref$par$mu))
  # membership matching to the fixed classifier of the same year
  fx <- panel_rows[[s]]$label
  hm <- hungarian(unclass(table(factor(rf$cluster, 1:rf$k), factor(fx, 1:K_ref))))
  rf$lab_ref <- hc$perm[rf$cluster]
  refits[[s]] <- rf
  matched_cos <- (rf$par$mu %*% t(ref$par$mu))[cbind(seq_len(rf$k), hc$perm)]
  refit_rows[[s]] <- data.frame(
    year = y, k_retained = rf$k,
    loglik_refit = rf$loglik, loglik_ref = yearly$loglik_ref[yearly$year == y],
    ari_refit_vs_fixed = ari(rf$cluster, fx),
    cen_margin = hc$margin, perm_agree_cen_vs_membership = mean(hc$perm == hm$perm),
    min_matched_cos = min(matched_cos), mean_matched_cos = mean(matched_cos),
    t(setNames(matched_cos[order(hc$perm)], paste0("cos_", hc$perm[order(hc$perm)]))),
    t(setNames(rf$par$rho[order(hc$perm)], paste0("rho_", hc$perm[order(hc$perm)]))))
  with(refit_rows[[s]], msg("  %d  k=%d  ARI(refit, fixed) %.3f  matched cos min %.3f mean %.3f  cen/memb agree %.2f",
                            y, k_retained, ari_refit_vs_fixed, min_matched_cos,
                            mean_matched_cos, perm_agree_cen_vs_membership))
}
refit_tab <- do.call(rbind, lapply(refit_rows, function(r) {
  miss <- setdiff(c(paste0("cos_", 1:K_ref), paste0("rho_", 1:K_ref)), names(r))
  r[miss] <- NA_real_; r }))
write.csv(refit_tab, out_file("refit_persistence.csv"), row.names = FALSE)

# alignment-free cross-check: consecutive refits matched on shared investors
af <- do.call(rbind, lapply(seq_len(nY - 1), function(t) {
  a <- refits[[as.character(YEARS[t])]]; b <- refits[[as.character(YEARS[t + 1])]]
  ia <- panel_rows[[as.character(YEARS[t])]]$investor_id
  ib <- panel_rows[[as.character(YEARS[t + 1])]]$investor_id
  com <- intersect(ia, ib)
  ca <- a$cluster[match(com, ia)]; cb <- b$cluster[match(com, ib)]
  if (b$k > a$k) return(NULL)                  # a component appeared; skip pair
  hm  <- hungarian(unclass(table(factor(cb, 1:b$k), factor(ca, 1:a$k))))  # b raw -> a raw
  hca <- hungarian(a$par$mu %*% t(ref$par$mu))$perm                       # a raw -> ref
  implied <- hca[hm$perm]                     # b raw -> ref, WITHOUT b's rotation
  via_cen <- hungarian(b$par$mu %*% t(ref$par$mu))$perm                   # b raw -> ref
  data.frame(year_from = YEARS[t], year_to = YEARS[t + 1], n_common = length(com),
             ari_consecutive_refits = ari(ca, cb),
             membership_vs_centroid_agree = mean(implied == via_cen))
}))
write.csv(af, out_file("refit_chain_check.csv"), row.names = FALSE)
msg("\nAlignment-free check (share of components whose rotation-based and\nmembership-based labels agree):")
print(format(af, digits = 3), row.names = FALSE)

# ---------------------------- transitions ------------------------------------
disp_file <- out_file("investor_displacement.rds")
disp <- if (file.exists(disp_file)) readRDS(disp_file) else NULL

make_steps <- function(label_col) do.call(rbind, lapply(seq_len(nY - 1), function(t) {
  a <- panel_rows[[as.character(YEARS[t])]]; b <- panel_rows[[as.character(YEARS[t + 1])]]
  com <- intersect(a$investor_id, b$investor_id)
  ia <- match(com, a$investor_id); ib <- match(com, b$investor_id)
  data.frame(investor_id = com, year_from = YEARS[t], year_to = YEARS[t + 1],
             from = a[[label_col]][ia], to = b[[label_col]][ib],
             post_from = a$max_post[ia], post_to = b$max_post[ib],
             stringsAsFactors = FALSE)
}))
steps <- make_steps("label")
steps$switch <- steps$from != steps$to

# refit-based labels as a robustness series
for (s in names(refits)) panel_rows[[s]]$label_refit <- refits[[s]]$lab_ref
steps_rf <- make_steps("label_refit"); steps_rf$switch <- steps_rf$from != steps_rf$to

if (!is.null(disp)) {
  k1 <- paste(steps$investor_id, steps$year_from)
  k2 <- paste(disp$investor_id, disp$year_from)
  m  <- match(k1, k2)
  steps$turnover <- disp$turnover[m]
  steps$delta <- disp$delta[m]
}

rate_table <- function(st, label) do.call(rbind, lapply(split(st, st$year_to), function(d) {
  pf <- tabulate(d$from, K_ref) / nrow(d); pt <- tabulate(d$to, K_ref) / nrow(d)
  ci <- wilson(sum(d$switch), nrow(d))
  conf <- d$post_from >= POST_CONF & d$post_to >= POST_CONF
  out <- data.frame(series = label, year_to = d$year_to[1], n = nrow(d),
                    switch_rate = mean(d$switch), lo = ci[["lo"]], hi = ci[["hi"]],
                    chance_rate = 1 - sum(pf * pt),
                    switch_rate_confident = mean(d$switch[conf]),
                    ari = ari(d$from, d$to))
  if (!is.null(d$turnover))
    out$switch_rate_stayers <- mean(d$switch[which(d$turnover <= 1 - STAB_THRESHOLD)])
  out
}))
rates <- rate_table(steps, sprintf("fixed classifier, %s", ANCHOR_RULE))
rates_rf <- rate_table(steps_rf, "refit, centroid-matched")
rates_rob <- lapply(ROB, function(r) {
  sr <- make_steps(paste0("label_", r)); sr$switch <- sr$from != sr$to
  rate_table(sr, sprintf("fixed classifier, %s", r))
})
cols <- names(rates_rf)
all_rates <- do.call(rbind, c(list(rates[, cols], rates_rf), lapply(rates_rob, `[`, cols)))
write.csv(all_rates, out_file("switch_rates.csv"), row.names = FALSE)
msg("\n== switching rates (fixed classifier) ==")
print(format(rates, digits = 3), row.names = FALSE)

# homogeneity: is the switching rate constant across steps?
ct <- table(steps$year_to, steps$switch)
chi <- suppressWarnings(chisq.test(ct))
msg("\nHomogeneity of switching across steps: chi2 = %.1f, df = %d, p = %.3g",
    chi$statistic, chi$parameter, chi$p.value)

# logistic model: step effects net of composition
ctrl <- "qlogis(pmin(pmax(post_from, 1e-4), 1 - 1e-4))"
if (!is.null(steps$turnover)) ctrl <- paste(ctrl, "+ turnover")
st_m <- steps[complete.cases(steps[, intersect(c("post_from", "turnover"), names(steps))]), ]
g1 <- glm(as.formula(paste("switch ~ factor(year_to) +", ctrl)), binomial, data = st_m)
g0 <- glm(as.formula(paste("switch ~", ctrl)), binomial, data = st_m)
lr <- anova(g0, g1, test = "LRT")
msg("LR test for step effects net of controls: chi2 = %.1f on %d df, p = %.3g",
    lr$Deviance[2], lr$Df[2], lr$`Pr(>Chi)`[2])
cf <- summary(g1)$coefficients
yr <- grep("factor\\(year_to\\)", rownames(cf))
step_fx <- data.frame(year_to = c(YEARS[2], as.integer(sub("factor\\(year_to\\)", "", rownames(cf)[yr]))),
                      log_odds = c(0, cf[yr, 1]), se = c(0, cf[yr, 2]))
step_fx$lo <- step_fx$log_odds - 1.96 * step_fx$se
step_fx$hi <- step_fx$log_odds + 1.96 * step_fx$se
write.csv(step_fx, out_file("switch_step_effects.csv"), row.names = FALSE)
write.csv(data.frame(term = rownames(cf)[-yr], cf[-yr, , drop = FALSE]),
          out_file("switch_controls.csv"), row.names = FALSE)

# transition matrices and per-component retention
trans <- do.call(rbind, lapply(split(steps, steps$year_to), function(d) {
  tb <- unclass(table(factor(d$from, 1:K_ref), factor(d$to, 1:K_ref)))
  data.frame(year_to = d$year_to[1], from = rep(1:K_ref, K_ref),
             to = rep(1:K_ref, each = K_ref), n = as.vector(tb),
             p = as.vector(tb / pmax(rowSums(tb), 1)))
}))
write.csv(trans, out_file("transitions.csv"), row.names = FALSE)
retention <- trans[trans$from == trans$to, c("year_to", "from", "n", "p")]
names(retention) <- c("year_to", "component", "stayed", "retention")
write.csv(retention, out_file("retention_by_component.csv"), row.names = FALSE)

# persistence over horizons
decay <- do.call(rbind, lapply(1:10, function(h) do.call(rbind, lapply(YEARS[YEARS + h <= max(YEARS)], function(y0) {
  a <- panel_rows[[as.character(y0)]]; b <- panel_rows[[as.character(y0 + h)]]
  com <- intersect(a$investor_id, b$investor_id)
  if (length(com) < 200) return(NULL)
  la <- a$label[match(com, a$investor_id)]; lb <- b$label[match(com, b$investor_id)]
  data.frame(h = h, year0 = y0, n = length(com), same = mean(la == lb),
             chance = sum(tabulate(la, K_ref) * tabulate(lb, K_ref)) / length(com)^2)
}))))
write.csv(decay, out_file("persistence_decay.csv"), row.names = FALSE)

# ---------------------------- noise floor ------------------------------------
if (!is.null(NOISE_EMB_FILE) && file.exists(NOISE_EMB_FILE)) {
  Xa <- load_embeddings(NOISE_YEAR); Xb <- read_vectors(NOISE_EMB_FILE)
  com0 <- intersect(rownames(Xa), rownames(Xb))
  # same quarter, same holdings: every investor is a valid anchor
  Rn <- procrustes(Xb[com0, ], Xa[com0, ]) %*% R_to_ref(NOISE_YEAR)
  a  <- panel_rows[[as.character(NOISE_YEAR)]]
  com <- intersect(a$investor_id, rownames(Xb))
  lb <- max.col(spcauchy_posterior(Xb[com, , drop = FALSE] %*% Rn, ref$par)$post, "first")
  la <- a$label[match(com, a$investor_id)]
  nf <- data.frame(year = NOISE_YEAR, n = length(com), switch_rate = mean(la != lb),
                   ari = ari(la, lb))
  write.csv(nf, out_file("switch_noise_floor.csv"), row.names = FALSE)
  msg("\nNoise floor (same holdings, different seed): switching %.3f, ARI %.3f",
      nf$switch_rate, nf$ari)
} else {
  msg("\n[skip] no seed pair configured: switching rates have no retraining noise floor.")
}

write.csv(yearly, out_file("fixed_classifier_by_year.csv"), row.names = FALSE)

# ---------------------------- figures ----------------------------------------
if (have_gg) {
  library(ggplot2)
  f1 <- ggplot(all_rates, aes(year_to, switch_rate, colour = series)) +
    geom_vline(xintercept = CRISIS_STEPS, colour = "grey85") +
    geom_line() + geom_point(size = .8) +
    geom_line(aes(y = chance_rate, colour = series), linetype = "dotted") +
    labs(x = "year t+1", y = "share of investors changing component",
         colour = NULL, caption = "dotted: rate under independent reassignment") +
    theme_minimal(base_size = 10) + theme(legend.position = "bottom")
  ggsave(out_file("fig_switch_rates.pdf"), f1, width = 8, height = 4)

  f2 <- ggplot(step_fx, aes(year_to, log_odds)) +
    geom_hline(yintercept = 0, colour = "grey70") +
    geom_vline(xintercept = CRISIS_STEPS, colour = "grey85") +
    geom_pointrange(aes(ymin = lo, ymax = hi), size = .25) +
    labs(x = "year t+1", y = "step effect on log-odds of switching") +
    theme_minimal(base_size = 10)
  ggsave(out_file("fig_switch_step_effects.pdf"), f2, width = 8, height = 4)

  sh <- do.call(rbind, lapply(seq_len(K_ref), function(j)
    data.frame(year = yearly$year, component = factor(j), share = yearly[[paste0("share_", j)]])))
  f3 <- ggplot(sh, aes(year, share, fill = component)) + geom_area() +
    labs(x = NULL, y = "share of investors (fixed classifier)") + theme_minimal(base_size = 10)
  ggsave(out_file("fig_component_shares.pdf"), f3, width = 8, height = 4.5)

  cs <- do.call(rbind, lapply(seq_len(K_ref), function(j)
    data.frame(year = refit_tab$year, component = factor(j), cos = refit_tab[[paste0("cos_", j)]])))
  f4 <- ggplot(cs, aes(year, cos, colour = component)) + geom_line() +
    labs(x = NULL, y = "cosine: refit centroid vs 2025 centroid") + theme_minimal(base_size = 10)
  ggsave(out_file("fig_centroid_persistence.pdf"), f4, width = 8, height = 4.5)

  f5 <- ggplot(retention, aes(year_to, retention)) + geom_line() +
    facet_wrap(~ component, nrow = 2) + geom_vline(xintercept = CRISIS_STEPS, colour = "grey85") +
    labs(x = "year t+1", y = "P(same component at t+1)") + theme_minimal(base_size = 9)
  ggsave(out_file("fig_retention.pdf"), f5, width = 9, height = 4.5)
}
msg("\nDone -> %s", OUT_DIR)