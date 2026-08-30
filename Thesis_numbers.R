# =============================================================================
# Thesis_numbers.R
#
# Computes the quantities still marked \todo{} in the thesis, excluding those
# belonging to Chapter 6 (cross-quarter dynamics). Each section writes a CSV
# to results/ and prints a LaTeX-ready table body, so figures can be pasted
# into the document without retyping.
#
#   A  Descriptive statistics of the panel            (Ch.3, todo 4)
#   B  Filtering attrition for one quarter            (Ch.3, todo 3)
#   C  Training diagnostics                           (Ch.4, todo 5)
#   D  Delta statistic + permutation benchmark        (Ch.5.1, todo 7)
#   E  Extended K grid                                (Ch.5.2, todo 8)
#   F  Rank-based association for the flat statistics (Ch.5.4, todo 9)
#   G  Multi-seed stability, within and between arms  (Ch.5.6, todo 10)
#   H  Robustness: minprior, k-means agreement, indices (Ch.5.8, todo 14)
#   I  US-only variants: baseline arm, 95% threshold  (Ch.5.7, todo 12)
#   J  The unnamed issuer in the index component      (Ch.5.7, todo 13)
#   K  Session information                            (App.A, todo 16)
#   L  Export full descriptive tables                 (App.B, todo 17)
#
# Sections E, G, H and I refit mixtures and are slow; each is behind its own
# flag. Everything else runs in seconds to a few minutes.
# =============================================================================

library(arrow)
library(dplyr)
library(tidyr)
library(purrr)
library(flexmix)
library(circlus)

# ------------------------------ CONFIG ---------------------------------------
QUARTER   <- "2026-03-31"
SEED      <- 42
K_GRID    <- 2:12
MINPRIOR  <- 0.01
MIN_COMP  <- 150
RES_DIR   <- "results"

RUN_A_DESCRIPTIVES <- TRUE
RUN_B_ATTRITION    <- FALSE    # needs WRDS; set FALSE to skip
RUN_C_TRAINING     <- TRUE
RUN_D_DELTA        <- TRUE
RUN_E_KGRID        <- FALSE   # slow: fits K = 13..20 for both arms
RUN_F_RANKASSOC    <- TRUE
RUN_G_SEEDS        <- FALSE   # slow: 3 seeds x 2 arms at the final K
RUN_H_ROBUST       <- FALSE   # slow: minprior sweep + k-means + indices
RUN_I_USVARIANTS   <- FALSE   # slow: 2 further US-restricted fits
RUN_J_UNNAMED      <- TRUE
RUN_K_SESSION      <- TRUE
RUN_L_EXPORT       <- TRUE

dir.create(RES_DIR, showWarnings = FALSE)

emb_path <- function(arm, q = QUARTER)
  switch(arm,
         baseline = sprintf("embeddings_v2/q_%s.parquet", q),
         weighted = sprintf("embeddings_weighted/q_%s__weighted_first-chunk.parquet", q))

# print a LaTeX tabular body from a data frame, so tables can be pasted in
latex_rows <- function(df, digits = 3) {
  # format COLUMN-wise: apply() coerces the whole frame to a character
  # matrix, which silently discards the numeric formatting
  df <- as.data.frame(df)
  cols <- lapply(df, function(x) {
    if (is.numeric(x)) {
      d <- if (all(x == round(x), na.rm = TRUE)) 0 else digits
      formatC(x, format = "f", digits = d, big.mark = "{,}")
    } else as.character(x)
  })
  body <- do.call(paste, c(cols, sep = " & "))
  cat("\n% --- LaTeX body ---\n")
  cat(paste0(body, " \\\\"), sep = "\n")
  cat("% ------------------\n")
}

save_res <- function(df, name) {
  write.csv(df, file.path(RES_DIR, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("  -> %s/%s.csv\n", RES_DIR, name))
}

# =========================================================================
# A. Panel descriptive statistics                          (Ch.3, todo 4)
# =========================================================================
if (RUN_A_DESCRIPTIVES) {
  cat("\n\n########## A. PANEL DESCRIPTIVE STATISTICS ##########\n")
  files <- sort(list.files("data", pattern = "^q_\\d{4}-\\d{2}-\\d{2}\\.parquet$",
                           full.names = TRUE))
  cat(sprintf("%d quarterly files\n", length(files)))

  panel <- map_dfr(seq_along(files), function(i) {
    cat(sprintf("\r  reading %d/%d", i, length(files))); flush.console()
    d <- read_parquet(files[i])
    q <- as.Date(sub(".*q_(\\d{4}-\\d{2}-\\d{2}).*", "\\1", files[i]))
    inv <- d |> distinct(investor_id, investor_type, n_assets_full)
    tibble(quarter = q,
           n_investors = nrow(inv),
           n_issuers   = n_distinct(unlist(d$tokens)),
           n_positions = sum(inv$n_assets_full),
           med_assets  = median(inv$n_assets_full),
           q25_assets  = quantile(inv$n_assets_full, .25),
           q75_assets  = quantile(inv$n_assets_full, .75),
           pct_over_62 = 100 * mean(inv$n_assets_full > 62)) |>
      bind_cols(inv |> count(investor_type) |>
                  pivot_wider(names_from = investor_type, values_from = n,
                              values_fill = 0))
  })
  cat("\n")
  panel <- panel |> arrange(quarter)
  save_res(panel, "A_panel_by_quarter")

  # seasonality, quantified: Q2/Q4 against Q1/Q3
  seas <- panel |>
    mutate(qtr = quarters(quarter)) |>
    group_by(qtr) |>
    summarise(n = n(), mean_investors = round(mean(n_investors)), .groups = "drop")
  cat("\n=== Investors by calendar quarter (the seasonal pattern) ===\n")
  print(seas)
  h1 <- mean(seas$mean_investors[seas$qtr %in% c("Q2","Q4")])
  h2 <- mean(seas$mean_investors[seas$qtr %in% c("Q1","Q3")])
  cat(sprintf("Q2/Q4 exceed Q1/Q3 by %.1f%%\n", 100 * (h1/h2 - 1)))

  cat("\n=== Panel summary (for the Chapter 3 table) ===\n")
  summ <- panel |>
    summarise(first = min(quarter), last = max(quarter), quarters = n(),
              inv_min = min(n_investors), inv_med = median(n_investors),
              inv_max = max(n_investors),
              iss_med = median(n_issuers),
              med_assets = median(med_assets),
              pct_over_62 = round(mean(pct_over_62), 1))
  print(as.data.frame(summ))
  save_res(summ, "A_panel_summary")

  cat("\n=== Reference-quarter portfolio size distribution ===\n")
  ref <- read_parquet(sprintf("data/q_%s.parquet", QUARTER)) |>
    distinct(investor_id, n_assets_full)
  print(summary(ref$n_assets_full))
  cat(sprintf("share above the 62-token window: %.1f%%\n",
              100 * mean(ref$n_assets_full > 62)))
  latex_rows(panel |> filter(quarter >= as.Date("2024-01-01")) |>
               select(quarter, n_investors, n_issuers, med_assets, pct_over_62),
             digits = 1)
}

# =========================================================================
# B. Filtering attrition                                   (Ch.3, todo 3)
# =========================================================================
# The pre-filter counts exist only upstream, so this reruns the cascade for
# ONE quarter against WRDS and records the row count after each step.
if (RUN_B_ATTRITION) {
  cat("\n\n########## B. FILTERING ATTRITION (one quarter) ##########\n")
  ok <- requireNamespace("RPostgres", quietly = TRUE) &&
        nzchar(Sys.getenv("WRDS_USER"))
  if (!ok) {
    cat("  WRDS credentials not available; skipping.\n",
        "  (Set WRDS_USER/WRDS_PASSWORD in ~/.Renviron and restart R.)\n")
  } else {
    library(RPostgres); library(dbplyr)
    wrds <- dbConnect(Postgres(), host = "wrds-pgdata.wharton.upenn.edu",
                      dbname = "wrds", port = 9737, sslmode = "require",
                      user = Sys.getenv("WRDS_USER"),
                      password = Sys.getenv("WRDS_PASSWORD"),
                      connect_timeout = 30)
    qe <- as.Date(QUARTER)
    # dbplyr translates `qe - 7` to `date - 7.0`, which Postgres rejects
    # (no date - numeric operator). Precomputing keeps them Date literals.
    lo13 <- qe - 7; hi13 <- qe + 7
    lofd <- qe - 5; hifd <- qe + 5
    sec_map <- tbl(wrds, in_schema("factset_own", "own_sec_entity_eq")) |>
      filter(!is.na(factset_entity_id)) |>
      select(fsym_id, issuer_id = factset_entity_id)
    fund_map <- tbl(wrds, in_schema("factset_own", "own_ent_funds")) |>
      filter(!is.na(fund_type)) |> select(factset_fund_id, fund_type)

    h13 <- tbl(wrds, in_schema("factset_own", "wrds_own_13f")) |>
      filter(entity_sub_type == "HF", report_date >= lo13,
             report_date <= hi13, adj_mv > 0) |>
      inner_join(sec_map, by = "fsym_id") |>
      select(investor_id = factset_entity_id, issuer_id, adj_mv) |> collect()
    hfd <- tbl(wrds, in_schema("factset_own", "wrds_own_fund")) |>
      inner_join(sec_map, by = "fsym_id") |>
      inner_join(fund_map, by = "factset_fund_id") |>
      filter(fund_type %in% c("OEF","ETF","CEF","VAR"),
             report_date >= lofd, report_date <= hifd, adj_mv > 0) |>
      select(investor_id = factset_fund_id, issuer_id, adj_mv,
             investor_type = fund_type, report_date) |> collect() |>
      arrange(investor_id, issuer_id, desc(report_date)) |>
      group_by(investor_id, issuer_id) |> slice_head(n = 1) |> ungroup() |>
      select(-report_date)
    dbDisconnect(wrds)

    h <- bind_rows(h13 |> mutate(investor_type = "HF"), hfd) |>
      group_by(investor_id, investor_type, issuer_id) |>
      summarise(adj_mv = sum(as.numeric(adj_mv)), .groups = "drop")

    steps <- tibble(step = "raw holdings (both blocks)",
                    positions = nrow(h),
                    investors = n_distinct(h$investor_id),
                    issuers   = n_distinct(h$issuer_id))

    h <- h |> group_by(investor_id) |>
      filter(max(adj_mv) / sum(adj_mv) <= 0.75) |> ungroup()
    steps <- add_row(steps, step = "after 75% concentration cap",
                     positions = nrow(h), investors = n_distinct(h$investor_id),
                     issuers = n_distinct(h$issuer_id))

    it <- 0
    repeat {
      n0 <- nrow(h); it <- it + 1
      h <- h |> group_by(investor_id) |> filter(n() >= 20) |> ungroup() |>
               group_by(issuer_id)   |> filter(n() >= 20) |> ungroup()
      steps <- add_row(steps, step = sprintf("bipartite pruning, pass %d", it),
                       positions = nrow(h), investors = n_distinct(h$investor_id),
                       issuers = n_distinct(h$issuer_id))
      if (nrow(h) == n0) break
    }
    steps <- steps |> mutate(pct_of_raw = round(100 * positions / positions[1], 1))
    cat(sprintf("\n=== Attrition cascade, %s ===\n", QUARTER))
    print(as.data.frame(steps))
    save_res(steps, "B_attrition")
    latex_rows(steps, digits = 1)
  }
}

# =========================================================================
# C. Training diagnostics                                  (Ch.4, todo 5)
# =========================================================================
if (RUN_C_TRAINING) {
  cat("\n\n########## C. TRAINING DIAGNOSTICS ##########\n")
  hist_files <- c(list.files("models_v2", pattern = "history.json$",
                             recursive = TRUE, full.names = TRUE),
                  list.files("models_weighted", pattern = "history.json$",
                             recursive = TRUE, full.names = TRUE))
  if (!length(hist_files)) {
    cat("  no history.json found under models_v2/ or models_weighted/\n")
  } else {
    tr <- map_dfr(hist_files, function(f) {
      j <- jsonlite::fromJSON(f, simplifyVector = FALSE)
      arm <- if (grepl("models_weighted", f)) "weighted" else "baseline"
      q   <- sub(".*q_(\\d{4}-\\d{2}-\\d{2}).*", "\\1", f)
      lg  <- j$pretrain_log
      tl  <- map_dbl(lg, ~ if (!is.null(.x$loss)) .x$loss else NA_real_)
      vl  <- map_dbl(lg, ~ if (!is.null(.x$eval_loss)) .x$eval_loss else NA_real_)
      ft  <- unlist(j$finetune$loss)
      vocab <- if (!is.null(j$vocab_size)) j$vocab_size else NA
      tibble(arm, quarter = q,
             vocab_size   = vocab,
             n_investors  = if (!is.null(j$n_investors)) j$n_investors else NA,
             mlm_first    = round(first(na.omit(tl)), 3),
             mlm_last     = round(last(na.omit(tl)), 3),
             mlm_val_last = round(last(na.omit(vl)), 3),
             # a uniform guess over the vocabulary is the reference point
             mlm_random   = if (!is.na(vocab)) round(log(vocab), 3) else NA_real_,
             ft_first     = if (length(ft)) round(ft[1], 4) else NA_real_,
             ft_last      = if (length(ft)) round(tail(ft, 1), 4) else NA_real_)
    }) |> arrange(arm, quarter)
    cat("\n=== Per-quarter training diagnostics ===\n")
    print(as.data.frame(tr))
    save_res(tr, "C_training")
    latex_rows(tr |> select(arm, quarter, vocab_size, mlm_random,
                            mlm_first, mlm_last, ft_first, ft_last), digits = 3)
    cat("\nNote: mlm_random = log(vocab) is the cross-entropy of a uniform\n",
        "guess and is the reference against which the fitted loss is read.\n")
  }
}

# =========================================================================
# D. Delta statistic and permutation benchmark             (Ch.5.1, todo 7)
# =========================================================================
# Exact O(n d) decomposition: for unit-norm rows, the sum of ordered pairwise
# inner products within a group g is ||sum_g x||^2 - n_g. Verified against a
# brute-force O(n^2 d) computation.
delta_stat <- function(X, g) {
  ok <- !is.na(g); X <- X[ok, , drop = FALSE]; g <- g[ok]
  n <- nrow(X); S <- colSums(X)
  tot_sum <- sum(S * S) - n;  tot_cnt <- n * (n - 1)
  w_sum <- 0; w_cnt <- 0
  for (lev in unique(g)) {
    Xg <- X[g == lev, , drop = FALSE]; ng <- nrow(Xg)
    if (ng < 2) next
    Sg <- colSums(Xg)
    w_sum <- w_sum + sum(Sg * Sg) - ng
    w_cnt <- w_cnt + ng * (ng - 1)
  }
  b_sum <- tot_sum - w_sum; b_cnt <- tot_cnt - w_cnt
  c(within = w_sum / w_cnt, between = b_sum / b_cnt,
    delta = w_sum / w_cnt - b_sum / b_cnt)
}

if (RUN_D_DELTA) {
  cat("\n\n########## D. DELTA STATISTIC ##########\n")
  res <- list()
  for (arm in c("baseline", "weighted")) {
    dir_ <- if (arm == "baseline") "embeddings_v2" else "embeddings_weighted"
    fs <- list.files(dir_, pattern = "\\.parquet$", full.names = TRUE)
    for (f in fs) {
      q <- sub(".*q_(\\d{4}-\\d{2}-\\d{2}).*", "\\1", basename(f))
      d <- read_parquet(f)
      X <- as.matrix(d[, grepl("^dim_", names(d))])
      X <- X / sqrt(rowSums(X^2))
      st <- delta_stat(X, d$investor_type)

      # permutation benchmark: shuffle types, recompute, 200 times
      set.seed(SEED)
      perm <- replicate(200, delta_stat(X, sample(d$investor_type))["delta"])
      res[[length(res) + 1]] <- tibble(
        arm, quarter = q, n = nrow(X),
        within = round(st["within"], 4), between = round(st["between"], 4),
        delta  = round(st["delta"], 4),
        perm_mean = round(mean(perm), 4), perm_sd = round(sd(perm), 4),
        z = round((st["delta"] - mean(perm)) / sd(perm), 1),
        p_perm = round(mean(perm >= st["delta"]), 4))
      cat(sprintf("  %-8s %s  delta = %+.4f  (permutation mean %+.4f, z = %.1f)\n",
                  arm, q, st["delta"], mean(perm),
                  (st["delta"] - mean(perm)) / sd(perm)))
    }
  }
  delta_tab <- bind_rows(res) |> arrange(arm, quarter)
  save_res(delta_tab, "D_delta")
  latex_rows(delta_tab |> select(arm, quarter, n, within, between, delta, z),
             digits = 4)
}

# =========================================================================
# E. Extended K grid                                       (Ch.5.2, todo 8)
# =========================================================================
fit_mixture <- function(X, k, seed = SEED, minprior = MINPRIOR) {
  set.seed(seed)
  eff <- max(minprior, MIN_COMP / nrow(X))
  f <- try(flexmix(X ~ 1, k = k, model = FLXMCspcauchy(),
                   control = list(minprior = eff)), silent = TRUE)
  if (inherits(f, "try-error") || !is.finite(as.numeric(logLik(f)))) return(NULL)
  f
}
load_X <- function(arm, q = QUARTER) {
  d <- read_parquet(emb_path(arm, q))
  X <- as.matrix(d[, grepl("^dim_", names(d))])
  list(X = X / sqrt(rowSums(X^2)), meta = d |> select(investor_id, investor_type))
}

if (RUN_E_KGRID) {
  cat("\n\n########## E. EXTENDED K GRID ##########\n")
  out <- list()
  for (arm in c("baseline", "weighted")) {
    dat <- load_X(arm)
    for (k in 13:20) {
      t0 <- Sys.time()
      f <- fit_mixture(dat$X, k)
      if (is.null(f)) { cat(sprintf("  %s k=%2d FAILED\n", arm, k)); next }
      out[[length(out)+1]] <- tibble(
        arm, k_asked = k, k_kept = length(unique(clusters(f))),
        logLik = as.numeric(logLik(f)), BIC = BIC(f),
        mins = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1))
      cat(sprintf("  %s k=%2d kept %2d  BIC = %.0f  (%.1f min)\n", arm, k,
                  tail(out,1)[[1]]$k_kept, tail(out,1)[[1]]$BIC,
                  tail(out,1)[[1]]$mins))
    }
  }
  kgrid <- bind_rows(out)
  save_res(kgrid, "E_kgrid_extended")
  cat("\nDoes BIC turn? (a minimum strictly inside the extended grid)\n")
  print(kgrid |> group_by(arm) |>
          summarise(best_k = k_asked[which.min(BIC)],
                    interior = best_k < max(k_asked), .groups = "drop"))
}

# =========================================================================
# F. Rank-based association for the flat statistics         (Ch.5.4, todo 9)
# =========================================================================
# eta^2 is not robust: extreme values inflate the denominator and drive the
# statistic toward zero. Two alternatives are computed for every continuous
# attribute: Kruskal-Wallis epsilon^2 (rank based) and eta^2 on winsorised
# inputs.
eta_sq <- function(x, g) {
  ok <- !is.na(x) & !is.na(g); if (sum(ok) < 50) return(NA_real_)
  x <- x[ok]; g <- factor(g[ok])
  sst <- sum((x - mean(x))^2); if (sst == 0) return(NA_real_)
  sum(tapply(x, g, function(v) length(v) * (mean(v) - mean(x))^2)) / sst
}
eps_sq <- function(x, g) {                   # Kruskal-Wallis epsilon squared
  ok <- !is.na(x) & !is.na(g); if (sum(ok) < 50) return(NA_real_)
  x <- x[ok]; g <- factor(g[ok]); n <- length(x); k <- nlevels(g)
  if (k < 2) return(NA_real_)
  H <- suppressWarnings(kruskal.test(x, g)$statistic)
  max(0, (as.numeric(H) - k + 1) / (n - k))
}
winsor <- function(x, p = 0.01) {
  q <- quantile(x, c(p, 1 - p), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}

if (RUN_F_RANKASSOC) {
  cat("\n\n########## F. RANK-BASED ASSOCIATION ##########\n")
  inv_ref <- read_parquet("reference/investors_master.parquet")
  num_vars <- c("beta", "pe_ratio", "pb_ratio", "dividend_yield",
                "sales_growth", "price_momentum", "relative_strength")
  out <- list()
  for (arm in c("baseline", "weighted")) {
    fa <- sprintf("cluster_assignment_%s_%s.rds", arm, QUARTER)
    if (!file.exists(fa)) { cat(sprintf("  %s: %s missing; skipped\n", arm, fa)); next }
    cl <- readRDS(fa)$cluster |> left_join(inv_ref, by = "investor_id")
    for (v in intersect(num_vars, names(cl))) {
      x <- cl[[v]]; g <- cl$cluster
      out[[length(out)+1]] <- tibble(
        arm, attribute = v,
        eta2_raw  = round(eta_sq(x, g), 3),
        eta2_wins = round(eta_sq(winsor(x), g), 3),
        eps2_rank = round(eps_sq(x, g), 3),
        n_obs = sum(!is.na(x)),
        # how extreme are the tails that break eta^2?
        max_over_p99 = round(max(x, na.rm = TRUE) /
                             quantile(x, .99, na.rm = TRUE), 1))
    }
  }
  ra <- bind_rows(out)
  cat("\n=== eta^2 (raw) vs winsorised vs rank-based epsilon^2 ===\n")
  print(as.data.frame(ra))
  save_res(ra, "F_rank_association")
  cat("\nAttributes whose raw eta^2 understates the rank-based measure by >0.05:\n")
  print(ra |> filter(eps2_rank - eta2_raw > 0.05) |>
          select(arm, attribute, eta2_raw, eta2_wins, eps2_rank))
  latex_rows(ra, digits = 3)
}

# =========================================================================
# G. Multi-seed stability, within and between arms         (Ch.5.6, todo 10)
# =========================================================================
ari <- function(a, b) {
  tab <- table(a, b); sc <- function(x) sum(choose(x, 2))
  si <- sc(tab); sr <- sc(rowSums(tab)); scl <- sc(colSums(tab))
  ex <- sr * scl / choose(length(a), 2)
  (si - ex) / ((sr + scl) / 2 - ex)
}

if (RUN_G_SEEDS) {
  cat("\n\n########## G. MULTI-SEED STABILITY ##########\n")
  K_FIN <- 12; seeds <- c(42, 123, 2024)
  labs <- list()
  for (arm in c("baseline", "weighted")) {
    dat <- load_X(arm)
    for (sd in seeds) {
      cat(sprintf("  fitting %s, seed %d ...\n", arm, sd)); flush.console()
      f <- fit_mixture(dat$X, K_FIN, seed = sd)
      if (is.null(f)) { cat("    failed\n"); next }
      labs[[paste(arm, sd, sep = "_")]] <-
        tibble(investor_id = dat$meta$investor_id, cluster = clusters(f))
    }
  }
  keys <- names(labs)
  pairs <- t(combn(keys, 2))
  cmp <- map_dfr(seq_len(nrow(pairs)), function(i) {
    a <- labs[[pairs[i,1]]]; b <- labs[[pairs[i,2]]]
    j <- inner_join(a, b, by = "investor_id", suffix = c("_a","_b"))
    tibble(pair_a = pairs[i,1], pair_b = pairs[i,2],
           same_arm = sub("_.*","",pairs[i,1]) == sub("_.*","",pairs[i,2]),
           n = nrow(j), ari = round(ari(j$cluster_a, j$cluster_b), 3))
  })
  cat("\n=== Pairwise ARI ===\n"); print(as.data.frame(cmp))
  save_res(cmp, "G_seed_stability")
  cat("\n=== Summary: is the between-arm difference larger than seed noise? ===\n")
  print(cmp |> group_by(same_arm) |>
          summarise(n_pairs = n(), mean_ari = round(mean(ari), 3),
                    min_ari = min(ari), max_ari = max(ari), .groups = "drop"))
  cat("\nIf within-arm ARI is high and between-arm ARI is near 0.25, the\n",
      "reorganisation is attributable to the aggregation rule rather than\n",
      "to EM initialisation.\n")
}

# =========================================================================
# H. Robustness: minprior, k-means agreement, internal indices (todo 14)
# =========================================================================
if (RUN_H_ROBUST) {
  cat("\n\n########## H. ROBUSTNESS ##########\n")
  arm <- "weighted"; dat <- load_X(arm); K_FIN <- 12

  cat("\n--- minprior sensitivity ---\n")
  mp <- map_dfr(c(0.005, 0.01, 0.02, 0.05), function(m) {
    f <- fit_mixture(dat$X, K_FIN, minprior = m)
    if (is.null(f)) return(tibble(minprior = m, k_kept = NA, BIC = NA))
    tibble(minprior = m, k_kept = length(unique(clusters(f))), BIC = BIC(f),
           smallest = min(table(clusters(f))))
  })
  print(as.data.frame(mp)); save_res(mp, "H_minprior")

  cat("\n--- agreement with spherical k-means at matched K ---\n")
  if (requireNamespace("skmeans", quietly = TRUE)) {
    set.seed(SEED)
    km <- skmeans::skmeans(dat$X, K_FIN, method = "pclust")
    f  <- fit_mixture(dat$X, K_FIN)
    a  <- round(ari(clusters(f), km$cluster), 3)
    cat(sprintf("  ARI(mixture, spherical k-means) at K = %d: %.3f\n", K_FIN, a))
    save_res(tibble(K = K_FIN, ari_mixture_vs_skmeans = a), "H_skmeans")
    cat("\n  contingency (rows = mixture, cols = k-means):\n")
    print(table(mixture = clusters(f), kmeans = km$cluster))
  } else cat("  package 'skmeans' not installed; skipped\n")

  cat("\n--- internal validity indices ---\n")
  f <- fit_mixture(dat$X, K_FIN); lab <- clusters(f)
  # cosine silhouette on a subsample: the full n x n matrix is ~3.7 GB
  set.seed(SEED); idx <- sample.int(nrow(dat$X), min(5000, nrow(dat$X)))
  Xs <- dat$X[idx, ]; ls <- lab[idx]
  D <- 1 - tcrossprod(Xs); diag(D) <- 0; D[D < 0] <- 0
  sil <- cluster::silhouette(ls, as.dist(D))
  mu <- t(sapply(sort(unique(lab)), function(c) {
    s <- colSums(dat$X[lab == c, , drop = FALSE]); s / sqrt(sum(s^2)) }))
  Sw <- sapply(seq_len(nrow(mu)), function(j)
    mean(1 - as.vector(dat$X[lab == sort(unique(lab))[j], , drop = FALSE] %*% mu[j, ])))
  M <- 1 - tcrossprod(mu); diag(M) <- Inf
  db <- mean(sapply(seq_along(Sw), function(i) max((Sw[i] + Sw[-i]) / M[i, -i])))
  idx_tab <- tibble(cosine_silhouette = round(mean(sil[, "sil_width"]), 4),
                    davies_bouldin_spherical = round(db, 4),
                    note = "silhouette on a 5,000-investor subsample")
  print(as.data.frame(idx_tab)); save_res(idx_tab, "H_internal_indices")
  cat("\nGeometric indices structurally favour hard clustering methods, which\n",
      "directly optimise compactness; they are reported for completeness.\n")
}

# =========================================================================
# I. US-only variants                                      (Ch.5.7, todo 12)
# =========================================================================
if (RUN_I_USVARIANTS) {
  cat("\n\n########## I. US-ONLY VARIANTS ##########\n")
  sec_ref <- read_parquet("reference/securities_master.parquet") |>
    select(issuer_id, iso_country)
  seqs <- read_parquet(sprintf("data/q_%s.parquet", QUARTER))
  seqs <- if ("chunk_id" %in% names(seqs)) filter(seqs, chunk_id == 1) else
    seqs |> group_by(investor_id) |> slice_head(n = 1) |> ungroup()
  us_share <- seqs |>
    mutate(iss = map(tokens, ~ tibble(issuer_id = as.character(.x)))) |>
    select(investor_id, iss) |> unnest(iss) |>
    left_join(sec_ref, by = "issuer_id") |>
    group_by(investor_id) |>
    summarise(n_known = sum(!is.na(iso_country)),
              pct_us = mean(iso_country == "US", na.rm = TRUE), .groups = "drop") |>
    mutate(pct_us = if_else(n_known == 0, 0, pct_us))

  grid <- expand.grid(arm = c("baseline", "weighted"), thr = c(0.90, 0.95),
                      stringsAsFactors = FALSE)
  out <- list()
  for (i in seq_len(nrow(grid))) {
    arm <- grid$arm[i]; thr <- grid$thr[i]
    dat <- load_X(arm)
    keep <- us_share$investor_id[us_share$pct_us >= thr]
    sel <- dat$meta$investor_id %in% keep
    Xs <- dat$X[sel, , drop = FALSE]
    cat(sprintf("\n  %s, threshold %.0f%%: %s investors\n", arm, 100*thr,
                format(nrow(Xs), big.mark = ",")))
    sw <- map_dfr(K_GRID, function(k) {
      f <- fit_mixture(Xs, k)
      if (is.null(f)) return(tibble(k_asked = k, k_kept = NA, BIC = NA))
      tibble(k_asked = k, k_kept = length(unique(clusters(f))), BIC = BIC(f))
    })
    best <- sw |> filter(!is.na(BIC)) |> slice_min(BIC, n = 1)
    out[[length(out)+1]] <- tibble(arm, threshold = thr, n = nrow(Xs),
                                   best_k = best$k_asked,
                                   k_kept = best$k_kept,
                                   BIC = round(best$BIC))
    cat(sprintf("    BIC-best k = %d, components kept = %d\n",
                best$k_asked, best$k_kept))
  }
  usv <- bind_rows(out)
  cat("\n=== US-restricted variants ===\n"); print(as.data.frame(usv))
  save_res(usv, "I_us_variants")
  cat("\nRe-run Clusters.r with ARM and US_THRESHOLD set to each row to obtain\n",
      "the full characterisation for whichever variants are reported.\n")
}

# =========================================================================
# J. The unnamed issuer in the index component             (Ch.5.7, todo 13)
# =========================================================================
if (RUN_J_UNNAMED) {
  cat("\n\n########## J. UNNAMED ISSUERS ##########\n")
  sec_ref <- read_parquet("reference/securities_master.parquet")
  seqs <- read_parquet(sprintf("data/q_%s.parquet", QUARTER))
  seqs <- if ("chunk_id" %in% names(seqs)) filter(seqs, chunk_id == 1) else
    seqs |> group_by(investor_id) |> slice_head(n = 1) |> ungroup()

  held <- seqs |>
    mutate(iss = map(tokens, ~ tibble(issuer_id = as.character(.x)))) |>
    select(investor_id, iss) |> unnest(iss)
  n_inv <- n_distinct(held$investor_id)

  unnamed <- held |>
    distinct(investor_id, issuer_id) |>
    count(issuer_id, name = "n_holders") |>
    left_join(sec_ref |> select(issuer_id, entity_proper_name, iso_country,
                                sector_code, market_cap),
              by = "issuer_id") |>
    filter(is.na(entity_proper_name)) |>
    mutate(pct_of_investors = round(100 * n_holders / n_inv, 1)) |>
    arrange(desc(n_holders))

  cat(sprintf("%s issuers held, of which %s (%.1f%%) have no name in the ",
              format(n_distinct(held$issuer_id), big.mark = ","),
              format(nrow(unnamed), big.mark = ","),
              100 * nrow(unnamed) / n_distinct(held$issuer_id)),
      "reference table.\n", sep = "")
  cat("\n=== Most widely held unnamed issuers ===\n")
  print(as.data.frame(head(unnamed, 20)))
  save_res(unnamed, "J_unnamed_issuers")

  cat("\nThese ids can be resolved with a single WRDS lookup:\n")
  cat("  SELECT factset_entity_id, entity_proper_name, iso_country\n")
  cat("  FROM factset.edm_standard_entity WHERE factset_entity_id IN (...)\n")
  cat("If they resolve, the reference table is incomplete and should be\n",
      "rebuilt; if they do not, the ids are absent from the entity master\n",
      "and the gap should be reported as a data limitation.\n")
}

# =========================================================================
# K. Session information                                    (App.A, todo 16)
# =========================================================================
if (RUN_K_SESSION) {
  cat("\n\n########## K. SESSION INFORMATION ##########\n")
  pk <- c("arrow","dplyr","tidyr","purrr","flexmix","circlus","skmeans",
          "cluster","RPostgres","dbplyr","jsonlite")
  vers <- map_dfr(pk, function(p) tibble(
    package = p,
    version = tryCatch(as.character(packageVersion(p)), error = function(e) NA)))
  info <- tibble(item = c("R version", "platform", "seed", "quarter"),
                 value = c(R.version.string, R.version$platform,
                           as.character(SEED), QUARTER))
  cat("\n=== Environment ===\n"); print(as.data.frame(info))
  cat("\n=== Package versions ===\n"); print(as.data.frame(vers))
  save_res(vers, "K_package_versions"); save_res(info, "K_environment")
}

# =========================================================================
# L. Export the full descriptive tables                     (App.B, todo 17)
# =========================================================================
if (RUN_L_EXPORT) {
  cat("\n\n########## L. EXPORTING SAVED CLUSTER TABLES ##########\n")
  for (f in list.files(pattern = "^cluster_assignment_.*\\.rds$")) {
    tag <- sub("^cluster_assignment_(.*)\\.rds$", "\\1", f)
    obj <- readRDS(f)
    for (el in c("style_tab", "distinctive", "assoc", "sweep")) {
      if (!is.null(obj[[el]])) {
        nm <- sprintf("L_%s_%s", tag, el)
        save_res(as.data.frame(obj[[el]]), nm)
      }
    }
    if (!is.null(obj$rho))
      save_res(tibble(cluster = seq_along(obj$rho), rho = round(obj$rho, 3)),
               sprintf("L_%s_rho", tag))
  }
}

cat("\n\nAll requested sections complete. Results in", RES_DIR, "/\n")