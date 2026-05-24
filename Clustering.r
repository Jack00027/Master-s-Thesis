library(arrow)
library(dplyr)
library(ggplot2)
library(skmeans)
library(tidyr)
library(cluster)   # silhouette()


# We normalize the embeddings
normalize <- function(path) {
  df <- read_parquet(path)
  
  meta <- df |> select(investor_id, quarter_end, investor_type)
  X    <- df |> select(starts_with("dim_")) |> as.matrix()
  
  norms  <- sqrt(rowSums(X^2))
  X_norm <- X / norms
  rownames(X_norm) <- df$investor_id
  
  list(meta = meta, X = X_norm, norms = norms)
}


# Internal cluster-quality metrics for L2-normalized embeddings on S^(d−1).
#   silhouette uses cosine distance (primary metric, range [-1, 1], higher better)
#   db_spherical uses spherical centroids + cosine distance (lower better)
#   db_euclidean is the textbook version (lower better; sanity check)
#   ch is textbook Euclidean Calinski–Harabasz (higher better)
#   pseudo_ch is a CH-style ratio on cosine dispersion (relative use only)
# Memory: silhouette materializes an n × n matrix (~800 MB at n = 10k).
# Pass `subsample` to evaluate silhouette on a random subset of investors.
cluster_metrics_sphere <- function(X, labels, subsample = NULL, seed = 42) {
  stopifnot(is.matrix(X), nrow(X) == length(labels))
  
  norms <- sqrt(rowSums(X^2))
  if (any(abs(norms - 1) > 1e-6)) {
    warning("Rows of X are not unit-norm; renormalizing.")
    X <- X / norms
  }
  
  n  <- nrow(X)
  cl <- sort(unique(labels))
  k  <- length(cl)
  if (k < 2) stop("Need at least 2 clusters to compute internal metrics.")
  
  # cosine silhouette: for unit-norm X, D = 1 − X X^T (one BLAS call)
  if (!is.null(subsample) && subsample < n) {
    set.seed(seed)
    idx <- sample.int(n, subsample)
    Xs  <- X[idx, , drop = FALSE]; ls <- labels[idx]
  } else {
    Xs <- X; ls <- labels
  }
  D <- 1 - tcrossprod(Xs)
  diag(D)  <- 0
  D[D < 0] <- 0
  sil      <- cluster::silhouette(ls, as.dist(D))
  sil_mean <- mean(sil[, "sil_width"])
  sil_per_cluster <- aggregate(
    sil[, "sil_width"],
    by  = list(cluster = sil[, "cluster"]),
    FUN = mean
  ) |> setNames(c("cluster", "mean_silhouette"))
  rm(D, Xs, ls); gc(verbose = FALSE)
  
  # cluster centers
  mu_sph <- t(sapply(cl, function(c) {
    s <- colSums(X[labels == c, , drop = FALSE]); s / sqrt(sum(s^2))
  }))
  mu_euc <- t(sapply(cl, function(c) colMeans(X[labels == c, , drop = FALSE])))
  nc <- as.integer(table(factor(labels, levels = cl)))
  
  # Davies–Bouldin (spherical: cosine distance + spherical centroids)
  S_sph <- sapply(seq_along(cl), function(j) {
    Xc <- X[labels == cl[j], , drop = FALSE]
    mean(1 - as.vector(Xc %*% mu_sph[j, ]))
  })
  M_sph <- 1 - tcrossprod(mu_sph); diag(M_sph) <- Inf
  db_spherical <- mean(sapply(seq_along(cl), function(i) {
    max((S_sph[i] + S_sph[-i]) / M_sph[i, -i])
  }))
  
  # Davies–Bouldin (textbook Euclidean)
  S_euc <- sapply(seq_along(cl), function(j) {
    Xc <- X[labels == cl[j], , drop = FALSE]
    mean(sqrt(rowSums(sweep(Xc, 2, mu_euc[j, ], `-`)^2)))
  })
  M_euc <- as.matrix(dist(mu_euc)); diag(M_euc) <- Inf
  db_euclidean <- mean(sapply(seq_along(cl), function(i) {
    max((S_euc[i] + S_euc[-i]) / M_euc[i, -i])
  }))
  
  # Calinski–Harabasz (textbook Euclidean)
  c_overall <- colMeans(X)
  W <- sum(sapply(seq_along(cl), function(j) {
    Xc <- X[labels == cl[j], , drop = FALSE]
    sum(rowSums(sweep(Xc, 2, mu_euc[j, ], `-`)^2))
  }))
  B  <- sum(nc * rowSums(sweep(mu_euc, 2, c_overall, `-`)^2))
  ch <- (B / (k - 1)) / (W / (n - k))
  
  # Pseudo-CH on the sphere (cosine dispersion; relative comparisons only)
  mu_overall_sph <- colSums(X)
  mu_overall_sph <- mu_overall_sph / sqrt(sum(mu_overall_sph^2))
  W_sph <- sum(sapply(seq_along(cl), function(j) {
    Xc <- X[labels == cl[j], , drop = FALSE]
    sum(1 - as.vector(Xc %*% mu_sph[j, ]))
  }))
  B_sph     <- sum(nc * (1 - as.vector(mu_sph %*% mu_overall_sph)))
  pseudo_ch <- (B_sph / (k - 1)) / (W_sph / (n - k))
  
  list(
    n = n, k = k,
    silhouette_mean        = sil_mean,
    silhouette_per_cluster = sil_per_cluster,
    db_spherical           = db_spherical,
    db_euclidean           = db_euclidean,
    ch                     = ch,
    pseudo_ch              = pseudo_ch
  )
}


q <- normalize("embeddings/q_2019-10-01.parquet")

# cluster the normalized embeddings using spherical k-means
set.seed(42)
k <- 6
skm <- skmeans(q$X, k, method = "pclust")

# internal cluster-quality metrics (computed on all investors, pre style join)
metrics <- cluster_metrics_sphere(q$X, skm$cluster)

cat(sprintf(
  "=== Internal cluster metrics — %s   n=%d   k=%d ===
  cosine silhouette  = %+.4f      (range [-1, 1], higher better)
  Davies–Bouldin     = %.4f spherical / %.4f Euclidean   (lower better)
  Calinski–Harabasz  = %.1f                              (higher better)
  pseudo-CH (sphere) = %.1f                              (relative only)\n",
  as.character(q$meta$quarter_end[1]),
  metrics$n, metrics$k,
  metrics$silhouette_mean,
  metrics$db_spherical, metrics$db_euclidean,
  metrics$ch, metrics$pseudo_ch
))


# load the style labels for the investors
tbl_style <- readRDS("factset_styles.rds")

# Join cluster labels with metadata and style
clustered_data <- q$meta %>%
  mutate(cluster = skm$cluster) %>%
  left_join(tbl_style, by = "investor_id") %>%
  filter(!is.na(style)) %>%
  collect()

# Visualize the distribution of styles across clusters
print(ggplot(clustered_data, aes(x = factor(cluster), fill = style)) +
  geom_bar(position = "fill") +
  labs(title = "Distribution of Investor Styles Across Clusters",
       x = "Cluster", y = "Proportion") +
  theme_minimal() +
  theme(legend.position = "bottom"))

sum(is.na(clustered_data$style))           # number of NAs
length(clustered_data$style)               # total rows
mean(is.na(clustered_data$style))          # proportion that are NA

# or as a one-liner with a formatted message:
cat(sprintf("style NAs: %d / %d (%.1f%%)\n",
            sum(is.na(clustered_data$style)),
            length(clustered_data$style),
            100 * mean(is.na(clustered_data$style))))

# check the number of investors in each cluster, with mean silhouette
print(
  clustered_data %>%
    group_by(cluster) %>%
    summarise(count = n(), .groups = "drop") %>%
    left_join(metrics$silhouette_per_cluster, by = "cluster") %>%
    arrange(cluster)
)


style_cluster_tables <- function(data, digits = 1) {
  # Cross-tab with all style × cluster combinations filled in
  counts <- data |>
    count(style, cluster, name = "n") |>
    complete(style, cluster, fill = list(n = 0))
  
  style_totals <- data |> count(style, name = "n_total")
  
  # Table 1: composition of each cluster (columns sum to 100%)
  # "Of the investors in cluster C, what % is each style?"
  style_within_cluster <- counts |>
    group_by(cluster) |>
    mutate(pct = round(100 * n / sum(n), digits)) |>
    ungroup() |>
    select(-n) |>
    pivot_wider(names_from = cluster, values_from = pct, names_prefix = "C") |>
    left_join(style_totals, by = "style") |>
    arrange(desc(n_total))
  
  # Table 2: distribution of each style across clusters (rows sum to 100%)
  # "Of the investors with style S, what % is in each cluster?"
  cluster_within_style <- counts |>
    group_by(style) |>
    mutate(pct = round(100 * n / sum(n), digits)) |>
    ungroup() |>
    select(-n) |>
    pivot_wider(names_from = cluster, values_from = pct, names_prefix = "C") |>
    left_join(style_totals, by = "style") |>
    arrange(desc(n_total))
  
  list(
    style_within_cluster = style_within_cluster,
    cluster_within_style = cluster_within_style
  )
}

# Usage
tables <- style_cluster_tables(clustered_data)

cat("=== Style composition within each cluster (columns sum to 100%) ===\n")
print(tables$style_within_cluster, n = Inf)

cat("\n=== Cluster distribution within each style (rows sum to 100%) ===\n")
print(tables$cluster_within_style, n = Inf)