library(arrow)
library(dplyr)
library(ggplot2)
library(skmeans)
library(tidyr)


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

q <- normalize("embeddings/q_2019-10-01.parquet")

# cluster the normalized embeddings using spherical k-means
set.seed(42)
k <- 6
skm <- skmeans(q$X, k, method = "pclust")

# load the style labels for the investors
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

# check the number of investors in each cluster
print(clustered_data %>%
  group_by(cluster) %>%
  summarise(count = n()) %>%
  arrange(cluster))


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