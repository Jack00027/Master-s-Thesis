library(arrow)
library(dplyr)
library(ggplot2)
library(skmeans)

# We normalize the embeddings

normalize <- function(path) {
  df <- read_parquet(path)
  
  meta <- df |> select(investor_id, quarter_end, investor_type)
  X    <- df |> select(starts_with("dim_")) |> as.matrix()
  
  norms  <- sqrt(rowSums(X^2))
  X_norm <- X / norms                       # row-wise division (column-major recycling)
  rownames(X_norm) <- df$investor_id
  
  list(meta = meta, X = X_norm, norms = norms)
}

q <- normalize("embeddings/q_2019-10-01.parquet")

# cluster the normalized embeddings using spherical k-means
set.seed(42)
k <- 6
skm <- skmeans(q$X, k, method = "pclust")

# ── WRDS connection ────────────────────────────────────────────
wrds <- dbConnect(
  Postgres(),
  host = "wrds-pgdata.wharton.upenn.edu", dbname = "wrds",
  port = 9737, sslmode = "require",
  user = Sys.getenv("WRDS_USER"), password = Sys.getenv("WRDS_PASSWORD")
)

tbl_style <- tbl(wrds, in_schema("factset_own", "own_ent_institutions")) |>
    select(investor_id = factset_entity_id, style) |>
    collect()

# Join cluster labels with metadata and style
clustered_data <- q$meta %>%
  mutate(cluster = skm$cluster) %>%
  left_join(tbl_style, by = "investor_id") %>%
  filter(!is.na(style)) %>%  # Optional: focus on investors with known styles
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

# visualize the embeddings in 2D using umap, colored by cluster
library(umap)
umap_model <- umap(q$X)
umap_data <- data.frame(UMAP1 = umap_model$layout[, 1], UMAP2 = umap_model$layout[, 2], cluster = factor(skm$cluster))
print(ggplot(umap_data, aes(x = UMAP1, y = UMAP2, color = cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "UMAP of Normalized Embeddings Colored by Cluster") +
  theme_minimal() +
  theme(legend.position = "bottom")
)
