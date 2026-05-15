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
k <- 10
skm <- skmeans(q$X, k, m = 1, method = "pclust", control = list(nruns = 100))


