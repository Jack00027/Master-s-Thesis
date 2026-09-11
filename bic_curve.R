# =============================================================================
# figures/bic_curve.pdf
# BIC of the spherical Cauchy mixture against k, fixed 5,000-investor subsample,
# weighted arm, 2025-Q4. Values already reported in tab:k-selection; this
# script only plots them.
#
# Single seed (4525) by design — the chapter now reports one seed rather than
# two, so this produces a one-line plot. Keeping IN_FILES as a vector in case
# a second seed is ever added back later.
# =============================================================================

library(tidyverse)

IN_FILES <- c(
  "choosing_k_weighted_2025-12-31.rds"
  # add a second path here if a second-seed run was saved separately, e.g.
  # , "choosing_k_weighted_2025-12-31_seed2.rds"
)
OUT_FILE <- "figures/bic_curve.pdf"

# ---- 1. Load each file and locate the BIC column ---------------------------
# The BIC values live in one of $res or $mix (both are one-row-per-k tibbles
# aligned with $k_grid). Column name is discovered by matching "bic",
# case-insensitively, rather than assumed — print what's found so a wrong
# match is caught before it's plotted.

find_bic_col <- function(tbl) {
  hit <- grep("bic", names(tbl), ignore.case = TRUE, value = TRUE)
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

load_one <- function(path) {
  raw <- readRDS(path)

  candidates <- list(res = raw$res, mix = raw$mix)
  found <- imap(candidates, ~ find_bic_col(.x))
  found <- found[!is.na(found)]

  if (length(found) == 0) {
    stop(sprintf(
      "No column matching 'bic' found in $res or $mix of %s.\n%s",
      path,
      "Print names(raw$res) and names(raw$mix) and check manually."
    ))
  }

  src_name <- names(found)[1]
  col_name <- found[[1]]
  cat(sprintf("%s: using %s$%s as the BIC column\n", path, src_name, col_name))

  tibble(
    k       = raw$k_grid,
    mix_bic = candidates[[src_name]][[col_name]],
    seed    = as.character(raw$seed)
  )
}

scan_df <- map_dfr(IN_FILES, load_one)

n_seeds <- n_distinct(scan_df$seed)
cat(sprintf("\nLoaded %d file(s), %d distinct seed(s): %s\n",
            length(IN_FILES), n_seeds, paste(unique(scan_df$seed), collapse = ", ")))
if (n_seeds < 2) {
  cat("Only one seed present — the plot will show a single line.\n",
      "Add a second file to IN_FILES for the two-seed comparison.\n", sep = "")
}

scan_df <- scan_df |>
  mutate(
    k    = as.integer(k),
    seed = factor(seed, labels = paste("Seed", seq_along(unique(seed))))
  )

cat(sprintf("\nk range in data: %d to %d\n", min(scan_df$k), max(scan_df$k)))

# ---- 2. Per-component penalty for the reference line -----------------------
# (d+1) * log(n), d = 64, n = 32,389 (full cross-section) — the value already
# quoted in the text as "roughly 675". Kept fixed rather than recomputed from
# the subsample, to stay consistent with the chapter narrative.

d <- 64
n_sub <- 5000  # subsample size the scan was fit on (raw$sub_n) — NOT n_full.
               # Using n_full here was the same error the chapter text had:
               # it inflates the penalty (675 vs ~554) and understates the
               # gain/penalty ratio relative to what's actually reported.
penalty <- (d + 1) * log(n_sub)
cat(sprintf("Per-component BIC penalty: %.1f\n", penalty))

ref_k <- scan_df |> filter(k == min(max(k, na.rm = TRUE), 6)) |> pull(k) |> unique() |> min()
ref_y_top    <- scan_df |> filter(k == ref_k) |> pull(mix_bic) |> min()
ref_y_bottom <- ref_y_top - penalty

# ---- 3. Plot ----------------------------------------------------------------

p <- ggplot(scan_df, aes(x = k, y = mix_bic, color = seed, group = seed)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.8) +
  annotate(
    "segment",
    x = ref_k, xend = ref_k,
    y = ref_y_top, yend = ref_y_bottom,
    linetype = "dashed", color = "grey30", linewidth = 0.5
  ) +
  annotate(
    "text",
    x = ref_k + 0.3, y = (ref_y_top + ref_y_bottom) / 2,
    label = sprintf("per-component\npenalty (%.0f)", penalty),
    hjust = 0, size = 3, color = "grey30"
  ) +
  scale_x_continuous(breaks = seq(2, 20, by = 2)) +
  labs(x = "k", y = "Mixture BIC", color = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = if (n_seeds > 1) "top" else "none",
        panel.grid.minor = element_blank())

print(p)

# ---- 4. Save ----------------------------------------------------------------

dir.create(dirname(OUT_FILE), showWarnings = FALSE, recursive = TRUE)
ggsave(OUT_FILE, plot = p, width = 6.5, height = 4.2, units = "in", device = "pdf")
cat(sprintf("\nSaved: %s\n", OUT_FILE))