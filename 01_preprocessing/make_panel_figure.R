## ---------------------------------------------------------------------------
## make_panel_figure.R
##
## Figure for Section 3.5: investors and issuers per quarter over the
## 85-quarter panel, with the semi-annual reporting cycle made visible.
##
## Input :  data/q_YYYY-MM-DD.parquet   (the filtered token sequences)
## Cache :  results/A_panel_by_quarter.csv
## Output:  figures/panel_by_quarter.pdf
##
## Run from the project root:  Rscript make_panel_figure.R
##                             Rscript make_panel_figure.R --rebuild
## Needs the arrow package only when the cache has to be rebuilt.
## ---------------------------------------------------------------------------

SEQ_DIR <- "data"
CACHE   <- "results/A_panel_by_quarter.csv"
OUT     <- "figures/panel_by_quarter.pdf"

rebuild <- "--rebuild" %in% commandArgs(TRUE)

dir.create(dirname(OUT),   showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(CACHE), showWarnings = FALSE, recursive = TRUE)

## ========================= counts from the sequences =======================
## One row per chunk; an investor with more than 62 positions occupies
## several chunks, so investors are counted by unique investor_id and issuers
## by the union of the token lists over all chunks of the quarter.

build_counts <- function(dir) {
  if (!requireNamespace("arrow", quietly = TRUE))
    stop("Rebuilding the cache needs the arrow package: install.packages(\"arrow\")")

  files <- sort(list.files(dir, pattern = "^q_[0-9]{4}-[0-9]{2}-[0-9]{2}\\.parquet$",
                           full.names = TRUE))
  if (!length(files)) stop("No files matched ", file.path(dir, "q_*.parquet"))

  ## column names, taken from the first file rather than assumed
  nm      <- names(arrow::open_dataset(files[1])$schema)
  col_inv <- grep("^investor(_id)?$|investor", nm, value = TRUE)[1]
  col_tok <- grep("^tokens?$|token", nm, value = TRUE)[1]
  if (is.na(col_inv) || is.na(col_tok))
    stop("Could not find the investor and token columns. Schema: ",
         paste(nm, collapse = ", "))
  message("Reading ", length(files), " quarters from ", dir,
          "  (", col_inv, " / ", col_tok, ")")

  res <- vector("list", length(files))
  for (k in seq_along(files)) {
    d  <- arrow::read_parquet(files[k], col_select = c(col_inv, col_tok))
    tk <- unlist(d[[col_tok]], use.names = FALSE)
    tk <- tk[!is.na(tk)]
    if (is.character(tk)) tk <- tk[!grepl("^\\[", tk)]   # drop [CLS], [PAD], ...
    res[[k]] <- data.frame(
      quarter     = sub("^q_(.*)\\.parquet$", "\\1", basename(files[k])),
      n_investors = length(unique(d[[col_inv]])),
      n_issuers   = length(unique(tk)),
      stringsAsFactors = FALSE)
    message(sprintf("  %s  investors %7s  issuers %7s", res[[k]]$quarter,
                    format(res[[k]]$n_investors, big.mark = ","),
                    format(res[[k]]$n_issuers,   big.mark = ",")))
  }
  do.call(rbind, res)
}

if (rebuild || !file.exists(CACHE)) {
  df <- build_counts(SEQ_DIR)
  write.csv(df, CACHE, row.names = FALSE)
  message("Wrote ", CACHE)
} else {
  df <- read.csv(CACHE, stringsAsFactors = FALSE)
  message("Read ", CACHE, " (delete it or pass --rebuild to recompute)")
}

## --- locate the three columns without assuming exact names -----------------
pick <- function(pattern, exclude = NULL) {
  nm <- names(df)[grepl(pattern, names(df), ignore.case = TRUE)]
  if (!is.null(exclude)) nm <- nm[!grepl(exclude, nm, ignore.case = TRUE)]
  if (length(nm) == 0) NA_character_ else nm[1]
}
col_q   <- pick("quarter|date|period")
col_inv <- pick("invest", exclude = "type|share|pct|pattern")
col_iss <- pick("issuer|stock|firm|security", exclude = "type|share|pct")

if (any(is.na(c(col_q, col_inv, col_iss)))) {
  stop("Could not identify columns. Found: ", paste(names(df), collapse = ", "),
       "\nSet col_q / col_inv / col_iss by hand.")
}
message("Using columns: ", col_q, " / ", col_inv, " / ", col_iss)

qlab <- as.character(df[[col_q]])
inv  <- as.numeric(df[[col_inv]])
iss  <- as.numeric(df[[col_iss]])

## --- parse the quarter label (handles "2005-Q1" and "2005-03-31") ----------
if (all(grepl("^[0-9]{4}-Q[1-4]$", qlab))) {
  yr <- as.integer(substr(qlab, 1, 4))
  qn <- as.integer(substr(qlab, 7, 7))
} else {
  d  <- as.Date(qlab)
  if (any(is.na(d))) stop("Cannot parse the quarter column: ", qlab[which(is.na(d))[1]])
  yr <- as.integer(format(d, "%Y"))
  qn <- (as.integer(format(d, "%m")) - 1L) %/% 3L + 1L
}

o   <- order(yr, qn)
yr  <- yr[o]; qn <- qn[o]; inv <- inv[o]; iss <- iss[o]
n   <- length(inv)
x   <- seq_len(n)

## --- console check: the seasonal gap ---------------------------------------
hi <- qn %in% c(2L, 4L)
cat(sprintf("Quarters: %d (%d-Q%d to %d-Q%d)\n", n, yr[1], qn[1], yr[n], qn[n]))
cat(sprintf("Investors  Q2/Q4 mean %8.0f | Q1/Q3 mean %8.0f | gap %5.1f%%\n",
            mean(inv[hi]), mean(inv[!hi]),
            100 * (mean(inv[hi]) / mean(inv[!hi]) - 1)))
cat(sprintf("Issuers    Q2/Q4 mean %8.0f | Q1/Q3 mean %8.0f | gap %5.1f%%\n",
            mean(iss[hi]), mean(iss[!hi]),
            100 * (mean(iss[hi]) / mean(iss[!hi]) - 1)))

## --- palette ---------------------------------------------------------------
c_inv  <- "#1F3B63"   # investors: dark navy
c_iss  <- "#8FA6C4"   # issuers:   light blue-grey
c_band <- "#EDEDED"   # stress-period shading

## --- device ----------------------------------------------------------------
pdf(OUT, width = 6.5, height = 2.9, pointsize = 10)
layout(matrix(c(1, 2), nrow = 1), widths = c(2.7, 1))

## ======================= LEFT PANEL: series over time ======================
par(mar = c(2.2, 4.9, 0.8, 0.6), mgp = c(3.6, 0.6, 0), las = 1, tcl = -0.25)

ylim <- c(0, max(inv, iss) * 1.08)
plot(x, inv, type = "n", axes = FALSE, xlab = "", ylab = "",
     xlim = c(0.5, n + 0.5), ylim = ylim, xaxs = "i")

## shaded stress periods
band <- function(y0, q0, y1, q1) {
  i <- which(yr == y0 & qn == q0); j <- which(yr == y1 & qn == q1)
  if (length(i) && length(j))
    rect(i - 0.5, ylim[1], j + 0.5, ylim[2], col = c_band, border = NA)
}
band(2008, 3, 2009, 2)
band(2020, 1, 2020, 1)

## gridlines
abline(h = pretty(ylim), col = "grey88", lwd = 0.6)

## series
lines(x, iss, col = c_iss, lwd = 1.8, lty = 2)
lines(x, inv, col = c_inv, lwd = 1.8)
points(x, inv, pch = 21, cex = 0.62, lwd = 0.8,
       col = c_inv, bg = ifelse(hi, c_inv, "white"))

## axes: year ticks at every Q1, labelled every second year
q1 <- which(qn == 1L)
axis(1, at = q1, labels = FALSE, tcl = -0.15, col = "grey40")
lab <- q1[seq(1, length(q1), by = 2)]
axis(1, at = lab, labels = yr[lab], cex.axis = 0.78, col = "grey40")
axis(2, at = pretty(ylim), labels = format(pretty(ylim), big.mark = ","),
     cex.axis = 0.78, col = "grey40")
title(ylab = "Count", line = 3.6)
box(col = "grey40", lwd = 0.7)

legend("topleft", bty = "n", cex = 0.76, inset = c(0.01, 0),
       legend = c("Investors (filled: Q2/Q4)", "Distinct issuers held"),
       col = c(c_inv, c_iss), lty = c(1, 2), lwd = 1.8,
       pch = c(21, NA), pt.bg = c(c_inv, NA), pt.cex = 0.62, seg.len = 2.2)

## ================= RIGHT PANEL: panel mean by calendar quarter =============
par(mar = c(2.2, 4.9, 0.8, 0.6), mgp = c(3.6, 0.6, 0), las = 1)

m <- rbind(tapply(inv, qn, mean)[as.character(1:4)],
           tapply(iss, qn, mean)[as.character(1:4)])

bp <- barplot(m, beside = TRUE, col = c(c_inv, c_iss), border = NA,
              names.arg = rep("", 4), axes = FALSE,
              ylim = c(0, max(m) * 1.18), ylab = "")
## labels drawn by hand: barplot() silently drops names it thinks will not fit
mtext(paste0("Q", 1:4), side = 1, at = colMeans(bp), line = 0.15, cex = 0.78)
axis(2, at = pretty(c(0, max(m))),
     labels = format(pretty(c(0, max(m))), big.mark = ","),
     cex.axis = 0.78, col = "grey40")
title(ylab = "Panel mean", line = 3.6)
box(col = "grey40", lwd = 0.7)

legend("topright", bty = "n", cex = 0.72, inset = c(0, -0.02),
       legend = c("Investors", "Issuers"),
       fill = c(c_inv, c_iss), border = NA)

invisible(dev.off())
cat("Wrote ", OUT, "\n", sep = "")