## ---------------------------------------------------------------------------
## make_panel_figure.R
##
## Figure for Section 3.5: investors and issuers per quarter over the
## 85-quarter panel, with the semi-annual reporting cycle made visible.
##
## Input :  results/A_panel_by_quarter.csv
## Output:  figures/panel_by_quarter.pdf
##
## Run from the project root:  Rscript make_panel_figure.R
## Base R only -- no package dependencies.
## ---------------------------------------------------------------------------

IN  <- "results/A_panel_by_quarter.csv"
OUT <- "figures/panel_by_quarter.pdf"

dir.create(dirname(OUT), showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(IN))

df <- read.csv(IN, stringsAsFactors = FALSE)

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
pdf(OUT, width = 8.2, height = 3.2, pointsize = 11)
layout(matrix(c(1, 2), nrow = 1), widths = c(2.7, 1))

## ======================= LEFT PANEL: series over time ======================
par(mar = c(3.2, 4.2, 1.2, 0.6), mgp = c(2.6, 0.6, 0), las = 1, tcl = -0.25)

ylim <- c(0, max(inv, iss) * 1.08)
plot(x, inv, type = "n", axes = FALSE, xlab = "", ylab = "Count",
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
box(col = "grey40", lwd = 0.7)

legend("topleft", bty = "n", cex = 0.76, inset = c(0.01, 0),
       legend = c("Investors (filled: Q2/Q4)", "Distinct issuers held"),
       col = c(c_inv, c_iss), lty = c(1, 2), lwd = 1.8,
       pch = c(21, NA), pt.bg = c(c_inv, NA), pt.cex = 0.62, seg.len = 2.2)

## ================= RIGHT PANEL: panel mean by calendar quarter =============
par(mar = c(3.2, 3.4, 1.2, 0.6), mgp = c(2.2, 0.6, 0), las = 1)

m <- rbind(tapply(inv, qn, mean)[as.character(1:4)],
           tapply(iss, qn, mean)[as.character(1:4)])

bp <- barplot(m, beside = TRUE, col = c(c_inv, c_iss), border = NA,
              names.arg = paste0("Q", 1:4), axes = FALSE,
              ylim = c(0, max(m) * 1.18), cex.names = 0.78,
              ylab = "Panel mean", cex.lab = 1)
axis(2, at = pretty(c(0, max(m))),
     labels = format(pretty(c(0, max(m))), big.mark = ","),
     cex.axis = 0.78, col = "grey40")
box(col = "grey40", lwd = 0.7)

legend("topright", bty = "n", cex = 0.72, inset = c(0, -0.02),
       legend = c("Investors", "Issuers"),
       fill = c(c_inv, c_iss), border = NA)

invisible(dev.off())
cat("Wrote ", OUT, "\n", sep = "")
