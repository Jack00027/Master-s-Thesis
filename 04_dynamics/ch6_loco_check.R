# =============================================================================
# ch6_loco_check.R -- leave-one-component-out check of the anchor rules
#
# For each Chapter 5 component c: fit the three rules WITHOUT c's investors,
# then measure displacement of c's stable members (wt_overlap >= threshold)
# and of its movers. Run after ch6_02_stability_profile.R in the same session
# (needs X0, X1, stab).
# =============================================================================
if (!exists("stab") || !exists("X0") || !exists("X1"))
  stop("run ch6_02_stability_profile.R first, in this session")

med_disp <- function(R, ids) {
  if (length(ids) < 5) return(NA_real_)
  A0 <- X0[ids, , drop = FALSE]
  A1 <- X1[ids, , drop = FALSE]
  median(1 - rowSums((A0 %*% R) * A1))
}

strat_anchors <- function(ids, s, comp) {
  groups <- split(seq_along(ids), comp)
  out <- lapply(groups, function(i) {
    cut <- quantile(s[i], 1 - STAB_TOP_SHARE)
    ids[i][s[i] >= cut]
  })
  unlist(out, use.names = FALSE)
}

loco_one <- function(c) {
  inc <- stab$component %in% c

  rest   <- stab$investor_id[!inc]
  s_rest <- stab$wt_overlap[!inc]
  c_rest <- stab$component[!inc]

  anc_thr <- rest[s_rest >= STAB_THRESHOLD]
  anc_str <- strat_anchors(rest, s_rest, c_rest)

  R_thr   <- procrustes(X0[anc_thr, ], X1[anc_thr, ])
  R_strat <- procrustes(X0[anc_str, ], X1[anc_str, ])
  R_all   <- procrustes(X0[rest, ],    X1[rest, ])

  mem   <- stab$investor_id[inc]
  s_mem <- stab$wt_overlap[inc]
  st <- mem[s_mem >= STAB_THRESHOLD]
  mv <- mem[s_mem <  STAB_THRESHOLD]

  data.frame(
    component = c,
    n_stable  = length(st),
    st_thr    = med_disp(R_thr,   st),
    st_strat  = med_disp(R_strat, st),
    st_all    = med_disp(R_all,   st),
    mv_thr    = med_disp(R_thr,   mv),
    mv_strat  = med_disp(R_strat, mv),
    mv_all    = med_disp(R_all,   mv)
  )
}

comps <- sort(unique(na.omit(stab$component)))
loco2 <- do.call(rbind, lapply(comps, loco_one))

best <- c("thr", "strat", "all")[apply(loco2[, c("st_thr", "st_strat", "st_all")], 1, which.min)]
loco2$best_on_stable <- best

print(format(loco2, digits = 3), row.names = FALSE)
cat("\nBest rule on stable members (lowest st_ displacement):\n")
print(table(loco2$best_on_stable))

write.csv(loco2, out_file("loco_check_", PY0, "_", PY1, ".csv"), row.names = FALSE)
