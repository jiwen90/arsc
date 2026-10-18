# R references for arsc.stats rank / stratified / rarefaction helpers (STATS_PARITY.md).
# Usage: Rscript benchmarks/r_reference_stats_resampling.R <dir written by benchmarks/stats_resampling_cases.py>
# Needs: stats (base), vegan, lme4.
args <- commandArgs(trailingOnly = TRUE)
d <- if (length(args)) args[1] else file.path(Sys.getenv("ARSC_SCRATCH"), "stats_resampling")
if (nzchar(Sys.getenv("ARSC_RLIB"))) .libPaths(c(Sys.getenv("ARSC_RLIB"), .libPaths()))
rd <- function(f) read.csv(file.path(d, f), stringsAsFactors = FALSE)

# --- wilcox.test(conf.int = TRUE)
cs <- rd("hl_cases.csv"); vv <- rd("hl_values.csv")
alt <- c("two-sided" = "two.sided", greater = "greater", less = "less")
hl <- do.call(rbind, lapply(seq_len(nrow(cs)), function(i) {
  k <- cs[i, ]
  x <- vv$value[vv$case == k$case & vv$grp == "x"]; y <- vv$value[vv$case == k$case & vv$grp == "y"]
  ex <- if (is.na(k$exact) || k$exact == "NA") NULL else as.logical(k$exact)
  r <- suppressWarnings(switch(k$kind,
    one = wilcox.test(x, alternative = alt[[k$alternative]], conf.int = TRUE, conf.level = k$conf_level,
                      exact = ex, correct = as.logical(k$correct)),
    paired = wilcox.test(x, y, paired = TRUE, alternative = alt[[k$alternative]], conf.int = TRUE,
                         conf.level = k$conf_level, exact = ex, correct = as.logical(k$correct)),
    two = wilcox.test(x, y, alternative = alt[[k$alternative]], conf.int = TRUE, conf.level = k$conf_level,
                      exact = ex, correct = as.logical(k$correct))))
  data.frame(case = k$case, statistic = unname(r$statistic), p = r$p.value, estimate = unname(r$estimate),
             ci_lo = r$conf.int[1], ci_hi = r$conf.int[2], conf_level = attr(r$conf.int, "conf.level"),
             method = r$method)
}))
write.csv(hl, file.path(d, "R_hl.csv"), row.names = FALSE)

# --- mantelhaen.test (2 x 2 x K, asymptotic)
mt <- rd("mh_tables.csv")
mh <- do.call(rbind, lapply(split(mt, mt$case), function(t) {
  arr <- array(0, c(2, 2, nrow(t)))
  arr[1, 1, ] <- t$a; arr[1, 2, ] <- t$b; arr[2, 1, ] <- t$c; arr[2, 2, ] <- t$d
  r <- mantelhaen.test(arr, correct = as.logical(t$correct[1]))
  data.frame(case = t$case[1], or_mh = unname(r$estimate), ci_lo = r$conf.int[1], ci_hi = r$conf.int[2],
             statistic = unname(r$statistic), p = r$p.value)
}))
write.csv(mh, file.path(d, "R_mh.csv"), row.names = FALSE)

# --- vegan::rarefy (expected richness + se)
rc <- rd("rarefy_counts.csv")
rf <- do.call(rbind, lapply(split(rc, rc$case), function(t) {
  r <- suppressWarnings(vegan::rarefy(t$size, t$depth[1], se = TRUE))
  data.frame(case = t$case[1], depth = t$depth[1], S = r[1], se = r[2])
}))
write.csv(rf, file.path(d, "R_rarefy.csv"), row.names = FALSE)

# --- ICC from lme4 REML variance components (balanced data: equals the ANOVA estimator when > 0)
ic <- rd("icc_values.csv")
icc <- do.call(rbind, lapply(split(ic, ic$case), function(t) {
  f <- if (length(unique(t$stratum)) > 1) y ~ stratum + (1 | group) else y ~ 1 + (1 | group)
  m <- lme4::lmer(f, data = t, REML = TRUE)
  vc <- as.data.frame(lme4::VarCorr(m))
  vb <- vc$vcov[vc$grp == "group"]; vw <- vc$vcov[vc$grp == "Residual"]
  data.frame(case = t$case[1], icc_reml = vb / (vb + vw), singular = lme4::isSingular(m))
}))
write.csv(icc, file.path(d, "R_icc.csv"), row.names = FALSE)

# --- OLS t of x adjusted for covariates (observed statistic of the Freedman-Lane test)
fl <- rd("fl_values.csv")
flr <- do.call(rbind, lapply(split(fl, fl$case), function(t) {
  zs <- c("z0", "z1")[c(any(is.finite(t$z0)), any(is.finite(t$z1)))]
  f <- as.formula(paste("y ~", paste(c(zs, "x"), collapse = " + ")))
  s <- summary(lm(f, data = t))$coefficients
  data.frame(case = t$case[1], beta = s["x", 1], t = s["x", 3], p = s["x", 4])
}))
write.csv(flr, file.path(d, "R_fl.csv"), row.names = FALSE)
cat("wrote R references to", d, "\n")
