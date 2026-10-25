ar_r <- "E:/autoresearch/tools/R/ar.R"; if (file.exists(ar_r)) source(ar_r)  # autoresearch toolkit, when vendored there
# R references for arsc.stats.wilcoxon_signed_rank and perm_test on 60 seeded random vectors (tests/test_stats_de.py).
# Usage: Rscript benchmarks/r_reference_stats_small.R <out_dir>   (writes R_small.csv; benchmarks/regen_refs.py stats_parity)
args <- commandArgs(trailingOnly = TRUE); stopifnot(length(args) == 1)
dir.create(args[1], recursive = TRUE, showWarnings = FALSE)
set.seed(42); rows <- list()
for (i in 1:60) {
  n <- sample(c(3:12, 20, 30, 49, 50, 60), 1); x <- rnorm(n, 0.3)
  if (i %% 3 == 0) x <- round(x, 1)                 # ties
  if (i %% 5 == 0) x[sample(n, 1)] <- 0              # zero
  p <- suppressWarnings(wilcox.test(x, exact = n < 50)$p.value)
  y <- rnorm(sample(3:8, 1)); 
  med <- function(a, b) median(a) - median(b)
  z <- c(x[1:min(n, 6)], y); nx <- min(n, 6); cmb <- combn(length(z), nx)
  obs <- med(z[1:nx], z[-(1:nx)]); null <- apply(cmb, 2, function(i) med(z[i], z[-i]))
  rows[[i]] <- data.frame(id = i, n = n, x = paste(x, collapse = ";"), p_wilcox = p, xp = paste(z[1:nx], collapse=";"), yp = paste(z[-(1:nx)], collapse=";"),
                          p_perm = mean(abs(null) >= abs(obs) - 1e-12), stat = obs)
}
write.csv(do.call(rbind, rows), file.path(args[1], "R_small.csv"), row.names = FALSE)
