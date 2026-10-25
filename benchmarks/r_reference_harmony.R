#!/usr/bin/env Rscript
# R harmony references for arsc.run_harmony: one and several batch variables, distinct theta and lambda per
# variable, fixed and automatic lambda. Runs with whichever harmony the library path provides (1.2.x, 2.0.x,
# 2.1.x), so the same script makes the reference for each version.
#
# Usage: Rscript benchmarks/r_reference_harmony.R <input_dir> [out_root] [harmony_lib]
#   input_dir  : pcs.{f64,json} + meta.csv written by benchmarks/harmony_input.py
#   out_root   : default $ARSC_SCRATCH/ref_harmony; outputs go to <out_root>/<harmony version>/
#   harmony_lib: library holding the harmony version to use (put first on .libPaths); ARSC_RLIB is added after it
#
# Per case: <case>_Z (corrected embedding), <case>_Y0 (R's k-means start, K x d, so arsc can start from the same
# centroids), <case>_trace.json (objective trajectory, clustering rounds, parameters) and <case>_seed2_Z (the
# same run after set.seed(2): R's own seed-to-seed spread, the noise floor for a comparison from different starts).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r_reference_harmony.R <input_dir> [out_root] [harmony_lib]")
if (nzchar(Sys.getenv("ARSC_RLIB"))) .libPaths(c(Sys.getenv("ARSC_RLIB"), .libPaths()))
if (length(args) >= 3) .libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages({ library(Matrix); library(harmony); library(jsonlite) })
inp <- args[1]
out_root <- if (length(args) >= 2) args[2] else file.path(Sys.getenv("ARSC_SCRATCH", "scratch"), "ref_harmony")
ver <- as.character(packageVersion("harmony"))
out <- file.path(out_root, ver)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "r_common.R"))
cat("harmony", ver, "from", find.package("harmony"), "\n")

dm <- fromJSON(file.path(inp, "pcs.json"))$dim
pcs <- matrix(readBin(file.path(inp, "pcs.f64"), "double", n = prod(dm), size = 8), dm[1], dm[2])
meta <- read.csv(file.path(inp, "meta.csv"), stringsAsFactors = FALSE)
P <- "patient_alias"; L <- "n10x_lane_name"; T <- "patient_timepoint"
# lambda = "default" leaves RunHarmony's own default (1 in 1.2, NULL = automatic in 2.x)
cases <- list(
  lane1    = list(vars = L, theta = NULL, lambda = "default"),
  mv2      = list(vars = c(P, L), theta = NULL, lambda = "default"),
  mv2_tl   = list(vars = c(P, L), theta = c(4, 1), lambda = c(0.5, 2)),
  mv2_auto = list(vars = c(P, L), theta = c(1, 3), lambda = NULL),
  mv3      = list(vars = c(P, L, T), theta = c(1, 2, 0.5), lambda = "default")
)
getZ <- function(h) if (ver >= "2") h$getZcorr() else h$Z_corr  # d x N
run <- function(cs, max_iter) {
  a <- list(data_mat = pcs, meta_data = meta, vars_use = cs$vars, max_iter = max_iter, return_object = TRUE,
            verbose = FALSE)
  if (!is.null(cs$theta)) a$theta <- cs$theta
  if (is.null(cs$lambda)) a["lambda"] <- list(NULL) else if (!identical(cs$lambda, "default")) a$lambda <- cs$lambda
  do.call(RunHarmony, a)
}
for (nm in names(cases)) {
  cs <- cases[[nm]]
  set.seed(1); h0 <- run(cs, 0)        # same RNG stream up to the k-means start
  Y0 <- t(h0$Y)                        # K x d
  set.seed(1)
  tm(nm, h <- run(cs, 10))
  write_dense(t(getZ(h)), file.path(out, paste0(nm, "_Z")))
  write_dense(Y0, file.path(out, paste0(nm, "_Y0")))
  write_json(list(harmony = ver, vars = cs$vars, theta = cs$theta,
                  lambda = if (is.null(cs$lambda)) "auto" else cs$lambda,
                  objective_harmony = h$objective_harmony, objective_kmeans = h$objective_kmeans,
                  kmeans_rounds = h$kmeans_rounds),
             file.path(out, paste0(nm, "_trace.json")), digits = NA, auto_unbox = TRUE)
  cat(nm, "objective:", round(h$objective_harmony, 3), "\n")
  set.seed(2)                          # a second seed: R's own run-to-run spread (the noise floor)
  write_dense(t(getZ(run(cs, 10))), file.path(out, paste0(nm, "_seed2_Z")))
}
write_timings(out)
