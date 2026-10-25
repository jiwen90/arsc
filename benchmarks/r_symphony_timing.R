# One timed R symphony build + map + knnPredict on the vignette data, after one untimed warm-up run in the same
# process (as the Python side). Usage: Rscript r_symphony_timing.R <symphony_ref dir> <out.json>
if (nzchar(Sys.getenv("ARSC_RLIB"))) .libPaths(c(Sys.getenv("ARSC_RLIB"), .libPaths()))
suppressMessages({library(symphony); library(Matrix)})
args <- commandArgs(TRUE)
d <- args[1]
X <- Matrix::readMM(file.path(d, "pbmcs_exprs.mtx")); rownames(X) <- readLines(file.path(d, "pbmcs_genes.txt"))
X <- as(X, "CsparseMatrix")
meta <- read.csv(file.path(d, "pbmcs_meta.csv"), row.names = 1)
colnames(X) <- rownames(meta)
isref <- meta$donor %in% c("3pv1", "3pv2")
run <- function() {
  uw <- tempfile("uwot")
  ref <- buildReference(X[, isref], meta[isref, ], vars = c("donor"), K = 100, verbose = FALSE, do_umap = TRUE,
                        do_normalize = FALSE, vargenes_method = "vst", topn = 2000, d = 20, theta = 2,
                        save_uwot_path = uw, seed = 111)
  q <- mapQuery(X[, !isref], meta[!isref, ], ref, vars = "donor", do_normalize = FALSE, do_umap = TRUE, verbose = FALSE)
  q <- knnPredict(q, ref, ref$meta_data$cell_type, k = 5)
  invisible(q)
}
t0 <- Sys.time(); suppressWarnings(run()); warm <- as.numeric(Sys.time() - t0, units = "secs")
t0 <- Sys.time(); suppressWarnings(run()); s <- as.numeric(Sys.time() - t0, units = "secs")
writeLines(sprintf('{"ok": true, "seconds": %.4f, "warmup_seconds": %.4f}', s, warm), args[2])
