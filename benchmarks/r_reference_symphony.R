# R reference outputs for arsc.symphony (SYMPHONY_PARITY.md, tests/test_symphony.py, benchmarks/symphony_parity.py).
# Needs R symphony 0.1.1 (+ harmony 1.2, uwot). Usage: Rscript benchmarks/r_reference_symphony.R [out_dir]
# (default $ARSC_SCRATCH/symphony_ref). Writes symphony internals on random inputs, and the package's vignette
# case (pbmcs_exprs_small: reference = donors 3pv1 + 3pv2, query = 5p).
suppressMessages({library(symphony); library(Matrix)})
args <- commandArgs(TRUE)
out <- if (length(args)) args[1] else file.path(Sys.getenv("ARSC_SCRATCH"), "symphony_ref")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
w <- function(x, f) write.table(as.matrix(x), file.path(out, f), sep = ",", row.names = FALSE, col.names = FALSE)
ns <- asNamespace("symphony")

# --- 1. internals on random inputs
set.seed(1)
d <- 6; K <- 4; N <- 50; Nq <- 30
Y <- matrix(rnorm(d * K), d, K); Y <- sweep(Y, 2, sqrt(colSums(Y^2)), "/")
Zq <- matrix(rnorm(d * Nq), d, Nq)
Zq_cos <- ns$cosine_normalize_cpp(Zq, 2)
w(Zq, "u_Zq.csv"); w(Y, "u_Y.csv"); w(Zq_cos, "u_Zq_cos.csv")
w(ns$soft_cluster(Y, Zq_cos, 0.1), "u_softcluster.csv")
Rr <- matrix(runif(K * N), K, N); Rr <- sweep(Rr, 2, colSums(Rr), "/")
Zr <- matrix(rnorm(d * N), d, N)
w(Rr, "u_Rr.csv"); w(Zr, "u_Zr.csv")
cache <- ns$compute_ref_cache(Rr, Zr)
w(cache[[1]], "u_Nr.csv"); w(cache[[2]], "u_C.csv")
w(t(ns$cosine_normalize_cpp(Rr %*% t(Zr), 1)), "u_centroids.csv")
Rq <- ns$soft_cluster(Y, Zq_cos, 0.1)
b <- factor(sample(c("a", "b", "c"), Nq, TRUE))
Xq <- t(cbind(1, intercept = model.matrix(~0 + b)))
w(Xq, "u_Xq.csv")
w(ns$moe_correct_ref(Zq, as.matrix(Xq), Rq, as.matrix(cache[[1]]), as.matrix(cache[[2]])), "u_moe.csv")
A <- Matrix(abs(matrix(rnorm(8 * 20), 8, 20)) * (runif(160) > 0.5), sparse = TRUE)
mu <- Matrix::rowMeans(A); sdv <- ns$rowSDs(A, mu)
w(as.matrix(A), "u_A.csv"); w(sdv, "u_sd.csv")
w(as.matrix(ns$scaleDataWithStats(A, mu, sdv, 1, thresh = 1.5)), "u_scaled_thresh1.5.csv")

# --- 2. vignette case
data(pbmcs_exprs_small); data(pbmcs_meta_small)
ref_idx <- pbmcs_meta_small$donor %in% c("3pv1", "3pv2")
ref_exp <- pbmcs_exprs_small[, ref_idx]; ref_meta <- pbmcs_meta_small[ref_idx, ]
q_exp <- pbmcs_exprs_small[, !ref_idx]; q_meta <- pbmcs_meta_small[!ref_idx, ]
Matrix::writeMM(pbmcs_exprs_small, file.path(out, "pbmcs_exprs.mtx"))
writeLines(rownames(pbmcs_exprs_small), file.path(out, "pbmcs_genes.txt"))
write.csv(pbmcs_meta_small, file.path(out, "pbmcs_meta.csv"))
uwot_path <- file.path(out, "ref_uwot_model")
t0 <- Sys.time()
ref <- buildReference(ref_exp, ref_meta, vars = c("donor"), K = 100, verbose = FALSE, do_umap = TRUE,
                      do_normalize = FALSE, vargenes_method = "vst", topn = 2000, d = 20, theta = 2,
                      save_uwot_path = uwot_path, seed = 111)
q <- mapQuery(q_exp, q_meta, ref, vars = "donor", do_normalize = FALSE, do_umap = TRUE, verbose = FALSE)
q <- knnPredict(q, ref, ref$meta_data$cell_type, k = 5)
cat("R build+map seconds:", as.numeric(Sys.time() - t0, units = "secs"), "\n")
writeLines(ref$vargenes$symbol, file.path(out, "ref_vargenes.txt"))
w(cbind(ref$vargenes$mean, ref$vargenes$stddev), "ref_meansd.csv")
w(ref$loadings, "ref_loadings.csv"); w(ref$Z_orig, "ref_Zorig.csv"); w(ref$Z_corr, "ref_Zcorr.csv")
w(ref$R, "ref_R.csv"); w(ref$centroids, "ref_centroids.csv"); w(ref$cache[[1]], "ref_Nr.csv")
w(ref$cache[[2]], "ref_C.csv"); w(ref$umap$embedding, "ref_umap.csv")
w(q$Zq_pca, "q_Zpca.csv"); w(q$R, "q_R.csv"); w(q$Z, "q_Z.csv"); w(q$umap, "q_umap.csv")
write.csv(data.frame(cell = rownames(q$meta_data), pred = q$meta_data$cell_type_pred_knn,
                     prob = q$meta_data$cell_type_pred_knn_prob), file.path(out, "q_knn.csv"), row.names = FALSE)
w(calcPerCellMappingMetric(ref, q), "q_maha.csv")

# --- 3. soft k-means and the per-cell mapping metric on random inputs
set.seed(2)
Z <- ns$cosine_normalize_cpp(matrix(rnorm(5 * 40), 5, 40), 2)
Y <- matrix(rnorm(5 * 3), 5, 3)
res <- ns$soft_kmeans_cpp(Y, Z, 4, 0.1)
w(Z, "sk_Z.csv"); w(Y, "sk_Y0.csv"); w(res$Y, "sk_Y.csv"); w(res$R, "sk_R.csv")
res1 <- ns$soft_kmeans_cpp(Y, Z, 1, 0.1)
w(res1$Y, "sk1_Y.csv"); w(res1$R, "sk1_R.csv")
# mapping metric on random small inputs
ref <- list(Z_orig = matrix(rnorm(4 * 60), 4, 60), Z_corr = matrix(rnorm(4 * 60), 4, 60))
Rr <- matrix(runif(3 * 60), 3, 60); ref$R <- sweep(Rr, 2, colSums(Rr), "/")
ref$centroids <- matrix(rnorm(4 * 3), 4, 3)
q <- list(Zq_pca = matrix(rnorm(4 * 10), 4, 10), exp = matrix(0, 2, 10))
Rq <- matrix(runif(3 * 10), 3, 10); q$R <- sweep(Rq, 2, colSums(Rq), "/")
w(ref$Z_orig, "mm_Zorig.csv"); w(ref$R, "mm_R.csv"); w(q$Zq_pca, "mm_Zq.csv"); w(q$R, "mm_Rq.csv")
w(calcPerCellMappingMetric(ref, q), "mm_maha.csv")

# --- 4. scaling: symmetric clipping, zero-s.d. rows
A <- Matrix::Matrix(rbind(c(0, 0, 0, 0, 10), c(5, 5, 5, 5, 0), c(1, 2, 3, 0, 0)), sparse = TRUE)
mu <- Matrix::rowMeans(A); s <- ns$rowSDs(A, mu)
w(as.matrix(A), "clip_A.csv")
w(as.matrix(ns$scaleDataWithStats(A, mu, s, 1, thresh = 1)), "clip_S.csv")
# zero sd row
B <- Matrix::Matrix(rbind(c(1, 1, 1), c(0, 2, 0)), sparse = TRUE)
w(as.matrix(ns$scaleDataWithStats(B, Matrix::rowMeans(B), ns$rowSDs(B, Matrix::rowMeans(B)), 1)), "clip_B.csv")
