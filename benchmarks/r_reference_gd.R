ar_r <- "E:/autoresearch/tools/R/ar.R"; if (file.exists(ar_r)) source(ar_r)  # autoresearch toolkit, when vendored there
#!/usr/bin/env Rscript
# Reference outputs + timings from the R ecosystem (Seurat 4.3, harmony 1.2,
# presto 1.0, UCell 2.6) on the reference gamma-delta T-cell object (38,440 cells). arsc's validation scripts
# compare against the files written here.
#
# Usage: Rscript benchmarks/r_reference_gd.R [rds] [out_dir]
# By default it writes only what tests/test_vs_r.py reads (HVG, a LogNormalize block, presto markers for the 2,000 variable genes; gzip, 3 MB).
# ARSC_REF_FULL=1 also writes PCA, harmony, UCell, AddModuleScore, pseudobulk, CLR and SNN outputs and timings for
# benchmarks/validation/ (about 1 GB). benchmarks/regen_refs.py ref_gd runs it on the shared input's Seurat object.
suppressPackageStartupMessages({
  if (nzchar(Sys.getenv("ARSC_RLIB"))) .libPaths(c(Sys.getenv("ARSC_RLIB"), .libPaths()))
  library(Matrix); library(Seurat); library(harmony); library(presto)
  library(UCell); library(data.table); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
rds <- if (length(args) >= 1) args[1] else stop("usage: Rscript r_reference_gd.R <object.rds> [out_dir]")
out <- if (length(args) >= 2) args[2] else file.path(Sys.getenv("ARSC_REFS", Sys.getenv("ARSC_SCRATCH", "scratch")), "ref_gd")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(future.globals.maxSize = 64 * 1024^3)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "r_common.R"))
full <- identical(Sys.getenv("ARSC_REF_FULL"), "1")

obj <- readRDS(rds)
DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = c("RNA", "TRIMMEDCITE"), counts = TRUE, data = TRUE)
cat("cells:", ncol(obj), "\n")

# ---- 1. LogNormalize / HVG / scale / PCA --------------------------------------
tm("NormalizeData", obj <- NormalizeData(obj, verbose = FALSE))
tm("FindVariableFeatures", obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE))
hvf <- obj[["RNA"]]@meta.features
fwrite(data.table(feature = rownames(hvf), hvf[, grep("^vst", names(hvf))]), file.path(out, "hvf.csv.gz"))
writeLines(VariableFeatures(obj), file.path(out, "variable_features.txt"))
# Normalized data for one cell block (checks LogNormalize numerics directly)
write_dense_gz(as.matrix(GetAssayData(obj, slot = "data")[VariableFeatures(obj)[1:200], 1:2000]),
               file.path(out, "lognorm_block"))
if (full) {
tm("ScaleData", obj <- ScaleData(obj, verbose = FALSE))
tm("RunPCA", obj <- RunPCA(obj, npcs = 30, verbose = FALSE))
write_dense(Embeddings(obj, "pca"), file.path(out, "pca_embeddings"))
write_dense(Loadings(obj, "pca"), file.path(out, "pca_loadings"))
writeLines(rownames(Loadings(obj, "pca")), file.path(out, "pca_loadings_features.txt"))
writeLines(as.character(Stdev(obj, "pca")), file.path(out, "pca_stdev.txt"))

# ---- 2. Harmony -------------------------------------------------------------
pcs <- Embeddings(obj, "pca")
meta <- obj@meta.data
set.seed(1)
tm("RunHarmony", hz <- RunHarmony(pcs, meta, "n10x_lane_name", verbose = FALSE))
write_dense(hz, file.path(out, "harmony_default"))
set.seed(1)
tm("RunHarmony_2vars", hz2 <- RunHarmony(pcs, meta, c("patient_alias", "n10x_lane_name"), verbose = FALSE))
write_dense(hz2, file.path(out, "harmony_2vars"))

# Step-by-step run exposing the k-means init so arsc can start from the same
# centroids. Mirrors RunHarmony.default exactly (setup -> init -> harmonize).
harmony_obj_run <- function(Z, meta, vars, block.size, seed = 1) {
  set.seed(seed)
  N <- nrow(meta); nclust <- min(round(N / 30), 100)
  theta <- rep(2, length(vars)); sigma <- rep(0.1, nclust)
  phi <- Reduce(rbind, lapply(vars, function(v) Matrix::t(Matrix::sparse.model.matrix(~0 + as.factor(meta[[v]])))))
  B_vec <- sapply(vars, function(v) nlevels(as.factor(meta[[v]])))
  lambda_vec <- c(0, rep(1, sum(B_vec)))
  theta <- unlist(lapply(seq_along(B_vec), function(b) rep(theta[b], B_vec[b])))
  h <- new(get("harmony", envir = asNamespace("harmony")))
  h$setup(t(Z), phi, sigma, theta, lambda_vec, 0.2, 20L, 1e-3, 1e-2, nclust, block.size, B_vec, FALSE)
  h$init_cluster_cpp()
  Y0 <- h$Y; R0 <- h$R; obj0 <- h$objective_harmony
  harmony:::harmonize(h, 10, FALSE)
  list(Y0 = Y0, R0 = R0, obj0 = obj0, Z = t(h$Z_corr), objective_harmony = h$objective_harmony,
       objective_kmeans = h$objective_kmeans, kmeans_rounds = h$kmeans_rounds, R = h$R)
}
for (bs in c(1, 0.05)) {
  tag <- if (bs == 1) "bs1" else "bs005"
  r <- harmony_obj_run(pcs, meta, "n10x_lane_name", bs)
  write_dense(r$Y0, file.path(out, paste0("harmony_", tag, "_Y0")))
  write_dense(r$Z, file.path(out, paste0("harmony_", tag, "_Z")))
  write_dense(r$R, file.path(out, paste0("harmony_", tag, "_R")))
  write_json(list(objective_harmony = r$objective_harmony, objective_kmeans = r$objective_kmeans,
                  kmeans_rounds = r$kmeans_rounds, obj0 = r$obj0),
             file.path(out, paste0("harmony_", tag, "_trace.json")), digits = NA)
}

}  # full

# ---- 3. presto markers ------------------------------------------------------
X <- GetAssayData(obj, slot = "data")
tm("presto_wilcoxauc", mk <- wilcoxauc(X, obj$cell_type))
# the tests compare the 2,000 variable genes (presto is per gene, so a subset is a full check of those rows; all genes: 30 MB)
fwrite(if (full) mk else mk[mk$feature %in% VariableFeatures(obj), ], file.path(out, "presto_cell_type.csv.gz"))
if (!full) {
  write_timings(out)
  cat("done (tests subset; ARSC_REF_FULL=1 for everything)
")
  quit(save = "no")
}

# ---- 4. Module scores -------------------------------------------------------
sigs <- list(
  cytotoxic = c("GZMB", "PRF1", "GNLY", "NKG7", "GZMH", "GZMA"),
  naive = c("CCR7", "SELL", "TCF7", "LEF1", "IL7R"),
  exhaustion = c("PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4"),
  isg = c("ISG15", "IFI6", "MX1", "IFIT3", "IFIT1", "OAS1"),
  gd = c("TRDC", "TRGC1", "TRGC2", "TRDV2", "TRGV9"),
  naive_vs_eff = c("CCR7+", "SELL+", "GZMB-", "PRF1-"),
  with_missing = c("CD3E", "CD3D", "NOT_A_GENE_1"))
write_json(sigs, file.path(out, "signatures.json"))
C <- GetAssayData(obj, slot = "counts")
tm("UCell", uc <- ScoreSignatures_UCell(C, features = sigs, ncores = 1))
fwrite(data.table(cell = rownames(uc), uc), file.path(out, "ucell.csv"))
sigs_pos <- sigs[c("cytotoxic", "naive", "exhaustion", "isg", "gd")]
tm("AddModuleScore", obj <- AddModuleScore(obj, features = sigs_pos, name = "AMS", seed = 1))
ams <- obj@meta.data[, paste0("AMS", seq_along(sigs_pos))]
fwrite(data.table(cell = rownames(ams), ams), file.path(out, "addmodulescore.csv"))

# ---- 5. Pseudobulk ----------------------------------------------------------
tm("AggregateExpression", pb <- AggregateExpression(obj, assays = "RNA", slot = "counts",
                                                    group.by = c("sample", "cell_type"), verbose = FALSE)$RNA)
pbs <- as(pb, "dgCMatrix")
writeBin(pbs@i, file.path(out, "pseudobulk.i.bin"), size = 4)
writeBin(pbs@p, file.path(out, "pseudobulk.p.bin"), size = 4)
writeBin(pbs@x, file.path(out, "pseudobulk.x.bin"), size = 8)
writeLines(colnames(pb), file.path(out, "pseudobulk_groups.txt"))
writeLines(rownames(pb), file.path(out, "pseudobulk_features.txt"))
writeLines(as.character(dim(pb)), file.path(out, "pseudobulk_dim.txt"))

# ---- 6. ADT CLR (margin = 2, per cell) -------------------------------------
tm("NormalizeData_CLR", obj <- NormalizeData(obj, assay = "TRIMMEDCITE", normalization.method = "CLR",
                                             margin = 2, verbose = FALSE))
write_dense(as.matrix(GetAssayData(obj, assay = "TRIMMEDCITE", slot = "data")), file.path(out, "adt_clr"))

# ---- 7. Neighbors / clusters / UMAP (timing + SNN graph) --------------------
tm("FindNeighbors", obj <- FindNeighbors(obj, dims = 1:30, verbose = FALSE))
snn <- as(obj@graphs$RNA_snn, "dgCMatrix")
writeBin(snn@i, file.path(out, "snn.i.bin"), size = 4)
writeBin(snn@p, file.path(out, "snn.p.bin"), size = 4)
writeBin(snn@x, file.path(out, "snn.x.bin"), size = 8)
tm("FindClusters_louvain", obj <- FindClusters(obj, resolution = 0.8, verbose = FALSE))
writeLines(as.character(obj$seurat_clusters), file.path(out, "louvain_clusters.txt"))
tm("RunUMAP", obj <- RunUMAP(obj, dims = 1:30, verbose = FALSE))

write_timings(out)
cat("done\n")
