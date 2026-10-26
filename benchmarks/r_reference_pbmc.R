ar_r <- "E:/autoresearch/tools/R/ar.R"; if (file.exists(ar_r)) source(ar_r)  # autoresearch toolkit, when vendored there
#!/usr/bin/env Rscript
# Scale reference on a PBMC CITE-seq WNN object (259,899 cells, RNA + 49 ADT): timings for the
# R ecosystem plus WNN outputs (stored weighted.nn from the original analysis
# and a fresh FindMultiModalNeighbors rerun = R-vs-R Annoy noise floor).
#
# Usage: Rscript benchmarks/r_reference_pbmc.R [qs] [out_dir]
suppressPackageStartupMessages({
  if (nzchar(Sys.getenv("ARSC_RLIB"))) .libPaths(c(Sys.getenv("ARSC_RLIB"), .libPaths()))
  library(Matrix); library(Seurat); library(harmony); library(presto)
  library(UCell); library(data.table); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args) >= 1) args[1] else stop("usage: Rscript r_reference_pbmc.R <object.qs> [out_dir]")
out <- if (length(args) >= 2) args[2] else file.path(Sys.getenv("ARSC_REFS", Sys.getenv("ARSC_SCRATCH", "scratch")), "ref_pbmc")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(future.globals.maxSize = 64 * 1024^3)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "r_common.R"))

tm("load_qs", obj <- qs::qread(src, nthreads = 16))
cells <- colnames(obj)

# ---- stored WNN neighbors (computed by the original analysis, Seurat 4.0.4)
wn <- obj@neighbors$weighted.nn
write_dense(wn@nn.idx, file.path(out, "stored_weighted_nn_idx"))
write_dense(wn@nn.dist, file.path(out, "stored_weighted_nn_dist"))

# ---- fresh FindMultiModalNeighbors with the stored parameters ------------
set.seed(1)
tm("FindMultiModalNeighbors", obj2 <- FindMultiModalNeighbors(
  obj, reduction.list = list("harmony_rna", "harmony_adt"), dims.list = list(1:30, 1:15),
  modality.weight.name = c("RNA.weight.rerun", "ADT.weight.rerun"),
  knn.graph.name = "wknn_rerun", snn.graph.name = "wsnn_rerun", weighted.nn.name = "wnn_rerun", verbose = FALSE))
write_dense(obj2@neighbors$wnn_rerun@nn.idx, file.path(out, "rerun_weighted_nn_idx"))
fwrite(data.table(cell = cells, RNA.weight.rerun = obj2$RNA.weight.rerun, RNA.weight.stored = obj$RNA.weight),
       file.path(out, "wnn_weights.csv"))
g <- as(obj2@graphs$wsnn_rerun, "dgCMatrix")
writeBin(g@i, file.path(out, "wsnn_rerun.i.bin"), size = 4)
writeBin(g@p, file.path(out, "wsnn_rerun.p.bin"), size = 4)
writeBin(g@x, file.path(out, "wsnn_rerun.x.bin"), size = 8)
rm(obj2, g); gc()

# ---- standard pipeline timings at 260k ------------------------------------
DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = c("RNA", "ADT"), counts = TRUE, data = TRUE,
                  dimreducs = c("pca", "harmony_rna", "harmony_adt"))
tm("NormalizeData", obj <- NormalizeData(obj, verbose = FALSE))
tm("FindVariableFeatures", obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE))
writeLines(VariableFeatures(obj), file.path(out, "variable_features.txt"))
tm("ScaleData", obj <- ScaleData(obj, verbose = FALSE))
tm("RunPCA", obj <- RunPCA(obj, npcs = 30, verbose = FALSE, reduction.name = "pca_new"))
write_dense(Embeddings(obj, "pca_new"), file.path(out, "pca_embeddings"))

pcs <- Embeddings(obj, "pca")  # stored PCA, so harmony input is identical for arsc
set.seed(1)
tm("RunHarmony", hz <- RunHarmony(pcs, obj@meta.data, "n10x_lane_name", verbose = FALSE))
write_dense(hz, file.path(out, "harmony_default"))

X <- GetAssayData(obj, slot = "data")
tm("presto_wilcoxauc", mk <- wilcoxauc(X, obj$wsnn_res.0.2))
fwrite(mk, file.path(out, "presto_wsnn_res02.csv"))
rm(mk, X); gc()

sigs <- jsonlite::read_json(file.path(dirname(out), "ref_gd", "signatures.json"), simplifyVector = TRUE)
C <- GetAssayData(obj, slot = "counts")
tm("UCell", uc <- ScoreSignatures_UCell(C, features = sigs, ncores = 1))
fwrite(data.table(cell = rownames(uc), uc), file.path(out, "ucell.csv"))
rm(C, uc); gc()
sigs_pos <- sigs[c("cytotoxic", "naive", "exhaustion", "isg", "gd")]
tm("AddModuleScore", obj <- AddModuleScore(obj, features = sigs_pos, name = "AMS", seed = 1))
ams <- obj@meta.data[, paste0("AMS", seq_along(sigs_pos))]
fwrite(data.table(cell = rownames(ams), ams), file.path(out, "addmodulescore.csv"))
tm("AggregateExpression", pb <- AggregateExpression(obj, assays = "RNA", slot = "counts",
                                                    group.by = c("sample", "azimuthl1"), verbose = FALSE)$RNA)
tm("FindNeighbors", obj <- FindNeighbors(obj, reduction = "harmony_rna", dims = 1:30, verbose = FALSE))
tm("FindClusters_louvain", obj <- FindClusters(obj, resolution = 0.8, verbose = FALSE))
write_timings(out)
cat("done\n")
