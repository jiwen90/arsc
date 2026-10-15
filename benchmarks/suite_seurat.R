# One benchmark step for the R/Seurat tool, run in a fresh Rscript process by run_suite.py.
#   Rscript suite_seurat.R <step> <data_dir> <result.json>
# Library path: env ARSC_RLIB (optional, prepended to .libPaths()).
# Only the analysis call is timed; reading the counts and building the Seurat object are not.
# Outputs for the agreement check go to <data_dir>/out/seurat/<step>.*
args <- commandArgs(trailingOnly = TRUE)
step <- args[1]; data <- args[2]; res_file <- args[3]
rlib <- Sys.getenv("ARSC_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages({ library(Matrix); library(jsonlite) })
N_HVG <- 2000; N_PC <- 30; K_NN <- 20; RES <- 0.8
options(future.globals.maxSize = 64 * 1024^3)   # Seurat parallel helpers refuse > 500 MB globals by default
od <- file.path(data, "out", "seurat"); dir.create(od, recursive = TRUE, showWarnings = FALSE)
common <- file.path(data, "common")
truth <- fromJSON(file.path(data, "truth.json"))
t0 <- NA_real_; t1 <- NA_real_
tic <- function() t0 <<- as.numeric(Sys.time())
toc <- function() t1 <<- as.numeric(Sys.time())

read_counts <- function() {
  rd <- file.path(data, "r")
  genes <- readLines(file.path(rd, "genes.txt")); cells <- readLines(file.path(rd, "cells.txt"))
  p <- readBin(file.path(rd, "p.f64"), "double", n = length(cells) + 1, size = 8)
  nnz <- p[length(p)]
  if (nnz >= 2^31) stop(sprintf("nnz %.0f exceeds the dgCMatrix limit (2^31 - 1)", nnz))
  x <- readBin(file.path(rd, "x.f32"), "double", n = nnz, size = 4)
  i <- readBin(file.path(rd, "i.i32"), "integer", n = nnz, size = 4)
  m <- new("dgCMatrix", i = i, p = as.integer(p), x = x, Dim = c(length(genes), length(cells)),
           Dimnames = list(genes, cells))
  obs <- read.csv(file.path(rd, "obs.csv"), row.names = 1, check.names = FALSE)
  list(m = m, obs = obs)
}
seurat_obj <- function(d) {
  suppressPackageStartupMessages(library(Seurat))
  CreateSeuratObject(d$m, meta.data = d$obs, min.cells = 0, min.features = 0)
}
read_pb <- function() {
  cnt <- as.matrix(read.csv(file.path(common, "pb_T0_counts.csv"), row.names = 1, check.names = FALSE))
  cd <- read.csv(file.path(common, "pb_T0_coldata.csv"), row.names = 1, check.names = FALSE)
  list(cnt = round(cnt), cd = cd[colnames(cnt), , drop = FALSE])
}
read_npy_f32 <- function(f, shape_json) {   # raw float32, column-major (written by run_suite.py)
  sh <- fromJSON(shape_json)
  matrix(readBin(f, "double", n = prod(sh), size = 4), nrow = sh[1], ncol = sh[2])
}
write_mat <- function(m, prefix) {   # dense float64 + names (row-major as written by Python is not needed)
  m <- as.matrix(m)
  writeBin(as.vector(m), paste0(prefix, ".f64"), size = 8)
  write_json(list(dim = dim(m), rownames = rownames(m), colnames = colnames(m)), paste0(prefix, ".json"), digits = NA)
}

res <- list(tool = "seurat", step = step, ok = FALSE)
out <- tryCatch({
  if (step == "normalize") {
    o <- seurat_obj(read_counts())
    tic(); o <- NormalizeData(o, verbose = FALSE); toc()
    L <- GetAssayData(o, slot = "data")[, 1:500]
    Matrix::writeMM(L, file.path(od, "normalize.mtx"))
  } else if (step == "hvg") {
    o <- seurat_obj(read_counts())
    tic(); o <- FindVariableFeatures(o, selection.method = "vst", nfeatures = N_HVG, verbose = FALSE); toc()
    hf <- HVFInfo(o)
    write.csv(data.frame(gene = rownames(hf), variance_standardized = hf$variance.standardized,
                         highly_variable = rownames(hf) %in% VariableFeatures(o)), file.path(od, "hvg.csv"), row.names = FALSE)
  } else if (step == "pca") {
    o <- seurat_obj(read_counts()); o <- NormalizeData(o, verbose = FALSE)
    hv <- readLines(file.path(common, "hvg.txt"))
    tic()
    o <- ScaleData(o, features = hv, verbose = FALSE)
    o <- RunPCA(o, features = hv, npcs = N_PC, verbose = FALSE)
    toc()
    write_mat(Embeddings(o, "pca"), file.path(od, "pca"))
  } else if (step %in% c("neighbors", "cluster")) {
    suppressPackageStartupMessages(library(Seurat))
    E <- read_npy_f32(file.path(common, "pca.f32"), file.path(common, "pca_shape.json"))
    rownames(E) <- readLines(file.path(data, "r", "cells.txt")); colnames(E) <- paste0("PC_", 1:ncol(E))
    o <- CreateSeuratObject(Matrix::sparseMatrix(i = 1, j = 1, x = 1, dims = c(2, nrow(E)),
                                                 dimnames = list(c("g1", "g2"), rownames(E))))
    o[["pca"]] <- CreateDimReducObject(embeddings = E, key = "PC_", assay = "RNA")
    tic()
    o <- FindNeighbors(o, reduction = "pca", dims = 1:N_PC, k.param = K_NN, verbose = FALSE)
    if (step == "cluster") o <- FindClusters(o, resolution = RES, verbose = FALSE)
    toc()
    if (step == "neighbors") {
      nn <- as(o@graphs$RNA_nn, "dgCMatrix")   # binary kNN incl. self, rows = cells
      tt <- as(Matrix::t(nn), "dgCMatrix")
      idx <- matrix(tt@i, nrow = K_NN)          # columns of t(nn) = rows of nn; K_NN entries each
      writeBin(as.integer(as.vector(idx)), file.path(od, "neighbors.i32"), size = 4)
    } else {
      writeLines(as.character(o$seurat_clusters), file.path(od, "cluster.txt"))
    }
  } else if (step == "markers") {
    d <- read_counts(); o <- seurat_obj(d); o <- NormalizeData(o, verbose = FALSE)
    L <- GetAssayData(o, slot = "data"); g <- as.character(d$obs$cell_type)
    tic(); r <- presto::wilcoxauc(L, g); toc()
    write.csv(r[, c("feature", "group", "auc", "pval", "statistic")], file.path(od, "markers.csv"), row.names = FALSE)
  } else if (step == "pseudobulk") {
    o <- seurat_obj(read_counts())
    tic(); a <- AggregateExpression(o, group.by = c("sample", "cell_type"), slot = "counts", assays = "RNA", verbose = FALSE)$RNA; toc()
    write_mat(a, file.path(od, "pseudobulk"))
  } else if (step == "de_deseq2") {
    suppressPackageStartupMessages(library(DESeq2))
    pb <- read_pb(); cd <- pb$cd; cd$patient <- factor(cd$patient); cd$timepoint <- factor(cd$timepoint, c("pre", "post"))
    tic()
    dds <- DESeqDataSetFromMatrix(pb$cnt, cd, ~ patient + timepoint)
    dds <- DESeq(dds, quiet = TRUE)
    r <- results(dds, contrast = c("timepoint", "post", "pre"), alpha = 0.1)
    toc()
    write.csv(data.frame(gene = rownames(r), log2FoldChange = r$log2FoldChange, pvalue = r$pvalue, padj = r$padj),
              file.path(od, "de_deseq2.csv"), row.names = FALSE)
  } else if (step == "de_voom") {
    suppressPackageStartupMessages({ library(limma); library(edgeR) })
    pb <- read_pb(); cd <- pb$cd; cd$timepoint <- factor(cd$timepoint, c("pre", "post"))
    tic()
    design <- model.matrix(~ timepoint, cd)
    y <- calcNormFactors(DGEList(pb$cnt))
    v <- voom(y, design)
    dc <- duplicateCorrelation(v, design, block = cd$patient)
    v <- voom(y, design, block = cd$patient, correlation = dc$consensus)
    dc <- duplicateCorrelation(v, design, block = cd$patient)
    fit <- eBayes(lmFit(v, design, block = cd$patient, correlation = dc$consensus))
    tt <- topTable(fit, coef = "timepointpost", number = Inf, sort.by = "none")
    toc()
    write.csv(data.frame(gene = rownames(tt), logFC = tt$logFC, t = tt$t, P.Value = tt$P.Value, adj.P.Val = tt$adj.P.Val),
              file.path(od, "de_voom.csv"), row.names = FALSE)
  } else if (step == "score_module") {
    o <- seurat_obj(read_counts()); o <- NormalizeData(o, verbose = FALSE)
    sigs <- truth$signatures
    tic(); o <- AddModuleScore(o, features = unname(sigs), name = "sig", seed = 1); toc()
    s <- o@meta.data[, paste0("sig", seq_along(sigs))]; colnames(s) <- names(sigs)
    write_mat(as.matrix(s), file.path(od, "score_module"))
  } else if (step == "score_ucell") {
    suppressPackageStartupMessages(library(UCell))
    d <- read_counts()
    tic(); s <- ScoreSignatures_UCell(d$m, features = truth$signatures); toc()
    write_mat(as.matrix(s), file.path(od, "score_ucell"))
  } else {
    stop(paste("no Seurat equivalent for", step), call. = FALSE)
  }
  res$ok <- TRUE
  NULL
}, error = function(e) {
  msg <- conditionMessage(e)
  if (grepl("^no Seurat equivalent", msg)) res$skipped <<- msg else res$error <<- msg
  NULL
})
res$t_start <- t0; res$t_end <- t1; res$seconds <- t1 - t0
write_json(res, res_file, auto_unbox = TRUE, digits = NA, na = "null")
