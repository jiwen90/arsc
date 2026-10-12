#!/usr/bin/env Rscript
# Export Seurat's own Annoy k-NN (as used inside FindMultiModalNeighbors) for
# both modalities, so arsc.wnn can be validated on identical neighbour input.
suppressPackageStartupMessages({ .libPaths(c("E:/autoresearch/.Rlib", .libPaths())); library(Seurat) })
out <- "E:/autoresearch/scratch/arsc/ref_pbmc"
options(future.globals.maxSize = 64 * 1024^3)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "r_common.R"))
obj <- qs::qread("E:/autoresearch/.cache/objects/autoresearch_data_2023_0129_V3_pbmc_wnn_RDS.qs", nthreads = 16)
for (spec in list(list("harmony_rna", 1:30), list("harmony_adt", 1:15))) {
  e <- Seurat:::L2Norm(Embeddings(obj, spec[[1]])[, spec[[2]]])
  for (k in c(20, 200)) {
    nn <- Seurat:::NNHelper(data = e, query = e, k = k, method = "annoy", metric = "euclidean")
    write_dense(Indices(nn), file.path(out, sprintf("annoy_%s_k%d_idx", spec[[1]], k)))
    write_dense(Distances(nn), file.path(out, sprintf("annoy_%s_k%d_dist", spec[[1]], k)))
    cat(spec[[1]], k, "done\n")
  }
}
