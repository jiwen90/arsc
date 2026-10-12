#!/usr/bin/env Rscript
# export_seurat.R -- dump a Seurat v4/v5 object to a fast on-disk layout that
# `arsc.read_seurat_export()` loads in seconds.
#
# Usage:
#   Rscript export_seurat.R <object.rds|object.qs> <out_dir> [--assays RNA,ADT] [--layers counts,data]
#                           [--no-graphs] [--no-reductions] [--threads 16]
#
# If an assay has none of the requested layers (e.g. an ADT assay that only
# kept `data`), the first non-empty of counts/data is exported instead and the
# manifest records which layer was written.
#
# Layout written (all binary is little-endian, raw, no headers):
#   out_dir/manifest.json             -- shapes, dtypes, file names, factor levels
#   out_dir/obs.csv                   -- cell metadata (row order == matrix columns)
#   out_dir/cells.txt                 -- cell barcodes, one per line
#   out_dir/assays/<A>/features.txt   -- feature names
#   out_dir/assays/<A>/<layer>.{i,p,x}.bin  -- CSC (genes x cells) == CSR (cells x genes)
#   out_dir/reductions/<R>.f64        -- cells x k column-major float64 embedding
#   out_dir/reductions/<R>.loadings.f64 (if present) -- features x k
#   out_dir/graphs/<G>.{i,p,x}.bin    -- cells x cells sparse graph (CSC)
#
# Why raw binary: R has no arrow here and writeMM/text is 50-100x slower. A
# dgCMatrix's @i/@p/@x slots are written verbatim with writeBin, so export is
# bounded by disk speed, and Python maps them straight into scipy.sparse.
#
# Seurat v5 (Assay5) split layers (counts.1, counts.2, ...) are joined back
# into one matrix in cell order.

suppressPackageStartupMessages({
  .libPaths(c("E:/autoresearch/.Rlib", .libPaths()))
  library(Matrix)
  library(SeuratObject)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: export_seurat.R <object.rds> <out_dir> [--assays A,B] [--layers counts,data] [--no-graphs] [--no-reductions]")
rds <- args[1]; out <- args[2]
opt <- function(flag, default = NULL) {
  k <- which(args == flag)
  if (length(k) && k < length(args)) args[k + 1] else default
}
want_assays <- opt("--assays")
want_layers <- strsplit(opt("--layers", "counts"), ",")[[1]]
do_graphs <- !("--no-graphs" %in% args)
do_reds <- !("--no-reductions" %in% args)
n_threads <- as.integer(opt("--threads", "16"))

t0 <- Sys.time()
msg <- function(...) cat(sprintf("[%6.1fs] ", as.numeric(Sys.time() - t0, units = "secs")), ..., "\n", sep = "")

obj <- if (grepl("[.]qs$", rds, ignore.case = TRUE)) qs::qread(rds, nthreads = n_threads) else readRDS(rds)
msg("loaded ", rds, " : ", ncol(obj), " cells")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cells <- colnames(obj)
writeLines(cells, file.path(out, "cells.txt"))

write_sparse <- function(m, prefix) {
  if (!inherits(m, "dgCMatrix")) m <- as(as(m, "CsparseMatrix"), "generalMatrix")
  if (!inherits(m, "dgCMatrix")) m <- as(m, "dgCMatrix")
  x <- m@x
  is_int <- length(x) == 0 || (all(x == round(x)) && max(abs(x)) < 2^31)
  writeBin(m@i, paste0(prefix, ".i.bin"), size = 4)
  writeBin(m@p, paste0(prefix, ".p.bin"), size = 4)
  if (is_int) writeBin(as.integer(x), paste0(prefix, ".x.bin"), size = 4)
  else writeBin(x, paste0(prefix, ".x.bin"), size = 8)
  list(shape = dim(m), nnz = length(x), xdtype = if (is_int) "int32" else "float64",
       prefix = basename(prefix))
}

get_layer <- function(assay_obj, layer) {
  if (inherits(assay_obj, "Assay5")) {
    lays <- SeuratObject::Layers(assay_obj, search = layer)
    lays <- lays[lays == layer | startsWith(lays, paste0(layer, "."))]
    if (!length(lays)) return(NULL)
    if (length(lays) == 1) {
      m <- SeuratObject::LayerData(assay_obj, layer = lays)
    } else {
      parts <- lapply(lays, function(l) SeuratObject::LayerData(assay_obj, layer = l))
      feats <- Reduce(union, lapply(parts, rownames))
      parts <- lapply(parts, function(p) {
        if (identical(rownames(p), feats)) return(p)
        z <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                  dims = c(length(feats), ncol(p)), dimnames = list(feats, colnames(p)))
        z[rownames(p), ] <- p
        z
      })
      m <- do.call(cbind, parts)
    }
    m
  } else {
    slot_name <- if (layer %in% c("counts", "data", "scale.data")) layer else return(NULL)
    m <- methods::slot(assay_obj, slot_name)
    if (!length(m) || nrow(m) == 0) return(NULL)
    m
  }
}

manifest <- list(source = normalizePath(rds, winslash = "/"), n_cells = length(cells),
                 seurat_version = as.character(obj@version), assays = list(),
                 default_assay = SeuratObject::DefaultAssay(obj),
                 reductions = list(), graphs = list(), obs = list())
assays <- names(obj@assays)
if (!is.null(want_assays)) assays <- intersect(assays, strsplit(want_assays, ",")[[1]])
for (a in assays) {
  ao <- obj@assays[[a]]
  adir <- file.path(out, "assays", a); dir.create(adir, recursive = TRUE, showWarnings = FALSE)
  entry <- list(class = class(ao)[1], layers = list())
  feats <- NULL
  write_layer <- function(l) {
    m <- tryCatch(get_layer(ao, l), error = function(e) { msg("  skip ", a, "/", l, ": ", conditionMessage(e)); NULL })
    if (is.null(m) || !inherits(m, c("Matrix", "matrix")) || nrow(m) == 0 || ncol(m) == 0) return(FALSE)
    if (!identical(colnames(m), cells)) m <- m[, cells, drop = FALSE]
    if (is.null(feats)) feats <<- rownames(m)
    else if (!identical(rownames(m), feats)) m <- m[feats, , drop = FALSE]
    entry$layers[[l]] <<- write_sparse(m, file.path(adir, l))
    msg("  wrote ", a, "/", l, " ", nrow(m), "x", ncol(m), " nnz=", entry$layers[[l]]$nnz)
    TRUE
  }
  for (l in want_layers) write_layer(l)
  if (!length(entry$layers)) {
    for (l in setdiff(c("counts", "data"), want_layers)) {
      msg("  ", a, ": none of [", paste(want_layers, collapse = ","), "] present; falling back to '", l, "'")
      if (write_layer(l)) break
    }
  }
  if (is.null(feats)) feats <- rownames(ao)
  writeLines(feats, file.path(adir, "features.txt"))
  vf <- tryCatch(SeuratObject::VariableFeatures(obj, assay = a), error = function(e) character(0))
  if (length(vf)) writeLines(vf, file.path(adir, "variable_features.txt"))
  manifest$assays[[a]] <- entry
}

if (do_reds && length(obj@reductions)) {
  dir.create(file.path(out, "reductions"), showWarnings = FALSE)
  for (r in names(obj@reductions)) {
    ro <- obj@reductions[[r]]
    emb <- ro@cell.embeddings
    if (!identical(rownames(emb), cells)) emb <- emb[cells, , drop = FALSE]
    writeBin(as.vector(emb), file.path(out, "reductions", paste0(r, ".f64")), size = 8)
    e <- list(shape = dim(emb), columns = colnames(emb), assay = ro@assay.used, key = ro@key)
    ld <- ro@feature.loadings
    if (length(ld) && nrow(ld) > 0) {
      writeBin(as.vector(ld), file.path(out, "reductions", paste0(r, ".loadings.f64")), size = 8)
      writeLines(rownames(ld), file.path(out, "reductions", paste0(r, ".loadings_features.txt")))
      e$loadings_shape <- dim(ld)
    }
    if (length(ro@stdev)) e$stdev <- ro@stdev
    manifest$reductions[[r]] <- e
    msg("  reduction ", r, " ", nrow(emb), "x", ncol(emb))
  }
}

if (do_graphs && length(obj@graphs)) {
  dir.create(file.path(out, "graphs"), showWarnings = FALSE)
  for (g in names(obj@graphs)) {
    gm <- obj@graphs[[g]]
    if (!identical(colnames(gm), cells)) gm <- gm[cells, cells]
    manifest$graphs[[g]] <- write_sparse(gm, file.path(out, "graphs", g))
    msg("  graph ", g)
  }
}

md <- obj@meta.data
md <- md[cells, , drop = FALSE]
for (cn in names(md)) {
  v <- md[[cn]]
  if (is.factor(v)) manifest$obs[[cn]] <- list(type = "factor", levels = levels(v))
  else if (is.logical(v)) manifest$obs[[cn]] <- list(type = "logical")
  else if (is.integer(v)) manifest$obs[[cn]] <- list(type = "integer")
  else if (is.numeric(v)) manifest$obs[[cn]] <- list(type = "double")
  else { md[[cn]] <- as.character(v); manifest$obs[[cn]] <- list(type = "character") }
}
md <- cbind(data.frame(cell = cells, stringsAsFactors = FALSE), md)
data.table::fwrite(md, file.path(out, "obs.csv"), na = "NA", quote = TRUE)
writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, digits = NA),
           file.path(out, "manifest.json"))
msg("done -> ", out)
