#!/usr/bin/env Rscript
# export_seurat.R -- write a Seurat v4/v5 object as an arsc store: a directory of
# memory-mappable .npy arrays that `arsc.open()` maps zero-copy (N worker
# processes share one physical copy through the OS page cache).
#
# Usage:
#   Rscript export_seurat.R <object.rds|object.qs> <out_dir> [--assays RNA,ADT]
#           [--layers counts,data] [--no-graphs] [--no-reductions] [--threads 16]
#           [--float64-counts]
#
# Layout (all arrays are NPY v1.0, little-endian):
#   manifest.json                         shapes, dtypes, factor levels, provenance
#   cells.txt, obs.csv                    cell barcodes / metadata (row order = cells)
#   assays/<A>/features.txt               feature names
#   assays/<A>/<layer>.csr.{data,indices,indptr}.npy
#                                         cells x features CSR. A dgCMatrix (features x
#                                         cells, CSC) has exactly these slots, so they are
#                                         written verbatim: @x -> data, @i -> indices,
#                                         @p -> indptr. Integer-valued layers (counts) are
#                                         stored as float32 (exact below 2^24) so Python
#                                         never has to cast (a cast would copy).
#   assays/<A>/variable_features.txt      (if set)
#   reductions/<R>.npy                    cells x k float64 (Fortran order)
#   reductions/<R>.loadings.npy + .loadings_features.txt
#   graphs/<G>.csr.{data,indices,indptr}.npy   cells x cells (stored transposed-CSC = CSR of t(G))
#
# Python's first `arsc.open()` adds obs.parquet and the per-feature CSC copies
# (<layer>.csc.*.npy) used for fast gene lookups.
#
# If an assay has none of the requested layers (e.g. an ADT assay that only
# kept `data`), the first non-empty of counts/data is exported instead and the
# manifest records which layer was written. Seurat v5 split layers
# (counts.1, counts.2, ...) are joined back into one matrix in cell order.

suppressPackageStartupMessages({
  .libPaths(c("E:/autoresearch/.Rlib", .libPaths()))
  library(Matrix)
  library(SeuratObject)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: export_seurat.R <object.rds|.qs> <out_dir> [--assays A,B] [--layers counts,data] [--no-graphs] [--no-reductions]")
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
counts_f64 <- "--float64-counts" %in% args

t0 <- Sys.time()
msg <- function(...) cat(sprintf("[%6.1fs] ", as.numeric(Sys.time() - t0, units = "secs")), ..., "\n", sep = "")

# ---------------------------------------------------------------- NPY writer
# Minimal NPY v1.0 writer; data written in chunks (writeBin is limited to
# 2^31-1 bytes per call).
write_npy <- function(x, path, type, shape = length(x), fortran = FALSE) {
  descr <- c(i4 = "<i4", f4 = "<f4", f8 = "<f8")[[type]]
  size <- c(i4 = 4L, f4 = 4L, f8 = 8L)[[type]]
  shp <- paste0(paste(format(shape, scientific = FALSE, trim = TRUE), collapse = ", "),
                if (length(shape) == 1) "," else "")
  hdr <- sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': (%s), }", descr,
                 if (fortran) "True" else "False", shp)
  pad <- (64 - ((10 + nchar(hdr) + 1) %% 64)) %% 64
  hdr <- paste0(hdr, strrep(" ", pad), "\n")
  tmp <- paste0(path, ".tmp")
  con <- file(tmp, "wb")
  writeBin(as.raw(0x93), con); writeChar("NUMPY", con, eos = NULL); writeBin(as.raw(c(1L, 0L)), con)
  writeBin(nchar(hdr), con, size = 2, endian = "little")
  writeChar(hdr, con, eos = NULL)
  n <- length(x); chunk <- 2^26; s <- 1
  while (s <= n) {
    e <- min(n, s + chunk - 1)
    v <- x[s:e]
    if (type == "i4") writeBin(as.integer(v), con, size = 4, endian = "little")
    else writeBin(as.double(v), con, size = size, endian = "little")
    s <- e + 1
  }
  close(con)
  file.rename(tmp, path)
  invisible(path)
}

obj <- if (grepl("[.]qs$", rds, ignore.case = TRUE)) qs::qread(rds, nthreads = n_threads) else readRDS(rds)
msg("loaded ", rds, " : ", ncol(obj), " cells")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cells <- colnames(obj)
writeLines(cells, file.path(out, "cells.txt"), useBytes = TRUE)

write_sparse <- function(m, prefix, allow_f32 = TRUE) {
  if (!inherits(m, "dgCMatrix")) m <- as(as(m, "CsparseMatrix"), "generalMatrix")
  if (!inherits(m, "dgCMatrix")) m <- as(m, "dgCMatrix")
  x <- m@x
  is_int <- length(x) == 0 || (all(x == round(x)) && max(abs(x)) < 2^24)
  xtype <- if (is_int && allow_f32 && !counts_f64) "f4" else "f8"
  write_npy(m@x, paste0(prefix, ".data.npy"), xtype)
  write_npy(m@i, paste0(prefix, ".indices.npy"), "i4")
  write_npy(m@p, paste0(prefix, ".indptr.npy"), "i4")
  list(shape = rev(dim(m)), nnz = length(x), dtype = c(f4 = "float32", f8 = "float64")[[xtype]],
       integer_valued = is_int, prefix = basename(prefix), format = "csr")
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
    if (!(layer %in% c("counts", "data", "scale.data"))) return(NULL)
    m <- methods::slot(assay_obj, layer)
    if (!length(m) || nrow(m) == 0) return(NULL)
    m
  }
}

manifest <- list(format = "arsc-store", format_version = 1L,
                 source = normalizePath(rds, winslash = "/"), n_cells = length(cells),
                 seurat_version = as.character(obj@version),
                 default_assay = SeuratObject::DefaultAssay(obj),
                 assays = list(), reductions = list(), graphs = list(), obs = list())
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
    if (!identical(colnames(m), cells)) {
      have <- intersect(cells, colnames(m))
      if (length(have) < length(cells)) {  # assay measured on a subset of cells (e.g. ADT absent for a cohort)
        msg("  ", a, "/", l, ": measured in ", length(have), "/", length(cells), " cells; others written as empty rows")
        entry$cells_measured <<- length(have)
        writeLines(have, file.path(adir, "cells_measured.txt"), useBytes = TRUE)
        pad <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = c(nrow(m), length(cells) - length(have)),
                                    dimnames = list(rownames(m), setdiff(cells, have)))
        m <- cbind(as(m[, have, drop = FALSE], "CsparseMatrix"), pad)
      }
      m <- m[, cells, drop = FALSE]
    }
    if (is.null(feats)) feats <<- rownames(m)
    else if (!identical(rownames(m), feats)) m <- m[feats, , drop = FALSE]
    entry$layers[[l]] <<- write_sparse(m, file.path(adir, paste0(l, ".csr")))
    msg("  wrote ", a, "/", l, " ", nrow(m), "x", ncol(m), " nnz=", entry$layers[[l]]$nnz,
        " (", entry$layers[[l]]$dtype, ")")
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
  writeLines(feats, file.path(adir, "features.txt"), useBytes = TRUE)
  vf <- tryCatch(SeuratObject::VariableFeatures(obj, assay = a), error = function(e) character(0))
  if (length(vf)) writeLines(vf, file.path(adir, "variable_features.txt"), useBytes = TRUE)
  manifest$assays[[a]] <- entry
}

if (do_reds && length(obj@reductions)) {
  dir.create(file.path(out, "reductions"), showWarnings = FALSE)
  for (r in names(obj@reductions)) {
    ro <- obj@reductions[[r]]
    emb <- ro@cell.embeddings
    if (!identical(rownames(emb), cells)) emb <- emb[cells, , drop = FALSE]
    write_npy(as.vector(emb), file.path(out, "reductions", paste0(r, ".npy")), "f8", dim(emb), fortran = TRUE)
    e <- list(shape = dim(emb), columns = colnames(emb), assay = ro@assay.used, key = ro@key)
    ld <- ro@feature.loadings
    if (length(ld) && nrow(ld) > 0) {
      write_npy(as.vector(ld), file.path(out, "reductions", paste0(r, ".loadings.npy")), "f8", dim(ld), fortran = TRUE)
      writeLines(rownames(ld), file.path(out, "reductions", paste0(r, ".loadings_features.txt")), useBytes = TRUE)
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
    # dgCMatrix slots of G are the CSR of t(G); arsc transposes on load.
    manifest$graphs[[g]] <- c(write_sparse(gm, file.path(out, "graphs", paste0(g, ".csr")), allow_f32 = FALSE),
                              list(transposed = TRUE))
    msg("  graph ", g)
  }
}

md <- obj@meta.data
md <- md[cells, , drop = FALSE]
for (cn in names(md)) {
  v <- md[[cn]]
  if (inherits(v, c("Date", "IDate", "POSIXt", "difftime"))) {  # dates are integer/double internally: export as text
    md[[cn]] <- as.character(v); manifest$obs[[cn]] <- list(type = "character")
  }
  else if (is.factor(v)) manifest$obs[[cn]] <- list(type = "factor", levels = levels(v))
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
