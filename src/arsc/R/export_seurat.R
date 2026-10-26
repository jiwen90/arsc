#!/usr/bin/env Rscript
# export_seurat.R -- write a Seurat v3/v4/v5 object as an arsc store: a directory of
# memory-mappable .npy arrays that `arsc.open()` maps zero-copy (N worker
# processes share one physical copy through the OS page cache).
#
# Usage:
#   Rscript export_seurat.R <object.rds|object.rds.gz|object.qs> <out_dir> [--full]
#           [--assays RNA,ADT] [--layers counts,data] [--no-graphs] [--no-reductions]
#           [--threads 16] [--float64-counts] [--scale-f64]
#
# --full writes EVERYTHING the object holds (the AutoResearch policy for an ingested object:
# no analysis should ever need R to read it): every assay and every layer (v5 split layers joined,
# with the per-cell split recorded; scale.data as a dense array; a v3 `data` slot that only repeats
# `counts` recorded as an alias, not a copy), per-feature metadata, VariableFeatures, every meta.data
# column with its type and factor levels, Idents (obs column `active.ident`), every reduction
# (embeddings, loadings, projected loadings, stdev, key, assay, misc), every graph and Neighbor
# object, and misc / commands / tools / images / assay and reduction extras (SCT models, ...) as
# uns.json (+ files under uns/). Anything that cannot be converted is listed in
# manifest.json "unconvertible" with its path, class and why.
#
# Layout (all arrays are NPY v1.0, little-endian):
#   manifest.json                         shapes, dtypes, factor levels, provenance, unconvertible
#   cells.txt, obs.csv                    cell barcodes / metadata (row order = cells)
#   assays/<A>/features.txt               feature names
#   assays/<A>/<layer>.csr.{data,indices,indptr}.npy
#                                         cells x features CSR. A dgCMatrix (features x
#                                         cells, CSC) has exactly these slots, so they are
#                                         written verbatim: @x -> data, @i -> indices,
#                                         @p -> indptr. Integer-valued layers (counts) are
#                                         stored as float32 (exact below 2^24) so Python
#                                         never has to cast (a cast would copy).
#   assays/<A>/<layer>.split.txt          (--full, v5 split layers) the split each cell came from
#   assays/<A>/<layer>.dense.npy + <layer>.dense_features.txt
#                                         (--full) dense layers (scale.data): cells x k, C order
#   assays/<A>/var.csv                    (--full) per-feature metadata (meta.features / meta.data)
#   assays/<A>/variable_features.txt      (if set)
#   reductions/<R>.npy                    cells x k float64 (Fortran order)
#   reductions/<R>.loadings.npy + .loadings_features.txt   (+ .projected.npy / .projected_features.txt)
#   graphs/<G>.csr.{data,indices,indptr}.npy   cells x cells (stored transposed-CSC = CSR of t(G))
#   neighbors/<N>.idx.npy, <N>.dist.npy   (--full) cells x k, 0-based positions in cells.txt
#                                         (index_space "cells"); a query -> reference Neighbor keeps
#                                         its 0-based reference positions (index_space "other")
#   uns.json, uns/*                       (--full) misc, commands, tools, images, seurat extras
#
# Python's first `arsc.open()` adds obs.parquet and the per-feature CSC copies
# (<layer>.csc.*.npy) used for fast gene lookups.
#
# If an assay has none of the requested layers (e.g. an ADT assay that only
# kept `data`), the first non-empty of counts/data is exported instead and the
# manifest records which layer was written. Seurat v5 split layers
# (counts.1, counts.2, ...) are joined back into one matrix in cell order.

# Extra R libraries: $ARSC_RLIB and $AR_ROOT/.Rlib, when set (R_LIBS / R_LIBS_USER are honored by R itself).
.arsc_libs <- c(Sys.getenv("ARSC_RLIB"), if (nzchar(Sys.getenv("AR_ROOT"))) file.path(Sys.getenv("AR_ROOT"), ".Rlib") else "")
.arsc_libs <- .arsc_libs[nzchar(.arsc_libs) & dir.exists(.arsc_libs)]
if (length(.arsc_libs)) .libPaths(unique(c(.arsc_libs, .libPaths())))
suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: export_seurat.R <object.rds|.qs> <out_dir> [--full] [--assays A,B] [--layers counts,data] [--no-graphs] [--no-reductions]")
rds <- args[1]; out <- args[2]
opt <- function(flag, default = NULL) {
  k <- which(args == flag)
  if (length(k) && k < length(args)) args[k + 1] else default
}
full <- "--full" %in% args
want_assays <- opt("--assays")
want_layers <- strsplit(opt("--layers", "counts"), ",")[[1]]
do_graphs <- !("--no-graphs" %in% args)
do_reds <- !("--no-reductions" %in% args)
n_threads <- as.integer(opt("--threads", "16"))
counts_f64 <- "--float64-counts" %in% args
scale_type <- if ("--scale-f64" %in% args) "f8" else "f4"
if (full && (!is.null(want_assays) || "--no-graphs" %in% args || "--no-reductions" %in% args || !is.null(opt("--layers"))))
  stop("--full exports everything; it cannot be combined with --assays/--layers/--no-graphs/--no-reductions")

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

read_object <- function(p) {
  if (grepl("[.]qs$", p, ignore.case = TRUE)) return(qs::qread(p, nthreads = n_threads))
  o <- tryCatch(readRDS(p), error = function(e) NULL)  # readRDS reads gzip/bzip2/xz-compressed files itself
  if (is.null(o)) o <- read_gz_twice(p, file.path(out, ".outer_unwrapped.rds"))  # an .rds that was gzipped again (GEO)
  o
}
read_gz_twice <- function(p, tmp) {  # unwrap the outer gzip into a file next to the output, then readRDS that
  dir.create(dirname(tmp), recursive = TRUE, showWarnings = FALSE)
  i <- gzfile(p, "rb"); o <- file(tmp, "wb")
  repeat { b <- readBin(i, "raw", 1e8); if (!length(b)) break; writeBin(b, o) }
  close(i); close(o)
  on.exit(unlink(tmp))
  readRDS(tmp)
}

obj <- read_object(rds)
if (!inherits(obj, "Seurat")) stop("not a Seurat object: ", paste(class(obj), collapse = "/"))
msg("loaded ", rds, " : ", ncol(obj), " cells", if (full) " (--full)" else "")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cells <- colnames(obj)
writeLines(cells, file.path(out, "cells.txt"), useBytes = TRUE)

unconvertible <- list()
# What a Neighbor object's nn.idx values index: "cells" (its own cells, the FindNeighbors case) or "other" (another
# object's cells: a query -> reference Neighbor such as Azimuth's "query_ref.nn", or any value beyond its cell count).
# tools/R/store_stats.R in autoresearch applies the same rule on the source side.
neighbor_index_space <- function(name, idx, n_cells) {
  if (grepl("query_ref|ref\\.nn$", name) || isTRUE(max(idx, na.rm = TRUE) > n_cells)) "other" else "cells"
}
note_unconv <- function(path, x, why) {
  unconvertible[[length(unconvertible) + 1]] <<- list(path = path, class = paste(class(x), collapse = "/"), reason = why)
  msg("  NOT CONVERTED ", path, " (", paste(class(x), collapse = "/"), "): ", why)
}

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

# the layer names of one group: v5 "counts" -> counts, counts.P1, counts.P2 ...; v3 the slot itself
layer_parts <- function(assay_obj, layer) {
  if (inherits(assay_obj, "Assay5")) {
    lays <- SeuratObject::Layers(assay_obj)
    lays[lays == layer | startsWith(lays, paste0(layer, "."))]
  } else layer
}

get_layer <- function(assay_obj, layer, parts = NULL) {
  if (inherits(assay_obj, "Assay5")) {
    lays <- if (is.null(parts)) layer_parts(assay_obj, layer) else parts
    if (!length(lays)) return(NULL)
    if (length(lays) == 1) {
      m <- SeuratObject::LayerData(assay_obj, layer = lays)
      if (is.null(rownames(m))) rownames(m) <- SeuratObject::Features(assay_obj, layer = lays)
      if (is.null(colnames(m))) colnames(m) <- SeuratObject::Cells(assay_obj, layer = lays)
    } else {
      parts_m <- lapply(lays, function(l) {
        p <- SeuratObject::LayerData(assay_obj, layer = l)
        if (is.null(rownames(p))) rownames(p) <- SeuratObject::Features(assay_obj, layer = l)
        if (is.null(colnames(p))) colnames(p) <- SeuratObject::Cells(assay_obj, layer = l)
        p
      })
      feats <- Reduce(union, lapply(parts_m, rownames))
      dense <- all(vapply(parts_m, is.matrix, TRUE))
      parts_m <- lapply(parts_m, function(p) {
        if (identical(rownames(p), feats)) return(p)
        z <- if (dense) matrix(NA_real_, length(feats), ncol(p), dimnames = list(feats, colnames(p)))
             else Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                       dims = c(length(feats), ncol(p)), dimnames = list(feats, colnames(p)))
        z[rownames(p), ] <- p
        z
      })
      m <- do.call(cbind, parts_m)
    }
    m
  } else {
    if (!(layer %in% c("counts", "data", "scale.data"))) return(NULL)
    m <- methods::slot(assay_obj, layer)
    if (!length(m) || nrow(m) == 0) return(NULL)
    m
  }
}

# every layer group of an assay (full mode): v3 slots that are not empty; v5 layer names folded by prefix
layer_groups <- function(ao) {
  if (inherits(ao, "Assay5")) {
    lays <- SeuratObject::Layers(ao)
    base <- vapply(lays, function(l) {
      for (b in c("scale.data", "counts", "data")) if (l == b || startsWith(l, paste0(b, "."))) return(b)
      l
    }, "")
    unique(base)
  } else {
    Filter(function(l) { m <- methods::slot(ao, l); length(m) > 0 && nrow(m) > 0 && ncol(m) > 0 },
           c("counts", "data", "scale.data"))
  }
}

same_sparse <- function(a, b) {
  if (is.null(a) || is.null(b) || !identical(dim(a), dim(b))) return(FALSE)
  if (!inherits(a, "dgCMatrix") || !inherits(b, "dgCMatrix")) return(FALSE)
  identical(a@p, b@p) && identical(a@i, b@i) && identical(a@x, b@x) && identical(dimnames(a), dimnames(b))
}

# ---------------------------------------------------------------- uns: R values -> JSON (+ files under uns/)
# JSON shapes the Python reader understands (arsc.formats._uns_from_json): scalars, lists, dicts,
# {"__ndarray__": values, "dtype", "shape", "order": "F", "names"/"levels"}, {"__dataframe__": {col: ...},
# "index"}, {"__npy__"|"__csv__"|"__mtx__"|"__txt__": "uns/<file>"}, {"__deparsed__": "<R code>"}; S4 objects
# become {"__class__": "<class>", <slot>: ...}. Anything else is listed as unconvertible.
UNS_SKIP <- structure(list(), class = "uns_skip")
uns_files <- 0L
uns_file <- function(path, ext) {
  dir.create(file.path(out, "uns"), showWarnings = FALSE)
  uns_files <<- uns_files + 1L
  f <- sprintf("uns/%04d_%s%s", uns_files, substr(gsub("[^A-Za-z0-9_.-]+", "_", path), 1, 80), ext)
  list(rel = f, abs = file.path(out, f))
}
BIG <- 1e5
atomic_json <- function(x, path, force_array = FALSE) {
  if (is.factor(x)) {
    return(list(`__ndarray__` = I(as.character(x)), dtype = "str", levels = I(levels(x)),
                shape = I(dim(x) %||% length(x))))
  }
  d <- dim(x)
  nm <- names(x)
  if (!force_array && is.null(d) && length(x) == 1 && is.null(nm)) return(if (is.na(x) && !is.character(x)) NULL else x)
  if (length(x) > BIG) {
    if (is.character(x)) {
      f <- uns_file(path, ".txt")
      con <- file(f$abs, "wb"); writeLines(enc2utf8(ifelse(is.na(x), "\\N", x)), con, useBytes = TRUE); close(con)
      return(list(`__txt__` = f$rel, shape = I(d %||% length(x)), na = "\\N"))
    }
    f <- uns_file(path, ".npy")
    type <- if (is.integer(x) && !anyNA(x)) "i4" else "f8"
    write_npy(as.vector(if (is.logical(x)) as.integer(x) else x), f$abs, type, d %||% length(x), fortran = TRUE)
    return(list(`__npy__` = f$rel, logical = is.logical(x)))
  }
  if (!force_array && !is.null(nm) && is.null(d) && all(nzchar(nm)) && !anyDuplicated(nm) && length(x) <= 1000) {
    return(lapply(stats::setNames(as.list(x), nm), function(v) if (is.na(v) && !is.character(v)) NULL else v))
  }
  dtype <- if (is.character(x)) "str" else if (is.logical(x)) (if (anyNA(x)) "object" else "bool")
           else if (is.integer(x) && !anyNA(x)) "<i4" else "<f8"
  vals <- as.vector(x)
  if (is.integer(vals) && anyNA(vals)) vals <- as.double(vals)
  r <- list(`__ndarray__` = I(vals), dtype = dtype, shape = I(d %||% length(x)), order = "F")
  if (!is.null(nm)) r$names <- I(nm)
  if (!is.null(d) && !is.null(dimnames(x))) r$dimnames <- lapply(dimnames(x), function(v) if (is.null(v)) NULL else I(v))
  r
}
`%||%` <- function(a, b) if (is.null(a)) b else a
to_json <- function(x, path, depth = 0) {
  if (depth > 40) { note_unconv(path, x, "nested more than 40 levels deep"); return(UNS_SKIP) }
  if (is.null(x)) return(NULL)
  if (inherits(x, c("Date", "POSIXt", "difftime"))) x <- as.character(x)
  if (inherits(x, "formula") || is.language(x) || is.symbol(x))
    return(list(`__deparsed__` = paste(deparse(x, width.cutoff = 500L), collapse = "\n")))
  if (is.function(x)) return(list(`__deparsed__` = paste(deparse(x, width.cutoff = 500L), collapse = "\n"), kind = "function"))
  if (is.environment(x) || typeof(x) %in% c("externalptr", "weakref", "promise", "bytecode", "char")) {
    note_unconv(path, x, paste0("an R ", typeof(x), " (no file representation)")); return(UNS_SKIP)
  }
  if (inherits(x, "Matrix")) {
    m <- tryCatch(as(as(x, "CsparseMatrix"), "generalMatrix"), error = function(e) NULL)
    if (is.null(m) || !inherits(m, "dMatrix")) m <- tryCatch(as(m, "dMatrix"), error = function(e) NULL)
    if (is.null(m)) { note_unconv(path, x, "Matrix class without a numeric sparse form"); return(UNS_SKIP) }
    f <- uns_file(path, ".mtx")
    Matrix::writeMM(m, f$abs)
    return(list(`__mtx__` = f$rel, rownames = if (is.null(rownames(m))) NULL else I(rownames(m)),
                colnames = if (is.null(colnames(m))) NULL else I(colnames(m))))
  }
  if (is.data.frame(x)) {
    if (as.numeric(nrow(x)) * max(1, ncol(x)) > BIG) {
      f <- uns_file(path, ".csv")
      df <- x
      for (cn in names(df)) if (is.list(df[[cn]]) || inherits(df[[cn]], c("Date", "POSIXt"))) df[[cn]] <- vapply(df[[cn]], function(v) paste(format(v), collapse = ";"), "")
      df <- cbind(data.frame(`_index` = rownames(x), check.names = FALSE, stringsAsFactors = FALSE), df)
      data.table::fwrite(df, f$abs, na = "NA", quote = TRUE)
      return(list(`__csv__` = f$rel, types = lapply(x, function(v) class(v)[1]),
                  levels = Filter(Negate(is.null), lapply(x, function(v) if (is.factor(v)) I(levels(v)) else NULL))))
    }
    cols <- list()
    for (cn in names(x)) {
      v <- if (is.atomic(x[[cn]]) && !inherits(x[[cn]], c("Date", "POSIXt", "difftime"))) atomic_json(unname(x[[cn]]), paste0(path, "/", cn), force_array = TRUE)
           else to_json(x[[cn]], paste0(path, "/", cn), depth + 1)
      if (!inherits(v, "uns_skip")) cols[[cn]] <- v
    }
    return(list(`__dataframe__` = cols, index = I(rownames(x)), columns = I(names(x))))
  }
  if (is.atomic(x)) return(atomic_json(x, path))
  if (isS4(x)) {
    r <- list(`__class__` = class(x)[1])
    for (s in methods::slotNames(x)) {
      v <- to_json(methods::slot(x, s), paste0(path, "@", s), depth + 1)
      if (!inherits(v, "uns_skip")) r[[s]] <- v
    }
    return(r)
  }
  if (is.list(x)) {
    nm <- names(x)
    if (is.null(nm)) {
      r <- lapply(seq_along(x), function(i) to_json(x[[i]], paste0(path, "[", i, "]"), depth + 1))
      return(lapply(r, function(v) if (inherits(v, "uns_skip")) NULL else v))
    }
    nm[!nzchar(nm)] <- paste0("_", which(!nzchar(nm)))
    nm <- make.unique(nm)
    r <- list()
    for (i in seq_along(x)) {
      v <- to_json(x[[i]], paste0(path, "/", nm[i]), depth + 1)
      if (!inherits(v, "uns_skip")) r[[nm[i]]] <- v
    }
    if (!length(r)) return(structure(list(), names = character(0)))
    return(r)
  }
  note_unconv(path, x, "no conversion for this R type")
  UNS_SKIP
}

manifest <- list(format = "arsc-store", format_version = 1L,
                 source = normalizePath(rds, winslash = "/"), n_cells = length(cells),
                 seurat_version = as.character(obj@version),
                 default_assay = SeuratObject::DefaultAssay(obj),
                 assays = list(), reductions = list(), graphs = list(), obs = list())
if (full) manifest$export <- list(mode = "full", exporter = "arsc/R/export_seurat.R", scale_dtype = c(f4 = "float32", f8 = "float64")[[scale_type]],
                                  seurat_object = as.character(packageVersion("SeuratObject")), r = R.version.string)
uns <- list()

# ---------------------------------------------------------------- assays
assays <- names(obj@assays)
if (!is.null(want_assays)) assays <- intersect(assays, strsplit(want_assays, ",")[[1]])
for (a in assays) {
  ao <- obj@assays[[a]]
  adir <- file.path(out, "assays", a); dir.create(adir, recursive = TRUE, showWarnings = FALSE)
  entry <- list(class = class(ao)[1], layers = list())
  feats <- NULL
  written <- list()  # layer -> the matrix written (to detect a v3 data slot that repeats counts)
  pad_cells <- function(m, l) {  # assay measured on a subset of cells (e.g. ADT absent for a cohort)
    if (identical(colnames(m), cells)) return(m)
    have <- intersect(cells, colnames(m))
    if (length(have) < length(cells)) {
      msg("  ", a, "/", l, ": measured in ", length(have), "/", length(cells), " cells; others written as empty rows")
      entry$cells_measured <<- length(have)
      writeLines(have, file.path(adir, "cells_measured.txt"), useBytes = TRUE)
      if (is.matrix(m)) {
        pad <- matrix(NA_real_, nrow(m), length(cells) - length(have), dimnames = list(rownames(m), setdiff(cells, have)))
        m <- cbind(m[, have, drop = FALSE], pad)
      } else {
        pad <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = c(nrow(m), length(cells) - length(have)),
                                    dimnames = list(rownames(m), setdiff(cells, have)))
        m <- cbind(as(m[, have, drop = FALSE], "CsparseMatrix"), pad)
      }
    }
    m[, cells, drop = FALSE]
  }
  write_split <- function(l, parts) {  # the split each cell came from (v5 layers counts.P1, counts.P2 ...)
    if (length(parts) < 2) return(NULL)
    lab <- rep("", length(cells))
    for (p in parts) {
      cc <- SeuratObject::Cells(ao, layer = p)
      lab[match(cc, cells)] <- sub(paste0("^", gsub(".", "\\.", l, fixed = TRUE), "\\."), "", p)
    }
    con <- file(file.path(adir, paste0(l, ".split.txt")), "wb"); writeLines(enc2utf8(lab), con, useBytes = TRUE); close(con)
    list(parts = I(parts), file = paste0(l, ".split.txt"))
  }
  write_dense <- function(l, m, parts) {
    m <- as.matrix(m)
    m <- pad_cells(m, l)
    df <- paste0(l, ".dense_features.txt")
    writeLines(rownames(m), file.path(adir, df), useBytes = TRUE)
    write_npy(as.vector(m), file.path(adir, paste0(l, ".dense.npy")), scale_type, c(ncol(m), nrow(m)), fortran = FALSE)
    e <- list(shape = c(ncol(m), nrow(m)), dtype = c(f4 = "float32", f8 = "float64")[[scale_type]],
              file = paste0(l, ".dense.npy"), features_file = df)
    sp <- write_split(l, parts); if (!is.null(sp)) e$split <- sp
    entry$dense[[l]] <<- e
    msg("  wrote ", a, "/", l, " dense ", nrow(m), "x", ncol(m), " (", e$dtype, ")")
    TRUE
  }
  write_layer <- function(l) {
    parts <- tryCatch(layer_parts(ao, l), error = function(e) l)
    m <- tryCatch(get_layer(ao, l, parts), error = function(e) { msg("  skip ", a, "/", l, ": ", conditionMessage(e)); NULL })
    if (is.null(m) || !inherits(m, c("Matrix", "matrix")) || nrow(m) == 0 || ncol(m) == 0) return(FALSE)
    if (full && (l == "scale.data" || (!is.null(feats) && !all(rownames(m) %in% feats))))
      return(write_dense(l, m, parts))  # scale.data, or a layer on other features: a dense array with its own features
    if (is.matrix(m)) m <- as(m, "CsparseMatrix")
    m <- pad_cells(m, l)
    if (is.null(feats)) feats <<- rownames(m)
    else if (!identical(rownames(m), feats)) {
      present <- rownames(m)
      if (full && length(present) < length(feats)) {  # a layer on fewer features: zero rows for the others, recorded
        z <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = c(length(feats) - length(present), ncol(m)),
                                  dimnames = list(setdiff(feats, present), colnames(m)))
        m <- rbind(as(m, "CsparseMatrix"), z)
        ff <- paste0(l, ".features_present.txt")
        writeLines(present, file.path(adir, ff), useBytes = TRUE)
        entry$features_present[[l]] <<- ff
      }
      m <- m[feats, , drop = FALSE]
    }
    if (!inherits(m, "dgCMatrix")) m <- as(as(m, "CsparseMatrix"), "generalMatrix")
    if (full && l == "data" && !is.null(written$counts) && same_sparse(written$counts, m)) {
      entry$layers[[l]] <<- c(entry$layers$counts[c("shape", "nnz", "dtype", "integer_valued")], list(alias_of = "counts"))
      msg("  ", a, "/data repeats counts (never normalized): recorded as an alias of counts, not written twice")
      return(TRUE)
    }
    entry$layers[[l]] <<- write_sparse(m, file.path(adir, paste0(l, ".csr")))
    sp <- if (full) write_split(l, parts) else NULL
    if (!is.null(sp)) entry$layers[[l]]$split <<- sp
    if (full && l == "counts") written$counts <<- m
    msg("  wrote ", a, "/", l, " ", nrow(m), "x", ncol(m), " nnz=", entry$layers[[l]]$nnz,
        " (", entry$layers[[l]]$dtype, ")")
    TRUE
  }
  lays <- if (full) layer_groups(ao) else want_layers
  if (full) lays <- c(intersect(c("counts", "data"), lays), setdiff(lays, c("counts", "data", "scale.data")), intersect("scale.data", lays))
  for (l in lays) write_layer(l)
  written <- NULL
  if (!length(entry$layers) && !length(entry$dense)) {
    for (l in setdiff(c("counts", "data"), want_layers)) {
      msg("  ", a, ": none of [", paste(want_layers, collapse = ","), "] present; falling back to '", l, "'")
      if (write_layer(l)) break
    }
  }
  if (is.null(feats)) feats <- rownames(ao)
  writeLines(feats, file.path(adir, "features.txt"), useBytes = TRUE)
  vf <- tryCatch(SeuratObject::VariableFeatures(obj, assay = a), error = function(e) character(0))
  if (length(vf)) writeLines(vf, file.path(adir, "variable_features.txt"), useBytes = TRUE)
  if (full) {
    entry$key <- tryCatch(SeuratObject::Key(ao), error = function(e) NULL)
    vt <- tryCatch(if (inherits(ao, "Assay5")) ao@meta.data else ao@meta.features, error = function(e) NULL)
    if (!is.null(vt) && ncol(vt) > 0) {
      af <- if (inherits(ao, "Assay5")) rownames(ao) else rownames(vt)
      if (is.null(rownames(vt)) || inherits(ao, "Assay5")) rownames(vt) <- af
      vt <- vt[match(feats, rownames(vt)), , drop = FALSE]
      types <- list()
      for (cn in names(vt)) {
        v <- vt[[cn]]
        if (is.list(v)) { vt[[cn]] <- vapply(v, function(z) paste(format(z), collapse = ";"), ""); types[[cn]] <- list(type = "character")
          note_unconv(paste0("assays/", a, "/meta.features$", cn), v, "a list column, written as text (values joined by ';')") }
        else if (inherits(v, c("Date", "POSIXt", "difftime"))) { vt[[cn]] <- as.character(v); types[[cn]] <- list(type = "character") }
        else if (is.factor(v)) types[[cn]] <- list(type = "factor", levels = I(levels(v)))
        else if (is.logical(v)) types[[cn]] <- list(type = "logical")
        else if (is.integer(v)) types[[cn]] <- list(type = "integer")
        else if (is.numeric(v)) types[[cn]] <- list(type = "double")
        else { vt[[cn]] <- as.character(v); types[[cn]] <- list(type = "character") }
      }
      vt <- cbind(data.frame(feature = feats, stringsAsFactors = FALSE), vt)
      data.table::fwrite(vt, file.path(adir, "var.csv"), na = "NA", quote = TRUE)
      entry$var <- list(file = "var.csv", columns = types)
    }
    extra <- list()
    skip <- c("counts", "data", "scale.data", "layers", "cells", "features", "meta.features", "meta.data", "var.features", "key")
    for (s in setdiff(methods::slotNames(ao), skip)) {
      v <- methods::slot(ao, s)
      if (is.null(v) || (length(v) == 0 && !isS4(v))) next
      j <- to_json(v, paste0("assays/", a, "@", s))
      if (!inherits(j, "uns_skip")) extra[[s]] <- j
    }
    if (length(extra)) uns$seurat$assays[[a]] <- extra
  }
  manifest$assays[[a]] <- entry
}

# ---------------------------------------------------------------- reductions
pad_rows <- function(emb, what) {  # a reduction computed on a subset of cells: NaN rows for the others
  if (identical(rownames(emb), cells)) return(list(m = emb, n = NULL))
  have <- intersect(cells, rownames(emb))
  if (length(have) == length(cells)) return(list(m = emb[cells, , drop = FALSE], n = NULL))
  if (!full) stop(what, ": covers ", length(have), " of ", length(cells), " cells (re-run with --full)")
  z <- matrix(NaN, length(cells), ncol(emb), dimnames = list(cells, colnames(emb)))
  z[have, ] <- emb[have, , drop = FALSE]
  msg("  ", what, ": ", length(have), "/", length(cells), " cells; others NaN")
  list(m = z, n = length(have))
}
if (do_reds && length(obj@reductions)) {
  dir.create(file.path(out, "reductions"), showWarnings = FALSE)
  for (r in names(obj@reductions)) {
    ro <- obj@reductions[[r]]
    pr <- pad_rows(ro@cell.embeddings, paste0("reduction ", r))
    emb <- pr$m
    write_npy(as.vector(emb), file.path(out, "reductions", paste0(r, ".npy")), "f8", dim(emb), fortran = TRUE)
    e <- list(shape = dim(emb), columns = colnames(emb), assay = ro@assay.used, key = ro@key)
    if (!is.null(pr$n)) e$cells_measured <- pr$n
    ld <- ro@feature.loadings
    if (length(ld) && nrow(ld) > 0) {
      write_npy(as.vector(ld), file.path(out, "reductions", paste0(r, ".loadings.npy")), "f8", dim(ld), fortran = TRUE)
      writeLines(rownames(ld), file.path(out, "reductions", paste0(r, ".loadings_features.txt")), useBytes = TRUE)
      e$loadings_shape <- dim(ld)
    }
    if (length(ro@stdev)) e$stdev <- I(ro@stdev)
    if (full) {
      lp <- ro@feature.loadings.projected
      if (length(lp) && nrow(lp) > 0) {
        write_npy(as.vector(lp), file.path(out, "reductions", paste0(r, ".projected.npy")), "f8", dim(lp), fortran = TRUE)
        writeLines(rownames(lp), file.path(out, "reductions", paste0(r, ".projected_features.txt")), useBytes = TRUE)
        e$projected_shape <- dim(lp)
      }
      e$global <- isTRUE(ro@global)
      extra <- list()
      if (length(ro@misc)) { j <- to_json(ro@misc, paste0("reductions/", r, "@misc")); if (!inherits(j, "uns_skip")) extra$misc <- j }
      js <- ro@jackstraw
      if (length(js@empirical.p.values) || length(js@overall.p.values)) {
        j <- to_json(js, paste0("reductions/", r, "@jackstraw")); if (!inherits(j, "uns_skip")) extra$jackstraw <- j
      }
      if (length(extra)) uns$seurat$reductions[[r]] <- extra
    }
    manifest$reductions[[r]] <- e
    msg("  reduction ", r, " ", nrow(emb), "x", ncol(emb))
  }
}

# ---------------------------------------------------------------- graphs and neighbors
if (do_graphs && length(obj@graphs)) {
  dir.create(file.path(out, "graphs"), showWarnings = FALSE)
  for (g in names(obj@graphs)) {
    gm <- obj@graphs[[g]]
    if (!identical(colnames(gm), cells) || !identical(rownames(gm), cells)) {
      if (!all(cells %in% colnames(gm)) || !all(cells %in% rownames(gm))) {
        if (!full) stop("graph ", g, " does not cover every cell")
        note_unconv(paste0("graphs/", g), gm, "the graph does not cover every cell of the object"); next
      }
      gm <- gm[cells, cells]
    }
    au <- tryCatch(methods::slot(gm, "assay.used"), error = function(e) character(0))
    # dgCMatrix slots of G are the CSR of t(G); arsc transposes on load.
    manifest$graphs[[g]] <- c(write_sparse(gm, file.path(out, "graphs", paste0(g, ".csr")), allow_f32 = FALSE),
                              list(transposed = TRUE), if (full && length(au)) list(assay = au) else NULL)
    msg("  graph ", g)
  }
}
if (full && length(obj@neighbors)) {
  dir.create(file.path(out, "neighbors"), showWarnings = FALSE)
  for (nn in names(obj@neighbors)) {
    nb <- obj@neighbors[[nn]]
    idx <- tryCatch(nb@nn.idx, error = function(e) NULL)
    dist <- tryCatch(nb@nn.dist, error = function(e) NULL)
    cn <- tryCatch(nb@cell.names, error = function(e) NULL)
    if (is.null(idx) || !length(idx)) { note_unconv(paste0("neighbors/", nn), nb, "no nn.idx (an index-only Neighbor object)"); next }
    if (is.null(cn) || !length(cn)) cn <- cells
    if (!all(cn %in% cells) || length(cn) != length(cells)) {
      note_unconv(paste0("neighbors/", nn), nb, "the neighbor graph does not cover exactly the object's cells"); next
    }
    pos <- match(cn, cells)  # row i of nn.idx is cell cn[i]; values are row numbers in cn
    ord <- order(pos)
    # A query -> reference Neighbor (Azimuth / MapQuery "query_ref.nn"): rows are this object's cells, but the values
    # are row numbers in ANOTHER object's cells (the reference), so they cannot be mapped through cn. Kept as they
    # are (0-based), flagged index_space = "other". Mapping them through cn wrote NA (-2^31) and wrong positions.
    other <- neighbor_index_space(nn, idx, length(cn)) == "other"
    idx0 <- if (other) matrix(as.integer(idx) - 1L, nrow(idx)) else matrix(pos[idx] - 1L, nrow(idx))
    if (anyNA(idx0)) {
      note_unconv(paste0("neighbors/", nn), nb, "nn.idx holds NA or values outside its cells"); next
    }
    idx0 <- idx0[ord, , drop = FALSE]
    write_npy(as.vector(idx0), file.path(out, "neighbors", paste0(nn, ".idx.npy")), "i4", dim(idx0), fortran = TRUE)
    e <- list(shape = dim(idx0), index_base = 0L, idx = paste0(nn, ".idx.npy"),
              index_space = if (other) "other" else "cells")
    if (!is.null(dist) && length(dist)) {
      write_npy(as.vector(dist[ord, , drop = FALSE]), file.path(out, "neighbors", paste0(nn, ".dist.npy")), "f8", dim(dist), fortran = TRUE)
      e$dist <- paste0(nn, ".dist.npy")
    }
    ai <- tryCatch(to_json(nb@alg.info, paste0("neighbors/", nn, "@alg.info")), error = function(e) NULL)
    if (!is.null(ai) && !inherits(ai, "uns_skip")) e$alg_info <- ai
    if (length(tryCatch(nb@alg.idx, error = function(e) NULL)))
      note_unconv(paste0("neighbors/", nn, "@alg.idx"), nb@alg.idx, "an Annoy/RcppAnnoy index pointer (live memory, rebuilt from the embedding)")
    manifest$neighbors[[nn]] <- e
    msg("  neighbors ", nn, " ", nrow(idx0), "x", ncol(idx0))
  }
}

# ---------------------------------------------------------------- meta.data (+ Idents)
md <- obj@meta.data
md <- md[cells, , drop = FALSE]
if (full) {
  idc <- if ("active.ident" %in% names(md)) "active.ident.arsc" else "active.ident"
  id <- SeuratObject::Idents(obj)
  md[[idc]] <- factor(as.character(id[cells]), levels = levels(id))
  manifest$idents_column <- idc
}
for (cn in names(md)) {
  v <- md[[cn]]
  if (inherits(v, c("Date", "IDate", "POSIXt", "difftime"))) {  # dates are integer/double internally: export as text
    md[[cn]] <- as.character(v); manifest$obs[[cn]] <- list(type = "character", from = class(v)[1])
  }
  else if (is.factor(v)) manifest$obs[[cn]] <- list(type = "factor", levels = I(levels(v)))  # I(): a 1-level factor stays a JSON array
  else if (is.logical(v)) manifest$obs[[cn]] <- list(type = "logical")
  else if (is.integer(v)) manifest$obs[[cn]] <- list(type = "integer")
  else if (is.numeric(v)) manifest$obs[[cn]] <- list(type = "double")
  else if (is.list(v)) {
    md[[cn]] <- vapply(v, function(z) paste(format(z), collapse = ";"), ""); manifest$obs[[cn]] <- list(type = "character", from = "list")
    if (full) note_unconv(paste0("meta.data$", cn), v, "a list column, written as text (values joined by ';')")
  }
  else { md[[cn]] <- as.character(v); manifest$obs[[cn]] <- list(type = "character") }
}
md <- cbind(data.frame(cell = cells, stringsAsFactors = FALSE), md)
data.table::fwrite(md, file.path(out, "obs.csv"), na = "NA", quote = TRUE)

# ---------------------------------------------------------------- misc, commands, tools, images
if (full) {
  for (s in c("misc", "commands", "tools", "images")) {
    v <- tryCatch(methods::slot(obj, s), error = function(e) NULL)
    if (is.null(v) || !length(v)) next
    j <- to_json(v, s)
    if (!inherits(j, "uns_skip")) uns[[s]] <- j
  }
  uns$seurat$project.name <- obj@project.name
  uns$seurat$version <- as.character(obj@version)
  known <- c("assays", "meta.data", "active.assay", "active.ident", "graphs", "neighbors", "reductions", "images",
             "project.name", "misc", "version", "commands", "tools")
  for (s in setdiff(methods::slotNames(obj), known)) {  # slots of a newer SeuratObject
    j <- to_json(methods::slot(obj, s), s)
    if (!inherits(j, "uns_skip")) uns$seurat$slots[[s]] <- j
  }
  writeLines(jsonlite::toJSON(uns, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = FALSE),
             file.path(out, "uns.json"), useBytes = TRUE)
  manifest$unconvertible <- unconvertible
}
writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null"),
           file.path(out, "manifest.json"))
msg("done -> ", out, if (full) sprintf(" (%d item(s) not convertible, listed in manifest.json)", length(unconvertible)) else "")
