if (nzchar(Sys.getenv("ARSC_R_PRELUDE"))) source(Sys.getenv("ARSC_R_PRELUDE"))   # a site's R setup, if any
# Seurat's FindAllMarkers split over processes: FindAllMarkers is FindMarkers(ident.1 = g) for every cell type g in
# turn, one thread each, so the cell types can be shared among several R processes and the per-type times summed
# (seurat_markers_shard.py). This process: read the same cells as r_seurat_timing.R, NormalizeData (Seurat's
# default), one untimed FindMarkers on a 2,000-cell subset (first-call costs), then FindMarkers(test.use = "wilcox")
# for each of its cell types, timed one by one.
#   Rscript r_seurat_markers_shard.R < spec.json   (spec on stdin: store, rows, cell_type, groups)
# Nothing is written to disk; the result goes to stdout as one JSON line after the line @@RESULT@@.
rlib <- Sys.getenv("ARSC_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages({ library(Matrix); library(jsonlite); library(Seurat) })
spec <- fromJSON(paste(readLines(file("stdin"), warn = FALSE), collapse = "\n"))

npy_header <- function(f) {   # numpy .npy v1/v2/v3 header -> data offset, dtype, length (as r_seurat_timing.R)
  con <- file(f, "rb"); on.exit(close(con))
  magic <- readBin(con, "raw", 8)
  stopifnot(identical(magic[1:6], as.raw(c(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59))))
  hl <- if (as.integer(magic[7]) == 1) readBin(con, "integer", 1, size = 2, signed = FALSE, endian = "little")
        else readBin(con, "integer", 1, size = 4, endian = "little")
  h <- rawToChar(readBin(con, "raw", hl))
  descr <- sub(".*'descr': *'([^']+)'.*", "\\1", h)
  shape <- as.numeric(strsplit(sub(".*'shape': *\\(([^)]*)\\).*", "\\1", h), ",")[[1]][1])
  list(offset = (if (as.integer(magic[7]) == 1) 10 else 12) + hl, descr = descr, n = shape)
}

read_store_rows <- function(store, rows) {   # rows: 0-based, sorted; genes x cells dgCMatrix (as r_seurat_timing.R)
  d <- file.path(store, "assays", "RNA")
  man <- fromJSON(file.path(store, "manifest.json"))
  pre <- man$assays$RNA$layers$counts$prefix
  hp <- npy_header(file.path(d, paste0(pre, ".indptr.npy")))
  hx <- npy_header(file.path(d, paste0(pre, ".data.npy")))
  hi <- npy_header(file.path(d, paste0(pre, ".indices.npy")))
  stopifnot(hp$descr == "<i4", hx$descr == "<f4", hi$descr == "<i4")
  cp <- file(file.path(d, paste0(pre, ".indptr.npy")), "rb")
  seek(cp, hp$offset); indptr <- readBin(cp, "integer", hp$n, size = 4, endian = "little"); close(cp)
  lo <- indptr[rows + 1]; hi_ <- indptr[rows + 2]; len <- hi_ - lo
  nnz <- sum(as.numeric(len)); if (nnz >= 2^31) stop("subset nnz exceeds the dgCMatrix limit")
  x <- numeric(nnz); ii <- integer(nnz)
  cx <- file(file.path(d, paste0(pre, ".data.npy")), "rb"); ci <- file(file.path(d, paste0(pre, ".indices.npy")), "rb")
  on.exit({ close(cx); close(ci) })
  pos <- 0
  brk <- c(0, which(diff(rows) != 1 | diff(cumsum(as.numeric(len)) %/% 8e6) != 0), length(rows))
  for (b in seq_len(length(brk) - 1)) {
    a <- brk[b] + 1; z <- brk[b + 1]
    n <- hi_[z] - lo[a]; if (n == 0) next
    seek(cx, hx$offset + 4 * lo[a]); seek(ci, hi$offset + 4 * lo[a])
    x[pos + seq_len(n)] <- readBin(cx, "double", n, size = 4, endian = "little")
    ii[pos + seq_len(n)] <- readBin(ci, "integer", n, size = 4, endian = "little")
    pos <- pos + n
  }
  stopifnot(pos == nnz)
  genes <- readLines(file.path(d, "features.txt"))
  cells <- readLines(file.path(store, "cells.txt"))[rows + 1]
  new("dgCMatrix", i = ii, p = c(0L, as.integer(cumsum(as.numeric(len)))), x = x,
      Dim = c(length(genes), length(rows)), Dimnames = list(genes, cells))
}

t0 <- as.numeric(Sys.time())
m <- read_store_rows(spec$store, as.integer(spec$rows))
o <- CreateSeuratObject(m, meta.data = data.frame(cell_type = spec$cell_type, row.names = colnames(m)),
                        min.cells = 0, min.features = 0)
rm(m); invisible(gc())
o <- NormalizeData(o, verbose = FALSE)
Idents(o) <- "cell_type"
t_prep <- as.numeric(Sys.time()) - t0
message(sprintf("[shard] %d cells read, object built and normalized in %.0f s", ncol(o), t_prep))

w <- subset(o, cells = colnames(o)[seq_len(min(2000, ncol(o)))])   # first-call costs, untimed
wg <- names(which(table(Idents(w)) >= 3))[1]
invisible(FindMarkers(w, ident.1 = wg, test.use = "wilcox", verbose = FALSE)); rm(w); invisible(gc())

per <- list()
for (g in spec$groups) {
  t1 <- as.numeric(Sys.time())
  m1 <- FindMarkers(o, ident.1 = g, test.use = "wilcox", verbose = FALSE)
  dt <- as.numeric(Sys.time()) - t1
  m1 <- m1[m1$p_val < 0.01, , drop = FALSE]   # FindAllMarkers keeps p_val < 0.01
  per[[g]] <- list(seconds = dt, cells = sum(Idents(o) == g), rows = nrow(m1))
  message(sprintf("[shard] timed markers, %s (%d cells): %.1f s", g, sum(Idents(o) == g), dt))
}
blas <- tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) NA)
res <- list(n = ncol(o), prep_s = t_prep, markers = per, blas_threads = blas,
            future_workers = as.integer(future::nbrOfWorkers()), r_version = R.version.string,
            seurat = as.character(packageVersion("Seurat")))
cat("\n@@RESULT@@\n")
cat(toJSON(res, auto_unbox = TRUE, digits = NA, na = "null"), "\n")
