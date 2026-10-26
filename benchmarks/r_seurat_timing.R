if (nzchar(Sys.getenv("ARSC_R_PRELUDE"))) source(Sys.getenv("ARSC_R_PRELUDE"))   # a site's R setup, if any
# Seurat side of seurat_timing.py: the Seurat steps on a cell subset of an arsc store, timed in one R process.
#   Rscript r_seurat_timing.R < spec.json      (the spec comes on stdin from seurat_timing.py, never from a file)
# spec: {"store": <store dir>, "rows": [0-based sorted cell rows], "cell_type": [...], "sample": [...],
#        "signatures": {name: [genes]}, "warm_n": 2000, "resolution": 0.6, "k": 20, "npcs": 30, "nhvg": 2000,
#        "markers_budget_s": 0}   (> 0: FindMarkers per cell type in a seeded random order until the budget is spent)
# The counts are read straight from the store's .npy files (CSR: cells x genes, float32 data, int32 indices and
# indptr), read-only, into a dgCMatrix in memory; nothing is written to disk. Every step runs once untimed on the
# first warm_n cells (warm-up), then once timed on all cells. The result goes to stdout as one JSON line after the
# line @@RESULT@@.
# Library path: env ARSC_RLIB (optional, prepended to .libPaths()).
rlib <- Sys.getenv("ARSC_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
t_load0 <- as.numeric(Sys.time())
suppressPackageStartupMessages({ library(Matrix); library(jsonlite); library(Seurat) })
t_load <- as.numeric(Sys.time()) - t_load0
options(future.globals.maxSize = 64 * 1024^3)

spec <- fromJSON(paste(readLines(file("stdin"), warn = FALSE), collapse = "\n"))
N_HVG <- spec$nhvg; N_PC <- spec$npcs; K_NN <- spec$k; RES <- spec$resolution

npy_header <- function(f) {   # numpy .npy v1/v2/v3 header -> data offset, dtype, length
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

read_store_rows <- function(store, rows) {   # rows: 0-based, sorted; returns genes x cells dgCMatrix
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
  # read runs of consecutive rows in blocks of at most 8M values (one seek + read per block)
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

threads_info <- function() {
  blas <- tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) NA)
  omp <- tryCatch(RhpcBLASctl::omp_get_max_threads(), error = function(e) NA)
  list(blas_threads = blas, omp_max_threads = omp, future_plan = class(future::plan())[1],
       future_workers = as.integer(future::nbrOfWorkers()), blas_lib = sessionInfo()$BLAS)
}

steps <- function(o, timed) {   # the Seurat steps in order; returns times and the outputs used for agreement
  t <- list(); r <- list()
  tm <- function(name, expr) {
    t0 <- as.numeric(Sys.time()); v <- force(expr); t[[name]] <<- as.numeric(Sys.time()) - t0
    message(sprintf("[seurat] %s %s: %.2f s (%d cells)", if (timed) "timed" else "warm-up", name, t[[name]], ncol(o)))
    v
  }
  o <- tm("normalize", NormalizeData(o, verbose = FALSE))
  o <- tm("hvg", FindVariableFeatures(o, selection.method = "vst", nfeatures = N_HVG, verbose = FALSE))
  hv <- VariableFeatures(o)
  o <- tm("pca", { o <- ScaleData(o, features = hv, verbose = FALSE); RunPCA(o, features = hv, npcs = N_PC, verbose = FALSE) })
  o <- tm("neighbors", FindNeighbors(o, reduction = "pca", dims = 1:N_PC, k.param = K_NN, verbose = FALSE))
  o <- tm("cluster", FindClusters(o, resolution = RES, algorithm = 1, random.seed = 0, verbose = FALSE))
  cl0 <- as.character(o$seurat_clusters)
  Idents(o) <- "cell_type"
  budget <- if (timed) as.numeric(spec$markers_budget_s) else 0
  if (length(budget) == 0 || is.na(budget) || budget <= 0) {
    mk <- tm("markers", FindAllMarkers(o, test.use = "wilcox", verbose = FALSE))
  } else {
    # FindAllMarkers is FindMarkers(ident.1 = g) for every cell type g, rbind-ed and filtered at p_val < 0.01. Here the
    # cell types run one by one in a seeded random order until the budget is spent; the total is extrapolated.
    set.seed(0); gs <- sample(levels(Idents(o))); per <- list(); mk <- NULL; used <- 0
    for (g in gs) {
      if (sum(Idents(o) == g) < 3) next
      t0 <- as.numeric(Sys.time())
      m1 <- FindMarkers(o, ident.1 = g, test.use = "wilcox", verbose = FALSE)
      dt <- as.numeric(Sys.time()) - t0; used <- used + dt
      message(sprintf("[seurat] timed markers, %s (%d cells): %.1f s (%.0f s of %.0f s budget)", g, sum(Idents(o) == g), dt, used, budget))
      m1 <- m1[order(m1$p_val, -m1[, 2]), , drop = FALSE]   # FindAllMarkers re-sorts by (p_val, -avg_log2FC)
      m1 <- m1[m1$p_val < 0.01, , drop = FALSE]
      if (nrow(m1)) { m1$cluster <- g; m1$gene <- rownames(m1); mk <- rbind(mk, m1) }
      per[[g]] <- list(seconds = dt, cells = sum(Idents(o) == g))
      if (used >= budget) break
    }
    t[["markers"]] <- NA_real_; t[["markers_partial_s"]] <- used
    r$markers_partial <- per
    r$markers_groups_total <- sum(table(Idents(o)) >= 3)
  }
  pb <- tm("pseudobulk", AggregateExpression(o, group.by = c("sample", "cell_type"), slot = "counts", assays = "RNA",
                                             verbose = FALSE)$RNA)
  o <- tm("score_module", AddModuleScore(o, features = unname(spec$signatures), name = "sig", seed = 1))
  if (timed) {
    o2 <- FindClusters(o, resolution = RES, algorithm = 1, random.seed = 1, verbose = FALSE)   # seed-to-seed reference
    sc <- o@meta.data[, paste0("sig", seq_along(spec$signatures))]; colnames(sc) <- names(spec$signatures)
    keep <- seq_len(min(5000, ncol(o)))
    r <- c(r, list(hvg = hv, cluster_seed0 = cl0, cluster_seed1 = as.character(o2$seurat_clusters),
              markers = mk[, c("gene", "cluster", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj")],
              pb_total = sum(pb), pb_dim = dim(pb), pb_colsums = setNames(as.list(colSums(pb)), colnames(pb)),
              score_module_first = as.list(sc[keep, , drop = FALSE])))
  }
  list(t = t, r = r)
}

t_read0 <- as.numeric(Sys.time())
m <- read_store_rows(spec$store, as.integer(spec$rows))
t_read <- as.numeric(Sys.time()) - t_read0
message(sprintf("[seurat] package load %.1f s; read %d cells from the store %.1f s", t_load, ncol(m), t_read))
meta <- data.frame(cell_type = spec$cell_type, sample = spec$sample, row.names = colnames(m))
t_obj0 <- as.numeric(Sys.time())
o <- CreateSeuratObject(m, meta.data = meta, min.cells = 0, min.features = 0)
t_obj <- as.numeric(Sys.time()) - t_obj0
rm(m); invisible(gc())

wn <- min(spec$warm_n, ncol(o))   # R compiles nothing; the warm-up only removes first-call costs
t_w0 <- as.numeric(Sys.time())
w <- steps(subset(o, cells = colnames(o)[seq_len(wn)]), timed = FALSE)
t_warm <- as.numeric(Sys.time()) - t_w0
invisible(gc())
s <- steps(o, timed = TRUE)

res <- list(tool = "seurat", n = ncol(o), times = s$t, warm_times = w$t, warm_n = wn,
            package_load_s = t_load, read_store_s = t_read, create_object_s = t_obj, warmup_s = t_warm,
            threads = threads_info(), r_version = R.version.string,
            versions = list(Seurat = as.character(packageVersion("Seurat")),
                            SeuratObject = as.character(packageVersion("SeuratObject")),
                            Matrix = as.character(packageVersion("Matrix")),
                            limma = if (requireNamespace("limma", quietly = TRUE)) as.character(packageVersion("limma")) else NA,
                            irlba = as.character(packageVersion("irlba")),
                            RcppAnnoy = as.character(packageVersion("RcppAnnoy"))),
            out = s$r)
cat("\n@@RESULT@@\n")
cat(toJSON(res, auto_unbox = TRUE, digits = NA, na = "null"), "\n")
