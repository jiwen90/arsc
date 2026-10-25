# Shared helpers for the R reference scripts.
.timings <- list()
tm <- function(name, expr) {
  t0 <- proc.time()[["elapsed"]]
  force(expr)
  dt <- proc.time()[["elapsed"]] - t0
  .timings[[name]] <<- dt
  cat(sprintf("[timing] %-28s %8.2f s\n", name, dt))
  invisible(dt)
}
write_timings <- function(out) {
  jsonlite::write_json(.timings, file.path(out, "timings_r.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
}
# Dense matrix -> raw little-endian float64 (column-major) + dims/dimnames sidecar.
write_dense <- function(m, prefix) {
  m <- as.matrix(m)
  writeBin(as.vector(m), paste0(prefix, ".f64"), size = 8)
  jsonlite::write_json(list(dim = dim(m), rownames = rownames(m), colnames = colnames(m)),
                       paste0(prefix, ".json"), digits = NA)
}
# Compact variant: gzip-compressed float32 (size = 4) or float64 (size = 8); read by refutil.read_dense.
write_dense_gz <- function(m, prefix, size = 8) {
  m <- as.matrix(m)
  con <- gzfile(paste0(prefix, if (size == 4) ".f32.gz" else ".f64.gz"), "wb")
  writeBin(as.vector(m), con, size = size)
  close(con)
  jsonlite::write_json(list(dim = dim(m), rownames = rownames(m), colnames = colnames(m)),
                       paste0(prefix, ".json"), digits = NA)
}
# Reads write_dense (.f64) or write_dense_gz (.f64.gz / .f32.gz) output as a matrix.
read_dense_any <- function(prefix) {
  dm <- jsonlite::fromJSON(paste0(prefix, ".json"))$dim
  for (ext in c(".f64", ".f64.gz", ".f32.gz")) {
    f <- paste0(prefix, ext)
    if (!file.exists(f)) next
    con <- if (grepl("gz$", ext)) gzfile(f, "rb") else file(f, "rb")
    x <- readBin(con, "double", n = prod(dm), size = if (ext == ".f32.gz") 4 else 8)
    close(con)
    return(matrix(x, dm[1], dm[2]))
  }
  stop("no matrix at ", prefix)
}
