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
