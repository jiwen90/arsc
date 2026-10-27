# read_uns.R -- read a store's uns (uns.json + files under uns/) back into R values, with their names restored.
# The inverse of the uns part of export_seurat.R (--full): misc / commands / tools / images and the Seurat extras.
#
#   source(file.path(<arsc package folder>, "R", "read_uns.R"))   # folder: python -c "import arsc; print(arsc.__path__[0])"
#   u <- arsc_read_uns("<store dir>")                # the whole uns as a named list
#   m <- arsc_read_uns("<store dir>", "misc/mat_cor")  # one item, by its path of names
#
# Values come back as R types: arrays with dim, names and dimnames (inline JSON, or a .npy / .txt file with its
# sidecar names files <file>.names.txt / <file>.dimnames<k>.txt); factors with their levels; data frames (inline or
# .csv) with row names and factor levels; sparse matrices (.mtx) with dimnames; an R formula or function stays its
# source text (class "arsc_deparsed", never evaluated). Needs jsonlite (Matrix for .mtx, data.table for .csv).

.arsc_npy_read <- function(f) {
  con <- file(f, "rb"); on.exit(close(con))
  magic <- readBin(con, "raw", 6)
  if (!identical(magic[2:6], charToRaw("NUMPY"))) stop("not an .npy file: ", f)
  ver <- as.integer(readBin(con, "raw", 2))
  hlen <- if (ver[1] == 1L) readBin(con, "integer", 1, size = 2, signed = FALSE, endian = "little")
          else readBin(con, "integer", 1, size = 4, endian = "little")
  hdr <- rawToChar(readBin(con, "raw", hlen))
  descr <- sub(".*'descr': *'([^']+)'.*", "\\1", hdr)
  fortran <- grepl("'fortran_order': *True", hdr)
  shp <- trimws(strsplit(sub(".*'shape': *\\(([^)]*)\\).*", "\\1", hdr), ",")[[1]])
  shp <- as.numeric(shp[nzchar(shp)])
  n <- prod(shp)
  end <- if (substr(descr, 1, 1) == ">") "big" else "little"
  code <- substring(descr, 2)
  x <- switch(code,
              f8 = readBin(con, "double", n, size = 8, endian = end),
              f4 = readBin(con, "double", n, size = 4, endian = end),
              i4 = readBin(con, "integer", n, size = 4, endian = end),
              b1 = as.logical(readBin(con, "integer", n, size = 1, signed = FALSE)),
              u1 = readBin(con, "integer", n, size = 1, signed = FALSE),
              stop("unsupported .npy dtype ", descr, " in ", f))
  if (length(shp) > 1) {
    if (fortran) dim(x) <- shp
    else x <- aperm(array(x, rev(shp)))
  }
  x
}

.arsc_lines <- function(f, na = "\\N") {
  v <- readLines(f, encoding = "UTF-8", warn = FALSE)
  v[v == na] <- NA_character_
  v
}

.arsc_set_names <- function(x, j, base) {
  sidecar <- function(rel) if (is.null(rel)) NULL else .arsc_lines(file.path(base, rel))
  dn <- NULL
  if (!is.null(j$dimnames_files)) dn <- lapply(j$dimnames_files, sidecar)
  else if (!is.null(j$dimnames)) dn <- lapply(j$dimnames, function(v) if (is.null(v)) NULL else unlist(v))
  if (!is.null(dn)) {
    if (is.null(dim(x))) dim(x) <- length(x)
    if (length(dn) == length(dim(x))) {
      if (!is.null(j$dimnames_names)) names(dn) <- unlist(j$dimnames_names)
      dimnames(x) <- dn
    }
    if (length(dim(x)) == 1 && is.null(j$dimnames_names)) { nm <- dn[[1]]; dim(x) <- NULL; names(x) <- nm }
    return(x)
  }
  if (!is.null(j$names_file)) names(x) <- sidecar(j$names_file)
  else if (!is.null(j$names)) names(x) <- unlist(j$names)
  x
}

.arsc_uns_value <- function(j, base) {
  if (!is.list(j)) return(j)
  if (!is.null(j[["__ndarray__"]])) {
    v <- j[["__ndarray__"]]
    na <- if (identical(j$dtype, "str")) NA_character_ else if (j$dtype %in% c("bool", "object")) NA else NA_real_
    v <- if (is.list(v)) vapply(v, function(z) if (is.null(z)) na else z, na) else v
    if (identical(j$dtype, "<i4")) v <- as.integer(v)
    if (identical(j$dtype, "<f8")) v <- as.double(v)
    shp <- unlist(j$shape)
    if (!is.null(j$levels)) v <- factor(v, levels = unlist(j$levels))
    if (length(shp) > 1 && prod(shp) == length(v)) dim(v) <- shp
    return(.arsc_set_names(v, j, base))
  }
  if (!is.null(j[["__dataframe__"]])) {
    cols <- lapply(j[["__dataframe__"]], function(c) { v <- .arsc_uns_value(c, base); dim(v) <- NULL; v })
    df <- as.data.frame(cols, stringsAsFactors = FALSE, check.names = FALSE, optional = TRUE)
    if (!is.null(j$index) && length(unlist(j$index)) == nrow(df)) rownames(df) <- unlist(j$index)
    return(df)
  }
  if (!is.null(j[["__npy__"]])) {
    x <- .arsc_npy_read(file.path(base, j[["__npy__"]]))
    if (isTRUE(j$logical)) { na <- is.na(x); x <- as.logical(x); x[na] <- NA }
    return(.arsc_set_names(x, j, base))
  }
  if (!is.null(j[["__txt__"]])) {
    x <- .arsc_lines(file.path(base, j[["__txt__"]]), if (is.null(j$na)) "\\N" else j$na)
    shp <- unlist(j$shape)
    if (length(shp) > 1 && prod(shp) == length(x)) dim(x) <- shp
    return(.arsc_set_names(x, j, base))
  }
  if (!is.null(j[["__csv__"]])) {
    df <- as.data.frame(data.table::fread(file.path(base, j[["__csv__"]]), na.strings = "NA"), stringsAsFactors = FALSE)
    if ("_index" %in% names(df)) { rownames(df) <- as.character(df[["_index"]]); df[["_index"]] <- NULL }
    for (cn in names(j$levels)) if (cn %in% names(df)) df[[cn]] <- factor(df[[cn]], levels = unlist(j$levels[[cn]]))
    return(df)
  }
  if (!is.null(j[["__mtx__"]])) {
    m <- methods::as(Matrix::readMM(file.path(base, j[["__mtx__"]])), "CsparseMatrix")
    dimnames(m) <- list(if (is.null(j$rownames)) NULL else unlist(j$rownames),
                        if (is.null(j$colnames)) NULL else unlist(j$colnames))
    return(m)
  }
  if (!is.null(j[["__deparsed__"]])) return(structure(j[["__deparsed__"]], class = "arsc_deparsed", kind = j$kind))
  lapply(j, .arsc_uns_value, base = base)
}

#' A store's uns as R values (names and dimnames restored). item: a path of names ("misc/mat_cor") or NULL for all.
arsc_read_uns <- function(store, item = NULL) {
  f <- file.path(store, "uns.json")
  if (!file.exists(f)) return(list())
  j <- jsonlite::fromJSON(f, simplifyVector = FALSE)
  if (!is.null(item)) for (k in strsplit(item, "/", fixed = TRUE)[[1]]) {
    if (is.null(j[[k]])) stop("no uns item ", item, " in ", store)
    j <- j[[k]]
  }
  .arsc_uns_value(j, store)
}
