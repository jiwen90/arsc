# Export one MSigDB collection (msigdbr) to csv.gz: gs_name, gene_symbol  (+ gs_collection, gs_subcollection when present)
# usage: Rscript msigdb_export.R <out.csv.gz> <species> <collection> [<subcollection>]
a <- commandArgs(TRUE)
.libPaths(unique(c(file.path(Sys.getenv("AR_ROOT", "E:/autoresearch"), ".Rlib"), .libPaths())))
suppressPackageStartupMessages(library(msigdbr))
newapi <- "collection" %in% names(formals(msigdbr::msigdbr))
args <- list(species = a[2]); args[[if (newapi) "collection" else "category"]] <- a[3]
if (length(a) >= 4 && nzchar(a[4])) args[[if (newapi) "subcollection" else "subcategory"]] <- a[4]
d <- as.data.frame(do.call(msigdbr::msigdbr, args))
keep <- intersect(c("gs_name", "gene_symbol", "gs_collection", "gs_subcollection", "gs_cat", "gs_subcat"), names(d))
d <- unique(d[, keep, drop = FALSE])
dir.create(dirname(a[1]), recursive = TRUE, showWarnings = FALSE)
con <- gzfile(a[1], "w"); write.csv(d, con, row.names = FALSE); close(con)
cat("wrote", a[1], nrow(d), "rows,", length(unique(d$gs_name)), "sets\n")
