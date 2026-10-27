#!/usr/bin/env Rscript
# r_bench_clonotrace.R -- per-step timing of the Clonotrace R code (Fu et al., bioRxiv 2025;
# github.com/yuntianf/Clonotrace, MIT) on the public demo, and the outputs that benchmarks/clonotrace_bench.py
# feeds to arsc (same inputs) and compares (agreement).
#
#   Rscript r_bench_clonotrace.R <size: S|M|L> <out dir> [n_boot] [workers] [steps]
#
# S = the 2,032-cell subset of r_reference_clonotrace.R (30 clones >= 15 cells + 1,200 background cells),
# M = 150 clones >= 15 cells + 3,500 background cells (the published timing size), L = the full demo. n_boot = bootstraps of the deviance
# step (default 48). workers = future workers (default 1: plan(sequential), what the package does unless the user
# sets a plan). steps = comma list (default all). R runs with the BLAS it was built with (reference BLAS on CRAN
# Windows builds). Each step: one timed run; steps under 20 s run 3 times and report all runs. Step start/end
# epoch times are written so an outside sampler can attribute memory.
# Environment: CLONOTRACE_SRC (pinned checkout), CLONOTRACE_SRC_MAIN (optional newer checkout: adds the step
# "nn_main" with that version's graph_clone_nn), ARSC_RLIB (optional extra R library).
for (p in strsplit(Sys.getenv("ARSC_RLIB"), .Platform$path.sep)[[1]]) if (nzchar(p)) .libPaths(c(p, .libPaths()))
suppressPackageStartupMessages({
  library(Matrix); library(dplyr); library(tidyr); library(igraph); library(future.apply); library(splines)
  library(rhdf5); library(jsonlite)
})
a <- commandArgs(TRUE)
size <- a[1]; out <- a[2]
n_boot <- if (length(a) >= 3) as.integer(a[3]) else 48L
workers <- if (length(a) >= 4) as.integer(a[4]) else 1L
steps <- if (length(a) >= 5 && nzchar(a[5])) strsplit(a[5], ",")[[1]] else "all"
src <- Sys.getenv("CLONOTRACE_SRC"); stopifnot(nzchar(src), dir.exists(src))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
for (f in c("auxiliary.R", "label_propagation.R", "clone_dis.R", "exact_ot_disk.R", "revised_distance.R", "cluster.R",
            "pseudotime.R", "profile_DEG.R", "coembed.R")) source(file.path(src, "R", f))
main_src <- Sys.getenv("CLONOTRACE_SRC_MAIN")
if (nzchar(main_src)) { main_env <- new.env(); for (f in c("auxiliary.R", "clone_dis.R")) sys.source(file.path(main_src, "R", f), envir = main_env) }
if (workers > 1) plan(multisession, workers = workers) else plan(sequential)
options(future.globals.maxSize = 8 * 1024^3)
want <- function(s) identical(steps, "all") || s %in% steps
wcsv <- function(x, name) write.csv(x, file.path(out, name), row.names = FALSE)
trip <- function(m, name) { s <- summary(as(m, "CsparseMatrix")); wcsv(data.frame(i = s$i - 1, j = s$j - 1, x = s$x), name) }
wbin <- function(m, name) { m <- as.matrix(m); con <- file(file.path(out, name), "wb"); writeBin(as.vector(m), con); close(con)
                             writeLines(as.character(dim(m)), file.path(out, paste0(name, ".dim"))) }
timing <- list()
now <- function() as.numeric(Sys.time())
tm <- function(name, fn) {
  runs <- c(); starts <- c(); ends <- c(); val <- NULL
  repeat {
    gc(); s <- now(); t0 <- proc.time()[["elapsed"]]
    val <- fn()
    e <- proc.time()[["elapsed"]] - t0; runs <- c(runs, e); starts <- c(starts, s); ends <- c(ends, now())
    if (runs[1] >= 20 || length(runs) >= 3) break
  }
  timing[[name]] <<- list(seconds = runs, start = starts, end = ends)
  message(sprintf("  %-12s %s s", name, paste(round(runs, 3), collapse = " ")))
  val
}

demo <- readRDS(file.path(src, "inst", "extdata", "vignette_demo_data.rds"))
meta <- demo$cell_meta; pca <- demo$pca
if (size %in% c("S", "M")) {
  sz <- table(meta$clone); big <- names(sz)[sz >= 15]
  n_cl <- if (size == "S") 30 else 150; n_bg <- if (size == "S") 1200 else 3500
  set.seed(7); keep_clones <- sort(sample(big, n_cl))
  set.seed(8); bg <- sample(which(!(meta$clone %in% keep_clones)), n_bg)
  cells <- sort(c(which(meta$clone %in% keep_clones), bg))
  pca <- pca[cells, ]; meta <- meta[cells, ]
  meta$clone_in <- ifelse(meta$clone %in% keep_clones, meta$clone, NA)
} else {
  meta$clone_in <- meta$clone
}
wcsv(data.frame(cell = rownames(pca), pca, check.names = FALSE), "pca.csv")
wcsv(data.frame(cell = rownames(meta), clone = meta$clone_in, cluster = as.character(meta$cluster),
                celltype = meta$Cell.type.annotation), "meta.csv")
labs <- as.integer(as.factor(meta$clone_in)); C <- max(labs, na.rm = TRUE)
wcsv(data.frame(level = levels(as.factor(meta$clone_in))), "clone_levels.csv")
message(sprintf("size %s: %d cells, %d clones, %d labeled cells; n_boot %d; workers %d", size, nrow(pca), C,
                sum(!is.na(labs)), n_boot, workers))

# 1-4. cell graph, transition, label spreading, bootstrap deviance -------------------------------------
knn <- tm("knn", function() embedding2knn(as.matrix(pca), k = 30, mode = "connectivity", if_self = FALSE))
trip(knn, "knn.csv")
Tm <- tm("transition", function() compute_transition(knn)); trip(Tm, "transition.csv")
F <- tm("spread", function() label_spreading_blocked(Tm, labs, label_n = C, alpha = 0.6, block_size = 128, verbose = FALSE))
wbin(F, "spread_raw.f64")
R <- F + 1e-12; R <- R / rowSums(R)
dev_file <- file.path(out, "deviance.csv")
if (want("bootstrap")) {
  set.seed(11)
  boot <- tm("bootstrap", function() label_spreading_bootstrap(Tm, meta$clone_in, alpha = 0.6, sample_rate = 0.8,
             sample_n = n_boot, backend = "blocked", block_size = 128, verbose = FALSE, tmpdir = file.path(out, "h5tmp")))
  wcsv(data.frame(deviance = boot$deviance), "deviance.csv")
  dev <- boot$deviance
} else dev <- read.csv(dev_file)$deviance

# 5. sparsify -----------------------------------------------------------------------------------
keep <- dev < 0.3
Q0 <- R[keep, ]; Q0 <- Q0 / rowSums(Q0)
wcsv(data.frame(keep = keep), "dev_keep.csv")
Q <- tm("sparsify", function() mat_sparsify(Q0, row_mass = 0.9, col_mass = 0.9))
Q <- Q / rowSums(Q); Q[is.na(Q)] <- 0
rownames(Q) <- rownames(pca)[keep]; colnames(Q) <- levels(as.factor(meta$clone_in))
Q <- Q[rowSums(Q) > 0, colSums(Q) > 0]
Q <- Q[, colSums(Q >= 0.1) > 0]          # every clone needs a cell at prob_thresh for graph_clone_nn
Q <- Q[rowSums(Q) > 0, ]; Q <- Q / rowSums(Q)
Qs <- Matrix(Q, sparse = TRUE)
wbin(Q, "Q.f64"); wcsv(data.frame(cell = rownames(Qs)), "Q_cells.csv"); wcsv(data.frame(clone = colnames(Qs)), "Q_clones.csv")
emb <- as.matrix(pca[rownames(Qs), ])

# 6-9. SNN cost graph, clone distances -------------------------------------------------------------
g <- tm("snn", function() .clonotrace_snn_similarity_to_cost(bluster::makeSNNGraph(emb, k = 10, type = "number")))
el <- as_edgelist(g, names = FALSE); wcsv(data.frame(i = el[, 1] - 1, j = el[, 2] - 1, w = E(g)$weight), "snn_cost.csv")
nn <- NULL
if (want("nn")) { nn <- tm("nn", function() graph_clone_nn(g, Qs, prob_thresh = 0.1, k = 2)); wcsv(as.data.frame(nn), "dist_nn.csv") }
if (want("nn_main") && nzchar(main_src)) {
  nn2 <- tm("nn_main", function() main_env$graph_clone_nn(g, Qs, prob_thresh = 0.1, k = 2))
  wcsv(as.data.frame(nn2), "dist_nn_main.csv"); if (is.null(nn)) nn <- nn2
}
if (is.null(nn)) nn <- as.matrix(read.csv(file.path(out, "dist_nn.csv")))
if (want("ot")) {
  # the package's default exact path (clone_disance(exact = TRUE) -> disk backend) on the same graph. The legacy
  # in-memory graph_clone_ot is not used: it loops forever when its greedy order reaches the last clone column
  # before the others are done (see CLONOTRACE_PARITY.md).
  ot <- tm("ot", function() {
    Q9 <- Qs[, 1:9]; cc <- rownames(Q9)[Matrix::rowSums(Q9 >= 0.05) > 0]
    unlink(file.path(out, c("ot_store.bin", "ot_ckpt")), recursive = TRUE)
    st <- build_graph_distance_store(g, path = file.path(out, "ot_store.bin"), cells = cc, cell_ids = rownames(emb),
                                     block_size = 256L, overwrite = TRUE, verbose = FALSE)
    graph_clone_ot_disk(st, Q9[cc, , drop = FALSE], prob_thresh = 0.05, target_cell_limit = 5000L, cores = 1L,
                        checkpoint_dir = file.path(out, "ot_ckpt"), overwrite_checkpoint = TRUE, monitor = FALSE,
                        verbose = FALSE)
  })
  ot <- as.data.frame(ot); ot$group1 <- match(as.character(ot[[1]]), colnames(Qs)[1:9], nomatch = 0) + 0
  ot$g1 <- if (all(ot$group1 > 0)) ot$group1 else as.numeric(ot[[1]])
  ot$g2 <- { m <- match(as.character(ot[[2]]), colnames(Qs)[1:9]); if (all(!is.na(m))) m else as.numeric(ot[[2]]) }
  wcsv(data.frame(group1 = ot$g1, group2 = ot$g2, dis = ot[[3]]), "dist_ot.csv")
}
Dn <- long2square(as.data.frame(nn), "group1", "group2", "dis", symmetric = TRUE)
Dn <- Dn[as.character(seq_len(ncol(Qs))), as.character(seq_len(ncol(Qs)))]; diag(Dn) <- 0
dimnames(Dn) <- list(colnames(Qs), colnames(Qs))
if (want("revised")) {
  cl <- data.frame(c1 = as.character(meta[rownames(emb), "cluster"]), c2 = meta[rownames(emb), "Cell.type.annotation"],
                   row.names = rownames(emb))
  rev <- tm("revised", function() revise_clone_distance(nn, emb, cell_clone_prob = Qs, clone_names = colnames(Qs),
                                                         clusters = cl, verbose = FALSE))
  wbin(unclass(rev)[, ], "dist_revised.f64")
  wcsv(data.frame(mean_weight = attr(rev, "mean_weight"), pick1 = attr(rev, "picks")[1], pick2 = attr(rev, "picks")[2]),
       "revised_weight.csv")
}

# 10-13. clone clustering, MDS, root + DPT ---------------------------------------------------------------
set.seed(1230)
lc <- tm("leiden", function() leiden_dis(Dn, k = 8, resolution = 0.5, if_umap = FALSE))
wcsv(data.frame(clone = names(lc), cluster = as.integer(lc)), "leiden_dis.csv")
cmd <- tm("cmdscale", function() cmdscale(Dn, k = 6)); wbin(cmd, "cmdscale.f64")
iso <- tm("isomds", function() MASS::isoMDS(Dn, k = 6, trace = FALSE)); wcsv(data.frame(stress = iso$stress), "isomds_stress.csv")
emb_c <- cmd; rownames(emb_c) <- colnames(Qs)
cm <- data.frame(clone = meta[rownames(emb), "clone_in"], cluster = as.character(meta[rownames(emb), "cluster"]))
dpt_out <- tm("dpt", function() {
  root <- clone_root(colnames(Qs), cm, "clone", "cluster", "0")
  list(root = root, t = clone_dpt(emb_c, cm, "clone", "cluster", start_cluster = "0", k = 5, dpt_k = 6))
})
tt <- dpt_out$t; wcsv(data.frame(root = dpt_out$root, t = tt), "clone_dpt.csv")

# 14-15. profiles, mass, enrichment --------------------------------------------------------------------
cpp <- tm("profiles", function() {
  P <- sparseMatrix(i = seq_along(lc), j = as.integer(lc), x = 1)
  cpp <- as.matrix(Qs %*% P); list(cpp = cpp, mass = cluster_profile_mass(cpp, cm$cluster))
})
wbin(cpp$cpp, "cell_profile_prob.f64"); wbin(cpp$mass, "profile_mass.f64")
wcsv(data.frame(cluster = rownames(cpp$mass)), "profile_mass_rows.csv")
set.seed(3)
en <- tm("enrichment", function() cluster_profile_enrich(cpp$cpp, cm$cluster, permute_n = 200)); wbin(en$pval, "enrich_pval.f64")

# 16. spline fit, F and LRT: 200 synthetic Poisson genes over the Q cells, time = cell pseudotime ----------
n <- nrow(Qs)
tcell <- as.numeric(Qs %*% tt); tcell <- (tcell - min(tcell)) / (max(tcell) - min(tcell))
pr <- cpp$cpp[, order(-colSums(cpp$cpp))[1:2], drop = FALSE] + 1e-6; Pm <- pr / rowSums(pr)
set.seed(5)
G <- matrix(rpois(200 * n, lambda = 3), nrow = 200); G[1:20, ] <- G[1:20, ] + round(10 * outer(rep(1, 20), tcell * Pm[, 1]))
wbin(G, "gam_G.f64"); wcsv(data.frame(t = tcell, p1 = Pm[, 1], p2 = Pm[, 2]), "gam_tP.csv")
gf <- tm("gam_f", function() soft_cluster_gam_fit(G, tcell, Pm, df = 3, lambda = 1e-4)); wcsv(gf$stat, "gam_f.csv")
gl <- tm("gam_lrt", function() soft_cluster_gam_fit(G, tcell, Pm, df = 2, lambda = 1e-4, test = "LRT")); wcsv(gl$stat, "gam_lrt.csv")

# 17. clone-weighted co-embedding graph ---------------------------------------------------------------------
co <- tm("coembed", function() cell_clone_coembed(emb, emb_c, Qs, cell_k = 15, clone_k = 5)); trip(co, "coembed.csv")

info <- list(size = size, n_cells = nrow(pca), n_clones = C, n_labeled = sum(!is.na(labs)), n_Q_cells = nrow(Qs),
             n_Q_clones = ncol(Qs), n_boot = n_boot, workers = workers, steps = timing,
             blas = La_library(), lapack = La_version(), r_version = R.version.string,
             clonotrace_src = src, clonotrace_src_main = main_src, session = capture.output(sessionInfo()))
write_json(info, file.path(out, sprintf("r_timing_w%d.json", workers)), auto_unbox = TRUE, digits = 8, pretty = TRUE)
message("done")
