ar_r <- "E:/autoresearch/tools/R/ar.R"; if (file.exists(ar_r)) source(ar_r)  # autoresearch toolkit, when vendored there
#!/usr/bin/env Rscript
# r_reference_clonotrace.R -- reference outputs of the Clonotrace R code (Fu et al., bioRxiv 2025;
# github.com/yuntianf/Clonotrace, MIT) for tests/test_clonotrace.py.
#
# Environment:
#   CLONOTRACE_SRC   checkout of the Clonotrace repository (its R/ files are sourced; no install needed)
#   ARSC_REFS        output root (old name ARSC_SCRATCH); results go to $ARSC_REFS/clonotrace_ref
#   ARSC_RLIB        optional extra R library (needs Matrix, dbscan, bluster, igraph, transport,
#                    RSpectra, uwot, dplyr, tidyr, future.apply, MASS, splines)
# Input data: the public demo bundled with Clonotrace (inst/extdata/vignette_demo_data.rds; Weinreb et
# al. 2020 in-vitro hematopoiesis, GSE140802). A deterministic subset keeps the run to ~1 minute.
for (p in strsplit(Sys.getenv("ARSC_RLIB"), .Platform$path.sep)[[1]]) if (nzchar(p)) .libPaths(c(p, .libPaths()))
suppressPackageStartupMessages({
  library(Matrix); library(dplyr); library(tidyr); library(igraph); library(future.apply); library(splines); library(rhdf5)
})
src <- Sys.getenv("CLONOTRACE_SRC"); out <- file.path(Sys.getenv("ARSC_REFS", Sys.getenv("ARSC_SCRATCH")), "clonotrace_ref")
stopifnot(nzchar(src), dir.exists(src), nzchar(Sys.getenv("ARSC_REFS", Sys.getenv("ARSC_SCRATCH"))))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
for (f in c("auxiliary.R", "label_propagation.R", "clone_dis.R", "revised_distance.R", "cluster.R",
            "pseudotime.R", "profile_DEG.R", "coembed.R")) source(file.path(src, "R", f))
plan(sequential)
wcsv <- function(x, name, ...) write.csv(x, file.path(out, name), row.names = FALSE, ...)
trip <- function(m, name) { s <- summary(as(m, "CsparseMatrix")); wcsv(data.frame(i = s$i - 1, j = s$j - 1, x = s$x), name) }
dense <- function(m, name) wcsv(as.data.frame(as.matrix(m)), name)

demo <- readRDS(file.path(src, "inst", "extdata", "vignette_demo_data.rds"))
meta <- demo$cell_meta; pca <- demo$pca
sz <- table(meta$clone); big <- names(sz)[sz >= 15]
set.seed(7); keep_clones <- sort(sample(big, 30))
set.seed(8); bg <- sample(which(!(meta$clone %in% keep_clones)), 1200)
cells <- sort(c(which(meta$clone %in% keep_clones), bg))
pca <- pca[cells, ]; meta <- meta[cells, ]
meta$clone_in <- ifelse(meta$clone %in% keep_clones, meta$clone, NA)
wcsv(data.frame(cell = rownames(pca), pca, check.names = FALSE), "pca.csv")
wcsv(data.frame(cell = rownames(meta), clone = meta$clone_in, cluster = as.character(meta$cluster),
                celltype = meta$Cell.type.annotation), "meta.csv")

# 1. cell graph ------------------------------------------------------------------------------
knn <- embedding2knn(as.matrix(pca), k = 30, mode = "connectivity", if_self = FALSE)
trip(knn, "knn_connectivity.csv")
Tm <- compute_transition(knn); trip(Tm, "transition.csv")

# 2. label spreading (blocked, deterministic reference) -------------------------------------
labs <- as.integer(as.factor(meta$clone_in)); C <- max(labs, na.rm = TRUE)
wcsv(data.frame(level = levels(as.factor(meta$clone_in))), "clone_levels.csv")
F <- label_spreading_blocked(Tm, labs, label_n = C, alpha = 0.6, block_size = 8, verbose = FALSE)
dense(F, "spread_raw.csv")
R <- F + 1e-12; R <- R / rowSums(R); dense(R, "spread_norm.csv")
set.seed(11)
boot <- label_spreading_bootstrap(Tm, meta$clone_in, alpha = 0.6, sample_rate = 0.8, sample_n = 12,
                                  backend = "blocked", block_size = 8, verbose = FALSE)
more <- sapply(c(21, 22), function(s) { set.seed(s); label_spreading_bootstrap(Tm, meta$clone_in, alpha = 0.6,
  sample_rate = 0.8, sample_n = 12, backend = "blocked", block_size = 8, verbose = FALSE)$deviance })
wcsv(data.frame(deviance = boot$deviance, deviance21 = more[, 1], deviance22 = more[, 2]), "deviance.csv")

# 3. sparsify ------------------------------------------------------------------------------
keep <- boot$deviance < 0.3
Q0 <- R[keep, ]; Q0 <- Q0 / rowSums(Q0)
wcsv(data.frame(keep = keep), "dev_keep.csv")
Q <- mat_sparsify(Q0, row_mass = 0.9, col_mass = 0.9)
Q <- Q / rowSums(Q); Q[is.na(Q)] <- 0
dense(Q, "Q.csv")
rownames(Q) <- rownames(pca)[keep]; colnames(Q) <- levels(as.factor(meta$clone_in))
Q <- Q[rowSums(Q) > 0, colSums(Q) > 0]
Qs <- Matrix(Q, sparse = TRUE)
wcsv(data.frame(cell = rownames(Qs)), "Q_cells.csv"); wcsv(data.frame(clone = colnames(Qs)), "Q_clones.csv")
emb <- as.matrix(pca[rownames(Qs), ])

# 4. SNN cost graph + clone distances ------------------------------------------------------
g <- bluster::makeSNNGraph(emb, k = 10, type = "number")
g <- .clonotrace_snn_similarity_to_cost(g)
el <- as_edgelist(g, names = FALSE); wcsv(data.frame(i = el[, 1] - 1, j = el[, 2] - 1, w = E(g)$weight), "snn_cost.csv")
nn <- graph_clone_nn(g, Qs, prob_thresh = 0.1, k = 2)
wcsv(as.data.frame(nn), "dist_nn.csv")
Qt <- as.matrix(Qs); Qt[Qt < 0.05] <- 0
D <- distances(g)
ot <- c()
for (a in 1:8) for (b in (a + 1):9) {
  ia <- which(Qt[, a] > 0); ib <- which(Qt[, b] > 0)
  ot <- rbind(ot, c(a, b, clone_2_ot(D[ia, ib, drop = FALSE], Qt[ia, a], Qt[ib, b])))
}
colnames(ot) <- c("group1", "group2", "dis"); wcsv(as.data.frame(ot), "dist_ot.csv")

# 5. revised distance with fixed clusterings --------------------------------------------------
cl <- data.frame(c1 = as.character(meta[rownames(emb), "cluster"]), c2 = meta[rownames(emb), "Cell.type.annotation"],
                 row.names = rownames(emb))
rev <- revise_clone_distance(nn, emb, cell_clone_prob = Qs, clone_names = colnames(Qs), clusters = cl, verbose = FALSE)
dense(unclass(rev)[, ], "dist_revised.csv")
wcsv(data.frame(mean_weight = attr(rev, "mean_weight"), pick1 = attr(rev, "picks")[1], pick2 = attr(rev, "picks")[2]),
     "revised_weight.csv")
dense(attr(rev, "selection_correlations"), "revised_curves.csv")

# 6. clone clustering, embedding, pseudotime ---------------------------------------------------
Dn <- long2square(as.data.frame(nn), "group1", "group2", "dis", symmetric = TRUE)
Dn <- Dn[as.character(seq_len(ncol(Qs))), as.character(seq_len(ncol(Qs)))]; diag(Dn) <- 0
dimnames(Dn) <- list(colnames(Qs), colnames(Qs))
set.seed(1230)
lc <- leiden_dis(Dn, k = 8, resolution = 0.5, if_umap = FALSE)
wcsv(data.frame(clone = names(lc), cluster = as.integer(lc)), "leiden_dis.csv")
cmd <- cmdscale(Dn, k = 6); dense(cmd, "cmdscale.csv")
iso <- MASS::isoMDS(Dn, k = 6, trace = FALSE); dense(iso$points, "isomds.csv")
wcsv(data.frame(stress = iso$stress), "isomds_stress.csv")
emb_c <- cmd; rownames(emb_c) <- colnames(Qs)
cm <- data.frame(clone = meta[rownames(emb), "clone_in"], cluster = as.character(meta[rownames(emb), "cluster"]))
root <- clone_root(colnames(Qs), cm, "clone", "cluster", "0")
tt <- clone_dpt(emb_c, cm, "clone", "cluster", start_cluster = "0", k = 5, dpt_k = 6)
wcsv(data.frame(root = root, t = tt), "clone_dpt.csv")

# 7. profiles, enrichment, fate-gene model -----------------------------------------------------
P <- sparseMatrix(i = seq_along(lc), j = as.integer(lc), x = 1)
cpp <- as.matrix(Qs %*% P); dense(cpp, "cell_profile_prob.csv")
mass <- cluster_profile_mass(cpp, cm$cluster); dense(mass, "profile_mass.csv")
wcsv(data.frame(cluster = rownames(mass)), "profile_mass_rows.csv")
set.seed(3); en <- cluster_profile_enrich(cpp, cm$cluster, permute_n = 200); dense(en$pval, "enrich_pval.csv")
for (df in 1:4) dense(ns(as.numeric(tt[1:20]) + seq(0, 1, length.out = 20), df = df), sprintf("ns_df%d.csv", df))
set.seed(5)
n <- 300; tcell <- sort(runif(n)); Pm <- cbind(runif(n), 0); Pm[, 2] <- 1 - Pm[, 1]
G <- matrix(rpois(40 * n, lambda = 3), nrow = 40); G[1:5, ] <- G[1:5, ] + round(10 * outer(rep(1, 5), tcell * Pm[, 1]))
dense(G, "gam_G.csv"); wcsv(data.frame(t = tcell, p1 = Pm[, 1], p2 = Pm[, 2]), "gam_tP.csv")
for (df in c(1, 3)) {
  fit <- soft_cluster_gam_fit(G, tcell, Pm, df = df, lambda = 1e-4)
  wcsv(fit$stat, sprintf("gam_df%d.csv", df))
}
fitl <- soft_cluster_gam_fit(G, tcell, Pm, df = 2, lambda = 1e-4, test = "LRT"); wcsv(fitl$stat, "gam_lrt_df2.csv")
mfl <- bin_filter_profile_mass(Pm * 3, tcell, thresh = 3, binsize = 0.005)
wcsv(data.frame(flag = mfl), "bin_filter.csv")

# 8. clone-weighted co-embedding graph -----------------------------------------------------------
co <- cell_clone_coembed(emb, emb_c, Qs, cell_k = 15, clone_k = 5)
trip(co, "coembed.csv")
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
cat("done\n")
