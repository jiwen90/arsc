# R references for arsc.clones / arsc.genesets (CLONES_PARITY.md, tests/test_clones.py, benchmarks/bench_clones.py).
# Needs the autoresearch R toolkit (tools/R/ar.R + ar_methods.R: the R implementations that arsc.clones ports), so it
# runs only inside autoresearch. Inputs (read-only, never copied):
#   ARSC_TEST_CLONES_META  CD8 cell metadata (csv.gz) with clone ids and the published Cyclone assignment
#   ARSC_DATA_V3           dataset folder holding tcr/HD/*clonotypes.csv (healthy-donor repertoires for the HD null)
# Usage: Rscript benchmarks/r_reference_clones.R [out_dir]   (default $ARSC_REFS/clones_ref; regen_refs.py clones_ref)
ar_r <- "E:/autoresearch/tools/R/ar.R"; if (!file.exists(ar_r)) stop("needs the autoresearch R toolkit (", ar_r, ")")
source(ar_r); source(file.path(dirname(ar_r), "ar_methods.R"))
args <- commandArgs(trailingOnly = TRUE)
O <- if (length(args)) args[1] else file.path(Sys.getenv("ARSC_REFS", Sys.getenv("ARSC_SCRATCH")), "clones_ref")
dir.create(O, FALSE, TRUE)
V3 <- Sys.getenv("ARSC_DATA_V3"); stopifnot(dir.exists(V3))
mf <- Sys.getenv("ARSC_TEST_CLONES_META"); stopifnot(file.exists(mf))
meta <- data.table::fread(mf,
  select = c("cell","patient_alias","timepoint","treatment","response","clone_id","k","metaclusters_exh","TRA_chain_cdr3_nt","TRB_chain_cdr3_nt"))
meta <- as.data.frame(meta); meta$treatment <- factor(meta$treatment, levels = c("aCTLA-4","aPD1","aPD-1 (naive)","Combination"))
tfile <- file.path(O, "timing.txt"); unlink(tfile)
tlog <- function(tag, t0) cat(tag, round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2), "\n", file = tfile, append = TRUE)
run <- function(tag, ...) {
  t0 <- Sys.time()
  cy <- ar_cyclone(meta, clone_col = "clone_id", patient_col = "patient_alias", time_col = "timepoint", timepoints = c(0,3,6,9), ...)
  tlog(paste0("cyclone_", tag), t0)
  write.csv(unique(cy$clonestats[, c("clone_id","k")]), file.path(O, paste0("cyclone_", tag, "_k.csv")), row.names = FALSE)
  write.csv(cy$counts, file.path(O, paste0("cyclone_", tag, "_counts.csv")), row.names = FALSE)
  write.csv(ar_cyclone_patient_table(cy), file.path(O, paste0("cyclone_", tag, "_ptable.csv")), row.names = FALSE)
  if (tag == "pub") {
    write.csv(as.data.frame(cy$mat_cor), file.path(O, "cyclone_pub_matcor.csv"))
    write.csv(cy$clonestats[, c("clone_id","patient_alias","timepoint","perc","delta_fc","delta_c","n_max")], file.path(O, "cyclone_pub_clonestats.csv"), row.names = FALSE)
    write.csv(data.frame(cell = meta$cell, k = cy$cell_k), file.path(O, "cyclone_pub_cellk.csv"), row.names = FALSE)
    cy_pub <<- cy }
}
run("pub", n_clust = 6, method = "hclust", custom_order = c(4,5,1,6,2,3), outlier_sd = 2.5)
run("noorder", n_clust = 6, method = "hclust")
run("sclone", n_clust = 6, method = "hclust", scale = "clone")
run("snone", method = "hclust", scale = "none", min_c = 0.5, n_clust = 4)
run("pam", n_clust = 6, method = "pam")
run("abs", n_clust = 6, method = "absolute")
run("reg", n_clust = 6, method = "hclust", reg = 0.5, min_fc = 1.5)
st <- ar_cyclone_states(cy_pub, meta, "metaclusters_exh", delta = "fc"); write.csv(st, file.path(O, "cyclone_states_fc.csv"), row.names = FALSE)
# HD-null deterministic pieces: pair test w/o downsampling
mp <- meta[meta$patient_alias == "12-153", ]
c1 <- mp$clone_id[mp$timepoint == 0]; c2 <- mp$clone_id[mp$timepoint == 3]
pt <- .ar_pair_test(c1, c2); pt$fdr <- p.adjust(pt$p, "BH"); write.csv(pt, file.path(O, "pairtest_12-153_0_3.csv"), row.names = FALSE)
hd <- ar_read_10x_clonotypes(list.files(file.path(V3, "tcr", "HD"), "P6 .*clonotypes.csv$", full.names = TRUE))
write.csv(as.data.frame(table(hd$timepoint)), file.path(O, "hd_P6_depth.csv"), row.names = FALSE)
nul <- .ar_pair_test(hd$clone_id[hd$timepoint == "T1"], hd$clone_id[hd$timepoint == "T2"])
write.csv(nul, file.path(O, "pairtest_hd_P6_full.csv"), row.names = FALSE)
meta$clone_pair <- paste(meta$TRA_chain_cdr3_nt, meta$TRB_chain_cdr3_nt); meta$week <- as.character(meta$timepoint)
t0 <- Sys.time()
ex <- ar_clone_expansion_null(meta, hd, clone_col = "clone_pair", time_col = "week", hd_time_col = "timepoint", hd_clone_col = "clone_id", reps = 5, seed = 1)
tlog("expnull_reps5", t0)
write.csv(ex$patients, file.path(O, "expnull_patients.csv"), row.names = FALSE)
ct <- ar_clone_table(meta, "clone_id"); write.csv(ct, file.path(O, "clone_table.csv"), row.names = FALSE)
t0 <- Sys.time(); dy <- ar_clone_dynamics(ct, 0, 3); tlog("dynamics", t0); write.csv(dy, file.path(O, "dynamics_0_3.csv"), row.names = FALSE)
set.seed(1); t0 <- Sys.time(); dv <- ar_clone_diversity(ct, reps = 20); tlog("diversity_reps20", t0); write.csv(dv, file.path(O, "diversity.csv"), row.names = FALSE)
t0 <- Sys.time(); stx <- ar_startrac(meta[meta$week == "3", ], clone_col = "clone_id", state_col = "metaclusters_exh", min_cells = 20)
stm <- ar_startrac(meta, clone_col = "clone_id", state_col = "metaclusters_exh", loc_col = "week", min_cells = 20); tlog("startrac", t0)
write.csv(stx$cluster, file.path(O, "startrac_w3_cluster.csv"), row.names = FALSE); write.csv(stx$pairwise, file.path(O, "startrac_w3_pairwise.csv"), row.names = FALSE)
write.csv(stm$cluster, file.path(O, "startrac_all_cluster.csv"), row.names = FALSE)
cf <- ar_cycling_fraction(meta, meta$metaclusters_exh == "Activated", by = "metaclusters_exh"); write.csv(cf, file.path(O, "cycling_fraction.csv"), row.names = FALSE)
cat("done\n")
