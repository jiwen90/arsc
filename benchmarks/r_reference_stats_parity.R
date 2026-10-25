ar_r <- "E:/autoresearch/tools/R/ar.R"; if (file.exists(ar_r)) source(ar_r)  # autoresearch toolkit, when vendored there
# R references for arsc.de / arsc.stats (STATS_PARITY.md, tests/test_stats_de.py): DESeq2, limma-voom +
# duplicateCorrelation, fgsea (Hallmark, Reactome), lme4 glmer / lmer on the pseudobulk and composition tables of
# the shared input store (benchmarks/stats_parity_inputs.py).
# Usage: Rscript benchmarks/r_reference_stats_parity.R <input_dir> <out_dir> <geneset_dir>
#   input_dir    written by benchmarks/stats_parity_inputs.py (tp_*, ct_*, comp.csv)
#   geneset_dir  holds Homo_sapiens__H__all.csv.gz and Homo_sapiens__C2__CP-REACTOME.csv.gz (arsc.de.msigdb cache)
# benchmarks/regen_refs.py stats_parity runs all of this.
if (nzchar(Sys.getenv("ARSC_RLIB"))) .libPaths(c(Sys.getenv("ARSC_RLIB"), .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(limma); library(edgeR); library(fgsea); library(lme4); library(lmerTest)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
ind <- normalizePath(args[1], winslash = "/"); gsd <- normalizePath(args[3], winslash = "/")
dir.create(args[2], recursive = TRUE, showWarnings = FALSE); setwd(args[2])
H <- read.csv(gzfile(file.path(gsd, "Homo_sapiens__H__all.csv.gz"))); H <- split(H$gene_symbol, H$gs_name)
R2 <- read.csv(gzfile(file.path(gsd, "Homo_sapiens__C2__CP-REACTOME.csv.gz"))); G2 <- split(R2$gene_symbol, R2$gs_name)
run_de <- function(tag, cnt, cd, sel, des_f, var, lvl, ref, vfm, block, coef) {
  keep <- rowSums(cnt[, sel] >= 10) >= 3
  cdp <- cd[sel, ]; for (v in all.vars(des_f)) cdp[[v]] <- factor(make.names(cdp[[v]]))
  dds <- DESeqDataSetFromMatrix(round(cnt[keep & TRUE, sel]), cdp, des_f)
  dds <- DESeq(dds, quiet = TRUE)
  r <- as.data.frame(results(dds, contrast = c(var, make.names(lvl), make.names(ref)))); r$gene <- rownames(r)
  write.csv(r, paste0("R_", tag, "_deseq2.csv"), row.names = FALSE)
  cat(tag, "deseq2:", ncol(dds), "samples", sum(keep), "genes; padj<0.1:", sum(r$padj < 0.1, na.rm = TRUE), "\n")
  y <- DGEList(cnt); des0 <- model.matrix(vfm, cd)
  kp <- filterByExpr(y, min.count = 10, min.total.count = 15, design = des0)
  y <- calcNormFactors(y[kp, , keep.lib.sizes = FALSE]); des <- model.matrix(vfm, cd)
  v <- voom(y, des); dc <- duplicateCorrelation(v, des, block = cd[[block]])
  v <- voom(y, des, block = cd[[block]], correlation = dc$consensus)
  fit <- eBayes(lmFit(v, des, block = cd[[block]], correlation = dc$consensus))
  tt <- topTable(fit, coef = coef, number = Inf); tt$gene <- rownames(tt)
  write.csv(tt, paste0("R_", tag, "_voom.csv"), row.names = FALSE)
  write.csv(data.frame(consensus = dc$consensus, kept = sum(kp), df_prior = fit$df.prior, s2_prior = fit$s2.prior,
                       df_res = fit$df.residual[1]), paste0("R_", tag, "_voom_meta.csv"), row.names = FALSE)
  write.csv(data.frame(sample = colnames(y), norm.factors = y$samples$norm.factors), paste0("R_", tag, "_tmm.csv"), row.names = FALSE)
  cat(tag, "voom: kept", sum(kp), "cor", dc$consensus, "df.prior", fit$df.prior, " padj<.05:", sum(tt$adj.P.Val < .05), "\n")
  st <- setNames(r$stat, r$gene); st <- st[is.finite(st)]; st <- sort(st, decreasing = TRUE)
  set.seed(1)
  for (nm in c("H", "reactome")) {
    fg <- fgsea(if (nm == "H") H else G2, st, minSize = 10, maxSize = 500, nproc = 1)
    fg$leadingEdge <- vapply(fg$leadingEdge, paste, "", collapse = ",")
    write.csv(as.data.frame(fg), paste0("R_", tag, "_fgsea_", nm, ".csv"), row.names = FALSE)
    cat(tag, "fgsea", nm, "sig(padj<.05):", sum(fg$padj < .05, na.rm = TRUE), "\n")
  }
  write.csv(data.frame(gene = names(st), stat = st), paste0("R_", tag, "_stat.csv"), row.names = FALSE)
}
# --- tp: Baseline vs Follow Up 1 in gdT pseudobulk (paired DESeq2 on patients with both; voom/dupCor on all samples)
cnt <- as.matrix(read.csv(file.path(ind, "tp_counts.csv.gz"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(ind, "tp_coldata.csv"), row.names = 1, check.names = FALSE)
pp <- names(which(table(cd$patient_alias) == 2))
run_de("tp", cnt, cd, cd$patient_alias %in% pp, ~ patient_alias + timepoint, "timepoint", "Follow Up 1", "Baseline",
       ~ timepoint, "patient_alias", "timepointFollow Up 1")
# --- ct: gd vs cd8_tem (strong effect) in matched patient x timepoint samples
cnt <- as.matrix(read.csv(file.path(ind, "ct_counts.csv.gz"), row.names = 1, check.names = FALSE))
cd <- read.csv(file.path(ind, "ct_coldata.csv"), row.names = 1, check.names = FALSE)
run_de("ct", cnt, cd, rep(TRUE, nrow(cd)), ~ pt + cell_type, "cell_type", "gd", "cd8_tem", ~ cell_type, "pt", "cell_typegd")
# ---- GLMM + LMM on composition
comp <- read.csv(file.path(ind, "comp.csv"), check.names = FALSE)
res <- list()
for (ct in c("gd", "cd8_tem", "cd4_t_memory")) {
  d <- comp[comp$celltype == ct, ]; d$sample_id <- seq_len(nrow(d))
  d <- d[d$patient_alias %in% names(which(table(d$patient_alias) >= 2)), ]
  d$timepoint <- factor(d$timepoint)
  f <- glmer(cbind(k, n - k) ~ timepoint + (1 | patient_alias) + (1 | sample_id), data = d, family = binomial,
             control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
  co <- as.data.frame(summary(f)$coefficients); co$term <- rownames(co); co$celltype <- ct
  vc <- as.data.frame(VarCorr(f)); co$sd_patient <- vc$sdcor[vc$grp == "patient_alias"]; co$sd_sample <- vc$sdcor[vc$grp == "sample_id"]
  co$logLik <- as.numeric(logLik(f)); res[[ct]] <- co
  d$lp <- qlogis(pmin(pmax(d$prop, 1e-3), 1 - 1e-3))
  l <- lmer(lp ~ timepoint + (1 | patient_alias), data = d)
  cl <- as.data.frame(summary(l)$coefficients); cl$term <- rownames(cl); cl$celltype <- ct
  cl$sd_patient <- attr(VarCorr(l)$patient_alias, "stddev")[[1]]; cl$sd_resid <- sigma(l)
  write.csv(cl, paste0("R_lmer_", ct, ".csv"), row.names = FALSE)
}
write.csv(do.call(rbind, res), "R_glmer.csv", row.names = FALSE)
