#!/usr/bin/env Rscript
# R reference outputs + timings for arsc.seurat_extra (Seurat 4.3 / SeuratObject 5.0) on gdT_1 (38,440 cells).
# Usage: Rscript benchmarks/r_reference_seurat_extra.R [rds] [out_dir]
suppressMessages({
  .libPaths(c("E:/autoresearch/.Rlib", .libPaths()))
  library(Matrix); library(Seurat); library(data.table); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
rds <- if (length(args) >= 1) args[1] else "E:/autoresearch/data/2023 0129 V3/gd/gdT_1.rds"
out <- if (length(args) >= 2) args[2] else "E:/autoresearch/scratch/arsc/ref_extra"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(future.globals.maxSize = 64 * 1024^3)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "r_common.R"))
sv <- function(df, name) fwrite(data.table(gene = rownames(df), df), file.path(out, name))

obj <- readRDS(rds)
DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = c("RNA", "TRIMMEDCITE"), counts = TRUE, data = TRUE)
obj <- NormalizeData(obj, verbose = FALSE)
obj <- NormalizeData(obj, assay = "TRIMMEDCITE", normalization.method = "CLR", margin = 2, verbose = FALSE)
cat("cells:", ncol(obj), "\n")
Idents(obj) <- "cell_type"
hvg <- readLines("E:/autoresearch/scratch/arsc/ref_gd/variable_features.txt")

# ---- FindMarkers: wilcox (user pattern: Follow Up 1 vs Baseline within one cell type) -------------------
gdc <- WhichCells(obj, idents = "gd")
sub <- subset(obj, cells = gdc)
Idents(sub) <- "timepoint"
tm("FindMarkers_wilcox_t01", fm1 <- FindMarkers(sub, ident.1 = "Follow Up 1", ident.2 = "Baseline", min.pct = 0.1,
                                                logfc.threshold = 0.1, verbose = FALSE))
sv(fm1, "fm_wilcox_gd_fu1_bl.csv")
tm("FindMarkers_wilcox_default", fm2 <- FindMarkers(sub, ident.1 = "Follow Up 2", ident.2 = "Baseline", verbose = FALSE))
sv(fm2, "fm_wilcox_default.csv")
tm("FindMarkers_wilcox_onlypos_maxcells", fm3 <- FindMarkers(sub, ident.1 = "Follow Up 3", ident.2 = "Follow Up 1", only.pos = TRUE,
                                                      min.pct = 0.25, max.cells.per.ident = 1500, random.seed = 7, verbose = FALSE))
sv(fm3, "fm_wilcox_maxcells.csv")
# vs-rest (ident.2 = NULL)
tm("FindMarkers_vs_rest", fm4 <- FindMarkers(sub, ident.1 = "Baseline", min.pct = 0.1, logfc.threshold = 0.1, verbose = FALSE))
sv(fm4, "fm_wilcox_vs_rest.csv")
# t-test
tm("FindMarkers_t", fm5 <- FindMarkers(sub, ident.1 = "Follow Up 1", ident.2 = "Baseline", test.use = "t", min.pct = 0.25,
                                       logfc.threshold = 0.25, verbose = FALSE))
sv(fm5, "fm_t.csv")
# LR with latent vars on a cell subsample (glm per gene is slow in R)
set.seed(3)
lrcells <- sample(colnames(sub)[sub$timepoint %in% c("Follow Up 1", "Baseline")], 3000)
writeLines(lrcells, file.path(out, "lr_cells.txt"))
lrobj <- subset(sub, cells = lrcells)
tm("FindMarkers_LR_latent", fm6 <- FindMarkers(lrobj, ident.1 = "Follow Up 1", ident.2 = "Baseline", test.use = "LR",
                                               latent.vars = c("nCount_RNA", "percent.mt"), min.pct = 0.25,
                                               logfc.threshold = 0.25, verbose = FALSE))
sv(fm6, "fm_lr_latent.csv")
tm("FindMarkers_LR_factor", fm7 <- FindMarkers(lrobj, ident.1 = "Follow Up 1", ident.2 = "Baseline", test.use = "LR",
                                               latent.vars = c("patient_alias"), min.pct = 0.3,
                                               logfc.threshold = 0.4, verbose = FALSE))
sv(fm7, "fm_lr_factor.csv")
# ADT (CLR) FindMarkers, user pattern
Idents(obj) <- "timepoint"
tm("FindMarkers_ADT", fm8 <- FindMarkers(obj, assay = "TRIMMEDCITE", ident.1 = "Follow Up 1", ident.2 = "Baseline",
                                         min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE))
sv(fm8, "fm_adt.csv")

# ---- FindAllMarkers (wilcox, only.pos) --------------------------------------------------------------------
Idents(obj) <- "cell_type"
big <- names(which(table(obj$cell_type) >= 200))
fam_obj <- subset(obj, cells = colnames(obj)[obj$cell_type %in% big])
Idents(fam_obj) <- "cell_type"
writeLines(big, file.path(out, "fam_types.txt"))
tm("FindAllMarkers", fam <- FindAllMarkers(fam_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, verbose = FALSE))
fwrite(as.data.table(fam), file.path(out, "fam_onlypos.csv"))
rm(fam_obj); gc()

# ---- AverageExpression -----------------------------------------------------------------------------------
tm("AverageExpression_data", ae <- AverageExpression(obj, assays = "RNA", group.by = c("patient_alias", "timepoint"),
                                                     slot = "data", verbose = FALSE)$RNA)
write_dense(ae, file.path(out, "avgexp_data"))
tm("AverageExpression_counts", ae2 <- AverageExpression(obj, assays = "RNA", group.by = "cell_type", slot = "counts",
                                                        verbose = FALSE)$RNA)
write_dense(ae2, file.path(out, "avgexp_counts"))
ae3 <- AverageExpression(obj, assays = "TRIMMEDCITE", group.by = "cell_type", slot = "data", verbose = FALSE)[[1]]
write_dense(ae3, file.path(out, "avgexp_adt"))

# ---- CellCycleScoring ------------------------------------------------------------------------------------
tm("CellCycleScoring", cc <- CellCycleScoring(obj, s.features = cc.genes$s.genes, g2m.features = cc.genes$g2m.genes,
                                              set.ident = FALSE))
fwrite(data.table(cell = colnames(cc), cc@meta.data[, c("S.Score", "G2M.Score", "Phase")]), file.path(out, "cellcycle.csv"))
rm(cc); gc()

# ---- ScaleData with vars.to.regress / split.by -------------------------------------------------------------
feat <- hvg[1:200]
writeLines(feat, file.path(out, "scale_features.txt"))
sc0 <- ScaleData(obj, features = feat, verbose = FALSE)
write_dense(GetAssayData(sc0, slot = "scale.data"), file.path(out, "scale_plain"))
tm("ScaleData_regress_200", sc1 <- ScaleData(obj, features = feat, vars.to.regress = c("nCount_RNA", "percent.mt"), verbose = FALSE))
write_dense(GetAssayData(sc1, slot = "scale.data"), file.path(out, "scale_regress_num"))
sc2 <- ScaleData(obj, features = feat, vars.to.regress = c("patient_alias", "nFeature_RNA"), verbose = FALSE)
write_dense(GetAssayData(sc2, slot = "scale.data"), file.path(out, "scale_regress_factor"))
# NB: ScaleData.default (4.3) cbind()s the per-split blocks and then re-assigns the ORIGINAL cell names, which
# scrambles cells unless the columns are already grouped by split.by (and Seurat's `[`/subset() always restores the
# original cell order, so the object route cannot be used). The reference therefore calls ScaleData.default on a
# lane-ordered matrix (the mathematically intended result) and writes columns back in original order.
ord <- order(obj$n10x_lane_name)
mm <- GetAssayData(obj, slot = "data")[feat, ord]
tm("ScaleData_regress_split_200", sc3 <- ScaleData(mm, features = feat, vars.to.regress = "percent.mt",
    latent.data = obj@meta.data[ord, "percent.mt", drop = FALSE], split.by = as.character(obj$n10x_lane_name)[ord], verbose = FALSE))
write_dense(sc3[, colnames(obj)], file.path(out, "scale_regress_split"))
rm(mm, sc3)
sc4 <- ScaleData(obj, features = feat, do.center = FALSE, scale.max = 3, verbose = FALSE)
write_dense(GetAssayData(sc4, slot = "scale.data"), file.path(out, "scale_nocenter_max3"))
rm(sc0, sc1, sc2, sc4); gc()

# ---- NormalizeData variants --------------------------------------------------------------------------------
cnt <- GetAssayData(obj, assay = "TRIMMEDCITE", slot = "counts")
tmp <- CreateSeuratObject(cnt, assay = "ADT")
n1 <- NormalizeData(tmp, normalization.method = "CLR", margin = 1, verbose = FALSE)
write_dense(as.matrix(GetAssayData(n1, slot = "data")[, 1:3000]), file.path(out, "clr_margin1"))
rc <- NormalizeData(CreateSeuratObject(GetAssayData(obj, assay = "RNA", slot = "counts")[, 1:2000]),
                    normalization.method = "RC", scale.factor = 1e4, verbose = FALSE)
write_dense(as.matrix(GetAssayData(rc, slot = "data")[1:300, ]), file.path(out, "rc_block"))
writeLines(rownames(GetAssayData(rc, slot = "data"))[1:300], file.path(out, "rc_features.txt"))

write_timings(out)
cat("done\n")
