# Seurat parity map for `arsc`

Which Seurat / SeuratObject functions the research agents and the user's original scripts actually call, and where each
lives in `arsc`. Counts are call sites (`name(` outside comments and `$`/`@` accessors) found by
`scratch/arsc/inv.py` over: `tools/R/*.R` (7 files), `projects/*/experiments/*/code/*.R` (52 files) and
`E:/2023 0129 V2/**/*.R|*.Rmd` (76 files, read-only). Reference implementation: Seurat 4.3.0 / SeuratObject 5.0.2 /
limma 3.58 / presto 1.0 (R 4.3).

## 1. Inventory

Columns: tools / projects / user = call count in that group; status: **P** = ported and R-validated, **N** = not a
compute step (object plumbing; the `arsc.Dataset` + pandas equivalent is noted), **S** = skipped (reason).

| Function | tools | proj. | user | Status / where |
|---|---:|---:|---:|---|
| `subset` | 0 | 8 | 306 | **P** `seurat_extra.which_cells` (+ `Dataset.subset`); pandas query on `obs` |
| `merge` | 6 | 65 | 49 | N (mostly base-R `merge()` on data.frames; Seurat object merge = `pd.concat`/`scipy.sparse.vstack`) |
| `DefaultAssay` | 3 | 1 | 66 | N (`Dataset["RNA"]`) |
| `GetAssayData` | 1 | 6 | 53 | N (`Dataset.X(assay, layer)`) |
| `AddMetaData` | 0 | 0 | 51 | **P** `seurat_extra.add_metadata` |
| `ScaleData` | 0 | 0 | 41 | **P** `preprocessing.scale` (no regression) and `seurat_extra.scale_data` (`vars.to.regress` linear, `split.by`, `do.center/scale`, `scale.max`) |
| `SplitObject` | 0 | 0 | 38 | **P** `seurat_extra.split_object` |
| `Idents` / `RenameIdents` | 0 | 0 | 37 | **P** `seurat_extra.rename_idents` (+ pandas column) |
| `RunPCA` | 0 | 0 | 22 | **P** `preprocessing.pca` |
| `NormalizeData` | 0 | 1 | 17 | **P** `preprocessing.log_normalize` / `clr`; `seurat_extra.normalize_data` (adds RC and a single entry point; CLR margin 1 and 2) |
| `RunUMAP` | 0 | 0 | 17 | **P** `cluster.umap` |
| `DietSeurat` | 0 | 0 | 17 | N (the Dataset only holds what you load) |
| `FindNeighbors` / `FindClusters` | 0 | 0 | 15 / 15 | **P** `neighbors.find_neighbors`, `cluster.cluster` |
| `FindVariableFeatures` / `VariableFeatures` | 0 | 0 | 14 / 7 | **P** `preprocessing.hvg_vst` |
| `Embeddings` | 0 | 0 | 14 | N (`Dataset.embedding`) |
| `AddModuleScore` | 1 | 0 | 12 | **P** `scores.add_module_score` (R-RNG exact), UCell `scores.score_ucell` |
| `RunHarmony` | 0 | 0 | 13 | **P** `harmony.run_harmony` |
| `CreateAssayObject` / `CreateDimReducObject` / `CreateSeuratObject` | 0 | 0 | 11 / 10 / 6 | N (construction of containers) |
| `FindMarkers` | 0 | 0 | 8 | **P** `seurat_extra.find_markers` (wilcox, LR+latent.vars, t; `DESeq2` call (1x) S, see below) |
| `AverageExpression` / `AggregateExpression` | 0 | 0 | 5 / 1 | **P** `seurat_extra.average_expression` (mean of `expm1(data)`, or counts); `pseudobulk.pseudobulk` for sums |
| `CellCycleScoring` | 0 | 1 | 3 | **P** `seurat_extra.cell_cycle_scoring` + `cc_genes` |
| `RunALRA` | 0 | 0 | 4 | S: the user already has `RunALRA.py` (python alra); randomized SVD, not bit-reproducible vs R |
| `Misc`, `Assays`, `Layers`, `Reductions`, `Cells`, `LayerData`, `SetAssayData`, `FetchData` | 9 | 1 | 10 | N (object plumbing) |
| `FindAllMarkers` | 0 | 0 | 2 | **P** `seurat_extra.find_all_markers` |
| `SCTransform` | 0 | 0 | 2 | S: v2 flavour (2 calls, exploratory); needs glmGamPoi-offset NB fits + kernel regularisation + R-RNG gene/cell subsampling. Not worth a partial port; keep LogNormalize+vst or run in R |
| `FindMultiModalNeighbors` | 0 | 0 | 2 | **P** `neighbors.wnn` |
| `Read10X`, `Read10X_h5` | 0 | 0 | 3 | N (`scanpy.read_10x_*` / `arsc.from_anndata`) |
| `RunAzimuth`, `FindTransferAnchors`, `MapQuery`, `IntegrateData`, `SelectIntegrationFeatures` | 0 | 0 | 2+1+1+1+1 | S: one-off calls, reference-based (Azimuth) or CCA anchors; no Python equivalent worth validating |
| `HTODemux` | 0 | 0 | 1 | S: single call |
| (everything else searched: `FindConservedMarkers`, `RunTSNE`, `RunLSI`, `PrepSCTFindMarkers`, `BuildClusterTree`, `PercentageFeatureSet`, `StashIdent`, `JoinLayers`...) | 0 | 0 | 0 | not used |

Priority was by call count; every compute-bearing function with >= 1 call is covered except those marked S.

## 2. What `seurat_extra.py` adds

| Python | Seurat | Notes |
|---|---|---|
| `find_markers(X, groups, ident_1, ident_2, ...)` | `FindMarkers` | `ident.2=None` (vs rest), list `ident_1`, `min_pct`, `min_diff_pct`, `logfc_threshold`, `only_pos`, `max_cells_per_ident`+`random_seed` (R RNG: same cells), `pseudocount_use`, `base`, `norm_method` (`LogNormalize` -> `log(mean(expm1 x)+1)`, otherwise e.g. CLR -> `log(mean x + 1)`), `test_features`, `test_use` in `wilcox` / `LR` (latent.vars: numeric or categorical, `glm.fit` IRLS replica) / `t` (Welch). Output columns/order identical (`p_val, avg_log2FC, pct.1, pct.2, p_val_adj`, sorted by `p_val`, `-avg_log2FC`; Bonferroni with `n_features_total`). |
| `find_all_markers(X, groups, ...)` | `FindAllMarkers` | `return_thresh`, `only_pos`; wilcox path is one rank pass for all clusters. `levels=` sets the cluster order (R `Idents<-"col"` gives first-appearance levels). |
| `fold_change` | `FoldChange` | pct rounded to 3 digits as Seurat. |
| `average_expression` | `AverageExpression` | multi-column `group_by` -> `"a_b"`; layer `data`/`counts`/`scale.data`; `method="aggregate"` gives sums. |
| `cell_cycle_scoring` | `CellCycleScoring` | uses validated `add_module_score`; `cc_genes` (= `Seurat::cc.genes`, what the scripts use) and `cc_genes_updated_2019`. |
| `scale_data` | `ScaleData` | `vars_to_regress` (columns of `latent_data` and/or genes), `split_by`, `do_center`, `do_scale`, `scale_max`; linear model only (`poisson`/`negbinom` raise). |
| `normalize_data` | `NormalizeData` | LogNormalize, CLR (`margin` 1 or 2), RC. |
| `add_metadata`, `which_cells`, `split_object`, `rename_idents` | `AddMetaData`, `WhichCells`/`subset`, `SplitObject`, `RenameIdents` | thin pandas helpers. |

Behaviours worth knowing (all verified against R):
* Seurat 4.3 `FindMarkers(test.use="wilcox")` uses `limma::rankSumTestWithCorrelation`; in current limma this has the
  tie correction, so p-values equal `presto::wilcoxauc`'s (identical formula) but are computed only for genes passing the
  pre-filters, and `p_val_adj` is Bonferroni over **all** assay features.
* `ScaleData(do.center=FALSE)` divides by the root-mean-square, not the s.d. (base `scale` semantics); `scale.max`
  clips the upper tail only.
* `ScaleData(split.by=, vars.to.regress=)` in Seurat 4.3 (`ScaleData.default`) cbind()s the per-split blocks and then
  re-assigns the original cell names; the result is only correct when cells are already grouped by split. `arsc` does the
  mathematically intended thing (per-split regression and scaling). The reference therefore feeds `ScaleData.default` a
  lane-ordered matrix.
* `scale.data` rows are stored by SeuratObject in assay order regardless of the `features` order; `scale_data`
  returns rows in the order of `select`.
* In splits with <= (number of regressors + 1) cells all residuals are exactly zero and the z-score of numerical
  noise is meaningless in both R and Python (excluded in the test).
* `Idents(obj) <- "meta_col"` yields factor levels in order of first appearance; `FindAllMarkers` output order follows that.

## 3. Validation (gdT_1, 38,440 cells; `tests/test_seurat_extra.py`, 17 tests, all pass)

R references: `benchmarks/r_reference_seurat_extra.R` -> `scratch/arsc/ref_extra` (run with
`Rscript benchmarks/r_reference_seurat_extra.R`). Python timings: `benchmarks/bench_seurat_extra.py`
(`benchmarks/results/seurat_extra_timings.json`). Agreement = max |diff| over all rows/cells.

| Check | Rows / size | Result vs R |
|---|---|---|
| FindMarkers wilcox gd FU1 vs Baseline (min.pct .1, lfc .1) | 2,360 genes | same genes, same order; max abs d log10(p) 3.9e-13; avg_log2FC 2.9e-15; pct exact |
| FindMarkers wilcox default thresholds | 157 | order equal; 5.7e-14; 2.2e-15 |
| FindMarkers wilcox vs rest (ident.2 = NULL) | 1,180 | order equal; 3.6e-13; 4.7e-15 |
| FindMarkers wilcox only.pos + max.cells.per.ident=1500 (R RNG) | 358 | order equal (same sampled cells); 5.7e-14 |
| FindMarkers t | 580 | order equal; 8.0e-12 |
| FindMarkers LR, numeric latent (nCount_RNA, percent.mt), 3,000 cells | 602 | order equal; 2.7e-12 |
| FindMarkers LR, factor latent (patient_alias, 20 levels, quasi-separated) | 235 | order equal; 9.6e-11 |
| FindMarkers ADT (CLR, `norm_method="CLR"`) | 3 | exact (weak check, few genes pass) |
| FindAllMarkers only.pos (8 cell types, 36,601 genes) | 736 rows | same gene/cluster sequence; 3.5e-13 |
| AverageExpression data (patient x timepoint), counts, ADT | 36,601 x groups | 5.7e-14 abs (1e-15 rel) |
| CellCycleScoring (cc.genes) | 38,440 cells | scores 5e-15; Phase identical for all cells |
| ScaleData plain / do.center=F,scale.max=3 | 200 genes | 7.5e-12 / 8.9e-16 |
| ScaleData vars.to.regress numeric / factor+numeric | 200 genes | 1.1e-11 / 1.4e-10 |
| ScaleData vars.to.regress + split.by (111 lanes), vs ScaleData.default on ordered matrix | 200 genes | ~1e-12 (cells in splits > 3 cells) |
| NormalizeData CLR margin 1 (ADT), RC | 3,000 cells / 300 genes | within test tolerances 1e-10 / 1e-9 |

Speed (single R process vs numba/threads; same machine):

| Step | R (s) | arsc (s) | speed-up |
|---|---:|---:|---:|
| FindMarkers wilcox gd FU1 vs Baseline | 46.0 | 0.24 | 189x |
| FindMarkers wilcox default | 3.4 | 0.24 | 14x |
| FindMarkers wilcox vs rest | 89.7 | 1.11 | 81x |
| FindMarkers wilcox only.pos max.cells | 4.0 | 0.31 | 13x |
| FindMarkers t | 4.8 | 0.24 | 20x |
| FindMarkers LR (numeric latent, 3k cells) | 16.1 | 0.16 | 102x |
| FindMarkers LR (factor latent) | 34.5 | 1.14 | 30x |
| FindAllMarkers only.pos (8 clusters) | 84.7 | 9.1 | 9x |
| AverageExpression data / counts | 2.3 / 0.4 | 0.73 / 0.23 | 3x / 2x |
| CellCycleScoring | 1.7 | 0.88 | 2x |
| ScaleData regress, 200 genes | 2.2 | 0.40 | 6x |
| ScaleData regress + split.by, 200 genes | 11.9 | 0.50 | 24x |
| ScaleData regress, 2,000 genes (arsc only) | n/a | 7.2 | |

(FindAllMarkers time is dominated by the CSC conversion of the 38k x 36k matrix; pass a gene-subset matrix to make it
much faster.)

## 4. Not ported / caveats

* `FindMarkers(test.use=)`: `DESeq2` (1 call), `MAST` (not installed in R either), `negbinom`, `poisson`, `roc`, `bimod`
  raise `NotImplementedError`. For sample-level DE use `arsc.pseudobulk` + DESeq2/edgeR in R (patient-level stats are the
  project standard anyway).
* `SCTransform`, `RunALRA`, label transfer / integration (`FindTransferAnchors`, `MapQuery`, `IntegrateData`, Azimuth),
  `HTODemux`: see inventory (1-4 calls each).
* `ScaleData` regression models other than linear.
