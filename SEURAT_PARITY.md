# Seurat parity map for `arsc`

Which Seurat / SeuratObject functions a reference workload of R analysis code actually calls, and where each
lives in `arsc`. Counts are call sites (`name(` outside comments and `$`/`@` accessors) found by a static scan of
three groups of R code: shared helper scripts (7 files), analysis-pipeline scripts (52 files) and an
investigator's original analysis scripts (76 `.R`/`.Rmd` files). Reference implementation: Seurat 4.3.0 / SeuratObject 5.0.2 /
limma 3.58 / presto 1.0 (R 4.3).

## 1. Inventory

Columns: tools (helper scripts) / proj. (pipeline scripts) / user (original analysis scripts) = call count in that group; status: **P** = ported and R-validated, **N** = not a
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
| `RunALRA` | 0 | 0 | 4 | S: a Python ALRA implementation already exists; randomized SVD, not bit-reproducible vs R |
| `Misc`, `Assays`, `Layers`, `Reductions`, `Cells`, `LayerData`, `SetAssayData`, `FetchData` | 9 | 1 | 10 | N (object plumbing) |
| `FindAllMarkers` | 0 | 0 | 2 | **P** `seurat_extra.find_all_markers` |
| `SCTransform` | 0 | 0 | 2 | S: v2 flavor (2 calls, exploratory); needs glmGamPoi-offset NB fits + kernel regularization + R-RNG gene/cell subsampling. Not worth a partial port; keep LogNormalize+vst or run in R |
| `FindMultiModalNeighbors` | 0 | 0 | 2 | **P** `neighbors.wnn` |
| `Read10X`, `Read10X_h5` | 0 | 0 | 3 | N (`arsc.read_10x_h5` / `arsc.read_10x_mtx`) |
| `RunAzimuth`, `FindTransferAnchors`, `MapQuery`, `IntegrateData`, `SelectIntegrationFeatures` | 0 | 0 | 2+1+1+1+1 | S: one-off calls, reference-based (Azimuth) or CCA anchors; no Python equivalent worth validating |
| `HTODemux` | 0 | 0 | 1 | S: single call |
| (everything else searched: `FindConservedMarkers`, `RunTSNE`, `RunLSI`, `PrepSCTFindMarkers`, `BuildClusterTree`, `PercentageFeatureSet`, `StashIdent`, `JoinLayers`...) | 0 | 0 | 0 | not used |

Priority was by call count; every compute-bearing function with >= 1 call is covered except those marked S.

## 2. What `seurat_extra.py` adds

| Python | Seurat | Notes |
|---|---|---|
| `find_markers(X, groups, ident_1, ident_2, ...)` | `FindMarkers` | `ident.2=None` (vs rest), list `ident_1`, `min_pct`, `min_diff_pct`, `logfc_threshold`, `only_pos`, `max_cells_per_ident`+`random_seed` (R RNG: same cells), `pseudocount_use`, `base`, `norm_method` (`LogNormalize` -> `log(mean(expm1 x)+1)`, otherwise e.g. CLR -> `log(mean x + 1)`), `test_features`, `test_use` in `wilcox` / `LR` (latent.vars: numeric or categorical, `glm.fit` IRLS replica) / `t` (Welch). Output columns/order identical (`p_val, avg_log2FC, pct.1, pct.2, p_val_adj`, sorted by `p_val` with ties in feature order, as Seurat 4.3's `order(p_val, -de.results[, 1])` whose column 1 is `p_val`; `find_all_markers` re-sorts each cluster by `p_val`, `-avg_log2FC` as `FindAllMarkers` does; Bonferroni with `n_features_total`). |
| `find_all_markers(X, groups, ...)` | `FindAllMarkers` | `return_thresh`, `only_pos`; wilcox path is one rank pass for all clusters. `levels=` sets the cluster order (R `Idents<-"col"` gives first-appearance levels). |
| `fold_change` | `FoldChange` | pct rounded to 3 digits as Seurat. |
| `average_expression` | `AverageExpression` | multi-column `group_by` -> `"a_b"`; layer `data`/`counts`/`scale.data`; `method="aggregate"` gives sums. |
| `cell_cycle_scoring` | `CellCycleScoring` | uses validated `add_module_score`; `cc_genes` (= `Seurat::cc.genes`) and `cc_genes_updated_2019`. |
| `scale_data` | `ScaleData` | `vars_to_regress` (columns of `latent_data` and/or genes), `split_by`, `do_center`, `do_scale`, `scale_max`; linear model only (`poisson`/`negbinom` raise). |
| `normalize_data` | `NormalizeData` | LogNormalize, CLR (`margin` 1 or 2), RC. |
| `add_metadata`, `which_cells`, `split_object`, `rename_idents` | `AddMetaData`, `WhichCells`/`subset`, `SplitObject`, `RenameIdents` | thin pandas helpers. |

Behaviors worth knowing (all verified against R):
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

## 3. Validation (a 38,440-cell gamma-delta T-cell CITE-seq dataset; `tests/test_seurat_extra.py`, 17 tests, all pass)

R references: `benchmarks/r_reference_seurat_extra.R` -> `$ARSC_REFS/ref_extra` (`python benchmarks/regen_refs.py ref_extra`) (run with
`Rscript benchmarks/r_reference_seurat_extra.R`). Python timings: `benchmarks/bench_seurat_extra.py`
(`benchmarks/results/seurat_extra_timings.json`). Agreement = max |diff| over all rows/cells.

| Check | Rows / size | Result vs R |
|---|---|---|
| FindMarkers wilcox, gamma-delta cells, follow-up vs baseline (min.pct .1, lfc .1) | 2,360 genes | same genes, same order; max abs d log10(p) 3.9e-13; avg_log2FC 2.9e-15; pct exact |
| FindMarkers wilcox default thresholds | 157 | order equal; 5.7e-14; 2.2e-15 |
| FindMarkers wilcox vs rest (ident.2 = NULL) | 1,180 | order equal; 3.6e-13; 4.7e-15 |
| FindMarkers wilcox only.pos + max.cells.per.ident=1500 (R RNG) | 358 | order equal (same sampled cells); 5.7e-14 |
| FindMarkers t | 580 | order equal; 8.0e-12 |
| FindMarkers LR, numeric latent (nCount_RNA, percent.mt), 3,000 cells | 602 | order equal; 2.7e-12 |
| FindMarkers LR, factor latent (patient, 20 levels, quasi-separated) | 235 | order equal; 9.6e-11 |
| FindMarkers ADT (CLR, `norm_method="CLR"`) | 3 | exact (weak check, few genes pass) |
| FindAllMarkers only.pos (8 cell types, 36,601 genes) | 736 rows | same gene/cluster sequence; 3.5e-13 |
| AverageExpression data (patient x timepoint), counts, ADT | 36,601 x groups | 5.7e-14 abs (1e-15 rel) |
| CellCycleScoring (cc.genes) | 38,440 cells | scores 5e-15; Phase identical for all cells |
| ScaleData plain / do.center=F,scale.max=3 | 200 genes | 7.5e-12 / 8.9e-16 |
| ScaleData vars.to.regress numeric / factor+numeric | 200 genes | 1.1e-11 / 1.4e-10 |
| ScaleData vars.to.regress + split.by (111 lanes), vs ScaleData.default on ordered matrix | 200 genes | ~1e-12 (cells in splits > 3 cells) |
| NormalizeData CLR margin 1 (ADT), RC | 3,000 cells / 300 genes | within test tolerances 1e-10 / 1e-9 |

Speed: section 6. Earlier single-process timings on this dataset: `benchmarks/results/seurat_extra_timings.json`
(arsc) and the git history of this file.

## 4. Not ported / caveats

* `FindMarkers(test.use=)`: `DESeq2` (1 call), `MAST` (not installed in R either), `negbinom`, `poisson`, `roc`, `bimod`
  raise `NotImplementedError`. For sample-level DE use `arsc.pseudobulk` + DESeq2/edgeR in R (the patient, not the cell, is the
  unit for condition comparisons).
* `SCTransform`, `RunALRA`, label transfer / integration (`FindTransferAnchors`, `MapQuery`, `IntegrateData`, Azimuth),
  `HTODemux`: see inventory (1-4 calls each).
* `ScaleData` regression models other than linear.

## 5. arsc's own graph engines: kNN, UMAP, Leiden / Louvain

Measured 2026-10-21 on a shared 16-core machine (load 40-90% from other jobs; times are indicative). Data: a real
flow-cytometry population, 80,000 events (15 PCs; 30 channels for kNN), not distributed; synthetic Gaussian mixtures
and `arsc.datasets.toy`. Scripts: `benchmarks/umap_parity.py`, `benchmarks/leiden_parity.py`; tests:
`tests/test_umap_knn.py`, `tests/test_leiden.py`.

### kNN (`arsc.knn`): arsc 0.3's own exact search (since replaced)

Since the scanpy-matching switch (CHANGELOG, Unreleased) `arsc.knn` searches exactly with scikit-learn brute force
up to 4,096 cells and with pynndescent above; the table below documents the partition search it replaced.

The exact search scans only the ring of each k-means cluster that can hold neighbors (points sorted by distance to
their center) and resolves ties at the k-th place by (distance, index), so the result no longer depends on the
partition seed or the scan order (on data rounded to one decimal, the old code gave 2 different results across
partition seeds and threads; now 1). On the real data the output is identical to the previous version (indices and
distances).

| 80,000 events, k = 15 | 15 PCs | 30 channels |
|---|---|---|
| recall@15 of `exact` vs brute force (2,000 queries) | 1.0000 | 1.0000 |
| recall@15 of `method="pynndescent"` (seed 0) | 0.978 | 0.945 |
| seconds, exact, 1 thread: before / after | 18.1 / 12.4 | 57.7 / 47.7 |
| seconds, exact, 8 threads (after) | 4.0 | about 10 |

pynndescent misses 2-5% of the neighbors on flow data (below a 0.99 recall bar); flow callers that need exact neighbors pass `method="exact"`.
Tree search is no alternative on these data (scikit-learn KDTree, 15 PCs: 124 s on one thread).

### UMAP (`arsc.umap(engine="arsc")`)

The same exact kNN for both engines; umap-learn 0.5.12's layout run seeded (single-threaded) on the same graph.
Median (min-max) over seeds 0-4; trustworthiness on a 5,000-event subsample with 15 neighbors; kNN preservation =
share of each event's 15 input neighbors among its 15 layout neighbors; disparity = Procrustes disparity.

| | real, 80,000 x 15 PCs | synthetic, 20,000 x 10, 8 separate blobs |
|---|---|---|
| fuzzy graph vs umap-learn | same edges, max weight difference 0.0 | same edges, 0.0 |
| a, b (min_dist 0.3) | equal to umap-learn | equal |
| spectral init vs umap-learn, abs(r) per axis | 0.999999998, 0.999999999 | 8 graph components: placement differs (0.0005, 0.14) |
| trustworthiness arsc / umap-learn | 0.9203 (0.9196-0.9205) / 0.9214 (0.9204-0.9215) | 0.9584 (0.9577-0.9588) / 0.9587 (0.9585-0.9591) |
| kNN preservation arsc / umap-learn | 0.0293 (0.0292-0.0296) / 0.0292 (0.0291-0.0293) | 0.0473 (0.0470-0.0489) / 0.0488 (0.0483-0.0494) |
| disparity arsc-arsc / ref-ref / arsc-ref | 0.0023 / 0.0036 / 0.0144 | 0.42 / 0.69 / 0.85 |
| identical coordinates at 1, 2, 4, 8 threads | yes | yes |

Reading: the graph and the initialization are umap-learn's, and the layouts reach the same quality (trustworthiness
within 0.1%, kNN preservation equal). On the real data two umap-learn seeds differ less from each other (0.004) than
from arsc (0.014): there the seed changes only the noise and the negative samples (the spectral start is the same),
and arsc's epoch differs systematically from umap-learn's sequential one (other points read from the epoch's
snapshot, tail moves added at the end of the epoch). The difference is small: 1.4% of the layout variance is not
explained after rotation and scaling. On the 8-blob data the components' placement varies with the seed in both
engines.

Seconds for the layout of the real 80,000 events (200 epochs, spectral start included): umap-learn 48-62 s
(single-threaded at any thread count when seeded); arsc 24.7 s on 1 thread and about 5 s on 8, identical output at
every thread count. umap-learn's own parallel mode (no seed) updates shared coordinates without locks ("Hogwild") and
differs from run to run.

### Leiden / Louvain (`arsc.cluster(engine="arsc")`)

Ports of libleidenalg's default optimizer (queue-based local moving, randomized refinement, aggregation on the refined
partition, the `n_iterations` loop) and of igraph `community_multilevel`; modularity with resolution and
`objective="cpm"`. Median (min-max) over seeds 0-9, ARI over seed pairs; Leiden `n_iterations=2`, Louvain
`n_start=10`.

| graph | method | quality arsc | quality reference | clusters arsc / ref | ARI arsc-arsc | ARI ref-ref | ARI arsc-ref | s arsc / ref (1 ; 8 threads) |
|---|---|---|---|---|---|---|---|---|
| planted 8x1000 | leiden modularity 0.5 | 0.72807 (all) | 0.72807 (all) | 8 / 8 | 1.000 | 1.000 | 1.000 | 0.11 / 0.66 ; 0.11 / 0.56 |
| planted 8x1000 | leiden cpm 0.005 | 278449.5 (all) | 278449.5 (all) | 8 / 8 | 1.000 | 1.000 | 1.000 | 0.12 / 0.60 ; 0.12 / 0.72 |
| planted 8x1000 | louvain modularity 0.5 | 0.72807 (all) | 0.72807 (all) | 8 / 8 | 1.000 | 1.000 | 1.000 | 0.61 / 1.69 ; 0.15 / 1.85 |
| toy SNN k20 | leiden modularity 0.5 | 0.78554 (0.78335-0.78709) | 0.78402 (0.77778-0.78642) | 6 / 6 | 0.701 (0.564-0.995) | 0.722 (0.554-0.997) | 0.711 (0.553-1.000) | 0.04 / 0.25 ; 0.04 / 0.27 |
| toy SNN k20 | leiden cpm 0.05 | 6636.6 (6607.7-6674.2) | 6615.9 (6569.9-6679.4) | 191 / 190.5 | 0.720 (0.670-0.783) | 0.691 (0.639-0.770) | 0.712 (0.651-0.790) | 0.05 / 0.29 ; 0.05 / 0.29 |
| toy SNN k20 | louvain modularity 0.5 | 0.78514 (0.78394-0.78572) | 0.78511 (0.78238-0.78511) | 6 / 6 | 0.952 (0.891-0.988) | 1.000 (0.721-1.000) | 0.948 (0.672-0.995) | 0.69 / 2.12 ; 0.14 / 2.17 |
| flow SNN k15 (80,000 x 15 PCs) | leiden modularity 0.5 | 0.86896 (0.86369-0.87022) | 0.86689 (0.86533-0.86964) | 9 / 10 | 0.915 (0.791-0.964) | 0.888 (0.812-0.958) | 0.894 (0.778-0.972) | 2.38 / 12.97 ; 2.88 / 12.85 |
| flow SNN k15 | leiden cpm 0.05 | 68604.5 (68498.1-68666.9) | 68612.6 (68536.9-68678.9) | 10076 / 10081 | 0.685 (0.677-0.692) | 0.687 (0.678-0.697) | 0.686 (0.673-0.695) | 3.64 / 17.02 ; 3.65 / 17.65 |
| flow SNN k15 | louvain modularity 0.5 | 0.86851 (0.86739-0.86896) | 0.86810 (0.86795-0.86889) | 10 / 10 | 0.943 (0.917-0.964) | 0.948 (0.928-1.000) | 0.946 (0.904-0.968) | 99.0 / 258.7 ; 20.0 / 305.3 |

Quality is within or above the reference's seed range in every row, and the ARI of arsc against the reference lies
within the reference's own seed-to-seed range. Labels are identical at 1, 2, 4 and 8 threads (Leiden is sequential;
the parallel Louvain starts are chosen by (modularity, start index)). The Louvain times predate a 15% speedup per
start. A seed does not reproduce leidenalg's partition (numpy `SeedSequence` streams).

**Leiden slower with more threads (arcyto: 18 s at 1 thread, 68 s at 8).** Run alone, the old path does not slow
down with threads (total 42.9 / 27.7 / 22.9 / 18.1 s at 1 / 2 / 4 / 8 threads: kNN 26.2 to 4.6 s, then about 2.5 s of
igraph construction and 12 s of leidenalg, sequential at every count). The 8-thread figure came from a fully loaded
machine, where 14 s of sequential work and parallel regions waiting for their slowest thread suffer most. The new
engine removes the sequential bottleneck: kNN + SNN + graph + Leiden = 26.6 / 15.9 / 9.9 / 7.0 s, Leiden 2.6 s.

**Defaults.** Since 0.3.0, `arsc.umap` and `arsc.cluster` run `engine="arsc"` by default (it needs neither
umap-learn, leidenalg nor igraph), and `arsc.cluster` defaults to Louvain at resolution 0.6. Before the switch the
fast suite and both parity benchmarks were re-run on the release candidate (2026-10-21; synthetic and toy data, 1 and
2 threads; the tables above were measured on real data and are unchanged): see the re-check below.
`engine="umap-learn"` / `"leidenalg"` stay available and reproduce results recorded with 0.2.x (whose defaults they
were, with Leiden at resolution 0.8 and 50 PCs).

Re-check before the default switch (2026-10-21, on the commit that added the engines; 2 threads available on a
shared machine, so thread determinism was checked at 1 and 2 threads):

| check | result |
|---|---|
| fast suite (`pytest -m "not slow"`) | 420 passed, 49 skipped (reference data not present), 0 failed |
| UMAP, synthetic 20,000 x 10 (`benchmarks/umap_parity.py`, seeds 0-4) | fuzzy graph: same edges, max weight difference 0.0; a, b equal; trustworthiness arsc 0.9584 (0.9577-0.9588) / umap-learn 0.9587 (0.9585-0.9591); kNN preservation 0.0473 / 0.0488; disparity arsc-arsc 0.42, ref-ref 0.69, arsc-ref 0.85; identical coordinates at 1 and 2 threads; layout 6.6 s (1 thread) and 3.8 s (2) vs umap-learn 8.4 s |
| Leiden / Louvain, planted 8x1000 and toy SNN k20 (`benchmarks/leiden_parity.py`, seeds 0-9) | every row equal to the table above (quality, cluster counts, ARI arsc-arsc / ref-ref / arsc-ref); same labels at 1 and 2 threads in every row |

The real-data rows (80,000 events) were not re-run: those data are not distributed with the benchmark.

## 6. Timing against Seurat

arsc's defaults against Seurat 4.3.0 (SeuratObject 5.0.2, R 4.3.1), step by step with the same parameters, on
deterministic subsets (seed 0) of a 1.37M-cell PBMC store (36,601 genes, 36 cell-type labels): the same cells as the
paired arsc vs scanpy tables (`SCANPY_PARITY.md`). Seurat was timed at 10,000 and 100,000 cells only; at
1,000,000 cells its steps take hours (the earlier three-way suite timed out on `ScaleData` + `RunPCA`).
`benchmarks/seurat_timing.py` (R side: `benchmarks/r_seurat_timing.R`; tables: `benchmarks/render_readme.py`), each
size under the machine-wide timing lock on a 16-core machine. Each tool runs its own chain (arsc's PCA feeds arsc's
graph, Seurat's feeds Seurat's), as a user would. Warm runs only; arsc is the median of 3 runs at 16 threads, Seurat
one run on one thread, which is Seurat's default: R's reference BLAS is single-threaded and `future` runs
sequentially unless a parallel plan is set. Seurat 4.3's `FindAllMarkers` runs limma's rank-sum test per gene
after extracting that gene's row from the column-compressed matrix, so its time grows with cells x genes x cell
types. Earlier runs: the 2026-10-15 three-way suite on simulated data (`benchmarks/results/suite/`) and the git history.

<!-- SEURAT_TIMING:START -->
**10,000 cells** (run 2026-10-26 11:38:48):

| Seurat call | arsc call | arsc, median (min-max) | Seurat | speed-up |
|---|---|---:|---:|---:|
| `NormalizeData` | `pp.log_normalize` | 0.02 s (0.02 s-0.02 s) | 0.70 s | 28.6x |
| `FindVariableFeatures` (vst, 2,000) | `pp.hvg_vst` | 0.20 s (0.20 s-0.21 s) | 1.22 s | 6.2x |
| `ScaleData` + `RunPCA` (30) | `pp.scale` + `pp.pca` | 1.35 s (1.32 s-1.35 s) | 3.81 s | 2.8x |
| `FindNeighbors` (k = 20) | `find_neighbors` | 0.07 s (0.07 s-0.08 s) | 1.64 s | 22.5x |
| `FindClusters` (Louvain, resolution 0.6) | `cluster` | 0.21 s (0.20 s-0.21 s) | 0.87 s | 4.2x |
| `FindAllMarkers` (Wilcoxon) | `find_all_markers` | 7.59 s (7.56 s-7.60 s) | 23 min | 178x |
| `AggregateExpression` (sample x cell type) | `pseudobulk` | 0.30 s (0.28 s-0.31 s) | 0.63 s | 2.1x |
| `AddModuleScore` (3 sets of 30 genes) | `add_module_score` | 0.40 s (0.40 s-0.40 s) | 0.54 s | 1.3x |

Not timed: R package load 1.77 s, reading the cells from the store 3.87 s, `CreateSeuratObject` 1.18 s, warm-up on 2,000 cells 229 s; arsc import 0.99 s, warm-up (numba compilation) on 10,000 cells 10 s. Seurat threads: BLAS 1 (R's reference BLAS), `future` workers 1; arsc 16 threads.

Agreement (each tool on its own chain):

* Variable genes: 2,000 of 2,000 shared.
* Clusters: ARI arsc-Seurat 0.815; Seurat seed 0-seed 1 0.951; arsc seed 0-seed 1 0.884 (15 / 13 / 14 clusters: arsc / Seurat / Seurat seed 1); against the cell-type labels arsc 0.126, Seurat 0.122.
* Markers: 24,217 Seurat rows, 24,217 arsc rows, Jaccard of (cell type, gene) 1.000; avg_log2FC max diff 5.8e-15, p max rel. diff 2.0e-13; top-10 positive markers per cell type shared 100% on average (min 100%).
* Pseudobulk: 1,367 groups each, total counts identical.
* Module scores: max diff 4.6e-15 on the first 5,000 cells.

**100,000 cells** (run 2026-10-26 12:10:46):

| Seurat call | arsc call | arsc, median (min-max) | Seurat | speed-up |
|---|---|---:|---:|---:|
| `NormalizeData` | `pp.log_normalize` | 0.22 s (0.20 s-0.22 s) | 6.90 s | 31.4x |
| `FindVariableFeatures` (vst, 2,000) | `pp.hvg_vst` | 1.25 s (1.24 s-1.26 s) | 6.58 s | 5.3x |
| `ScaleData` + `RunPCA` (30) | `pp.scale` + `pp.pca` | 3.21 s (3.17 s-3.33 s) | 31 s | 9.6x |
| `FindNeighbors` (k = 20) | `find_neighbors` | 0.70 s (0.67 s-0.71 s) | 22 s | 31.8x |
| `FindClusters` (Louvain, resolution 0.6) | `cluster` | 8.84 s (8.58 s-9.01 s) | 28 s | 3.2x |
| `FindAllMarkers` (Wilcoxon) | `find_all_markers` | 75 s (75 s-75 s) | ~176 min (estimate\*) | ~141x |
| `AggregateExpression` (sample x cell type) | `pseudobulk` | 0.48 s (0.47 s-0.49 s) | 4.62 s | 9.6x |
| `AddModuleScore` (3 sets of 30 genes) | `add_module_score` | 2.83 s (2.80 s-2.83 s) | 5.23 s | 1.8x |

\*`FindAllMarkers` did not fit the time budget. It is `FindMarkers` for each cell type in turn, so the cell types were run one by one in a seeded random order until 70 min had passed: cd4_tfh (2,545 cells) 40 s, b_memory_exhausted (91 cells) 334 s, unidentified (29,780 cells) 142 s, gd_non_v2 (223 cells) 346 s, cd8_tem (3,155 cells) 110 s, mait (1,766 cells) 131 s, cdc (72 cells) 12 min, innate_t (533 cells) 72 s, b_memory_switched (50 cells) 420 s, cd8_t_exhausted (745 cells) 246 s, intermediate_monocyte (407 cells) 12 min, cd4_t_cytotoxic (1,925 cells) 107 s, cd4_th1 (1,712 cells) 89 s, pdc (46 cells) 12 min. Estimate = 14 of 35 cell types timed (4214 s, 40-736 s each); mean x 35.

Not timed: R package load 1.82 s, reading the cells from the store 12 s, `CreateSeuratObject` 6.62 s, warm-up on 2,000 cells 247 s; arsc import 1.05 s, warm-up (numba compilation) on 12,000 cells 12 s. Seurat threads: BLAS 1 (R's reference BLAS), `future` workers 1; arsc 16 threads.

Agreement (each tool on its own chain):

* Variable genes: 1,998 of 2,000 shared.
* Clusters: ARI arsc-Seurat 0.819; Seurat seed 0-seed 1 0.809; arsc seed 0-seed 1 0.780 (24 / 25 / 25 clusters: arsc / Seurat / Seurat seed 1); against the cell-type labels arsc 0.100, Seurat 0.111.
* Markers (14 cell types Seurat finished): 13,014 Seurat rows, 13,014 arsc rows, Jaccard of (cell type, gene) 1.000; avg_log2FC max diff 5.8e-15, p max rel. diff 3.6e-11; top-10 not compared: this run's per-cell-type loop kept `FindMarkers`' tie order instead of `FindAllMarkers`' re-sort (see the note below the tables).
* Pseudobulk: 2,030 groups each, total counts identical.
* Module scores: max diff 4.7e-15 on the first 5,000 cells.
<!-- SEURAT_TIMING:END -->

Marker order at 100,000 cells. Many p values underflow to 0 there, so the order among tied rows decides the top of
each list. Seurat 4.3's `FindMarkers` sorts by `order(p_val, -de.results[, 1])`, and column 1 is `p_val`, so ties keep
the feature order; `FindAllMarkers` then re-sorts each cell type by `p_val` and `-avg_log2FC`. The 100,000-cell run
above called `FindMarkers` per cell type without that re-sort, so its tie order differs from `FindAllMarkers` (top-10
overlap 62% on average, min 0%, although every row, p value and fold change agreed). A separate check on the same
100,000 cells (cd4_tfh, cd8_tem, mait; not timed): `arsc.find_markers` gives `FindMarkers`' exact order for
cd4_tfh and cd8_tem, and `find_all_markers` gives `FindAllMarkers`' exact order for them; for mait the order differs
only where one tool's p is a denormal near 1e-308 and the other's is 0 (21 vs 22 zeros); the top 10 matched in all
three. The check also found that `arsc.find_markers` broke ties by `-avg_log2FC`, unlike `FindMarkers`; it now
keeps the feature order. The benchmark's R loop now re-sorts as `FindAllMarkers` does.
