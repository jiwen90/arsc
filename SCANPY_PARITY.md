# scanpy / scverse parity

Inventory source: the 81 Jupyter notebooks of a published scanpy-based clinical single-cell analysis (reference workload),
code cells only, commented lines removed. About 850 scanpy/scverse calls; 294 are plotting (`sc.pl.*`,
`palantir.plot.*`, out of scope: arsc does not plot). Counts below are call counts across notebooks.
New code: `src/arsc/scanpy_extra.py`; tests: `tests/test_scanpy_extra.py` (19 tests, all pass; reference scanpy 1.11.5,
anndata 0.12.19). Validation data: 20,000-cell random subsample (seed 0) of a 132,989-cell CD8 T-cell dataset (3,000 HVG
counts), harmony PCs for graph steps (3,000 cells, because scanpy switches from exact to approximate kNN at 4,096 cells).

## Function map

| scanpy / scverse call (n) | Params used in the reference notebooks | arsc equivalent | Status | Validation vs scanpy |
|---|---|---|---|---|
| `sc.pp.calculate_qc_metrics` (20) | `qc_vars=[mt,ribo,hb]`, `percent_top=[20]`, `log1p=True` | `scanpy_extra.qc_metrics` | NEW | All 11 columns equal (max abs diff 1.4e-14) |
| `sc.pp.filter_cells/filter_genes` (1/3) | `min_genes=200`, `min_cells=3/10` | numpy on `qc_metrics` / column nnz | trivial, not wrapped | n/a |
| `sc.pp.normalize_total` (37) + `sc.pp.log1p` (50) | `target_sum=1e4` | `scanpy_extra.normalize_total_log1p` (`arsc.pp.log_normalize` equal to 1.8e-15) | NEW (bitwise) | max diff 0.0. Use the new one for rank tests: Seurat arithmetic `x/s*sf` differs by 1 ulp and changes tie structure |
| `sc.pp.normalize_per_cell` (12) | `counts_per_cell_after=1e4` | same as above | covered | same op as normalize_total |
| `sc.pp.highly_variable_genes` (31+) | default `seurat` with `n_top_genes` 2000/3000/4000/5000; `seurat_v3` on `layer="counts"`, 3000 | `scanpy_extra.highly_variable_genes` (`seurat`, `cell_ranger`, `seurat_v3`/`_paper`, `batch_key` for v3); v3 delegates to `arsc.pp.hvg_vst` | NEW | seurat/cell_ranger: gene set identical, means/dispersions 1e-14, dispersions_norm 3e-7 (scanpy stores float32). v3: `variances_norm` 3e-12, 999/1000 genes shared (the 1 swap is an exact tie at the cutoff). v3 with `batch_key`: norm. variance r > 0.999999, 940-950/1000 genes shared (ties) |
| `sc.pp.scale` (41) | `max_value=10` | `arsc.pp.scale` | covered | max diff 1e-13 |
| `sc.pp.regress_out` (1) | numeric covariates | `scanpy_extra.regress_out` (numeric, or single categorical key) | NEW | max diff 1e-13 |
| `sc.tl.pca` / `sc.pp.pca` (20+9) | `svd_solver='arpack'`, `n_comps=30/50` | `arsc.pp.pca` | covered | per-PC abs r >= 0.998 (arpack on 1,000 HVG vs gram), variance ratio diff 1.6e-5; scanpy 1.7 s vs arsc 0.46 s |
| `sce.pp.harmony_integrate` (12) | key = sample | `arsc.run_harmony` (R harmony port) | covered, NOT validated against harmonypy | validated vs R harmony 2.0.5 / 2.1.0 / 1.2 (see HARMONY_PARITY.md). harmonypy is a different port, so expect r ~0.99 per dim, not identity |
| `sc.pp.neighbors` (42) | `n_neighbors=30`, `use_rep='X_pca_harmony'`; 16 with `X_pca` euclidean / `X_scANVI` cosine | `scanpy_extra.neighbors` (kNN as scanpy: exact up to 4,096 cells, approximate above; + UMAP fuzzy simplicial set) | NEW | connectivities exact (max diff 0.0), distances 6e-14, euclidean and cosine |
| `sc.tl.leiden` (25+) | `resolution` 0.4-2.0, default `leidenalg`, `random_state=0` | `scanpy_extra.leiden` | NEW | labels identical (res 0.5, 2.0), ARI 1.0 |
| `sc.tl.umap` (25+) | default, or `min_dist=0.4, spread=1.0` | `scanpy_extra.umap_from_connectivities` | NEW | coordinates identical (max diff 0.0) |
| `sc.tl.rank_genes_groups` (36+) | `method='wilcoxon'`, `pts=True`, `use_raw=True/False`, group by leiden / celltype / antigen_type | `scanpy_extra.rank_genes_groups` | NEW | wilcoxon (+ `tie_correct`), t-test, t-test_overestim_var, reference `rest` and a group: p-values equal to 1e-13, scores equal to float32 storage (rel 6e-8), pts exact |
| `sc.get.rank_genes_groups_df` (50) | `group=None` | the output of `rank_genes_groups` is already that table | covered | same columns: `group, names, scores, logfoldchanges, pvals, pvals_adj` (+ `pct_nz_group`, `pct_nz_reference`) |
| `sc.tl.score_genes` (51) | default, `ctrl_size=50`, `use_raw=False` | `scanpy_extra.score_genes` | NEW | exact (max diff 5e-15) incl. `ctrl_size`, `n_bins`, `random_state` variants |
| `sc.tl.dendrogram` (29) | `groupby` leiden / celltype | `scanpy_extra.dendrogram` | NEW | linkage 2.5e-8, same leaf order |
| `sc.get.obs_df` (4) | single gene extraction | column slice of `store.X()` | trivial | n/a |
| `palantir.utils.run_diffusion_maps` (10) | `n_components=5`, `pca_key='X_pca_harmony'` | `scanpy_extra.palantir_diffusion_maps` | NEW | eigenvalues 1e-10, kernel 5e-15, T 3e-16 (3,000 cells, exact-kNN regime) |
| `palantir.utils.determine_multiscale_space` (10) | default | `scanpy_extra.palantir_multiscale_space` | NEW | per-component abs r = 1.0 |
| `palantir.utils.run_magic_imputation` (10) | default `n_steps=3` | `scanpy_extra.magic_impute` | NEW | max diff 4e-6 (float32); 13x faster (T applied stepwise, T^3 never formed) |
| `palantir.core.run_palantir`, `presults.select_branch_cells`, `compute_gene_trends`, `cluster_gene_trends` (5/5/4/1) | `num_waypoints=500`, terminal states, `q=0.01, eps=0.01` | none | GAP | stochastic waypoint sampling, shortest paths and fate probabilities; not ported (see caveats) |
| `hotspot` (`legacy_init model='danb'`, `create_knn_graph`, `compute_autocorrelations`, `compute_local_correlations`, `create_modules`, `calculate_module_scores`) (1 each) | `n_neighbors=50`, `min_gene_threshold=150`, `core_only=True`, `fdr_threshold=0.05` | none | GAP | module scores were precomputed in the reference workload; not ported |
| `scv.tl.recover_dynamics`, `scv.tl.velocity` (1 each) | `dynamical`, `n_jobs=8` | none | GAP, single exploratory cell | out of scope |
| `gp.enrichr/prerank/ssgsea/get_library/read_gmt` (10/3/3/8/1) | various | `arsc.fgsea`, `arsc.genesets`, `arsc.score_ucell` for the prerank / rank-based cases | partial | Enrichr needs web access; use `arsc.de.fgsea` (validated vs R fgsea) |
| `dc.run_ulm`, `get_progeny`, `get_dorothea` (2/1/1) | `top=5000`; levels A-C | none (run decoupler directly) | not needed | single-call use on pseudobulk |
| `sm.OLS`, `sm.add_constant` (9) | `cov_type='cluster'/'HC3'` | statsmodels directly | n/a | |
| `ad.AnnData`, `ad.concat`, `var_names_make_unique`, `sc.settings.*`, `sc.logging.*` | | AnnData bookkeeping | n/a | |
| `sc.tl.diffmap`, `sc.tl.dpt` (not used in the notebooks, which used palantir) | | `scanpy_extra.diffmap`, `scanpy_extra.dpt` | NEW (bonus) | eigenvalues 1e-6 (scanpy float32), eigenvectors r = 1.0, DPT r = 0.99999999 |
| `sc.tl.paga`, `sc.tl.embedding_density` (not used) | | none | not implemented | not used in the notebooks |
| `sc.pl.*` (294), `palantir.plot.*` (36), `sm.set_array` | | plotting libraries | out of scope | |

## Speed: paired runs (same session, same threads)

**Speed gate, 10,000 cells, 16 cores** (2026-10-25 23:25, after the switch to arsc's own k-NN kernel;
`ARSC_SPEED_GATE=1 python -m pytest -q tests/test_speed_gate.py` under `tools/timing_lock.py run`): toy data of
`benchmarks/suite_sim.py` generated in memory, arsc and scanpy in one process, interleaved (alternating which runs first), 16 threads for
both, warm: every step is run once untimed first, so numba compilation is not in the times (first-call cost: see
the k-NN backend section). Median (min-max) of 3 runs each. 16 cores; other processes used 8.9% of the machine's
CPU on average during the run (at most 27.9%).

| Step | arsc | scanpy | speed-up (median) | slowest pair, arsc / scanpy |
|---|---:|---:|---:|---:|
| Log-normalize | 0.022 s (0.021-0.025) | 0.134 s (0.134-0.137) | **6.2x** | 0.18 |
| Highly variable genes (2,000) | 0.204 s (0.203-0.206) | 0.428 s (0.428-0.429) | **2.1x** | 0.48 |
| Scale + PCA (30 PCs) | 1.29 s (1.28-1.29) | 2.74 s (2.74-2.80) | **2.1x** | 0.47 |
| k-NN graph (k = 20) | 0.093 s (0.092-0.097) | 1.20 s (1.20-1.23) | **12.9x** | 0.08 |
| k-NN + Leiden clustering | 0.215 s (0.213-0.220) | 1.43 s (1.43-1.44) | **6.7x** | 0.15 |
| Markers, scanpy-style Wilcoxon | 0.515 s (0.510-0.522) | 2.77 s (2.73-2.78) | **5.4x** | 0.19 |
| Pseudobulk sums | 0.029 s (0.028-0.031) | 0.096 s (0.096-0.097) | **3.3x** | 0.32 |
| Pseudobulk DESeq2 (paired) | 1.55 s (1.47-1.72) | 2.12 s (1.93-2.19) | **1.4x** | 0.89 |
| scanpy gene-set score (5 sets) | 0.404 s (0.400-0.408) | 0.723 s (0.706-0.733) | **1.8x** | 0.58 |

Agreement in the same run: k-NN recall against exact neighbors arsc 0.993, scanpy 0.949 (overlap arsc-scanpy
0.943); Leiden ARI against the cell types arsc 0.995, scanpy 0.996, arsc-scanpy 0.997 (10 clusters each).

**100,000 and 1,000,000 real cells, 16 cores** (2026-10-26; `benchmarks/paired_bench.py --sizes 100000` and
`--sizes 1000000 --step-repeats rank_genes=1`, `--threads 16 --repeats 3 --quality`, each size under
`tools/timing_lock.py run` with nothing else admitted). arsc was faster than scanpy on all 9 steps at both sizes:
1.2x-16x at 100,000 cells and 1.2x-84x at 1,000,000 (median of 3 runs; scanpy's Wilcoxon markers took 78 min at
1M cells, so that step ran once at 1M). Notes:

* **Agreement.** k-NN recall against exact float64 neighbors of 2,000 query cells: arsc 0.990 / 0.988 (100k /
  1M), scanpy 0.949 / 0.946; arsc's overlap with scanpy (0.940 / 0.937) equals scanpy's overlap with itself under
  another seed (0.939 / 0.934). The Leiden partitions agree less: ARI arsc-scanpy 0.54 / 0.62 against scanpy
  seed 0-seed 1 0.65 / 0.66. The two tools cluster different graphs at the same resolution (arsc: the
  Seurat-style shared-nearest-neighbor graph; scanpy: UMAP connectivities), and arsc finds more clusters (26 vs
  20-21 at 100k, 33 vs 22-24 at 1M). On these PBMC data, where 30% of cells carry the label "unidentified", both
  tools' ARI against the cell-type labels is about 0.1.
* **The slowest pair.** arsc's first DESeq2 run of each session was slower (5.8 s at 100k, 12.8 s at 1M, against
  3.8-3.9 s and 5.7 s for the next two), so its slowest pair loses to scanpy (0.8x, 0.5x); the medians favor arsc
  (1.2x at both sizes).
* **Machine state.** At 100k another process briefly used up to 50% of the machine's CPU (4.1% on average); at
  1M up to 27% (2.8% on average).

These tables replace the 100,000- and 1,000,000-cell tables measured on 2026-10-25 at 3 threads on synthetic
data beside other jobs (superseded; kept in `benchmarks/results/paired/` and the git history).

<!-- PAIRED:START -->
Same session, interleaved (arsc and scanpy alternate first), the same thread count for both; median (min-max) of the runs; machine state = system CPU use in the second before each run and the framework's CPU leases (other jobs on the shared machine). Real data: a deterministic subset (seed 0) of 100,000 or 1,000,000 cells of a 1.37M-cell PBMC store (36,601 genes, 32 patients, 36 cell-type labels), read memory-mapped and held in memory; nothing written to disk but these tables. Both tools in one process at 16 threads; every step run once untimed first (on a 12,000-cell toy dataset, so numba compilation is excluded).

**100,000 cells**:

| Step | arsc | scanpy | speed-up (median; slowest pair) | runs each | threads | CPU use before runs | agreement |
|---|---:|---:|---|---:|---:|---|---|
| Log-normalize | 0.17 s (0.16 s-0.21 s) | 1.20 s (1.20 s-1.23 s) | **7.1x**; 5.8x | 3 | 16 | 4-4% |  |
| Highly variable genes (2,000) | 1.26 s (1.24 s-1.27 s) | 3.70 s (3.70 s-3.70 s) | **2.9x**; 2.9x | 3 | 16 | 4-4% |  |
| Scale + PCA (30 PCs) | 2.83 s (2.80 s-2.86 s) | 9.57 s (9.56 s-9.78 s) | **3.4x**; 3.4x | 3 | 16 | 4-4% |  |
| k-NN graph (k = 20) | 0.68 s (0.66 s-0.70 s) | 2.47 s (2.44 s-2.49 s) | **3.6x**; 3.5x | 3 | 16 | 4-4% | recall vs exact: arsc 0.990, scanpy 0.949; overlap arsc-scanpy 0.940 vs scanpy seed 0-seed 1 0.939 |
| k-NN + Leiden clustering | 2.28 s (2.27 s-2.34 s) | 7.73 s (7.71 s-7.75 s) | **3.4x**; 3.3x | 3 | 16 | 4-4% | ARI arsc-scanpy 0.544 vs scanpy seed 0-seed 1 0.647 (26 / 20 / 21 clusters); vs the cell-type labels arsc 0.106, scanpy 0.120 |
| Markers, scanpy-style Wilcoxon (36,601 genes, 36 cell types) | 5.32 s (5.22 s-5.32 s) | 87 s (87 s-87 s) | **16.4x**; 16.3x | 3 | 16 | 4-4% |  |
| Pseudobulk sums (sample x cell type) | 0.43 s (0.41 s-0.43 s) | 0.84 s (0.84 s-0.84 s) | **2.0x**; 1.9x | 3 | 16 | 4-4% |  |
| Pseudobulk DESeq2 (paired, 25 patients x 2 timepoints) | 3.90 s (3.84 s-5.77 s) | 4.68 s (4.64 s-4.79 s) | **1.2x**; 0.8x | 3 | 16 | 4-4% |  |
| scanpy gene-set score (3 sets of 30 genes) | 1.74 s (1.73 s-1.74 s) | 3.46 s (3.45 s-3.47 s) | **2.0x**; 2.0x | 3 | 16 | 4-4% |  |

Run 2026-10-26 01:53:21 under the machine-wide timing lock (16 threads, 16 physical cores, 11 min including data loading and warm-up): other processes used 4.1% of the machine's CPU on average (at most 50.4%; 1.31 cores on average), sampled every second (650 samples).

**1,000,000 cells**:

| Step | arsc | scanpy | speed-up (median; slowest pair) | runs each | threads | CPU use before runs | agreement |
|---|---:|---:|---|---:|---:|---|---|
| Log-normalize | 1.67 s (1.60 s-1.98 s) | 12 s (12 s-12 s) | **7.2x**; 6.1x | 3 | 16 | 4-4% |  |
| Highly variable genes (2,000) | 12 s (12 s-12 s) | 36 s (36 s-37 s) | **3.1x**; 3.0x | 3 | 16 | 4-4% |  |
| Scale + PCA (30 PCs) | 19 s (18 s-19 s) | 99 s (99 s-100 s) | **5.3x**; 5.2x | 3 | 16 | 4-4% |  |
| k-NN graph (k = 20) | 13 s (13 s-13 s) | 24 s (24 s-24 s) | **1.8x**; 1.8x | 3 | 16 | 4-4% | recall vs exact: arsc 0.988, scanpy 0.946; overlap arsc-scanpy 0.937 vs scanpy seed 0-seed 1 0.934 |
| k-NN + Leiden clustering | 37 s (36 s-37 s) | 104 s (104 s-105 s) | **2.9x**; 2.8x | 3 | 16 | 4-4% | ARI arsc-scanpy 0.619 vs scanpy seed 0-seed 1 0.660 (33 / 24 / 22 clusters); vs the cell-type labels arsc 0.107, scanpy 0.122 |
| Markers, scanpy-style Wilcoxon (36,601 genes, 36 cell types) | 55 s (55 s-55 s) | 77.6 min (77.6 min-77.6 min) | **84.4x**; 84.4x | 1 | 16 | 4-4% |  |
| Pseudobulk sums (sample x cell type) | 0.96 s (0.95 s-0.98 s) | 5.23 s (5.21 s-5.24 s) | **5.4x**; 5.3x | 3 | 16 | 4-4% |  |
| Pseudobulk DESeq2 (paired, 25 patients x 2 timepoints) | 5.71 s (5.71 s-13 s) | 6.87 s (6.85 s-6.90 s) | **1.2x**; 0.5x | 3 | 16 | 4-4% |  |
| scanpy gene-set score (3 sets of 30 genes) | 17 s (17 s-17 s) | 34 s (33 s-34 s) | **2.0x**; 2.0x | 3 | 16 | 4-4% |  |

Run 2026-10-26 02:04:27 under the machine-wide timing lock (16 threads, 16 physical cores, 114 min including data loading and warm-up): other processes used 2.8% of the machine's CPU on average (at most 26.6%; 0.90 cores on average), sampled every second (6,752 samples).

<!-- PAIRED:END -->

## k-NN backend

**arsc uses its own k-NN kernel** (2026-10-25): it was the fastest backend at recall@k >= 0.95 and >= 0.99 on every
real dataset tested, ahead of FAISS (Meta), hnswlib, Voyager and Annoy (Spotify), pynndescent and scanpy. pynndescent
and scikit-learn are no longer used. `arsc.knn` (and `find_neighbors`, `scanpy_extra.neighbors`) searches exactly
up to 4,096 cells and approximately above, as scanpy.

* **Approximate** (`method="approximate"`, the default above 4,096 cells): a seeded k-means splits the cells into
  2 sqrt(n) clusters; every cell scans all cells of its 24 nearest clusters. The distances are float32 matrix
  products (BLAS) of the cells that probe a cluster and the cluster's cells, centered on its center; the k + 8 best
  candidates are re-scored exactly in float64 and ordered by (distance, index). Deterministic for a seed and
  independent of the thread count.
* **Exact** (`method="exact"`): the same partition, every cluster a candidate, skipped or narrowed by the triangle
  inequality, a float32 filter with a proven rounding margin and float64 distances: the exact k-NN at any size.

Time to build the full graph (build + every cell queries) at the cheapest setting of each library that reaches the
recall, from a parameter sweep of each library (FAISS IVF-Flat nlist/nprobe, IVF-PQ, HNSW M/efConstruction/efSearch;
hnswlib and Voyager M/ef; Annoy trees/search_k; pynndescent max_candidates/n_neighbors; arsc clusters/probes).
Recall@k is scored against exact float64 neighbors of 2,000 random cells (self included; ties counted). Data: real
PCs held in memory, never written: "PBMC CITE-seq" = the stored 30-PC PCA of a 260k-cell PBMC CITE-seq store,
100k random cells; "PBMC 1.37M" = 50 PCs (vst top 2,000 genes, log1p CP10k, z clip 10) of a 1.37M-cell PBMC store,
100k or 1M random cells. Pinned to the same 16 logical CPUs; most sweep points at 6 threads, arsc's and the later
1M points at 4 threads (the machine was shared), so arsc's lead at 100k is understated:

| Data | k | arsc | hnswlib | FAISS HNSW | FAISS IVF-Flat | Voyager | pynndescent | Annoy | scanpy |
|---|---:|---|---|---|---|---|---|---|---|
| PBMC CITE-seq, 100k, 30 PCs | 15 | **0.50 / 0.67 s** | 0.62 / 0.80 | 0.58 / 0.85 | 0.66 / 0.86 | 0.77 / 1.08 | 1.03 / 1.03 | 2.3 / 3.8 | recall 0.931 (best) |
| PBMC CITE-seq, 100k, 30 PCs | 30 | **0.49 / 0.63 s** | 0.73 / 0.98 | 0.68 / 1.21 | 0.80 / 1.13 | 0.86 / 1.38 | 2.2 / 2.2 | 3.5 / 5.7 | 12.2 / - |
| PBMC 1.37M, 100k, 50 PCs | 15 | **0.67 / 0.91 s** | 0.93 / 1.60 | 1.00 / 1.54 | 1.10 / 1.76 | 1.54 / 3.11 | 1.28 / 2.18 | 5.8 / 18 | recall 0.914 (best) |
| PBMC 1.37M, 100k, 50 PCs | 30 | **0.75 / 1.07 s** | 1.11 / 1.99 | 1.10 / 2.11 | 1.20 / 2.51 | 1.89 / 3.45 | 2.47 / 2.82 | 7.7 / 18 | 11.5 / - |
| PBMC 1.37M, 1M, 50 PCs | 15 | **23 / 37 s** | 33 / 64 | 44 / 80 | 58 / 110 | 85 / 195 | 45 / - | - / - | recall 0.902 |

Cells: seconds to reach recall 0.95 / 0.99 ("-": not reached in the sweep). The 1M row mixes 6 threads (hnswlib, FAISS)
and 4 threads (arsc, Voyager, pynndescent, Annoy, scanpy). A Gaussian copy of the CITE-seq PCs (1M cells, 30 dims) was
swept for the HNSW and IVF libraries before the sweep was stopped; arsc was not run on it.

**Confirmation at 1M cells, k = 15, everything at 4 threads, one run each, in memory** (`benchmarks/results/knn_backend/confirm_1m.csv`):
arsc defaults 22.4 s (recall 0.961); pynndescent (scanpy's settings) 46.8 s (0.958); hnswlib M 16 / ef 96 56.1 s (0.952);
FAISS HNSW M 32 / efSearch 32 71.0 s (0.957); FAISS IVF-Flat nlist 2,000 / nprobe 24 73.8 s (0.962); Voyager M 32 / ef 64
95.9 s (0.953); scanpy `pp.neighbors` 141.5 s (0.902). arsc is 2.1x faster than the next library at the same recall.

Defaults: 2 sqrt(n) clusters and 24 probes give recall 0.961-0.996 on the data above (k = 15 and 30); more probes
(`arsc.neighbors.APPROX_PROBE`) raise recall: 32 probes 0.978 and 48 probes 0.991 at 1M cells.

![time vs recall](benchmarks/results/knn_backend/knn_backend_recall.png)

Every configuration: `benchmarks/results/knn_backend/sweep_points.csv`, frontiers `pareto.csv`.

First call in a fresh process (`benchmarks/cold_start.py --what knn`, 100,000 toy cells, 4 threads, 2026-10-25):
`arsc.knn` 7.1 s with an empty numba cache (one-off compilation) and 1.1 s once compiled kernels are cached on disk;
`scanpy.pp.neighbors` 50-55 s. pynndescent's own one-off compilation (about 30 s) is gone from arsc.

## Speed, earlier single measurements (32-thread workstation, not paired)

| step | scanpy | arsc |
|---|---|---|
| `rank_genes_groups` wilcoxon, 20k cells x 3,000 genes x 24 groups (`pts=True`) | 1.8 s (4.1 s with `tie_correct`) | 0.26 s (7-18x) |
| `rank_genes_groups` wilcoxon, all 132,989 cells | 12.7 s | 1.3 s (10x) |
| t-test (either), 20k cells | 1.0 s | 0.15-0.18 s (6x) |
| `score_genes`, 20k cells | 0.12 s | 0.03 s |
| `neighbors` (3,000 cells, scanpy includes numba compile) | 16 s | 1.3 s |
| `neighbors`, all 132,989 cells (30 NN, harmony PCs) | 55 s (pynndescent, approximate) | 24 s (exact kNN; arsc 0.3 default, now `method="exact"`) |
| `leiden` res 2.0, `n_iterations=-1`, 133k cells (22 clusters) | 880 s | 1,086 s (same leidenalg; single-threaded; no gain). Use `n_iterations=2` for speed |
| `calculate_qc_metrics`, 20k cells | 0.12 s | 0.02 s |
| `regress_out` (2 covariates) | 3-10 s | 1.1 s |
| `pca` (30 comps, 1,000 HVG, 20k cells) | 1.7 s | 0.46 s |
| `run_diffusion_maps` (3,000 cells) | 0.17 s | 0.05 s |
| `run_magic_imputation` (3,000 cells x 3,000 genes) | 6.4 s | 0.5 s (13x) |
| HVG seurat / cell_ranger | 0.1-0.3 s | 0.04-0.1 s |

Full-size graph check (132,989 cells): arsc exact-kNN connectivities share 99.05% of scanpy's approximate-kNN edges (6.47M nnz each); Leiden res 2.0 on the two graphs agree at ARI 0.936. Leiden itself is not accelerated.

## Caveats
- scanpy stores `scores`, `logfoldchanges` and `dispersions_norm` as float32; "exact" means equal at that precision.
- `rank_genes_groups` expects log1p data (as scanpy: `logfoldchanges` use `expm1`). The notebooks ran it on `.raw` (log-normalized,
  all genes) in most places and on scaled `X` in a few; both paths work (negative values go through a dense rank path).
  Scanpy's Wilcoxon default is NOT tie-corrected and has no continuity correction, unlike `arsc.wilcoxauc` / Seurat.
- `neighbors`: arsc switches like scanpy (exact below 4,096 cells, pynndescent with scanpy's settings above). Above 4,096
  cells the graphs are compared by recall against exact neighbors, overlap with scanpy (against scanpy's own seed-to-seed
  overlap) and cluster ARI (paired table above). `n_neighbors` counts the cell
  itself as in scanpy.
- `highly_variable_genes`: `batch_key` is supported only for `seurat_v3`. Genes with exactly tied statistics at the top-n cutoff
  are broken arbitrarily by scanpy (unstable sort); tie order cannot be reproduced bit-for-bit.
- `score_genes` draws control genes from a local `RandomState(random_state)`, the same MT19937 stream scanpy draws after
  `np.random.seed`, so results match scanpy without touching the global RNG (`SEEDS.md`).
- scanpy's `leiden` default (leidenalg, directed graph, `n_iterations=-1`) is reproduced; scanpy warns that the default will change
  to igraph. Use `flavor="igraph"` to get the future default.
- Palantir/hotspot validation used pip packages installed into a separate directory (scikit-misc 0.5.3,
  palantir 1.4.5, hotspotsc 1.1.3), not into the main environment; and the corresponding tests skip if the packages are absent.
- Not ported (GAPs): the Palantir pseudotime run itself (waypoints, fate probabilities, branch selection), Hotspot
  (danb autocorrelation, local correlations, modules), scVelo.
