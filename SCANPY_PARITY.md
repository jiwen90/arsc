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
| `sc.pp.neighbors` (42) | `n_neighbors=30`, `use_rep='X_pca_harmony'`; 16 with `X_pca` euclidean / `X_scANVI` cosine | `scanpy_extra.neighbors` (kNN as scanpy: exact up to 4,096 cells, pynndescent above; + UMAP fuzzy simplicial set) | NEW | connectivities exact (max diff 0.0), distances 6e-14, euclidean and cosine |
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

<!-- PAIRED:START -->
Same session, interleaved (arsc and scanpy alternate first), the same thread count for both; median (min-max) of the runs; machine state = system CPU use in the second before each run and the framework's CPU leases (other jobs on the shared machine). Synthetic data of `benchmarks/suite_sim.py`.

**100,000 cells**:

| Step | arsc | scanpy | speed-up (median; slowest pair) | runs each | threads | CPU use before runs | agreement |
|---|---:|---:|---|---:|---:|---|---|
| Log-normalise | 0.89 s (0.83 s-1.12 s) | 2.23 s (2.23 s-2.34 s) | **2.5x**; 2.0x | 3 | 3 | 22-68% |  |
| Highly variable genes (2,000) | 2.29 s (2.20 s-2.34 s) | 4.92 s (4.79 s-5.01 s) | **2.1x**; 2.1x | 3 | 3 | 39-59% |  |
| Scale + PCA (30 PCs) | 6.18 s (5.83 s-6.70 s) | 68 s (58 s-72 s) | **11.0x**; 9.9x | 3 | 3 | 17-63% |  |
| k-NN graph (k = 20) | 5.37 s (4.48 s-5.39 s) | 5.81 s (5.79 s-6.44 s) | **1.1x**; 1.1x | 3 | 3 | 33-73% | recall vs exact: arsc 0.955, scanpy 0.933; overlap arsc-scanpy 0.907 vs scanpy seed 0-seed 1 0.901 |
| k-NN + Leiden clustering | 6.35 s (6.29 s-6.37 s) | 14 s (14 s-16 s) | **2.1x**; 2.1x | 3 | 3 | 30-54% | ARI arsc-scanpy 0.998; vs truth arsc 0.997, scanpy 0.997 |
| Markers, scanpy-style Wilcoxon (20k genes) | 12 s (11 s-13 s) | 101 s (99 s-102 s) | **8.3x**; 7.9x | 3 | 3 | 18-38% |  |
| Pseudobulk sums | 0.37 s (0.34 s-0.41 s) | 0.90 s (0.83 s-0.90 s) | **2.4x**; 2.2x | 3 | 3 | 41-48% |  |
| Pseudobulk DESeq2 (paired, 16 samples) | 15 s (14 s-15 s) | 15 s (15 s-15 s) | **1.0x**; 1.0x | 3 | 3 | 36-52% |  |
| scanpy gene-set score (5 sets) | 5.17 s (4.54 s-5.23 s) | 9.44 s (9.23 s-9.62 s) | **1.8x**; 1.8x | 3 | 3 | 34-52% |  |

Measured 2026-10-25 07:40:05 to 2026-10-25 08:13:19; leases seen: leased 16/16 by 4 process(es); machine load 14%; leased 16/16 by 4 process(es); machine load 17%; leased 16/16 by 4 process(es); machine load 18%.

**1,000,000 cells**:

| Step | arsc | scanpy | speed-up (median; slowest pair) | runs each | threads | CPU use before runs | agreement |
|---|---:|---:|---|---:|---:|---|---|
| Log-normalise | 9.81 s (9.81 s-12 s) | 21 s (21 s-21 s) | **2.1x**; 1.7x | 3 | 3 | 20-30% |  |
| Highly variable genes (2,000) | 18 s (17 s-18 s) | 44 s (44 s-45 s) | **2.5x**; 2.4x | 3 | 3 | 12-30% |  |
| Scale + PCA (30 PCs) | 48 s (48 s-49 s) | 12.0 min (11.3 min-13.8 min) | **15.0x**; 14.1x | 3 | 3 | 19-41% |  |
| k-NN graph (k = 20) | 78 s (75 s-79 s) | 90 s (84 s-92 s) | **1.2x**; 1.1x | 3 | 3 | 37-50% | recall vs exact: arsc 0.914, scanpy 0.910; overlap arsc-scanpy 0.868 vs scanpy seed 0-seed 1 0.861 |
| k-NN + Leiden clustering | 101 s (95 s-109 s) | 214 s (194 s-222 s) | **2.1x**; 2.0x | 3 | 3 | 23-51% | ARI arsc-scanpy 0.997; vs truth arsc 0.995, scanpy 0.997 |
| Markers, scanpy-style Wilcoxon (20k genes) | 155 s (155 s-155 s) | 72.3 min (72.3 min-72.3 min) | **27.9x**; 27.9x | 1 | 3 | 24-34% |  |
| Pseudobulk sums | 3.14 s (3.09 s-3.31 s) | 7.10 s (7.04 s-7.24 s) | **2.3x**; 2.1x | 3 | 3 | 17-34% |  |
| Pseudobulk DESeq2 (paired, 16 samples) | 17 s (17 s-18 s) | 20 s (20 s-20 s) | **1.2x**; 1.1x | 3 | 3 | 13-33% |  |
| scanpy gene-set score (5 sets) | 42 s (41 s-42 s) | 83 s (80 s-86 s) | **2.0x**; 2.0x | 3 | 3 | 14-20% |  |

Measured 2026-10-25 08:13:44 to 2026-10-25 10:04:12; leases seen: leased 10/16 by 4 process(es); machine load 16%; leased 10/16 by 4 process(es); machine load 17%; leased 10/16 by 4 process(es); machine load 18%.

<!-- PAIRED:END -->

k-NN: `arsc.knn` (and `find_neighbors`, `scanpy_extra.neighbors`) searches exactly up to 4,096 cells and with
pynndescent above, with the settings scanpy passes (UMAP's `nearest_neighbors`), seeded; distances are re-computed
in float64. Library choice (2026-10-25, 2 threads each, a 1M-cell Gaussian copy of the benchmark PCA, 500 random
queries against brute force; one run each, machine shared): pynndescent 105 s, recall 0.951; FAISS IVFFlat
(nlist 2,000 / nprobe 64) 251 s, 0.961; FAISS HNSW (M 16, efSearch 48) 128 s, 0.776; hnswlib (M 16, ef 48) 181 s,
0.854; scanpy `pp.neighbors` 121 s, 0.913. pynndescent was the fastest established library at scanpy's quality, so
it is the only backend (no GPU path: pynndescent has none, and FAISS GPU wheels exist only for Linux).
First call in a fresh process (numba compilation included; `benchmarks/cold_start.py`, 3 threads, 3 interleaved
runs, 2026-10-25): k-NN graph at 10,000 cells arsc 30.6 s vs scanpy 54.2 s, at 100,000 cells 33.2 s vs 42.6 s. Most of
it is pynndescent's one-off compilation, which both pay; arsc 0.3's exact search was compiled once and cached on disk,
so a first call on 4,097-100,000 cells is now about 30 s slower than in arsc 0.3 (pass `method="exact"` in short
scripts on small data). Reproducibility: the same seed and thread count give the same graph; another thread count changes about 2% of the
neighbour entries (pynndescent's parallel updates).

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
  cells the graphs are compared by recall against exact neighbours, overlap with scanpy (against scanpy's own seed-to-seed
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
