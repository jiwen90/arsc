# Changelog

All notable changes to arsc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may change the API).

## [Unreleased]

### Tests
- Tests run on GitHub CI on every push (any branch, not only main). Locally, name the test files a change can break,
  or a group by area (`python tests/run.py knn`; the table is in `tests/run.py`); `tests/conftest.py`
  refuses a whole-directory run or `tests/run.py everything` unless `ARSC_FULL=1` (set in CI). The `tests/testsel.py`
  selector was removed.
- The tests import the checkout's own `src`, not the editable install's (a git worktree's tests had run the main
  checkout's code); `tests/test_source_pin.py` checks it with a sentinel in a copied checkout.
- Test references no longer live in a scratch folder (pruned after 2 days, which had silently turned most R-parity
  tests into skips). Stored R outputs go to `ARSC_REFS` (autoresearch: `.cache/arsc`); `benchmarks/regen_refs.py`
  rebuilds any set (`status` lists folders, sizes, inputs and readers). One shared input for the large-object tests:
  the registered 38,440-cell gamma-delta store (`ARSC_STORE_GD`, memory-mapped, never copied); R reads the Seurat
  object it was converted from (`ARSC_GD_RDS`). Inputs are rebuilt in memory (stats pseudobulk and composition:
  `benchmarks/stats_parity_inputs.py`; harmony PCA: `harmony_input.build`; resampling tables:
  `stats_resampling_cases.build`), so only R outputs are stored. The lost stats-parity and clones R scripts are back
  in the repo (`r_reference_stats_parity.R`, `r_reference_stats_small.R`, `r_reference_clones.R`); `r_reference_gd.R`
  writes only what the tests read unless `ARSC_REF_FULL=1` (gzip, 3 MB instead of ~1 GB); `r_reference_seurat_extra.R` stores gzip and every 10th cell of the
  scaled data (31 MB instead of 364 MB); clones references are gzipped (8 MB instead of 57 MB).
- Every reference skip names the missing file and the command that rebuilds it.
- `tests/test_repo_hygiene.py`: fails when a path in `tests/local.env` (or a resolved reference folder) is under a
  scratch folder, or when git tracks a data file (datasets are never committed; `.gitignore` covers the formats).
- `genesets.msigdb` no longer searches `$AR_ROOT/scratch/arsc/msigdb`.

### Changed
- k-NN backend: arsc's own kernel replaces pynndescent (approximate) and scikit-learn (exact); neither is a
  dependency any more. Approximate: seeded k-means into 2 sqrt(n) clusters, each cell scans its 24 nearest
  clusters with float32 matrix products, the k + 8 best re-scored in float64. Exact: the same partition with
  triangle-inequality pruning and a proven float32 margin. Chosen by a time-vs-recall sweep against FAISS, hnswlib,
  Voyager, Annoy, pynndescent and scanpy on real PCs (SCANPY_PARITY.md, "k-NN backend"): fastest at recall 0.95 and
  0.99 at 100k and 1M cells; 22 s vs 47 s (pynndescent) and 56 s (hnswlib) at 1M cells. The graph no longer
  depends on the thread count. `method="pynndescent"` maps to `"approximate"` with a DeprecationWarning;
  `n_clusters` is used again.
- Benchmarks keep their data in memory: `paired_bench.py`, the speed gate and `cold_start.py` build the input from a
  real store (`ARSC_BENCH_STORE`, memory-mapped, read-only) or the toy data generated in memory from a fixed seed
  (`benchdata.py`); nothing is written under `benchmarks/` (tested). The speed gate defaults to 10,000 toy cells in
  one process. `run_suite.py` (Seurat side) writes data only with an explicit `--bench-dir`; no `.bench_data` cache.
- k-NN (`knn`, `find_neighbors`, `wnn`, `scanpy_extra.neighbors`, `palantir_diffusion_maps`, and through them
  `cluster`/`umap` graph building): `method="auto"` (new default) searches exactly up to 4,096 points and
  approximately above, as `scanpy.pp.neighbors`. Approximate search is pynndescent with the settings scanpy uses
  (UMAP's `nearest_neighbors`: `n_trees = min(64, 5 + sqrt(n)/20)`, `n_iters = max(5, log2 n)`), seeded; it is
  now a hard dependency (`pynndescent>=0.5`, BSD-2). Exact search is scikit-learn's brute force for k + 8
  candidates, re-scored in float64 and ordered by (distance, index), with a proven rounding check that re-scores
  a row against every point when needed; arsc's own partition search is removed. Distances are always the exact
  float64 values. arsc 0.3.x searched exactly at every size: pass `method="exact"` to reproduce those graphs.
  `knn(n_clusters=)` is accepted and ignored. Methods: `"auto"`, `"exact"`, `"approximate"` (aliases `"approx"`,
  `"pynndescent"`). The 1M-cell k-NN step went from 13 min (exact) to scanpy's speed (benchmark table in
  SCANPY_PARITY.md). `clonotrace` keeps exact search (dbscan::kNN semantics).
- `snn_graph`: only the shared-neighbour pairs that pass `prune` are sorted (3x faster at 100,000 cells; output
  identical).
- `run_harmony` reproduces R harmony 2.0.5 (the current CRAN release, "Harmony2") by default; `version="1.2"`
  gives the previous behaviour (harmony 1.2), `version="2.1"` the GitHub tag v2.1.0 (per-variable penalties
  multiplied). 2.x differs from 1.2 in its k-means start, fixed centroids during clustering (set from the
  regression intercept), pseudocount penalty, per-cluster removal of absent batch levels (`batch_prop_cutoff`)
  and defaults (`lambda` automatic, 4 clustering iterations): the embeddings differ, so pass `version="1.2"` to
  keep results made with 1.2. `lambda_` and `max_iter_cluster` now default to the version's R defaults.
  `symphony.build_reference` stays on 1.2 (R symphony 0.1.1 was validated with harmony 1.2).
  Validated against each R version with one, two and three batch variables and distinct theta / lambda per
  variable (HARMONY_PARITY.md; `benchmarks/r_reference_harmony.R`, `benchmarks/harmony_parity.py`,
  `tests/test_vs_r.py::test_harmony_versions_track_r`, `tests/test_harmony_versions.py`). The 1.2 port needed no
  change with several variables.

### Added
- Harmony R references get their own location, `ARSC_REF_HARMONY` (default `$ARSC_SCRATCH/ref_harmony`), so they
  can sit where no scratch cleanup prunes them, in a compact layout (gzip; corrected embeddings in float32: about
  115 MB instead of 229 MB). `benchmarks/r_reference_harmony_all.R` rebuilds all three versions (1.2.0, 2.0.5,
  2.1.0), each from its own R library, and checks the version of each output.
  `tests/test_vs_r.py::test_harmony_tracks_r` (one batch variable, each tool from its own start) now uses them for
  all three versions, with R's own seed-to-seed spread as the bar, instead of the full reference-dataset export.
  `refutil.read_dense` reads the gzip layout (`r_common.R::write_dense_gz`).
- `benchmarks/paired_bench.py`: arsc vs scanpy timed in the same session (interleaved, alternating order, the same
  thread count, machine load recorded per run; median and range of repeats; k-NN recall vs brute force, overlap with
  scanpy and scanpy's own seed-to-seed overlap; cluster ARI). `--gate` fails when arsc is slower; the speed
  regression test `tests/test_speed_gate.py` (marker `bench`, `ARSC_SPEED_GATE=1`) runs it.
  `benchmarks/symphony_timing.py`: the same pairing for Symphony against R.
- Subset stores as views (`arsc.views`): a store whose cells are rows of other (parent) stores keeps only what is new
  (metadata, reductions, graphs, neighbors, `uns`, dense layers, features, and every layer whose values differ) and
  reads each layer row that equals a parent row bit for bit from the parent, through a feature map and the store's own
  dtypes, with rows from several parents and rows kept locally. `arsc.open(view)` returns a normal `Store`
  (`is_view`, `view_parents`); `manifest.json` and `cells.txt` stay byte-identical to the full store. `views.build`
  (row candidates in, bit-for-bit row matching, dry run), `views.compare` (every accessor of a view against the full
  store it replaces), `views.materialize` (back to a full store), `views.array_sha256`, `datasets.toy_view`. A view's
  parent that is missing or changed raises `ViewError` at the first matrix read. `st.X()` of an inherited layer
  gathers the view's rows into read-only RAM once per process; `get_features` reads only the requested parent
  columns; `subset` gathers only the requested rows.

### Fixed
- Full Seurat export of a query -> reference Neighbor object (Azimuth / MapQuery `query_ref.nn`): its `nn.idx` values
  index the reference's cells, but the exporter mapped them through the object's own cells, writing NA (-2^31) for
  every reference row beyond the object's cell count and wrong positions for the rest. Such a Neighbor now keeps its
  0-based reference positions with `index_space: "other"` in the manifest (own-cell Neighbors: `"cells"`), and a
  Neighbor whose positions would hold NA is listed as unconvertible instead of written.

### Added
- Full Seurat export: `arsc.export_seurat(obj, out, full=True)` / `arsc.read_seurat(..., full=True)` /
  `export_seurat.R --full` writes everything a Seurat v3/v4/v5 object holds: every assay and layer (v5 split layers
  joined with the per-cell split recorded; a v3 `data` slot that repeats `counts` stored as an alias, not a copy),
  `scale.data` as a dense array (float32 by default, `scale_f64=True`), per-feature metadata, VariableFeatures, every
  meta.data column typed with all factor levels, Idents, reductions with projected loadings, graphs with their assay,
  Neighbor objects, and misc / commands / tools / images / assay extras (SCT models) as `uns`. Anything it cannot
  convert (an R environment, an external pointer) is listed in `manifest["unconvertible"]`. Reads `.rds` files that
  were gzipped a second time (GEO `.rds.gz`).
- `Store.dense()` / `dense_layers()`, `Store.split()`, `Store.neighbors` / `neighbor()`, `Store.idents`,
  `Store.unconvertible`; `Reduction.projected` and `Reduction.key`; `Store.var()` reads a full export's `var.csv`
  (and writes `var.parquet` on finalize); `Store.uns` resolves arrays with shape/levels, data frames and the files
  under `uns/` (npy, csv, mtx, txt).

### Fixed
- `export_seurat.R`: a reduction or graph that does not cover every cell no longer fails with a subscript error in
  full mode (NaN rows, or listed as unconvertible); a typed obs/var csv whose manifest names a column the csv lacks
  reads instead of raising.

## [0.3.0] - 2026-10-21

### Changed (defaults; pass the old values to reproduce results computed with 0.2.x)
- `arsc.umap` runs arsc's own engine by default (`engine="arsc"`; was `"umap-learn"`). Re-checked before the switch
  (SEURAT_PARITY.md section 5): the fuzzy graph is bit-identical to umap-learn's, the layout's trustworthiness and
  kNN preservation equal umap-learn's, and the coordinates are identical at any thread count. `engine="umap-learn"`
  stays available (and is needed for metrics other than cosine / euclidean).
- `arsc.cluster` runs arsc's own engine by default (`engine="arsc"`; was `"leidenalg"`), with Louvain at resolution
  0.6 (`method="louvain"`, `resolution=0.6`; was Leiden at 0.8), Seurat `FindClusters(resolution = 0.6,
  algorithm = 1)`. The ARI of arsc's Leiden / Louvain against leidenalg / igraph lies within the references' own
  seed-to-seed range and labels are identical at any thread count. `engine="leidenalg"` stays available.
- `pp.pca` computes 30 components by default (was 50).
- These defaults are the settings of the CD8 T-cell workflow (Seurat 4) that arsc was validated on: LogNormalize
  (scale factor 1e4), vst variable genes (2,000), 30 PCs, Louvain (`algorithm = 1`) at resolution 0.6, RunUMAP with
  30 neighbours, min_dist 0.3, cosine metric, spread 1, seed 42. `log_normalize`, `hvg_vst`, `find_neighbors`
  (k = 20) and `umap`'s layout parameters already had these values and are unchanged.
- `clonotrace.revise_distance`'s default Leiden ladder pins `method="leiden"`, `engine="leidenalg"` (its documented
  scanpy-style behaviour), so its output does not change.
- To reproduce 0.2.x results: `arsc.cluster(snn, method="leiden", resolution=0.8, engine="leidenalg")`,
  `arsc.umap(emb, engine="umap-learn")`, `arsc.pp.pca(Z, n_comps=50)` (or the values the call passed).

### Added
- `arsc.umap(engine="arsc")` and `arsc.fuzzy_simplicial_set`: arsc's own UMAP. The fuzzy graph is umap-learn's
  (bit-identical weights on the same kNN); the layout runs on all threads with identical coordinates at any thread
  count (snapshot epochs, edges partitioned by head point, per-edge counter-based negative sampling); about 10x
  faster than umap-learn's seeded single-threaded layout at 8 threads, same trustworthiness (SEURAT_PARITY.md
  section 5).
- `arsc.cluster(engine="arsc")`: arsc's own Leiden (a port of libleidenalg's default optimiser) and Louvain (a port of
  igraph community_multilevel) in numba, for modularity and `objective="cpm"`. Quality within the references'
  seed-to-seed range, Leiden 4-6x and Louvain 3-15x faster, labels identical at any thread count (the default
  engine since 0.3.0, see above).

- `arsc.stats.perm_pvalue` (the p of an observed statistic against a null you built: never-zero `(1 + #) / (B + 1)`,
  NaN null values dropped, tie tolerance; `exact=True` for a full enumeration), `strat_perm_test` (two-group
  permutation test with labels permuted within strata: clones within patient, patients within batch; exact
  enumeration or Monte Carlo) and `boot_ci_diff` (percentile bootstrap CI of `f(x) - f(y)`, groups or pairs
  resampled). Validated by brute force (STATS_PARITY.md). They replace hand-written versions found in experiment
  code (a p without the +1, NaN counted as not extreme, permutation across strata that should have been within).
- `arsc.stats` rank, randomisation, stratified and depth helpers (STATS_PARITY.md): `signflip_test` (exact to n = 20,
  never-zero Monte Carlo p, NaN dropped), `signflip_maxt` (Westfall-Young single-step / step-down over K readouts),
  `hodges_lehmann` (estimate + CI + p, port of R `wilcox.test(conf.int = TRUE)`, exact and normal branches),
  `mantel_haenszel` (R `mantelhaen.test`), `icc_oneway` (ICC(1) with strata, F test, CI), `freedman_lane`
  (covariate-adjusted permutation of one coefficient), `downsample_counts` (hypergeometric / binomial thinning),
  `rarefy_richness` (vegan `rarefy`, closed form) and `poisson_binomial`.
- `arsc.symphony`: Symphony reference mapping (port of R symphony 0.1.1, GPL-3): `build_reference`,
  `reference_from_harmony`, `map_query`, `transfer_labels` (kNN with vote-share confidence), `mapping_score`,
  `Reference.save` / `load_reference`, UMAP placement of query cells. Parity in `SYMPHONY_PARITY.md`.
- Seed recording: every stochastic function in SEEDS.md records the seed it used. Pandas results carry
  `attrs["seed"]` / `attrs["seed_function"]`; `arsc.seeds_used()` / `arsc.reset_seeds_used()` give the
  per-process log (outermost calls only, counted per function and seed). Results are unchanged.
- Tests of 1-thread vs N-thread determinism for `knn(method="pynndescent")`, `fgsea`, `propagate_clones` and
  `add_module_score` (`tests/test_seedlog.py`). fgsea, propagate_clones and add_module_score are
  thread-invariant; pynndescent is repeatable at a fixed thread count but not across thread counts
  (documented in SEEDS.md, recorded as an expected failure).

### Changed
- `arsc.knn` (exact): ring search inside each cluster (1.2-1.5x faster on one thread) and ties at the k-th place
  resolved by (distance, index), so the neighbour set no longer depends on the partition seed or scan order; the
  per-row sort no longer allocates inside the parallel loop. `snn_graph` sorts in place; the igraph conversion no
  longer goes through Python lists; `group_singletons` runs in numba (same rule).
- `stats.perm_test`: the sampled (Monte Carlo) p is now `(1 + #extreme) / (B + 1)` and can no longer be 0 (it was
  `#extreme / B`); exact enumeration is unchanged. An empty group or a non-finite observed statistic now gives
  p = NaN (it gave 0). The result gains `exact`. `stats.arm_contrast` drops non-finite values before the
  Hodges-Lehmann shift and the leave-one-out range, as `perm_test` already did for the p.
- `pp.pca` and `run_harmony` are now thread-count-invariant: their large BLAS products run in fixed blocks in a fixed
  summation order (`arsc._detla`), and `eigh` / QR / ARPACK updates / the k-means init run on one BLAS/OpenMP
  thread. Measured before: PCA differed by up to 3e-11 and harmony by 5e-14 to 1.4e-12 between 2 and 8 threads
  (SEEDS.md had claimed harmony was thread-invariant; that test varied only numba threads). Values change at float
  rounding versus earlier runs. Tested 2 vs 8 threads.

### Fixed
- `HarmonyResult` docstring: `R` is `(n_cells, nclust)` and `Y` is `(nclust, d)` (the docstring had them transposed).
- Seurat import: a factor with a single level (e.g. `orig.ident` of a one-sample object) was written by jsonlite
  as a JSON string, and `Store.obs` failed with "Categorical categories must be unique". The exporter now writes
  the levels as an array (`I()`), and the reader also accepts the old string form (stores already on disk).

### Changed
- Licence: arsc is now GPL-3.0-or-later (was labelled MIT). Several modules translate or closely
  follow GPL code (harmony 1.2, limma/edgeR voom chain, presto, UCell). `NOTICE` lists each ported original
  with its licence; the licence texts of the MIT/BSD originals (Seurat, fgsea, Clonotrace, scanpy) are in
  `LICENSES/`.

### Added
- Agent entry points: `AGENTS.md`, `llms.txt`, `CLAUDE.md` and a Claude Code skill, with a
  task-to-function map and runnable recipes.
- Human documentation in `docs/`: getting-started tutorial on a synthetic dataset, API reference
  rendered from docstrings, migration guides from scanpy and from Seurat.
- `arsc.datasets.toy()` / `toy_h5ad()`: small synthetic study (6 patients x pre/post, planted
  effects) for tutorials, examples and tests.
- Docstring standard (summary, parameters, returns, example) enforced by `tests/test_docstrings.py`;
  every code block in the docs is executed by `tests/test_docs_snippets.py`.
- `arsc.help()` (task -> function map) and `arsc.recipes()` (runnable recipes, found from a plain-English
  question); recipes live in `src/arsc/_recipes.py` and `docs/build_docs.py` renders them into the docs.
- `clones.read_10x_contigs`: CellRanger `filtered_contig_annotations.csv` -> one row per barcode with a
  paired `clone_id`. `Dataset.get_features` (same as `Store.get_features`).
- Clearer error messages across the API (missing columns, absent levels, misaligned inputs, unknown
  options, design problems) that say how to fix the problem.
- Reproducible benchmark suite against scanpy and Seurat (`benchmarks/`).
- Packaging for publication: `LICENSE`, `NOTICE` (method credits), `CITATION.cff`,
  `CONTRIBUTING.md`, GitHub Actions CI (tests on Linux and Windows, Python 3.10-3.12; build check),
  `stats` and `clonotrace` extras, project URLs, logo.

### Changed
- Parity documents, docstrings and tests no longer name internal projects, people or local paths.
  Data-dependent tests read their locations from environment variables (`tests/conftest.py`, optional
  untracked `tests/local.env`).
- Gene-set cache: `$ARSC_CACHE`, else `$AR_ROOT/.cache`, else `~/.cache/arsc` (no machine-specific default).
  `genesets.load_v3` needs `ARSC_DATA_V3`; the R scripts add `$ARSC_RLIB` / `$AR_ROOT/.Rlib` to the library
  path only when set.
- `stats.paired_test` raises when `t0`/`t1` is not a level of the time column (it returned an empty result).

## [0.2.0] - 2026-10-14

### Added
- `arsc.read` dispatcher with streamed readers for `.h5ad`, 10x `.h5` (v2/v3, split by feature
  type) and 10x mtx directories; Seurat `.qs`/`.rds` via the bundled R exporter.
- Streamed `.h5ad` writer (`write_h5ad`, `Store.write_h5ad`) and `Store.to_anndata`.
- `arsc.clonotrace`: CloneTrace port (clone probability propagation, clone distances, clone
  pseudotime, fate genes), validated against the R code (`CLONOTRACE_PARITY.md`).
- Seed policy: every stochastic function takes a validated integer seed with a fixed default,
  rejects `None` and never touches the global RNG (`SEEDS.md`, `tests/test_seeds.py`).
- Packaging metadata and extras `h5ad`, `graph`, `all`, `dev`.

### Changed
- Tests skip (rather than error) when R reference outputs or optional packages are missing.

## [0.1.0] - 2026-10-12

### Added
- Memory-mapped store (`arsc.open`): NPY components opened zero-copy, per-feature CSC copies,
  shared across processes through the OS page cache; Seurat v4/v5 export bridge.
- Preprocessing with Seurat semantics: LogNormalize, CLR, vst HVG (R-compatible loess), ScaleData,
  PCA.
- Harmony 1.2 port; presto-equivalent `wilcoxauc`; UCell and AddModuleScore (R-compatible RNG);
  exact kNN, SNN, WNN, Leiden/Louvain and UMAP wrappers; pseudobulk.
- `seurat_extra` (FindMarkers, FindAllMarkers, AverageExpression, CellCycleScoring, ScaleData with
  regression and split.by), `scanpy_extra` (rank_genes_groups, score_genes, HVG flavours,
  neighbors, leiden, umap, regress_out, QC metrics, diffmap, dpt, palantir diffusion maps, MAGIC).
- Patient-level statistics (`arsc.stats`: composition, paired and permutation tests, binomial GLMM,
  LMM, GEE, minimum detectable effect) and pseudobulk DE (`arsc.de`: DESeq2 via pydeseq2,
  limma-voom with duplicateCorrelation, fgsea).
- Clonal methods (`arsc.clones`: Cyclone, healthy-donor expansion null, STARTRAC, diversity and
  dynamics helpers).
- Validation suite against R and scanpy (`tests/`, `*_PARITY.md`).

[Unreleased]: https://github.com/jiwen90/arsc/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jiwen90/arsc/releases/tag/v0.2.0
[0.1.0]: https://github.com/jiwen90/arsc/releases/tag/v0.1.0
