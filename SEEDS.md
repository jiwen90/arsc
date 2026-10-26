# Seeds and determinism

arsc results must be exactly reproducible. Every function that uses randomness, directly or through a
dependency, takes an explicit integer seed with a fixed default. Tests: `tests/test_seeds.py`.

## Rules

1. **The seed is a parameter, and it always has a fixed default.** Where arsc reproduces an R or scanpy
   function, the parameter keeps that function's name and default: Seurat `seed.use = 42`
   (`pca`, `umap`), `random.seed = 0` (`cluster`), `seed = 1` (`AddModuleScore`), `random.seed = 1`
   (`FindMarkers`), scanpy `random_state = 0`, palantir `seed = 0`. The other functions use the defaults
   they always had.
2. **`None` is rejected.** `seed=None` would mean fresh entropy on every call, which gives a different
   answer each time. `arsc._utils.check_seed` raises `ValueError` for `None`, `TypeError` for a
   non-integer, and `ValueError` outside `[0, 2**32)`. Functions that take an R `set.seed` value accept
   any 32-bit signed integer, so negative seeds work there as they do in R.
3. **Global RNG state is never touched.** arsc never calls `np.random.seed` or `random.seed`. Each call
   builds a local `np.random.Generator` or `RandomState`, an R-compatible `_rrng.RRandom`, or a private
   `random.Random` for igraph, and passes it down. Where igraph needs an RNG installed, the private one is
   installed only for that call, and igraph's default (the `random` module) is put back in a `finally`
   block. The test suite checks that numpy's and Python's global RNG state is unchanged after every
   function.
4. **Seeds reach every dependency:** sklearn `random_state`, umap-learn `random_state` and
   `transform_seed`, leidenalg `seed`, pynndescent `random_state`, the igraph RNG, ARPACK `v0`, and the
   numba kernel in fgsea, which uses an explicitly seeded splitmix64 stream per permutation.
5. **One seed per pipeline.** A wrapper that runs several stochastic steps takes one `seed` and passes
   the same value to each step. For example, `clonotrace.run(seed=s)` uses `s` for the bootstrap, the
   Leiden ladder and the clone Leiden; `umap(seed=s)` uses it for the kNN and the layout;
   `wnn`/`find_neighbors` pass it to the kNN. So one seed reproduces the whole pipeline, and changing it
   changes every stochastic step.
6. **To test stability, vary the seed explicitly:**

   ```python
   layouts = {s: arsc.umap(emb, seed=s) for s in (1, 2, 3)}   # three layouts, each reproducible
   labels  = {s: arsc.cluster(snn, seed=s) for s in range(10)} # clustering stability (e.g. pairwise ARI)
   ```

## Function table

"Bit-identical" means that the same input and the same seed give `np.array_equal` output (tested), on
the same machine and library versions. "Thread-invariant" means the output also does not depend on the
number of threads.

| Function | Seed parameter (default) | What the seed drives | Determinism |
|---|---|---|---|
| `pp.pca` | `seed` (42) | ARPACK start vector (`method="arpack"`), Gaussian test matrix (`"randomized"`); `"gram"` ignores it | bit-identical and thread-invariant (BLAS products in fixed blocks, `arsc._detla`; 2 vs 8 threads tested). arpack and gram converge to the same subspace for any seed. randomized depends on the seed by design |
| `run_harmony` | `seed` (0) | k-means start (1.2: sklearn `KMeans(random_state=seed)`; 2.x: the k-means++-style seeding draws from a local `default_rng`); block order each clustering round (local `default_rng`) | bit-identical and thread-invariant (fixed numba reduction chunks, fixed-block BLAS products, k-means on one thread; 2 vs 8 threads tested, see below). `init_centroids` bypasses k-means |
| `knn` | `seed` (0) | `"exact"`: k-means partition used only for pruning, so the answer is exact and the same for every seed (tested). `"pynndescent"`: `random_state` | bit-identical. pynndescent output depends on the seed and on the thread count (see below) |
| `find_neighbors` | `seed` (0) | passed to `knn` | as `knn` |
| `wnn` | `seed` (0) | passed to `knn` for every modality | as `knn` |
| `cluster` | `seed` (0) | `engine="arsc"` (default): numpy `SeedSequence(seed)`, Louvain start `s` its `s`-th child. `engine="leidenalg"`: Leiden uses leidenalg `seed`; Louvain start `s` uses a private `random.Random(seed + s)` as the igraph RNG | bit-identical; arsc engine also thread-invariant |
| `umap` | `seed` (42) | `engine="arsc"` (default): spectral-init noise and per-edge counter-based negative-sampling streams; `engine="umap-learn"`: `random_state` and `transform_seed`; also the `knn` seed | bit-identical; arsc engine identical at any thread count, umap-learn runs its layout single-threaded when seeded. `seed=None`, the old fast parallel mode, is no longer allowed |
| `scanpy_extra.score_genes` | `random_state` (0) | control-gene sampling from a local `RandomState(random_state)`. This is the same MT19937 stream scanpy draws after `np.random.seed`, so scanpy parity is kept (tested) | bit-identical |
| `scanpy_extra.neighbors` | `random_state` (0) | passed to `knn` (matters only for `method="pynndescent"`) | bit-identical |
| `scanpy_extra.leiden` | `random_state` (0) | leidenalg `seed`; with `flavor="igraph"`, a private `random.Random` as the igraph RNG | bit-identical |
| `scanpy_extra.umap_from_connectivities` | `random_state` (0) | local `RandomState` for spectral-init noise and SGD (`parallel=False`) | bit-identical |
| `scanpy_extra.diffmap` | `random_state` (0) | ARPACK `v0` (local `RandomState`) | bit-identical. A different seed changes only ARPACK's convergence path (differences at the solver tolerance) |
| `scanpy_extra.palantir_diffusion_maps` | `seed` (0) | ARPACK `v0` (`default_rng(seed).random`, as palantir); `knn` seed | bit-identical |
| `add_module_score` | `seed` (1, R seed) | R-compatible RNG (`_rrng.RRandom`): bin tie-break noise and control-gene draws, identical to R `AddModuleScore(seed = s)` | bit-identical, equal to R, thread-invariant (tested) |
| `cell_cycle_scoring` | `seed` (1, R seed) | passed to `add_module_score` | bit-identical, equal to R |
| `find_markers` | `random_seed` (1, R seed) | `max_cells_per_ident` downsampling (R-compatible RNG); not used otherwise | bit-identical, equal to R |
| `find_all_markers` | `random_seed` (1, R seed) | passed unchanged to each per-cluster `find_markers`, as Seurat does | bit-identical |
| `stats.boot_ci` | `seed` (1) | bootstrap resamples (local `default_rng`) | bit-identical |
| `stats.paired_test` | `seed` (1) | `boot_ci`, same seed for every group | bit-identical |
| `stats.perm_test` | `seed` (1) | sampled permutations when `choose(n_x+n_y, n_x) > B`; exact enumeration otherwise | bit-identical |
| `stats.arm_contrast` | `seed` (1), **new parameter** | passed to `perm_test` (it used `perm_test`'s default 1 before) | bit-identical |
| `stats.signflip_test` | `seed` (1) | random sign patterns when n > `max_exact` (20); exact enumeration otherwise | bit-identical |
| `stats.signflip_maxt` | `seed` (1) | random whole-row sign patterns when n > `max_exact` (16) | bit-identical |
| `stats.freedman_lane` | `seed` (1) | random residual permutations when n > `max_exact` (8) | bit-identical |
| `stats.downsample_counts` | `seed` (1) | hypergeometric (or binomial) thinning draws | bit-identical |
| `stats.strat_perm_test` | `seed` (1) | random within-stratum permutations when the arrangements exceed `max_exact` (20,000); exact enumeration otherwise | bit-identical |
| `stats.boot_ci_diff` | `seed` (1) | bootstrap resamples within each group (or of pairs) | bit-identical |
| `de.fgsea` | `seed` (1) | permutation `i` for set size `k` uses splitmix64 seeded from `(seed, k, i)` inside the numba `prange` kernel | bit-identical and thread-invariant (one independent stream per permutation) |
| `clones.clone_expansion_null` | `seed` (1) | patient and healthy-donor downsampling (local `default_rng`, consumed in input order) | bit-identical for the same input row order |
| `clones.startrac` | `seed` (1) | optional per-group `downsample` | bit-identical |
| `clones.clone_diversity` | `seed` (1) | rarefaction (multivariate hypergeometric draws, sorted sample order) | bit-identical |
| `clonotrace.propagate_clones` | `seed` (0) | bootstrap label subsets (local `default_rng`) | bit-identical and thread-invariant (tested). Label spreading is a deterministic numba kernel with per-row sums |
| `clonotrace.revise_distance` | `seed` (0) | leidenalg seed of the default Leiden ladder (not used when `clusters` or `mean_weight` is given) | bit-identical |
| `clonotrace.cluster_clones` | `seed` (0) | private `random.Random(seed)` as the igraph RNG. This used to be `random.seed` on the global RNG, then restored; the stream is identical, so the labels are unchanged | bit-identical |
| `clonotrace.profile_enrichment` | `seed` (0) | label permutations (local `default_rng`) | bit-identical |
| `clonotrace.fate_genes` | `seed` (0) | permutation null (local `default_rng`) | bit-identical |
| `clonotrace.run` | `seed` (0) | one seed for the whole pipeline: `propagate_clones`, `revise_distance`, `cluster_clones` | bit-identical |
| `symphony.build_reference` | `seed` (111, R symphony's `seed`) | one seed for the build: Harmony (k-means init, block order), the soft k-means when `vars=None`, the UMAP layout | bit-identical. R's `set.seed(111)` drives R's own k-means, so R and arsc references differ at Harmony's stochastic steps (SYMPHONY_PARITY.md) |
| `symphony.transfer_labels` | `seed` (0, R `knnPredict(seed = 0)`) | breaks kNN vote ties (local `default_rng`) | bit-identical; a different seed changes only tied votes |

These functions are deterministic and take no seed: `clones.pam` (BUILD + SWAP), `cyclone`,
`clonotrace.isomds` (L-BFGS from `cmdscale`), `clone_dpt` and `clone_distance` (`eigh` / Dijkstra / EMD
LP), `markers`, `pseudobulk`, `hvg_vst`, `scale`, `glmm_prop` / `gee_prop` / `lmm_score`, `deseq2` /
`voom`, `rank_genes_groups`, `highly_variable_genes`, `dpt`, and `magic_impute`.

## Recording the seed

Every function in the table is wrapped by `arsc._seedlog.records_seed`, so the seed a call used is kept
without any extra code (tests: `tests/test_seedlog.py`):

* A pandas result (DataFrame or Series) gets `result.attrs["seed"]` and `result.attrs["seed_function"]`.
* `arsc.seeds_used()` returns the seeds used in this process, one entry per function and seed value:
  `[{"function": "stats.paired_test", "param": "seed", "seed": 1, "calls": 4}, ...]`. Only the
  outermost call is logged: `paired_test` calling `boot_ci`, or `clonotrace.run` calling
  `propagate_clones`, is one entry (they pass the same seed down, rule 5). Calls that raise are not
  logged. `arsc.seeds_used(reset=True)` reads and clears the log; `arsc.reset_seeds_used()` clears it.
* The log is in memory only and touches no RNG. A run-config writer (for example the AutoResearch
  `arseed.py`) reads it at the end of a script and records it next to the results.

## Parallelism and floating-point order

* The other numba `prange` kernels (normalization, markers, pseudobulk, kNN, SNN, label spreading,
  Dijkstra, fgsea null) give each iteration its own output rows or columns, and each sum runs
  sequentially inside one iteration, so the result is the same for any thread count.
* **harmony** had cross-thread reductions whose chunk count followed the thread count (`get_num_threads()`
  for the objective, `min(8, threads)` for the R update). That made the float32 objective, and so the
  convergence decisions, depend on the number of threads. The chunk counts are now fixed (32 and 8).
  That test varied only the numba threads, so it missed the BLAS products: measured with 2 vs 8 BLAS
  threads, harmony still differed by 5e-14 (and up to 1.4e-12 on a 9000-cell test), from
  `R.T @ Z_cos` (a sum over all cells, which multithreaded BLAS splits by thread count) and the k-means init.
  Fixed in this revision (next item); the 2 vs 8 thread test now covers BLAS and numba together.
* **BLAS** products over a long dimension (`Z.T @ Z` in `pca`, `R.T @ Z` in harmony) are summed in an
  order that depends on the BLAS thread count (measured, 2 vs 8 threads: PCA max |diff| 6e-12 gram,
  3e-11 arpack, 2e-11 randomized). `pca` and `run_harmony` now compute their large products with
  `arsc._detla.matmul`: fixed 4096-row blocks, each on single-threaded BLAS, run in parallel on arsc's
  threads and summed in block order; `eigh`, QR, ARPACK's own updates, `inv` and the sklearn k-means init run
  with BLAS/OpenMP pinned to one thread. Results are bit-identical for any thread count (tested 2 vs 8,
  `tests/test_seedlog.py`) and as fast or faster (50,000 x 2,000 Gram: 0.98 s blocked vs 1.78 s plain BLAS
  at 8 threads; `eigh` of 2,000 x 2,000: 1.4 s on one thread vs 3.4 s on eight). Values differ from earlier
  runs at float rounding (about 1e-10 absolute on the Gram matrix), which can move a harmony convergence
  decision. Across different BLAS builds, only agreement to about 1e-12 relative is expected. Other `@`
  products in arsc (small, or row-wise) are not covered by this guarantee.
* **pynndescent** with `random_state` set is deterministic in the tested version (0.6) for a fixed thread
  count, but **not thread-invariant**: its parallel NN-descent updates depend on the numba thread count
  (measured on 600 synthetic cells, k = 10: 5 of 6000 neighbor entries differ between 1 and 4 threads;
  1 and 2 threads agreed). Record `arsc.get_threads()` with the seed, or use `knn(method="exact")`, when the
  graph must not depend on threads. Across versions, only statistical agreement is guaranteed.

## Behavior changes in this revision (could alter previous results)

* `seed=None` / `random_state=None` now raises instead of drawing unseeded. This affects `umap`,
  `add_module_score` and `cell_cycle_scoring`, `score_genes`, `umap_from_connectivities`, and every other
  function in the table. Earlier calls with `None` were not reproducible anyway.
* `score_genes` no longer reseeds numpy's **global** RNG. The control genes for a given `random_state`
  are unchanged. Code that relied on the side effect, by drawing from `np.random` after the call, will
  now see different global draws.
* harmony uses fixed reduction chunks. Results are unchanged on a 32-thread machine with `n_threads`
  unset. Runs with fewer than 32 threads, or with `n_threads < 32`, may differ from earlier runs by
  float-rounding (objective sums in a different order), which can shift a convergence decision.
* All other defaults are unchanged. New seed parameters (`find_neighbors`, `wnn`,
  `scanpy_extra.neighbors`, `arm_contrast`) default to the value that was used internally before.
