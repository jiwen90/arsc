# CloneTrace parity (`arsc.clonotrace`)

Port of the Clonotrace R package: Fu Y, Mathew D, Wang M, Chen XE, Lin KZ, Schaff D, Shaffer SM,
Pardoll DM, Jackson C, Zhang NR, *Deciphering Cell Fate and Clonal Dynamics via Integrative
Single-Cell Lineage Modeling*, bioRxiv 2025.09.01.673503. The reference is
github.com/yuntianf/Clonotrace, MIT, v1.2.1, commit f0c397a. The R code is sourced, not installed.

**Reference data:** the public demo bundled with Clonotrace (Weinreb et al. 2020, GSE140802). The
script `benchmarks/r_reference_clonotrace.R` writes the reference outputs to
`$ARSC_REFS/clonotrace_ref` (`python benchmarks/regen_refs.py clonotrace_ref`). It uses a deterministic subset: 30 clones with at least 15 cells
plus 1,200 background cells, 2,032 cells in total. To run it, set `CLONOTRACE_SRC` (a checkout of
the repo), `ARSC_REFS`, and `ARSC_RLIB` (an R library with dbscan and transport).

**Tests:** `tests/test_clonotrace.py`, 19 tests. 11 are R-parity tests, which skip when the
references are absent. 8 are synthetic tests that always run.

| step | R function | arsc | agreement |
|---|---|---|---|
| kNN kernel graph | `embedding2knn` | `knn_connectivity` | max diff < 1e-12, same edge set |
| transition | `compute_transition` | `cell_transition` | < 1e-14 |
| label spreading | `label_spreading_blocked` | `label_spreading` | < 1e-13 (block semantics reproduced) |
| bootstrap deviance | `label_spreading_bootstrap` | `propagate_clones` | statistical (RNG differs). R 3x12-bootstrap estimate vs Python 48: r = 0.88 (Python 36 vs 48: 0.92); means within 1.5% |
| sparsify | `mat_sparsify` | `sparsify` | exact (< 1e-14) |
| SNN cost graph | `bluster::makeSNNGraph(type="number")` + `exp(-w)` | `snn_cost_graph` | identical edges, < 1e-15 |
| NN clone distance | `graph_clone_nn` | `clone_distance(method="nn")` | < 1e-12 on the subset. **Full size** (31,433 cells; 8 clones vs current R): 4e-16 |
| exact OT distance | `clone_2_ot` (transport) | `clone_distance(method="exact")` | < 1e-9 (POT if installed, else HiGHS LP) |
| revised distance | `revise_clone_distance` (fixed clusterings) | `revise_distance` | weights and picks equal; curves and matrix < 1e-12 |
| clone Leiden | `leiden_dis` | `cluster_clones` | ARI > 0.8 (RNG differs) |
| classical MDS | `cmdscale` | `cmdscale` | < 1e-10 (up to column sign) |
| non-metric MDS | `MASS::isoMDS` | `isomds` | approximate: a different optimizer; stress within 10% |
| root, DPT | `clone_root`, `clone_dpt` | `clone_root`, `clone_dpt` | same root; pseudotime < 1e-8 |
| profiles, mass | `Q %*% P`, `cluster_profile_mass` | `cell_profiles`, `profile_mass` | < 1e-14 |
| enrichment p | `cluster_profile_enrich` | `profile_enrichment` | statistical (permutations) |
| ns basis | `splines::ns` | `ns_basis` | same column space (< 1e-10); df=1 equal up to sign |
| ridge spline F / LRT | `soft_cluster_gam_fit` | `soft_cluster_fit` | rtol 1e-7 |
| bin filter | `bin_filter_profile_mass` | `bin_filter` | identical, including R's NA-bin behavior |
| co-embedding graph | `cell_clone_coembed` | `coembed_graph` | same edges, < 1e-12 |

## Speed and agreement, R vs arsc, step by step

arsc is 20 times faster than the R code on one thread and 150 times faster on 16 threads over the whole pipeline,
with the same results: on the same inputs every deterministic step agrees with R to 1e-9 or better (most to 1e-12), and the
random steps agree statistically.

- **Data:** a deterministic subset of the public demo: 150 clones of at least 15 cells (seed 7) plus 3,500 other
  cells (seed 8), 7,524 cells; 7,447 cells x 150 clones remain after the deviance filter (11,175 clone pairs).
  This is one mid-size on purpose: large enough that every step does real work, small enough that the R code
  finishes every step (23 minutes in all). The 2,032-cell test subset is too small to time.
- **Same inputs:** each arsc step reads R's output of the step before (R's kNN graph, R's transition matrix, R's
  `cell_clone_prob`, R's distance matrix, R's classical MDS), so every row times one step on identical input and
  the agreement column compares that step alone.
- **R:** R 4.3.1 with its bundled reference BLAS (one thread), Clonotrace v1.2.1 (commit f0c397a) sourced, and
  `future::plan(sequential)`, which is what the package does unless the user sets a parallel plan. One run of each
  step; a step under 20 s was run 3 times and the median is shown. R's timer has a resolution of about 10 ms on
  Windows, so R times below 0.1 s are rough. R ran as an ordinary job: the machine's other work used 26% of the CPU
  on average while it ran. The row "Clonotrace main" is the newer `graph_clone_nn` of commit eba2001 (2026-10-26,
  a distance cache), on the same input.
- **arsc:** commit 993dadc (recorded as 5284f77 in the result files, the same code before a rebase), Python 3.11, numba 0.68, numpy 1.26, scipy 1.17, POT 0.9.7 (exact OT). One process
  per thread count, the thread variables set before import; every step first ran once untimed on the small subset
  (numba compilation excluded); a step under 20 s was timed 3 times (median). The 16-thread and 1-thread runs ran
  under a machine-wide lock (no other job admitted; other processes used 2.6% of the CPU on average).
- **Memory:** the largest increase of resident memory over the step's starting value, sampled every 20 ms (R: the
  R process; arsc: the Python process). Steps shorter than about 0.1 s can be missed by the sampler.
- **Machine:** AMD Threadripper PRO 5955WX, 16 cores, 512 GB RAM, Windows 11.

<!-- CLONOTRACE-SPEED:START -->

#### M: 7,524 cells, 150 clones (7,447 cells x 150 clones after the deviance filter)

| step | R | arsc 1 thread | arsc 16 threads | speed-up, 1 thread | speed-up, 16 threads | R peak mem | arsc peak mem (16 threads) | agreement with R |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| kNN kernel graph (k = 30) | 1.48 s | 0.30 s | 0.25 s | 4.9x | 5.9x | 261 MB | 38 MB | max abs diff 1.6e-14; same edges: yes |
| transition matrix | 10 ms | 10 ms | 9 ms | 1.0x | 1.1x | 0 MB | 0 MB | max abs diff 6.7e-16 |
| label spreading (reference) | 0.89 s | 0.38 s | 61 ms | 2.4x | 14.5x | 129 MB | 21 MB | max abs diff 2.8e-17 |
| spreading + bootstrap deviance | 3 min | 17 s | 3.07 s | 11.5x | 64.8x | 106 MB | 30 MB | corr with R 0.972; mean R 0.183; mean arsc 0.183; pass 0.3 R 7,447; pass 0.3 arsc 7,444 |
| sparsify | 0.62 s | 68 ms | 69 ms | 9.1x | 9.0x | 208 MB | 8 MB | max abs diff 2.2e-16; same shape: yes |
| SNN cost graph (k = 10) | 0.25 s | 93 ms | 65 ms | 2.7x | 3.9x | 117 MB | 6 MB | max abs diff 3.3e-16; same edges: yes |
| NN clone distance, all pairs | 12 min | 27 s | 2.31 s | 26.0x | 299.0x | 67 MB | 18 MB | max abs diff 5.1e-16; same pairs: yes |
| exact OT distance, 9 clones (36 pairs) | 20 s | 6.42 s | 0.76 s | 3.2x | 26.6x | 116 MB | 27 MB | max abs diff 4.7e-16 |
| revised distance (fixed clusterings) | 0.19 s | 60 ms | 62 ms | 3.1x | 3.1x | 26 MB | 1 MB | max abs diff 2.2e-15; same weight and picks: yes |
| clone Leiden | 50 ms | 3 ms | 3 ms | 16.4x | 15.2x | 2 MB | 0 MB | ari vs R 0.940 |
| classical MDS | 0 ms | 2 ms | 4 ms | - | - | 0 MB | 0 MB | max abs diff up to sign 3.2e-16 |
| non-metric MDS (k = 6) | 0.22 s | 89 ms | 0.11 s | 2.5x | 2.0x | 3 MB | 0 MB | stress R 5.231; stress arsc 5.159 |
| root + DPT | 70 ms | 21 ms | 23 ms | 3.3x | 3.1x | 1 MB | 0 MB | same root: yes; max abs diff 4.9e-13 |
| profiles + mass | 10 ms | 14 ms | 12 ms | 0.7x | 0.9x | 0 MB | 1 MB | max abs diff cpp 0; max abs diff mass 2.8e-14 |
| enrichment (200 permutations) | 2.13 s | 0.12 s | 0.12 s | 17.8x | 17.6x | 100 MB | 0 MB | frac p within 0.1 0.992 |
| spline fit, F test (200 genes) | 80 ms | 31 ms | 29 ms | 2.6x | 2.8x | 91 MB | 35 MB | max rel diff stat 2.6e-10; max rel diff p 2.2e-11 |
| spline fit, LRT (200 genes) | 70 ms | 31 ms | 28 ms | 2.3x | 2.5x | 68 MB | 7 MB | max rel diff stat 2.7e-10 |
| co-embedding graph | 8.67 s | 0.33 s | 0.29 s | 26.5x | 29.5x | 101 MB | 68 MB | max abs diff 5.1e-13; same edges: yes |
| NN clone distance, Clonotrace main | 6 min | 27 s | 2.31 s | 12.6x | 144.8x | 109 MB | 18 MB | as the row above |
| **pipeline** (sum of the steps knn, transition, bootstrap, sparsify, snn, nn, leiden, isomds, dpt, profiles) | **15 min** | **45 s** | **5.93 s** | **20.1x** | **150.6x** | 792 MB (process) | 422 MB (process) | |

<!-- CLONOTRACE-SPEED:END -->

Notes on the table:

- **Where arsc is not faster:** "profiles + mass" (a 7,447 x 150 matrix times a clone-to-profile matrix) takes 10-14 ms in both, and
  classical MDS of 150 clones takes a few ms in both; R's 0 ms is below its timer resolution. A few small steps run
  slightly slower on 16 threads than on 1 thread (thread start-up costs more than the work).
- **The two slow R steps** are the bootstrap deviance (48 label-spreading runs; the blocked backend writes each
  bootstrap to an HDF5 file) and the NN clone distance (shortest paths from every clone's cells, then a loop over
  clone pairs in R). Together they are 99% of R's pipeline time (14.8 of 14.9 min).
- **Exact OT** in arsc needs POT (`pip install pot`, or `arsc[clonotrace]`). Without it arsc falls back to scipy's
  HiGHS linear program and warns: on the 2,032-cell subset that took 23 s for the 36 pairs, against 0.5 s with POT
  and 3.1 s in R.
- **Agreement** of the random steps: the bootstrap deviance correlates r = 0.97 with R's (both 48 bootstraps;
  numpy's RNG is not R's), with the same mean (0.183) and 7,444 vs 7,447 cells passing the 0.3 filter; clone Leiden
  ARI 0.94 with R (one seed); permutation p-values within 0.1 of R's for 99% of entries (200 permutations each);
  non-metric MDS reaches a slightly lower stress than `MASS::isoMDS` (5.16 vs 5.23; a different optimizer).
- **Reproduce:** `python benchmarks/clonotrace_bench.py r --size M` (R; writes the R outputs and
  `results/clonotrace/M_r_w1.json`), then `python benchmarks/clonotrace_bench.py arsc --size M --threads 1,16`
  (needs the size S outputs too, for the warm-up: `... r --size S`), then `... render`.

### Earlier full-size check (public demo, 34,782 cells, 802 clones; 32 threads; arsc only)

| step | arsc |
|---|---|
| `cell_transition` (k=30) | 1.6 s |
| reference spreading, 802 clones | 1.3 s |
| spreading + 48 bootstraps | 64 s |
| NN clone distance, 31,433 cells x 802 clones (321k pairs) | 32 s, 2.4 GB peak private |

In that run the R blocked spreading with 4 bootstraps took 6.5 min, and R's NN distance for 8 of the 802 clones
took 12 s.

**Full-size deviance check (4 bootstraps each):**

| | R (current code) | arsc |
|---|---|---|
| mean deviance | 0.236 | 0.237 |
| cells with deviance < 0.3 | 24,584 | 24,664 |
| correlation with a 48-bootstrap arsc estimate | 0.85 | 0.84 |

## Known differences and caveats
- **R dense bootstrap shifts clone columns (Clonotrace 1.0.0; `backend = "dense"` in 1.2.x):** `label_spreading()`
  re-factors its labels (`as.numeric(as.factor(labels))`) even when `label_n` is given. In a bootstrap that drops
  every labeled cell of one clone, every later clone moves one column left and is compared with the wrong column of
  the reference, so the deviance of all cells of later clones rises. Minimal case: labels `1, 1, NA, NA, 3, 3` with
  `label_n = 3` put clone 3's mass in column 2. This matters when small clones are labeled (a 3-cell clone loses all
  its labels in about 0.8% of bootstraps at `sample_rate = 0.8`, so with hundreds of such clones nearly every
  bootstrap shifts). The default blocked backend (1.1.0 and later) keeps the original indices, and so does arsc.
- **R `graph_clone_ot` can loop forever (Clonotrace main, `exact_backend = "memory"`):** `graph_clone_ot_sub` has
  `if (global_id == ncol(cell_clone_prob)) next` before it picks the next clone, so when its greedy order reaches the
  last column while other clones remain, the `while` loop repeats that clone forever (for example 3 clones on a path
  graph where the largest is the last column). The default disk backend does not use this code; the benchmark times
  that backend. arsc's `clone_distance(method="exact")` has no such loop.
- **R `graph_clone_nn` in 1.0.0 with an empty clone:** a clone with no cell at `prob_thresh` fails with "subscript
  out of bounds" (`1:nrow(x)` on a 0-row matrix); 1.2.x stops with a clear message, and so does arsc.
- **R 1.0.0 dense spreading starts from `F = Y`, the blocked solver (and arsc) from `F = 0`:** both stop at the
  same tolerance (1e-3), so their results differ by about that much; arsc matches the blocked solver to 1e-13.
- **Bundled cell-clone matrix:** the bundled `cell_clone_prob.rds` keeps 31,433 cells, but the
  current code, in R or in Python, keeps about 24.6k cells at deviance < 0.3. The bundled
  matrix therefore predates the current defaults.
- **Bundled distance file:** the bundled `inst/extdata/clone_graph_dis.tsv` is not reproduced by
  the current code, in R or in Python. It correlates r = 0.90 with the current NN distance, and it
  is not the exact OT distance either. It was presumably produced by an earlier version or with
  other settings; arsc matches the current R code.
- **Clone order:** R orders clones with `as.factor`, which follows the session locale. arsc uses
  code-point order by default; pass `levels=` to reproduce a given R run.
- **Default revised distance:** `revise_distance` without `clusters` builds its own Leiden ladder
  (scanpy-style UMAP graph + leidenalg). The partitions are not R's, so pass fixed clusterings
  for reproducibility.
- **Not ported:** `diffusionMap::diffuse` (visualization only; archived on CRAN) and the R
  `umap` layout of `leiden_dis`. For layouts, use `arsc.umap` or `umap-learn` with
  `metric="precomputed"` on `coembed_graph`.
