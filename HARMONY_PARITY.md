# Harmony parity (arsc.run_harmony vs R harmony 1.2.0, 2.0.5, 2.1.0)

`arsc.run_harmony(version=...)` reproduces three R releases of harmony (immunogenomics/harmony, GPL-3):

| `version` | R release | Where it comes from | Status |
|---|---|---|---|
| `"2.0"` (default) | 2.0.5, 2026-06-12 | CRAN, current release | "Harmony2" rewrite, first on CRAN as 2.0.2 (2026-04-25) |
| `"2.1"` | 2.1.0, 2026-10-15 | GitHub tag `v2.1.0` (= `master` on 2026-10-25), not on CRAN yet | 2.0.5 plus PR #292: per-variable penalties multiplied instead of summed |
| `"1.2"` | 1.2.0 (1.2.1, 1.2.3, 1.2.4 identical algorithm) | CRAN archive | the Seurat-era release; used by `arsc.symphony` |

Checked on 2026-10-25 against the CRAN page and archive (1.2.0 2023-11-29, 1.2.1 2024-08-28, 1.2.3 2024-11-28,
1.2.4 2025-10-10, 2.0.2 2026-04-25, 2.0.3 2026-05-13, 2.0.4 2026-06-04, 2.0.5 2026-06-12) and the GitHub tags
(`v2.1.0`, `0.1`) and commit history (head `7e21bd66`, "Version bump", 2026-10-15). The package NEWS stops at
1.2.0 (plus a 2.1.0 entry on GitHub), so the differences below come from diffing the sources:

* 1.2.0 -> 1.2.4: documentation, Makevars and timer files only; no algorithmic change.
* 1.2.4 -> 2.0.2 (Harmony2, `R/ui.R`, `src/harmony.cpp`, `src/utils.cpp`): k-means++-style seeding plus 10 Lloyd
  steps (`arma::kmeans`) instead of `stats::kmeans` with 10 starts; centroids fixed during clustering and set from
  the per-cluster regression intercept; R recomputed from the corrected embedding at the start of each round;
  pseudocount penalty `((2E + 1) / (O + E + 1)) ** theta` and cross-entropy `log((O + E + 1) / (2E + 1))`;
  per-cluster removal of batch levels with share `O[k, b] / N_b <= batch.prop.cutoff` (new option, 1e-5) and of
  variables left with one level; new defaults `lambda = NULL` (automatic `alpha * E`) and
  `max.iter.cluster = 4` (was 20; 4 iterations never reach the clustering convergence check); objective sums
  accumulated serially in float; `block.size = 1` allowed; float32 scalars where the BLAS has single-precision
  routines (R on Windows: double, as here).
* 2.0.2 -> 2.0.5: fix of the covariate bounds in the level-removal step (multi-variable runs) and of the seeding
  index range; startup message.
* 2.0.5 -> 2.1.0: the per-variable penalties of a cell are multiplied (2.0.x and 1.2 sum them), so a variable with
  `theta = 0` no longer changes the clustering.

## Validation

Data: the 38,440-cell reference gamma-delta T-cell CITE-seq dataset, its 25-dim PCA (store reduction) with three
batch variables: patient (32 levels), 10x lane (111 levels, 1 to 12,977 cells) and patient x timepoint (129
levels). Input written by `benchmarks/harmony_input.py`; R references by `benchmarks/r_reference_harmony.R` (R 4.3.1;
harmony 1.2.0 in the framework library, 2.0.5 and 2.1.0 installed from the CRAN tarball / GitHub tag into
separate libraries); comparison by `benchmarks/harmony_parity.py` (`benchmarks/results/harmony_parity.json`), asserted
by `tests/test_vs_r.py::test_harmony_versions_track_r` (15 cases). Each case runs with the version's own defaults
except where listed (`lambda` "default" = 1 for 1.2, automatic for 2.x).

The R side seeds with `set.seed(1)`. **From R's start**, arsc gets R's k-means centroids (`init_centroids`), so
only the random block order differs (R's RNG vs numpy). **From arsc's start** (`seed=1`), the two runs start from
different centroids; the reference for that comparison is R itself run with `set.seed(2)`.

Per-dimension Pearson r between the corrected embeddings (min / median over the 25 dims), final objective
and harmony rounds (R vs arsc):

| R harmony | batch variables (theta, lambda per variable) | r from R's start | objective (rounds) from R's start | r from arsc's start | R seed 2 vs R seed 1 |
|---|---|---|---|---|---|
| 1.2.0 | lane | 0.9841 / 0.9991 | 702.8 vs 701.8 (10 vs 10 rounds) | 0.979 / 0.995 | 0.976 / 0.995 |
| 1.2.0 | patient + lane | 0.9976 / 0.9996 | 948.1 vs 944.3 (9 vs 10 rounds) | 0.949 / 0.994 | 0.938 / 0.995 |
| 1.2.0 | patient + lane, theta 4 / 1, lambda 0.5 / 2 | 0.9992 / 0.9998 | 1129.7 vs 1130.3 (7 vs 7 rounds) | 0.985 / 0.997 | 0.988 / 0.996 |
| 1.2.0 | patient + lane, theta 1 / 3, lambda auto | 0.9997 / 0.9999 | 1376.2 vs 1376.3 (5 vs 5 rounds) | 0.984 / 0.997 | 0.993 / 0.996 |
| 1.2.0 | patient + lane + patient x timepoint, theta 1 / 2 / 0.5 | 0.9992 / 0.9999 | 929.2 vs 929.4 (8 vs 8 rounds) | 0.979 / 0.995 | 0.986 / 0.995 |
| 2.0.5 | lane | 0.9992 / 0.9997 | 769.0 vs 769.1 (5 vs 5 rounds) | 0.977 / 0.992 | 0.975 / 0.993 |
| 2.0.5 | patient + lane | 0.9996 / 0.9998 | 801.0 vs 801.2 (5 vs 5 rounds) | 0.975 / 0.992 | 0.965 / 0.992 |
| 2.0.5 | patient + lane, theta 4 / 1, lambda 0.5 / 2 | 0.9995 / 0.9999 | 445.3 vs 445.0 (10 vs 10 rounds) | 0.960 / 0.992 | 0.963 / 0.990 |
| 2.0.5 | patient + lane, theta 1 / 3, lambda auto | 0.9998 / 0.9999 | 801.1 vs 801.1 (5 vs 5 rounds) | 0.969 / 0.992 | 0.963 / 0.992 |
| 2.0.5 | patient + lane + patient x timepoint, theta 1 / 2 / 0.5 | 0.9995 / 0.9999 | 881.1 vs 881.1 (4 vs 4 rounds) | 0.964 / 0.991 | 0.971 / 0.992 |
| 2.1.0 | lane (one variable: identical to 2.0.5) | 0.9992 / 0.9997 | 769.0 vs 769.1 (5 vs 5 rounds) | 0.977 / 0.992 | 0.975 / 0.993 |
| 2.1.0 | patient + lane | 0.9979 / 0.9996 | 792.1 vs 792.4 (5 vs 5 rounds) | 0.980 / 0.992 | 0.969 / 0.991 |
| 2.1.0 | patient + lane, theta 4 / 1, lambda 0.5 / 2 | 0.9969 / 0.9995 | 442.7 vs 442.2 (9 vs 9 rounds) | 0.956 / 0.991 | 0.936 / 0.988 |
| 2.1.0 | patient + lane, theta 1 / 3, lambda auto | 0.9944 / 0.9993 | 789.7 vs 789.8 (5 vs 5 rounds) | 0.975 / 0.991 | 0.969 / 0.991 |
| 2.1.0 | patient + lane + patient x timepoint, theta 1 / 2 / 0.5 | 0.9955 / 0.9995 | 873.1 vs 873.1 (4 vs 4 rounds) | 0.977 / 0.991 | 0.963 / 0.991 |

The objective trajectories coincide round by round from R's start (2.0.5, patient + lane with theta 4 / 1:
R 1251.7, 1120.7, 695.9, 559.2, 508.5, ..., 445.3; arsc 1251.7, 1120.6, 695.9, 559.3, 508.7, ..., 445.0). From
arsc's start the agreement with R equals R's own agreement across two seeds, so the residual is the stochastic
start and block order, not the algorithm. The test asserts, from R's start, per-dim r > 0.98 on every dim,
rounds within 1 and the final objective within 1 %.

Multi-variable 1.2 (the existing port, before this table only validated with one variable): the cases with
distinct theta and lambda per variable, three variables and automatic lambda agree with R as closely as the
single-variable case, so theta / lambda follow the `vars_use` order as in R; no change to the 1.2 code was needed.

Not matched bit for bit: R's k-means start and block shuffling use R's RNG (pass R's centroids to start
identically); harmony 2.x's serial float32 objective sums are emulated in the same order. `tests/test_harmony_versions.py` covers the version switches, the version
defaults, `block_size = 1` (2.x only), `batch_prop_cutoff` and R 2.1.0's own unit test (a variable with
`theta = 0` leaves the clustering unchanged in 2.1, not in 2.0).
