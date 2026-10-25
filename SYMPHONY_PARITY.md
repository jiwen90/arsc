# Symphony parity (`arsc.symphony` vs R `symphony` 0.1.1)

`arsc.symphony` ports Symphony reference mapping (Kang et al. 2021, Nat Commun 12:5890; R package `symphony`
0.1.1, GPL-3): `build_reference` (`buildReference`), `reference_from_harmony` (`buildReferenceFromHarmonyObj`),
`map_query` (`mapQuery`), `transfer_labels` (`knnPredict`), `mapping_score` (`calcPerCellMappingMetric`). Measured
on 2026-10-16 with R 4.3.1, symphony 0.1.1, harmony 1.2.0, uwot 0.2.2, against arsc on Windows (AR_THREADS 8).

Reproduce: `Rscript benchmarks/r_reference_symphony.R` (writes `$ARSC_REFS/symphony_ref`; or `python benchmarks/regen_refs.py symphony_ref`), then
`python benchmarks/symphony_parity.py`; `tests/test_symphony.py::test_parity_with_r_given_r_reference` asserts the
deterministic rows.

## 1. Internals on random inputs (exact)

| Step (R function) | max abs difference |
|---|---|
| `soft_cluster` (query soft assignment) | 3.6e-15 |
| `compute_ref_cache` (`Nr`, `C`) | 3.6e-14 |
| reference centroids (`cosine_normalize(R Z_corr')`) | 1.2e-15 |
| `moe_correct_ref` (mixture-of-experts query correction, 3-level batch) | 8.0e-15 |
| `scaleDataWithStats` (clip at both tails) | 4.4e-16 |
| `soft_kmeans_cpp` (4 rounds, same initial centroids) | 3.1e-14 |
| `calcPerCellMappingMetric` | 9.8e-15 |

## 2. Mapping a query onto R's own reference (deterministic steps, exact)

Symphony vignette data (`pbmcs_exprs_small`, 1,767 genes): reference = donors 3pv1 + 3pv2 (750 cells, Harmony on
`donor`), query = donor 5p (450 cells, `vars = "donor"`). arsc wraps R's reference matrices with
`reference_from_harmony` and maps the query itself.

| Quantity | arsc vs R |
|---|---|
| centroids; `Nr` / `C` | max abs diff 1.9e-15; 5.7e-13 |
| query projection `Zq_pca` | 1.3e-13 |
| query soft clusters `R` | 3.2e-15 |
| corrected query embedding `Z` | 1.1e-13 |
| kNN labels (k = 5) / `prob` | identical (450/450) / max abs diff 0 |
| mapping score | 6.5e-9 |

## 3. End to end (arsc builds its own reference)

The stochastic steps are Harmony's k-means start and block order (R's RNG vs numpy) and the UMAP layout
(uwot vs umap-learn), so these rows are correlations, not equalities.

| Quantity | Vignette data (750 ref / 450 query) | clinical CD8 T-cell subsample (6,000 ref / 2,000 query) |
|---|---|---|
| variable genes (2,000) | identical, same order | identical, same order |
| gene means / s.d.s | max abs diff 9.3e-15 | - |
| loadings, reference PCs (per PC, abs r) | min 1.000000 | min 1.000000 |
| reference Harmony embedding (per dim r, median / min) | 0.998 / 0.986 | 0.993 / 0.992 |
| corrected query embedding (per dim r, median / min) | 0.999 / 0.996 | 0.998 / 0.994 |
| kNN label agreement arsc vs R | 0.991 | 0.888 (0.996 where both `prob` >= 0.8, 53 % of cells) |
| R vs R with another seed (calibration) | - | 0.892 (0.999 where both `prob` >= 0.8) |
| accuracy vs annotation, arsc / R | 0.935 / 0.935 (`cell_type`) | - |
| `prob` Pearson r arsc vs R | 0.959 | 0.807 (R vs R other seed: 0.836) |
| mapping score Spearman arsc vs R | 0.992 | - |
| UMAP placement: label of the 10 nearest reference cells on each package's own UMAP, agreement / accuracy arsc, R | 0.967 / 0.942, 0.937 | - |
| time, build + map (arsc incl. first-call compilation; R) | 31.5 s; 5.3 s | 7.7 s; 12.4 s |
| time, build + map + kNN labels, paired (vignette; `benchmarks/symphony_timing.py`, 2026-10-25, 3 threads, 3 interleaved runs each, after one untimed run in the same process) | arsc 3.40 s (3.22-3.62), R 4.00 s (3.84-4.01) | - |
| first call in a fresh process (same runs) | arsc 31-32 s, R 4.3-4.7 s | - |

The earlier 31.5 s for arsc on the vignette was its first call: about 28 s of it is the one-off numba compilation of
umap-learn (the reference UMAP model and the query placement use umap-learn, whose functions are not cached on disk),
which R's uwot does not pay. Warm, arsc is 1.2x faster than R on the vignette and 1.6x on the clinical case. Plan for
the first call: build the reference UMAP with arsc's own engine (compiled once and cached on disk), which needs a port
of umap-learn's `transform` for the query cells.

Clinical case (a longitudinal CD8 T-cell dataset): a 6,000-cell pan-T-sort reference (Harmony on patient x timepoint, d = 30, K = 100) and a 2,000-cell
CD8-sort query (`vars` = patient x timepoint x sort, sigma 0.1), labels = the reference's 9 CD8 clusters, genes
detected in >= 10 of the 8,000 cells. CD8 clusters lie on a continuum, so about half the query cells have mixed
neighbourhoods; there arsc and R disagree as often as R disagrees with itself under another seed.

## Differences that remain

* Harmony starts from sklearn k-means and numpy block order, so references built by arsc and R differ at the
  stochastic steps (same objective; see the harmony row of README.md). Pass R's Harmony output through
  `reference_from_harmony` to reproduce an R reference exactly.
* With harmony 1.2, R symphony's `max.iter.harmony = 20` and `tau` are ignored (harmony warns), so R runs
  10 rounds with tau 0; arsc's defaults (`max_iter = 10, tau = 0`) match what R actually runs.
* `transfer_labels` breaks vote ties with numpy, not R's RNG, and does not widen the vote to ties at the k-th
  distance (`class::knn` does); neither occurred in the vignette case (labels identical given R's reference).
* A zero-s.d. gene scales to 0 (R gives NaN; reference variable genes never have zero s.d.).
* UMAP: umap-learn with uwot's settings (30 neighbours, cosine, min_dist 0.1, learning rate 0.5, spectral
  init); coordinates differ from uwot, neighbourhoods agree (table).
* Not ported: `vargenes_method = "mvp"`, `calcPerClusterMappingMetric`, the k-NN correlation metrics.
