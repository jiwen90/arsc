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

**Tests:** `tests/test_clonotrace.py`, 18 tests. 11 are R-parity tests, which skip when the
references are absent. 7 are synthetic tests that always run.

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

## Speed (full public demo: 34,782 cells, 802 clones; 32 threads)

| step | arsc |
|---|---|
| `cell_transition` (k=30) | 1.6 s |
| reference spreading, 802 clones | 1.3 s |
| spreading + 48 bootstraps | 64 s |
| NN clone distance, 31,433 cells x 802 clones (321k pairs) | 32 s, 2.4 GB peak private |

R's NN distance for 8 clones alone took 12 s on the same graph. The R blocked spreading with 4
bootstraps took 6.5 min; arsc does 48 bootstraps in 64 s.

**Full-size deviance check (4 bootstraps each):**

| | R (current code) | arsc |
|---|---|---|
| mean deviance | 0.236 | 0.237 |
| cells with deviance < 0.3 | 24,584 | 24,664 |
| correlation with a 48-bootstrap arsc estimate | 0.85 | 0.84 |

## Known differences and caveats
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
