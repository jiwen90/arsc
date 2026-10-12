# arsc.clones / arsc.genesets: parity with R

Python port of the TCR / clonal methods in `tools/R/ar_methods.R` (+ clone helpers of `tools/R/ar.R`). No plotting.
Data: CD8 FORCYCLONE meta (83,287 cells, 18 patients, 32,080 clones; `cache/cd8_nn_labeled_FORCYCLONE_FINAL_cyclone.meta.csv.gz`).
R references: `scratch/arsc/clones_ref.R` -> `scratch/arsc/clones_ref/`. Tests: `tests/test_clones.py` (23 pass, ~13 s).
Timing: `benchmarks/bench_clones.py`.

| Python | R reference | Agreement |
|---|---|---|
| `cyclone` (hclust, scale timepoint/clone/none, pam, absolute, custom_order, outlier_sd, reg, stats_only) | `ar_cyclone` (port of published `cyclone_find_trajectories`) | **Published assignment reproduced exactly**: 32,080/32,080 clones and all 83,287 cell labels identical to the published `k`, ARI 1.0 (500 clustered clones: Traj1-6 = 111/29/104/104/64/88 + 8 outlier "Other"). All 7 settings variants match R exactly (clone labels, per-patient counts, perc to 5e-14, correlation matrix to 2e-15, delta_fc/delta_c). |
| `cyclone_patient_table`, `cyclone_states` | same names | exact (counts equal; values 5e-15) |
| `read_10x_clonotypes`, `pair_test` (signed Yates chi + Fisher), `fisher_exact_2x2`, `p_adjust_bh` | `ar_read_10x_clonotypes`, `.ar_pair_test` | HD depth 5313/3118 identical; chi 5e-15, Fisher p 2e-11 (own hypergeometric sum, R relErr 1+1e-7), BH 2e-15; checked on 2,205 patient clones and 2,584 HD clones |
| `clone_expansion_null` | `ar_clone_expansion_null` | Stochastic (downsampling): numpy RNG, not bitwise equal to R. Deterministic columns identical (N_1, N_2, n_fdr_up/down; n_clones within 0.75%). Over 54 patient-steps (reps=5): mean cutoff 0.7540 vs R 0.7500 (Pearson 0.945); n_sig_up mean 16.67 vs 16.58 (r 0.985); n_sig_down 19.85 vs 19.86 (r 0.973); frac_cells_sig_up_t2 0.2133 vs 0.2133 (r 0.998). |
| `startrac` (expa/tran/migr, pairwise tran) | `ar_startrac` | exact: 132 + 152 patient-state rows and 426 pairwise rows, max diff 3e-14, n_shared_clones identical |
| `clone_table` | `ar_clone_table` | exact (39,103 rows) |
| `clone_dynamics` | `ar_clone_dynamics` | exact (1,466 clones: p 1e-11, fdr 6e-12, lfc 4e-14, class identical) |
| `clone_diversity` | `ar_clone_diversity` | Monte-Carlo (20 draws, exact multivariate-hypergeometric rarefaction vs R `sample`); same depth 203 and samples (70/72); Pearson > 0.998 for all 5 metrics, median relative diff 0.06-2.9%. |
| `clone_tissue_matrix`, `clone_tissue_entropy` | new (no R twin) | unit-tested on toy data (bits entropy, normalised entropy, breadth, dominant tissue) |
| `genesets.read_genesets / load_v3 / cycling_fraction` | `ar_read_genesets`, `ar_cycling_fraction` | jo_pathways 4 sets, pathways / jo_pathways_subsets 19, hotspot modules 832/882/754/234, CPDB 4,589 (174 curated); cycling_fraction logit_cc 5e-15 vs R (519 rows) |

## Method notes / deviations
* Cyclone numbering: R `cutree` numbers clusters by first observation (sorted `clone_id` order); implemented identically
  (scipy `ward` on the Euclidean distance of the correlation matrix == `hclust(ward.D2)`). `pam` is an own BUILD+SWAP
  implementation on the same distance, also numbered by first observation as `cluster::pam` does (matches R exactly).
* `delta_fc` 0/0 -> NaN -> 1 and `delta_c` NA -> 0 exactly as the original; Inf kept.
* Random downsampling cannot be bitwise-identical to R's Mersenne Twister; the deterministic core (tests, chi, p, BH) is.
* MSigDB: no local data (R `msigdbr` 26.1 is installed without `msigdbdf`, no network). `genesets.msigdb()` reads local GMTs
  (`$ARSC_MSIGDB_DIR`, `scratch/arsc/msigdb`, `data/msigdb`); `export_msigdb_gmt()` fills the cache via R when data are available.
  Untested against real MSigDB files (unit-tested with a synthetic GMT).

## Speed (R 4.3 / data.table vs Python, same machine)
| method | R s | Python s | speedup |
|---|---|---|---|
| cyclone (published settings) | 2.47 | 0.52 | 4.8x |
| clone_expansion_null (reps=5) | 12.61 | 2.51 | 5.0x |
| startrac (W3 + all-tp migr) | 6.93 | 0.26 | 26.7x |
| clone_dynamics 0->3 | 0.86 | 0.18 | 4.9x |
| clone_diversity (reps=20) | 1.80 | 0.18 | 9.9x |

## Exports to add to `arsc/__init__.py` (not edited)
`from . import clones, genesets` (public names: `clones.cyclone, cyclone_patient_table, cyclone_states, read_10x_clonotypes,
clone_expansion_null, startrac, clone_table, clone_diversity, clone_dynamics, clone_tissue_matrix, clone_tissue_entropy,
ari, p_adjust_bh, fisher_exact_2x2, pam, Cyclone`; `genesets.read_genesets, load_v3, V3_SETS, msigdb, export_msigdb_gmt,
filter_genesets, cycling_fraction`). `clones.pair_test` is also public (not in `__all__`).
