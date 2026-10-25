# arsc benchmarks

Two kinds of evidence live here:

1. **The benchmark suite** (`run_suite.py`): arsc vs scanpy vs Seurat on the same synthetic data at
   several sizes, measuring wall time, peak memory and output agreement. Fully reproducible on any
   machine; no data needs to be downloaded.
2. **The R-parity references** (`r_reference_*.R`, `validation/`, `bench_*.py`): validation of arsc
   against R on real datasets. These need the datasets and are not runnable without them; their
   results are recorded in the `*_PARITY.md` files at the repo root.

## No benchmark data on disk

Benchmark inputs are never written to disk and never committed, at any size (no toy files either):

* `paired_bench.py` and the speed gate (`tests/test_speed_gate.py`) build their data **in memory**
  (`benchdata.py`). By default (`--source auto`) that is the real arsc store named by `ARSC_BENCH_STORE`,
  opened memory-mapped and read-only, with a seeded subset of `--sizes` cells (column names mapped with
  `ARSC_BENCH_STORE_COLUMNS`, see `benchdata.py`); without it, the toy data of `suite_sim.py`, generated in
  memory from a fixed seed at 20,000 cells (larger sizes only when asked with `--sizes`, still in memory).
  The shared 30-PC embedding is computed once by the parent and passed to the child processes through
  shared memory (RAM); children return their results, including the neighbour lists and cluster labels
  used for the agreement check, on stdout. Only the small result tables go to `--out`.
  `tests/test_speed_gate.py::test_paired_bench_writes_nothing_under_benchmarks` checks that a run writes
  nothing under `benchmarks/`.
* `run_suite.py` (arsc vs scanpy vs Seurat) is the one exception: Seurat runs in R and reads files, so this
  suite writes its simulated data, and only when given an explicit `--bench-dir` (or `ARSC_BENCH_DIR`) outside
  the repository. There is no default data directory and no `.bench_data` cache.

## Benchmark suite

```bash
# from the repo root, with the dev extras installed (pip install -e ".[dev]" scikit-misc)
python benchmarks/paired_bench.py --gate                         # arsc vs scanpy, 20k toy cells in memory
ARSC_BENCH_STORE=<store> python benchmarks/paired_bench.py --sizes 100000,1000000   # a real store, in memory
python benchmarks/run_suite.py --bench-dir <scratch dir>         # three tools; WRITES data to <scratch dir>
python benchmarks/run_suite.py --bench-dir <scratch dir> --steps markers,pseudobulk --sizes 10000
python benchmarks/run_suite.py --bench-dir <scratch dir> --compare-only   # recompute agreement
python benchmarks/render_table.py                                # headline table -> README.md (BENCH markers)
```

Environment variables: `ARSC_BENCH_STORE` / `ARSC_BENCH_STORE_COLUMNS` (real store for the in-memory
benchmarks), `ARSC_BENCH_DIR` (run_suite.py only: where it writes data and raw outputs; no default),
`ARSC_RLIB` (an extra R library path), `RSCRIPT` (default `Rscript`). Finished steps are skipped on a
re-run (results are appended to `runs.jsonl`); `--rerun` forces them.

R side: R 4.3 with Seurat 4.3, presto, DESeq2, limma, edgeR, UCell, Matrix and jsonlite. Python side:
scanpy (with `scikit-misc` for `flavor="seurat_v3"`), pydeseq2, psutil.

### Data

`suite_sim.py` generates, for each size, 20,000 genes of negative-binomial counts (gamma-Poisson,
dispersion 0.3) with a log-normal abundance profile and library sizes around 2,500 UMI, so about
1,600 genes are detected per cell. 10 cell types each up-regulate 60 marker genes about 6x; 8 patients
x 2 timepoints give 16 samples; in cell type T0, 150 genes go up 2x at `post` (the DE ground truth).
T-cell-like types carry clone ids, with 5 clones per patient expanding at `post`. `simulate_memory(n, seed)`
returns them in memory (paired_bench, speed gate); only `run_suite.py --bench-dir` writes the same counts as an
arsc store, an `.h5ad` (for scanpy) and raw dgCMatrix binaries (for R).

### What is measured

* Every (size, tool, step) runs in a **fresh process**. The parent samples the child's resident set
  size (process + children, every 50 ms).
* **Time** = wall time of the analysis call only. Loading data and computing inputs the step does not
  own (e.g. log-normalised data for `markers`) happen before the timed region.
* **Memory** = peak RSS of the process *during* the timed call (so it includes the loaded data).
  For arsc, the counts are memory-mapped; touched pages count towards RSS.
* **JIT warm-up**: each Python step is first run untimed on a separate 12,000-cell dataset in the same
  process, so one-off compilation (numba in arsc; pynndescent/umap in scanpy) is not counted. This
  favors scanpy most: its first `neighbors` call in a fresh process spends 25-40 s compiling, and the
  compiled code depends on the data size (a 5,000-cell warm-up did not remove it at 10,000 cells).
  The warm-up time is kept in `summary.csv` (`warmup_seconds`).
* **Threads**: each tool's default. arsc (numba) and scanpy use all cores; most Seurat steps are
  single-threaded R, which is part of the measured difference.
* **Shared inputs**: PCA uses a common 2,000-gene HVG list; neighbors and clustering use a common
  30-PC embedding; DE uses a common pseudobulk matrix (T0 cells, 16 samples, genes with >= 10 counts
  in >= 3 samples). All three are computed once with arsc, so the agreement of a step is not
  confounded by upstream differences.

### Agreement metrics (`results/suite/agreement.csv`)

| step | metric(s) |
|---|---|
| normalize, pseudobulk, module / gene-set / UCell scores | max absolute difference |
| hvg | genes shared among the 2,000 selected; Pearson r of the standardized variance |
| pca | per-PC abs. Pearson r (signs are arbitrary) |
| neighbors | mean fraction of the k-NN set shared with arsc; arsc recall vs brute force on 1,000 cells |
| cluster | adjusted Rand index vs the simulated cell types and between tools |
| markers | vs presto: max AUC difference, max relative p difference; vs scanpy: top-50 overlap per group, r of -log10 p |
| de_deseq2 | vs R DESeq2: r of log2FC and of -log10 p, Jaccard of padj < 0.1 calls; sensitivity and false-discovery proportion vs the planted genes |
| de_voom | vs R limma: max difference of logFC and t, r of t |

The README headline table is generated by `render_table.py` from `summary.csv` + `agreement.csv`.
It says "exact" only when the measured maximum difference is below 1e-10 and "identical" only when it
is 0; otherwise it prints the measured number.

### Known differences that are not bugs

* scanpy's `neighbors` uses approximate pynndescent above 4,096 cells and Seurat's `FindNeighbors`
  uses approximate Annoy; arsc 0.2's k-NN (timed in that suite) was exact. Overlap below 100% reflects the approximation
  in the other tools (see arsc's brute-force recall).
* Clustering: scanpy runs Leiden (igraph flavor), Seurat runs Louvain (its default), arsc runs Leiden
  on the Seurat-style SNN graph. ARI vs the simulated truth is the fairer comparison.
* scanpy's Wilcoxon reports z-scores without tie correction; presto/arsc report U, AUC and
  tie-corrected p-values, so p-values differ while the rankings agree.

### Simulated data: regenerate, then delete

Everything under `ARSC_BENCH_DIR` (default `benchmarks/.bench_data/`) is generated and safe to delete: the
counts of each size (`n<cells>/store`, `data.h5ad`, `r/`), the shared inputs (`common/`) and the raw outputs of
the runs (`out/`). The timings and agreement tables that matter are committed under `results/`. Sizes on disk:
about 0.5 GB at 10,000 cells, 1 GB at 20,000, 5 GB at 100,000 and 51 GB at 1,000,000 cells (store, h5ad and R
binaries each hold the full counts). `run_suite.py` and `paired_bench.py` simulate a missing size on demand
(same seed, so the same counts), including the 12,000-cell warm-up set; to build one ahead of time:

```bash
python benchmarks/suite_sim.py 20000                 # -> $ARSC_BENCH_DIR/n20000 (seed 0; --seed, --root)
python benchmarks/suite_sim.py 12000 --seed 1        # the JIT warm-up set paired_bench.py uses
```

The speed gate (`ARSC_SPEED_GATE=1 python -m pytest -q tests/test_speed_gate.py`) needs no stored data: it runs
`paired_bench.py --gate`, which simulates the 12,000- and 20,000-cell sets (about 1.6 GB, 2-3 min) when they are
missing. Delete the 1M-cell set after a large benchmark (`rm -rf benchmarks/.bench_data/n1000000`).

## R-parity references (real data, not shipped)

Only R outputs are stored; nothing in this repository is a dataset (`tests/test_repo_hygiene.py` fails if one is
tracked). The stored outputs live under `ARSC_REFS` (one folder per set) in a place no scratch cleanup prunes: in
autoresearch `.cache/arsc`, set in the untracked `tests/local.env`; the default, `benchmarks/.refs`, is git-ignored.
`tests/test_repo_hygiene.py` also fails when any path in `tests/local.env` resolves under a `scratch` folder.

**One shared input.** Every test and benchmark that needs a large real object reads the same registered arsc store,
`ARSC_STORE_GD` (the 38,440-cell gamma-delta T-cell CITE-seq store), memory-mapped and read-only, never copied; a
smaller size is a deterministic subset of its cells taken in memory. R computes its references from the Seurat object
the store was converted from (`ARSC_GD_RDS`), or from tables exported from the store into a temporary folder that the
regeneration deletes afterwards (harmony input, pseudobulk, composition). Synthetic inputs are seeded and built in
memory (`stats_resampling_cases.build`, `suite_sim.py`).

Rebuild with `benchmarks/regen_refs.py` (`status` lists every set with its folder, size, input and the test that reads
it; a test whose reference is missing skips with a message naming the missing file and this command):

| set | input | R script | read by | size |
|---|---|---|---|---|
| `ref_gd` | shared store (rds) | `r_reference_gd.R` (tests subset; `ARSC_REF_FULL=1` writes PCA, harmony, UCell, pseudobulk, CLR, SNN for `validation/`) | `test_vs_r.py` | 3 MB |
| `ref_extra` | shared store (rds) | `r_reference_seurat_extra.R` | `test_seurat_extra.py` | 31 MB |
| `ref_harmony` | shared store PCA (temporary export) | `r_reference_harmony_all.R` | `test_vs_r.py` | 102 MB |
| `stats_parity` | shared store pseudobulk / composition (`stats_parity_inputs.py`) | `r_reference_stats_parity.R`, `r_reference_stats_small.R` | `test_stats_de.py` | 4 MB |
| `stats_resampling` | seeded synthetic tables | `r_reference_stats_resampling.R` | `test_stats_resampling.py` | <1 MB |
| `clones_ref` | CD8 clone metadata (`ARSC_TEST_CLONES_META`, `ARSC_DATA_V3`) | `r_reference_clones.R` (needs autoresearch's `tools/R`) | `test_clones.py` | 8 MB (gzip) |
| `clonotrace_ref` | Clonotrace's public demo (downloaded at the pinned commit) | `r_reference_clonotrace.R` | `test_clonotrace.py` | 11 MB (+ 4.5 MB source in `$ARSC_REFS/src`) |
| `symphony_ref` | R symphony's vignette data + seeded random inputs | `r_reference_symphony.R` | `test_symphony.py` | 12 MB |

Three sets cannot use the shared input: Cyclone and the clone helpers need TCR clones with the published Cyclone
assignment, CloneTrace needs lineage-traced cells, and the Symphony reference/query case is the package's own vignette.

```bash
python benchmarks/regen_refs.py status
python benchmarks/regen_refs.py ref_gd ref_extra stats_parity   # or: all (about 1 h)
```

`r_reference_pbmc.R` and `r_annoy_knn_pbmc.R` (WNN timing on a 259,899-cell PBMC CITE-seq object) feed only
`benchmarks/validation/` and the README timing table, not the tests. `mmap_concurrency.py` measures memory sharing of
one store across processes. Results: `../*_PARITY.md`, `results/*.json`.

The harmony references (about 105 MB: per harmony version and case the corrected embedding in gzip float32, R's
k-means start and the objective trace) live in `ARSC_REF_HARMONY` (default `$ARSC_REFS/ref_harmony`). Their input (the
store's PCA and batch columns) is rebuilt in memory by `harmony_input.build`; `regen_refs.py ref_harmony` writes it to
`<ARSC_REF_HARMONY>/input` for R and deletes it afterwards. Each harmony version runs from its own R library (about
9 min per version):

```bash
python benchmarks/regen_refs.py ref_harmony          # all three versions
python benchmarks/harmony_parity.py                  # results/harmony_parity.json
```

`r_reference_harmony_all.R` finds the 2.0.5 / 2.1.0 libraries at `<AR_HOME>/.Rlib_harmony_<version>` or in
`ARSC_HARMONY_LIB_2_0_5` / `ARSC_HARMONY_LIB_2_1_0`; 1.2.0 comes from the default library. Keep every cell: the
per-dimension r is driven by a few hundred cells that change cluster, and a 1-in-10 subset moved it by up to 0.015.
