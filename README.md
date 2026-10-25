<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <img src="docs/assets/logo.png" alt="arsc" width="420">
  </picture>
</p>

<p align="center"><b>Single-cell analysis in Python that gives Seurat's and scanpy's answers, usually much faster.</b></p>

<p align="center">
  <a href="https://github.com/jiwen90/arsc/actions/workflows/ci.yml"><img src="https://github.com/jiwen90/arsc/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg" alt="License: GPL-3.0-or-later"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10-3.12">
</p>

arsc is a single-cell RNA-seq, CITE-seq and TCR analysis engine.

* **Validated.** Each method is checked numerically against the tool it replaces: Seurat 4.3, scanpy 1.11,
  presto, UCell, harmony 1.2 / 2.0 / 2.1, DESeq2, limma-voom, fgsea, lme4 and the published Cyclone and STARTRAC code.
  The agreement for every function is measured and written down in the parity tables.
* **Fast.** Numba and threaded kernels: at 100,000 cells, PCA is 14x faster than scanpy and 113x faster than
  Seurat, and UCell is 651x faster than the R package (benchmarks below). Memory-mapped stores let any number of
  processes open one copy of a dataset without loading it. One exception: arsc's exact k-NN search is slower
  than scanpy's approximate one at 1M cells.
* **Statistics at the patient level.** Pseudobulk DE, composition tests, exact paired and permutation tests,
  GLMMs and minimum-detectable-effect checks are built in, so the patient (not the cell) is the unit.
* **Made for agents.** `AGENTS.md` and `llms.txt` let an AI assistant turn a question like *"compare responders
  and non-responders in CD8 T cells"* into correct arsc code, and every snippet in the docs is executed by the tests.

arsc computes; it does not plot. Results are arrays and tidy `pandas` tables for any plotting tool.

## Benchmarks

### arsc vs scanpy, same session and threads

Every step arsc shares with scanpy, timed in pairs (`benchmarks/paired_bench.py`; the speed gate
`tests/test_speed_gate.py` reruns it and fails if arsc is slower):

<!-- PAIRED:START -->
<!-- PAIRED:END -->

### Earlier full suite (arsc 0.2.0, 2026-10-15; each tool at its default threading, not paired)

Kept for the Seurat columns and memory. Its k-NN rows timed arsc 0.2's exact search at every size (13 min at
1M cells); arsc now searches like scanpy (exact up to 4,096 cells, pynndescent above): see the paired table above.

<!-- BENCH:START -->
Synthetic data generated in code (`benchmarks/suite_sim.py`: 20,000 genes, negative-binomial counts, ~1,600 genes detected per cell, 10 cell types, 8 patients x 2 timepoints). Time = wall time of the analysis call only; memory = peak resident memory of the process during that call (GB, includes the loaded data). Shared inputs (HVGs, PCs, pseudobulk) are computed once so every step is compared on identical input. Speed-up = the other tool's time / arsc's time.

**100,000 cells**:

| Step | arsc | scanpy | Seurat / R | speed-up vs scanpy | speed-up vs Seurat | memory arsc / scanpy / Seurat | agreement with scanpy | agreement with Seurat / R |
|---|---:|---:|---:|---:|---:|---|---|---|
| Log-normalise | 0.58 s | 1.26 s | 7.70 s | **2.2x** | **13x** | 3.2 / 2.3 / 9.2 | max diff 3.4e-07 | exact (max diff 9e-16) |
| Highly variable genes (2,000) | 1.50 s | 3.78 s | 6.19 s | **2.5x** | **4.1x** | 2.6 / 2.6 / 8.7 | 2,000/2,000 genes shared; r = 1.000000 | 2,000/2,000 genes shared; r = 1.000000 |
| Scale + PCA (30 PCs) | 2.62 s | 37 s | 298 s | **14x** | **113x** | 4.7 / 4.6 / 8.6 | per-PC abs r >= 1.0000 (PC1-10), >= 0.998 (all 30) | per-PC abs r >= 1.0000 (PC1-10), >= 1.000 (all 30) |
| k-NN graph (k = 20) | 4.82 s | 6.68 s | 28 s | **1.4x** | **5.8x** | 1.1 / 0.8 / 2.3 | 98.2% of neighbours shared | 78.8% of neighbours shared |
| k-NN + clustering | 11 s | 12 s | 38 s | 1.04x | **3.4x** | 0.9 / 1.3 / 2.3 | ARI 0.998 | ARI 0.998 |
| Markers, presto-style Wilcoxon + AUC (20k genes) | 3.57 s | - | 24 s | - | **6.6x** | 7.9 / - / 8.6 | - | AUC exact (max diff 6e-16); p max rel. diff 1e-10 |
| Markers, scanpy-style Wilcoxon (20k genes) | 4.71 s | 55 s | - | **12x** | - | 8.5 / 2.7 / - | z-score max diff 1.1e-04; p max rel. diff 7e-04; top-50 overlap 100% | - |
| Pseudobulk sums (sample x cell type) | 0.09 s | 0.56 s | 1.57 s | **6.2x** | **17x** | 1.5 / 2.6 / 5.3 | identical (max diff 0) | identical (max diff 0) |
| Pseudobulk DESeq2 (paired, 16 samples) | 5.13 s | 5.20 s | 7.53 s | 1.01x | **1.5x** | 1.8 / 1.8 / 0.9 | - | log2FC r = 0.9997; -log10 p r = 0.9990; padj<0.1 Jaccard 0.96 |
| Pseudobulk limma-voom + duplicateCorrelation | 10 s | - | 24 s | - | **2.4x** | 0.2 / - / 0.3 | - | t r = 1.000000; max diff t 1.1e-03, logFC 1.5e-06 |
| Module score (5 sets) | 4.74 s | - | 8.18 s | - | **1.7x** | 2.8 / - / 7.4 | - | exact (max diff 1e-14) |
| scanpy gene-set score (5 sets) | 3.27 s | 6.34 s | - | **1.9x** | - | 2.6 / 2.8 / - | max diff 2.8e-07 | - |
| UCell score (5 sets) | 0.50 s | - | 326 s | - | **651x** | 2.7 / - / 7.7 | - | identical (max diff 0) |
| Clone table + dynamics + STARTRAC | 0.14 s | - | - | - | - | 0.2 / - / - | - | - |

**1,000,000 cells**:

| Step | arsc | scanpy | Seurat / R | speed-up vs scanpy | speed-up vs Seurat | memory arsc / scanpy / Seurat | agreement with scanpy | agreement with Seurat / R |
|---|---:|---:|---:|---:|---:|---|---|---|
| Log-normalise | 5.50 s | 12 s | 78 s | **2.3x** | **14x** | 29.7 / 23.8 / 88.7 | max diff 3.0e-07 | exact (max diff 9e-16) |
| Highly variable genes (2,000) | 13 s | 33 s | 61 s | **2.6x** | **4.7x** | 23.8 / 24.0 / 82.9 | 2,000/2,000 genes shared; r = 1.000000 | 2,000/2,000 genes shared; r = 1.000000 |
| Scale + PCA (30 PCs) | 21 s | 554 s | timed out (> 3600 s) | **27x** | **> 173x** | 46.2 / 43.6 / - | per-PC abs r >= 0.9999 (PC1-10), >= 0.993 (all 30) | - |
| k-NN graph (k = 20) | 13 min | 118 s | 373 s | 0.15x | 0.48x | 3.5 / 2.7 / 20.5 | 95.6% of neighbours shared | 56.3% of neighbours shared |
| k-NN + clustering | 14 min | 191 s | did not finish | 0.22x | - | 3.7 / 7.6 / - | ARI 0.992 | - |
| Markers, presto-style Wilcoxon + AUC (20k genes) | 45 s | - | 181 s | - | **4.0x** | 76.8 / - / 66.7 | - | AUC max diff 1.6e-07; p max rel. diff 5e-03 |
| Markers, scanpy-style Wilcoxon (20k genes) | 55 s | 48 min | - | **52x** | - | 82.7 / 23.9 / - | z-score max diff 7.6e-05; p max rel. diff 6e-04; top-50 overlap 100% | - |
| Pseudobulk sums (sample x cell type) | 0.74 s | 5.24 s | 14 s | **7.1x** | **19x** | 12.1 / 24.0 / 48.0 | identical (max diff 0) | identical (max diff 0) |
| Pseudobulk DESeq2 (paired, 16 samples) | 7.02 s | 7.59 s | 7.75 s | **1.1x** | **1.1x** | 1.9 / 1.8 / 0.9 | - | log2FC r = 0.9987; -log10 p r = 0.9869; padj<0.1 Jaccard 0.97 |
| Pseudobulk limma-voom + duplicateCorrelation | 16 s | - | 24 s | - | **1.5x** | 0.3 / - / 0.3 | - | t r = 1.000000; max diff t 3.3e-03, logFC 2.6e-07 |
| Module score (5 sets) | 45 s | - | 87 s | - | **1.9x** | 26.5 / - / 69.7 | - | exact (max diff 1e-14) |
| scanpy gene-set score (5 sets) | 31 s | 62 s | - | **2.0x** | - | 23.8 / 25.5 / - | max diff 2.8e-07 | - |
| UCell score (5 sets) | 4.97 s | - | not run | - | - | 24.9 / - / - | - | - |
| Clone table + dynamics + STARTRAC | 1.02 s | - | - | - | - | 0.4 / - / - | - | - |

Scaling (seconds; `-` = no equivalent, `n/r` = not run (time budget), `t/o` = timed out, `dnf` = did not finish, `err` = failed):

| Step | 10k arsc / scanpy / Seurat | 100k arsc / scanpy / Seurat | 1,000k arsc / scanpy / Seurat |
|---|---|---|---|
| Log-normalise | 0.08 / 0.16 / 0.94 | 0.58 / 1.26 / 7.70 | 5.50 / 12 / 78 |
| Highly variable genes (2,000) | 0.29 / 0.42 / 1.21 | 1.50 / 3.78 / 6.19 | 13 / 33 / 61 |
| Scale + PCA (30 PCs) | 0.84 / 3.03 / 15 | 2.62 / 37 / 298 | 21 / 554 / t/o |
| k-NN graph (k = 20) | 0.28 / 1.77 / 2.08 | 4.82 / 6.68 / 28 | 13 min / 118 / 373 |
| k-NN + clustering | 1.39 / 1.95 / 2.82 | 11 / 12 / 38 | 14 min / 191 / dnf |
| Markers, presto-style Wilcoxon + AUC (20k genes) | 0.52 / - / 1.93 | 3.57 / - / 24 | 45 / - / 181 |
| Markers, scanpy-style Wilcoxon (20k genes) | 0.66 / 3.72 / - | 4.71 / 55 / - | 55 / 48 min / - |
| Pseudobulk sums (sample x cell type) | 0.03 / 0.10 / 0.39 | 0.09 / 0.56 / 1.57 | 0.74 / 5.24 / 14 |
| Pseudobulk DESeq2 (paired, 16 samples) | 2.12 / 2.10 / 2.96 | 5.13 / 5.20 / 7.53 | 7.02 / 7.59 / 7.75 |
| Pseudobulk limma-voom + duplicateCorrelation | 4.60 / - / 6.64 | 10 / - / 24 | 16 / - / 24 |
| Module score (5 sets) | 0.67 / - / 1.03 | 4.74 / - / 8.18 | 45 / - / 87 |
| scanpy gene-set score (5 sets) | 0.45 / 0.75 / - | 3.27 / 6.34 / - | 31 / 62 / - |
| UCell score (5 sets) | 0.06 / - / 23 | 0.50 / - / 326 | 4.97 / - / n/r |
| Clone table + dynamics + STARTRAC | 0.07 / - / - | 0.14 / - / - | 1.02 / - / - |

Notes: arsc's k-NN is exact (recall vs brute force on 1,000 random cells: 1.000 at 10,000, 1.000 at 100,000, 1.000 at 1,000,000); scanpy (pynndescent) and Seurat (Annoy) are approximate, so their overlap with arsc is their recall. Exact search is the slowest arsc step at 1M cells; `arsc.knn(method="pynndescent")` exists but is not benchmarked here. Python steps are timed after an untimed warm-up run in the same process (JIT compilation excluded; scanpy's first `neighbors` call in a fresh process spends 25-40 s compiling).

Machine: AMD Ryzen Threadripper PRO 5955WX 16-Cores, 32 threads, 511.9 GB RAM, Windows 11 (10.0.26200). arsc 0.2.0, scanpy 1.11.5, anndata 0.12.19, pydeseq2 0.5.2, numpy 2.4.6; R version 4.3.1 (2023-06-16 ucrt), Seurat 4.3.0, presto 1.0.0, DESeq2 1.42.1, limma 3.58.1, UCell 2.6.2. Default threading for every tool (arsc and scanpy use all cores; most Seurat steps are single-threaded). Measured 2026-10-14 to 2026-10-15; reproduce with `python benchmarks/run_suite.py` (details in `benchmarks/README.md`).
<!-- BENCH:END -->

## Install

```bash no-run
pip install "arsc[all] @ git+https://github.com/jiwen90/arsc.git"
```

The PyPI name `arsc` belongs to an unrelated project, so install from git for now. The extras are `h5ad`
(anndata, h5py), `graph` (Leiden, UMAP), `stats` (pydeseq2, statsmodels, patsy) and `clonotrace` (POT); `all`
installs them all.

## Quickstart

```python
import arsc

ds = arsc.datasets.toy()                     # or arsc.read("pbmc.h5ad"), arsc.open("stores/pbmc")
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
L = arsc.pp.log_normalize(counts)                                    # Seurat NormalizeData, exact
markers = arsc.wilcoxauc(L, obs["cell_type"], features=genes)        # presto, exact
cd8 = (obs["cell_type"] == "CD8 T").to_numpy()
pb = arsc.pseudobulk(counts[cd8], obs[cd8], ["patient", "timepoint"], features=genes, min_cells=10)
de = arsc.de.deseq2(pb.to_frame(), pb.groups, "~ patient + timepoint", ("timepoint", "post", "pre"))
print(de.head()[["gene", "log2FoldChange", "padj"]])                 # paired by patient
```

Large files: `arsc.read("big.h5ad", store="stores/big")` converts once in bounded memory, then
`st = arsc.open("stores/big")` is instant and zero-copy. `st.get_features(["CD8A"])` reads only that gene, and
`st.subset(mask)` reads only those cells. Seurat objects (`.qs`, `.rds`) convert through the bundled R exporter.

## For AI agents

Point your assistant at **[AGENTS.md](AGENTS.md)** (Claude Code reads it through `CLAUDE.md`; other tools through
[llms.txt](llms.txt)). It contains the mental model, an "I want to... -> call" map of the whole workflow, runnable
recipes for the common biology questions, the statistics rules and the gotchas. From Python:

```python
import arsc
arsc.help("clone")                                          # task -> function map, filtered
code = arsc.recipes("do these clones expand after treatment")   # prints the matching recipe
```

Cloning the repository also gives Claude Code users an `arsc` skill (`.claude/skills/arsc/SKILL.md`).

## Validation summary

Agreement with the reference implementation, from the parity tables (each row is asserted by a test where the
reference outputs are available):

| arsc | Reference | Agreement |
|---|---|---|
| `pp.log_normalize`, `pseudobulk`, `pp.clr`, `average_expression` | Seurat `NormalizeData`, `AggregateExpression`, CLR, `AverageExpression` | exact (max diff 0 to 6e-14) |
| `pp.hvg_vst` | Seurat `FindVariableFeatures` (vst) | values rel. < 1e-9; identical 2,000-gene set |
| `pp.pca` | Seurat `RunPCA` | top-20 subspace cosine >= 0.9999 |
| `wilcoxauc` | `presto::wilcoxauc` | statistic and AUC exact; p rel. 4e-8 |
| `find_markers`, `find_all_markers` | Seurat `FindMarkers` / `FindAllMarkers` (wilcox, t, LR) | same gene order; p within 1e-13 to 1e-10 |
| `score_ucell`, `add_module_score`, `cell_cycle_scoring` | UCell, `AddModuleScore`, `CellCycleScoring` | exact (<= 1e-14; Phase identical for 38,440 cells) |
| `run_harmony(version="2.0")` (default) | `harmony::RunHarmony` 2.0.5 (CRAN) | 1-3 batch variables, distinct theta / lambda: same objective path; per-dim r min 0.9992-0.9998 from R's start ([HARMONY_PARITY.md](HARMONY_PARITY.md)) |
| `run_harmony(version="2.1")` | `harmony::RunHarmony` 2.1.0 (GitHub tag) | same cases: per-dim r min 0.994-0.999 from R's start |
| `run_harmony(version="1.2")` | `harmony::RunHarmony` 1.2 | same objective path (801.4 vs 801.7); per-dim r 0.989-0.998 (one variable); 1-3 variables: r min 0.984-0.9997 from R's start |
| `knn`, `find_neighbors`, `cluster` | scanpy `pp.neighbors` k-NN / Seurat `FindNeighbors` / `FindClusters` | `method="auto"`: exact up to 4,096 cells (scikit-learn brute force, float64 re-scoring), pynndescent with scanpy's settings above; `"exact"` at any size. Recall, overlap with scanpy and cluster ARI: [SCANPY_PARITY.md](SCANPY_PARITY.md); Leiden ARI 0.92 vs Seurat |
| `wnn` | Seurat `FindMultiModalNeighbors` | modality weight r 0.97; cluster ARI 0.87 |
| `symphony.build_reference` / `map_query` / `transfer_labels` | R `symphony` 0.1.1 (`buildReference`, `mapQuery`, `knnPredict`) | deterministic steps <= 1e-12 and identical labels given R's reference; own reference: query r >= 0.99 ([SYMPHONY_PARITY.md](SYMPHONY_PARITY.md)) |
| scanpy functions (`qc_metrics`, `normalize_total_log1p`, `highly_variable_genes`, `rank_genes_groups`, `score_genes`, `neighbors`, `leiden`, UMAP) | scanpy 1.11.5 | exact to 1e-13; identical Leiden labels and UMAP coordinates below 4,096 cells |
| `de.deseq2` | DESeq2 1.42 | log2FC r 0.9998; same padj < 0.1 calls |
| `de.voom` | limma-voom + duplicateCorrelation | logFC and t r 1.00000 (max diff 1e-4) |
| `de.fgsea` | fgsea 1.28 | ES exact; NES r 0.99995 |
| `stats.wilcoxon_signed_rank`, `perm_test`, `glmm_prop` | `wilcox.test`, exact enumeration, `lme4::glmer` | 6e-16, identical, fixed effects 2.2e-4 |
| `clones.cyclone`, `startrac`, `clone_dynamics` | published Cyclone, STARTRAC formulas | published Cyclone assignment reproduced (32,080 clones, ARI 1.0); 3e-14 |

Details, datasets and caveats: [SEURAT_PARITY.md](SEURAT_PARITY.md), [SCANPY_PARITY.md](SCANPY_PARITY.md),
[STATS_PARITY.md](STATS_PARITY.md), [CLONES_PARITY.md](CLONES_PARITY.md), [CLONOTRACE_PARITY.md](CLONOTRACE_PARITY.md),
[HARMONY_PARITY.md](HARMONY_PARITY.md).
Every stochastic function takes an integer seed and never touches the global RNG ([SEEDS.md](SEEDS.md)).

Earlier measurements on a 38,440-cell dataset (R single process vs arsc, same machine): `FindMarkers` wilcox 46 s
vs 0.24 s, `FindAllMarkers` 85 s vs 9 s, UCell 142 s vs 1.3 s, harmony 41 s vs 6.5 s; WNN on 259,899 cells
1,504 s vs 53 s. Three processes sharing one 260k-cell store used about 0.77 GB private memory each over 2.7 GB of
shared counts.

## Formats

| Input | Reader | Notes |
|---|---|---|
| arsc store directory | `arsc.open` | memory-mapped; the first open builds per-gene (CSC) copies |
| `.h5ad` | `arsc.read_h5ad` | `X`, `layers`, `raw`, `obs`/`var`, `obsm`, `varm`, `obsp`, JSON-able `uns`; streamed |
| 10x `.h5` (v2, v3) | `arsc.read_10x_h5` | split by feature type: Gene Expression -> `RNA`, Antibody Capture -> `ADT` |
| 10x `mtx` directory | `arsc.read_10x_mtx` | `matrix.mtx[.gz]`, `features.tsv[.gz]`, `barcodes.tsv[.gz]` |
| Seurat `.qs` / `.rds` | `arsc.read_seurat` | needs R + SeuratObject (+ qs) |

Write with `arsc.write_h5ad` / `Store.write_h5ad`; the output reads back identically in anndata and scanpy.

## Documentation

* [Getting started](docs/getting-started.md): a complete analysis on the toy dataset
* [Recipes](docs/recipes.md) and [AGENTS.md](AGENTS.md): runnable answers to common questions
* [API reference](docs/api.md), generated from the docstrings
* Migrating: [from scanpy](docs/from-scanpy.md), [from Seurat](docs/from-seurat.md)
* [Contributing](CONTRIBUTING.md), [Changelog](CHANGELOG.md), [Roadmap](ROADMAP.md)

## Citation

If you use arsc, please cite it ([CITATION.cff](CITATION.cff)) and the methods it implements ([NOTICE](NOTICE)).

## License

GPL-3.0-or-later ([LICENSE](LICENSE)). Several modules translate or closely follow GPL code (harmony, limma/edgeR,
presto, UCell), so arsc as a whole is GPL. You may use it freely, including privately and in a lab; if you
distribute arsc or a program that includes it, the GPL applies to that distribution (source available, same
licence). [NOTICE](NOTICE) credits each ported original with its licence; the MIT/BSD originals' licence
texts (Seurat, fgsea, Clonotrace, scanpy) are kept in [LICENSES/](LICENSES).
