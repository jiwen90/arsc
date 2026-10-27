<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <img src="docs/assets/logo.png" alt="arsc" width="420">
  </picture>
</p>

<p align="center"><b>Single-cell analysis in Python, following Seurat's algorithms.</b></p>

<p align="center">
  <a href="https://github.com/jiwen90/arsc/actions/workflows/ci.yml"><img src="https://github.com/jiwen90/arsc/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg" alt="License: GPL-3.0-or-later"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10-3.12">
</p>

arsc is a Python library for single-cell RNA-seq, CITE-seq and TCR analysis. It is meant for people who would
otherwise use Seurat but want to work in Python, or whose data have grown too large to analyze comfortably in R.

arsc follows Seurat's algorithms by default and reimplements them as compiled, multithreaded code (numba). Unless
you ask for something else, it gives Seurat's answer:

* normalization (`NormalizeData`), variable genes (`FindVariableFeatures`, vst), scaling and PCA (`ScaleData`,
  `RunPCA`), markers (`FindMarkers`, `FindAllMarkers`) and module scores (`AddModuleScore`) use Seurat's
  conventions, checked against Seurat in [SEURAT_PARITY.md](SEURAT_PARITY.md);
* the neighbor graph is Seurat's shared-nearest-neighbor graph (`FindNeighbors`);
* clustering is Louvain at resolution 0.6 (`FindClusters`, algorithm 1), run on arsc's own engine.

scanpy-compatible paths exist for like-for-like comparison: `arsc.scanpy_extra` (scanpy's normalization, variable
genes, neighbors, Leiden, markers and gene-set scores) and `arsc.cluster(..., refinement="igraph")`, checked against
scanpy in [SCANPY_PARITY.md](SCANPY_PARITY.md). The arsc vs scanpy timings below use those paths.

Data live in memory-mapped stores, so several processes can share one copy of a large dataset without loading it.

* **Validated.** Each method is compared numerically with the tool it replaces (Seurat 4.3, scanpy 1.11, presto,
  UCell, harmony, DESeq2, limma-voom, fgsea, lme4, Cyclone, STARTRAC); the parity tables record the agreement.
* **Statistics with the patient as the unit.** Pseudobulk DE, composition tests, exact paired and permutation tests,
  GLMMs and minimum-detectable-effect checks.
* **Usable by AI assistants.** `AGENTS.md` and `llms.txt` describe the workflow and the statistics rules, and every
  code example in the docs is run by the tests.

arsc computes; it does not plot. Results are arrays and tidy `pandas` tables for any plotting tool.

## Benchmarks

Real data: deterministic subsets (seed 0) of a 1.37M-cell PBMC store (36,601 genes, 32 patients, 36 cell-type
labels), read memory-mapped and held in memory. One machine with 16 cores (AMD Threadripper PRO 5955WX, 512 GB
RAM, Windows 11). Every time is a warm run: numba compilation is excluded.

* **arsc vs scanpy** (first three columns): both at 16 threads in one process, under a machine-wide lock that admits
  no new jobs (other processes still used 3-11% of the CPU on average); arsc uses its scanpy-compatible calls.
  Ranges, machine load and agreement: [SCANPY_PARITY.md](SCANPY_PARITY.md#speed-paired-runs-same-session-same-threads).
* **arsc vs Seurat** (last three columns): arsc at 16 threads with its Seurat-path calls, under the same lock;
  Seurat 4.3 with its defaults, which run on one thread, timed separately as an ordinary job. At 1,000,000 cells the
  Seurat run was stopped before it finished, so three of its values are imputed, marked ~value\*, with the method
  and range below that table. Calls, ranges and agreement:
  [SEURAT_PARITY.md](SEURAT_PARITY.md).

<!-- SPEED:START -->
#### 10,000 cells

| Step | arsc | scanpy | speed-up | arsc | Seurat | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| Log-normalize | 0.02 s | 0.12 s | 6.3x | 0.02 s | 0.71 s | 31.9x |
| Variable genes (2,000) | 0.20 s | 0.41 s | 2.1x | 0.19 s | 1.51 s | 7.8x |
| Scale + PCA (30 PCs) | 0.49 s | 1.21 s | 2.5x | 0.52 s | 5.18 s | 10.0x |
| Neighbor graph (k = 20) | 0.08 s | 1.22 s | 15.8x | 0.07 s | 2.20 s | 30.3x |
| Clustering | 0.14 s | 1.50 s | 10.9x | 0.21 s | 1.06 s | 5.0x |
| Markers, Wilcoxon, each cell type vs rest | 0.87 s | 5.51 s | 6.3x | 7.44 s | 31 min | 248x |
| Pseudobulk sums | 0.10 s | 0.27 s | 2.6x | 0.16 s | 0.82 s | 5.2x |
| Pseudobulk DESeq2 | 1.80 s | 2.05 s | 1.1x | - | - | - |
| Gene-set score (3 sets) | 0.23 s | 0.39 s | 1.7x | 0.40 s | 0.67 s | 1.7x |

16 cores; arsc and scanpy at 16 threads in one process, interleaved, median of 3 warm runs (numba compilation excluded). arsc vs Seurat: a separate session, arsc median of 3 warm runs at 16 threads; Seurat 4.3 one run after an untimed warm-up, on 1 thread (R's reference BLAS, `future` with 1 worker: Seurat's default), timed apart from arsc as an ordinary job beside other work (machine CPU use 25% on average while it ran).

#### 100,000 cells

| Step | arsc | scanpy | speed-up | arsc | Seurat | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| Log-normalize | 0.19 s | 1.23 s | 6.5x | 0.22 s | 6.97 s | 31.8x |
| Variable genes (2,000) | 1.27 s | 3.69 s | 2.9x | 1.26 s | 8.08 s | 6.4x |
| Scale + PCA (30 PCs) | 1.73 s | 9.67 s | 5.6x | 2.12 s | 38 s | 18.1x |
| Neighbor graph (k = 20) | 0.67 s | 2.48 s | 3.7x | 0.69 s | 29 s | 42.6x |
| Clustering | 1.99 s | 7.76 s | 3.9x | 8.56 s | 36 s | 4.2x |
| Markers, Wilcoxon, each cell type vs rest | 5.51 s | 87 s | 15.7x | 74 s | 233 min | 189x |
| Pseudobulk sums | 0.19 s | 0.84 s | 4.4x | 0.25 s | 4.72 s | 18.5x |
| Pseudobulk DESeq2 | 3.99 s | 5.02 s | 1.3x | - | - | - |
| Gene-set score (3 sets) | 1.74 s | 3.43 s | 2.0x | 2.87 s | 6.27 s | 2.2x |

16 cores; arsc and scanpy at 16 threads in one process, interleaved, median of 3 warm runs (numba compilation excluded). arsc vs Seurat: a separate session, arsc median of 3 warm runs at 16 threads; Seurat 4.3 one run after an untimed warm-up, on 1 thread (R's reference BLAS, `future` with 1 worker: Seurat's default), timed apart from arsc as an ordinary job beside other work (machine CPU use 21% on average while it ran).

#### 1,000,000 cells

| Step | arsc | scanpy | speed-up | arsc | Seurat | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| Log-normalize | 1.69 s | 12 s | 7.1x | 2.14 s | 71 s | 33.3x |
| Variable genes (2,000) | 12 s | 36 s | 3.1x | 12 s | 75 s | 6.3x |
| Scale + PCA (30 PCs) | 15 s | 101 s | 6.9x | 18 s | 401 s | 21.9x |
| Neighbor graph (k = 20) | 13 s | 24 s | 1.8x | 13 s | 368 s | 28.1x |
| Clustering | 41 s | 114 s | 2.8x | 298 s | 26 min | 5.2x |
| Markers, Wilcoxon, each cell type vs rest | 58 s | 78 min | 80.7x | 13 min | ~14 h\* | ~64.6x |
| Pseudobulk sums | 0.69 s | 5.21 s | 7.5x | 0.87 s | ~47 s\* | ~54.3x |
| Pseudobulk DESeq2 | 5.86 s | 6.82 s | 1.2x | - | - | - |
| Gene-set score (3 sets) | 17 s | 34 s | 2.0x | 28 s | ~63 s\* | ~2.2x |

16 cores; arsc and scanpy at 16 threads in one process, interleaved, median of 3 warm runs (numba compilation excluded) (markers ran 1 time per tool). arsc vs Seurat: a separate session, arsc median of 3 warm runs at 16 threads; Seurat 4.3 one run after an untimed warm-up, on 1 thread (R's reference BLAS, `future` with 1 worker: Seurat's default). \*Imputed, not measured: Seurat's one-thread run at this size was stopped during the markers step, so the pseudobulk and gene-set score steps never ran; its earlier steps are measured. Markers: `FindMarkers` (one thread) was timed for 16 of the 36 cell types, in that run and in separate one-thread R processes (6.2 h in all); the other 20 are counted at the mean measured time per type (23 min), giving 14.0 h (95% range 11.4-17.0 h, from resampling the measured per-type times). Time per type does not follow the number of cells in it: the slowest type (6,470 cells) took 71 min, the fastest (32,694 cells) 408 s. Pseudobulk sums and gene-set score: 10 times the 100,000-cell Seurat time; from 10,000 to 100,000 cells these grew 5.7-9.4x, while the measured steps grew 10-43x from 100,000 to 1,000,000, so the true time may be lower or higher.
<!-- SPEED:END -->

Rows: "Neighbor graph" is the k-NN graph (`sc.pp.neighbors`), "Clustering" k-NN plus Leiden
(`sc.tl.leiden(flavor="igraph")`), markers `sc.tl.rank_genes_groups` (Wilcoxon) and "Gene-set score"
`sc.tl.score_genes`.

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
| `knn`, `find_neighbors`, `cluster` | scanpy `pp.neighbors` k-NN / Seurat `FindNeighbors` / `FindClusters` | `method="auto"`: exact up to 4,096 cells, approximate above (arsc's own clustered kernel, float64 re-scoring; faster than FAISS, hnswlib, Voyager, Annoy and pynndescent at equal recall); `"exact"` at any size. Recall, overlap with scanpy and cluster ARI: [SCANPY_PARITY.md](SCANPY_PARITY.md); Leiden ARI 0.92 vs Seurat |
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
license). [NOTICE](NOTICE) credits each ported original with its license; the MIT/BSD originals' license
texts (Seurat, fgsea, Clonotrace, scanpy) are kept in [LICENSES/](LICENSES).
