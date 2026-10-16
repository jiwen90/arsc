# Changelog

All notable changes to arsc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may change the API).

## [Unreleased]

### Added
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
