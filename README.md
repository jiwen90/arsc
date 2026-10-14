# arsc — fast, validated single-cell compute for AutoResearch

Python replacements for the Seurat-ecosystem steps the research agents use. Each one is validated
numerically against its R reference. **No plotting**: arsc returns arrays and tidy tables, and all
figures are made in R ggplot (`tools/R/ar_theme.R`).

## Quickstart
```bash
pip install -e ".[h5ad]"        # core + h5ad/10x readers; ".[dev]" adds scanpy/pytest for the tests
```
```python
import arsc

# 1. Read anything. Small files load into memory; large ones convert once into a memory-mapped store.
ds = arsc.read("pbmc.h5ad")                                  # in-memory Dataset
st = arsc.read("pbmc.h5ad", store="stores/pbmc")             # Store: streamed conversion, bounded RAM
st = arsc.read("filtered_feature_bc_matrix.h5", store="stores/run1")   # RNA + ADT assays
st = arsc.read("filtered_feature_bc_matrix/")                # 10x mtx directory (Dataset)
st = arsc.read("obj.qs", store="stores/obj")                 # Seurat v4/v5 via the bundled R exporter
st = arsc.open("stores/pbmc")                                # reopen later: zero-copy, shared by processes

# 2. Compute (arrays and tidy tables in, arrays and tidy tables out).
L  = arsc.pp.log_normalize(st.X("RNA", "counts"))
mk = arsc.wilcoxauc(L, st.obs["cell_type"], features=st.features("RNA"))
pb = arsc.pseudobulk(st.X("RNA", "counts"), st.obs["sample"])

# 3. Hand off to scanpy / anndata (all or a subset; only the selected rows are read).
adata = st.to_anndata(cells=st.obs["cell_type"] == "CD8 T", genes=["CD8A", "GZMK", "PDCD1"])
st.write_h5ad("cd8.h5ad", cells=st.obs["cell_type"] == "CD8 T")    # streamed, any size
```

### Formats
| input | reader | notes |
|---|---|---|
| arsc store directory | `arsc.open` | memory-mapped; first open builds per-gene CSC copies |
| `.h5ad` | `arsc.read_h5ad` | dense/CSR/CSC `X`, `layers`, `raw` (own assay), `obs`/`var` dtypes, `obsm` -> reductions, `varm`, `obsp` -> graphs, JSON-able `uns`; streamed |
| 10x `.h5` (v3, v2) | `arsc.read_10x_h5` | split by feature type: Gene Expression -> `RNA`, Antibody Capture -> `ADT`, ... |
| 10x mtx directory | `arsc.read_10x_mtx` | `matrix.mtx[.gz]`, `features.tsv[.gz]`/`genes.tsv`, `barcodes.tsv[.gz]` |
| Seurat `.qs` (preferred) / `.rds` | `arsc.read_seurat` | needs R + SeuratObject (+ qs); `R/export_seurat.R` |
| `.h5Seurat` | not native | in R: `SeuratDisk::Convert(f, dest = "h5ad")`, or `qs::qsave(LoadH5Seurat(f), "x.qs")` |

Write: `arsc.write_h5ad` / `Store.write_h5ad` / `Dataset.write_h5ad`. The output reads back
identically in `anndata`/`scanpy`; matrices are written as CSR, and the source dtype is restored.
Integer-valued matrices are stored as float32 (exact below 2^24); non-integer float64 stays float64.

## Design
- **Memory-mapped store** (`arsc.open(dir)`): CSR/CSC components are stored as `.npy` and opened
  zero-copy with `mmap_mode="r"`. N processes share one physical copy through the OS page cache.
  Measured with 3 concurrent processes on the 260k-cell PBMC store: about 0.77 GB private memory each,
  over 2.7 GB of shared counts.
- **Threads, not process copies** (numba / BLAS; respects `AR_THREADS`).
- **Bridge from Seurat:** `export_seurat()` / `exporter_script()` write the store from any Seurat
  v4/v5 object; `read_seurat_export(dir)` reads it.

## Validation and speed (gdT_1, 38,440 cells; PBMC WNN on 259,899 cells)
| function | R reference | agreement | arsc | R |
|---|---|---|---|---|
| `pp.log_normalize` | `NormalizeData` | exact (< 1e-10) | — | 2.8 s |
| `pp.hvg_vst` | `FindVariableFeatures` (vst) | values rel. < 1e-9; identical 2,000-gene set (tie order differs) | — | 2.5 s |
| PCA (randomized) | `RunPCA` | top-20 subspace cos ≥ 0.9999; per-PC r ≥ 0.992 | 2.1 s | 13.6 s |
| `pseudobulk` | `AggregateExpression` | exact (max diff 0) | 0.28 s | 1.7 s |
| ADT CLR | Seurat CLR (margin 2) | exact (2e-15) | — | — |
| `wilcoxauc` | `presto::wilcoxauc` | statistic/AUC exact; p rel. 4e-8 | 1.3 s | 6.1 s |
| `score_ucell` / `add_module_score` | UCell / `AddModuleScore` | exact (≤ 1e-14) | ~1.3 s | 142 s / 2.2 s |
| `run_harmony` | `harmony::RunHarmony` 1.2 | same objective trajectory (1371 → 801.4 vs R 801.7); per-dim r 0.989–0.998 (stochastic block order) | 6.5 s | 40.8 s |
| `knn` / `snn_graph` | `FindNeighbors` | exact kNN (Seurat uses approximate Annoy: SNN Jaccard 0.66); Leiden ARI vs Seurat 0.92 | 1.2 s | — |
| `wnn` | `FindMultiModalNeighbors` | RNA.weight r 0.97; neighbor overlap 0.92; cluster ARI 0.87 | **53 s** | **1,504 s** |

Reproduce: the R references come from `benchmarks/r_reference_gd.R` and `r_reference_pbmc.R`, into
`$ARSC_SCRATCH/ref_*`. Then run `.venv/Scripts/python.exe -m pytest -q tests` (asserts the
thresholds above).

## Use from the framework
```python
import sys; sys.path.insert(0, "E:/autoresearch/vendor/arsc/src"); import arsc
st = arsc.open("E:/autoresearch/data/<store>")          # mmap, zero-copy
L  = arsc.pp.log_normalize(st.X())
mk = arsc.wilcoxauc(L, st.obs["cell_type"], features=st.features())
```
Status: core validated. h5ad/10x IO validated against anndata/scanpy on synthetic data
(`tests/test_formats.py`). Not yet included: scVI-class models, a native h5Seurat/loom reader.
See `ROADMAP.md`.
