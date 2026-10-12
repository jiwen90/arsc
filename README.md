# arsc — fast, validated single-cell compute for AutoResearch

Python replacements for the Seurat-ecosystem steps the research agents use. Each one is validated
numerically against its R reference. **No plotting**: arsc returns arrays and tidy tables, and all
figures are made in R ggplot (`tools/R/ar_theme.R`).

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
Status: core validated. Not yet included: scVI-class models, trajectory tools, and an AnnData
(`.h5ad`) → store converter for scanpy-native projects (planned).
