# arsc roadmap

## 0.2 (this round): IO completeness, API polish, packaging

1. **Store builder** (`arsc/formats.py`): one streaming writer used by every reader. Row chunks
   (or column chunks for CSC sources) go straight into `.npy` files of unknown final length, so a
   conversion never holds a full matrix in RAM. The same builder can target memory, which returns an
   in-memory `Dataset` for small files.
2. **h5ad read**: dense/CSR/CSC `X`, `layers`, `raw` (stored as its own assay), `obs`/`var`
   (categoricals, nullable, strings), `obsm` -> reductions, `varm["PCs"]` -> PCA loadings, other
   `varm`, `obsp` -> graphs, and JSON-able `uns`. The matrix is streamed with h5py; the small
   elements go through `anndata.io.read_elem`. The source dtype is recorded so export can restore it.
3. **h5ad write**: `Store.write_h5ad` / `Dataset.write_h5ad` / `arsc.write_h5ad`. The matrix is
   streamed in row chunks (cell and gene subsets are supported); the rest goes through
   `anndata.io.write_elem`. `Store.to_anndata(cells, genes)` / `Dataset.to_anndata` are the
   in-memory hand-off to scanpy.
4. **10x**: `filtered_feature_bc_matrix.h5` (CellRanger v3+ and v2), split by `feature_type`
   (Gene Expression -> RNA, Antibody Capture -> ADT, ...), and the `matrix.mtx[.gz]` directory.
5. **Seurat**: `read("obj.qs", store=...)` runs the bundled R exporter (`export_seurat.R`; qs is
   preferred over RDS). h5Seurat is documented as a SeuratDisk -> h5ad route; it is not parsed natively.
6. **`arsc.read(path, store=None)`**: dispatches on format. It returns an in-memory `Dataset`, or a
   memory-mapped `Store` when `store=` is given.
7. Polish: clear errors, docstrings, type hints, pyproject extras (`h5ad`, `graph`, `dev`), version
   0.2.0, and a portable README Quickstart.
8. Tests: synthetic data generated inside the tests (no lab data). They cover parity with
   anndata/scanpy readers and round trips (h5ad -> store -> h5ad, 10x h5/mtx vs scanpy).

## Next
- **CloneTrace** (clonal fate modeling): port next to `arsc.clones.cyclone`. The port will be
  scoped after reading the published code.
- Native h5Seurat reader; loom reader.
- MuData (`.h5mu`) export of multi-assay stores (RNA + ADT).
- Rebuild the R reference outputs that the R-parity tests depend on (`benchmarks/r_reference_*.R`).
- Make the R-library path in `R/*.R` and the default data paths in `de.py`/`genesets.py`
  environment-only before the repo is shared.
