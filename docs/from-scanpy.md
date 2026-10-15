# Coming from scanpy

arsc functions take arrays and return arrays or tidy tables. There is no `AnnData` that every function mutates:
you pass `X`, the labels and the gene names, and you keep the results where you want them. The `scanpy_extra`
functions reproduce scanpy's numbers (see `SCANPY_PARITY.md`); the Seurat-style functions are listed in
`docs/from-seurat.md`.

| scanpy | arsc | Agreement (SCANPY_PARITY.md) |
|---|---|---|
| `sc.read_h5ad(f)` | `arsc.read(f)` (memory) or `arsc.read(f, store=dir)` then `arsc.open(dir)` (memory-mapped) | round trip identical |
| `adata.X`, `adata.layers["counts"]` | `ds.X("RNA", "X")`, `ds.X("RNA", "counts")` | |
| `adata.obs`, `adata.var_names` | `ds.obs`, `ds.features("RNA")` | |
| `adata[mask]` | `ds.subset(mask)` / `st.subset(mask)` | |
| `sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])` | `arsc.qc_metrics(counts, genes, qc_vars={"mt": mask})` | all columns, 1.4e-14 |
| `sc.pp.normalize_total(target_sum=1e4); sc.pp.log1p` | `arsc.normalize_total_log1p(counts)` | bitwise |
| `sc.pp.highly_variable_genes(flavor="seurat", n_top_genes=n)` | `arsc.highly_variable_genes(L, genes, flavor="seurat", n_top_genes=n)` | identical gene set |
| `sc.pp.highly_variable_genes(flavor="seurat_v3", layer="counts")` | `arsc.highly_variable_genes(counts, genes, flavor="seurat_v3", n_top_genes=n)` | 999/1000 genes (a tie) |
| `sc.pp.scale(max_value=10)` | `arsc.pp.scale(X, scale_max=10)` | 1e-13 |
| `sc.pp.regress_out` | `arsc.regress_out(X, covariates)` | 1e-13 |
| `sc.tl.pca(svd_solver="arpack")` | `arsc.pp.pca(Z, n_comps, method="arpack")` | per-PC r >= 0.998 |
| `sce.pp.harmony_integrate` | `arsc.run_harmony(pcs, obs, key)` (R harmony port, not harmonypy) | r ~0.99 per dim vs harmonypy |
| `sc.pp.neighbors(n_neighbors=15)` | `arsc.scanpy_extra.neighbors(emb, n_neighbors=15)` | connectivities exact below 4,096 cells |
| `sc.tl.leiden(resolution=r)` | `arsc.scanpy_extra.leiden(conn, resolution=r)` | identical labels on the same graph |
| `sc.tl.umap` | `arsc.scanpy_extra.umap_from_connectivities(conn)` | identical coordinates |
| `sc.tl.rank_genes_groups(method="wilcoxon")` + `sc.get.rank_genes_groups_df` | `arsc.rank_genes_groups(L, labels, genes)` | p 1e-13 |
| `sc.tl.score_genes` | `arsc.score_genes(L, genes, gene_list)` | 5e-15 |
| `sc.tl.dendrogram` | `arsc.scanpy_extra.dendrogram(rep, labels)` | same leaf order |
| `sc.tl.diffmap`, `sc.tl.dpt` | `arsc.diffmap(conn)`, `arsc.dpt(dm, root)` | DPT r 0.99999999 |
| palantir diffusion maps / MAGIC | `arsc.scanpy_extra.palantir_diffusion_maps`, `arsc.magic_impute` | 1e-10 / 4e-6 |
| `decoupler.get_pseudobulk` | `arsc.pseudobulk(counts, obs, ["patient", "cell_type"])` | exact sums |
| `pydeseq2` on pseudobulk | `arsc.de.deseq2(cnt, coldata, design, contrast)` | uses pydeseq2 |
| `gseapy.prerank` | `arsc.de.fgsea(stat, genesets)` | fgsea (R) parity |
| `sc.pl.*` | not provided: plot from the returned tables | |

The same analysis, side by side. scanpy:

```python no-run
import scanpy as sc
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=100)
sc.pp.scale(adata, max_value=10); sc.tl.pca(adata, n_comps=20, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=15); sc.tl.leiden(adata, resolution=1.0); sc.tl.umap(adata)
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
markers = sc.get.rank_genes_groups_df(adata, group=None)
```

arsc:

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
counts, genes = ds.X("RNA", "counts"), ds.features("RNA")
L = arsc.normalize_total_log1p(counts)
hv = arsc.highly_variable_genes(L, genes, flavor="seurat", n_top_genes=100)
idx = genes.get_indexer(hv.index[hv["highly_variable"]])
Z, _, _ = arsc.pp.scale(L[:, idx], scale_max=10)
pcs = arsc.pp.pca(Z, n_comps=20, method="arpack").embedding
nb = arsc.scanpy_extra.neighbors(pcs, n_neighbors=15)
leiden = arsc.scanpy_extra.leiden(nb["connectivities"], resolution=1.0)
xy = arsc.scanpy_extra.umap_from_connectivities(nb["connectivities"])
markers = arsc.rank_genes_groups(L, pd.Series(leiden.astype(str)), genes)
print(markers.head())
```

Hand data back to scanpy at any point with `ds.to_anndata(cells=..., genes=...)` or `st.write_h5ad(path)`.
