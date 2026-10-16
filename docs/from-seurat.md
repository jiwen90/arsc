# Coming from Seurat

arsc reproduces Seurat 4.3's numbers in Python (`SEURAT_PARITY.md`), usually 10-200x faster. Two differences in
habit matter. Matrices are **cells x features** (Seurat stores features x cells). And there is no object that
functions modify: you pass a matrix, the identities and the feature names, and you get arrays or tables back.

Bring a Seurat object over once. This needs R with SeuratObject (and `qs` for `.qs` files):

```python no-run
st = arsc.read("obj.qs", store="stores/obj")   # runs the bundled R exporter, then opens the store
st = arsc.open("stores/obj")                   # afterwards: memory-mapped, no R needed
```

| Seurat | arsc | Agreement (SEURAT_PARITY.md / README) |
|---|---|---|
| `obj@meta.data` | `st.obs` | |
| `GetAssayData(obj, "counts")` | `st.X("RNA", "counts")` (cells x genes) | |
| `FetchData(obj, c("CD8A", "GZMK"))` | `st.get_features(["CD8A", "GZMK"])` | |
| `subset(obj, cells = ...)` / `WhichCells` | `st.subset(mask)` / `arsc.which_cells(obs, "nCount_RNA > 500")` | |
| `NormalizeData` | `arsc.pp.log_normalize(counts)` | exact |
| `NormalizeData(normalization.method = "CLR", margin = 2)` (ADT) | `arsc.pp.clr(adt_counts)` | exact |
| `FindVariableFeatures` (vst) | `arsc.pp.hvg_vst(counts, nfeatures=2000, features=genes)` | identical 2,000 genes |
| `ScaleData` (`vars.to.regress`, `split.by`) | `arsc.pp.scale(X)` / `arsc.scale_data(X, vars_to_regress=..., latent_data=obs)` | 1e-11 to 1e-10 |
| `RunPCA` | `arsc.pp.pca(Z, n_comps=50)` | subspace cos >= 0.9999 |
| `RunHarmony` | `arsc.run_harmony(pcs, obs, "batch")` | objective path as R; r 0.989-0.998 |
| `FindNeighbors` | `arsc.find_neighbors(emb, k=20)` (exact kNN; Seurat's Annoy is approximate) | Leiden ARI 0.92 |
| `FindMultiModalNeighbors` | `arsc.wnn([rna_emb, adt_emb])` | ARI 0.87; 28x faster |
| `FindClusters(algorithm = 4)` | `arsc.cluster(snn, method="leiden", resolution=0.8)` | |
| `RunUMAP` | `arsc.umap(emb)` | |
| `FindMarkers(ident.1, ident.2)` | `arsc.find_markers(L, idents, "A", "B", features=genes)` | p within 1e-13 |
| `FindMarkers(test.use = "LR", latent.vars = ...)` | `arsc.find_markers(..., test_use="LR", latent_vars=obs[[...]])` | 1e-11 |
| `FindAllMarkers(only.pos = TRUE)` | `arsc.find_all_markers(L, idents, features=genes, only_pos=True)` | same rows |
| `presto::wilcoxauc` | `arsc.wilcoxauc(L, idents, features=genes)` | exact |
| `AddModuleScore` | `arsc.add_module_score(L, genes, {"sig": [...]})` | exact (same R RNG) |
| `UCell::AddModuleScore_UCell` | `arsc.score_ucell(counts, genes, {"sig": [...]})` | exact |
| `CellCycleScoring` | `arsc.cell_cycle_scoring(L, genes)` (`arsc.cc_genes`) | Phase identical |
| `AverageExpression` / `AggregateExpression` | `arsc.average_expression(L, obs, "cluster", features=genes)` / `arsc.pseudobulk(counts, obs, by)` | exact |
| `RenameIdents`, `AddMetaData`, `SplitObject` | `arsc.rename_idents`, `arsc.add_metadata`, `arsc.split_object` | |
| DESeq2 / limma-voom on pseudobulk | `arsc.de.deseq2`, `arsc.de.voom(..., block="patient")` | r 0.9998 / 1.00000 |
| `fgsea::fgsea`, `msigdbr` | `arsc.de.fgsea`, `arsc.de.msigdb("H")` | ES exact |
| `symphony::buildReference` / `mapQuery` / `knnPredict` | `arsc.symphony.build_reference` / `map_query` / `transfer_labels` | exact given the same reference (SYMPHONY_PARITY.md) |
| `SCTransform`, `IntegrateData`, Azimuth / Seurat `MapQuery`, `HTODemux` | not ported; run them in R | |
| `DimPlot`, `FeaturePlot`, `VlnPlot` | not provided: plot the returned tables | |

A standard Seurat workflow in arsc:

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
L = arsc.pp.log_normalize(counts)                                        # NormalizeData
vf = arsc.pp.hvg_vst(counts, nfeatures=200, features=genes)              # FindVariableFeatures
vf_genes = vf.sort_values("rank").index[:200]                            # VariableFeatures(obj) order
Z, _, _ = arsc.pp.scale(L[:, genes.get_indexer(vf_genes)])               # ScaleData
pca = arsc.pp.pca(Z, n_comps=20)                                         # RunPCA
emb = arsc.run_harmony(pca.embedding, obs, "patient")                    # RunHarmony
snn = arsc.find_neighbors(emb, k=20)["snn"]                              # FindNeighbors
obs["seurat_clusters"] = pd.Categorical(arsc.cluster(snn, resolution=0.5).astype(str))   # FindClusters
markers = arsc.find_all_markers(L, obs["seurat_clusters"], features=genes, only_pos=True)  # FindAllMarkers
cc = arsc.cell_cycle_scoring(L, genes, ctrl=5)                           # CellCycleScoring (ctrl: small toy)
avg = arsc.average_expression(L, obs, "seurat_clusters", features=genes)  # AverageExpression (genes x groups)
print(markers.groupby("cluster", observed=True).head(2))
```

Seurat's `FindMarkers` between conditions treats cells as replicates. For condition effects, aggregate per
patient with `arsc.pseudobulk` and use `arsc.de.deseq2` (see the `de_paired` and `de_groups` recipes in AGENTS.md).
