# Getting started with arsc

This tutorial walks through a complete analysis on `arsc.datasets.toy()`: a small synthetic study of 6 patients
sampled before and after treatment. Three patients respond (`R`), three do not (`NR`). The data are simulated, so
we know the answer: in responders, CD8 T cells become more frequent after treatment, switch on a cytotoxic
programme (`GZMK GZMB IFNG PRF1 CXCL13`), and a few T-cell clones expand. Every block below runs as written
(the test suite executes it), and each one continues from the previous block.

## 1. Data: Dataset and Store

```python
import os, tempfile
import numpy as np
import pandas as pd
import arsc

ds = arsc.datasets.toy()      # in-memory Dataset: 3,000 cells x 300 genes, plus an ADT assay
print(ds)
print(ds.obs.head())
```

Your data usually comes as a file. `arsc.read` reads `.h5ad`, 10x `.h5`, a 10x `mtx` directory or a Seurat `.qs`
file. With `store=`, it converts the file once into a **store**: a directory of memory-mapped arrays. Opening a
store costs nothing, many processes can share it, and you read only the cells and genes you touch.

```python
work = tempfile.mkdtemp()
path = arsc.datasets.toy_h5ad(os.path.join(work, "toy.h5ad"))   # pretend this is your file
st = arsc.read(path, store=os.path.join(work, "toy_store"))      # one-time conversion
st = arsc.open(os.path.join(work, "toy_store"))                  # what you do every other time
print(st)
print(st.get_features(["CD8A", "GZMK"]).head(3))                 # reads only these two genes
```

The rest of the tutorial uses `ds`; `st` has the same `obs`, `X()`, `features()` and `subset()`.

## 2. Quality control

```python
counts, genes = ds.X("RNA", "counts"), ds.features("RNA")
qc = arsc.qc_metrics(counts, genes, qc_vars={"mt": genes.str.startswith("MT-")})
print(qc[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe().round(1))
keep = ((qc["n_genes_by_counts"] >= 20) & (qc["pct_counts_mt"] < 20)).to_numpy()
ds = ds.subset(keep)
counts, obs = ds.X("RNA", "counts"), ds.obs
```

## 3. Normalise, reduce, cluster

These steps reproduce Seurat's `NormalizeData`, `FindVariableFeatures`, `ScaleData`, `RunPCA`, `RunHarmony`,
`FindNeighbors`, `FindClusters` and `RunUMAP` (the parity tables give the agreement for each step).

```python
L = arsc.pp.log_normalize(counts)
hvg = arsc.pp.hvg_vst(counts, nfeatures=200, features=genes)
top = hvg.sort_values("rank").index[:200]
Z, _, _ = arsc.pp.scale(L[:, genes.get_indexer(top)])
pcs = arsc.pp.pca(Z, n_comps=20).embedding
emb = arsc.run_harmony(pcs, obs, "patient")            # remove patient-specific shifts
snn = arsc.find_neighbors(emb, k=20)["snn"]
obs["cluster"] = pd.Categorical(arsc.cluster(snn, resolution=0.5).astype(str))
umap_xy = arsc.umap(emb)
print(pd.crosstab(obs["cluster"], obs["cell_type"]))
```

arsc does not plot. To draw the UMAP, write a table and plot it in your tool of choice:

```python
arsc.write_table(obs.assign(UMAP_1=umap_xy[:, 0], UMAP_2=umap_xy[:, 1]), os.path.join(work, "umap.csv"))
```

## 4. What defines each cluster?

`wilcoxauc` (presto) tests every gene in every cluster against the rest in one pass. These are cell-level
statistics: they describe clusters, and they are not evidence about patients.

```python
mk = arsc.wilcoxauc(L, obs["cluster"], features=genes)
top = arsc.top_markers(mk, n=3)
print(top[["group", "feature", "auc", "logFC", "pct_in", "pct_out"]])
```

Name clusters with signature scores (UCell, which is rank based and works on raw counts):

```python
sigs = {"CD8 T": ["CD8A", "CD8B"], "CD4 T": ["CD4", "IL7R"], "B": ["MS4A1", "CD79A"],
        "NK": ["NKG7", "GNLY"], "Mono": ["LYZ", "CD14"]}
uc = arsc.score_ucell(counts, genes, sigs)
obs["predicted"] = uc.idxmax(axis=1).str.removesuffix("_UCell").to_numpy()
print(pd.crosstab(obs["cluster"], obs["predicted"]))
```

## 5. Do CD8 T cells expand after treatment? (patient = unit)

Proportions are computed per sample (patient x timepoint), then compared within patients.

```python
comp = arsc.stats.composition(obs, "cell_type", sample_cols=["patient", "timepoint"], extra_cols=["response"])
cd8 = comp[comp["celltype"] == "CD8 T"]
res = arsc.stats.paired_test(cd8, "prop", t0="pre", t1="post", patient="patient", time="timepoint",
                             by="response", transform="logit")
print(res[["group", "n_patients", "k_up", "median_delta", "ci_lo", "ci_hi", "p_wilcox_exact", "min_possible_p"]])
```

All three responders go up and all three non-responders do not, but with 3 patients the smallest possible exact
p-value is 0.25. That is what the data can say: a consistent direction in every patient, with a confidence interval,
not a small p-value. `arsc.stats.mde(3)` tells you this before you test.

## 6. What changes inside CD8 T cells? (pseudobulk DE)

Sum raw counts per patient x timepoint, then run DESeq2 with the patient in the design (a paired analysis).
Here we use only the responders.

```python
is_cd8 = (obs["cell_type"] == "CD8 T").to_numpy()
pb = arsc.pseudobulk(counts[is_cd8], obs[is_cd8], ["patient", "timepoint"], features=genes, min_cells=10)
cnt, cd = pb.to_frame(), pb.groups
resp = cd.index[cd["patient"].isin(["P1", "P2", "P3"])]
de = arsc.de.deseq2(cnt[resp], cd.loc[resp], "~ patient + timepoint", ("timepoint", "post", "pre"))
print(de.head(8)[["gene", "log2FoldChange", "padj"]])
```

The five planted genes come out on top. Rank all genes by the Wald statistic and test gene sets:

```python
sets = {"cytotoxic": ["GZMK", "GZMB", "PRF1", "IFNG", "CXCL13"], "naive": ["TCF7", "SELL", "CCR7"]}
print(arsc.de.fgsea(de.set_index("gene")["stat"], sets, min_size=3)[["pathway", "NES", "pval"]])
```

## 7. Do clones expand?

```python
ct = arsc.clones.clone_table(obs, "clone_id", sample_cols=["patient", "timepoint"])
dyn = arsc.clones.clone_dynamics(ct, "pre", "post", patient="patient", time="timepoint")
print(dyn.sort_values("p").head(6)[["patient", "clone", "n_0", "n_1", "lfc", "fdr"]])
div = arsc.clones.clone_diversity(ct, sample_cols=["patient", "timepoint"], floor=50)
div["response"] = div["patient"].map(obs.groupby("patient", observed=True)["response"].first())
print(div.pivot_table(index=["response", "patient"], columns="timepoint", values="clonality", observed=True).round(3))
```

The clones that grow belong to the responders. As in step 5, report the per-patient pattern; the test across
patients is limited by n.

## Next

* `arsc.help()` lists every task; `arsc.recipes()` prints runnable recipes for common questions.
* `docs/from-scanpy.md` and `docs/from-seurat.md` map familiar calls to arsc.
* `docs/api.md` documents every function.
