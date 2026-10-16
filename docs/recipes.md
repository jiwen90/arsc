# Recipes

Generated from `src/arsc/_recipes.py` by `docs/build_docs.py`; do not edit by hand.
Each recipe runs as written on `arsc.datasets.toy()` (tested in `tests/test_docs_snippets.py`).
In Python: `arsc.recipes()` lists them, `arsc.recipes("<id or question>")` prints one.

## `load`: Load data or build a memory-mapped store

Asks like: "load my h5ad"; "open the dataset"; "convert this Seurat object"; "read 10x output"

```python
import os, tempfile
import arsc

tmp = tempfile.mkdtemp()
h5ad = arsc.datasets.toy_h5ad(os.path.join(tmp, "toy.h5ad"))   # stand-in for your own file
st = arsc.read(h5ad, store=os.path.join(tmp, "toy_store"))      # convert once; streamed, bounded RAM
st = arsc.open(os.path.join(tmp, "toy_store"))                  # later: memory-mapped, zero-copy
print(st)                                        # lists assays and layers
print(st.layers("RNA"))                          # an h5ad's .X arrives as layer "X"; its layers keep their names
X = st.X("RNA")                                  # cells x genes CSR (default layer: counts, else the first)
genes = st.features("RNA")
two = st.get_features(["CD8A", "GZMK"])          # DataFrame; reads only these two genes
cd8 = st.subset(st.obs["cell_type"] == "CD8 T")  # in-memory Dataset with only these cells
print(cd8.n_cells, two.shape)
```

## `qc`: QC metrics and cell filtering

Asks like: "filter low-quality cells"; "percent mitochondrial"; "QC"

```python
import arsc

ds = arsc.datasets.toy()
counts, genes = ds.X("RNA", "counts"), ds.features("RNA")
qc = arsc.qc_metrics(counts, genes, qc_vars={"mt": genes.str.startswith("MT-")})   # scanpy columns
keep = ((qc["n_genes_by_counts"] >= 20) & (qc["pct_counts_mt"] < 20)).to_numpy()     # tune on real data
ds = ds.subset(keep)
print(f"kept {keep.sum()} of {len(keep)} cells")
```

## `cluster`: Normalise, HVG, PCA, Harmony, neighbours, clusters, UMAP

Asks like: "cluster the cells"; "make a UMAP"; "standard Seurat workflow"; "integrate patients"

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
L = arsc.pp.log_normalize(counts)                                   # Seurat NormalizeData (exact)
hvg = arsc.pp.hvg_vst(counts, nfeatures=200, features=genes)        # use 2000 on real data
top = hvg.sort_values("rank").index[:200]
Z, _, _ = arsc.pp.scale(L[:, genes.get_indexer(top)])
pcs = arsc.pp.pca(Z, n_comps=20).embedding                          # cells x 20
emb = arsc.run_harmony(pcs, obs, "patient")                         # optional batch correction
snn = arsc.find_neighbors(emb, k=20)["snn"]
obs["cluster"] = pd.Categorical(arsc.cluster(snn, resolution=0.5).astype(str))   # 0 = largest
xy = arsc.umap(emb)                                                 # cells x 2 (plot elsewhere)
print(pd.crosstab(obs["cluster"], obs["cell_type"]))
```

## `reference_mapping`: Map a query dataset onto a reference (Symphony) and transfer its labels

Asks like: "map my cells onto a reference"; "transfer labels from a reference"; "symphony mapQuery"; "project a new dataset onto an existing atlas"

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
isref = (obs["timepoint"] == "pre").to_numpy()                    # reference = the cleanest, most complete set
ref = arsc.symphony.build_reference(counts[isref], genes, obs[isref], vars="patient",
                                    topn=200, d=10, K=20, umap=False)  # real data: topn 2000, d 20-30, K 100
q = arsc.symphony.map_query(counts[~isref], genes, obs[~isref], ref, vars="patient", umap=False)
lab = arsc.symphony.transfer_labels(ref, q, "cell_type", k=5)       # label + prob (share of the k votes)
lab["mapping_dist"] = arsc.symphony.mapping_score(ref, q)           # high = far from the reference
print(pd.crosstab(lab["label"], obs.loc[~isref, "cell_type"]))
print("low-confidence cells (prob < 0.6):", int((lab["prob"] < 0.6).sum()))
```

## `wnn`: Cluster on RNA + protein (CITE-seq) with weighted nearest neighbours

Asks like: "cluster on RNA and ADT together"; "weighted nearest neighbour clustering"; "CITE-seq WNN"

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()                                            # has assays RNA and ADT
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
L = arsc.pp.log_normalize(counts)
top = arsc.pp.hvg_vst(counts, nfeatures=200, features=genes).sort_values("rank").index[:200]
Z, _, _ = arsc.pp.scale(L[:, genes.get_indexer(top)])
rna = arsc.pp.pca(Z, n_comps=20).embedding
A = arsc.pp.clr(ds.X("ADT", "counts"))                              # Seurat CLR, margin 2
Za, _, _ = arsc.pp.scale(A)
adt = arsc.pp.pca(Za, n_comps=5).embedding                          # n_comps < number of proteins
w = arsc.wnn([rna, adt], k_nn=20)                                   # Seurat FindMultiModalNeighbors
obs["wnn_cluster"] = pd.Categorical(arsc.cluster(w.wsnn, resolution=0.5).astype(str))
obs["rna_weight"] = w.weights[:, 0]                                 # per-cell modality weights
print(pd.crosstab(obs["wnn_cluster"], obs["cell_type"]))
```

## `markers`: Marker genes of a cluster (e.g. the biggest one)

Asks like: "which genes define cluster 3"; "markers of the cluster with the most cells"; "FindAllMarkers"

```python
import arsc

ds = arsc.datasets.toy()
genes = ds.features("RNA")
L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
labels = ds.obs["cell_type"].astype(str)          # or your cluster column
res = arsc.wilcoxauc(L, labels, features=genes)   # presto: every group vs the rest, all genes
biggest = labels.value_counts().index[0]
top = arsc.top_markers(res, n=10)
print(top[top["group"] == biggest][["feature", "auc", "logFC", "pct_in", "pct_out", "padj"]])
# The same question as a Seurat FindAllMarkers table (matches Seurat 4.3):
fam = arsc.find_all_markers(L, labels, features=genes, only_pos=True)
print(fam[fam["cluster"] == biggest].head(10))
```

## `two_groups_cells`: Genes that differ between two clusters (descriptive, cell level)

Asks like: "compare cluster A with cluster B"; "FindMarkers ident.1 ident.2"

```python
import arsc

ds = arsc.datasets.toy()
genes = ds.features("RNA")
L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
fm = arsc.find_markers(L, ds.obs["cell_type"].astype(str), "CD8 T", "CD4 T", features=genes)
print(fm.head(10))      # Seurat columns: p_val, avg_log2FC, pct.1, pct.2, p_val_adj
# Cell-level p-values describe; they are not evidence about patients. For conditions use pseudobulk (below).
```

## `annotate`: Annotate cells with signature scores

Asks like: "annotate the clusters"; "score a gene signature"; "cell cycle phase"

```python
import arsc

ds = arsc.datasets.toy()
counts, genes = ds.X("RNA", "counts"), ds.features("RNA")
sigs = {"CD8 T": ["CD8A", "CD8B", "CD3E"], "CD4 T": ["CD4", "IL7R", "CD3E"], "B": ["MS4A1", "CD79A", "CD19"],
        "NK": ["NKG7", "GNLY", "KLRD1"], "Mono": ["LYZ", "CD14", "S100A8"]}
uc = arsc.score_ucell(counts, genes, sigs)            # UCell (rank based); columns "<name>_UCell"
pred = uc.idxmax(axis=1).str.removesuffix("_UCell").to_numpy()
print("agreement with truth:", (pred == ds.obs["cell_type"].astype(str).to_numpy()).mean().round(2))
L = arsc.pp.log_normalize(counts)
ms = arsc.add_module_score(L, genes, {"cytotoxic": ["GZMK", "GZMB", "PRF1", "IFNG"]}, ctrl=5)  # Seurat; ctrl=100 on full data
cc = arsc.cell_cycle_scoring(L, genes, ctrl=5)        # S.Score, G2M.Score, Phase (row order = cells)
ds.obs["cytotoxic"] = ms["cytotoxic"].to_numpy()
ds.obs["phase"] = cc["Phase"].to_numpy()
```

## `composition`: Does a cell type change in frequency (paired, and between arms)?

Asks like: "do CD8 T cells increase after treatment"; "compare cell-type proportions between responders and non-responders"

```python
import numpy as np
import arsc

ds = arsc.datasets.toy()
comp = arsc.stats.composition(ds.obs, "cell_type", sample_cols=["patient", "timepoint"], extra_cols=["response"])
cd8 = comp[comp["celltype"] == "CD8 T"]                       # one row per sample: k, n, prop
# 1) within-patient change pre -> post, per arm (unit = patient; exact Wilcoxon; bootstrap CI; LOO range)
print(arsc.stats.paired_test(cd8, "prop", t0="pre", t1="post", patient="patient", time="timepoint",
                             by="response", transform="logit").T)
# 2) does the change differ between arms? compare per-patient deltas
w = cd8.pivot_table(index=["patient", "response"], columns="timepoint", values="prop", observed=True).reset_index()
logit = lambda p: np.log(p / (1 - p))
w["delta"] = logit(w["post"]) - logit(w["pre"])
print(arsc.stats.arm_contrast(w.loc[w.response == "R", "delta"], w.loc[w.response == "NR", "delta"]))
# 3) count model: binomial GLMM with a patient random intercept (lme4::glmer equivalent)
print(arsc.stats.glmm_prop(cd8, fixed="timepoint", groups=("patient",)))
```

## `de_paired`: Pseudobulk DE within patients (post vs pre) in one cell type

Asks like: "what changes in CD8 T cells after treatment"; "pseudobulk DE with patient as unit"; "paired DESeq2"

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
cd8 = (obs["cell_type"] == "CD8 T").to_numpy()
pb = arsc.pseudobulk(ds.X("RNA", "counts")[cd8], obs[cd8], ["patient", "timepoint"],
                     features=ds.features("RNA"), min_cells=10)     # one column per patient x timepoint
cnt, cd = pb.to_frame(), pb.groups                                  # genes x samples, sample table
cd["timepoint"] = pd.Categorical(cd["timepoint"], categories=["pre", "post"])   # reference first
# DESeq2, paired by patient (patient in the design)
res = arsc.de.deseq2(cnt, cd, "~ patient + timepoint", ("timepoint", "post", "pre"))
print(res.head(10)[["gene", "log2FoldChange", "pvalue", "padj"]])
# limma-voom alternative: patient as a duplicateCorrelation block (also handles unpaired samples)
vm = arsc.de.voom(cnt, cd, "~ timepoint", coef=("timepoint", "post"), block="patient")
print(vm.head(10)[["gene", "logFC", "P.Value", "adj.P.Val"]])
```

## `de_groups`: Compare responders and non-responders in one cell type (pseudobulk, between patients)

Asks like: "compare responders and non-responders in CD8 T cells"; "DE between two patient groups"

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
m = ((obs["cell_type"] == "CD8 T") & (obs["timepoint"] == "post")).to_numpy()   # one sample per patient
pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], obs[m], ["patient"], features=ds.features("RNA"), min_cells=10)
cnt, cd = pb.to_frame(), pb.groups
cd["response"] = cd["patient"].map(obs.groupby("patient", observed=True)["response"].first().astype(str))
print(cd)                                                             # check n per group before trusting p
res = arsc.de.deseq2(cnt, cd, "~ response", ("response", "R", "NR"))
print(res.head(10)[["gene", "log2FoldChange", "pvalue", "padj"]])
print(arsc.stats.mde(3, 3))   # 3 vs 3 patients: no exact two-group test can reach p < 0.05
```

## `gsea`: Pathways from a DE ranking (preranked GSEA)

Asks like: "which pathways change"; "run GSEA on the DE result"; "hallmark enrichment"

```python
import pandas as pd
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
cd8 = (obs["cell_type"] == "CD8 T").to_numpy()
pb = arsc.pseudobulk(ds.X("RNA", "counts")[cd8], obs[cd8], ["patient", "timepoint"], features=ds.features("RNA"), min_cells=10)
cnt, cd = pb.to_frame(), pb.groups
res = arsc.de.deseq2(cnt, cd, "~ patient + timepoint", ("timepoint", "post", "pre"))
genesets = {"cytotoxic": ["GZMK", "GZMB", "PRF1", "IFNG", "CXCL13", "NKG7", "GNLY"],
            "naive": ["TCF7", "SELL", "CCR7", "IL7R"],
            "b_cell": ["MS4A1", "CD79A", "CD19"]}
# real data: genesets = arsc.de.msigdb("H")  (exports once with R msigdbr, then cached)
#            or arsc.genesets.read_genesets("sets.gmt")
gs = arsc.de.fgsea(res.set_index("gene")["stat"], genesets, min_size=3)
print(gs[["pathway", "NES", "pval", "padj", "size"]])
```

## `tf_activity`: Transcription-factor or pathway activity (regulons as gene sets)

Asks like: "which transcription factors are active"; "TF activity"; "PROGENy pathway activity"

```python
import pandas as pd
import arsc

# arsc has no decoupler-style ULM/VIPER yet. A validated route: GSEA of the DE statistic over each TF's
# positive targets (e.g. CollecTRI or DoRothEA exported to GMT; signs of repressed targets are ignored).
ds = arsc.datasets.toy()
obs = ds.obs
cd8 = (obs["cell_type"] == "CD8 T").to_numpy()
pb = arsc.pseudobulk(ds.X("RNA", "counts")[cd8], obs[cd8], ["patient", "timepoint"], features=ds.features("RNA"), min_cells=10)
res = arsc.de.deseq2(pb.to_frame(), pb.groups, "~ patient + timepoint", ("timepoint", "post", "pre"))
regulons = {"TBX21_targets": ["IFNG", "GZMB", "PRF1", "CXCL13", "GZMK"],   # illustration only
            "TCF7_targets": ["TCF7", "SELL", "CCR7", "IL7R"]}            # real: arsc.genesets.read_genesets("collectri.gmt")
print(arsc.de.fgsea(res.set_index("gene")["stat"], regulons, min_size=3)[["pathway", "NES", "pval"]])
```

## `score_change`: Does a signature score change per patient?

Asks like: "is the exhaustion score higher after treatment"; "per-patient signature change"

```python
import arsc

ds = arsc.datasets.toy()
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
obs["cyto"] = arsc.score_ucell(counts, genes, {"cyto": ["GZMK", "GZMB", "PRF1", "IFNG"]})["cyto_UCell"].to_numpy()
cd8 = obs[obs["cell_type"] == "CD8 T"]
per_sample = arsc.stats.sample_means(cd8, ["cyto"], sample_cols=["patient", "timepoint"], min_cells=10)
per_sample["response"] = per_sample["patient"].map(obs.groupby("patient", observed=True)["response"].first())
print(arsc.stats.paired_test(per_sample, "cyto", t0="pre", t1="post", patient="patient", time="timepoint",
                             by="response").T)
```

## `clone_expansion`: Do clones expand after treatment, and more in responders?

Asks like: "do these clones expand after treatment"; "clonal expansion between timepoints"; "TCR expansion responders vs non-responders"

```python
import arsc

ds = arsc.datasets.toy()
obs = ds.obs                                                        # one row per cell; clone_id NaN = no TCR
ct = arsc.clones.clone_table(obs, "clone_id", sample_cols=["patient", "timepoint"])   # n, freq per sample
dyn = arsc.clones.clone_dynamics(ct, "pre", "post", patient="patient", time="timepoint")  # Fisher per clone, BH within patient
print(dyn.sort_values("p").head())
# patient-level summary: share of the post repertoire in clones called expanded (BH FDR < 0.05 within patient)
dyn["up"] = dyn["class"].isin(["expanded", "novel_expanded"])
per_pt = dyn.groupby("patient").apply(lambda d: d.loc[d.up, "n_1"].sum() / d["N1"].iloc[0], include_groups=False)
per_pt = per_pt.rename("expanded_frac").reset_index()
per_pt["response"] = per_pt["patient"].map(obs.groupby("patient", observed=True)["response"].first())
print(per_pt)
# exact permutation test on the difference in means (the default, difference in medians, ties when many values are 0)
print(arsc.stats.perm_test(per_pt.loc[per_pt.response == "R", "expanded_frac"].to_numpy(),
                           per_pt.loc[per_pt.response == "NR", "expanded_frac"].to_numpy(),
                           stat=lambda a, b: a.mean() - b.mean()))   # 3 vs 3: the smallest possible p is 0.1
```

## `clone_diversity`: Repertoire diversity and STARTRAC indices

Asks like: "is the repertoire more clonal after treatment"; "clonality"; "STARTRAC expansion index"

```python
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
ct = arsc.clones.clone_table(obs, "clone_id", sample_cols=["patient", "timepoint"])
div = arsc.clones.clone_diversity(ct, sample_cols=["patient", "timepoint"], floor=50)   # rarefied to a common depth
print(arsc.stats.paired_test(div, "clonality", t0="pre", t1="post", patient="patient", time="timepoint").T)
cluster_idx, pairwise = arsc.clones.startrac(obs, clone_col="clone_id", state_col="cell_type",
                                             group_cols="patient", min_cells=10)   # per patient
print(cluster_idx.head())
```

## `power`: Before testing: how many patients, and what effect can I detect?

Asks like: "is this comparison powered"; "how many samples do I have"; "minimum detectable effect"

```python
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
cd8 = obs[obs["cell_type"] == "CD8 T"]
print(cd8.groupby(["response", "timepoint"], observed=True)["patient"].nunique())   # patients per cell
print(cd8.groupby("sample").size().describe())                                     # cells per sample
print(arsc.stats.mde(6))       # paired, 6 patients: effect in SD units detectable at 80% power
print(arsc.stats.mde(3, 3))    # two groups of 3: infeasible (minimum exact p = 0.1)
```

## `handoff`: Hand results to scanpy, R or a CSV

Asks like: "give me an AnnData"; "export for plotting in R"; "save the table"

```python
import os, tempfile
import arsc

ds = arsc.datasets.toy()
cd8 = (ds.obs["cell_type"] == "CD8 T").to_numpy()
adata = ds.to_anndata(cells=cd8, genes=["CD8A", "GZMK", "PDCD1"])      # in-memory AnnData for scanpy
out = tempfile.mkdtemp()
ds.write_h5ad(os.path.join(out, "cd8.h5ad"), cells=cd8)              # streamed, any size
arsc.write_table(ds.obs, os.path.join(out, "obs.csv"))                # tidy table for ggplot
print(adata)
```
