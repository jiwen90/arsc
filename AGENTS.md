# arsc: instructions for AI agents

arsc is a Python single-cell analysis engine. It reproduces the numerical results of Seurat, scanpy, presto, UCell,
harmony, DESeq2/limma-voom, fgsea and published TCR-clone methods (each validated against its reference; see the
`*_PARITY.md` files), and it runs on memory-mapped stores that many processes share without copying. It returns
arrays and tidy `pandas` tables. **It does not plot**: write a tidy table and plot it with your plotting tool.

Use this file to turn a biologist's question into correct arsc code. Every code block below runs as written on
the built-in toy data (`tests/test_docs_snippets.py` executes them). In Python, `arsc.help()` prints the task map
and `arsc.recipes("<question>")` prints the matching recipe.

## Install

```bash no-run
pip install "arsc[all] @ git+https://github.com/jiwen90/arsc.git"   # PyPI name "arsc" belongs to another project
# extras: h5ad (anndata, h5py), graph (leiden, umap), stats (pydeseq2, statsmodels, patsy), all = everything
```

## Mental model (read this first)

* **Store** (`arsc.open(dir)`): a directory of `.npy` files opened with `mmap`. Nothing is read until you ask;
  `st.X()` costs no RAM of its own, and N processes share one physical copy. Build a store once with
  `arsc.read(file, store=dir)` from `.h5ad`, a 10x `.h5` or `mtx` directory, or a Seurat `.qs`/`.rds` (Seurat needs R).
* **Dataset**: the same interface held in memory. `arsc.read(file)` (no `store=`), `st.subset(mask)` and
  `arsc.datasets.toy()` return one.
* Both have `.obs` (a cell metadata DataFrame indexed by barcode), `.X(assay="RNA", layer=None)` (a cells x features
  CSR matrix; the default layer is `counts`, otherwise the first layer), `.features(assay)` (a gene `Index`
  aligned to the columns of `X`), `.subset(cells)`, `.to_anndata()` and `.write_h5ad()`. Assays are modalities
  (`RNA`, `ADT`); layers are matrices within an assay (`counts`, `data`, ...). A Store also has `.get_features(genes)`.
* **Functions take arrays, not objects**: `f(X, labels_or_obs, features=genes, ...)`. Rows of `X` are cells in
  the same order as `obs`. You pass the layer you mean (raw counts or log-normalised), and the column names you mean.
* **Labels from elsewhere** (an annotation CSV): join by barcode, `obs["label"] = labels.reindex(obs.index)`; then
  work on `st.subset(obs["label"] == "CD8 T")`.
* **The toy data** (`arsc.datasets.toy()`) has 3,000 cells x 300 genes, 6 patients (`P1`-`P3` responders `R`,
  `P4`-`P6` non-responders `NR`) x 2 timepoints (`pre`, `post`), 5 cell types, `clone_id` for T cells and an `ADT`
  assay. Planted truth: in responders, CD8 T cells up-regulate `GZMK GZMB IFNG PRF1 CXCL13` after treatment,
  become more frequent, and three clones per responder expand.

## Statistics rules (non-negotiable)

1. **The patient (sample) is the unit of inference; cells are not replicates.** To compare conditions, aggregate
   first (`arsc.pseudobulk`, `stats.composition`, `stats.sample_means`) and test per patient. Never report a
   cell-level p-value (`wilcoxauc`, `find_markers`) as evidence that conditions differ; those rank genes that
   describe a cluster.
2. **Paired designs stay paired**: put `patient` in the DESeq2 design (`~ patient + timepoint`), use
   `block="patient"` in voom, or use `stats.paired_test`.
3. **Know the floor before testing**: `stats.mde(n)` / `stats.mde(n1, n2)`. With 3 vs 3 patients no exact
   two-group test reaches p < 0.05; with 6 pairs the minimum exact paired p is 0.031. Report effect sizes, CIs and
   per-patient values (each test table includes them), not just p.
4. **Check robustness**: the leave-one-patient-out range (`loo_min`/`loo_max` in `paired_test`, `arm_contrast`;
   `stats.loo_effect` for anything else) and a second seed for anything stochastic.
5. Correct for the family you tested (`padj`/`fdr` columns; `stats.bh(p)`).

## Task map: "I want to ..." -> call

<!-- TASKS:START -->
| Area | I want to ... | Call |
|---|---|---|
| Load | open an existing arsc store (memory-mapped, zero-copy) | `st = arsc.open(dir)` |
| Load | read h5ad / 10x .h5 / 10x mtx dir / Seurat .qs|.rds into memory | `ds = arsc.read(path)` |
| Load | convert a big file once into a store (streamed, bounded RAM) | `st = arsc.read(path, store=dir)` |
| Load | get a matrix (cells x features, CSR, no copy) | `st.X('RNA', 'counts')  # also ds.X(...)` |
| Load | a few genes for all cells (reads only those genes) | `st.get_features(['CD8A', 'GZMK'])` |
| Load | materialise a cell subset | `sub = st.subset(mask)  # -> in-memory Dataset` |
| Load | hand off to scanpy / write h5ad | `st.to_anndata(cells, genes); st.write_h5ad(path, cells)` |
| Load | toy data for trying things | `arsc.datasets.toy()` |
| QC | per-cell QC metrics (scanpy calculate_qc_metrics) | `arsc.qc_metrics(counts, genes, qc_vars={'mt': mask})` |
| Normalise | LogNormalize (Seurat NormalizeData) | `arsc.pp.log_normalize(counts)` |
| Normalise | normalize_total + log1p (scanpy, bitwise) | `arsc.normalize_total_log1p(counts)` |
| Normalise | CLR for ADT (Seurat) | `arsc.pp.clr(adt_counts)` |
| HVG | Seurat vst highly variable genes | `arsc.pp.hvg_vst(counts, nfeatures=2000, features=genes)` |
| HVG | scanpy highly_variable_genes (seurat / cell_ranger / seurat_v3) | `arsc.highly_variable_genes(X, genes, flavor=...)` |
| Reduce | scale (Seurat ScaleData, optional regression) | `arsc.pp.scale(L[:, idx]) / arsc.scale_data(...)` |
| Reduce | PCA (Seurat RunPCA conventions) | `arsc.pp.pca(Z, n_comps=50).embedding` |
| Reduce | Harmony batch correction (R harmony 1.2 port) | `arsc.run_harmony(pcs, obs, 'batch')` |
| Graph | kNN + SNN graph (Seurat FindNeighbors, exact kNN) | `arsc.find_neighbors(emb, k=20)['snn']` |
| Graph | multimodal WNN (Seurat FindMultiModalNeighbors) | `arsc.wnn([rna_emb, adt_emb]).wsnn` |
| Graph | scanpy neighbors / leiden | `arsc.scanpy_extra.neighbors(emb); arsc.scanpy_extra.leiden(conn)` |
| Cluster | Leiden / Louvain clusters | `arsc.cluster(snn, resolution=0.8)` |
| Cluster | UMAP coordinates (Seurat RunUMAP defaults) | `arsc.umap(emb)` |
| Markers | markers of every group vs rest (presto wilcoxauc) | `arsc.wilcoxauc(L, labels, features=genes)` |
| Markers | top n markers per group | `arsc.top_markers(res, n=10)` |
| Markers | Seurat FindMarkers / FindAllMarkers | `arsc.find_markers(L, labels, 'A', 'B', features=genes)` |
| Markers | scanpy rank_genes_groups | `arsc.rank_genes_groups(L, labels, genes)` |
| Scores | UCell signature scores (rank-based, raw counts ok) | `arsc.score_ucell(counts, genes, {name: genes})` |
| Scores | Seurat AddModuleScore / scanpy score_genes | `arsc.add_module_score(L, genes, sigs) / arsc.score_genes(L, genes, list)` |
| Scores | cell-cycle phase (Seurat CellCycleScoring) | `arsc.cell_cycle_scoring(L, genes)` |
| Aggregate | pseudobulk counts per sample (x cell type) | `arsc.pseudobulk(counts, obs, ['patient', 'timepoint'], features=genes)` |
| Aggregate | average expression per group (Seurat) | `arsc.average_expression(L, obs, 'cell_type', features=genes)` |
| Aggregate | cell-type proportions per sample | `arsc.stats.composition(obs, 'cell_type', sample_cols=[...])` |
| Aggregate | per-sample means of a per-cell score | `arsc.stats.sample_means(obs, ['score'], sample_cols=[...])` |
| DE | pseudobulk DESeq2 (pydeseq2; paired via ~ patient + condition) | `arsc.de.deseq2(cnt, coldata, design, (var, level, ref))` |
| DE | pseudobulk limma-voom (+ duplicateCorrelation block) | `arsc.de.voom(cnt, coldata, '~ cond', coef=(var, level), block='patient')` |
| Gene sets | preranked GSEA (fgsea) | `arsc.de.fgsea(stat_series, genesets)` |
| Gene sets | MSigDB sets (needs R msigdbr once, then cached) | `arsc.de.msigdb('H')` |
| Gene sets | read GMT / JSON / CSV gene sets | `arsc.genesets.read_genesets(path)` |
| Stats | paired pre/post change per patient (exact Wilcoxon, bootstrap CI, LOO) | `arsc.stats.paired_test(df, 'value', t0, t1, patient=..., time=...)` |
| Stats | two-group permutation test on per-patient values | `arsc.stats.perm_test(a, b)` |
| Stats | between-arm contrast of per-patient deltas | `arsc.stats.arm_contrast(delta_a, delta_b)` |
| Stats | binomial GLMM on proportions (lme4 glmer) | `arsc.stats.glmm_prop(comp, fixed='timepoint', groups=('patient',))` |
| Stats | linear mixed model on a per-sample score | `arsc.stats.lmm_score(df, 'score ~ timepoint', group='patient')` |
| Stats | minimum detectable effect before testing | `arsc.stats.mde(n_patients) / arsc.stats.mde(n1, n2)` |
| Stats | leave-one-patient-out effect | `arsc.stats.loo_effect(df, 'patient', fn)` |
| Stats | Benjamini-Hochberg FDR | `arsc.stats.bh(p)` |
| Clones | clone counts / frequency per sample | `arsc.clones.clone_table(obs, 'clone_id', sample_cols=[...])` |
| Clones | clones expanding/contracting between two timepoints (per patient) | `arsc.clones.clone_dynamics(ct, t0, t1, patient=..., time=...)` |
| Clones | repertoire diversity (rarefied) | `arsc.clones.clone_diversity(ct, sample_cols=[...])` |
| Clones | clone trajectories over >= 3 timepoints (Cyclone, exact) | `arsc.clones.cyclone(obs, clone_col=..., patient_col=..., time_col=...)` |
| Clones | STARTRAC expansion / transition indices | `arsc.clones.startrac(obs, clone_col=..., state_col=..., group_cols=...)` |
| Clones | clonal fate (CloneTrace port) | `arsc.clonotrace.run(embedding, clones, ...)` |
| Trajectory | diffusion map / DPT / Palantir diffusion / MAGIC | `arsc.diffmap, arsc.dpt, arsc.scanpy_extra.palantir_diffusion_maps, arsc.magic_impute` |
| Utility | threads | `arsc.set_threads(n); env AR_THREADS` |
<!-- TASKS:END -->

## Translate English to arsc

| The question sounds like | Recipe | Unit / test |
|---|---|---|
| "load / open / convert my data", "read the 10x output" | `load` | |
| "filter bad cells", "percent mito" | `qc` | |
| "cluster the cells", "make a UMAP", "integrate the patients" | `cluster` | |
| "cluster on RNA and protein", "CITE-seq WNN" | `wnn` | |
| "which genes define cluster 3", "markers of the biggest cluster" | `markers` | descriptive, cell level |
| "how does cluster A differ from cluster B" | `two_groups_cells` | descriptive, cell level |
| "annotate the clusters", "score this signature", "cell cycle" | `annotate` | |
| "do CD8 T cells increase after treatment", "proportions in responders vs non-responders" | `composition` | patient; paired Wilcoxon, permutation, GLMM |
| "what changes in CD8 T cells after treatment", "pseudobulk DE with patient as unit" | `de_paired` | patient x timepoint pseudobulk; DESeq2 `~ patient + timepoint` |
| "compare responders and non-responders in CD8 T cells" | `de_groups` | one pseudobulk per patient; DESeq2 `~ response` |
| "which pathways change", "GSEA", "hallmarks" | `gsea` | ranking from a pseudobulk DE |
| "which TFs are active", "pathway activity" | `tf_activity` | ranking from a pseudobulk DE |
| "is the signature score higher after treatment" | `score_change` | per-sample mean score; paired test |
| "do these clones expand after treatment", "more expansion in responders?" | `clone_expansion` | clone Fisher within patient, then patient-level summary |
| "is the repertoire more clonal", "STARTRAC" | `clone_diversity` | per-sample rarefied diversity; paired test |
| "is this powered", "how many patients do I have" | `power` | |
| "give me an AnnData", "export for R" | `handoff` | |

## Recipes

<!-- RECIPES:START -->
### `load`: Load data or build a memory-mapped store

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

### `qc`: QC metrics and cell filtering

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

### `cluster`: Normalise, HVG, PCA, Harmony, neighbours, clusters, UMAP

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

### `wnn`: Cluster on RNA + protein (CITE-seq) with weighted nearest neighbours

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

### `markers`: Marker genes of a cluster (e.g. the biggest one)

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

### `two_groups_cells`: Genes that differ between two clusters (descriptive, cell level)

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

### `annotate`: Annotate cells with signature scores

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

### `composition`: Does a cell type change in frequency (paired, and between arms)?

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

### `de_paired`: Pseudobulk DE within patients (post vs pre) in one cell type

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

### `de_groups`: Compare responders and non-responders in one cell type (pseudobulk, between patients)

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

### `gsea`: Pathways from a DE ranking (preranked GSEA)

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

### `tf_activity`: Transcription-factor or pathway activity (regulons as gene sets)

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

### `score_change`: Does a signature score change per patient?

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

### `clone_expansion`: Do clones expand after treatment, and more in responders?

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

### `clone_diversity`: Repertoire diversity and STARTRAC indices

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

### `power`: Before testing: how many patients, and what effect can I detect?

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

### `handoff`: Hand results to scanpy, R or a CSV

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
<!-- RECIPES:END -->

## Gotchas

* **Layers after import**: an `.h5ad`'s `X` becomes layer `"X"` (its `layers` keep their names, `raw` becomes
  its own assay or layer); 10x and Seurat imports give `"counts"`. Check `st.layers("RNA")` and pass the layer
  that holds raw counts. An `.h5ad` holds one modality: for RNA + ADT, import the 10x `.h5` or the Seurat object.
  `arsc.datasets.toy(store=dir)` writes a two-assay store directly.
* **Marker direction**: `find_markers` / `find_all_markers` return up- and down-regulated genes unless
  `only_pos=True`; for "markers of X" pass `only_pos=True` (or keep `logFC > 0` from `wilcoxauc`).
* **Always pass column names.** Some defaults are `patient_alias`, `timepoint` or `sample_id` (`stats.*`,
  `clones.*`). Pass `patient=`, `time=`, `sample_cols=`, `patient_col=`, `time_col=`, `group_cols=` explicitly.
* **Reference levels**: `pseudobulk(...).groups` holds plain strings, so patsy (voom) takes the alphabetical first
  level as reference ("post" before "pre"). Set `cd[v] = pd.Categorical(cd[v], categories=[reference, other])`
  before `de.voom`. `de.deseq2` takes an explicit `(variable, level, reference)` contrast.
* **Which layer**: `log_normalize`, `hvg_vst`, `qc_metrics`, `pseudobulk`, `score_ucell` and `de.*` take **raw
  counts**. `wilcoxauc`, `find_markers`, `find_all_markers`, `add_module_score`, `cell_cycle_scoring` and
  `average_expression` take **log-normalised** data. `rank_genes_groups` and `highly_variable_genes(flavor="seurat")`
  take `normalize_total_log1p` output. `pp.pca` takes the **scaled** HVG matrix, so it does not re-centre.
* **Positional outputs**: `qc_metrics`, `cell_cycle_scoring`, `add_module_score` and `score_ucell` return one row
  per cell in input order; assign with `.to_numpy()` (`obs["x"] = res["x"].to_numpy()`), not by index alignment.
  `cluster` and `umap` return numpy arrays.
* `hvg_vst(...).highly_variable` is not ranked: use `hvg.sort_values("rank").index[:n]` for Seurat's order.
* `stats.composition` renames the label column to `celltype` and returns `k`, `n`, `prop`.
* `stats.perm_test` defaults to the difference of medians, which ties across permutations: 4 vs 4 completely
  separated patients give p = 0.057, not the 0.029 floor that `stats.mde(4, 4)` reports. Pass
  `stat=lambda a, b: a.mean() - b.mean()` to reach the floor.
* `arsc.read(file, store=dir)` reopens an existing store built from a file of the same name instead of
  rebuilding it (it warns if the file is newer). Pass `overwrite=True` after the source file changes.
* `add_module_score` / `cell_cycle_scoring` sample `ctrl` control genes per expression bin; on small gene sets
  (the toy has 300 genes) pass `ctrl=5`. The real-data default is 100.
* Seeds are integers with fixed defaults (`seed`, or `random_state` in the scanpy-parity functions); `None` is
  rejected and the global RNG is never touched. Same input + seed gives an identical result (`SEEDS.md`).
* The first `umap` call in a process takes about 15 s (umap-learn's numba compile); later calls are fast.
* `de.msigdb("H")` needs R + msigdbr once, then reads a cache (`ARSC_CACHE`, default `~/.cache/arsc`). Without R,
  pass your own sets: `arsc.genesets.read_genesets("sets.gmt")`. `de.deseq2` needs `pydeseq2` (`arsc[stats]`).
* Preranked GSEA (`de.fgsea`) permutes genes. Sets of co-expressed genes (cell cycle, ribosomal) get p-values that
  are too small. The smallest p is 1/(nperm+1). Back a pathway claim with sample-level evidence.
* **Memory**: `st.X()` is free; `X.toarray()` on a full matrix is not (cells x genes x 8 bytes). Subset cells
  (`st.subset`) and genes (`X[:, idx]`, `st.get_features`) before densifying. Pseudobulk and marker functions
  stay sparse.
* **Windows vs Linux**: identical results. Threads come from `AR_THREADS` or `arsc.set_threads(n)`. The page-cache
  sharing of stores works on both; on Linux, forked workers also share the parent's open maps.

### Exact vs approximate (summary; full tables in the PARITY files)

| Function | Agreement with the reference |
|---|---|
| `pp.log_normalize`, `pseudobulk`, `pp.clr`, `average_expression`, `normalize_total_log1p` | exact (max diff 0 to 1e-14) |
| `score_ucell`, `add_module_score`, `cell_cycle_scoring`, `score_genes`, `qc_metrics` | exact (<= 1e-14; Phase identical) |
| `wilcoxauc` (presto), `find_markers` / `find_all_markers` (Seurat), `rank_genes_groups` (scanpy) | statistics exact; p within 1e-13 to 4e-8 relative; same gene order |
| `pp.hvg_vst`, `highly_variable_genes` | values within 1e-9; same gene set (ties at the cutoff can swap one gene) |
| `pp.pca` | top-20 subspace cosine >= 0.9999 vs Seurat; per-PC r >= 0.992 (signs fixed) |
| `find_neighbors` / `knn` | exact kNN. Seurat uses approximate Annoy, so SNN Jaccard vs Seurat is 0.66 and Leiden ARI 0.92 |
| `scanpy_extra.neighbors` / `leiden` / `umap_from_connectivities` | identical to scanpy below 4,096 cells (scanpy switches to approximate kNN above) |
| `run_harmony` | same objective path as R harmony 1.2; per-dimension r 0.989-0.998 |
| `wnn` | modality weight r 0.97; neighbour overlap 0.92; cluster ARI 0.87 vs Seurat |
| `de.deseq2` (pydeseq2) | log2FC r 0.9998, top-100 overlap 98-100 vs R DESeq2 |
| `de.voom` (+ duplicateCorrelation) | logFC/t r 1.00000; max abs diff 1e-4 vs limma |
| `de.fgsea` | ES exact; NES r 0.99995; p sampled (fgseaSimple) |
| `stats.wilcoxon_signed_rank`, `perm_test` | exact vs R `wilcox.test` / enumeration |
| `stats.glmm_prop` | fixed effects within 2.2e-4 of `lme4::glmer` |
| `clones.cyclone`, `startrac`, `clone_dynamics`, `clone_table` | exact (published Cyclone assignment reproduced, ARI 1.0) |
| `clones.clone_diversity`, `clone_expansion_null` | Monte-Carlo; r > 0.998 vs R |

## How to verify a result

1. Run the recipe on `arsc.datasets.toy()` first; it should recover the planted effect described above.
2. Print the n: patients per group and cells per sample (`power` recipe). Say so if a group has fewer than 3 patients.
3. Look at per-patient values (the `paired_test` table, the pseudobulk per-sample matrix), not only the pooled test.
4. Leave one patient out (`loo_min`/`loo_max`, `stats.loo_effect`). If one patient carries the effect, report that.
5. Re-run stochastic steps (harmony, clustering, UMAP, GSEA) with another `seed`; claims must survive it.
6. To check arsc against Seurat or scanpy, use the PARITY tables; do not expect bitwise agreement where the table
   gives a correlation.

## Do not

* Do not treat cells as replicates or quote cell-level p-values for condition effects.
* Do not run DE on log-normalised data or on cells: pseudobulk raw counts, then `de.deseq2` / `de.voom`.
* Do not cluster and then test markers of those clusters as if independent (circular); markers describe clusters.
* Do not call `.toarray()` on a full store matrix, or load a whole Seurat object when a store exists.
* Do not change the input files; write results to a new directory.
* Do not pass `seed=None`, and do not rely on default column names.
* Do not claim "exact" agreement where the parity table gives a correlation.

## More

* `README.md`: overview, benchmarks. `docs/getting-started.md`: tutorial. `docs/recipes.md`: all recipes.
* `docs/api.md`: every public function (generated from docstrings). `docs/from-scanpy.md`, `docs/from-seurat.md`.
* `SEURAT_PARITY.md`, `SCANPY_PARITY.md`, `STATS_PARITY.md`, `CLONES_PARITY.md`, `CLONOTRACE_PARITY.md`, `SEEDS.md`.
