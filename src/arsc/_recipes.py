"""Task map and copy-paste recipes (the single source for ``arsc.help()``, ``arsc.recipes()`` and AGENTS.md).

Every recipe is a complete, runnable snippet on :func:`arsc.datasets.toy` data. ``tests/test_docs_snippets.py``
runs them all, and ``docs/build_docs.py`` copies them into AGENTS.md and docs/recipes.md (a test checks
that the copies are in sync). Replace ``arsc.datasets.toy()`` with ``arsc.open("your_store")`` (or
``arsc.read("file.h5ad")``) and the column names with yours.
"""
from __future__ import annotations

# (section, task in plain English, call)
TASKS = [
    ("Load", "open an existing arsc store (memory-mapped, zero-copy)", "st = arsc.open(dir)"),
    ("Load", "read h5ad / 10x .h5 / 10x mtx dir / Seurat .qs|.rds into memory", "ds = arsc.read(path)"),
    ("Load", "convert a big file once into a store (streamed, bounded RAM)", "st = arsc.read(path, store=dir)"),
    ("Load", "get a matrix (cells x features, CSR, no copy)", "st.X('RNA', 'counts')  # also ds.X(...)"),
    ("Load", "a few genes for all cells (reads only those genes)", "st.get_features(['CD8A', 'GZMK'])"),
    ("Load", "materialise a cell subset", "sub = st.subset(mask)  # -> in-memory Dataset"),
    ("Load", "hand off to scanpy / write h5ad", "st.to_anndata(cells, genes); st.write_h5ad(path, cells)"),
    ("Load", "toy data for trying things", "arsc.datasets.toy()"),
    ("QC", "per-cell QC metrics (scanpy calculate_qc_metrics)", "arsc.qc_metrics(counts, genes, qc_vars={'mt': mask})"),
    ("Normalise", "LogNormalize (Seurat NormalizeData)", "arsc.pp.log_normalize(counts)"),
    ("Normalise", "normalize_total + log1p (scanpy, bitwise)", "arsc.normalize_total_log1p(counts)"),
    ("Normalise", "CLR for ADT (Seurat)", "arsc.pp.clr(adt_counts)"),
    ("HVG", "Seurat vst highly variable genes", "arsc.pp.hvg_vst(counts, nfeatures=2000, features=genes)"),
    ("HVG", "scanpy highly_variable_genes (seurat / cell_ranger / seurat_v3)", "arsc.highly_variable_genes(X, genes, flavor=...)"),
    ("Reduce", "scale (Seurat ScaleData, optional regression)", "arsc.pp.scale(L[:, idx]) / arsc.scale_data(...)"),
    ("Reduce", "PCA (Seurat RunPCA conventions)", "arsc.pp.pca(Z, n_comps=50).embedding"),
    ("Reduce", "Harmony batch correction (R harmony 1.2 port)", "arsc.run_harmony(pcs, obs, 'batch')"),
    ("Reduce", "map a query onto a reference (Symphony: build once, map, kNN labels)", "ref = arsc.symphony.build_reference(counts, genes, obs, 'batch'); q = arsc.symphony.map_query(qcounts, qgenes, qobs, ref, 'batch')"),
    ("Reduce", "transfer reference labels to the query, with confidence", "arsc.symphony.transfer_labels(ref, q, 'cell_type', k=5)"),
    ("Graph", "kNN + SNN graph (Seurat FindNeighbors, exact kNN)", "arsc.find_neighbors(emb, k=20)['snn']"),
    ("Graph", "multimodal WNN (Seurat FindMultiModalNeighbors)", "arsc.wnn([rna_emb, adt_emb]).wsnn"),
    ("Graph", "scanpy neighbors / leiden", "arsc.scanpy_extra.neighbors(emb); arsc.scanpy_extra.leiden(conn)"),
    ("Cluster", "Leiden / Louvain clusters", "arsc.cluster(snn, resolution=0.8)"),
    ("Cluster", "UMAP coordinates (Seurat RunUMAP defaults)", "arsc.umap(emb)"),
    ("Markers", "markers of every group vs rest (presto wilcoxauc)", "arsc.wilcoxauc(L, labels, features=genes)"),
    ("Markers", "top n markers per group", "arsc.top_markers(res, n=10)"),
    ("Markers", "Seurat FindMarkers / FindAllMarkers", "arsc.find_markers(L, labels, 'A', 'B', features=genes)"),
    ("Markers", "scanpy rank_genes_groups", "arsc.rank_genes_groups(L, labels, genes)"),
    ("Scores", "UCell signature scores (rank-based, raw counts ok)", "arsc.score_ucell(counts, genes, {name: genes})"),
    ("Scores", "Seurat AddModuleScore / scanpy score_genes", "arsc.add_module_score(L, genes, sigs) / arsc.score_genes(L, genes, list)"),
    ("Scores", "cell-cycle phase (Seurat CellCycleScoring)", "arsc.cell_cycle_scoring(L, genes)"),
    ("Aggregate", "pseudobulk counts per sample (x cell type)", "arsc.pseudobulk(counts, obs, ['patient', 'timepoint'], features=genes)"),
    ("Aggregate", "average expression per group (Seurat)", "arsc.average_expression(L, obs, 'cell_type', features=genes)"),
    ("Aggregate", "cell-type proportions per sample", "arsc.stats.composition(obs, 'cell_type', sample_cols=[...])"),
    ("Aggregate", "per-sample means of a per-cell score", "arsc.stats.sample_means(obs, ['score'], sample_cols=[...])"),
    ("DE", "pseudobulk DESeq2 (pydeseq2; paired via ~ patient + condition)", "arsc.de.deseq2(cnt, coldata, design, (var, level, ref))"),
    ("DE", "pseudobulk limma-voom (+ duplicateCorrelation block)", "arsc.de.voom(cnt, coldata, '~ cond', coef=(var, level), block='patient')"),
    ("Gene sets", "preranked GSEA (fgsea)", "arsc.de.fgsea(stat_series, genesets)"),
    ("Gene sets", "MSigDB sets (needs R msigdbr once, then cached)", "arsc.de.msigdb('H')"),
    ("Gene sets", "read GMT / JSON / CSV gene sets", "arsc.genesets.read_genesets(path)"),
    ("Stats", "paired pre/post change per patient (exact Wilcoxon, bootstrap CI, LOO)", "arsc.stats.paired_test(df, 'value', t0, t1, patient=..., time=...)"),
    ("Stats", "two-group permutation test on per-patient values", "arsc.stats.perm_test(a, b)"),
    ("Stats", "between-arm contrast of per-patient deltas", "arsc.stats.arm_contrast(delta_a, delta_b)"),
    ("Stats", "binomial GLMM on proportions (lme4 glmer)", "arsc.stats.glmm_prop(comp, fixed='timepoint', groups=('patient',))"),
    ("Stats", "linear mixed model on a per-sample score", "arsc.stats.lmm_score(df, 'score ~ timepoint', group='patient')"),
    ("Stats", "minimum detectable effect before testing", "arsc.stats.mde(n_patients) / arsc.stats.mde(n1, n2)"),
    ("Stats", "leave-one-patient-out effect", "arsc.stats.loo_effect(df, 'patient', fn)"),
    ("Stats", "Benjamini-Hochberg FDR", "arsc.stats.bh(p)"),
    ("Stats", "sign-flip test of per-patient deltas (exact to n = 20, never-zero Monte Carlo p)", "arsc.stats.signflip_test(d, stat='mean')"),
    ("Stats", "family-wise sign-flip over K readouts on the same patients (maxT)", "arsc.stats.signflip_maxt(D)"),
    ("Stats", "Hodges-Lehmann estimate + Wilcoxon CI (R wilcox.test conf.int)", "arsc.stats.hodges_lehmann(x, y=None, paired=False)"),
    ("Stats", "common odds ratio over strata (Mantel-Haenszel, R mantelhaen.test)", "arsc.stats.mantel_haenszel(a, b, c, d)"),
    ("Stats", "intraclass correlation ICC(1), optionally within strata", "arsc.stats.icc_oneway(y, groups, strata=None)"),
    ("Stats", "covariate-adjusted permutation test of one coefficient (Freedman-Lane)", "arsc.stats.freedman_lane(y, x, covariates=Z)"),
    ("Stats", "chance that a 'k of n patients' rule passes (Poisson-binomial)", "arsc.stats.poisson_binomial(p)[k:].sum()"),
    ("Depth", "thin counts to a common depth per cell (hypergeometric / binomial)", "arsc.stats.downsample_counts(X, target)"),
    ("Depth", "expected clone richness at a common depth (vegan rarefy, closed form)", "arsc.stats.rarefy_richness(sizes, depth)"),
    ("Clones", "clone counts / frequency per sample", "arsc.clones.clone_table(obs, 'clone_id', sample_cols=[...])"),
    ("Clones", "clones expanding/contracting between two timepoints (per patient)", "arsc.clones.clone_dynamics(ct, t0, t1, patient=..., time=...)"),
    ("Clones", "repertoire diversity (rarefied)", "arsc.clones.clone_diversity(ct, sample_cols=[...])"),
    ("Clones", "clone trajectories over >= 3 timepoints (Cyclone, exact)", "arsc.clones.cyclone(obs, clone_col=..., patient_col=..., time_col=...)"),
    ("Clones", "STARTRAC expansion / transition indices", "arsc.clones.startrac(obs, clone_col=..., state_col=..., group_cols=...)"),
    ("Clones", "clonal fate (CloneTrace port)", "arsc.clonotrace.run(embedding, clones, ...)"),
    ("Trajectory", "diffusion map / DPT / Palantir diffusion / MAGIC", "arsc.diffmap, arsc.dpt, arsc.scanpy_extra.palantir_diffusion_maps, arsc.magic_impute"),
    ("Utility", "threads", "arsc.set_threads(n); env AR_THREADS (default 4)"),
]

RECIPES = [
    dict(
        id="load",
        title="Load data or build a memory-mapped store",
        asks=["load my h5ad", "open the dataset", "convert this Seurat object", "read 10x output"],
        code='''\
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
''',
    ),
    dict(
        id="qc",
        title="QC metrics and cell filtering",
        asks=["filter low-quality cells", "percent mitochondrial", "QC"],
        code='''\
import arsc

ds = arsc.datasets.toy()
counts, genes = ds.X("RNA", "counts"), ds.features("RNA")
qc = arsc.qc_metrics(counts, genes, qc_vars={"mt": genes.str.startswith("MT-")})   # scanpy columns
keep = ((qc["n_genes_by_counts"] >= 20) & (qc["pct_counts_mt"] < 20)).to_numpy()     # tune on real data
ds = ds.subset(keep)
print(f"kept {keep.sum()} of {len(keep)} cells")
''',
    ),
    dict(
        id="cluster",
        title="Normalise, HVG, PCA, Harmony, neighbours, clusters, UMAP",
        asks=["cluster the cells", "make a UMAP", "standard Seurat workflow", "integrate patients"],
        code='''\
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
''',
    ),
    dict(
        id="reference_mapping",
        title="Map a query dataset onto a reference (Symphony) and transfer its labels",
        asks=["map my cells onto a reference", "transfer labels from a reference", "symphony mapQuery",
              "project a new dataset onto an existing atlas"],
        code='''import pandas as pd
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
''',
    ),
    dict(
        id="wnn",
        title="Cluster on RNA + protein (CITE-seq) with weighted nearest neighbours",
        asks=["cluster on RNA and ADT together", "weighted nearest neighbour clustering", "CITE-seq WNN"],
        code='''\
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
''',
    ),
    dict(
        id="markers",
        title="Marker genes of a cluster (e.g. the biggest one)",
        asks=["which genes define cluster 3", "markers of the cluster with the most cells", "FindAllMarkers"],
        code='''\
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
''',
    ),
    dict(
        id="two_groups_cells",
        title="Genes that differ between two clusters (descriptive, cell level)",
        asks=["compare cluster A with cluster B", "FindMarkers ident.1 ident.2"],
        code='''\
import arsc

ds = arsc.datasets.toy()
genes = ds.features("RNA")
L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
fm = arsc.find_markers(L, ds.obs["cell_type"].astype(str), "CD8 T", "CD4 T", features=genes)
print(fm.head(10))      # Seurat columns: p_val, avg_log2FC, pct.1, pct.2, p_val_adj
# Cell-level p-values describe; they are not evidence about patients. For conditions use pseudobulk (below).
''',
    ),
    dict(
        id="annotate",
        title="Annotate cells with signature scores",
        asks=["annotate the clusters", "score a gene signature", "cell cycle phase"],
        code='''\
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
''',
    ),
    dict(
        id="composition",
        title="Does a cell type change in frequency (paired, and between arms)?",
        asks=["do CD8 T cells increase after treatment", "compare cell-type proportions between responders and non-responders"],
        code='''\
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
''',
    ),
    dict(
        id="de_paired",
        title="Pseudobulk DE within patients (post vs pre) in one cell type",
        asks=["what changes in CD8 T cells after treatment", "pseudobulk DE with patient as unit", "paired DESeq2"],
        code='''\
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
''',
    ),
    dict(
        id="de_groups",
        title="Compare responders and non-responders in one cell type (pseudobulk, between patients)",
        asks=["compare responders and non-responders in CD8 T cells", "DE between two patient groups"],
        code='''\
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
''',
    ),
    dict(
        id="gsea",
        title="Pathways from a DE ranking (preranked GSEA)",
        asks=["which pathways change", "run GSEA on the DE result", "hallmark enrichment"],
        code='''\
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
''',
    ),
    dict(
        id="tf_activity",
        title="Transcription-factor or pathway activity (regulons as gene sets)",
        asks=["which transcription factors are active", "TF activity", "PROGENy pathway activity"],
        code='''\
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
''',
    ),
    dict(
        id="score_change",
        title="Does a signature score change per patient?",
        asks=["is the exhaustion score higher after treatment", "per-patient signature change"],
        code='''\
import arsc

ds = arsc.datasets.toy()
counts, genes, obs = ds.X("RNA", "counts"), ds.features("RNA"), ds.obs
obs["cyto"] = arsc.score_ucell(counts, genes, {"cyto": ["GZMK", "GZMB", "PRF1", "IFNG"]})["cyto_UCell"].to_numpy()
cd8 = obs[obs["cell_type"] == "CD8 T"]
per_sample = arsc.stats.sample_means(cd8, ["cyto"], sample_cols=["patient", "timepoint"], min_cells=10)
per_sample["response"] = per_sample["patient"].map(obs.groupby("patient", observed=True)["response"].first())
print(arsc.stats.paired_test(per_sample, "cyto", t0="pre", t1="post", patient="patient", time="timepoint",
                             by="response").T)
''',
    ),
    dict(
        id="clone_expansion",
        title="Do clones expand after treatment, and more in responders?",
        asks=["do these clones expand after treatment", "clonal expansion between timepoints", "TCR expansion responders vs non-responders"],
        code='''\
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
''',
    ),
    dict(
        id="clone_diversity",
        title="Repertoire diversity and STARTRAC indices",
        asks=["is the repertoire more clonal after treatment", "clonality", "STARTRAC expansion index"],
        code='''\
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
ct = arsc.clones.clone_table(obs, "clone_id", sample_cols=["patient", "timepoint"])
div = arsc.clones.clone_diversity(ct, sample_cols=["patient", "timepoint"], floor=50)   # rarefied to a common depth
print(arsc.stats.paired_test(div, "clonality", t0="pre", t1="post", patient="patient", time="timepoint").T)
cluster_idx, pairwise = arsc.clones.startrac(obs, clone_col="clone_id", state_col="cell_type",
                                             group_cols="patient", min_cells=10)   # per patient
print(cluster_idx.head())
''',
    ),
    dict(
        id="power",
        title="Before testing: how many patients, and what effect can I detect?",
        asks=["is this comparison powered", "how many samples do I have", "minimum detectable effect"],
        code='''\
import arsc

ds = arsc.datasets.toy()
obs = ds.obs
cd8 = obs[obs["cell_type"] == "CD8 T"]
print(cd8.groupby(["response", "timepoint"], observed=True)["patient"].nunique())   # patients per cell
print(cd8.groupby("sample").size().describe())                                     # cells per sample
print(arsc.stats.mde(6))       # paired, 6 patients: effect in SD units detectable at 80% power
print(arsc.stats.mde(3, 3))    # two groups of 3: infeasible (minimum exact p = 0.1)
''',
    ),
    dict(
        id="handoff",
        title="Hand results to scanpy, R or a CSV",
        asks=["give me an AnnData", "export for plotting in R", "save the table"],
        code='''\
import os, tempfile
import arsc

ds = arsc.datasets.toy()
cd8 = (ds.obs["cell_type"] == "CD8 T").to_numpy()
adata = ds.to_anndata(cells=cd8, genes=["CD8A", "GZMK", "PDCD1"])      # in-memory AnnData for scanpy
out = tempfile.mkdtemp()
ds.write_h5ad(os.path.join(out, "cd8.h5ad"), cells=cd8)              # streamed, any size
arsc.write_table(ds.obs, os.path.join(out, "obs.csv"))                # tidy table for ggplot
print(adata)
''',
    ),
]
