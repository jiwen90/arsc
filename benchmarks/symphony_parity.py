"""arsc.symphony vs R symphony 0.1.1 on the package's vignette data (reference 3pv1 + 3pv2, query 5p).

Run benchmarks/r_reference_symphony.R first (writes $ARSC_REFS/symphony_ref); then
    python benchmarks/symphony_parity.py
prints the SYMPHONY_PARITY.md table and writes symphony_parity.csv next to the R outputs.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io

import arsc  # noqa: E402
from arsc import symphony as sy  # noqa: E402

D = os.path.join((os.environ.get("ARSC_REFS") or os.environ.get("ARSC_SCRATCH", ".")), "symphony_ref")
r = lambda f: np.loadtxt(os.path.join(D, f), delimiter=",", ndmin=2)  # noqa: E731
rows = []


def rec(stage, what, value, note=""):
    rows.append(dict(stage=stage, quantity=what, value=value, note=note))
    print(f"{stage:10s} {what:55s} {value}")


def maxabs(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def col_r(A, B):
    """per-column |Pearson r| (signs of PCs are arbitrary)."""
    return np.array([abs(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(A.shape[1])])


# ---------------- 1. internals on random inputs
Y, Zc = r("u_Y.csv"), r("u_Zq_cos.csv")
rec("internal", "soft_cluster max abs diff", maxabs(sy._soft_cluster(Y.T, Zc.T, 0.1), r("u_softcluster.csv")))
Rr, Zr = r("u_Rr.csv"), r("u_Zr.csv")
C = Rr @ Zr.T
rec("internal", "Nr, C max abs diff", max(maxabs(Rr.sum(1), r("u_Nr.csv").ravel()), maxabs(C, r("u_C.csv"))))
rec("internal", "centroids max abs diff", maxabs(sy._l2_rows(C).T, r("u_centroids.csv")))
Zq, Xq = r("u_Zq.csv"), r("u_Xq.csv")
pats, inv = np.unique(Xq.T, axis=0, return_inverse=True)
Zcorr = sy._moe_correct_ref(Zq.T, pats, inv.ravel(), r("u_softcluster.csv"), Rr.sum(1), C)
rec("internal", "moe_correct_ref max abs diff", maxabs(Zcorr.T, r("u_moe.csv")))
A = r("clip_A.csv")
rec("internal", "scaleDataWithStats (clip 1) max abs diff",
    maxabs(sy._scale_with_stats(A.T, A.mean(1), A.std(1, ddof=1), 1.0).T, r("clip_S.csv")))
Zs, Y0 = r("sk_Z.csv"), r("sk_Y0.csv")
Yk = Y0.T.copy()
for _ in range(4):
    Yk = sy._l2_rows(Yk); Rk = sy._soft_cluster(Yk, Zs.T, 0.1); Yk = Rk @ Zs.T
rec("internal", "soft_kmeans (4 rounds, same init) R max abs diff", maxabs(Rk, r("sk_R.csv")))
mmref = sy.Reference(pd.Index([]), np.zeros(0), np.zeros(0), np.zeros((0, 4)), r("mm_Zorig.csv").T,
                     r("mm_Zorig.csv").T, r("mm_R.csv"), np.zeros((3, 4)), np.zeros(3), np.zeros((3, 4)),
                     pd.DataFrame(index=range(60)))
mmq = sy.QueryMap(np.zeros((10, 4)), r("mm_Zq.csv").T, r("mm_Rq.csv"), pd.DataFrame(index=range(10)))
rec("internal", "calcPerCellMappingMetric max abs diff", maxabs(sy.mapping_score(mmref, mmq), r("mm_maha.csv").ravel()))

# ---------------- data
X = scipy.io.mmread(os.path.join(D, "pbmcs_exprs.mtx")).tocsr().T.tocsr()  # cells x genes, log-normalized
genes = pd.Index(open(os.path.join(D, "pbmcs_genes.txt")).read().split())
meta = pd.read_csv(os.path.join(D, "pbmcs_meta.csv"), index_col=0)
isref = meta["donor"].isin(["3pv1", "3pv2"]).to_numpy()
Xr, Xqm, mr, mq = X[isref], X[~isref], meta[isref], meta[~isref]
rvg = open(os.path.join(D, "ref_vargenes.txt")).read().split()
rms = r("ref_meansd.csv")
Rl, RZo, RZc, RR = r("ref_loadings.csv"), r("ref_Zorig.csv").T, r("ref_Zcorr.csv").T, r("ref_R.csv")
Rq_Z, Rq_pca, Rq_R = r("q_Z.csv").T, r("q_Zpca.csv").T, r("q_R.csv")
rknn = pd.read_csv(os.path.join(D, "q_knn.csv"))

# ---------------- 2. mapping given R's reference (deterministic steps only)
ref_r = sy.reference_from_harmony(RZo, RZc, RR, rvg, rms[:, 0], rms[:, 1], Rl, mr, umap=False)
rec("given-R", "centroids vs R max abs diff", maxabs(ref_r.centroids, r("ref_centroids.csv").T))
rec("given-R", "Nr / C vs R max abs diff", max(maxabs(ref_r.Nr, r("ref_Nr.csv").ravel()), maxabs(ref_r.C, r("ref_C.csv"))))
q_r = sy.map_query(Xqm, genes, mq, ref_r, vars="donor", normalize=False, umap=False)
rec("given-R", "query Zq_pca max abs diff", maxabs(q_r.Z_pca, Rq_pca))
rec("given-R", "query R max abs diff", maxabs(q_r.R, Rq_R))
rec("given-R", "query Z (corrected) max abs diff", maxabs(q_r.Z, Rq_Z))
lab_r = sy.transfer_labels(ref_r, q_r, "cell_type", k=5)
rec("given-R", "kNN label agreement with R knnPredict", float((lab_r["label"].to_numpy() == rknn["pred"].to_numpy()).mean()))
rec("given-R", "kNN prob max abs diff", maxabs(lab_r["prob"], rknn["prob"]))
rec("given-R", "mapping score max abs diff", maxabs(sy.mapping_score(ref_r, q_r), r("q_maha.csv").ravel()))

# ---------------- 3. end to end in arsc
t0 = time.time()
ref = sy.build_reference(Xr, genes, mr, vars="donor", normalize=False, topn=2000, d=20, K=100, umap=True)
q = sy.map_query(Xqm, genes, mq, ref, vars="donor", normalize=False, umap=True)
lab = sy.transfer_labels(ref, q, "cell_type", k=5)
rec("end-to-end", "arsc build+map+transfer seconds", round(time.time() - t0, 2), "R: 8.3 s incl. uwot")
rec("end-to-end", "variable genes identical to R (same order)", bool(list(ref.genes) == rvg),
    f"{len(set(ref.genes) & set(rvg))}/{len(rvg)} shared")
rec("end-to-end", "gene mean / sd max abs diff", max(maxabs(ref.mean, rms[:, 0]), maxabs(ref.sd, rms[:, 1])))
rec("end-to-end", "loadings per-PC |r| min", round(float(col_r(ref.loadings, Rl).min()), 6))
rec("end-to-end", "Z_orig per-PC |r| min", round(float(col_r(ref.Z_orig, RZo).min()), 6))
sign = np.sign([np.corrcoef(ref.Z_orig[:, j], RZo[:, j])[0, 1] for j in range(20)])
czr = col_r(ref.Z_corr, RZc)
rec("end-to-end", "ref Z_corr per-dim r (median, min)", (round(float(np.median(czr)), 4), round(float(czr.min()), 4)),
    "Harmony stochastic (k-means init, block order)")
cq = col_r(q.Z, Rq_Z)
rec("end-to-end", "query Z per-dim r (median, min)", (round(float(np.median(cq)), 4), round(float(cq.min()), 4)))
truth = mq["cell_type"].to_numpy()
rec("end-to-end", "label agreement arsc vs R", round(float((lab["label"].to_numpy() == rknn["pred"].to_numpy()).mean()), 4))
rec("end-to-end", "accuracy vs annotated cell_type: arsc", round(float((lab["label"].to_numpy() == truth).mean()), 4))
rec("end-to-end", "accuracy vs annotated cell_type: R", round(float((rknn["pred"].to_numpy() == truth).mean()), 4))
rec("end-to-end", "prob Pearson r arsc vs R", round(float(np.corrcoef(lab["prob"], rknn["prob"])[0, 1]), 4))
ms = sy.mapping_score(ref, q)
rec("end-to-end", "mapping score Spearman arsc vs R",
    round(float(pd.Series(ms).corr(pd.Series(r("q_maha.csv").ravel()), method="spearman")), 4))
# UMAP: the layouts differ (umap-learn vs uwot); compare the label read off the query's 10 nearest reference
# cells in each package's own UMAP
from arsc.neighbors import knn  # noqa: E402
ru, rqu = r("ref_umap.csv"), r("q_umap.csv")
def umap_label(refxy, qxy, y):
    idx, _ = knn(refxy, 10, query=qxy)
    return pd.DataFrame(y[idx]).mode(axis=1)[0].to_numpy()
y = mr["cell_type"].to_numpy()
la, lr = umap_label(ref.umap, q.umap, y), umap_label(ru, rqu, y)
rec("end-to-end", "UMAP placement: 10-NN label on own UMAP, arsc vs R agreement", round(float((la == lr).mean()), 4))
rec("end-to-end", "UMAP placement: accuracy arsc / R", (round(float((la == truth).mean()), 4), round(float((lr == truth).mean()), 4)))
pd.DataFrame(rows).to_csv(os.path.join(D, "symphony_parity.csv"), index=False)
