import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, "E:/autoresearch/vendor/arsc/benchmarks")
from refutil import *
import arsc
from arsc.harmony import run_harmony, Harmony
ds = arsc.read_seurat_export(EXP_GD, assays=[], graphs=False)
pcs = read_dense(REF_GD/"pca_embeddings").values
meta = ds.obs
def cmp(a, b, tag):
    r = [np.corrcoef(a[:,k], b[:,k])[0,1] for k in range(a.shape[1])]
    print(f"{tag}: max|diff| {np.abs(a-b).max():.3e}  rel {np.abs(a-b).max()/np.abs(b).max():.3e}  min per-dim r {min(r):.10f}")
tag="bs005"
Y0 = read_dense(REF_GD/f"harmony_{tag}_Y0").values  # d x K
Zr = read_dense(REF_GD/f"harmony_{tag}_Z").values
Rr = read_dense(REF_GD/f"harmony_{tag}_R").values  # K x N
tr = json.load(open(REF_GD/f"harmony_{tag}_trace.json"))
# (a) init objective from R's centroids
res0 = run_harmony(pcs, meta, "n10x_lane_name", init_centroids=Y0.T, max_iter=0, return_object=True)
print("init objective  py", res0.objective_harmony[0], " R", tr["obj0"][0])
# (b) exact correction from R's final R
res = run_harmony(pcs, meta, "n10x_lane_name", init_centroids=Y0.T, max_iter=1, return_object=True)
from arsc.harmony import _r_factor_levels
c, lev = _r_factor_levels(meta["n10x_lane_name"])
h = Harmony(pcs, c[:,None], [len(lev)], theta=np.full(len(lev),2.0), sigma=np.full(100,0.1), lambda_vec=np.r_[0, np.ones(len(lev))], K=100)
h.R = np.ascontiguousarray(Rr.T); h.moe_correct_ridge()
cmp(h.Z_corr, Zr, "ridge correction given R's final R")
# (c) full run from R's centroids
for seed in range(3):
    t=time.perf_counter()
    res = run_harmony(pcs, meta, "n10x_lane_name", init_centroids=Y0.T, return_object=True, seed=seed)
    print("time", round(time.perf_counter()-t,2))
    cmp(res.Z_corr, Zr, f"full run from R init, seed {seed}")
    print(" py obj", np.round(res.objective_harmony,3), res.kmeans_rounds)
print(" R  obj", np.round(tr["objective_harmony"],3), tr["kmeans_rounds"])
t=time.perf_counter(); Zd = run_harmony(pcs, meta, "n10x_lane_name", seed=1); print("default time", time.perf_counter()-t)
cmp(Zd, read_dense(REF_GD/"harmony_default").values, "default (independent init)")
