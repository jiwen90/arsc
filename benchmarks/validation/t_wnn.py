import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "E:/autoresearch/vendor/arsc/benchmarks")
from refutil import *
import arsc
st = arsc.open(EXP_PBMC)
er = st.embedding("harmony_rna", range(30)); ea = st.embedding("harmony_adt", range(15))
t=time.perf_counter(); w = arsc.wnn([er, ea]); print("wnn", time.perf_counter()-t, flush=True)
rw = st.obs["RNA.weight"].values
print("RNA.weight r", np.corrcoef(w.weights[:,0], rw)[0,1], "mean|diff|", np.abs(w.weights[:,0]-rw).mean(), "median|diff|", np.median(np.abs(w.weights[:,0]-rw)))
S = read_dense(REF_PBMC/"stored_weighted_nn_idx").values.astype(np.int64) - 1
ov = np.array([len(set(a)&set(b)) for a,b in zip(S, w.idx)])/S.shape[1]
print("weighted-nn overlap vs stored Seurat: mean", ov.mean(), "median", np.median(ov))
np.save(SCRATCH/"wnn_idx.npy", w.idx); np.save(SCRATCH/"wnn_weights.npy", w.weights)
G = st.graph("wsnn").tocsr()
t=time.perf_counter(); lab = arsc.cluster(w.wsnn, "leiden", resolution=0.2); print("leiden", time.perf_counter()-t, lab.max()+1)
labS = arsc.cluster(G, "leiden", resolution=0.2)
from sklearn.metrics import adjusted_rand_score as ari
ref = st.obs["wsnn_res.0.2"].astype(str).values
print("ARI leiden(our wsnn) vs Seurat wsnn_res.0.2:", ari(ref, lab), " leiden(Seurat wsnn) vs wsnn_res.0.2:", ari(ref, labS), " ours vs leiden(Seurat wsnn):", ari(labS, lab))
