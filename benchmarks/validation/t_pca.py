import sys, time, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from refutil import *
import arsc
from arsc import preprocessing as pp
ds = arsc.read_seurat_export(EXP_GD)
C = ds.X("RNA","counts"); feats = ds["RNA"].features
L = pp.log_normalize(C)
rv = open(REF_GD/"variable_features.txt").read().split()
idx = feats.get_indexer(rv)
t=time.perf_counter(); Z,mu,sd = pp.scale(L[:, idx]); print("scale", time.perf_counter()-t)
for m in ["gram","arpack","randomized"]:
    t=time.perf_counter(); P = pp.pca(Z, 30, method=m); print(m, "pca", time.perf_counter()-t)
    R = read_dense(REF_GD/"pca_embeddings").values
    rs = np.loadtxt(REF_GD/"pca_stdev.txt")
    cors = [abs(np.corrcoef(P.embedding[:,k], R[:,k])[0,1]) for k in range(30)]
    print(" per-PC |r| min", min(cors), "first 10", np.round(cors[:10],8))
    print(" stdev max rel diff", np.max(np.abs(P.stdev-rs)/rs))
    # principal angles between 30-dim subspaces
    qa,_=np.linalg.qr(P.embedding); qb,_=np.linalg.qr(R)
    s=np.linalg.svd(qa.T@qb, compute_uv=False); print(" subspace cos min", s.min(), "mean", s.mean())
    for kk in (10,20):
        s=np.linalg.svd(np.linalg.qr(P.embedding[:,:kk])[0].T@np.linalg.qr(R[:,:kk])[0], compute_uv=False); print(f" top{kk} subspace cos min", s.min())
