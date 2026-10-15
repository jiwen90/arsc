import sys, time, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from refutil import *
import arsc
from arsc import preprocessing as pp
ds = arsc.read_seurat_export(EXP_GD)
C = ds.X("RNA","counts"); feats = ds["RNA"].features
t=time.perf_counter(); L = pp.log_normalize(C); print("lognorm", time.perf_counter()-t)
t=time.perf_counter(); L = pp.log_normalize(C); print("lognorm", time.perf_counter()-t)
blk = read_dense(REF_GD/"lognorm_block")
mine = pd.DataFrame(L[:2000][:, feats.get_indexer(blk.index)].toarray().T, index=blk.index)
print("lognorm max abs diff", np.abs(mine.values-blk.values).max())
t=time.perf_counter(); h = pp.hvg_vst(C, features=feats); print("hvg", time.perf_counter()-t)
r = pd.read_csv(REF_GD/"hvf.csv", index_col=0)
for a,b in [("mean","vst.mean"),("variance","vst.variance"),("variance_expected","vst.variance.expected"),("variance_standardized","vst.variance.standardized")]:
    x=h[a].values; y=r.loc[h.index,b].values
    rel = np.abs(x-y)/np.maximum(np.abs(y),1e-300)
    print(a, "max rel diff", np.nanmax(rel[y>0]) if (y>0).any() else 0, "median", np.nanmedian(rel[y>0]))
rv = open(REF_GD/"variable_features.txt").read().split()
mv = list(h.sort_values("rank").index[:2000])
print("HVG overlap", len(set(rv)&set(mv))/2000, "same order", rv==mv)
d=[(i,a,b) for i,(a,b) in enumerate(zip(rv,mv)) if a!=b]
print(len(d), d[:6])
for i,a,b in d[:4]: print(a, r.loc[a,"vst.variance.standardized"], h.loc[a,"variance_standardized"], b, r.loc[b,"vst.variance.standardized"], h.loc[b,"variance_standardized"])
