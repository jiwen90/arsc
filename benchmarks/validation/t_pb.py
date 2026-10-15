import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from refutil import *
import arsc
st = arsc.open(EXP_GD); feats = st.features(); C = st.X()
for _ in range(2):
    t=time.perf_counter(); pb = arsc.pseudobulk(C, st.obs, ["sample","cell_type"], features=feats); print("pb", time.perf_counter()-t, pb.X.shape)
dim = [int(x) for x in open(REF_GD/"pseudobulk_dim.txt").read().split()]
R = read_dgc(REF_GD/"pseudobulk", dim).toarray()
rg = open(REF_GD/"pseudobulk_groups.txt").read().splitlines(); rf = open(REF_GD/"pseudobulk_features.txt").read().splitlines()
print("n groups py/R", pb.X.shape[0], len(rg)); 
mine = pb.to_frame()
common = [g for g in rg if g in mine.columns]; print("matched group names", len(common))
missing = [g for g in rg if g not in mine.columns][:5]; print("unmatched examples", missing)
Rdf = pd.DataFrame(R, index=rf, columns=rg)
d = np.abs(mine.loc[rf, common].values - Rdf[common].values); print("max abs diff", d.max(), "total counts", Rdf.values.sum(), mine.values.sum())
pb10 = arsc.pseudobulk(C, st.obs, ["sample","cell_type"], features=feats, min_cells=10); print("min_cells=10 kept", pb10.X.shape[0], "dropped", len(pb10.dropped))
# CLR
A = st.X("TRIMMEDCITE")
t=time.perf_counter(); Z = arsc.pp.clr(A, margin=2); print("clr", time.perf_counter()-t)
Rc = read_dense(REF_GD/"adt_clr").values  # features x cells
print("CLR max abs diff", np.abs(Z.toarray().T - Rc).max())
