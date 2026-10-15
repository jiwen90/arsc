import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from refutil import *
import arsc
st = arsc.open(EXP_GD); feats = st.features(); C = st.X()
sigs = json.load(open(REF_GD/"signatures.json"))
for _ in range(2):
    t=time.perf_counter(); u = arsc.score_ucell(C, feats, sigs, cells=st.cells); print("ucell", time.perf_counter()-t)
ur = pd.read_csv(REF_GD/"ucell.csv", index_col=0)
for c in ur.columns:
    d = np.abs(u[c].values - ur.loc[u.index, c].values); print(f"{c:25s} max abs diff {d.max():.3e}  r {np.corrcoef(u[c], ur.loc[u.index,c])[0,1]:.12f}")
L = arsc.pp.log_normalize(C)
pos = {k: sigs[k] for k in ["cytotoxic","naive","exhaustion","isg","gd"]}
for _ in range(2):
    t=time.perf_counter(); a = arsc.add_module_score(L, feats, list(pos.values()), name="AMS", seed=1, cells=st.cells); print("AMS", time.perf_counter()-t)
ar = pd.read_csv(REF_GD/"addmodulescore.csv", index_col=0)
for c in ar.columns:
    d = np.abs(a[c].values - ar.loc[a.index, c].values); print(f"{c:6s} max abs diff {d.max():.3e}")
