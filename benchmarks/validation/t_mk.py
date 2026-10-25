import sys, time, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from refutil import *
import arsc
st = arsc.open(EXP_GD); feats = st.features()
L = arsc.pp.log_normalize(st.X())
lab = st.obs["cell_type"]
import os
compat = os.environ.get("COMPAT")=="1"
for _ in range(2):
    t=time.perf_counter(); res = arsc.wilcoxauc(L, lab, features=feats); print("wilcoxauc", time.perf_counter()-t)
ref = pd.read_csv(REF_GD/"presto_cell_type.csv.gz")
m = res.merge(ref, on=["feature","group"], suffixes=("","_r"))
print("rows", len(res), len(ref), len(m))
for c in ["avgExpr","logFC","statistic","auc","pval","padj","pct_in","pct_out"]:
    a, b = m[c].values, m[c+"_r"].values
    ok = np.isfinite(a)&np.isfinite(b)
    d = np.abs(a-b)[ok]; rel = d/np.maximum(np.abs(b[ok]),1e-300)
    print(f"{c:10s} max abs {d.max():.3e}  max rel {rel.max():.3e}  nan mismatch {(np.isfinite(a)!=np.isfinite(b)).sum()}")
# p-value relative diff where p>1e-300
ok = (m.pval_r>1e-300)&(m.pval>0)
print("log10 p max abs diff (p>1e-300):", np.abs(np.log10(m.pval[ok])-np.log10(m.pval_r[ok])).max())
