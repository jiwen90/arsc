"""Measured agreement of arsc.stats rank / stratified / rarefaction helpers with R (numbers for STATS_PARITY.md)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arsc import stats as st

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["ARSC_SCRATCH"]) / "stats_resampling"
rd = lambda f: pd.read_csv(d / f, float_precision="round_trip")  # noqa: E731

cs, vv, R = rd("hl_cases.csv"), rd("hl_values.csv"), rd("R_hl.csv").set_index("case")
dif = {"exact": {"p": 0, "est": 0, "ci": 0, "cl": 0, "n": 0}, "normal": {"p": 0, "est": 0, "ci": 0, "cl": 0, "n": 0}}
for _, k in cs.iterrows():
    x = vv[(vv.case == k.case) & (vv.grp == "x")].sort_values("i").value.to_numpy()
    y = vv[(vv.case == k.case) & (vv.grp == "y")].sort_values("i").value.to_numpy()
    kw = dict(alternative=k.alternative, conf_level=k.conf_level, correct=bool(k.correct))
    r = (st.hodges_lehmann(x, **kw) if k.kind == "one" else
         st.hodges_lehmann(x, y, paired=True, **kw) if k.kind == "paired" else st.hodges_lehmann(x, y, **kw))
    ref = R.loc[k.case]
    m = dif[r["method"]]
    m["n"] += 1
    m["p"] = max(m["p"], abs(r["p"] - ref.p))
    m["est"] = max(m["est"], abs(r["estimate"] - ref.estimate))
    m["cl"] = max(m["cl"], abs(r["conf_level"] - ref.conf_level))
    for a, b in ((r["ci_lo"], ref.ci_lo), (r["ci_hi"], ref.ci_hi)):
        if np.isfinite(b):
            m["ci"] = max(m["ci"], abs(a - b))
print("hodges_lehmann vs wilcox.test:", dif)

T, R = rd("mh_tables.csv"), rd("R_mh.csv").set_index("case")
mx = {k: 0.0 for k in ("or_mh", "ci_lo", "ci_hi", "statistic", "p")}
for case, t in T.groupby("case"):
    r = st.mantel_haenszel(t.a, t.b, t.c, t.d, correct=bool(t.correct.iloc[0]))
    for k in mx:
        mx[k] = max(mx[k], abs(r[k] - R.loc[case, k]) / abs(R.loc[case, k]))
print("mantel_haenszel vs mantelhaen.test, max rel diff:", {k: f"{v:.1e}" for k, v in mx.items()}, "cases", T.case.nunique())

C, R = rd("rarefy_counts.csv"), rd("R_rarefy.csv").set_index("case")
e1 = e2 = 0.0
for case, t in C.groupby("case"):
    e, s = st.rarefy_richness(t.sort_values("clone")["size"].to_numpy(), int(t.depth.iloc[0]), se=True)
    e1 = max(e1, abs(e - R.loc[case, "S"]) / R.loc[case, "S"]); e2 = max(e2, abs(s - R.loc[case, "se"]))
print(f"rarefy_richness vs vegan::rarefy: S max rel {e1:.1e}, se max abs {e2:.1e}, cases {C.case.nunique()}")

V, R = rd("icc_values.csv"), rd("R_icc.csv").set_index("case")
dd = []
for case, t in V.groupby("case"):
    r = st.icc_oneway(t.y, t.group, strata=t.stratum if t.stratum.nunique() > 1 else None)
    if r["icc"] > 0 and not R.loc[case, "singular"]:
        dd.append(abs(r["icc"] - R.loc[case, "icc_reml"]))
print(f"icc_oneway vs lme4 REML (balanced, positive): n {len(dd)}, max abs {max(dd):.1e}, median {np.median(dd):.1e}")

V, R = rd("fl_values.csv"), rd("R_fl.csv").set_index("case")
db = dt = dp = 0.0
for case, t in V.groupby("case"):
    zs = [c for c in ("z0", "z1") if t[c].notna().any()]
    r = st.freedman_lane(t.y.to_numpy(), t.x.to_numpy(), covariates=t[zs].to_numpy() if zs else None, B=200)
    db = max(db, abs(r["beta"] - R.loc[case, "beta"])); dt = max(dt, abs(r["t"] - R.loc[case, "t"]))
    dp = max(dp, abs(r["p_param"] - R.loc[case, "p"]))
print(f"freedman_lane observed beta/t/p vs lm: {db:.1e} / {dt:.1e} / {dp:.1e}")
