"""Inputs of the R parity of arsc.de / arsc.stats (STATS_PARITY.md, tests/test_stats_de.py), built in memory from the
shared input store ($ARSC_STORE_GD, the 38,440-cell gamma-delta T-cell store; benchmarks/refutil.py).

    tp    gdT pseudobulk per patient x timepoint (Baseline vs Follow Up 1, >= 20 cells)
    ct    gd vs cd8_tem pseudobulk in matched patient x timepoint samples (>= 20 cells each)
    comp  cell-type composition per patient x timepoint (parent >= 50 cells)

Nothing here is stored: the tests rebuild these tables, and ``python benchmarks/stats_parity_inputs.py <dir>`` writes
them only for benchmarks/r_reference_stats_parity.R (benchmarks/regen_refs.py stats_parity deletes them afterwards).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refutil import STORE_GD  # noqa: E402


def _open(store=None):
    import arsc
    st = arsc.open(store or STORE_GD)
    obs = st.obs.copy()
    for c in ("patient_alias", "timepoint", "cell_type"):
        obs[c] = obs[c].astype(str)
    return st, obs


def tp(store=None):
    """(counts genes x samples, coldata): gdT cells, Baseline vs Follow Up 1, patient x timepoint."""
    import arsc
    st, obs = _open(store)
    m = (obs.cell_type == "gd") & obs.timepoint.isin(["Baseline", "Follow Up 1"])
    rows = np.flatnonzero(m.values)
    pb = arsc.pseudobulk(st.X()[rows], obs[m].reset_index(drop=True), ["patient_alias", "timepoint"],
                         features=st.features(), min_cells=20)
    cnt = pb.to_frame()
    cd = pb.groups.copy()
    cd.index = cnt.columns
    cnt = cnt.loc[cnt.sum(axis=1) > 0].round().astype(int)
    return _tidy(cnt, cd)


def ct(store=None):
    """(counts, coldata): gd vs cd8_tem in the patient x timepoint samples that have >= 20 cells of both."""
    import arsc
    st, obs = _open(store)
    obs["pt"] = obs.patient_alias + "|" + obs.timepoint
    m = obs.cell_type.isin(["gd", "cd8_tem"])
    rows = np.flatnonzero(m.values)
    pb = arsc.pseudobulk(st.X()[rows], obs[m].reset_index(drop=True), ["pt", "cell_type"], features=st.features(),
                         min_cells=20)
    cnt = pb.to_frame()
    cd = pb.groups.copy()
    cd.index = cnt.columns
    both = cd.groupby("pt").cell_type.nunique()
    sel = cd.pt.isin(both[both == 2].index).values
    cnt, cd = cnt.loc[:, sel], cd[sel].copy()
    cd["patient_alias"] = cd.pt.str.split("|").str[0]
    cnt = cnt.loc[cnt.sum(axis=1) > 0].round().astype(int)
    return _tidy(cnt, cd)


def comp(store=None) -> pd.DataFrame:
    """Composition of every cell type per patient x timepoint (parent >= 50 cells)."""
    from arsc import stats
    _, obs = _open(store)
    out = stats.composition(obs, "cell_type", ["patient_alias", "timepoint"], min_parent=50)
    return pd.DataFrame({c: (out[c].astype(str) if out[c].dtype.name in ("category", "object") else out[c])
                         for c in out.columns})


def _tidy(cnt, cd):
    """Same types as a write/read round trip of the tables (string labels and sample names)."""
    cnt.columns = cnt.columns.astype(str)
    cd.index = cd.index.astype(str)
    for c in cd.columns:
        if cd[c].dtype.name in ("category", "object"):
            cd[c] = cd[c].astype(str)
    return cnt, cd


def write(out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for tag, f in (("tp", tp), ("ct", ct)):
        cnt, cd = f()
        cnt.to_csv(out / f"{tag}_counts.csv.gz")
        cd.to_csv(out / f"{tag}_coldata.csv")
    comp().to_csv(out / "comp.csv", index=False)
    return out


if __name__ == "__main__":
    print("wrote", write(sys.argv[1]))
