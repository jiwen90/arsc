"""TCR / clonal analysis methods (port of tools/R/ar_methods.R + the clone helpers of tools/R/ar.R).

Everything takes a plain cell-level metadata ``pandas.DataFrame`` (one row per cell), exactly like the R
functions, so it works on any lineage.  The patient is the unit of inference: every method also returns a
per-patient summary.  No plotting (figures stay in R).

    cyclone()                 clone trajectories (Cancer Cell 2024 algorithm; exact port, ARI 1.0 vs published)
    cyclone_patient_table()   clones per trajectory x patient (wide)
    cyclone_states()          per-patient state composition of trajectory clones over time
    read_10x_clonotypes()     healthy-donor clonotypes.csv -> one row per cell
    clone_expansion_null()    per-clone expansion vs a healthy-donor null (downsampling, Fisher + signed chi)
    startrac()                STARTRAC-style expansion / migration / transition indices (per patient)
    clone_table(), clone_diversity(), clone_dynamics()
    clone_tissue_matrix(), clone_tissue_entropy()
    ari(), p_adjust_bh(), fisher_exact_2x2()
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaln

__all__ = [
    "Cyclone", "cyclone", "cyclone_patient_table", "cyclone_states", "read_10x_clonotypes",
    "clone_expansion_null", "startrac", "clone_table", "clone_diversity", "clone_dynamics",
    "clone_tissue_matrix", "clone_tissue_entropy", "ari", "p_adjust_bh", "fisher_exact_2x2", "pam",
]


# ============================================================================================ helpers
def _fmt_tp(v) -> str:
    """as.character() of a timepoint label (3.0 -> "3")."""
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _as_str(s: pd.Series) -> pd.Series:
    """Vectorised string labels (NaN preserved), integral floats printed without ".0"."""
    codes, uniq = pd.factorize(s, sort=False)
    lab = np.array([_fmt_tp(u) for u in uniq], dtype=object)
    out = np.where(codes >= 0, lab[np.maximum(codes, 0)], None)
    return pd.Series(out, index=s.index, dtype=object)


def _tp_levels(x: pd.Series, timepoints=None) -> list[str]:
    if timepoints is not None:
        return [_fmt_tp(t) for t in timepoints]
    if isinstance(x.dtype, pd.CategoricalDtype):
        used = set(x.dropna().unique())
        return [_fmt_tp(c) for c in x.cat.categories if c in used]
    u = _as_str(x.dropna()).unique().tolist()
    try:
        return sorted(u, key=float)
    except ValueError:
        return sorted(u)


def ari(a, b) -> float:
    """Adjusted Rand index between two labelings."""
    tab = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy().astype(float)
    n = tab.sum()
    c2 = lambda x: float((x * (x - 1) / 2).sum())  # noqa: E731
    s, sa, sb = c2(tab), c2(tab.sum(1)), c2(tab.sum(0))
    e = sa * sb / (n * (n - 1) / 2)
    return (s - e) / ((sa + sb) / 2 - e)


def p_adjust_bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjustment (== R p.adjust(p, "BH"); NaN preserved)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    o = np.argsort(-q, kind="stable")
    ranked = np.minimum.accumulate((q[o] * n / (n - np.arange(n)))[::1])
    adj = np.empty(n)
    adj[o] = np.minimum(1.0, ranked)
    out[ok] = adj
    return out


def _lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def fisher_exact_2x2(a, b, N1, N2) -> np.ndarray:
    """Two-sided Fisher exact p (R fisher.test, relErr 1+1e-7) for tables [[a, N1-a], [b, N2-b]].

    ``a``/``b`` arrays (clone counts at the two timepoints), ``N1``/``N2`` scalars or arrays.  Unique tables
    are evaluated once.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n1 = np.broadcast_to(np.asarray(N1, dtype=np.int64), a.shape)
    n2 = np.broadcast_to(np.asarray(N2, dtype=np.int64), a.shape)
    key = np.stack([a, b, n1, n2], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    pu = np.empty(len(uk))
    for i, (x, y, m1, m2) in enumerate(uk):
        N = m1 + m2
        s = x + y                                   # successes (column total)
        lo, hi = max(0, m1 - (N - s)), min(m1, s)
        k = np.arange(lo, hi + 1)
        lp = _lchoose(s, k) + _lchoose(N - s, m1 - k) - _lchoose(N, m1)
        d = np.exp(lp)
        dobs = d[x - lo]
        pu[i] = min(1.0, float(d[d <= dobs * (1 + 1e-7)].sum()))
    return pu[inv]


def pam(D: np.ndarray, k: int, max_iter: int = 100):
    """Partitioning around medoids (BUILD + SWAP) on a dissimilarity matrix (cluster::pam equivalent).

    Returns (labels 0..k-1 numbered in medoid order, medoid indices).
    """
    n = D.shape[0]
    med = [int(np.argmin(D.sum(1)))]
    dnear = D[med[0]].copy()
    while len(med) < k:
        gain = np.maximum(dnear[None, :] - D, 0).sum(1)
        gain[med] = -1
        m = int(np.argmax(gain))
        med.append(m)
        dnear = np.minimum(dnear, D[m])
    for _ in range(max_iter):
        Dm = D[med]                                  # k x n
        order = np.argsort(Dm, axis=0)
        near = order[0]
        d1 = Dm[near, np.arange(n)]
        d2 = Dm[order[1], np.arange(n)] if k > 1 else np.full(n, np.inf)
        cur = d1.sum()
        best, bm, bh = 0.0, -1, -1
        nonmed = np.setdiff1d(np.arange(n), med)
        for mi in range(k):
            wo = np.where(near == mi, d2, d1)        # nearest distance without medoid mi
            cost = np.minimum(D[nonmed], wo[None, :]).sum(1)
            j = int(np.argmin(cost))
            if cost[j] - cur < best - 1e-12:
                best, bm, bh = cost[j] - cur, mi, int(nonmed[j])
        if bm < 0:
            break
        med[bm] = bh
    lab = np.argmin(D[med], axis=0)
    return lab, np.array(med)


# ============================================================================================ Cyclone
@dataclass
class Cyclone:
    """Result of :func:`cyclone` (mirrors the R ``ar_cyclone`` list)."""
    clonestats: pd.DataFrame      # clone x patient x timepoint long table with perc, deltas, k
    assignment: pd.DataFrame      # one row per clone kept by the filter (k may be "Other" after outlier rule)
    counts: pd.DataFrame          # patient x trajectory: n_clones and summed perc per timepoint
    mat: pd.DataFrame             # timepoint x clone matrix of perc (missing -> 0)
    mat_cor: pd.DataFrame         # clone x clone Pearson correlation of the standardized matrix
    linkage: np.ndarray | None    # scipy linkage (ward == R ward.D2), None for pam/absolute
    cell_k: pd.Series             # trajectory per row of ``meta`` ("Other" unclustered, None without clone)
    params: dict = field(default_factory=dict)

    def patient_table(self, value="n_clones", at=None) -> pd.DataFrame:
        return cyclone_patient_table(self, value, at)

    def __repr__(self):
        a = self.assignment
        return (f"<Cyclone> {int((a.k != 'Other').sum())} clustered clones, "
                f"{a[self.params['patient_col']].nunique()} patients, timepoints {'/'.join(self.params['timepoints'])}")


def _patient_groups(meta, patient_col, group_cols) -> pd.DataFrame:
    pg = meta[[patient_col, *group_cols]].copy()
    pg[patient_col] = pg[patient_col].astype(str)
    pg = pg.drop_duplicates(patient_col).rename(columns={patient_col: "patient"})
    return pg


def _clonestats(meta, clone_col, patient_col, time_col, lv, min_c, min_fc, n_max_min, reg,
                drop_incomplete, min_cells_sample, verbose=True):
    d = pd.DataFrame({
        "clone_id": meta[clone_col].to_numpy(),
        "patient": meta[patient_col].astype(str).to_numpy(),
        "tp": _as_str(meta[time_col]).to_numpy(),
    })
    d = d[d.clone_id.notna() & (d.clone_id.astype(str) != "") & d.tp.isin(lv)].copy()
    d["clone_id"] = d.clone_id.astype(str)
    if d.empty:
        raise ValueError("[cyclone] no clonotyped cells at the requested timepoints")
    ns = d.groupby(["patient", "tp"], sort=False).size().rename("N").reset_index()
    ok = ns[ns.N >= min_cells_sample][["patient", "tp"]]
    d = d.merge(ok, on=["patient", "tp"])
    if drop_incomplete:
        have = d.groupby("patient", sort=False).tp.nunique()
        bad = have[have < len(lv)].index.tolist()
        if bad:
            if verbose:
                print("[cyclone] dropping patients missing a timepoint: " + ", ".join(bad))
            d = d[~d.patient.isin(bad)]
    cnt = d.groupby(["patient", "clone_id", "tp"], sort=False).size().rename("n").reset_index()
    ptp = d.groupby(["patient", "tp"], sort=False).size().rename("n_tcells_tp").reset_index()
    clones = cnt[["patient", "clone_id"]].drop_duplicates()
    grid = clones.merge(pd.DataFrame({"tp": lv}), how="cross")
    cs = grid.merge(cnt, on=["patient", "clone_id", "tp"], how="left").merge(ptp, on=["patient", "tp"], how="left")
    cs["n"] = np.where(cs.n.isna() & cs.n_tcells_tp.notna(), 0, cs.n)
    cs["perc"] = cs.n / cs.n_tcells_tp * 100
    cs["tp_i"] = cs.tp.map({t: i for i, t in enumerate(lv)}).astype(int)
    cs = cs.sort_values(["clone_id", "tp_i"], kind="stable").reset_index(drop=True)
    g = cs.groupby("clone_id", sort=False)
    cs["n_max"] = g.n.transform("max")
    cs["n_min"] = g.n.transform("min")
    cs["range"] = cs.n_max - cs.n_min
    cs["range_perc"] = g.perc.transform("max") - g.perc.transform("min")
    prev = g.perc.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dfc = (cs.perc + reg) / (prev + reg)
    cs["delta_fc"] = dfc.where(dfc.notna(), 1.0)     # NA/NaN -> 1 (as the original)
    dc = cs.perc - prev
    cs["delta_c"] = dc.where(dc.notna(), 0.0)
    cs["avg_freq"] = (cs.perc + prev) / 2
    g = cs.groupby("clone_id", sort=False)
    cs["min_perc"] = g.perc.transform("min")
    cs["max_perc"] = g.perc.transform("max")
    imax = g.perc.idxmax()
    imin = g.perc.idxmin()
    cs["timepoint_max"] = cs.clone_id.map(cs.tp.loc[imax.dropna().astype(int)].set_axis(imax.dropna().index))
    cs["timepoint_min"] = cs.clone_id.map(cs.tp.loc[imin.dropna().astype(int)].set_axis(imin.dropna().index))
    cs["delta_fc_max"] = g.delta_fc.transform("max")
    cs["delta_fc_min"] = g.delta_fc.transform("min")
    cs["delta_c_max"] = g.delta_c.transform("max")
    cs["delta_c_min"] = g.delta_c.transform("min")
    t0 = lv[0]
    cond = (((cs.delta_fc > min_fc) & (cs.delta_c > min_c) & (cs.timepoint_max != t0)) |
            ((cs.delta_fc < 1 / min_fc) & (cs.delta_c < -min_c) & (cs.timepoint_max == t0)))
    cs["pass_dyn"] = cond.groupby(cs.clone_id).transform("any")
    cs["pass"] = cs.pass_dyn & (cs.n_max >= n_max_min)
    return cs, clones


def cyclone(meta: pd.DataFrame, clone_col="clone_id", patient_col="patient_alias", time_col="timepoint",
            timepoints=None, n_clust=6, min_fc=2.0, min_c=0.25, n_max_min=5, method="hclust", scale="timepoint",
            custom_order=None, outlier_sd=None, reg=0.0, group_cols=("treatment", "response"),
            drop_incomplete=True, min_cells_sample=1, stats_only=False, verbose=True):
    """Cyclone: cluster expanding/contracting clones by their temporal trajectory.

    Exact port of ``ar_cyclone`` (itself a faithful port of the published ``cyclone_find_trajectories``):
    per patient x timepoint perc -> consecutive deltas (fc, abs) -> keep clones with any increasing
    (fc > min_fc & d > min_c & peak not at t0) or decreasing (fc < 1/min_fc & d < -min_c & peak at t0) step
    and n_max >= n_max_min -> timepoint x clone matrix standardized (``scale``: "timepoint" = faithful default
    (z-score per timepoint across clones), "clone" or "none" ~ per-clone z / raw) -> Pearson correlation between
    clones -> Euclidean distance on the correlation matrix -> Ward.D2 -> cutree(n_clust), numbered by first
    appearance in sorted(clone_id) order; ``custom_order`` renames cutree ids to Traj 1..n.  ``outlier_sd``:
    among rows with perc > min_c and n_max >= n_max_min, clones with any perc >= mean + outlier_sd * sd -> "Other".
    method "pam" (PAM on the correlation matrix) or "absolute" (trajectory = timepoint index of max perc).
    ``stats_only`` returns just the clone x patient x timepoint change table with logical ``pass``.
    """
    if method not in ("hclust", "pam", "absolute"):
        raise ValueError("method must be hclust|pam|absolute")
    if scale not in ("timepoint", "clone", "none"):
        raise ValueError("scale must be timepoint|clone|none")
    meta = pd.DataFrame(meta)
    group_cols = [c for c in (group_cols or ()) if c in meta.columns]
    lv = _tp_levels(meta[time_col], timepoints)
    cs, clones = _clonestats(meta, clone_col, patient_col, time_col, lv, min_c, min_fc, n_max_min, reg,
                             drop_incomplete, min_cells_sample, verbose)
    pg = _patient_groups(meta, patient_col, group_cols)

    def finish(cs):
        if group_cols:
            cs = pg.merge(cs, on="patient", how="right")
        cs["timepoint"] = pd.Categorical(cs.tp, categories=lv)
        return cs.rename(columns={"patient": patient_col})

    if stats_only:
        out = finish(cs.copy())
        return out.drop(columns=["tp", "pass_dyn"])
    keep = sorted(cs.loc[cs["pass"], "clone_id"].unique())
    if verbose:
        print(f"[cyclone] {len(keep)}/{len(clones)} clones pass (FC>{min_fc:g}, |delta|>{min_c:g} pp, "
              f"n_max>={n_max_min}) in {cs.patient.nunique()} patients")
    if len(keep) < n_clust:
        raise ValueError("[cyclone] fewer passing clones than n_clust")
    sub = cs[cs.clone_id.isin(keep)]
    cidx = {c: i for i, c in enumerate(keep)}
    mat = np.zeros((len(lv), len(keep)))
    mat[sub.tp_i.to_numpy(), sub.clone_id.map(cidx).to_numpy()] = np.nan_to_num(sub.perc.to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        if scale == "timepoint":
            z = (mat - mat.mean(1, keepdims=True)) / mat.std(1, ddof=1, keepdims=True)
        elif scale == "clone":
            z = (mat - mat.mean(0)) / mat.std(0, ddof=1)
        else:
            z = mat
        mat_cor = np.corrcoef(z, rowvar=False)
    Z = None
    if method == "hclust":
        Z = linkage(pdist(mat_cor, "euclidean"), method="ward")
        raw = cut_tree(Z, n_clusters=n_clust).ravel()
        _, first = np.unique(raw, return_index=True)
        rank = {raw[i]: r + 1 for r, i in enumerate(sorted(first))}   # R cutree: numbered by first observation
        ct = np.array([rank[v] for v in raw])
    elif method == "pam":
        D = squareform(pdist(mat_cor, "euclidean"))
        raw = pam(D, n_clust)[0]
        order = {v: r + 1 for r, v in enumerate(pd.unique(raw))}      # cluster::pam numbers by first observation
        ct = np.array([order[v] for v in raw])
    else:
        ct = mat.argmax(0) + 1
    if custom_order is not None and len(custom_order) == n_clust and set(custom_order) <= set(np.unique(ct)):
        pos = {c: i + 1 for i, c in enumerate(custom_order)}
        ct = np.array([pos[v] for v in ct])
    k = pd.Series([f"Traj {v}" for v in ct], index=keep)
    if outlier_sd is not None:
        ref = cs[cs.perc.notna() & (cs.perc > min_c) & (cs.n_max >= n_max_min)]
        maxperc = ref.perc.mean() + outlier_sd * ref.perc.std(ddof=1)
        out = set(ref.loc[ref.perc >= maxperc, "clone_id"])
        isout = k.index.isin(out)
        k[isout] = "Other"
        if verbose:
            print(f"[cyclone] outlier rule (perc >= {maxperc:.2f}%): {int(isout.sum())} clustered clones set to 'Other'")
    cs["k"] = cs.clone_id.map(k).fillna("Other")
    cs = finish(cs).drop(columns=["tp", "pass_dyn"])
    cs = cs.reset_index(drop=True)
    acols = [patient_col, *group_cols, "clone_id", "k", "n_max"]
    assignment = cs[(cs.k != "Other") | cs.clone_id.isin(keep)][acols].drop_duplicates().reset_index(drop=True)
    kc = cs[cs.k != "Other"]
    counts = (kc.groupby([patient_col, *group_cols, "k", "timepoint"], observed=True, sort=True, dropna=False)
              .agg(n_clones=("clone_id", "nunique"), perc=("perc", lambda s: s.sum(skipna=True)))
              .reset_index())
    mc = clone_col
    cell_k = pd.Series(None, index=meta.index, dtype=object)
    cl = meta[mc]
    has = cl.notna() & (cl.astype(str) != "")
    cell_k[has] = "Other"
    cstr = cl.astype(str)
    hit = has & cstr.isin(k.index)
    cell_k[hit] = cstr[hit].map(k)
    mat_df = pd.DataFrame(mat, index=lv, columns=keep)
    cor_df = pd.DataFrame(mat_cor, index=keep, columns=keep)
    return Cyclone(cs, assignment, counts, mat_df, cor_df, Z, cell_k,
                   dict(clone_col=clone_col, patient_col=patient_col, time_col=time_col, timepoints=lv,
                        n_clust=n_clust, min_fc=min_fc, min_c=min_c, n_max_min=n_max_min, method=method,
                        scale=scale, custom_order=custom_order, outlier_sd=outlier_sd, group_cols=group_cols))


def cyclone_patient_table(cy: Cyclone, value="n_clones", at=None) -> pd.DataFrame:
    """Per-patient trajectory table: clones per trajectory (or summed perc at timepoint ``at``), wide."""
    pc = cy.params["patient_col"]
    d = cy.counts
    if value == "n_clones":
        d = d.drop_duplicates([pc, "k"])
    elif at is not None:
        d = d[d.timepoint.astype(str) == _fmt_tp(at)]
    idx = [c for c in d.columns if c not in ("timepoint", "n_clones", "perc", "k")]
    w = d.groupby([*idx, "k"], dropna=False, observed=True, sort=True)[value].sum().unstack("k", fill_value=0)
    w.columns.name = None
    return w.reset_index()


def cyclone_states(cy: Cyclone, meta: pd.DataFrame, state_col: str, delta="fc", reg=2.0, mki67=None,
                   group_col="treatment") -> pd.DataFrame:
    """Per-patient state composition of trajectory clones over time (``ar_cyclone_states``).

    value n = % of the patient's clonotyped cells at that timepoint in trajectory k AND state s;
    delta "fc" = log2((n+reg)/(n_t0+reg)), "abs" = n - n_t0, "none" = n.  One row per patient x time x k x state.
    """
    pc, tc = cy.params["patient_col"], cy.params["time_col"]
    tps = cy.params["timepoints"]
    meta = pd.DataFrame(meta)
    d = pd.DataFrame({"patient": meta[pc].astype(str).to_numpy(), "tp": _as_str(meta[tc]).to_numpy(),
                      "state": meta[state_col].astype(str).to_numpy(), "k": cy.cell_k.to_numpy(),
                      "grp": meta[group_col].astype(str).to_numpy() if group_col in meta.columns else "all"})
    if mki67 is not None:
        d["mki"] = np.asarray(mki67, dtype=float) > 0
    d = d[d.k.notna() & d.tp.isin(tps) & d.patient.isin(set(cy.assignment[pc].astype(str)))]
    tot = d.groupby(["patient", "tp"]).size().rename("count").reset_index()
    x = (d[d.k != "Other"].groupby(["grp", "patient", "tp", "k", "state"]).size().rename("cells").reset_index())
    ks, ss = sorted(x.k.unique()), sorted(x.state.unique())
    gp = d[["grp", "patient"]].drop_duplicates()
    full = gp.merge(pd.DataFrame(list(itertools.product(tps, ks, ss)), columns=["tp", "k", "state"]), how="cross")
    x = full.merge(x, on=["grp", "patient", "tp", "k", "state"], how="left")
    x["cells"] = x.cells.fillna(0)
    x = x.merge(tot, on=["patient", "tp"], how="left")
    x["n"] = x.cells / x["count"] * 100
    x["tp_i"] = x.tp.map({t: i + 1 for i, t in enumerate(tps)})
    x = x.sort_values(["grp", "patient", "k", "state", "tp_i"], kind="stable").reset_index(drop=True)
    n0 = x[x.tp_i == 1].set_index(["grp", "patient", "k", "state"]).n
    x["n0"] = pd.MultiIndex.from_frame(x[["grp", "patient", "k", "state"]]).map(n0.to_dict())
    if delta == "fc":
        x["value"] = np.log2((x.n + reg) / (x.n0 + reg))
    elif delta == "abs":
        x["value"] = x.n - x.n0
    else:
        x["value"] = x.n
    if mki67 is not None:
        cyc = d[d.k != "Other"].groupby(["grp", "tp", "k", "state"]).mki.mean().mul(100).rename("cycling").reset_index()
        x = x.merge(cyc, on=["grp", "tp", "k", "state"], how="left")
    x["timepoint"] = pd.Categorical(x.tp, categories=tps)
    x = x.rename(columns={"patient": pc, "grp": group_col}).drop(columns=["tp", "tp_i"])
    x.attrs["delta"] = delta
    return x


# ============================================================================================ HD-null expansion
def read_10x_clonotypes(files, require_paired=True) -> pd.DataFrame:
    """Read 10x ``clonotypes.csv`` files ("<donor> <timepoint> clonotypes.csv") -> one row per cell.

    clone = TRA nt + " " + TRB nt (multiple chains comma-joined), the paired definition of the original.
    """
    import os
    out = []
    for f in files:
        d = pd.read_csv(f)
        parts = os.path.basename(str(f)).removesuffix(" clonotypes.csv").split(" ")
        tra, trb = [], []
        for s in d["cdr3s_nt"].fillna("").astype(str):
            ch = [c for c in s.split(";") if c]
            tra.append(",".join(c[4:] for c in ch if c.startswith("TRA:")))
            trb.append(",".join(c[4:] for c in ch if c.startswith("TRB:")))
        tra, trb = np.array(tra, dtype=object), np.array(trb, dtype=object)
        ok = (tra != "") & (trb != "") if require_paired else np.ones(len(d), bool)
        clone = np.char.add(np.char.add(tra[ok].astype(str), " "), trb[ok].astype(str))
        out.append(pd.DataFrame({"patient_alias": parts[0], "timepoint": parts[1],
                                 "clone_id": np.repeat(clone, d["frequency"].to_numpy()[ok])}))
    return pd.concat(out, ignore_index=True)


def _signed_chi(a, b, N1, N2):
    a = a.astype(float)
    b = b.astype(float)
    c1, c2 = N1 - a, N2 - b
    N = N1 + N2
    r1, r2 = a + b, c1 + c2
    O = np.stack([a, b, c1, c2], 1)
    E = np.stack([r1 * N1, r1 * N2, r2 * N1, r2 * N2], 1) / N
    dev = np.abs(O - E)
    yates = np.minimum(0.5, dev.min(1))             # == chisq.test(correct = TRUE)
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = ((dev - yates[:, None]) ** 2 / E).sum(1)
        sgn = np.where(b / c2 > a / c1, 1.0, -1.0)  # original: tp2/tot3 > tp1/tot0
    return sgn * np.sqrt(x2)


def _pair_counts(c1, c2):
    u1, n1 = np.unique(c1, return_counts=True)
    u2, n2 = np.unique(c2, return_counts=True)
    cl = np.union1d(u1, u2)
    a = np.zeros(len(cl), np.int64)
    b = np.zeros(len(cl), np.int64)
    a[np.searchsorted(cl, u1)] = n1
    b[np.searchsorted(cl, u2)] = n2
    return cl, a, b


def pair_test(c1, c2) -> pd.DataFrame:
    """Per-clone Fisher p and signed sqrt(Yates chi-square) between two cell-label vectors."""
    N1, N2 = len(c1), len(c2)
    cl, a, b = _pair_counts(np.asarray(c1), np.asarray(c2))
    return pd.DataFrame({"clone": cl, "n_1": a, "n_2": b, "N_1": N1, "N_2": N2,
                         "chi": _signed_chi(a, b, N1, N2), "p": fisher_exact_2x2(a, b, N1, N2)})


def clone_expansion_null(meta, hd, pairs=((0, 3), (3, 6), (6, 9)), hd_times=("T1", "T2"),
                         clone_col="clone_id", patient_col="patient_alias", time_col="timepoint",
                         ref_depth=None, null_q=0.95, reps=1, seed=1, downsample=True,
                         hd_time_col=None, hd_clone_col=None):
    """Per-clone expansion/contraction between consecutive timepoints calibrated by a healthy-donor null.

    Port of ``ar_clone_expansion_null``: per patient and pair, each timepoint is downsampled to at most the HD
    depth (``ref_depth``, default HD cell counts); per clone a Fisher test (BH within patient x pair -> ``fdr``)
    and a signed sqrt(Yates chi2) ``chi``.  Null: HD T1 vs T2 downsampled to the patient's depths, cutoff =
    quantile(|chi_HD|, null_q); ``sig_null`` if |chi| > cutoff.  ``reps`` repeated downsamplings are averaged in
    the patient summary.  Random draws use numpy (``seed``), so they are statistically but not bitwise equal to R.

    Returns ``(clones, patients)`` DataFrames.
    """
    hd_time_col = hd_time_col or time_col
    hd_clone_col = hd_clone_col or clone_col
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(meta)
    cl = meta[clone_col]
    meta = meta[cl.notna() & (cl.astype(str) != "")]
    hts = hd[hd_time_col].astype(str)
    h1 = hd.loc[hts == str(hd_times[0]), hd_clone_col].astype(str).to_numpy()
    h2 = hd.loc[hts == str(hd_times[1]), hd_clone_col].astype(str).to_numpy()
    if not len(h1) or not len(h2):
        raise ValueError("[clones] no HD cells at " + "/".join(map(str, hd_times)))
    ref_depth = ref_depth or (len(h1), len(h2))

    def sub(v, n):
        return v[rng.choice(len(v), n, replace=False)] if len(v) > n else v

    tps = _as_str(meta[time_col]).to_numpy()
    codes, uniq = pd.factorize(meta[clone_col].astype(str))   # integer clone codes: fast unique/counts
    uniq = np.asarray(uniq, dtype=object)
    hcodes_map = {u: i for i, u in enumerate(uniq)}
    # HD labels get their own integer space (the null is HD vs HD only)
    hu, h1c = np.unique(h1, return_inverse=True)
    h2c = np.searchsorted(hu, h2) if np.isin(h2, hu).all() else None
    if h2c is None:
        hu, inv = np.unique(np.concatenate([h1, h2]), return_inverse=True)
        h1c, h2c = inv[:len(h1)], inv[len(h1):]
    pts = meta[patient_col].to_numpy()
    res = []
    for pt in pd.unique(pts):
        pm = pts == pt
        for pr in pairs:
            c1 = codes[pm & (tps == _fmt_tp(pr[0]))]
            c2 = codes[pm & (tps == _fmt_tp(pr[1]))]
            if not len(c1) or not len(c2):
                continue
            for r in range(1, reps + 1):
                d1 = sub(c1, ref_depth[0]) if downsample else c1
                d2 = sub(c2, ref_depth[1]) if downsample else c2
                tp = pair_test(d1, d2)
                tp["clone"] = uniq[tp["clone"].to_numpy()]
                tp["fdr"] = p_adjust_bh(tp.p.to_numpy())
                nul = pair_test(sub(h1c, len(d1)), sub(h2c, len(d2)))
                cut = float(np.quantile(np.abs(nul.chi.to_numpy()), null_q))
                tp["cutoff"] = cut
                tp["sig_null"] = np.abs(tp.chi) > cut
                tp["hd_frac_sig_fdr"] = float(np.mean(p_adjust_bh(nul.p.to_numpy()) < 0.05))
                tp[patient_col] = pt
                tp["t1"], tp["t2"], tp["rep"] = pr[0], pr[1], r
                res.append(tp)
    del hcodes_map
    clones = pd.concat(res, ignore_index=True)
    clones["_up"] = clones.sig_null & (clones.chi > 0)
    clones["_dn"] = clones.sig_null & (clones.chi < 0)
    clones["_fu"] = (clones.fdr < 0.05) & (clones.chi > 0)
    clones["_fd"] = (clones.fdr < 0.05) & (clones.chi < 0)
    clones["_cu"] = np.where(clones._up, clones.n_2, 0)
    gk = [patient_col, "t1", "t2", "rep"]
    s = clones.groupby(gk, sort=False).agg(
        N_1=("N_1", "first"), N_2=("N_2", "first"), n_clones=("clone", "size"), cutoff=("cutoff", "first"),
        n_sig_up=("_up", "sum"), n_sig_down=("_dn", "sum"), n_fdr_up=("_fu", "sum"), n_fdr_down=("_fd", "sum"),
        cu=("_cu", "sum"), hd_frac_sig_fdr=("hd_frac_sig_fdr", "first")).reset_index()
    s["frac_cells_sig_up_t2"] = s.cu / s.N_2
    num = ["N_1", "N_2", "n_clones", "cutoff", "n_sig_up", "n_sig_down", "n_fdr_up", "n_fdr_down",
           "frac_cells_sig_up_t2", "hd_frac_sig_fdr"]
    pat = s.groupby([patient_col, "t1", "t2"], sort=False)[num].mean()
    pat["reps"] = s.groupby([patient_col, "t1", "t2"], sort=False).size()
    clones = clones.drop(columns=["_up", "_dn", "_fu", "_fd", "_cu"])
    return clones, pat.reset_index()


# ============================================================================================ STARTRAC
def _row_entropy(M: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of each row of a count matrix."""
    tot = M.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(M > 0, M / np.where(tot > 0, tot, 1), 1.0)
        return -(np.where(M > 0, p * np.log2(p), 0.0)).sum(1)


def _count_matrix(codes_r, codes_c, nr, nc):
    M = np.zeros((nr, nc))
    np.add.at(M, (codes_r, codes_c), 1)
    return M


def startrac(meta, clone_col="clone_id", state_col="metaclusters_exh", group_cols="patient_alias", loc_col=None,
             min_cells=20, downsample=None, seed=1):
    """STARTRAC-style indices (Zhang et al. 2018) computed per patient (or sample).

    expa(state) = 1 - H(clone sizes in state)/log2(n cells in state); tran(state) = sum_c w_c H_c(states);
    migr(state) = same with ``loc_col``; pairwise tran(A,B) over states A,B.  Entropies in bits, unnormalised.
    Returns ``(cluster, pairwise)`` DataFrames.
    """
    rng = np.random.default_rng(seed)
    gc = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    cols = list(dict.fromkeys([clone_col, state_col, *gc, *([loc_col] if loc_col else [])]))
    d = pd.DataFrame(meta)[cols].copy()
    d = d.rename(columns={clone_col: "clone", state_col: "state"}) if clone_col != state_col else d
    d = d[d.clone.notna() & (d.clone.astype(str) != "") & d.state.notna()]
    d["clone"], d["state"] = d.clone.astype(str), d.state.astype(str)
    if downsample is not None:
        d = d.groupby([*gc, "state"], sort=False, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), downsample), random_state=int(rng.integers(2**31))))
    crows, prows = [], []
    for key, g in d.groupby(gc, sort=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        cc, cu = pd.factorize(g.clone)
        sc, su = pd.factorize(g.state, sort=True)
        M = _count_matrix(cc, sc, len(cu), len(su))
        Hc = _row_entropy(M)
        n_s = M.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Hs = _row_entropy(M.T)
            expa = 1 - Hs / np.log2(np.maximum(n_s, 2))
            tran = (M * Hc[:, None]).sum(0) / n_s
            if loc_col:
                lc, lu = pd.factorize(g[loc_col].astype(str))
                Hm = _row_entropy(_count_matrix(cc, lc, len(cu), len(lu)))
                migr = (M * Hm[:, None]).sum(0) / n_s
        for j, s in enumerate(su):
            row = dict(zip(gc, key))
            row.update(state=s, n=int(n_s[j]), expa=expa[j], tran=tran[j])
            if loc_col:
                row["migr"] = migr[j]
            crows.append(row)
        sel = [j for j in range(len(su)) if n_s[j] >= min_cells]
        for i, j in itertools.combinations(sel, 2):
            tot = M[:, i] + M[:, j]
            pair = np.stack([M[:, i], M[:, j]], 1)
            h = _row_entropy(pair)
            row = dict(zip(gc, key))
            row.update(state_a=su[i], state_b=su[j], tran=float((tot * h).sum() / tot.sum()),
                       n_shared_clones=int(((M[:, i] > 0) & (M[:, j] > 0)).sum()))
            prows.append(row)
    cluster = pd.DataFrame(crows)
    cluster = cluster[cluster.n >= min_cells].reset_index(drop=True)
    pairwise = pd.DataFrame(prows, columns=[*gc, "state_a", "state_b", "tran", "n_shared_clones"])
    return cluster, pairwise


# ============================================================================================ clone helpers
def clone_table(meta, clone_col, sample_cols=("patient_alias", "timepoint"), by=None) -> pd.DataFrame:
    """Clone counts and within-sample frequency per sample (cells without a clone dropped)."""
    sc = list(sample_cols) + ([by] if isinstance(by, str) else list(by or []))
    d = pd.DataFrame(meta)
    d = d[d[clone_col].notna() & (d[clone_col].astype(str) != "")]
    ct = d.groupby([*sc, d[clone_col].rename("clone")], observed=True, sort=True).size().rename("n").reset_index()
    ct["freq"] = ct.n / ct.groupby(list(sample_cols), observed=True).n.transform("sum")
    return ct


def clone_diversity(ct, sample_cols=("patient_alias", "timepoint"), depth=None, floor=200, reps=20, seed=1,
                    verbose=True) -> pd.DataFrame:
    """Diversity per sample after rarefying to a common depth (default = min sample depth >= ``floor``).

    Metrics (means over ``reps`` draws): shannon (nats), simpson, richness, top10_frac, clonality.
    Rarefaction uses an exact multivariate hypergeometric draw (== sampling cells without replacement).
    """
    sc = list(sample_cols)
    rng = np.random.default_rng(seed)
    sizes = ct.groupby(sc, observed=True).n.sum().rename("N")
    if depth is None:
        eligible = sizes[sizes >= floor]
        if eligible.empty:
            raise ValueError("no sample reaches the depth floor")
        depth = int(max(floor, eligible.min()))
    ok = sizes[sizes >= depth].index
    if verbose:
        print(f"[clones] diversity rarefied to {depth} cells; {len(ok)}/{len(sizes)} samples retained")
    rows = []
    for key, d in ct.groupby(sc, observed=True, sort=True):
        if key not in ok:
            continue
        counts = d.n.to_numpy().astype(np.int64)
        draws = rng.multivariate_hypergeometric(counts, depth, size=reps)
        m = []
        for s in draws:
            s = s[s > 0]
            p = s / s.sum()
            h = -(p * np.log(p)).sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                clon = 1 - h / np.log(len(s))
            m.append([h, 1 - (p ** 2).sum(), len(s), np.sort(p)[::-1][:10].sum(), clon])
        mm = np.mean(m, axis=0)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(sc, key)), "shannon": mm[0], "simpson": mm[1], "richness": mm[2],
                     "top10_frac": mm[3], "clonality": mm[4], "depth": depth})
    return pd.DataFrame(rows)


def clone_dynamics(ct, t0, t1, patient="patient_alias", time="timepoint", min_cells=3) -> pd.DataFrame:
    """Clone dynamics between two timepoints within each patient (Fisher per clone, BH within patient)."""
    out = []
    for pt, d in ct.groupby(patient, sort=False, observed=True):
        tt = d[time].astype(str)
        a = d[tt == _fmt_tp(t0)]
        b = d[tt == _fmt_tp(t1)]
        if a.empty or b.empty:
            continue
        Na, Nb = int(a.n.sum()), int(b.n.sum())
        m = (a[["clone", "n"]].merge(b[["clone", "n"]], on="clone", how="outer", suffixes=("_0", "_1"))
             .fillna({"n_0": 0, "n_1": 0}).sort_values("clone"))
        m = m[(m.n_0 + m.n_1) >= min_cells].copy()
        if m.empty:
            continue
        m["n_0"], m["n_1"] = m.n_0.astype(int), m.n_1.astype(int)
        m["p"] = fisher_exact_2x2(m.n_0.to_numpy(), m.n_1.to_numpy(), Na, Nb)
        m["fdr"] = p_adjust_bh(m.p.to_numpy())
        m["lfc"] = np.log2((m.n_1 / Nb + 1e-5) / (m.n_0 / Na + 1e-5))
        sig = m.fdr < 0.05
        m["class"] = np.select([sig & (m.lfc > 0) & (m.n_0 == 0), sig & (m.lfc > 0), sig & (m.lfc < 0)],
                               ["novel_expanded", "expanded", "contracted"], "stable")
        m[patient] = pt
        m["N0"], m["N1"] = Na, Nb
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def clone_tissue_matrix(meta, clone_col="clone_id", tissue_col="tissue", min_cells=1) -> pd.DataFrame:
    """Clone x tissue cell-count matrix (clones with fewer than ``min_cells`` cells dropped)."""
    d = pd.DataFrame(meta)
    d = d[d[clone_col].notna() & (d[clone_col].astype(str) != "") & d[tissue_col].notna()]
    m = pd.crosstab(d[clone_col].astype(str), d[tissue_col].astype(str))
    return m[m.sum(1) >= min_cells]


def clone_tissue_entropy(mat: pd.DataFrame) -> pd.DataFrame:
    """Per-clone tissue distribution: bits entropy, entropy normalised by log2(#tissues), tissue breadth,
    dominant tissue and its fraction (low entropy = tissue-restricted, high = disseminated)."""
    M = mat.to_numpy().astype(float)
    H = _row_entropy(M)
    nt = (M > 0).sum(1)
    n = M.sum(1)
    k = M.shape[1]
    return pd.DataFrame({"n_cells": n.astype(int), "n_tissues": nt, "entropy": H,
                         "entropy_norm": H / np.log2(k) if k > 1 else np.zeros(len(H)),
                         "dominant": mat.columns[M.argmax(1)], "dominant_frac": M.max(1) / n}, index=mat.index)
