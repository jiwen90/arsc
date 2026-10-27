"""arsc.clonotrace vs the Clonotrace R code (Fu et al. 2025; github.com/yuntianf/Clonotrace v1.2.1).

R references: benchmarks/r_reference_clonotrace.R -> $ARSC_REFS/clonotrace_ref (regen_refs.py clonotrace_ref) (public demo data shipped
with Clonotrace; no lab data). Reference tests skip when those files are absent; synthetic tests always run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from arsc import clonotrace as ct
from arsc.clones import ari

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from refutil import CLONOTRACE_REF as REF, missing  # noqa: E402

_SKIP = missing(REF / "coembed.csv", regen="clonotrace_ref")
have_ref = pytest.mark.skipif(bool(_SKIP), reason=_SKIP or "")


def rd(name, **kw):
    return pd.read_csv(REF / name, **kw)


def trip(name, n):
    t = rd(name)
    return sp.csr_matrix((t.x.to_numpy(), (t.i.to_numpy(), t.j.to_numpy())), shape=(n, n))


@pytest.fixture(scope="module")
def ref():
    pca = rd("pca.csv").set_index("cell")
    meta = rd("meta.csv", keep_default_na=False, na_values=["NA"]).set_index("cell")
    levels = rd("clone_levels.csv").level.tolist()
    X = pca.to_numpy()
    return {"X": X, "meta": meta, "levels": levels, "n": X.shape[0], "cells": pca.index}


@pytest.fixture(scope="module")
def Q(ref):
    """R's Q (after its own bootstrap filter + sparsify), as a cells x clones frame."""
    keep = rd("dev_keep.csv").keep.to_numpy()
    q = rd("Q.csv").to_numpy()
    df = pd.DataFrame(q, index=ref["cells"][keep], columns=ref["levels"])
    return df.loc[df.sum(1) > 0, df.sum(0) > 0]


# ------------------------------------------------------------------------------------------ graph + spreading
@have_ref
def test_knn_connectivity_and_transition(ref):
    K = ct.knn_connectivity(ref["X"], 30)
    Kr = trip("knn_connectivity.csv", ref["n"])
    assert (K != 0).sum() == (Kr != 0).sum()
    assert abs(K - Kr).max() < 1e-12
    T = ct.cell_transition(ref["X"], 30)
    assert abs(T - trip("transition.csv", ref["n"])).max() < 1e-14


@have_ref
def test_label_spreading_exact(ref):
    T = trip("transition.csv", ref["n"])
    codes, lv = ct.encode_clones(ref["meta"].clone, ref["levels"])
    F = ct.label_spreading(T, codes, len(lv), alpha=0.6, block_size=8)
    Fr = rd("spread_raw.csv").to_numpy()
    assert np.abs(F - Fr).max() < 1e-13
    cp = ct.propagate_clones(T, ref["meta"].clone, min_size=1, alpha=0.6, block_size=8, n_boot=0, levels=lv)
    assert np.abs(cp.prob - rd("spread_norm.csv").to_numpy()).max() < 1e-13


@have_ref
def test_bootstrap_deviance_statistical(ref):
    T = trip("transition.csv", ref["n"])
    kw = dict(min_size=1, alpha=0.6, block_size=8, levels=ref["levels"])
    big = ct.propagate_clones(T, ref["meta"].clone, seed=1, n_boot=48, **kw).deviance
    dr = rd("deviance.csv").mean(1).to_numpy()  # R: 3 runs x 12 bootstraps, R's RNG
    # same estimand: the R estimate tracks a 48-bootstrap Python estimate as closely as Python
    # estimates of the same size do (measured: R 0.88, Python 36 vs 48 boots 0.92), and the means agree
    assert np.corrcoef(dr, big)[0, 1] > 0.85
    assert abs(np.mean(dr) - np.mean(big)) < 0.05 * np.mean(big)


@have_ref
def test_sparsify_exact(ref):
    keep = rd("dev_keep.csv").keep.to_numpy()
    R = rd("spread_norm.csv").to_numpy()[keep]
    R = R / R.sum(1, keepdims=True)
    S = ct.sparsify(R, 0.9, 0.9)
    with np.errstate(invalid="ignore"):
        S = np.nan_to_num(S / S.sum(1, keepdims=True))
    assert np.abs(S - rd("Q.csv").to_numpy()).max() < 1e-14


# ------------------------------------------------------------------------------------------ distances
@have_ref
def test_snn_cost_graph(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    G = ct.snn_cost_graph(ref["X"][pos], 10)
    t = rd("snn_cost.csv")
    Gr = sp.coo_matrix((t.w, (t.i, t.j)), shape=G.shape).tocsr()
    Gr = Gr + Gr.T
    assert G.nnz == Gr.nnz and abs(G - Gr).max() < 1e-15


@have_ref
def test_clone_distance_nn_exact(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    L = ct.clone_distance(ref["X"][pos], Q, method="nn", square=False)
    r = rd("dist_nn.csv")
    np.testing.assert_array_equal(L.group1, r.group1)
    np.testing.assert_array_equal(L.group2, r.group2)
    assert np.abs(L.dis.to_numpy() - r.dis.to_numpy()).max() < 1e-12


@have_ref
def test_clone_distance_exact_ot(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    D = ct.clone_distance(ref["X"][pos], Q.iloc[:, :9], method="exact")
    r = rd("dist_ot.csv")
    got = D.to_numpy()[r.group1 - 1, r.group2 - 1]
    assert np.abs(got - r.dis.to_numpy()).max() < 1e-9


@have_ref
def test_revised_distance_fixed_clusters(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    Dn = ct.clone_distance(ref["X"][pos], Q, method="nn")
    m = ref["meta"].iloc[pos]
    cl = pd.DataFrame({"c1": m.cluster.astype(str).to_numpy(), "c2": m.celltype.to_numpy()})
    Rv, info = ct.revise_distance(Dn, ref["X"][pos], Q, clusters=cl)
    w = rd("revised_weight.csv")
    assert info["mean_weight"] == pytest.approx(w.mean_weight[0])
    assert info["picks"] == pytest.approx([w.pick1[0], w.pick2[0]])
    assert np.nanmax(np.abs(info["selection_correlations"] - rd("revised_curves.csv").to_numpy())) < 1e-12
    assert np.abs(Rv.to_numpy() - rd("dist_revised.csv").to_numpy()).max() < 1e-12


# ------------------------------------------------------------------------------------------ clone level
@have_ref
def test_cluster_cmdscale_dpt(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    Dn = ct.clone_distance(ref["X"][pos], Q, method="nn")
    lr = rd("leiden_dis.csv")
    best = max(ari(ct.cluster_clones(Dn, k=8, resolution=0.5, seed=s).to_numpy(), lr.cluster.to_numpy())
               for s in range(5))
    assert best > 0.8
    cm = ct.cmdscale(Dn, 6)
    cr = rd("cmdscale.csv").to_numpy()
    assert np.abs(np.abs(cm.to_numpy()) - np.abs(cr)).max() < 1e-10
    pts, stress = ct.isomds(Dn, 6)
    assert stress <= rd("isomds_stress.csv").stress[0] * 1.1 + 0.5
    cell_clone = ref["meta"].clone.iloc[pos].to_numpy()
    root = ct.clone_root(Q.columns, cell_clone, ref["meta"].cluster.iloc[pos].astype(str).to_numpy(), "0")
    dr = rd("clone_dpt.csv")
    assert root == dr.root[0]
    emb = pd.DataFrame(cr, index=Q.columns)  # R's cmdscale (sign convention) as input
    t = ct.clone_dpt(emb, root, k=5, dpt_k=6)
    assert np.abs(t.to_numpy() - dr.t.to_numpy()).max() < 1e-8


@have_ref
def test_profiles_and_enrichment(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    lr = rd("leiden_dis.csv").set_index("clone").cluster
    cpp, lab = ct.cell_profiles(Q, lr)
    assert np.abs(cpp.to_numpy() - rd("cell_profile_prob.csv").to_numpy()).max() < 1e-14
    clus = ref["meta"].cluster.iloc[pos].astype(str).to_numpy()
    mass, pv = ct.profile_enrichment(cpp, clus, n_perm=400, seed=0)
    assert np.abs(mass.to_numpy() - rd("profile_mass.csv").to_numpy()).max() < 1e-12
    assert list(mass.index) == rd("profile_mass_rows.csv").cluster.astype(str).tolist()
    pr = rd("enrich_pval.csv").to_numpy()
    assert np.mean(np.abs(pv.to_numpy() - pr) < 0.1) > 0.95  # permutation noise


@have_ref
def test_ns_basis_and_gam(ref):
    dr = rd("clone_dpt.csv")
    x = dr.t.to_numpy()[:20] + np.linspace(0, 1, 20)
    for df in range(1, 5):
        B = ct.ns_basis(x, df)
        Br = rd(f"ns_df{df}.csv").to_numpy()
        # same column space: projection residual ~ 0
        res = Br - B @ np.linalg.lstsq(B, Br, rcond=None)[0]
        assert np.abs(res).max() < 1e-10
        if df == 1:
            assert np.abs(np.abs(B[:, 0]) - np.abs(Br[:, 0])).max() < 1e-12
    G = rd("gam_G.csv").to_numpy().T  # cells x genes
    tP = rd("gam_tP.csv")
    for df in (1, 3):
        s = ct.soft_cluster_fit(G, tP.t, tP[["p1", "p2"]], df=df)
        r = rd(f"gam_df{df}.csv")
        for c in ("stat", "pval", "mean_diff", "cohen"):
            np.testing.assert_allclose(s[c].to_numpy(), r[c].to_numpy(), rtol=1e-7, atol=1e-12)
    s = ct.soft_cluster_fit(G, tP.t, tP[["p1", "p2"]], df=2, test="LRT")
    np.testing.assert_allclose(s.stat.to_numpy(), rd("gam_lrt_df2.csv").stat.to_numpy(), rtol=1e-7)
    flag = ct.bin_filter(tP[["p1", "p2"]].to_numpy() * 3, tP.t.to_numpy(), thresh=3, binsize=0.005)
    np.testing.assert_array_equal(flag, rd("bin_filter.csv").flag.to_numpy())


@have_ref
def test_coembed_graph(ref, Q):
    pos = ref["cells"].get_indexer(Q.index)
    emb = pd.DataFrame(rd("cmdscale.csv").to_numpy(), index=Q.columns)
    M = ct.coembed_graph(ref["X"][pos], emb.to_numpy(), Q, cell_k=15, clone_k=5)
    Mr = trip("coembed.csv", len(pos))
    assert M.nnz == Mr.nnz
    assert abs(M - Mr).max() < 1e-12


# ------------------------------------------------------------------------------------------ synthetic (always run)
def _toy(seed=0, n_per=60, n_clones=6):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=1.5, size=(n_clones, 5))
    X = np.vstack([c + rng.normal(size=(n_per, 5)) for c in centers])
    clones = np.repeat([f"c{i}" for i in range(n_clones)], n_per).astype(object)
    clones[rng.random(clones.size) < 0.3] = None
    return X, clones


def test_pipeline_runs_and_separates_clones():
    X, clones = _toy()
    res = ct.run(X, clones, n_boot=4, n_mds=3, cluster_k=3, dpt_k=4, dpt_nn=3, k=10)
    assert res.Q.shape[1] == 6
    np.testing.assert_allclose(res.Q.sum(1).to_numpy(), 1.0)
    D = res.distance.to_numpy()
    assert np.allclose(D, D.T) and (np.diag(D) == 0).all()
    tab = res.table()
    assert set(tab.columns) >= {"clone", "mass", "n_cells", "profile", "clone_t"}
    assert tab.clone_t.min() == 0 and tab.clone_t.max() == pytest.approx(1)


def test_label_spreading_block_invariance_when_converged():
    X, clones = _toy(1)
    T = ct.cell_transition(X, 10)
    codes, lv = ct.encode_clones(pd.Series(clones))
    a = ct.label_spreading(T, codes, len(lv), alpha=0.6, tol=1e-12, max_iter=2000, block_size=2)
    b = ct.label_spreading(T, codes, len(lv), alpha=0.6, tol=1e-12, max_iter=2000, block_size=128)
    assert np.abs(a - b).max() < 1e-10


def test_dijkstra_matches_scipy():
    from scipy.sparse.csgraph import dijkstra
    X, _ = _toy(2)
    G = ct.snn_cost_graph(X, 8)
    src = np.arange(0, X.shape[0], 7)
    np.testing.assert_allclose(ct.shortest_paths(G, src), dijkstra(G, directed=False, indices=src), rtol=1e-13)


def test_emd_highs_matches_pot_or_bruteforce():
    rng = np.random.default_rng(3)
    a = rng.random(4); a /= a.sum()
    b = rng.random(5); b /= b.sum()
    M = rng.random((4, 5))
    v = ct._emd_highs(a, b, M)
    from scipy.optimize import linprog
    # dense LP as an independent check
    A = np.zeros((9, 20))
    for i in range(4):
        A[i, i * 5:(i + 1) * 5] = 1
    for j in range(5):
        A[4 + j, j::5] = 1
    ref = linprog(M.ravel(), A_eq=A, b_eq=np.r_[a, b], bounds=(0, None), method="highs").fun
    assert v == pytest.approx(ref, abs=1e-10)


def test_sparsify_semantics():
    m = np.array([[0.5, 0.3, 0.15, 0.05], [0.95, 0.05, 0, 0], [0.25, 0.25, 0.25, 0.25]])
    s = ct._mass_filter_rows(m, 0.9)
    np.testing.assert_array_equal(s[0], [0.5, 0.3, 0, 0])      # entry crossing 0.9 dropped
    np.testing.assert_array_equal(s[1], [0.95, 0, 0, 0])       # top alone >= 0.9
    np.testing.assert_array_equal(s[2], m[2])                  # ties kept


def test_cyclone_join():
    from arsc import clones as C
    X, clones = _toy(4)
    res = ct.run(X, clones, n_boot=2, n_mds=3, cluster_k=3, dpt_k=4, dpt_nn=3, k=10)

    class FakeCy:  # minimal stand-in with the Cyclone fields clone_table uses
        params = {"clone_col": "clone_id", "patient_col": "patient_alias"}
        assignment = pd.DataFrame({"clone_id": ["c0", "c1"], "k": ["1", "2"], "patient_alias": ["P1", "P1"]})

    tab = ct.clone_table(res, FakeCy())
    assert tab.set_index("clone").loc["c0", "cyclone_k"] == "1"
    assert tab.set_index("clone").loc["c3", "cyclone_k"] != tab.set_index("clone").loc["c3", "cyclone_k"]  # NaN
    assert C is not None


def test_pava_and_isomds_regression():
    # _pava (numba) vs a plain pool-adjacent-violators; isomds stress vs the value of the former np.add.at version
    def pava_ref(y):
        blocks = []
        for v in y:
            blocks.append([v, 1])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2 = blocks.pop()
                v1, w1 = blocks.pop()
                blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
        return np.concatenate([[v] * w for v, w in blocks])
    rng = np.random.default_rng(4)
    y = np.cumsum(rng.normal(size=500)) + rng.normal(scale=3, size=500)
    np.testing.assert_allclose(ct._pava(y), pava_ref(y), rtol=1e-12, atol=1e-12)
    from scipy.spatial.distance import pdist, squareform
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    D = squareform(pdist(X)) * (1 + 0.1 * rng.random((60, 60)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    pts, stress = ct.isomds(D, 6)
    assert stress == pytest.approx(6.120126444083615, rel=1e-6)
    assert pts.shape == (60, 6)
