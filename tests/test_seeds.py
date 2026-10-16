"""Seed determinism of every stochastic public function (see SEEDS.md).

For each function: (a) the same seed twice -> identical output (exact equality: np.array_equal on every
array / frame / sparse matrix), (b) a different seed -> a different output where the randomness can show
(not for converged eigensolvers, the exact kNN or exact permutation enumeration), (c) the default call
twice -> identical output. Plus: ``seed=None`` is rejected, and no function touches the global numpy /
Python RNG. Synthetic inputs only.
"""
from __future__ import annotations

import dataclasses
import random

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import arsc
from arsc import clones as C
from arsc import clonotrace as ct
from arsc import scanpy_extra as sx
from arsc import stats as st


# ------------------------------------------------------------------------------------------ helpers
def same(a, b) -> bool:
    """Exact (bitwise for floats, NaN == NaN) equality of nested results."""
    if type(a) is not type(b):
        return False
    if isinstance(a, pd.DataFrame):
        return a.shape == b.shape and a.index.equals(b.index) and a.columns.equals(b.columns) and \
            all(same(a.iloc[:, i].to_numpy(), b.iloc[:, i].to_numpy()) for i in range(a.shape[1]))
    if isinstance(a, pd.Series):
        return a.index.equals(b.index) and same(a.to_numpy(), b.to_numpy())
    if isinstance(a, pd.Index):
        return a.equals(b)
    if sp.issparse(a):
        a, b = sp.csr_matrix(a), sp.csr_matrix(b)
        a.sort_indices(); b.sort_indices()
        return a.shape == b.shape and same(a.indptr, b.indptr) and same(a.indices, b.indices) and same(a.data, b.data)
    if isinstance(a, np.ndarray):
        if a.dtype.kind in "fc":
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return a.shape == b.shape and all(same(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if dataclasses.is_dataclass(a):
        return all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return a == b


# ------------------------------------------------------------------------------------------ synthetic data
def _blobs(n_per=40, k=4, d=10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3, size=(k, d))
    return np.vstack([c + rng.normal(size=(n_per, d)) for c in centers])


def _expr(n=200, g=300, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.6, 1.5, size=g)
    counts = rng.poisson(lam[None, :] * rng.uniform(0.5, 2, size=(n, 1)))
    X = sp.csr_matrix(np.log1p(counts / counts.sum(1, keepdims=True).clip(1) * 1e4))
    feats = [f"g{i}" for i in range(g)]
    groups = np.repeat(["a", "b", "c", "d"], n // 4)
    return X, feats, groups


E = _blobs()
E2 = _blobs(seed=1)[:, :6]
X, FEATS, GROUPS = _expr()
SNN = arsc.find_neighbors(E, k=10)["snn"]
CONN = sx.neighbors(E, n_neighbors=10)["connectivities"]
_NOISE = np.random.default_rng(0).normal(size=(300, 10))  # no cluster structure: partitions depend on the seed
SNN_N = arsc.find_neighbors(_NOISE, k=10)["snn"]
CONN_N = sx.neighbors(_NOISE, n_neighbors=10)["connectivities"]
ZS = np.asarray(X.todense())
ZS = (ZS - ZS.mean(0)) / ZS.std(0).clip(1e-8)
TOY_X, TOY_CLONES = (lambda r: (np.vstack([c + r.normal(size=(60, 5)) for c in r.normal(scale=1.5, size=(6, 5))]),
                                np.where(r.random(360) < 0.3, None, np.repeat([f"c{i}" for i in range(6)], 60))
                                .astype(object)))(np.random.default_rng(0))


def _clone_meta(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in ("P1", "P2", "P3"):
        for t in ("0", "3"):
            n = 300
            cl = rng.zipf(1.6, size=n) % 80
            rows.append(pd.DataFrame({"clone_id": [f"{p}_c{c}" for c in cl], "patient_alias": p, "timepoint": t,
                                      "state": rng.choice(["s1", "s2", "s3"], size=n)}))
    meta = pd.concat(rows, ignore_index=True)
    hd = pd.DataFrame({"clone_id": [f"h{c}" for c in rng.zipf(1.6, size=400) % 90],
                       "timepoint": np.repeat(["T1", "T2"], 200)})
    return meta, hd


META, HD = _clone_meta()
CT_TABLE = C.clone_table(META, "clone_id")


@pytest.fixture(scope="module")
def cp():
    T = ct.cell_transition(TOY_X, 10)
    return ct.propagate_clones(T, TOY_CLONES, n_boot=3)


_SOBS = pd.DataFrame({"b": GROUPS}, index=[f"c{i}" for i in range(len(GROUPS))])
_SREF = arsc.symphony.build_reference(X, FEATS, _SOBS, vars="b", normalize=False, topn=100, d=5, K=4, max_iter=2,
                                      umap=False)
_SQ = arsc.symphony.map_query(X, FEATS, _SOBS, _SREF, vars="b", normalize=False, umap=False)


def _q(cp_):
    return ct.filter_and_sparsify(cp_, 0.3)


# ------------------------------------------------------------------------------------------ cases
# id -> (fn(**seedkw), seed parameter name, different seeds give different output?)
def _cases():
    sigs = {"m1": FEATS[:15], "m2": FEATS[40:60]}
    gs = {f"set{i}": FEATS[i * 20:(i * 20) + 25] for i in range(6)}
    stats_ser = pd.Series(np.random.default_rng(5).normal(size=len(FEATS)), index=FEATS)
    pdf = pd.DataFrame({"patient_alias": np.repeat([f"p{i}" for i in range(8)], 2), "timepoint": ["t0", "t1"] * 8,
                        "value": np.random.default_rng(6).normal(size=16)})
    a12, b12 = np.random.default_rng(7).normal(size=12), np.random.default_rng(8).normal(size=12)
    cc_s, cc_g = FEATS[:20], FEATS[20:40]
    return {
        "pp.pca[arpack]": (lambda **k: arsc.pp.pca(ZS, 10, method="arpack", **k), "seed", False),
        "pp.pca[randomized]": (lambda **k: arsc.pp.pca(ZS, 10, method="randomized", n_iter=0, **k), "seed", True),
        "run_harmony": (lambda **k: arsc.run_harmony(E, np.tile(["x", "y"], len(E) // 2), None, nclust=4, max_iter=3,
                                                     **k), "seed", True),
        "knn[exact]": (lambda **k: arsc.knn(E, 8, **k), "seed", False),
        "knn[pynndescent]": (lambda **k: arsc.knn(_blobs(300, 5, 10), 8, method="pynndescent", **k), "seed", False),
        "find_neighbors": (lambda **k: arsc.find_neighbors(E, k=10, **k), "seed", False),
        "wnn": (lambda **k: arsc.wnn([E, E2], k_nn=10, knn_range=40, **k), "seed", False),
        "cluster[leiden]": (lambda **k: arsc.cluster(SNN_N, **k), "seed", True),
        "cluster[louvain]": (lambda **k: arsc.cluster(SNN_N, method="louvain", n_start=2, **k),
                             "seed", True),
        "umap": (lambda **k: arsc.umap(E, n_neighbors=10, n_epochs=50, **k), "seed", True),
        "scanpy.score_genes": (lambda **k: sx.score_genes(X, FEATS, FEATS[:10], ctrl_size=10, n_bins=10, **k),
                               "random_state", True),
        "scanpy.neighbors": (lambda **k: sx.neighbors(E, n_neighbors=10, **k), "random_state", False),
        "scanpy.leiden[leidenalg]": (lambda **k: sx.leiden(CONN_N, **k), "random_state", True),
        "scanpy.leiden[igraph]": (lambda **k: sx.leiden(CONN_N, flavor="igraph", **k),
                                  "random_state", True),
        "scanpy.umap_from_connectivities": (lambda **k: sx.umap_from_connectivities(CONN, maxiter=50, **k),
                                            "random_state", True),
        "scanpy.diffmap": (lambda **k: sx.diffmap(CONN, n_comps=5, **k), "random_state", False),
        "scanpy.palantir_diffusion_maps": (lambda **k: sx.palantir_diffusion_maps(E, n_components=5, knn=12, **k),
                                           "seed", False),
        "add_module_score": (lambda **k: arsc.add_module_score(X, FEATS, sigs, ctrl=20, nbin=10,
                                                               return_controls=True, **k), "seed", True),
        "cell_cycle_scoring": (lambda **k: arsc.cell_cycle_scoring(X, FEATS, cc_s, cc_g, nbin=10, **k), "seed",
                               True),
        "find_markers": (lambda **k: arsc.find_markers(X, GROUPS, "a", features=FEATS, max_cells_per_ident=20,
                                                       logfc_threshold=0, min_pct=0, **k), "random_seed", True),
        "find_all_markers": (lambda **k: arsc.find_all_markers(X, GROUPS, features=FEATS, max_cells_per_ident=20,
                                                               logfc_threshold=0, min_pct=0, return_thresh=1,
                                                               **k), "random_seed", True),
        "stats.boot_ci": (lambda **k: st.boot_ci(a12, B=500, **k), "seed", True),
        "stats.paired_test": (lambda **k: st.paired_test(pdf, t0="t0", t1="t1", B=50, **k), "seed", True),
        "stats.perm_test": (lambda **k: st.perm_test(a12, b12, B=500, **k), "seed", True),
        "stats.perm_test[exact]": (lambda **k: st.perm_test(a12[:4], b12[:4], B=500, **k), "seed", False),
        "stats.arm_contrast": (lambda **k: st.arm_contrast(a12, b12, B=500, **k), "seed", True),
        "de.fgsea": (lambda **k: arsc.fgsea(stats_ser, gs, nperm=300, **k), "seed", True),
        "clones.clone_expansion_null": (lambda **k: C.clone_expansion_null(META, HD, pairs=((0, 3),),
                                                                           ref_depth=(150, 150), reps=2, **k),
                                        "seed", True),
        "clones.startrac[downsample]": (lambda **k: C.startrac(META, state_col="state", min_cells=1,
                                                               downsample=40, **k), "seed", True),
        "clones.clone_diversity": (lambda **k: C.clone_diversity(CT_TABLE, floor=100, depth=100, reps=3,
                                                                 verbose=False, **k), "seed", True),
        "clonotrace.propagate_clones": (lambda **k: ct.propagate_clones(ct.cell_transition(TOY_X, 10), TOY_CLONES,
                                                                        n_boot=3, **k), "seed", True),
        "symphony.build_reference": (lambda **k: arsc.symphony.build_reference(
            X, FEATS, _SOBS, vars="b", normalize=False, topn=100, d=5, K=4, max_iter=2, umap=False, **k), "seed", True),
        "symphony.transfer_labels": (lambda **k: arsc.symphony.transfer_labels(_SREF, _SQ, GROUPS, k=4, **k), "seed",
                                     False),
        "clonotrace.run": (lambda **k: ct.run(TOY_X, TOY_CLONES, n_boot=3, n_mds=3, cluster_k=3, dpt_k=4,
                                              dpt_nn=3, k=10, revised=True, **k), "seed", True),
    }


CASES = _cases()


@pytest.mark.parametrize("name", list(CASES))
def test_same_seed_identical(name):
    fn, par, _ = CASES[name]
    assert same(fn(**{par: 7}), fn(**{par: 7})), name


@pytest.mark.parametrize("name", list(CASES))
def test_default_call_identical(name):
    fn, _, _ = CASES[name]
    assert same(fn(), fn()), name


@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c[2]])
def test_different_seed_differs(name):
    fn, par, _ = CASES[name]
    outs = [fn(**{par: s}) for s in (1, 2, 3)]
    assert not (same(outs[0], outs[1]) and same(outs[0], outs[2])), name


@pytest.mark.parametrize("name", list(CASES))
def test_none_seed_rejected(name):
    fn, par, _ = CASES[name]
    with pytest.raises(ValueError, match="not allowed"):
        fn(**{par: None})


@pytest.mark.parametrize("name", list(CASES))
def test_global_rng_untouched(name):
    fn, par, _ = CASES[name]
    np.random.seed(12345)
    random.seed(12345)
    np_state, py_state = np.random.get_state(), random.getstate()
    fn(**{par: 3})
    after = np.random.get_state()
    assert all(same(np.asarray(x), np.asarray(y)) for x, y in zip(np_state, after)), "numpy global RNG changed"
    assert random.getstate() == py_state, "python global RNG changed"


# ------------------------------------------------------------------------------------------ specific checks
def test_umap_layouts_seed_controlled():
    """Same seed -> bit-identical UMAP coordinates; seeds 1, 2, 3 -> three different layouts."""
    lay = {s: arsc.umap(E, n_neighbors=10, n_epochs=100, seed=s) for s in (1, 2, 3)}
    assert np.array_equal(lay[1], arsc.umap(E, n_neighbors=10, n_epochs=100, seed=1))
    assert not np.array_equal(lay[1], lay[2]) and not np.array_equal(lay[2], lay[3])
    assert not np.allclose(lay[1], lay[3], atol=1e-3)
    assert np.array_equal(arsc.umap(E, n_neighbors=10, n_epochs=100), arsc.umap(E, n_neighbors=10, n_epochs=100,
                                                                                    seed=42))  # default = 42


def test_scanpy_umap_seed_controlled():
    a = sx.umap_from_connectivities(CONN, maxiter=100, random_state=1)
    assert np.array_equal(a, sx.umap_from_connectivities(CONN, maxiter=100, random_state=1))
    assert not np.allclose(a, sx.umap_from_connectivities(CONN, maxiter=100, random_state=2), atol=1e-3)


def test_cluster_clones_seeded(cp):
    Q = _q(cp)
    D = ct.clone_distance(TOY_X[pd.RangeIndex(len(TOY_X)).get_indexer(Q.index)], Q)
    a, b = ct.cluster_clones(D, k=3, seed=5), ct.cluster_clones(D, k=3, seed=5)
    assert same(a, b) and same(ct.cluster_clones(D, k=3), ct.cluster_clones(D, k=3))
    with pytest.raises(ValueError):
        ct.cluster_clones(D, k=3, seed=None)
    random.seed(1); s0 = random.getstate()
    ct.cluster_clones(D, k=3, seed=9)
    assert random.getstate() == s0


def test_revise_distance_seeded(cp):
    Q = _q(cp)
    Xq = TOY_X[pd.RangeIndex(len(TOY_X)).get_indexer(Q.index)]
    D = ct.clone_distance(Xq, Q)
    r1 = ct.revise_distance(D, Xq, Q, seed=4, resolutions=(0.5, 1.0), cluster_k=8)
    r2 = ct.revise_distance(D, Xq, Q, seed=4, resolutions=(0.5, 1.0), cluster_k=8)
    assert same(r1, r2)
    with pytest.raises(ValueError):
        ct.revise_distance(D, Xq, Q, seed=None)


def test_profile_enrichment_and_fate_genes_seeded(cp):
    Q = _q(cp)
    cl = pd.Series([1, 1, 2, 2, 3, 3][:Q.shape[1]], index=[str(c) for c in Q.columns])
    cpp, _ = ct.cell_profiles(Q, cl)
    lab = np.random.default_rng(0).integers(0, 3, size=len(cpp))
    pe = lambda **k: ct.profile_enrichment(cpp, lab, n_perm=50, **k)  # noqa: E731
    assert same(pe(seed=2), pe(seed=2)) and same(pe(), pe())
    assert not same(pe(seed=1), pe(seed=2))
    rng = np.random.default_rng(1)
    n = len(cpp)
    G = sp.csr_matrix(rng.poisson(2.0, size=(n, 40)).astype(float))
    t = rng.random(n)
    fg = lambda **k: ct.fate_genes(1, "0", G, [f"g{i}" for i in range(40)], np.zeros(n, int), t, cpp,  # noqa: E731
                                   n_perm=5, min_cells=5, mass_thresh=0, binsize=0.05, **k)
    a = fg(seed=3)
    assert a is not None and same(a, fg(seed=3)) and same(fg(), fg())
    assert not same(a, fg(seed=4))
    for f in (pe, fg):
        with pytest.raises(ValueError):
            f(seed=None)


def test_exact_knn_independent_of_seed():
    """The seed of the exact kNN only steers pruning; the neighbours are exact for every seed."""
    a = arsc.knn(E, 12, seed=0)
    for s in (1, 99):
        b = arsc.knn(E, 12, seed=s)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_harmony_thread_count_invariant():
    """Fixed reduction chunks: harmony is bit-identical with 1 thread and with all threads."""
    import numba as nb
    prev = nb.get_num_threads()
    batch = np.tile(["x", "y"], len(E) // 2)
    try:
        a = arsc.run_harmony(E, batch, None, nclust=4, max_iter=3, seed=2, n_threads=1)
        b = arsc.run_harmony(E, batch, None, nclust=4, max_iter=3, seed=2, n_threads=prev)
    finally:
        nb.set_num_threads(prev)
    assert np.array_equal(a, b)


def test_score_genes_matches_global_seed_stream():
    """The local RandomState draws exactly what scanpy's np.random.seed + Series.sample would."""
    gl = FEATS[:10]
    s, ctrl = sx.score_genes(X, FEATS, gl, ctrl_size=10, n_bins=10, random_state=11, return_controls=True)
    # emulate scanpy: reseed the *global* RNG, then sample without random_state
    import arsc.scanpy_extra as mod
    np.random.seed(11)
    pos = pd.Series(np.arange(len(FEATS)), index=pd.Index(FEATS).astype("string"))
    mu = mod._col_means(X)
    obs_avg = pd.Series(mu, index=pos.index)
    n_items = int(np.round(len(obs_avg) / 9))
    cut = obs_avg.rank(method="min") // n_items
    ref = pd.Index([], dtype="string")
    for c in np.unique(cut.loc[gl]):
        r = cut[cut == c].index
        if 10 < len(r):
            r = r.to_series().sample(10).index
        ref = ref.union(r.difference(gl))
    assert ctrl.equals(ref)


def test_seed_type_validation():
    with pytest.raises(TypeError):
        arsc.cluster(SNN, seed=1.5)
    with pytest.raises(ValueError):
        arsc.cluster(SNN, seed=-1)
    arsc.add_module_score(X, FEATS, {"m": FEATS[:10]}, ctrl=10, nbin=10, seed=-5)  # R seeds may be negative
