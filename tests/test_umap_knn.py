"""arsc's own UMAP engine (``arsc._umap``) and the exact kNN: agreement with umap-learn where the arithmetic is
the same (fuzzy graph, a/b curve, spectral initialization), layout determinism across thread counts, and
scan-order-independent kNN ties."""
import numpy as np
import pytest
import scipy.sparse as sp

import arsc
from arsc import _umap


def _blobs(n=1500, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(k), n // k)
    return rng.normal(size=(lab.size, d)) + 6.0 * np.eye(k, d)[lab], lab


def test_fuzzy_graph_and_ab_match_umap_learn():
    umap_ = pytest.importorskip("umap.umap_")
    X, _ = _blobs()
    idx, d = arsc.knn(X, 15)
    G = arsc.fuzzy_simplicial_set(idx, d)
    R = sp.csr_matrix(umap_.fuzzy_simplicial_set(sp.coo_matrix((X.shape[0], 1)), 15, None, None,
                                                 knn_indices=idx, knn_dists=d)[0])
    G.sort_indices()
    R.sort_indices()
    assert np.array_equal(G.indptr, R.indptr) and np.array_equal(G.indices, R.indices)
    assert np.array_equal(G.data, R.data)  # bit-identical float32 weights
    for md in (0.1, 0.3, 0.5):
        assert np.allclose(_umap.find_ab_params(1.0, md), umap_.find_ab_params(1.0, md), rtol=1e-12)


def test_spectral_init_matches_umap_learn():
    spectral = pytest.importorskip("umap.spectral")
    from sklearn.utils import check_random_state
    X, _ = _blobs(3000, 4, 1)
    G = arsc.fuzzy_simplicial_set(*arsc.knn(X, 15))
    ours = _umap.spectral_init(G, 2, X=X, seed=0)
    ref = spectral.spectral_layout(X, sp.coo_matrix(G), 2, check_random_state(0))
    for j in range(2):
        assert abs(np.corrcoef(ours[:, j], ref[:, j])[0, 1]) > 0.999


def test_layout_identical_across_threads_and_separates_clusters():
    X, lab = _blobs()
    G = arsc.fuzzy_simplicial_set(*arsc.knn(X, 15))
    n0 = arsc.get_threads()
    outs = []
    try:
        for th in (1, 2, 3):
            arsc.set_threads(th)
            outs.append(_umap.embed(G, seed=7, X=X, n_epochs=120))
    finally:
        arsc.set_threads(n0)
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    other = _umap.embed(G, seed=8, X=X, n_epochs=120)
    assert not np.array_equal(outs[0], other)
    Y = outs[0]
    cent = np.array([Y[lab == c].mean(0) for c in range(3)])
    spread = max(np.linalg.norm(Y[lab == c] - cent[c], axis=1).mean() for c in range(3))
    gaps = [np.linalg.norm(cent[a] - cent[b]) for a in range(3) for b in range(a + 1, 3)]
    assert min(gaps) > 3 * spread


def test_umap_engine_switch():
    X, _ = _blobs(600)
    a = arsc.umap(X, n_neighbors=10, metric="euclidean", seed=1, n_epochs=40, engine="arsc")
    b = arsc.umap(X, n_neighbors=10, metric="euclidean", seed=1, n_epochs=40, engine="arsc",
                  knn=arsc.knn(X, 10))
    assert a.shape == (600, 2) and np.array_equal(a, b)
    with pytest.raises(ValueError):
        arsc.umap(X, engine="nope")


def test_several_components_are_laid_out():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(0, 1, (300, 3)), rng.normal(50, 1, (300, 3)), rng.normal(-50, 1, (5, 3))]
    G = arsc.fuzzy_simplicial_set(*arsc.knn(X, 5))
    Y = _umap.embed(G, seed=0, X=X, n_epochs=50)
    assert np.isfinite(Y).all() and Y.shape == (605, 2)


def test_knn_ties_resolve_by_index_whatever_the_scan_order():
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(3000, 3)), 1)  # many exact distance ties
    n0 = arsc.get_threads()
    outs = set()
    try:
        for th in (1, 2):
            arsc.set_threads(th)
            for s in (0, 1, 2):
                i, d = arsc.knn(X, 8, seed=s, n_clusters=[20, 60, 110][s])
                outs.add(i.tobytes() + d.tobytes())
    finally:
        arsc.set_threads(n0)
    assert len(outs) == 1
    D = ((X[:300, None, :] - X[None, :, :]) ** 2).sum(-1)
    ref = np.lexsort((np.broadcast_to(np.arange(X.shape[0]), D.shape), D))[:, :8]
    i, _ = arsc.knn(X, 8)
    # the point itself first, then (distance, index) order
    assert np.array_equal(i[:300, 1:], np.array([[j for j in r if j != q][:7] for q, r in enumerate(ref)]))


def test_defaults_are_arsc_engines_and_cd8_workflow_settings():
    """0.3.0 defaults: arsc's own UMAP / Leiden-Louvain engines, Louvain at 0.6, 30 PCs; the reference engines
    stay selectable (CHANGELOG 0.3.0)."""
    import inspect

    import arsc
    sig = {f: inspect.signature(getattr(arsc, f)).parameters for f in ("cluster", "umap")}
    assert sig["cluster"]["engine"].default == "arsc" and sig["umap"]["engine"].default == "arsc"
    assert sig["cluster"]["method"].default == "louvain" and sig["cluster"]["resolution"].default == 0.6
    assert sig["cluster"]["seed"].default == 0 and sig["umap"]["seed"].default == 42
    assert (sig["umap"]["n_neighbors"].default, sig["umap"]["min_dist"].default, sig["umap"]["metric"].default,
            sig["umap"]["spread"].default) == (30, 0.3, "cosine", 1.0)
    assert inspect.signature(arsc.pp.pca).parameters["n_comps"].default == 30
    assert inspect.signature(arsc.pp.hvg_vst).parameters["nfeatures"].default == 2000
    assert inspect.signature(arsc.pp.log_normalize).parameters["scale_factor"].default == 1e4
    assert inspect.signature(arsc.find_neighbors).parameters["k"].default == 20
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, (150, 6)) for c in (0.0, 6.0, 12.0)])
    g = arsc.find_neighbors(X, k=20)["snn"]
    assert np.array_equal(arsc.cluster(g), arsc.cluster(g, method="louvain", resolution=0.6, engine="arsc"))
    pytest.importorskip("igraph")
    ref = arsc.cluster(g, method="louvain", resolution=0.6, engine="leidenalg")
    assert len(np.unique(ref)) == len(np.unique(arsc.cluster(g)))
    a = arsc.umap(X, n_neighbors=10, n_epochs=30)
    assert np.array_equal(a, arsc.umap(X, n_neighbors=10, n_epochs=30, engine="arsc"))
