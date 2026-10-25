"""arsc's kNN: the exact method (clustered search with proven pruning, float64 distances) equals float64 brute
force (ties, duplicates, odd dimensions, external queries, large coordinates, any partition); the approximate method
(the same kernel scanning each query's nearest clusters) finds nearly all true neighbours, reports exact distances and
is reproducible with a seed; "auto" switches at 4,096 points like scanpy; the SNN graph matches its definition."""
import numpy as np
import pytest
import scipy.sparse as sp

import arsc
from arsc import neighbors as N


def _mix(n, d=12, seed=0, noise_dims=6, offset=0.0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 5, (8, d))
    mu[:, d - noise_dims:] = 0
    X = mu[rng.integers(0, 8, n)] + rng.normal(size=(n, d))
    return X + offset


def _brute(X, Q, k):
    """(distance, index)-ordered brute force with the same float64 sum of squared differences."""
    D = np.zeros((Q.shape[0], X.shape[0]))
    for t in range(X.shape[1]):
        D += (Q[:, t:t + 1] - X[None, :, t]) ** 2
    order = np.lexsort((np.broadcast_to(np.arange(X.shape[0]), D.shape), D))[:, :k]
    return order, np.sqrt(np.take_along_axis(D, order, 1))


@pytest.mark.parametrize("d,offset", [(12, 0.0), (7, 0.0), (5, 1e4)])
def test_exact_equals_brute_force(d, offset):
    X = _mix(2500, d=d, seed=d, noise_dims=min(4, d - 1), offset=offset)
    i, dist = arsc.knn(X, 10, method="exact")
    ri, rd = _brute(X, X, 10)
    assert np.array_equal(i, ri)
    assert np.allclose(dist, rd, rtol=1e-12, atol=1e-12)


def test_exact_ties_duplicates_and_queries():
    rng = np.random.default_rng(3)
    X = np.round(rng.normal(size=(3000, 4)), 1)
    X[1500:1550] = X[:50]                                   # exact duplicates
    i, d = arsc.knn(X, 8, method="exact")
    assert (i[:, 0] == np.arange(3000)).all()               # self first even when tied with a duplicate
    Q = np.round(rng.normal(size=(400, 4)), 1)
    qi, qd = arsc.knn(X, 8, query=Q, method="exact")
    ri, rd = _brute(X, Q, 8)
    assert np.array_equal(qi, ri) and np.allclose(qd, rd, rtol=1e-12, atol=1e-12)


def test_approx_recall_exact_distances_and_seed():
    X = _mix(30000, d=20, seed=5, noise_dims=12)
    i, d = arsc.knn(X, 15, method="approximate", seed=0)
    i2, d2 = arsc.knn(X, 15, method="approximate", seed=0)
    assert np.array_equal(i, i2) and np.array_equal(d, d2)  # same seed, same graph
    qs = np.random.default_rng(0).choice(X.shape[0], 300, replace=False)
    ri, _ = _brute(X, X[qs], 15)
    rec = np.mean([len(set(i[q]) & set(ri[a])) / 15 for a, q in enumerate(qs)])
    assert rec > 0.95, rec
    # the distances reported are the exact ones for the neighbours found
    exact = np.sqrt(((X[qs, None, :] - X[i[qs]]) ** 2).sum(-1))
    assert np.allclose(d[qs], exact, rtol=1e-12)
    assert np.all(np.diff(d, axis=1) >= 0) and (i[:, 0] == np.arange(X.shape[0])).all()
    i3, _ = arsc.knn(X, 15, method="approximate", seed=1)  # another seed: nearly the same graph
    assert np.mean([len(set(a) & set(b)) for a, b in zip(i, i3)]) / 15 > 0.95
    with pytest.warns(DeprecationWarning):   # the old backend name maps to the approximate method
        assert all(np.array_equal(p, q) for p, q in zip(arsc.knn(X, 15, method="pynndescent"), (i, d)))


def test_exact_any_partition_and_short_clusters():
    X = _mix(2000, d=9, seed=3)
    ri, rd = _brute(X, X, 12)
    for nc, seed in [(1, 0), (7, 1), (45, 2), (400, 3), (2000, 4)]:   # the partition changes speed, never the answer
        i, d = arsc.knn(X, 12, method="exact", n_clusters=nc, seed=seed)
        assert np.array_equal(i, ri) and np.array_equal(d, rd), (nc, seed)
    # approximate search over clusters smaller than k: rows that found fewer than k points are completed exactly
    i, d = arsc.knn(X, 12, method="approximate", n_clusters=1500, seed=0)
    assert (i >= 0).all() and np.all(np.diff(d, axis=1) >= 0) and (i[:, 0] == np.arange(2000)).all()
    exact = np.sqrt(((X[:, None, :] - X[i]) ** 2).sum(-1))
    assert np.allclose(d, exact, rtol=1e-12)


def test_auto_switch(monkeypatch):
    X = _mix(3000, d=8, seed=1)
    e = arsc.knn(X, 10, method="exact")
    assert all(np.array_equal(a, b) for a, b in zip(arsc.knn(X, 10), e))   # n <= AUTO_EXACT_MAX: exact
    monkeypatch.setattr(N, "AUTO_EXACT_MAX", 1000)
    a = arsc.knn(X, 10)
    b = arsc.knn(X, 10, method="approximate")
    assert all(np.array_equal(p, q) for p, q in zip(a, b))                 # above it: approximate
    q = arsc.knn(X, 10, query=X[:50])                                       # queries stay exact
    assert all(np.array_equal(p, r) for p, r in zip(q, arsc.knn(X, 10, query=X[:50], method="exact")))
    with pytest.raises(ValueError):
        arsc.knn(X, 10, method="nope")
    monkeypatch.undo()
    assert N.AUTO_EXACT_MAX == 4096  # scanpy.pp.neighbors switches to approximate kNN above 4,096 cells


def test_snn_graph_matches_reference_definition():
    X = _mix(1500, d=6, seed=2)
    idx, _ = arsc.knn(X, 12)
    G = arsc.snn_graph(idx, prune=1 / 15)
    A = sp.csr_matrix((np.ones(idx.size), (np.repeat(np.arange(1500), 12), idx.ravel())), shape=(1500, 1500))
    S = (A @ A.T).toarray()
    J = S / (24 - S)
    J[J < 1 / 15] = 0
    assert G.has_sorted_indices
    assert np.allclose(G.toarray(), J, rtol=0, atol=1e-15)
    # prune 0 keeps every pair with an overlap: more pairs per row than the first-pass buffer holds
    G0 = arsc.snn_graph(idx, prune=0.0)
    J0 = S / (24 - S)
    assert G0.has_sorted_indices and np.allclose(G0.toarray(), J0, rtol=0, atol=1e-15)
