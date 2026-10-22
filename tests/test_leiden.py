"""arsc's own Leiden / Louvain (arsc._leiden, arsc.cluster(engine="arsc")): planted partitions recovered,
quality at least the reference's (leidenalg / igraph) over seeds, identical labels at any thread count."""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import arsc
from arsc import _leiden as L


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def _planted(n_blocks=6, size=150, p_in=0.08, p_out=0.004, seed=0):
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    truth = np.repeat(np.arange(n_blocks), size)
    P = np.where(truth[:, None] == truth[None, :], p_in, p_out)
    U = np.triu(rng.random((n, n)) < P, 1)
    W = U * rng.uniform(0.5, 1.5, (n, n))
    A = sp.csr_matrix(W + W.T)
    return A, truth


def _snn(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(c, 1.0, (n // 5, 8)) for c in (0, 3, 6, 9, 12)])
    return arsc.find_neighbors(X, k=15)["snn"]


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_planted_partition_recovered(method):
    A, truth = _planted()
    lab = arsc.cluster(A, method=method, resolution=1.0, seed=0, engine="arsc")
    assert _ari(lab, truth) > 0.95


def test_cpm_planted():
    A, truth = _planted(p_in=0.2, p_out=0.002)
    lab = L.leiden(A, resolution=0.05, objective="cpm", seed=0)
    assert _ari(lab, truth) > 0.95


def test_quality_matches_igraph_and_leidenalg():
    ig = pytest.importorskip("igraph")
    la = pytest.importorskip("leidenalg")
    A, _ = _planted(seed=3)
    lab = np.random.default_rng(0).integers(0, 9, A.shape[0])
    U = sp.triu(A, 1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=np.column_stack([U.row, U.col]), directed=False)
    g.es["weight"] = U.data
    assert L.quality(A, lab, 0.7) == pytest.approx(g.modularity(lab.tolist(), weights="weight", resolution=0.7),
                                                  rel=1e-10)
    p = la.CPMVertexPartition(g, initial_membership=lab.tolist(), weights="weight", resolution_parameter=0.05)
    assert L.quality(A, lab, 0.05, "cpm") == pytest.approx(p.quality(), rel=1e-10)


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_quality_not_below_reference(method):
    pytest.importorskip("leidenalg")
    G = _snn()
    ours = [L.quality(G, arsc.cluster(G, method=method, resolution=0.8, seed=s, engine="arsc",
                                      group_singletons=False), 0.8) for s in range(5)]
    ref = [L.quality(G, arsc.cluster(G, method=method, resolution=0.8, seed=s, engine="leidenalg",
                                     group_singletons=False), 0.8) for s in range(5)]
    assert np.median(ours) >= min(ref) - 1e-3


def test_cpm_quality_not_below_leidenalg():
    pytest.importorskip("leidenalg")
    G = _snn()
    ours = [L.quality(G, arsc.cluster(G, method="leiden", resolution=0.05, seed=s, engine="arsc", objective="cpm",
                                      group_singletons=False), 0.05, "cpm") for s in range(8)]
    ref = [L.quality(G, arsc.cluster(G, method="leiden", resolution=0.05, seed=s, engine="leidenalg", objective="cpm",
                                     group_singletons=False), 0.05, "cpm") for s in range(8)]
    assert np.median(ours) >= min(ref)  # CPM optima vary by about 1% between seeds in both


def test_n_iterations_until_stable_not_worse():
    G = _snn(seed=2)
    q2 = L.quality(G, L.leiden(G, 1.0, seed=0, n_iterations=2), 1.0)
    qm = L.quality(G, L.leiden(G, 1.0, seed=0, n_iterations=-1), 1.0)
    assert qm >= q2 - 1e-12


def test_seed_reproducible():
    G = _snn(seed=1)
    a = arsc.cluster(G, resolution=2.0, seed=0, engine="arsc")
    b = arsc.cluster(G, resolution=2.0, seed=0, engine="arsc")
    assert np.array_equal(a, b)


def test_engine_validation():
    G = _snn(n=500)
    with pytest.raises(ValueError):
        arsc.cluster(G, engine="nope")
    with pytest.raises(ValueError):
        arsc.cluster(G, method="louvain", objective="cpm", engine="arsc")


def test_group_singletons_matches_seurat_rule():
    G = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 3, 0.]]))
    # labels 0 = {0, 1}, 1 = {2, 3}: no singleton -> unchanged
    assert L.group_singletons(np.array([0, 0, 1, 1]), G).tolist() == [0, 0, 1, 1]
    # node 3 alone joins the cluster of its only neighbour (node 2, cluster 0)
    out = L.group_singletons(np.array([0, 0, 0, 1]), G)
    assert out.tolist() == [0, 0, 0, 0]


_DET = r"""
import hashlib, numpy as np, numba, arsc
rng = np.random.default_rng(0)
X = np.concatenate([rng.normal(c, 1.0, (600, 6)) for c in range(5)])
G = arsc.find_neighbors(X, k=15)["snn"]
out = []
for t in (1, 2, 4, 8):
    numba.set_num_threads(t)
    h = hashlib.sha1()
    for m in ("leiden", "louvain"):
        h.update(arsc.cluster(G, method=m, resolution=1.0, seed=3, engine="arsc").tobytes())
    out.append(h.hexdigest())
print(" ".join(out))
"""


def test_identical_across_thread_counts():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_THREADS") and k != "AR_THREADS"}
    env.update(NUMBA_NUM_THREADS="8", AR_THREADS="8", PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(arsc.__file__))] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    r = subprocess.run([sys.executable, "-c", _DET], capture_output=True, text=True, env=env, check=True)
    digests = r.stdout.split()
    assert len(digests) == 4 and len(set(digests)) == 1, r.stdout + r.stderr
