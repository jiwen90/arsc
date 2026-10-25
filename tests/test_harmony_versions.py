"""run_harmony versions (R harmony 1.2 / 2.0 / 2.1): defaults, argument checks and the algorithmic differences.

The numerical agreement with each R version is in tests/test_vs_r.py::test_harmony_versions_track_r.
"""
import numpy as np
import pandas as pd
import pytest

import arsc
from arsc.harmony import Harmony


def _data(n=600, d=8, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    a = np.tile(["a1", "a2", "a3"], n // 3)
    b = np.repeat(["b1", "b2"], n // 2)
    Z[a == "a2"] += 1.5  # a batch shift to correct
    return Z, pd.DataFrame({"a": a, "b": b})


def test_version_names_and_errors():
    Z, meta = _data()
    out = {v: arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=2, seed=1, version=v)
           for v in ["1.2", "1.2.0", "2.0", "2.0.5", "2.1", "2.1.0"]}
    assert np.array_equal(out["1.2"], out["1.2.0"]) and np.array_equal(out["2.0"], out["2.0.5"])
    assert np.array_equal(out["2.1"], out["2.1.0"])
    assert not np.allclose(out["1.2"], out["2.0"])
    with pytest.raises(ValueError, match="version"):
        arsc.run_harmony(Z, meta, "a", nclust=5, version="3")
    with pytest.raises(ValueError, match="lambda_"):
        arsc.run_harmony(Z, meta, "a", nclust=5, lambda_="auto")


def test_defaults_follow_the_version():
    """1.2: lambda 1, 20 clustering iterations; 2.x: automatic lambda (alpha * E), 4 clustering iterations."""
    Z, meta = _data()
    for v, lam, iters in [("1.2", 1.0, 20), ("2.0", None, 4)]:
        a = arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=2, seed=1, version=v)
        b = arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=2, seed=1, version=v, lambda_=lam,
                             max_iter_cluster=iters)
        assert np.array_equal(a, b)
    res = arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=3, seed=1, version="2.0", early_stop=False,
                           return_object=True)
    assert res.kmeans_rounds == [4, 4, 4]  # 4 iterations never reach the convergence check (iteration > 3)


def test_v2_block_size_one_is_allowed():
    """harmony 2.x's pseudocounts keep E / O finite with one block; 1.2 rejects it (R returns NaN)."""
    Z, meta = _data()
    out = arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=2, seed=1, version="2.0", block_size=1.0)
    assert np.isfinite(out).all()
    with pytest.raises(ValueError, match="block_size"):
        arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=2, seed=1, version="1.2", block_size=1.0)


def _first_R(Z, codes, B_vec, theta, version, Y0):
    h = Harmony(Z, codes, B_vec, theta=theta, sigma=np.full(5, 0.1), lambda_vec=None, K=5, seed=3,
                version=version, max_iter_cluster=4)
    h.init_cluster(init_centroids=Y0)
    h.cluster()
    return h.R


@pytest.mark.parametrize("version,same", [("2.1", True), ("2.0", False)])
def test_theta_zero_covariate(version, same):
    """harmony 2.1 multiplies the per-variable penalties, so a variable with theta = 0 leaves the cluster
    assignment unchanged (R harmony 2.1.0's own unit test); 2.0 (and 1.2) sum them, so it does not."""
    Z, meta = _data()
    ca = pd.Categorical(meta["a"]).codes.astype(np.int64)
    cb = pd.Categorical(meta["b"]).codes.astype(np.int64) + 3
    Y0 = Z[[0, 100, 200, 300, 400]]
    r1 = _first_R(Z, ca[:, None], [3], np.full(3, 2.0), version, Y0)
    r2 = _first_R(Z, np.stack([ca, cb], 1), [3, 2], np.r_[np.full(3, 2.0), np.zeros(2)], version, Y0)
    assert np.allclose(r1, r2, atol=1e-12) == same


def test_v2_drops_absent_batches_from_a_cluster():
    """A batch level with (almost) no cells in a cluster is left out of that cluster's regression (2.x
    batch_prop_cutoff); a cutoff of 1 leaves every cluster uncorrected, so the output is the input."""
    Z, meta = _data()
    out = arsc.run_harmony(Z, meta, "a", nclust=5, max_iter=1, seed=1, version="2.0", batch_prop_cutoff=1.0)
    assert np.allclose(out, Z)
