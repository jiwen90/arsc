"""arsc.symphony: Symphony reference mapping on tiny synthetic data (in memory), plus the R parity check.

The parity test reads R symphony outputs written by benchmarks/r_reference_symphony.R into
$ARSC_REFS/symphony_ref and skips when they are absent (SYMPHONY_PARITY.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import arsc
from arsc import symphony as sy


def _synthetic(n_per=120, n_genes=300, types=("A", "B", "C"), batches=("b1", "b2", "q1"), seed=0):
    """Counts with 3 cell types (20 marker genes each) and a multiplicative per-gene batch effect."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0.0, 1.0, n_genes) * 0.5
    rows, ty, ba = [], [], []
    for b in batches:
        shift = rng.lognormal(0.0, 0.4, n_genes)
        for t_i, t in enumerate(types):
            mu = base * shift
            mu[t_i * 20:(t_i + 1) * 20] *= 8.0
            rows.append(rng.poisson(mu, size=(n_per, n_genes)))
            ty += [t] * n_per
            ba += [b] * n_per
    X = sp.csr_matrix(np.vstack(rows).astype(np.float64))
    obs = pd.DataFrame({"cell_type": ty, "batch": ba}, index=[f"c{i}" for i in range(X.shape[0])])
    genes = pd.Index([f"G{i}" for i in range(n_genes)])
    return X, genes, obs


@pytest.fixture(scope="module")
def data():
    X, genes, obs = _synthetic()
    isref = (obs["batch"] != "q1").to_numpy()
    ref = sy.build_reference(X[isref], genes, obs[isref], vars="batch", topn=150, d=10, K=12, umap=False)
    return X, genes, obs, isref, ref


def test_moe_correct_ref_matches_dense_loop():
    rng = np.random.default_rng(1)
    N, d, K = 80, 5, 4
    Zq = rng.normal(size=(N, d))
    Rq = rng.random((K, N)); Rq /= Rq.sum(0)
    Nr = rng.random(K) * 50 + 5
    C = rng.normal(size=(K, d)) * 10
    obs = pd.DataFrame({"a": rng.choice(["x", "y", "z"], N), "b": rng.choice(["u", "v"], N)})
    pats, pat_of = sy._design(obs, ["a", "b"], N)
    got = sy._moe_correct_ref(Zq, pats, pat_of, Rq, Nr, C)
    Xq = pats[pat_of].T  # (B+1) x N, symphony's Xq
    ref = Zq.T.copy()
    for k in range(K):
        XR = Xq * Rq[k]
        m1 = XR @ Xq.T
        m1[0, 0] += Nr[k]
        m1 += np.diag(np.r_[0.0, np.ones(Xq.shape[0] - 1)])
        m2 = XR @ Zq
        m2[0] += C[k]
        beta = np.linalg.solve(m1, m2)
        beta[0] = 0
        ref -= beta.T @ XR
    assert np.abs(got - ref.T).max() < 1e-10


def test_scaling_clips_both_tails_and_soft_cluster_normalises():
    E = np.array([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0], [10.0, 0.0]])
    S = sy._scale_with_stats(E, E.mean(0), E.std(0, ddof=1), thresh=1.0)
    assert S.max() == 1.0 and S.min() == -1.0
    rng = np.random.default_rng(2)
    R = sy._soft_cluster(sy._l2_rows(rng.normal(size=(6, 4))), sy._l2_rows(rng.normal(size=(30, 4))), 0.1)
    assert R.shape == (6, 30) and np.allclose(R.sum(0), 1.0)


def test_reference_structure(data):
    X, genes, obs, isref, ref = data
    n = int(isref.sum())
    assert ref.Z_corr.shape == (n, 10) and ref.R.shape == (12, n) and ref.loadings.shape == (150, 10)
    assert np.allclose(ref.R.sum(0), 1.0)
    assert np.allclose(np.linalg.norm(ref.centroids, axis=1), 1.0)
    assert np.allclose(ref.Nr, ref.R.sum(1)) and np.allclose(ref.C, ref.R @ ref.Z_corr)


def test_reference_cells_project_onto_their_own_pcs(data):
    X, genes, obs, isref, ref = data
    q = sy.map_query(X[isref], genes, obs[isref], ref, vars=None, umap=False)
    assert np.abs(q.Z_pca - ref.Z_orig).max() < 1e-8  # same scaling + loadings -> the reference PCs


def test_query_maps_to_right_types_and_batch_shrinks(data):
    X, genes, obs, isref, ref = data
    q = sy.map_query(X[~isref], genes, obs[~isref], ref, vars="batch", umap=False)
    lab = sy.transfer_labels(ref, q, "cell_type", k=5)
    truth = obs.loc[~isref, "cell_type"].to_numpy()
    assert (lab["label"].to_numpy() == truth).mean() > 0.95
    assert lab["prob"].between(0, 1).all()
    # the correction moves query cells toward reference cells of the same type
    yr = obs.loc[isref, "cell_type"].to_numpy()
    def gap(Zq):
        return np.mean([np.linalg.norm(Zq[truth == t].mean(0) - ref.Z_corr[yr == t].mean(0)) for t in "ABC"])
    assert gap(q.Z) < gap(q.Z_pca)
    ms = sy.mapping_score(ref, q)
    assert ms.shape == (int((~isref).sum()),) and np.isfinite(ms).all()


def test_missing_query_genes_and_from_harmony(data, tmp_path):
    X, genes, obs, isref, ref = data
    keep = np.arange(len(genes)) % 10 != 0  # drop 10 % of the genes from the query
    q = sy.map_query(X[~isref][:, keep], genes[keep], obs[~isref], ref, vars="batch", umap=False)
    assert q.n_shared_genes == int(np.isin(ref.genes, genes[keep]).sum()) < len(ref.genes)
    r2 = sy.reference_from_harmony(ref.Z_orig, ref.Z_corr, ref.R.T, ref.genes, ref.mean, ref.sd, ref.loadings,
                                   ref.obs, umap=False)
    assert np.allclose(r2.centroids, ref.centroids) and np.allclose(r2.C, ref.C)
    r3 = sy.load_reference(ref.save(str(tmp_path / "ref")))
    assert list(r3.genes) == list(ref.genes) and np.array_equal(r3.R, ref.R)


def _no_batch(umap):
    X, genes, obs = _synthetic(n_per=60, batches=("b1", "q1"), seed=3)
    isref = (obs["batch"] == "b1").to_numpy()
    ref = sy.build_reference(X[isref], genes, obs[isref], vars=None, topn=150, d=8, K=6, umap=umap, seed=1)
    q = sy.map_query(X[~isref], genes, obs[~isref], ref, vars="batch", umap=umap)
    return X, obs, isref, ref, q


def test_no_batch_reference():
    X, obs, isref, ref, q = _no_batch(False)
    assert np.allclose(ref.R.sum(0), 1.0) and np.array_equal(ref.Z_corr, ref.Z_orig)
    lab = sy.transfer_labels(ref, q, "cell_type", k=5)
    assert (lab["label"].to_numpy() == obs.loc[~isref, "cell_type"].to_numpy()).mean() > 0.9


@pytest.mark.slow  # synthetic, but umap-learn's numba compilation takes ~40 s
def test_umap_model_places_query():
    X, obs, isref, ref, q = _no_batch(True)
    assert ref.umap.shape == (int(isref.sum()), 2) and q.umap.shape == (int((~isref).sum()), 2)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from refutil import SYMPHONY_REF as REF, missing  # noqa: E402

_SKIP = missing(REF / "q_Z.csv", regen="symphony_ref")


@pytest.mark.slow
@pytest.mark.skipif(bool(_SKIP), reason=_SKIP or "")
def test_parity_with_r_given_r_reference():
    """Deterministic steps against R symphony 0.1.1 on its vignette data (reference 3pv1+3pv2, query 5p)."""
    import scipy.io
    r = lambda f: np.loadtxt(REF / f, delimiter=",", ndmin=2)  # noqa: E731
    X = scipy.io.mmread(REF / "pbmcs_exprs.mtx").tocsr().T.tocsr()
    genes = pd.Index((REF / "pbmcs_genes.txt").read_text().split())
    meta = pd.read_csv(REF / "pbmcs_meta.csv", index_col=0)
    isref = meta["donor"].isin(["3pv1", "3pv2"]).to_numpy()
    ms = r("ref_meansd.csv")
    ref = sy.reference_from_harmony(r("ref_Zorig.csv").T, r("ref_Zcorr.csv").T, r("ref_R.csv"),
                                    (REF / "ref_vargenes.txt").read_text().split(), ms[:, 0], ms[:, 1],
                                    r("ref_loadings.csv"), meta[isref], umap=False)
    assert np.abs(ref.centroids - r("ref_centroids.csv").T).max() < 1e-10
    q = sy.map_query(X[~isref], genes, meta[~isref], ref, vars="donor", normalize=False, umap=False)
    assert np.abs(q.Z_pca - r("q_Zpca.csv").T).max() < 1e-9
    assert np.abs(q.R - r("q_R.csv")).max() < 1e-10
    assert np.abs(q.Z - r("q_Z.csv").T).max() < 1e-9
    knn = pd.read_csv(REF / "q_knn.csv")
    lab = sy.transfer_labels(ref, q, "cell_type", k=5)
    assert (lab["label"].to_numpy() == knn["pred"].to_numpy()).all()
    assert np.abs(sy.mapping_score(ref, q) - r("q_maha.csv").ravel()).max() < 1e-6
