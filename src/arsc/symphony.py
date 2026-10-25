"""Symphony reference mapping -- a port of R ``symphony`` 0.1.1 (Kang et al. 2021, Nat Commun 12:5890; GPL-3).

Build a compressed reference once (log-normalised expression -> variable genes -> scaling with the reference's
own means and s.d.s -> PCA -> Harmony on the reference's batch variables), then map any query onto it without
touching the reference cells again: the query is scaled with the reference's means/s.d.s, projected onto the
reference loadings, soft-assigned to the reference's Harmony centroids and batch-corrected by the mixture-of-experts
ridge regression anchored on the reference's per-cluster terms (``Nr``, ``C``). Labels move across by k nearest
neighbours in the corrected space, with a vote-share confidence; query cells can also be placed on the
reference UMAP.

  ref = arsc.symphony.build_reference(counts_ref, genes, obs_ref, vars="donor")      # buildReference
  q   = arsc.symphony.map_query(counts_q, genes_q, obs_q, ref, vars="donor")          # mapQuery
  lab = arsc.symphony.transfer_labels(ref, q, "cell_type", k=5)                       # knnPredict
  score = arsc.symphony.mapping_score(ref, q)                                         # calcPerCellMappingMetric

Orientation: arsc arrays are cells first (``Z_corr`` is cells x d, ``loadings`` genes x d, ``centroids`` K x d).
R symphony stores the transposes (``Z_corr`` d x cells, ``centroids`` d x K); ``R`` is K x cells in both.

Reproduced from the R source (internals agree to < 1e-13, SYMPHONY_PARITY.md):
* variable genes: symphony's ``vargenes_vst`` runs the vst statistic on the *log-normalised* matrix (not counts,
  unlike Seurat); per-group top genes are united (``vargenes_groups``).
* scaling: (x - mean) / sd with the s.d. on n - 1, clipped to [-10, 10] (both tails, unlike Seurat's ScaleData).
* PCA: exact SVD of the scaled reference (irlba in R); signs may differ from R, which changes nothing downstream.
* Harmony: ``arsc.run_harmony(version="1.2")`` (harmony 1.2 port) with ``nclust = K = 100``, theta 2, sigma 0.1, lambda 1. With
  harmony 1.2 installed, R symphony's ``max.iter.harmony = 20`` and ``tau`` are ignored (harmony warns), so the
  effective R run is ``max_iter = 10, tau = 0``: these are the defaults here.
* centroids: L2-normalised rows of ``R @ Z_corr``; ``Nr = R.sum(1)``, ``C = R @ Z_corr``.
* query soft clustering: ``R = softmax(-2 (1 - Y Zq_cos) / sigma)`` over clusters (sigma as float32, as in C++).
* correction: per cluster k, ``beta = (Xq R_k Xq' + Nr_k e1 e1' + lambda I_-0)^-1 (Xq R_k Zq + e1 C_k)``,
  intercept row zeroed, ``Zq -= beta' Xq R_k`` (lambda 1 on the batch terms, not on the intercept).
* kNN: Euclidean on the corrected embeddings, majority vote, ``prob`` = share of the k votes (``class::knn``).
  Vote ties are broken by an integer-seeded generator, not R's RNG, and ties at the k-th distance are not
  widened (R includes them); both are rare with continuous embeddings.
Not ported: ``vargenes_method = "mvp"``, ``calcPerClusterMappingMetric``, the k-NN correlation metrics.
UMAP: umap-learn with uwot's settings (30 neighbours, cosine, min_dist 0.1, learning rate 0.5, spectral init);
layouts differ from uwot's, so compare neighbourhoods, not coordinates.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._seedlog import records_seed
from ._utils import as_csr, check_seed

__all__ = ["Reference", "QueryMap", "build_reference", "reference_from_harmony", "map_query",
           "transfer_labels", "mapping_score", "load_reference"]

_SIGMA32 = np.float32  # symphony's C++ soft_cluster takes ``float sigma``


# --------------------------------------------------------------------------- containers
@dataclass
class Reference:
    """A compressed Symphony reference (``buildReference`` output), cells-first orientation.

    Attributes
    ----------
    genes : Index of the reference variable genes (order of ``mean``, ``sd`` and the ``loadings`` rows).
    mean, sd : per-gene mean and s.d. (n - 1) of the log-normalised reference expression.
    loadings : (genes x d) PCA gene loadings (R ``loadings``).
    Z_orig : (cells x d) reference PCs before Harmony.
    Z_corr : (cells x d) Harmony-corrected reference embedding.
    R : (K x cells) soft cluster assignments.
    centroids : (K x d) L2-normalised Harmony centroids (R ``centroids``, transposed).
    Nr : (K,) cluster sizes ``R.sum(1)`` (R ``cache[[1]]``).
    C : (K x d) ``R @ Z_corr`` (R ``cache[[2]]``).
    obs : reference cell metadata (index = cell barcodes, row-aligned with ``Z_corr``).
    umap : (cells x 2) reference UMAP, or None.
    umap_model : fitted umap-learn model (for placing query cells), or None.
    params : the settings used to build it.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
    ...                                     topn=200, d=10, K=20, umap=False)
    >>> ref.Z_corr.shape, ref.centroids.shape
    ((3000, 10), (20, 10))
    """

    genes: pd.Index
    mean: np.ndarray
    sd: np.ndarray
    loadings: np.ndarray
    Z_orig: np.ndarray
    Z_corr: np.ndarray
    R: np.ndarray
    centroids: np.ndarray
    Nr: np.ndarray
    C: np.ndarray
    obs: pd.DataFrame
    umap: np.ndarray | None = None
    umap_model: object | None = None
    params: dict = field(default_factory=dict)

    @property
    def centroids_pc(self) -> np.ndarray:
        """(K x d) cluster centroids in the corrected space, ``C / Nr`` (R ``centroids_pc``)."""
        return self.C / self.Nr[:, None]

    def save(self, path: str) -> str:
        """Write the reference to a directory (compressed arrays, metadata CSV, pickled UMAP model).

        Parameters
        ----------
        path : directory to create (or reuse).

        Returns
        -------
        str, the directory.

        Examples
        --------
        >>> import arsc, tempfile
        >>> ds = arsc.datasets.toy()
        >>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
        ...                                     topn=200, d=10, K=20, umap=False)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     ref2 = arsc.symphony.load_reference(ref.save(d))
        >>> bool((ref2.Z_corr == ref.Z_corr).all())
        True
        """
        os.makedirs(path, exist_ok=True)
        arrs = dict(mean=self.mean, sd=self.sd, loadings=self.loadings, Z_orig=self.Z_orig, Z_corr=self.Z_corr,
                    R=self.R, centroids=self.centroids, Nr=self.Nr, C=self.C,
                    genes=np.asarray(self.genes, dtype=str))
        if self.umap is not None:
            arrs["umap"] = self.umap
        np.savez_compressed(os.path.join(path, "reference.npz"), **arrs)
        self.obs.to_csv(os.path.join(path, "obs.csv.gz"))
        with open(os.path.join(path, "params.json"), "w", encoding="utf-8") as fh:
            json.dump(self.params, fh, indent=1, default=str)
        if self.umap_model is not None:
            with open(os.path.join(path, "umap_model.pkl"), "wb") as fh:
                pickle.dump(self.umap_model, fh)
        return path


def load_reference(path: str) -> Reference:
    """Read a reference written by :meth:`Reference.save`.

    The UMAP model is a pickle: load only references you (or a trusted pipeline) wrote.

    Parameters
    ----------
    path : directory written by ``Reference.save``.

    Returns
    -------
    Reference.

    Examples
    --------
    >>> import arsc, tempfile
    >>> ds = arsc.datasets.toy()
    >>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
    ...                                     topn=200, d=10, K=20, umap=False)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     len(arsc.symphony.load_reference(ref.save(d)).genes)
    200
    """
    z = np.load(os.path.join(path, "reference.npz"), allow_pickle=False)
    obs = pd.read_csv(os.path.join(path, "obs.csv.gz"), index_col=0)
    with open(os.path.join(path, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    model = None
    mp = os.path.join(path, "umap_model.pkl")
    if os.path.exists(mp):
        with open(mp, "rb") as fh:
            model = pickle.load(fh)
    return Reference(pd.Index(z["genes"]), z["mean"], z["sd"], z["loadings"], z["Z_orig"], z["Z_corr"], z["R"],
                     z["centroids"], z["Nr"], z["C"], obs, z["umap"] if "umap" in z.files else None, model, params)


@dataclass
class QueryMap:
    """A query mapped onto a :class:`Reference` (``mapQuery`` output), cells first.

    Attributes
    ----------
    Z : (cells x d) corrected query embedding, in the reference's Harmony space (R ``Z``).
    Z_pca : (cells x d) query projected on the reference loadings, before correction (R ``Zq_pca``).
    R : (K x cells) soft assignment of query cells to the reference clusters.
    obs : query metadata.
    umap : (cells x 2) query cells placed on the reference UMAP, or None.
    n_shared_genes : reference variable genes found in the query (missing ones are set to 0 after scaling).

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
    >>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
    >>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
    ...                                     umap=False)
    >>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
    >>> q.Z.shape, q.n_shared_genes
    ((1517, 10), 200)
    """

    Z: np.ndarray
    Z_pca: np.ndarray
    R: np.ndarray
    obs: pd.DataFrame
    umap: np.ndarray | None = None
    n_shared_genes: int = 0


# --------------------------------------------------------------------------- building blocks
def _l2_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _lognorm_columns(X, cols, normalize: bool, scale_factor: float = 1e4) -> np.ndarray:
    """Dense cells x len(cols) log-normalised values; library sizes from ALL genes (only ``cols`` are densified)."""
    X = as_csr(X)
    sub = X[:, cols].tocsr().astype(np.float64)
    if normalize:
        tot = np.asarray(X.sum(1)).ravel().astype(np.float64)
        tot[tot == 0] = 1.0
        sub = sp.diags(scale_factor / tot) @ sub
        sub.data = np.log1p(sub.data)
    return sub.toarray()


def _scale_with_stats(E, mean, sd, thresh: float = 10.0) -> np.ndarray:
    """symphony ``scaleDataWithStats``: (x - mean) / sd, clipped to [-thresh, thresh]; zero-s.d. genes -> 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        S = (E - mean[None, :]) / np.where(sd > 0, sd, np.nan)[None, :]
    S[~np.isfinite(S)] = 0.0
    np.clip(S, -thresh, thresh, out=S)
    return S


def _soft_cluster(centroids, Zcos, sigma) -> np.ndarray:
    """(K x cells) ``softmax_k(-2 (1 - y_k . z) / sigma)`` (symphony ``soft_cluster``)."""
    s = float(_SIGMA32(sigma))
    Rq = -2.0 * (1.0 - centroids @ Zcos.T) / s
    Rq -= Rq.max(0, keepdims=True)
    np.exp(Rq, out=Rq)
    Rq /= Rq.sum(0, keepdims=True)
    return Rq


def _soft_kmeans(Z, K, seed, max_iter=20, sigma=0.1):
    """symphony ``soft_kmeans`` (reference without batch variables): k-means init on the cosine-normalised PCs,
    then ``max_iter`` rounds of {normalise Y, soft-assign, Y = R Z}."""
    from sklearn.cluster import KMeans
    Zc = _l2_rows(Z)
    Y = KMeans(n_clusters=K, n_init=10, max_iter=25, random_state=seed, algorithm="lloyd").fit(Zc).cluster_centers_
    for _ in range(max_iter):
        Y = _l2_rows(Y)
        R = _soft_cluster(Y, Zc, sigma)
        Y = R @ Zc
    return _l2_rows(Y), R


def _design(obs, vars, n):
    """Query design as (pattern matrix P x (B+1), per-cell pattern index): rows of symphony's Xq.

    Xq = [1, one-hot(var1), one-hot(var2), ...]; a variable with one level contributes a column of ones;
    no variables -> two columns of ones (symphony's placeholder)."""
    if vars is None:
        return np.ones((1, 2)), np.zeros(n, dtype=np.int64)
    vars = [vars] if isinstance(vars, str) else list(vars)
    miss = [v for v in vars if v not in obs.columns]
    if miss:
        raise KeyError(f"map_query: batch column(s) {miss} not in the query obs")
    blocks = []
    for v in vars:
        col = obs[v]
        if col.isna().any():
            raise ValueError(f"map_query: batch column {v!r} has missing values")
        cat = pd.Categorical(col.astype(str) if not isinstance(col.dtype, pd.CategoricalDtype)
                             else col.cat.remove_unused_categories())
        L = len(cat.categories)
        blocks.append(np.ones((n, 1)) if L == 1 else np.eye(L)[cat.codes])
    Xfull = np.hstack([np.ones((n, 1))] + blocks)
    pats, inv = np.unique(Xfull, axis=0, return_inverse=True)
    return pats, inv.ravel().astype(np.int64)


def _moe_correct_ref(Zq, pats, pat_of, Rq, Nr, C, lam: float = 1.0) -> np.ndarray:
    """symphony ``moe_correct_ref`` with the query design compressed to its distinct rows (exact; O(K N d))."""
    K = Rq.shape[0]
    P, B1 = pats.shape
    Zcorr = Zq.copy()
    order = np.argsort(pat_of, kind="stable")
    bounds = np.searchsorted(pat_of[order], np.arange(P + 1))
    # per distinct design row p: Rp[:, p] = sum of R over its cells, S[p] = R[:, cells] @ Zq[cells] (K x d)
    Rp = np.empty((K, P))
    S = np.empty((P, K, Zq.shape[1]))
    for p in range(P):
        idx = order[bounds[p]:bounds[p + 1]]
        Rp[:, p] = Rq[:, idx].sum(1)
        S[p] = Rq[:, idx] @ Zq[idx]
    pen = np.full(B1, lam)
    pen[0] = 0.0
    W = np.empty((P, K, Zq.shape[1]))
    for k in range(K):
        m1 = (pats.T * Rp[k]) @ pats
        m1[0, 0] += Nr[k]
        m1[np.diag_indices(B1)] += pen
        m2 = np.einsum("pb,pd->bd", pats, S[:, k, :])
        m2[0] += C[k]
        beta = np.linalg.solve(m1, m2)
        beta[0] = 0.0
        W[:, k, :] = pats @ beta
    for p in range(P):
        idx = order[bounds[p]:bounds[p + 1]]
        Zcorr[idx] -= Rq[:, idx].T @ W[p]
    return Zcorr


def _fit_umap(Z, seed, min_dist, n_neighbors):
    import warnings
    import umap as umap_learn
    model = umap_learn.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", learning_rate=0.5,
                            init="spectral", random_state=seed, transform_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = model.fit_transform(Z)
    return emb, model


# --------------------------------------------------------------------------- public API
def _finish_reference(Z_orig, Z_corr, R, genes, mean, sd, loadings, obs, *, umap, umap_min_dist, umap_n_neighbors,
                      seed, params) -> Reference:
    R = np.ascontiguousarray(R, dtype=np.float64)
    C = R @ Z_corr
    Nr = R.sum(1)
    centroids = _l2_rows(C)
    emb = model = None
    if umap:
        emb, model = _fit_umap(Z_corr, seed, umap_min_dist, umap_n_neighbors)
    return Reference(pd.Index(genes), np.asarray(mean, float), np.asarray(sd, float), np.asarray(loadings, float),
                     np.asarray(Z_orig, float), np.asarray(Z_corr, float), R, centroids, Nr, C, obs.copy(), emb,
                     model, params)


@records_seed("seed")
def build_reference(X, genes, obs, vars=None, *, normalize: bool = True, topn: int = 2000, vargenes_groups=None,
                    hvg=None, additional_genes=None, d: int = 20, K: int = 100, theta=2.0, sigma: float = 0.1,
                    lambda_: float = 1.0, tau: float = 0.0, max_iter: int = 10, umap: bool = True,
                    umap_min_dist: float = 0.1, umap_n_neighbors: int = 30, seed: int = 111) -> Reference:
    """Build a Symphony reference (R ``symphony::buildReference``).

    Parameters
    ----------
    X : reference cells x genes matrix: raw counts (``normalize=True``, the default, as R ``do_normalize``) or
        log-normalised values (``normalize=False``). A memory-mapped store matrix works (``st.X("RNA", "counts")``).
    genes : gene names aligned to the columns of ``X``.
    obs : reference metadata, row-aligned with ``X`` (index = barcodes); kept for label transfer.
    vars : batch column(s) for Harmony (R ``vars``); None = no Harmony (soft k-means clusters instead).
    normalize : log-normalise counts first (LogNormalize, scale factor 1e4).
    topn : variable genes (per group when ``vargenes_groups`` is given; groups' top genes are united).
    vargenes_groups : obs column whose groups each contribute ``topn`` genes (R ``vargenes_groups``).
    hvg : optional explicit gene list instead of the vst selection (e.g. the HVGs of an existing Seurat object).
    additional_genes : genes added to the selection (R ``additional_genes``).
    d : number of PCs (R ``d``).
    K : Harmony clusters (R ``K``).
    theta, sigma, lambda_, tau, max_iter : Harmony settings (see module notes: R with harmony 1.2 runs
        ``max_iter = 10`` and ignores ``tau``).
    umap : fit a reference UMAP (kept as a model so query cells can be placed on it).
    umap_min_dist, umap_n_neighbors : UMAP settings (R: 0.1, 30).
    seed : integer seed (R ``seed = 111``) for every stochastic step: Harmony's k-means init and block order,
        the k-means of the no-Harmony case, and the UMAP layout (SEEDS.md rule 5).

    Returns
    -------
    Reference.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
    ...                                     topn=200, d=10, K=20, umap=False)
    >>> ref.loadings.shape, round(float(ref.R.sum(0).mean()), 6)
    ((200, 10), 1.0)
    """
    from .harmony import run_harmony
    from .preprocessing import hvg_vst, log_normalize
    seed = check_seed(seed)
    X = as_csr(X)
    genes = pd.Index(genes)
    if X.shape[1] != len(genes) or X.shape[0] != len(obs):
        raise ValueError(f"build_reference: X is {X.shape}, genes {len(genes)}, obs {len(obs)} (must align)")
    if hvg is None:
        L = log_normalize(X) if normalize else X.astype(np.float64)
        if vargenes_groups is None:
            hv = hvg_vst(L, nfeatures=topn, features=genes)
            var_genes = list(hv.sort_values("rank", kind="stable").index[:topn])
        else:
            var_genes = []
            grp = obs[vargenes_groups].astype(str).to_numpy()
            for gname in sorted(pd.unique(grp)):  # R split() orders groups by factor level (sorted)
                m = grp == gname
                hv = hvg_vst(L[m], nfeatures=topn, features=genes)
                for g in hv.sort_values("rank", kind="stable").index[:topn]:
                    if g not in var_genes:
                        var_genes.append(g)
        del L
    else:
        var_genes = [g for g in hvg if g in genes]
        if len(var_genes) < len(hvg):
            raise KeyError(f"build_reference: {len(hvg) - len(var_genes)} hvg genes not in genes")
    if additional_genes is not None:
        var_genes += [g for g in additional_genes if g not in var_genes and g in genes]
    cols = genes.get_indexer(var_genes)
    E = _lognorm_columns(X, cols, normalize)
    mean = E.mean(0)
    sd = E.std(0, ddof=1)
    S = _scale_with_stats(E, mean, sd)
    del E
    from .preprocessing import pca
    pc = pca(S, n_comps=d, method="gram" if S.shape[1] <= 6000 else "arpack")
    del S
    Z_orig = pc.embedding
    params = dict(normalize=normalize, topn=topn, d=d, K=K, theta=theta, sigma=sigma, lambda_=lambda_, tau=tau,
                  max_iter=max_iter, vars=vars, seed=seed, umap_min_dist=umap_min_dist,
                  umap_n_neighbors=umap_n_neighbors, method="build_reference")
    if vars is not None:
        vlist = [vars] if isinstance(vars, str) else list(vars)
        th = theta if np.ndim(theta) else [theta] * len(vlist)
        h = run_harmony(Z_orig, obs, vlist, theta=th, sigma=sigma, lambda_=lambda_, nclust=K, max_iter=max_iter,
                        tau=tau, seed=seed, return_object=True, version="1.2")
        Z_corr = h.Z_corr
        R = h.R.T if h.R.shape[0] == Z_orig.shape[0] else h.R
    else:
        _, R = _soft_kmeans(Z_orig, K, seed, sigma=sigma)
        Z_corr = Z_orig
    return _finish_reference(Z_orig, Z_corr, R, var_genes, mean, sd, pc.loadings, obs, umap=umap,
                             umap_min_dist=umap_min_dist, umap_n_neighbors=umap_n_neighbors, seed=seed,
                             params=params)


def reference_from_harmony(Z_orig, Z_corr, R, genes, mean, sd, loadings, obs, *, umap: bool = True,
                           umap_min_dist: float = 0.1, umap_n_neighbors: int = 30, seed: int = 111) -> Reference:
    """Build a reference from an existing PCA + Harmony run (R ``buildReferenceFromHarmonyObj``).

    Use it to wrap a reference made elsewhere, e.g. the matrices of a saved R symphony reference or a Seurat
    object (``Z_orig`` = PCs, ``Z_corr`` = Harmony embedding, ``R`` = Harmony's soft clusters, ``loadings`` =
    feature loadings, ``mean``/``sd`` = the variable genes' log-normalised mean and s.d.).

    Parameters
    ----------
    Z_orig, Z_corr : (cells x d) PCs and Harmony embedding.
    R : (K x cells) or (cells x K) soft cluster assignments.
    genes : the variable genes (rows of ``loadings``).
    mean, sd : per-gene mean and s.d. used to scale the reference.
    loadings : (genes x d) loadings.
    obs : reference metadata, row-aligned with ``Z_corr``.
    umap : fit a reference UMAP model on ``Z_corr``.
    umap_min_dist, umap_n_neighbors, seed : UMAP settings.

    Returns
    -------
    Reference.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> ds = arsc.datasets.toy()
    >>> r = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
    ...                                   topn=200, d=10, K=20, umap=False)
    >>> r2 = arsc.symphony.reference_from_harmony(r.Z_orig, r.Z_corr, r.R, r.genes, r.mean, r.sd, r.loadings,
    ...                                          r.obs, umap=False)
    >>> bool(np.allclose(r2.C, r.C))
    True
    """
    seed = check_seed(seed)
    Z_orig, Z_corr, R = (np.asarray(a, dtype=np.float64) for a in (Z_orig, Z_corr, R))
    if R.shape[0] == Z_corr.shape[0] and R.shape[1] != Z_corr.shape[0]:
        R = R.T
    return _finish_reference(Z_orig, Z_corr, R, list(genes), mean, sd, loadings, obs, umap=umap,
                             umap_min_dist=umap_min_dist, umap_n_neighbors=umap_n_neighbors, seed=seed,
                             params=dict(method="reference_from_harmony", seed=seed))


def map_query(X, genes, obs, ref: Reference, vars=None, *, normalize: bool = True, sigma: float = 0.1,
              lambda_: float = 1.0, umap: bool = True) -> QueryMap:
    """Map query cells onto a Symphony reference (R ``symphony::mapQuery``).

    Parameters
    ----------
    X : query cells x genes, raw counts (``normalize=True``) or log-normalised. Only the reference's variable
        genes are densified; library sizes use all genes. A memory-mapped store matrix works.
    genes : gene names of ``X``'s columns.
    obs : query metadata, row-aligned with ``X``.
    ref : the :class:`Reference`.
    vars : query batch column(s) to correct (R ``vars``). The correction removes what these columns explain, so
        never include the biology you want to compare (e.g. timepoint or response).
    normalize : log-normalise counts first.
    sigma : soft-clustering width (R ``sigma = 0.1``).
    lambda_ : ridge penalty on the batch terms (symphony fixes 1).
    umap : place query cells on the reference UMAP (needs ``ref.umap_model``).

    Returns
    -------
    QueryMap.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
    >>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
    >>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
    ...                                     umap=False)
    >>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
    >>> q.R.shape
    (20, 1517)
    """
    X = as_csr(X)
    genes = pd.Index(genes)
    if X.shape[1] != len(genes) or X.shape[0] != len(obs):
        raise ValueError(f"map_query: X is {X.shape}, genes {len(genes)}, obs {len(obs)} (must align)")
    pos = genes.get_indexer(ref.genes)
    shared = pos >= 0
    E = np.zeros((X.shape[0], len(ref.genes)))
    if shared.any():
        E[:, shared] = _lognorm_columns(X, pos[shared], normalize)
    S = _scale_with_stats(E, ref.mean, ref.sd)
    S[:, ~shared] = 0.0  # symphony: missing genes are 0 in the scaled, synchronised matrix
    del E
    Zq = S @ ref.loadings
    del S
    Rq = _soft_cluster(ref.centroids, _l2_rows(Zq), sigma)
    pats, pat_of = _design(obs, vars, Zq.shape[0])
    Zc = _moe_correct_ref(Zq, pats, pat_of, Rq, ref.Nr, ref.C, lam=lambda_)
    um = None
    if umap and ref.umap_model is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            um = ref.umap_model.transform(Zc)
    return QueryMap(Zc, Zq, Rq, obs.copy(), um, int(shared.sum()))


@records_seed("seed")
def transfer_labels(ref: Reference, query: QueryMap, labels, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """k-NN label transfer from the reference to the mapped query (R ``symphony::knnPredict``).

    Parameters
    ----------
    ref : the :class:`Reference`.
    query : the :class:`QueryMap`.
    labels : a column of ``ref.obs`` or an array of reference labels (row-aligned with ``ref.Z_corr``).
    k : neighbours (R ``k = 5``).
    seed : integer seed for breaking vote ties.

    Returns
    -------
    DataFrame indexed by query cell: ``label`` (majority among the k nearest reference cells, Euclidean on the
    corrected embeddings) and ``prob`` (share of the k votes for it; R ``_prob``). Low ``prob`` marks cells whose
    neighbourhood is mixed, e.g. a state absent from the reference.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
    >>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
    >>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
    ...                                     umap=False)
    >>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
    >>> lab = arsc.symphony.transfer_labels(ref, q, "cell_type", k=5)
    >>> float((lab["label"] == ds.obs.loc[~isref, "cell_type"]).mean()) > 0.8   # the toy types overlap
    True
    """
    from .neighbors import knn
    seed = check_seed(seed)
    y = ref.obs[labels].to_numpy() if isinstance(labels, str) else np.asarray(labels)
    if len(y) != ref.Z_corr.shape[0]:
        raise ValueError("transfer_labels: labels must be one per reference cell")
    cat = pd.Categorical(y)
    codes = np.asarray(cat.codes)
    idx, _ = knn(ref.Z_corr, k, query=query.Z)
    votes = np.zeros((idx.shape[0], len(cat.categories)))
    np.add.at(votes, (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), codes[idx].ravel()), 1)
    top = votes.max(1)
    rng = np.random.default_rng(seed)
    tie = (votes == top[:, None])
    win = np.argmax(tie, axis=1)
    multi = np.flatnonzero(tie.sum(1) > 1)
    for i in multi:
        win[i] = rng.choice(np.flatnonzero(tie[i]))
    return pd.DataFrame({"label": np.asarray(cat.categories)[win], "prob": top / idx.shape[1]},
                        index=query.obs.index)


def mapping_score(ref: Reference, query: QueryMap, use_orig: bool = True) -> np.ndarray:
    """Per-cell mapping distance (R ``symphony::calcPerCellMappingMetric``, Mahalanobis).

    For each reference cluster k, the R_k-weighted mean and covariance of the reference PCs (``Z_orig``; or the
    corrected ``Z_corr`` with ``use_orig=False``); each query cell's Mahalanobis distance of its uncorrected
    projection to every cluster, averaged with the cell's soft assignment. High values = the cell sits far from
    the reference distribution (possibly a state the reference lacks); compare against the reference's own cells
    or a held-out reference batch to set a cut-off.

    Parameters
    ----------
    ref : the :class:`Reference`.
    query : the :class:`QueryMap`.
    use_orig : use the reference PCs before Harmony (R default ``Z_orig = TRUE``).

    Returns
    -------
    numpy.ndarray, one distance per query cell.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
    >>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
    >>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
    ...                                     umap=False)
    >>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
    >>> arsc.symphony.mapping_score(ref, q).shape
    (1517,)
    """
    Zr = ref.Z_orig if use_orig else ref.Z_corr
    Zq = query.Z_pca
    out = np.zeros(Zq.shape[0])
    for k in range(ref.R.shape[0]):
        w = ref.R[k] / ref.R[k].sum()
        c = w @ Zr
        D = Zr - c
        cov = (D * w[:, None]).T @ D / (1.0 - np.sum(w ** 2))  # stats::cov.wt (unbiased)
        Q = Zq - c
        m = np.einsum("ij,ij->i", Q @ np.linalg.inv(cov), Q)
        out += np.sqrt(np.maximum(m, 0.0)) * query.R[k]
    return out
