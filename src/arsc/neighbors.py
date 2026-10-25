"""Nearest neighbours, SNN graphs and weighted nearest neighbours (WNN).

=======================  =====================================================
arsc                     Seurat equivalent
=======================  =====================================================
``knn``                  ``NNHelper`` / ``AnnoyNN`` (here: FAISS, exact or IVF)
``snn_graph``            ``ComputeSNN`` (Jaccard of k-NN sets, pruned)
``find_neighbors``       ``FindNeighbors`` (nn + snn graphs)
``wnn``                  ``FindMultiModalNeighbors`` (Hao et al. 2021)
=======================  =====================================================

Seurat uses Annoy (approximate, randomized). arsc searches with FAISS: exact up to
4,096 cells and seeded IVF (approximate, deterministic on the CPU) above, as scanpy
switches to pynndescent; method="exact" gives the exact graph at any size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
import scipy.sparse as sp

from ._utils import check_seed
from ._seedlog import records_seed

__all__ = ["knn", "snn_graph", "find_neighbors", "wnn", "WNNResult", "l2_normalize"]


# --------------------------------------------------------------------------- kNN
# The search runs in FAISS (Meta; MIT licence; a hard dependency, ``faiss-cpu``): IndexFlatL2 (BLAS brute force) for
# the exact method, IndexIVFFlat for the approximate one, the same indexes on a CUDA GPU with ``faiss-gpu``. arsc only
# re-scores the candidates FAISS returns in float64 and orders them by (distance, index).
AUTO_EXACT_MAX = 4096   # method="auto": exact up to this many points (scanpy's switch to approximate), approximate above
IVF_LISTS = 4.0         # approximate: IVF lists = IVF_LISTS * sqrt(n)
IVF_PROBE = 0.04        # approximate: lists probed per query = IVF_PROBE * lists (at least 8)
_EXTRA = 8              # exact: extra candidates per query beyond k for the float64 re-scoring
_U32 = 2.0 ** -24


@nb.njit(parallel=True, cache=True)
def _exact_d2(Q, X, cand, out):
    """out[i, j] = squared distance of Q[i] to X[cand[i, j]] in float64 (plain left-to-right sum; inf for -1)."""
    m, kk = cand.shape
    d = Q.shape[1]
    for i in nb.prange(m):
        for j in range(kk):
            c = cand[i, j]
            if c < 0:
                out[i, j] = np.inf
                continue
            s = 0.0
            for t in range(d):
                df = Q[i, t] - X[c, t]
                s += df * df
            out[i, j] = s


def _rescore(Q, X, cand, k):
    """Exact float64 distances of the candidates, ordered by (distance, index); the k best per row."""
    d2 = np.empty(cand.shape)
    _exact_d2(Q, X, cand, d2)
    big = np.iinfo(np.int64).max
    order = np.lexsort((np.where(cand < 0, big, cand), d2), axis=-1)[:, :k]
    return np.take_along_axis(cand, order, 1), np.take_along_axis(d2, order, 1)


def _faiss(device: str):
    """faiss with arsc's thread count; returns (faiss, GPU resources or None)."""
    import faiss
    from ._threads import get_threads
    faiss.omp_set_num_threads(get_threads())  # never all cores
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unknown device {device!r}; use 'auto', 'cpu' or 'gpu'")
    ngpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    if device == "gpu" and ngpu == 0:
        raise RuntimeError("device='gpu' needs a CUDA GPU and a GPU build of faiss (pip install 'arsc[gpu]' on Linux); "
                           "use device='cpu' or 'auto'")
    if device == "cpu" or ngpu == 0:
        return faiss, None
    return faiss, faiss.StandardGpuResources()


def _to_device(faiss, res, index):
    return index if res is None else faiss.index_cpu_to_gpu(res, 0, index)


def _centred32(X, Q):
    mu = X.mean(0)
    X32 = np.ascontiguousarray(X - mu, dtype=np.float32)
    Q32 = X32 if Q is X else np.ascontiguousarray(Q - mu, dtype=np.float32)
    return X32, Q32


def _knn_exact(X, Q, k, faiss, res):
    n, d = X.shape
    X32, Q32 = _centred32(X, Q)
    k2 = min(n, k + _EXTRA)
    index = _to_device(faiss, res, faiss.IndexFlatL2(d))
    index.add(X32)
    D32, cand = index.search(Q32, k2)
    idx, d2 = _rescore(Q, X, cand.astype(np.int64), k)
    if k2 < n:
        # A point outside the k2 candidates can be closer than the k-th only if float32 rounding of FAISS's
        # |q|^2 + |x|^2 - 2 q.x moved it past the k2-th candidate: bound that error and search those rows in full.
        qn = (Q32.astype(np.float64) ** 2).sum(1)
        xmax = float((X32.astype(np.float64) ** 2).sum(1).max())
        err = 8.0 * (d + 8) * _U32 * (qn + xmax) + 4.0 * _U32 * np.abs(D32[:, -1])
        bad = np.flatnonzero(D32[:, -1].astype(np.float64) - err <= d2[:, -1])
        if bad.size:
            allc = np.arange(n, dtype=np.int64)
            for a in range(0, bad.size, 256):
                rows = bad[a:a + 256]
                full = np.ascontiguousarray(np.broadcast_to(allc, (rows.size, n)))
                idx[rows], d2[rows] = _rescore(np.ascontiguousarray(Q[rows]), X, full, k)
    return idx, d2


def _knn_ivf(X, Q, k, faiss, res, seed, n_lists, n_probe):
    n, d = X.shape
    X32, Q32 = _centred32(X, Q)
    nl = int(n_lists) if n_lists is not None else max(1, int(IVF_LISTS * np.sqrt(n)))
    nl = max(1, min(nl, n // 39))  # faiss wants >= 39 training points per list
    npr = int(n_probe) if n_probe is not None else max(8, int(round(IVF_PROBE * nl)))
    npr = max(1, min(npr, nl))
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nl)
    index.cp.seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain = min(n, 64 * nl)
    train = X32 if ntrain == n else X32[np.sort(rng.choice(n, ntrain, replace=False))]
    index.train(train)  # on the CPU, so the partition does not depend on the device
    index.add(X32)
    index.nprobe = npr
    index = _to_device(faiss, res, index)
    _, cand = index.search(Q32, k)
    return _rescore(Q, X, cand.astype(np.int64), k)


@records_seed("seed")
def knn(X, k: int, query=None, *, method: str = "auto", device: str = "auto", n_lists: int | None = None,
        n_probe: int | None = None, seed: int = 0, n_clusters: int | None = None):
    """k nearest neighbours (Euclidean) with FAISS: exact up to 4,096 points, approximate above (``"auto"``).

    ``method="exact"``: FAISS brute force (``IndexFlatL2``, BLAS) for ``k + 8`` candidates, re-scored in float64
    and ordered by (distance, index); a row whose float32 candidate list could have missed a closer point (by a
    proven rounding bound) is searched again in full. The answer is the exact float64 k-NN. Deterministic; the
    query point itself comes first when ``query`` is None.

    ``method="approximate"`` (alias ``"approx"``): FAISS ``IndexIVFFlat``: k-means into ``n_lists`` lists (default
    ``4 * sqrt(n)``, trained on a seeded sample), each query scans its ``n_probe`` nearest lists (default 4% of
    the lists, at least 8). Distances of the neighbours found are exact (float64). Seeded; deterministic on the
    CPU. Recall against exact neighbours and downstream clusters vs scanpy's pynndescent: ``SCANPY_PARITY.md``.

    ``method="auto"`` (default): exact when n <= 4,096 or ``query`` is given, approximate above, as
    ``scanpy.pp.neighbors`` (exact below 4,096 cells, pynndescent above). arsc 0.3.x used exact search at every
    size: pass ``method="exact"`` to reproduce those graphs.

    ``method="pynndescent"``: pynndescent's NN-descent (scanpy's large-n route; needs the ``graph`` extra).

    ``device``: ``"auto"`` (default) runs on a CUDA GPU when the installed faiss has GPU support (``arsc[gpu]``,
    Linux) and a GPU is present, else on the CPU; ``"cpu"`` forces the CPU; ``"gpu"`` requires the GPU.

    Parameters
    ----------
    X : (n x d) dense embedding (e.g. PCs).
    k : neighbours per query (self included when ``query`` is None); capped at n.
    query : optional (m x d) query points; default ``X`` itself.
    method : ``"auto"``, ``"exact"``, ``"approximate"`` (``"approx"``) or ``"pynndescent"``.
    device : ``"auto"``, ``"cpu"`` or ``"gpu"``.
    n_lists, n_probe : IVF lists, and lists scanned per query (approximate method; defaults above).
    seed : integer seed of the IVF training sample and k-means (approximate) or pynndescent's ``random_state``;
        the exact method does not use it.
    n_clusters : older name of ``n_lists`` (kept for compatibility).

    Returns
    -------
    tuple ``(idx, dist)``: int64 and float64 arrays, each (n_query x k), nearest first.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L)
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> idx, dist = arsc.knn(pcs, 15)
    >>> idx.shape, bool((idx[:, 0] == range(3000)).all())
    ((3000, 15), True)
    """
    seed = check_seed(seed)
    if hasattr(X, "toarray"):
        raise TypeError("knn expects a dense embedding (e.g. PCs), not a sparse expression matrix")
    X = np.ascontiguousarray(X, dtype=np.float64)
    Q = X if query is None else np.ascontiguousarray(query, dtype=np.float64)
    n = X.shape[0]
    k = int(min(k, n))
    if method == "pynndescent":
        from pynndescent import NNDescent
        from ._threads import get_threads
        index = NNDescent(X, n_neighbors=k, random_state=seed, n_jobs=get_threads())  # never -1 (all cores)
        if query is None:
            idx, dist = index.neighbor_graph
        else:
            idx, dist = index.query(Q, k=k)
        return idx.astype(np.int64), dist.astype(np.float64)
    if method == "approx":
        method = "approximate"
    if method == "auto":
        method = "exact" if (query is not None or n <= AUTO_EXACT_MAX) else "approximate"
    if method not in ("exact", "approximate"):
        raise ValueError(f"unknown method {method!r}; use 'auto', 'exact', 'approximate' or 'pynndescent'")
    faiss, res = _faiss(device)
    if method == "exact":
        idx, d2 = _knn_exact(X, Q, k, faiss, res)
    else:
        idx, d2 = _knn_ivf(X, Q, k, faiss, res, seed, n_lists if n_lists is not None else n_clusters, n_probe)
    if query is None:
        # the point itself first: when it tied with a duplicate at distance 0, or the approximate search missed it
        rows = np.arange(n)
        is_self = idx == rows[:, None]
        has = is_self.any(1)
        self_pos = np.argmax(is_self, axis=1)
        for r in np.flatnonzero(has & (self_pos > 0) & (d2[rows, self_pos] <= d2[:, 0])):
            p = self_pos[r]
            idx[r, 1:p + 1] = idx[r, 0:p].copy()
            d2[r, 1:p + 1] = d2[r, 0:p].copy()
            idx[r, 0] = r
            d2[r, 0] = 0.0
        for r in np.flatnonzero(~has):
            idx[r, 1:] = idx[r, :-1].copy()
            d2[r, 1:] = d2[r, :-1].copy()
            idx[r, 0] = r
            d2[r, 0] = 0.0
    return idx, np.sqrt(np.maximum(d2, 0.0))


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay 0 (Seurat ``L2Norm``).

    Parameters
    ----------
    X : (n x d) array.

    Returns
    -------
    numpy.ndarray (float64) with unit-norm rows.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> arsc.neighbors.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    array([[0.6, 0.8],
           [0. , 0. ]])
    """
    X = np.asarray(X, dtype=np.float64)
    nrm = np.sqrt((X * X).sum(1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = X / nrm
    out[~np.isfinite(out)] = 0.0
    return out


# --------------------------------------------------------------------------- SNN
@nb.njit(parallel=True, cache=True)
def _snn_count(idx, rev_ptr, rev_idx, n, prune, k, out_ptr, out_idx, out_val, fill, nt):
    """Shared-neighbour Jaccard |N(i) & N(j)| / (2k - |N(i) & N(j)|) for all pairs with overlap."""
    chunk = (n + nt - 1) // nt
    for t in nb.prange(nt):
        cnt = np.zeros(n, dtype=np.int32)
        touched = np.empty(n, dtype=np.int64)
        surv = np.empty(n, dtype=np.int64)
        for i in range(t * chunk, min(n, (t + 1) * chunk)):
            nt_ = 0
            for a in range(idx.shape[1]):
                m = idx[i, a]
                for p in range(rev_ptr[m], rev_ptr[m + 1]):
                    j = rev_idx[p]
                    if cnt[j] == 0:
                        touched[nt_] = j
                        nt_ += 1
                    cnt[j] += 1
            # keep the pairs that pass the cutoff; only those are sorted (by column), and only when filling
            w = 0
            for q in range(nt_):
                j = touched[q]
                s = cnt[j]
                v = s / (k + (k - s))
                if v >= prune and v > 0:
                    surv[w] = j
                    w += 1
            if fill:
                surv[:w].sort()  # in place: no allocation inside the parallel loop (Windows heap contention)
                base = out_ptr[i]
                for q in range(w):
                    j = surv[q]
                    s = cnt[j]
                    out_idx[base + q] = j
                    out_val[base + q] = s / (k + (k - s))
            for q in range(nt_):
                cnt[touched[q]] = 0
            if not fill:
                out_ptr[i + 1] = w


def snn_graph(idx: np.ndarray, prune: float = 1 / 15) -> sp.csr_matrix:
    """Seurat ``ComputeSNN``: Jaccard similarity of k-NN sets, entries < ``prune`` removed.

    Parameters
    ----------
    idx : (n x k) neighbour index matrix, e.g. from :func:`knn` (include self for
        ``FindNeighbors`` semantics).
    prune : Jaccard cutoff (Seurat ``prune.SNN``, default 1/15).

    Returns
    -------
    scipy.sparse.csr_matrix, symmetric (n x n) SNN weights.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L)
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> idx, _ = arsc.knn(pcs, 20)
    >>> arsc.snn_graph(idx).shape
    (3000, 3000)
    """
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    n, k = idx.shape
    # reverse index: for each cell m, the cells i having m among their neighbours
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    A = sp.csr_matrix((np.ones(n * k, dtype=np.int8), (idx.ravel(), rows)), shape=(n, n))
    A.sum_duplicates()
    rev_ptr, rev_idx = A.indptr.astype(np.int64), A.indices.astype(np.int64)
    counts = np.zeros(n + 1, dtype=np.int64)
    _snn_count(idx, rev_ptr, rev_idx, n, float(prune), float(k), counts, np.empty(0, np.int64),
               np.empty(0), False, nb.get_num_threads())
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts[1:], out=ptr[1:])
    out_idx = np.empty(ptr[-1], dtype=np.int64)
    out_val = np.empty(ptr[-1])
    _snn_count(idx, rev_ptr, rev_idx, n, float(prune), float(k), ptr, out_idx, out_val, True, nb.get_num_threads())
    return sp.csr_matrix((out_val, out_idx, ptr), shape=(n, n))


def _nn_matrix(idx: np.ndarray, n: int) -> sp.csr_matrix:
    m, k = idx.shape
    return sp.csr_matrix((np.ones(m * k), (np.repeat(np.arange(m), k), idx.ravel())), shape=(m, n))


@records_seed("seed")
def find_neighbors(emb: np.ndarray, k: int = 20, prune: float = 1 / 15, l2_norm: bool = False,
                   method: str = "auto", seed: int = 0) -> dict:
    """Seurat ``FindNeighbors``: k-NN (self included) and SNN graphs from an embedding.

    ``method`` and ``seed`` are passed to :func:`knn`: by default (``"auto"``) the k-NN is exact up to
    50,000 cells and arsc's approximate search above (``method="exact"`` forces exact search at any size).
    Seurat's Annoy is approximate, so SNN graphs differ slightly from Seurat's (see the README validation
    table).

    Parameters
    ----------
    emb : (n x d) embedding (PCs, harmony).
    k : neighbours including self (Seurat ``k.param``).
    prune : SNN Jaccard cutoff (Seurat ``prune.SNN``).
    l2_norm : L2-normalize rows first (Seurat ``l2.norm``).
    method : ``"auto"`` (default), ``"exact"``, ``"approx"`` or ``"pynndescent"`` (see :func:`knn`).
    seed : integer seed.

    Returns
    -------
    dict with ``"idx"``, ``"dist"`` (n x k arrays), ``"nn"`` (binary CSR) and ``"snn"`` (CSR), the
    input for :func:`arsc.cluster`.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L)
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> g = arsc.find_neighbors(pcs, k=20)
    >>> sorted(g)
    ['dist', 'idx', 'nn', 'snn']
    >>> labels = arsc.cluster(g["snn"], resolution=0.5)
    >>> labels.shape
    (3000,)
    """
    X = l2_normalize(emb) if l2_norm else np.asarray(emb, dtype=np.float64)
    idx, dist = knn(X, k, method=method, seed=seed)
    return {"idx": idx, "dist": dist, "nn": _nn_matrix(idx, X.shape[0]), "snn": snn_graph(idx, prune)}


# --------------------------------------------------------------------------- WNN
@nb.njit(parallel=True, cache=True)
def _snn_width(ptr, ind, val, emb, n_far, nearest):
    """Seurat ``SNN_SmallestNonzero_Dist``: mean distance to the n least-shared SNN neighbours."""
    N = ptr.size - 1
    out = np.empty(N)
    for i in nb.prange(N):
        s, e = ptr[i], ptr[i + 1]
        m = e - s
        vals = val[s:e]
        o = np.argsort(vals, kind="mergesort")
        n_i = min(n_far, m)
        if n_i == 0:
            out[i] = 0.0
            continue
        thr = vals[o[n_i - 1]]
        dists = np.empty(m)
        nd = 0
        for jj in range(m):
            q = o[jj]
            if nd < n_i or vals[q] == thr:
                c = ind[s + q]
                acc = 0.0
                for t in range(emb.shape[1]):
                    df = emb[c, t] - emb[i, t]
                    acc += df * df
                r = np.sqrt(acc)
                if nearest[i] > 0:
                    r = r - nearest[i]
                    if r < 0:
                        r = 0.0
                dists[nd] = r
                nd += 1
            else:
                break
        if nd > n_i:
            ds = np.sort(dists[:nd])[::-1]
            out[i] = ds[:n_i].sum() / n_i
        else:
            out[i] = dists[:nd].sum() / nd
    return out


@nb.njit(parallel=True, cache=True)
def _neighbor_mean(emb, idx, start):
    """Mean of emb over idx[:, start:] for every row (Seurat PredictAssay on a reduction)."""
    n, k = idx.shape
    d = emb.shape[1]
    out = np.zeros((n, d))
    for i in nb.prange(n):
        for a in range(start, k):
            c = idx[i, a]
            for t in range(d):
                out[i, t] += emb[c, t]
        for t in range(d):
            out[i, t] /= (k - start)
    return out


@nb.njit(parallel=True, cache=True)
def _wnn_select(cands, n_cand, embs, offsets, dims, nearest, sigma, weight, k_nn, out_idx, out_aff):
    """Weighted affinities over the union of per-modality candidates; keep top k_nn (stable)."""
    n = cands.shape[0]
    M = nearest.shape[1]
    for i in nb.prange(n):
        c = n_cand[i]
        aff = np.zeros(c)
        for mm in range(M):
            off = offsets[mm]
            dm = dims[mm]
            for a in range(c):
                j = cands[i, a]
                acc = 0.0
                for t in range(dm):
                    df = embs[j, off + t] - embs[i, off + t]
                    acc += df * df
                r = np.sqrt(acc) - nearest[i, mm]
                if r < 0:
                    r = 0.0
                aff[a] += np.exp(-(r / sigma[i, mm])) * weight[i, mm]
        o = np.argsort(-aff[:c], kind="mergesort")  # decreasing, ties keep candidate order
        for t in range(k_nn):
            out_idx[i, t] = cands[i, o[t]]
            out_aff[i, t] = aff[o[t]]


def _union_candidates(nn_lists):
    """Per cell: union of neighbour lists in modality order (R ``Reduce(union, ...)``)."""
    n = nn_lists[0].shape[0]
    width = sum(l.shape[1] for l in nn_lists)
    cands = np.full((n, width), -1, dtype=np.int64)
    n_cand = np.zeros(n, dtype=np.int64)
    _union_kernel(np.hstack(nn_lists).astype(np.int64), cands, n_cand)
    return cands, n_cand


@nb.njit(parallel=True, cache=True)
def _union_kernel(allnn, cands, n_cand):
    n, w = allnn.shape
    for i in nb.prange(n):
        c = 0
        for a in range(w):
            j = allnn[i, a]
            dup = False
            for b in range(c):
                if cands[i, b] == j:
                    dup = True
                    break
            if not dup:
                cands[i, c] = j
                c += 1
        n_cand[i] = c


@dataclass
class WNNResult:
    """Output of :func:`wnn`.

    Attributes
    ----------
    weights : n x M modality weights (rows sum to 1; Seurat ``RNA.weight`` etc.).
    idx, dist : n x k weighted neighbours, self excluded (Seurat ``weighted.nn``).
    wknn : binary, symmetrized k-NN graph, diagonal 1.
    wsnn : pruned Jaccard SNN graph (cluster this).
    sigma : per-cell kernel bandwidths, per modality.
    nearest_dist : distance to the nearest neighbour, per modality.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> rna = arsc.pp.pca(arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))[0], n_comps=10).embedding
    >>> adt = arsc.pp.pca(arsc.pp.scale(arsc.pp.clr(ds.X("ADT", "counts")))[0], n_comps=4).embedding
    >>> w = arsc.wnn([rna, adt], k_nn=20, knn_range=50)
    >>> w.weights.shape, w.wsnn.shape
    ((3000, 2), (3000, 3000))
    """

    weights: np.ndarray
    idx: np.ndarray
    dist: np.ndarray
    wknn: sp.csr_matrix
    wsnn: sp.csr_matrix
    sigma: np.ndarray
    nearest_dist: np.ndarray


@records_seed("seed")
def wnn(embeddings: list, k_nn: int = 20, knn_range: int = 200, prune_snn: float = 1 / 15,
        sd_scale: float = 1.0, cross_constant=1e-4, l2_norm: bool = True, smooth: bool = False,
        knn_method: str = "auto", seed: int = 0) -> WNNResult:
    """Weighted nearest neighbours (Seurat ``FindMultiModalNeighbors`` / ``FindModalityWeights``).

    Parameters
    ----------
    embeddings : list of (n x d_m) arrays, one per modality (e.g. ``[harmony_rna[:, :30],
        harmony_adt[:, :15]]``).
    k_nn, knn_range, prune_snn, sd_scale, cross_constant, l2_norm, smooth : as in Seurat
        (``cross.contant.list`` -> ``cross_constant``, scalar or per modality).
    knn_method, seed : passed to :func:`knn` (``"auto"``: exact up to 50,000 cells, approximate above).

    Returns
    -------
    WNNResult with ``weights``, ``idx``, ``dist``, ``wknn``, ``wsnn``, ``sigma``, ``nearest_dist``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> rna = arsc.pp.pca(arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))[0], n_comps=10).embedding
    >>> adt = arsc.pp.pca(arsc.pp.scale(arsc.pp.clr(ds.X("ADT", "counts")))[0], n_comps=4).embedding
    >>> w = arsc.wnn([rna, adt], k_nn=20, knn_range=50)
    >>> labels = arsc.cluster(w.wsnn, resolution=0.5)
    """
    M = len(embeddings)
    if M < 2:
        raise ValueError("wnn needs at least two modalities: pass a list of embeddings, e.g. [rna_pcs, adt_pcs]")
    if len({np.asarray(e).shape[0] for e in embeddings}) != 1:
        raise ValueError("all embeddings must have the same cells (rows); got shapes "
                         f"{[np.asarray(e).shape for e in embeddings]}")
    cc = np.broadcast_to(np.asarray(cross_constant, dtype=np.float64), (M,))
    E = [l2_normalize(e) if l2_norm else np.asarray(e, dtype=np.float64) for e in embeddings]
    n = E[0].shape[0]
    # -- FindModalityWeights --------------------------------------------------
    big = [knn(e, max(knn_range, k_nn), method=knn_method, seed=seed) for e in E]  # self first
    nn = [(i[:, :k_nn], d[:, :k_nn]) for i, d in big]  # exact k-NN are prefixes of each other
    nearest = np.stack([d[:, 1] for _, d in nn], axis=1)  # Distances[, 2]
    # PredictAssay drops the self column when the query cell is its own first neighbour
    within = [_neighbor_mean(E[m], nn[m][0], 1) for m in range(M)]
    cross = {(m, m2): _neighbor_mean(E[m], nn[m2][0], 1) for m in range(M) for m2 in range(M) if m2 != m}

    def impute_dist(x, y, nd):
        return np.maximum(np.sqrt(((x - y) ** 2).sum(1)) - nd, 0.0)

    within_d = [impute_dist(E[m], within[m], nearest[:, m]) for m in range(M)]
    cross_d = {key: impute_dist(E[key[0]], v, nearest[:, key[0]]) for key, v in cross.items()}
    sigma = np.empty((n, M))
    for m in range(M):
        g = snn_graph(nn[m][0][:, :k_nn], prune=0.0)  # s.nn = k.nn, prune.SNN = 0
        sigma[:, m] = _snn_width(g.indptr.astype(np.int64), g.indices.astype(np.int64), g.data, E[m],
                                 k_nn, nearest[:, m]) * sd_scale
    scores = []
    for m in range(M):
        wk = np.exp(-within_d[m] / sigma[:, m])
        cols = []
        for m2 in range(M):
            if m2 == m:
                continue
            ck = np.exp(-cross_d[(m, m2)] / sigma[:, m])
            cols.append(np.clip(wk / (ck + cc[m]), 0, 200))
        sc = np.stack(cols, axis=1)
        if smooth:
            sc = np.stack([sc[nn[m][0][:, 1:], c].mean(1) for c in range(sc.shape[1])], axis=1)
        scores.append(sc)
    all_score = np.exp(np.hstack(scores)).sum(1)
    weights = np.stack([np.exp(s).sum(1) / all_score for s in scores], axis=1)
    # -- MultiModalNN -----------------------------------------------------------------
    rng_nn = [i[:, 1:knn_range] for i, _ in big]
    cands, n_cand = _union_candidates(rng_nn)
    embs = np.ascontiguousarray(np.hstack(E))
    dims = np.array([e.shape[1] for e in E], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(dims)[:-1]]).astype(np.int64)
    sel = np.empty((n, k_nn), dtype=np.int64)
    aff = np.empty((n, k_nn))
    _wnn_select(cands, n_cand, embs, offsets, dims, np.ascontiguousarray(nearest),
                np.ascontiguousarray(sigma), np.ascontiguousarray(weights), k_nn, sel, aff)
    dist = np.sqrt(np.clip((1 - aff) / 2, 0, 1))
    A = _nn_matrix(sel, n).tolil()
    A.setdiag(1)
    A = A.tocsr()
    AT = A.T.tocsr()
    wknn = (A + AT - AT.multiply(A)).tocsr()
    wknn.eliminate_zeros()
    wsnn = snn_graph(sel, prune=prune_snn)
    return WNNResult(weights, sel, dist, wknn, wsnn, sigma, nearest)
