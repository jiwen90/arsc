"""Nearest neighbours, SNN graphs and weighted nearest neighbours (WNN).

=======================  =====================================================
arsc                     Seurat equivalent
=======================  =====================================================
``knn``                  ``NNHelper`` / ``AnnoyNN`` (here: pynndescent / exact brute force)
``snn_graph``            ``ComputeSNN`` (Jaccard of k-NN sets, pruned)
``find_neighbors``       ``FindNeighbors`` (nn + snn graphs)
``wnn``                  ``FindMultiModalNeighbors`` (Hao et al. 2021)
=======================  =====================================================

Seurat uses Annoy (approximate, randomized). arsc, like scanpy, searches exactly up to
4,096 cells (scikit-learn brute force) and with pynndescent above (seeded);
method="exact" gives the exact graph at any size.
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
# Approximate search: pynndescent (BSD-2; scanpy's own large-n kNN; a hard dependency) with UMAP's / scanpy's
# settings. Exact search: scikit-learn's brute force (BLAS, as scanpy below 4,096 cells). arsc only re-scores the
# candidates in float64 and orders them by (distance, index).
AUTO_EXACT_MAX = 4096   # method="auto": exact up to this many points (scanpy's switch to approximate), approximate above
_EXTRA = 8              # exact: extra candidates per query beyond k for the float64 re-scoring
_U64 = 2.0 ** -53


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


def _knn_exact(X, Q, k):
    """scikit-learn brute force (|q|^2 + |x|^2 - 2 q.x, BLAS) for k + 8 candidates, then the float64 re-scoring; rows
    where rounding of that formula could have hidden a closer point (proven bound) are re-scored against all points."""
    from sklearn.neighbors import NearestNeighbors
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    Qc = Xc if Q is X else Q - mu
    k2 = min(n, k + _EXTRA)
    D, cand = NearestNeighbors(n_neighbors=k2, algorithm="brute").fit(Xc).kneighbors(Qc)
    idx, d2 = _rescore(Q, X, cand.astype(np.int64), k)
    if k2 < n:
        qn = (Qc ** 2).sum(1)
        xmax = float((Xc ** 2).sum(1).max())
        err = 8.0 * (d + 8) * _U64 * (qn + xmax) + 1e-12 * D[:, -1] ** 2
        bad = np.flatnonzero(D[:, -1] ** 2 - err <= d2[:, -1])
        allc = np.arange(n, dtype=np.int64)
        for a in range(0, bad.size, 256):
            rows = bad[a:a + 256]
            full = np.ascontiguousarray(np.broadcast_to(allc, (rows.size, n)))
            idx[rows], d2[rows] = _rescore(np.ascontiguousarray(Q[rows]), X, full, k)
    return idx, d2


def _knn_pynndescent(X, Q, k, seed):
    """pynndescent with UMAP's nearest_neighbors / scanpy's settings (n_trees, n_iters scale with n)."""
    from pynndescent import NNDescent
    from ._threads import get_threads
    n = X.shape[0]
    index = NNDescent(X, n_neighbors=k, metric="euclidean", random_state=seed, n_jobs=get_threads(),  # never -1
                      n_trees=min(64, 5 + int(round(n ** 0.5 / 20.0))), n_iters=max(5, int(round(np.log2(n)))),
                      low_memory=True, verbose=False)
    if Q is X:
        cand, _ = index.neighbor_graph
    else:
        cand, _ = index.query(Q, k=k)
    return _rescore(Q, X, np.ascontiguousarray(cand, dtype=np.int64), k)


@records_seed("seed")
def knn(X, k: int, query=None, *, method: str = "auto", seed: int = 0, n_clusters: int | None = None):
    """k nearest neighbours (Euclidean): exact up to 4,096 points, approximate (pynndescent) above (``"auto"``).

    ``method="exact"``: scikit-learn brute force (BLAS) for ``k + 8`` candidates, re-scored in float64 and ordered
    by (distance, index); a row whose candidate list could have missed a closer point through rounding (proven
    bound) is re-scored against every point. The answer is the exact float64 k-NN. Deterministic; the query point
    itself comes first when ``query`` is None.

    ``method="approximate"`` (alias ``"approx"``, ``"pynndescent"``): pynndescent's NN-descent with the settings of
    UMAP's ``nearest_neighbors`` that ``scanpy.pp.neighbors`` uses (``n_trees = min(64, 5 + sqrt(n) / 20)``,
    ``n_iters = max(5, log2 n)``), seeded with ``seed``; the distances reported are re-computed exactly (float64).
    Recall and downstream clusters vs scanpy: ``SCANPY_PARITY.md``.

    ``method="auto"`` (default): exact when n <= 4,096 or ``query`` is given, approximate above, as
    ``scanpy.pp.neighbors``. arsc 0.3.x used exact search at every size: pass ``method="exact"`` to reproduce
    those graphs.

    Parameters
    ----------
    X : (n x d) dense embedding (e.g. PCs).
    k : neighbours per query (self included when ``query`` is None); capped at n.
    query : optional (m x d) query points; default ``X`` itself.
    method : ``"auto"``, ``"exact"`` or ``"approximate"`` (``"approx"``, ``"pynndescent"``).
    seed : integer ``random_state`` of pynndescent (the exact method does not use it).
    n_clusters : ignored (arsc 0.3's exact-search partition count; kept so old calls still run).

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
    if method in ("approx", "pynndescent"):
        method = "approximate"
    if method == "auto":
        method = "exact" if (query is not None or n <= AUTO_EXACT_MAX) else "approximate"
    if method not in ("exact", "approximate"):
        raise ValueError(f"unknown method {method!r}; use 'auto', 'exact' or 'approximate'")
    if method == "exact":
        idx, d2 = _knn_exact(X, Q, k)
    else:
        idx, d2 = _knn_pynndescent(X, Q, k, seed)
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
@nb.njit(cache=True)
def _reverse_lists(idx, n):
    """For each cell m, the cells i (increasing) that have m among their neighbours (CSR; a neighbour listed twice
    in a row enters m's list once, as with scipy's sum_duplicates in earlier versions)."""
    m, k = idx.shape
    ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(m):
        for a in range(k):
            if not _seen_before(idx, i, a):
                ptr[idx[i, a] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    pos = ptr[:-1].copy()
    out = np.empty(ptr[n], dtype=np.int32)
    for i in range(m):
        for a in range(k):
            if not _seen_before(idx, i, a):
                j = idx[i, a]
                out[pos[j]] = i
                pos[j] += 1
    return ptr, out


@nb.njit(inline="always")
def _seen_before(idx, i, a):
    for b in range(a):
        if idx[i, b] == idx[i, a]:
            return True
    return False


@nb.njit(parallel=True, cache=True)
def _snn_blocks(idx, rev_ptr, rev_idx, n, smin, B, cap, buf_j, buf_s, rowcnt, blk_used, nt):
    """One pass over the shared-neighbour counts: block b (rows b*B..) writes its rows' pairs with at least smin
    shared neighbours, each row sorted by column, into its own region of buf_j / buf_s (cap entries per row of
    the block); blk_used[b] = entries written, or -1 when the region overflowed (the caller then counts first)."""
    nblk = (n + B - 1) // B
    for t in nb.prange(nt):
        cnt = np.zeros(n, dtype=np.int32)
        touched = np.empty(n, dtype=np.int32)
        for b in range(t, nblk, nt):
            base = b * B * cap
            lim = base + min(B, n - b * B) * cap
            w = base
            over = False
            for i in range(b * B, min(n, (b + 1) * B)):
                nt_ = 0
                for a in range(idx.shape[1]):
                    m = idx[i, a]
                    for p in range(rev_ptr[m], rev_ptr[m + 1]):
                        j = rev_idx[p]
                        if cnt[j] == 0:
                            touched[nt_] = j
                            nt_ += 1
                        cnt[j] += 1
                w0 = w
                for q in range(nt_):
                    j = touched[q]
                    s = cnt[j]
                    cnt[j] = 0
                    if s >= smin and not over:
                        if w < lim:
                            # insertion into the row's sorted run (rows have tens of entries)
                            u = w - 1
                            while u >= w0 and buf_j[u] > j:
                                buf_j[u + 1] = buf_j[u]
                                buf_s[u + 1] = buf_s[u]
                                u -= 1
                            buf_j[u + 1] = j
                            buf_s[u + 1] = s
                            w += 1
                        else:
                            over = True
                rowcnt[i] = w - w0
            blk_used[b] = -1 if over else w - base


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
    rev_ptr, rev_idx = _reverse_lists(idx, n)
    # s shared neighbours pass when s / (2k - s) >= prune, i.e. s >= 2k * prune / (1 + prune)
    smin = 1
    while smin <= k and not (smin / (k + (k - smin)) >= prune):
        smin += 1
    nt = nb.get_num_threads()
    B, cap = 2048, max(8, 2 * k)
    buf_j = np.empty(n * cap, dtype=np.int32)
    buf_s = np.empty(n * cap, dtype=np.int32)
    rowcnt = np.zeros(n, dtype=np.int64)
    nblk = (n + B - 1) // B
    used = np.zeros(nblk, dtype=np.int64)
    _snn_blocks(idx, rev_ptr, rev_idx, n, smin, B, cap, buf_j, buf_s, rowcnt, used, nt)
    if (used < 0).any():
        # some block had more pairs than its region: size every row exactly and fill again
        cap = int(_snn_rowmax(idx, rev_ptr, rev_idx, n, smin, nt)) + 1
        buf_j = np.empty(n * cap, dtype=np.int32)
        buf_s = np.empty(n * cap, dtype=np.int32)
        _snn_blocks(idx, rev_ptr, rev_idx, n, smin, B, cap, buf_j, buf_s, rowcnt, used, nt)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(rowcnt, out=ptr[1:])
    out_idx = np.empty(ptr[-1], dtype=np.int32)
    sv = np.empty(ptr[-1], dtype=np.int32)
    for b in range(nblk):
        r0 = b * B
        o = ptr[r0]
        out_idx[o:o + used[b]] = buf_j[r0 * cap:r0 * cap + used[b]]
        sv[o:o + used[b]] = buf_s[r0 * cap:r0 * cap + used[b]]
    del buf_j, buf_s
    out_val = sv / (k + (k - sv.astype(np.float64)))
    return sp.csr_matrix((out_val, out_idx, ptr), shape=(n, n))


@nb.njit(parallel=True, cache=True)
def _snn_rowmax(idx, rev_ptr, rev_idx, n, smin, nt):
    """Largest number of passing pairs in any row (sizes the buffers when a block overflowed)."""
    best = np.zeros(nt, dtype=np.int64)
    chunk = (n + nt - 1) // nt
    for t in nb.prange(nt):
        cnt = np.zeros(n, dtype=np.int32)
        touched = np.empty(n, dtype=np.int32)
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
            c = 0
            for q in range(nt_):
                if cnt[touched[q]] >= smin:
                    c += 1
                cnt[touched[q]] = 0
            best[t] = max(best[t], c)
    return best.max()


def _nn_matrix(idx: np.ndarray, n: int) -> sp.csr_matrix:
    m, k = idx.shape
    return sp.csr_matrix((np.ones(m * k), (np.repeat(np.arange(m), k), idx.ravel())), shape=(m, n))


@records_seed("seed")
def find_neighbors(emb: np.ndarray, k: int = 20, prune: float = 1 / 15, l2_norm: bool = False,
                   method: str = "auto", seed: int = 0) -> dict:
    """Seurat ``FindNeighbors``: k-NN (self included) and SNN graphs from an embedding.

    ``method`` and ``seed`` are passed to :func:`knn`: by default (``"auto"``) the k-NN is exact up to
    4,096 cells and approximate (pynndescent, scanpy's settings) above; ``method="exact"`` forces exact search.
    Seurat's Annoy is approximate, so SNN graphs differ slightly from Seurat's (see the README validation
    table).

    Parameters
    ----------
    emb : (n x d) embedding (PCs, harmony).
    k : neighbours including self (Seurat ``k.param``).
    prune : SNN Jaccard cutoff (Seurat ``prune.SNN``).
    l2_norm : L2-normalize rows first (Seurat ``l2.norm``).
    method : ``"auto"`` (default), ``"exact"`` or ``"approximate"`` (see :func:`knn`).
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
    knn_method, seed : passed to :func:`knn` (``"auto"``: exact up to 4,096 cells, approximate above).

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
