"""Nearest neighbours, SNN graphs and weighted nearest neighbours (WNN).

=======================  =====================================================
arsc                     Seurat equivalent
=======================  =====================================================
``knn``                  ``NNHelper`` / ``AnnoyNN`` (here: exact, BLAS + numba)
``snn_graph``            ``ComputeSNN`` (Jaccard of k-NN sets, pruned)
``find_neighbors``       ``FindNeighbors`` (nn + snn graphs)
``wnn``                  ``FindMultiModalNeighbors`` (Hao et al. 2021)
=======================  =====================================================

Seurat uses Annoy (approximate, randomized); arsc uses exact search by
default, so results are deterministic. Exact search is a few seconds for 260k
cells x 30 dims.
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
@nb.njit(parallel=True, cache=True)
def _heap_finish(hd, hi, out_idx, out_d2, row0):
    """Sort each row's heap by (distance, index), in place (insertion sort of k entries: no allocation inside the
    parallel loop, which contends on the Windows heap)."""
    m, k = hd.shape
    n_chunks = min(m, 256)
    step = (m + n_chunks - 1) // max(n_chunks, 1)
    for ch in nb.prange(n_chunks):
        for r in range(ch * step, min(m, (ch + 1) * step)):
            for t in range(k):
                out_d2[row0 + r, t] = hd[r, t]
                out_idx[row0 + r, t] = hi[r, t]
            for t in range(1, k):
                dv = out_d2[row0 + r, t]
                iv = out_idx[row0 + r, t]
                u = t - 1
                while u >= 0 and (out_d2[row0 + r, u] > dv or (out_d2[row0 + r, u] == dv and out_idx[row0 + r, u] > iv)):
                    out_d2[row0 + r, u + 1] = out_d2[row0 + r, u]
                    out_idx[row0 + r, u + 1] = out_idx[row0 + r, u]
                    u -= 1
                out_d2[row0 + r, u + 1] = dv
                out_idx[row0 + r, u + 1] = iv
            for t in range(k):
                if not out_d2[row0 + r, t] > 0:
                    out_d2[row0 + r, t] = 0.0


@nb.njit(inline="always")
def _heap_push(hd, hi, r, d, j, k):
    """Replace the top of row r's max-heap, keyed by (distance, index), with (d, j) and sift down."""
    pos = 0
    while True:
        c = 2 * pos + 1
        if c >= k:
            break
        if c + 1 < k and (hd[r, c + 1] > hd[r, c] or (hd[r, c + 1] == hd[r, c] and hi[r, c + 1] > hi[r, c])):
            c += 1
        if hd[r, c] < d or (hd[r, c] == d and hi[r, c] < j):
            break
        hd[r, pos] = hd[r, c]
        hi[r, pos] = hi[r, c]
        pos = c
    hd[r, pos] = d
    hi[r, pos] = j


@nb.njit(inline="always")
def _scan_cluster(Q, r, Xp, ids, starts, rpt, c, dcc, hd, hi, k, d):
    """Scan cluster c for query r. The cluster's points are sorted by their distance to its centre (``rpt``), so
    only the ring |rpt - dcc| <= current k-th distance can hold neighbours: binary search to its inner edge, stop
    past its outer edge. A point enters on (distance, index) order, so ties at the k-th place resolve to the
    smaller index whatever the scan order."""
    top = hd[r, 0]
    a = starts[c]
    e = starts[c + 1]
    if top < np.inf:
        lo_v = dcc - np.sqrt(top)
        b = e
        while a < b:
            m = (a + b) >> 1
            if rpt[m] < lo_v:
                a = m + 1
            else:
                b = m
    for p in range(a, e):
        b = rpt[p] - dcc
        if b > 0 and b * b > top:
            break
        if (dcc - rpt[p]) ** 2 > top:
            continue
        acc = 0.0
        for t in range(d):
            df = Q[r, t] - Xp[p, t]
            acc += df * df
        if acc < top or (acc == top and ids[p] < hi[r, 0]):
            _heap_push(hd, hi, r, acc, ids[p], k)
            top = hd[r, 0]


@nb.njit(parallel=True, cache=True)
def _knn_pruned(Q, Xp, ids, starts, centers, radius, rpt, hd, hi, n_seed, n_chunks):
    """Exact k-NN over a clustered reference set using triangle-inequality pruning.

    Per query: scan the ``n_seed`` clusters with the nearest centres first (fills
    the heap with good candidates), then sweep all remaining clusters, skipping
    any whose lower bound max(0, |q - c| - radius_c) exceeds the current k-th
    distance; single points are skipped when | |q - c| - |x - c| | exceeds it.
    No allocation inside the per-query loop (Windows heap contention).
    """
    m, d = Q.shape
    C = centers.shape[0]
    k = hd.shape[1]
    step = (m + n_chunks - 1) // n_chunks
    for ch in nb.prange(n_chunks):
        dc = np.empty(C)
        seen = np.zeros(C, dtype=np.bool_)
        seeds = np.empty(n_seed, dtype=np.int64)
        for r in range(ch * step, min(m, (ch + 1) * step)):
            for c in range(C):
                acc = 0.0
                for t in range(d):
                    df = Q[r, t] - centers[c, t]
                    acc += df * df
                dc[c] = np.sqrt(acc)
                seen[c] = False
            # nearest n_seed centres by repeated selection (n_seed small)
            ns = min(n_seed, C)
            for a in range(ns):
                best = -1
                bv = np.inf
                for c in range(C):
                    if not seen[c] and dc[c] < bv:
                        bv = dc[c]
                        best = c
                seen[best] = True
                seeds[a] = best
            for a in range(ns):
                c = seeds[a]
                _scan_cluster(Q, r, Xp, ids, starts, rpt, c, dc[c], hd, hi, k, d)
            for c in range(C):
                if seen[c]:
                    continue
                lb = dc[c] - radius[c]
                if lb > 0 and lb * lb > hd[r, 0]:
                    continue
                _scan_cluster(Q, r, Xp, ids, starts, rpt, c, dc[c], hd, hi, k, d)


def _partition(X, n_clusters, seed, iters=3, chunk=65536):
    """Cheap k-means partition (BLAS assignment); returns permuted data and cluster geometry."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    C = int(min(n_clusters, n))
    centers = X[rng.choice(n, C, replace=False)].copy()
    xn = (X * X).sum(1)
    assign = np.empty(n, dtype=np.int64)
    for it in range(iters + 1):
        cn = (centers * centers).sum(1)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            D = cn[None, :] - 2.0 * (X[a:b] @ centers.T)
            assign[a:b] = np.argmin(D, axis=1)
        if it == iters:
            break
        counts = np.bincount(assign, minlength=C)
        sums = np.zeros_like(centers)
        np.add.at(sums, assign, X)
        nz = counts > 0
        centers[nz] = sums[nz] / counts[nz, None]
    counts = np.bincount(assign, minlength=C)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    rpt_all = np.sqrt(((X - centers[assign]) ** 2).sum(1))
    order = np.lexsort((np.arange(n), rpt_all, assign))  # by cluster, then distance to its centre (ring search)
    Xp = np.ascontiguousarray(X[order])
    rpt = rpt_all[order]
    radius = np.zeros(C)
    np.maximum.at(radius, assign[order], rpt)
    return Xp, order.astype(np.int64), starts, centers, radius, rpt


@records_seed("seed")
def knn(X, k: int, query=None, *, method: str = "exact", n_clusters: int | None = None, seed: int = 0):
    """k nearest neighbours (Euclidean), exact by default.

    ``method="exact"`` (default): exact search in float64 with cluster-based
    triangle-inequality pruning (numba, parallel over queries). Deterministic;
    ties are ordered by index and the query point itself comes first when
    ``query`` is None. ``method="pynndescent"``: approximate (for very large data).

    ``seed`` (default 0): for ``"exact"`` it only seeds the k-means partition used for
    pruning, which never changes the (exact) answer; for ``"pynndescent"`` it is the
    ``random_state`` of the random-projection forest and NN-descent.

    Parameters
    ----------
    X : (n x d) dense embedding (e.g. PCs).
    k : neighbours per query (self included when ``query`` is None); capped at n.
    query : optional (m x d) query points; default ``X`` itself.
    method : ``"exact"`` or ``"pynndescent"``.
    n_clusters : partitions used for pruning (default ``2 * sqrt(n)``).
    seed : integer seed.

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
    if method != "exact":
        raise ValueError(f"unknown method {method!r}; use 'exact' or 'pynndescent'")
    if n_clusters is None:
        n_clusters = max(1, int(2 * np.sqrt(n)))
    Xp, ids, starts, centers, radius, rpt = _partition(X, n_clusters, seed)
    # Process queries in partition order: consecutive queries share candidate
    # clusters, so their data stays cache-resident (the scan is bandwidth-bound).
    if query is None:
        qorder = ids
        Qs = Xp
    else:
        cn = (centers * centers).sum(1)
        qa = np.argmin(cn[None, :] - 2.0 * (Q @ centers.T), axis=1)
        qorder = np.argsort(qa, kind="stable").astype(np.int64)
        Qs = np.ascontiguousarray(Q[qorder])
    m = Q.shape[0]
    hd = np.full((m, k), np.inf)
    hi = np.full((m, k), -1, dtype=np.int64)
    _knn_pruned(Qs, Xp, ids, starts, centers, radius, rpt, hd, hi, 8, max(1, 8 * nb.get_num_threads()))
    idx_s = np.empty((m, k), dtype=np.int64)
    d2_s = np.empty((m, k))
    _heap_finish(hd, hi, idx_s, d2_s, 0)
    idx = np.empty_like(idx_s)
    d2 = np.empty_like(d2_s)
    idx[qorder] = idx_s
    d2[qorder] = d2_s
    if query is None:
        # put self first if it tied with a duplicate point at distance 0
        rows = np.arange(m)
        is_self = idx == rows[:, None]
        self_pos = np.argmax(is_self, axis=1)
        fix = is_self.any(1) & (self_pos > 0) & (d2[rows, self_pos] <= d2[:, 0])
        for r in np.flatnonzero(fix):
            p = self_pos[r]
            idx[r, 1 : p + 1] = idx[r, 0:p].copy()
            idx[r, 0] = r
    return idx, np.sqrt(d2)


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
            touched[:nt_].sort()  # in place: no allocation inside the parallel loop (Windows heap contention)
            w = 0
            base = out_ptr[i]
            for q in range(nt_):
                j = touched[q]
                s = cnt[j]
                v = s / (k + (k - s))
                if v >= prune and v > 0:
                    if fill:
                        out_idx[base + w] = j
                        out_val[base + w] = v
                    w += 1
                cnt[j] = 0
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
                   method: str = "exact", seed: int = 0) -> dict:
    """Seurat ``FindNeighbors``: k-NN (self included) and SNN graphs from an embedding.

    ``seed`` is passed to :func:`knn` (matters only for ``method="pynndescent"``). The k-NN is
    exact; Seurat's Annoy is approximate, so SNN graphs differ slightly from Seurat's
    (see the README validation table).

    Parameters
    ----------
    emb : (n x d) embedding (PCs, harmony).
    k : neighbours including self (Seurat ``k.param``).
    prune : SNN Jaccard cutoff (Seurat ``prune.SNN``).
    l2_norm : L2-normalize rows first (Seurat ``l2.norm``).
    method : ``"exact"`` or ``"pynndescent"``.
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
        knn_method: str = "exact", seed: int = 0) -> WNNResult:
    """Weighted nearest neighbours (Seurat ``FindMultiModalNeighbors`` / ``FindModalityWeights``).

    Parameters
    ----------
    embeddings : list of (n x d_m) arrays, one per modality (e.g. ``[harmony_rna[:, :30],
        harmony_adt[:, :15]]``).
    k_nn, knn_range, prune_snn, sd_scale, cross_constant, l2_norm, smooth : as in Seurat
        (``cross.contant.list`` -> ``cross_constant``, scalar or per modality).
    knn_method, seed : passed to :func:`knn` (``seed`` matters only for ``"pynndescent"``).

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
