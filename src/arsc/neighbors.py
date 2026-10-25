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


_U32 = 2.0 ** -24  # float32 unit roundoff
_SLACK = 1e-9      # relative slack on float64 pruning bounds (rounding of reassociated sums; only widens the search)


@nb.njit(fastmath=True)
def _dot_min(qc, X32, a, e, acc, xs):
    """float32 dot products of one centred query with the centred reference rows a..e (rows padded to a multiple
    of 8 so the reduction vectorises), and the min over rows of xs[p] - 2 * dot. ``acc`` gets the dots."""
    d = qc.size
    mn = np.inf
    for j in range(e - a):
        x = X32[a + j]
        s = np.float32(0.0)
        for t in range(d):
            s += qc[t] * x[t]
        acc[j] = s
        mn = min(mn, xs[a + j] - 2.0 * s)
    return mn


@nb.njit(fastmath=True)
def _centre_dists(Qb, centersT, out):
    """|q - c| for a block of queries vs all centres: sums of squared differences (no cancellation), vectorised
    over centres; used only for pruning, with slack."""
    m, d = Qb.shape
    C = centersT.shape[1]
    for i in range(m):
        for c in range(C):
            out[i, c] = 0.0
        for t in range(d):
            q = Qb[i, t]
            row = centersT[t]
            for c in range(C):
                df = q - row[c]
                out[i, c] += df * df
        for c in range(C):
            out[i, c] = np.sqrt(out[i, c])


@nb.njit(parallel=True, cache=True)
def _knn_blocks(Q, wq0, wq1, wc, Xp, X32, xs, ids, starts, centers, centersT, radius, rpt, hd, hi, maxsz, qb,
                n_chunks, n_probe, shrink):
    """Exact k-NN over a clustered reference set (triangle-inequality pruning, vectorised filter).

    Queries are sorted by their nearest centre and cut into work items Q[wq0[w]:wq1[w]] of at most ``qb`` queries
    that share that centre (wc[w]); ``n_chunks`` contiguous runs of items go to the threads. For each candidate
    cluster c, in order of its centre's distance to the item's centre (so the first ones fill the heaps): skip it
    when every query's bound max(0, |q - c| - radius_c) exceeds its k-th distance; else, per query, keep only the
    ring | |x - c| - |q - c| | <= k-th distance (points are sorted by |x - c|), compute float32 dot products of
    (q - c) with the ring's (x - c), vectorised over points, and give every point whose float32 squared distance
    lies within a proven rounding bound of the k-th distance the exact float64 check: the same sum of squared
    differences, in the same order, as a plain scan, so distances are bit-identical to brute force. A point enters
    on (distance, index) order, so the result does not depend on the scan order. Buffers are allocated once per
    chunk, never inside the per-query loops (Windows heap contention). ``n_probe`` < number of clusters scans, for
    each query, only its ``n_probe`` nearest clusters (approximate search)."""
    nw = wq0.size
    d = Q.shape[1]
    C = centers.shape[0]
    k = hd.shape[1]
    eps = 4.0 * (d + 8) * _U32
    step = (nw + n_chunks - 1) // n_chunks
    for ch in nb.prange(n_chunks):
        dc = np.empty((qb, C))
        acc = np.empty(maxsz, dtype=np.float32)
        qc = np.zeros(X32.shape[1], dtype=np.float32)
        cc = np.empty(C)
        corder = np.arange(C)
        pth = np.full(qb, np.inf)
        rowbuf = np.empty(C)
        prev = -1
        for w_ in range(ch * step, min(nw, (ch + 1) * step)):
            s0 = wq0[w_]
            s1 = wq1[w_]
            m = s1 - s0
            ca = wc[w_]
            if ca != prev:
                for c in range(C):
                    s = 0.0
                    for t in range(d):
                        df = centers[ca, t] - centers[c, t]
                        s += df * df
                    cc[c] = s
                corder = np.argsort(cc)
                prev = ca
            _centre_dists(Q[s0:s1], centersT, dc)
            if n_probe < C:
                for i in range(m):
                    for c in range(C):
                        rowbuf[c] = dc[i, c]
                    pth[i] = np.partition(rowbuf, n_probe - 1)[n_probe - 1]
            for oi in range(C):
                c = corder[oi]
                ac = starts[c]
                ec = starts[c + 1]
                if ec <= ac:
                    continue
                rc = radius[c]
                need = False
                for i in range(m):
                    if dc[i, c] > pth[i]:
                        continue
                    lb = dc[i, c] * (1.0 - _SLACK) - rc * (1.0 + _SLACK)
                    if not (lb > 0 and lb * lb > shrink * hd[s0 + i, 0]):
                        need = True
                        break
                if not need:
                    continue
                for i in range(m):
                    r = s0 + i
                    top = hd[r, 0]
                    dcc = dc[i, c]
                    if dcc > pth[i]:
                        continue
                    lb = dcc * (1.0 - _SLACK) - rc * (1.0 + _SLACK)
                    if lb > 0 and lb * lb > shrink * top:
                        continue
                    a = ac
                    e = ec
                    if top < np.inf:
                        wd = np.sqrt(shrink * top) * (1.0 + _SLACK) + _SLACK * (dcc + rc)
                        lo_v = dcc - wd
                        hi_v = dcc + wd
                        hb = e
                        while a < hb:
                            mid = (a + hb) >> 1
                            if rpt[mid] < lo_v:
                                a = mid + 1
                            else:
                                hb = mid
                        lo = a
                        hb = e
                        while lo < hb:
                            mid = (lo + hb) >> 1
                            if rpt[mid] <= hi_v:
                                lo = mid + 1
                            else:
                                hb = mid
                        e = lo
                    if e <= a:
                        continue
                    qq = 0.0
                    for t in range(d):
                        v = Q[r, t] - centers[c, t]
                        qc[t] = v
                        qq += v * v
                    thr = top - (1.0 - eps) * qq
                    if _dot_min(qc, X32, a, e, acc, xs) > thr:
                        continue
                    for j in range(e - a):
                        if xs[a + j] - 2.0 * acc[j] > thr:
                            continue
                        p = a + j
                        s = 0.0
                        for t in range(d):
                            df = Q[r, t] - Xp[p, t]
                            s += df * df
                        if s < top or (s == top and ids[p] < hi[r, 0]):
                            _heap_push(hd, hi, r, s, ids[p], k)
                            top = hd[r, 0]
                            thr = top - (1.0 - eps) * qq


# --------------------------------------------------------------------------- NN-descent (approximate)
@nb.njit(fastmath=True, inline="always")
def _dfast(Xf, i, j):
    """Squared distance with a reassociated (vectorised) sum over rows padded to a multiple of 8."""
    s = 0.0
    for t in range(Xf.shape[1]):
        df = Xf[i, t] - Xf[j, t]
        s += df * df
    return s


@nb.njit(inline="always")
def _d64(X, i, j):
    s = 0.0
    for t in range(X.shape[1]):
        df = X[i, t] - X[j, t]
        s += df * df
    return s


@nb.njit(inline="always")
def _try(X, Xf, i, l, hd, hi):
    """Offer point l to row i's heap (fast filter with a rounding margin, then the exact float64 sum in the
    plain order, so reported distances equal the exact method's; no duplicates). Returns 1 when it entered."""
    top = hd[i, 0]
    if _dfast(Xf, i, l) * (1.0 - 1e-12) > top:
        return 0
    s = _d64(X, i, l)
    if not (s < top or (s == top and l < hi[i, 0])):
        return 0
    k = hd.shape[1]
    for t in range(k):
        if hi[i, t] == l:
            return 0
    _heap_push(hd, hi, i, s, l, k)
    return 1


@nb.njit(parallel=True, cache=True)
def _nnd_round(X, Xf, oi, onew, rptr, ridx, rnew, hd, hi, stamp):
    """One pull-style NN-descent round. Row i reads only the previous graph (oi, its 'new' flags onew, and the
    reverse lists rptr/ridx with flags rnew) and writes only its own heap (hd, hi; started as a copy of row i),
    so the result does not depend on the thread count. Candidates: neighbours of neighbours, neighbours of
    reverse neighbours and reverse neighbours of neighbours, each pair skipped when neither hop is new; each
    candidate is evaluated once per row (``stamp``: one row of marks per thread)."""
    n, k = oi.shape
    upd = np.zeros(n, dtype=np.int64)
    for i in nb.prange(n):
        st = stamp[nb.get_thread_id()]
        mark = i + 1
        st[i] = mark
        for a in range(k):
            j = oi[i, a]
            if j >= 0:
                st[j] = mark
        u = 0
        for a in range(k):
            j = oi[i, a]
            if j < 0 or j == i:
                continue
            na = onew[i, a]
            for b in range(k):
                l = oi[j, b]
                if l < 0 or st[l] == mark or not (na or onew[j, b]):
                    continue
                st[l] = mark
                u += _try(X, Xf, i, l, hd, hi)
            for q in range(rptr[j], rptr[j + 1]):
                l = ridx[q]
                if st[l] == mark or not (na or rnew[q]):
                    continue
                st[l] = mark
                u += _try(X, Xf, i, l, hd, hi)
        for q in range(rptr[i], rptr[i + 1]):
            j = ridx[q]
            nq = rnew[q]
            for b in range(k):
                l = oi[j, b]
                if l < 0 or st[l] == mark or not (nq or onew[j, b]):
                    continue
                st[l] = mark
                u += _try(X, Xf, i, l, hd, hi)
        upd[i] = u
    return upd.sum()


@nb.njit(cache=True)
def _reverse(oi, onew):
    """Reverse lists of a k-NN graph (CSR, sources in increasing order) with the 'new' flag of each edge."""
    n, k = oi.shape
    cnt = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        for a in range(k):
            j = oi[i, a]
            if j >= 0 and j != i:
                cnt[j + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    pos = cnt[:-1].copy()
    ridx = np.empty(cnt[n], dtype=np.int64)
    rnew = np.empty(cnt[n], dtype=np.bool_)
    for i in range(n):
        for a in range(k):
            j = oi[i, a]
            if j >= 0 and j != i:
                ridx[pos[j]] = i
                rnew[pos[j]] = onew[i, a]
                pos[j] += 1
    return cnt, ridx, rnew


@nb.njit(parallel=True, cache=True)
def _new_flags(oi, hi, nf):
    """nf[i, a] = entry hi[i, a] was not in the previous row oi[i]."""
    n, k = hi.shape
    for i in nb.prange(n):
        for a in range(k):
            v = hi[i, a]
            f = True
            for b in range(k):
                if oi[i, b] == v:
                    f = False
                    break
            nf[i, a] = f


def _nn_descent(Xp, hd, hi, max_iter, delta):
    """Refine a start graph (heaps hd/hi, indices into Xp) by NN-descent until fewer than delta * n * k entries
    change in a round, or max_iter rounds."""
    n, k = hi.shape
    d = Xp.shape[1]
    Xf = np.zeros((n, -(-d // 8) * 8))
    Xf[:, :d] = Xp
    onew = np.ones((n, k), dtype=np.bool_)
    stamp = np.zeros((nb.config.NUMBA_NUM_THREADS, n), dtype=np.int64)
    rounds = 0
    for _ in range(max_iter):
        oi = hi.copy()
        rptr, ridx, rnew = _reverse(oi, onew)
        upd = _nnd_round(Xp, Xf, oi, onew, rptr, ridx, rnew, hd, hi, stamp)
        rounds += 1
        _new_flags(oi, hi, onew)
        if upd < delta * n * k:
            break
    return rounds


_QB = 128  # queries per work item
AUTO_EXACT_MAX = 50_000  # method="auto": exact search up to this many points, NN-descent above
APPROX_EPS = 0.0
NND_PROBE = 4            # NN-descent start graph: exact search within each point's nearest clusters
NND_MAX_ITER = 12
NND_DELTA = 0.001        # stop when fewer than delta * n * k graph entries change in a round (pynndescent's default)


def _work_items(qstarts, qb):
    """Cut each centre's query range into items of at most qb queries: (start, end, centre) arrays."""
    sz = np.diff(qstarts)
    nper = (sz + qb - 1) // qb
    wc = np.repeat(np.arange(sz.size, dtype=np.int64), nper)
    first = np.repeat(np.cumsum(nper) - nper, nper)
    j = np.arange(wc.size) - first
    wq0 = qstarts[wc] + j * qb
    wq1 = np.minimum(wq0 + qb, qstarts[wc + 1])
    return wq0.astype(np.int64), wq1.astype(np.int64), wc


def _assign(X, centers, chunk=65536):
    """Nearest centre of every row (float32 BLAS; only the partition geometry depends on it, never the answer)."""
    C32 = np.ascontiguousarray(centers, dtype=np.float32)
    cn = (C32.astype(np.float64) ** 2).sum(1).astype(np.float32)
    out = np.empty(X.shape[0], dtype=np.int64)
    for a in range(0, X.shape[0], chunk):
        b = min(X.shape[0], a + chunk)
        D = X[a:b].astype(np.float32) @ C32.T
        D *= -2.0
        D += cn
        out[a:b] = np.argmin(D, axis=1)
    return out


def _partition(X, n_clusters, seed, iters=3, chunk=65536):
    """Cheap k-means partition: Lloyd iterations on a sample of at most 64 points per centre, then one assignment
    of every point (float32 BLAS); returns permuted data and the cluster geometry (float64)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    C = int(min(n_clusters, n))
    centers = X[rng.choice(n, C, replace=False)].copy()
    S = X if n <= 64 * C else X[np.sort(rng.choice(n, 64 * C, replace=False))]
    for it in range(iters):
        sa = _assign(S, centers, chunk)
        counts = np.bincount(sa, minlength=C)
        sums = np.zeros_like(centers)
        np.add.at(sums, sa, S)
        nz = counts > 0
        centers[nz] = sums[nz] / counts[nz, None]
    assign = _assign(X, centers, chunk)
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
    if method == "auto":
        method = "exact" if (query is not None or n <= AUTO_EXACT_MAX) else "nndescent"
    if method not in ("exact", "nndescent"):
        raise ValueError(f"unknown method {method!r}; use 'auto', 'exact', 'nndescent' or 'pynndescent'")
    if method == "nndescent" and query is not None:
        method = "exact"   # NN-descent builds a graph on X itself; queries against X use the exact search
    if n_clusters is None:
        n_clusters = max(1, int(2 * np.sqrt(n)))
    Xp, ids, starts, centers, radius, rpt = _partition(X, n_clusters, seed)
    # Process queries in partition order: consecutive queries share candidate
    # clusters, so their data stays cache-resident.
    C = centers.shape[0]
    if query is None:
        qorder = ids
        Qs = Xp
        qstarts = starts
    else:
        cn = (centers * centers).sum(1)
        qa = np.argmin(cn[None, :] - 2.0 * (Q @ centers.T), axis=1)
        qorder = np.argsort(qa, kind="stable").astype(np.int64)
        Qs = np.ascontiguousarray(Q[qorder])
        qstarts = np.concatenate([[0], np.cumsum(np.bincount(qa, minlength=C))]).astype(np.int64)
    wq0, wq1, wc = _work_items(qstarts, _QB)
    own = np.repeat(np.arange(C), np.diff(starts))
    R = Xp - centers[own]
    dpad = -(-X.shape[1] // 8) * 8
    X32 = np.zeros((R.shape[0], dpad), dtype=np.float32)
    X32[:, :X.shape[1]] = R
    xs = (1.0 - 4.0 * (X.shape[1] + 8) * _U32) * (R * R).sum(1)
    del R
    maxsz = int(max(1, np.diff(starts).max()))
    m = Q.shape[0]
    hd = np.full((m, k), np.inf)
    hi = np.full((m, k), -1, dtype=np.int64)
    n_probe = C if method == "exact" else min(C, NND_PROBE)
    shrink = 1.0 if method == "exact" else 1.0 / (1.0 + APPROX_EPS) ** 2
    _knn_blocks(Qs, wq0, wq1, wc, Xp, X32, xs, ids, starts, centers, np.ascontiguousarray(centers.T), radius, rpt,
                hd, hi, maxsz, _QB, max(1, 8 * nb.get_num_threads()), n_probe, shrink)
    del X32, xs
    if method == "nndescent":
        inv = np.empty(n, dtype=np.int64)
        inv[ids] = np.arange(n)
        hp = np.where(hi >= 0, inv[np.maximum(hi, 0)], -1)
        _nn_descent(Xp, hd, hp, NND_MAX_ITER, NND_DELTA)
        hi = np.where(hp >= 0, ids[np.maximum(hp, 0)], -1)
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
