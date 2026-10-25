"""Nearest neighbours, SNN graphs and weighted nearest neighbours (WNN).

=======================  =====================================================
arsc                     Seurat equivalent
=======================  =====================================================
``knn``                  ``NNHelper`` / ``AnnoyNN`` (here: clustered search, numba)
``snn_graph``            ``ComputeSNN`` (Jaccard of k-NN sets, pruned)
``find_neighbors``       ``FindNeighbors`` (nn + snn graphs)
``wnn``                  ``FindMultiModalNeighbors`` (Hao et al. 2021)
=======================  =====================================================

Seurat uses Annoy (approximate, randomized). arsc, like scanpy, searches exactly up to
4,096 cells and approximately above (its own clustered search, seeded); method="exact" gives
the exact graph at any size.
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


@nb.njit(inline="always")
def _kth_smallest(v, C, p, buf):
    """The p-th smallest of v[:C] (insertion into a sorted buffer of p values; no allocation)."""
    cnt = 0
    for c in range(C):
        x = v[c]
        if cnt == p and x >= buf[p - 1]:
            continue
        j = cnt if cnt < p else p - 1
        while j > 0 and buf[j - 1] > x:
            buf[j] = buf[j - 1]
            j -= 1
        buf[j] = x
        if cnt < p:
            cnt += 1
    return buf[p - 1]


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
        rowbuf = np.empty(max(1, n_probe))
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
                    pth[i] = _kth_smallest(dc[i], C, n_probe, rowbuf)
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


@nb.njit(cache=True)
def _centre_sums(S, lab, C):
    sums = np.zeros((C, S.shape[1]))
    cnt = np.zeros(C, dtype=np.int64)
    for i in range(S.shape[0]):
        c = lab[i]
        cnt[c] += 1
        for t in range(S.shape[1]):
            sums[c, t] += S[i, t]
    return sums, cnt


def _kmeans_centres(X, n_clusters, seed, iters=3, chunk=65536):
    """Cheap k-means: Lloyd iterations on a seeded sample of at most 64 points per centre (float32 BLAS
    assignment); only the partition geometry depends on it, never an exact answer."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    C = int(min(n_clusters, n))
    centers = X[rng.choice(n, C, replace=False)].copy()
    S = X if n <= 64 * C else X[np.sort(rng.choice(n, 64 * C, replace=False))]
    for it in range(iters):
        sa = _assign(S, centers, chunk)
        sums, counts = _centre_sums(S, sa, C)
        nz = counts > 0
        centers[nz] = sums[nz] / counts[nz, None]
    return centers


def _partition(X, n_clusters, seed, iters=3, chunk=65536):
    """k-means partition for the exact search: every point assigned to its nearest centre, points ordered by
    cluster and then by distance to the centre (ring search); returns permuted data and the geometry (float64)."""
    n = X.shape[0]
    centers = _kmeans_centres(X, n_clusters, seed, iters, chunk)
    C = centers.shape[0]
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


# Parameters of the approximate search (benchmark: SCANPY_PARITY.md, "k-NN backend")
AUTO_EXACT_MAX = 4096    # method="auto": exact up to this many points (scanpy's switch), approximate above
APPROX_CF = 2.0          # approximate: n_clusters = APPROX_CF * sqrt(n)
APPROX_PROBE = 24        # approximate: each query scans its APPROX_PROBE nearest clusters
EXACT_CF = 1.0           # exact: n_clusters = EXACT_CF * sqrt(n) (pruning only; never changes the answer)
_QB = 128                # queries per work item
_QB_APPROX = 256         # approximate: queries per work item (rows of the per-cluster matrix products)
_EXTRA = 8               # approximate: candidates kept beyond k for the exact float64 re-scoring


@nb.njit(parallel=True, cache=True)
def _top_p(G, cn, p, out):
    """out[i] = the p centres nearest to row i, nearest first (ties by index), from G = rows . centres:
    |x - c|^2 - |x|^2 = cn[c] - 2 G[i, c]. Insertion into a sorted buffer of p values, which almost every centre
    fails at the first comparison."""
    m, C = G.shape
    nb_ = min(m, 256)
    step = (m + nb_ - 1) // max(nb_, 1)
    for b in nb.prange(nb_):
        bv = np.empty(p, dtype=np.float32)
        bi = np.empty(p, dtype=np.int32)
        for i in range(b * step, min(m, (b + 1) * step)):
            cnt = 0
            for c in range(C):
                x = cn[c] - 2.0 * G[i, c]
                if cnt == p and x >= bv[p - 1]:
                    continue
                j = cnt if cnt < p else p - 1
                while j > 0 and bv[j - 1] > x:
                    bv[j] = bv[j - 1]
                    bi[j] = bi[j - 1]
                    j -= 1
                bv[j] = x
                bi[j] = c
                if cnt < p:
                    cnt += 1
            for j in range(p):
                out[i, j] = bi[j]


@nb.njit(parallel=True, cache=True)
def _ivf_items(Q, wq0, wq1, probes, R32, xn, starts, centers, ids, hd, hi, maxsz, qb, n_chunks):
    """Approximate search over work items (up to qb queries sharing a nearest centre). For every cluster that some
    query of the item probes: one float32 matrix product of those queries and the cluster's points, both centred on
    the cluster's centre (BLAS, single-threaded inside the parallel loop); |q - x|^2 ~ |q - c|^2 + |x - c|^2 -
    2 (q - c).(x - c) is offered to each query's heap. An item owns its queries' heaps (no write races)."""
    nw = wq0.size
    d = Q.shape[1]
    C = centers.shape[0]
    p = probes.shape[1]
    kk = hd.shape[1]
    step = (nw + n_chunks - 1) // n_chunks
    for ch in nb.prange(n_chunks):
        Qc = np.empty(qb * d, dtype=np.float32)
        G = np.empty(qb * maxsz, dtype=np.float32)
        qn = np.empty(qb, dtype=np.float32)
        loc = np.full(C, -1, dtype=np.int64)        # cluster -> slot in this item's list
        clist = np.empty(qb * p, dtype=np.int64)
        cnt = np.zeros(qb * p + 1, dtype=np.int64)
        rows = np.empty(qb * p, dtype=np.int64)
        fill = np.empty(qb * p, dtype=np.int64)
        for w_ in range(ch * step, min(nw, (ch + 1) * step)):
            s0 = wq0[w_]
            m = wq1[w_] - s0
            nc = 0
            for i in range(m):
                for j in range(p):
                    c = probes[s0 + i, j]
                    if loc[c] < 0:
                        loc[c] = nc
                        clist[nc] = c
                        cnt[nc + 1] = 0
                        nc += 1
                    cnt[loc[c] + 1] += 1
            cnt[0] = 0
            for u in range(nc):
                cnt[u + 1] += cnt[u]
                fill[u] = cnt[u]
            for i in range(m):
                for j in range(p):
                    u = loc[probes[s0 + i, j]]
                    rows[fill[u]] = i
                    fill[u] += 1
            for u in range(nc):
                c = clist[u]
                loc[c] = -1
                a = starts[c]
                sz = starts[c + 1] - a
                mr = cnt[u + 1] - cnt[u]
                if sz == 0 or mr == 0:
                    continue
                Qv = Qc[:mr * d].reshape((mr, d))
                for ii in range(mr):
                    i = rows[cnt[u] + ii]
                    acc = 0.0
                    for t in range(d):
                        v = Q[s0 + i, t] - centers[c, t]
                        Qv[ii, t] = v
                        acc += v * v
                    qn[ii] = acc
                Gv = G[:mr * sz].reshape((mr, sz))
                np.dot(Qv, R32[a:a + sz].T, Gv)
                for ii in range(mr):
                    r = s0 + rows[cnt[u] + ii]
                    base = qn[ii]
                    top = hd[r, 0]
                    for j in range(sz):
                        v = base + xn[a + j] - 2.0 * Gv[ii, j]
                        if v < top:
                            _heap_push(hd, hi, r, v, ids[a + j], kk)
                            top = hd[r, 0]


@nb.njit(parallel=True, cache=True)
def _rescore_sorted(Q, X, cand, k, out_idx, out_d2):
    """Exact float64 squared distances of each row's candidates (plain left-to-right sum, as brute force), the k
    best ordered by (distance, index); -1 candidates last."""
    m, kk = cand.shape
    d = Q.shape[1]
    for i in nb.prange(m):
        dv = np.empty(kk)
        iv = np.empty(kk, dtype=np.int64)
        for j in range(kk):
            c = cand[i, j]
            if c < 0:
                dv[j] = np.inf
                iv[j] = np.iinfo(np.int64).max
                continue
            s = 0.0
            for t in range(d):
                df = Q[i, t] - X[c, t]
                s += df * df
            dv[j] = s
            iv[j] = c
        for j in range(1, kk):
            a = dv[j]
            b = iv[j]
            u = j - 1
            while u >= 0 and (dv[u] > a or (dv[u] == a and iv[u] > b)):
                dv[u + 1] = dv[u]
                iv[u + 1] = iv[u]
                u -= 1
            dv[u + 1] = a
            iv[u + 1] = b
        for j in range(k):
            out_d2[i, j] = dv[j]
            out_idx[i, j] = iv[j] if dv[j] < np.inf else -1


def _ivf_search(X, Q, k, seed, n_clusters, n_probe, extra=_EXTRA):
    """Approximate search: every query scans all points of its n_probe nearest clusters (seeded k-means), with
    float32 matrix products per (group of queries, cluster); the k + ``extra`` best candidates are re-scored exactly
    in float64 and ordered by (distance, index)."""
    n, d = X.shape
    centers = _kmeans_centres(X, n_clusters, seed)
    C = centers.shape[0]
    p = int(min(C, n_probe))
    kk = int(min(n, k + extra))
    C32 = np.ascontiguousarray(centers, dtype=np.float32)
    cn = (C32.astype(np.float64) ** 2).sum(1).astype(np.float32)

    def nearest(A):
        pr = np.empty((A.shape[0], p), dtype=np.int32)
        for a in range(0, A.shape[0], 16384):
            b = min(A.shape[0], a + 16384)
            _top_p(A[a:b].astype(np.float32) @ C32.T, cn, p, pr[a:b])
        return pr

    pr_x = nearest(X)
    assign = pr_x[:, 0].astype(np.int64)
    ids = np.argsort(assign, kind="stable").astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(np.bincount(assign, minlength=C))]).astype(np.int64)
    Xp = np.ascontiguousarray(X[ids])
    if Q is X:
        qorder, Qs, qstarts = ids, Xp, starts
        probes = np.ascontiguousarray(pr_x[ids])
    else:
        pq = nearest(Q)
        qa = pq[:, 0].astype(np.int64)
        qorder = np.argsort(qa, kind="stable").astype(np.int64)
        Qs = np.ascontiguousarray(Q[qorder])
        qstarts = np.concatenate([[0], np.cumsum(np.bincount(qa, minlength=C))]).astype(np.int64)
        probes = np.ascontiguousarray(pq[qorder])
    del pr_x
    m = Qs.shape[0]
    own = np.repeat(np.arange(C), np.diff(starts))
    R32 = np.ascontiguousarray(Xp - centers[own], dtype=np.float32)
    xn = (R32.astype(np.float64) ** 2).sum(1).astype(np.float32)
    del own
    wq0, wq1, wc = _work_items(qstarts, _QB_APPROX)
    hd = np.full((m, kk), np.inf, dtype=np.float32)
    hi = np.full((m, kk), -1, dtype=np.int64)
    maxsz = int(max(1, np.diff(starts).max()))
    try:
        from threadpoolctl import threadpool_limits
        lim = threadpool_limits(limits=1, user_api="blas")   # BLAS calls inside numba threads: one thread each
    except Exception:  # noqa: BLE001 - threadpoolctl optional
        lim = None
    try:
        _ivf_items(Qs, wq0, wq1, probes, R32, xn, starts, centers, ids, hd, hi, maxsz, _QB_APPROX,
                   max(1, 8 * nb.get_num_threads()))
    finally:
        if lim is not None:
            lim.restore_original_limits()
    del R32, xn, probes
    idx_s = np.empty((m, k), dtype=np.int64)
    d2_s = np.empty((m, k))
    _rescore_sorted(Qs, X, hi, k, idx_s, d2_s)
    idx = np.empty_like(idx_s)
    d2 = np.empty_like(d2_s)
    idx[qorder] = idx_s
    d2[qorder] = d2_s
    return idx, d2


def _exact_search(X, Q, k, seed, n_clusters):
    """Exact search: k-means partition (only the speed depends on it), every cluster a candidate, skipped or narrowed
    by the triangle inequality; float32 filter with a proven margin, exact float64 distances."""
    n = X.shape[0]
    Xp, ids, starts, centers, radius, rpt = _partition(X, n_clusters, seed)
    C = centers.shape[0]
    if Q is X:
        qorder, Qs, qstarts = ids, Xp, starts
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
    _knn_blocks(Qs, wq0, wq1, wc, Xp, X32, xs, ids, starts, centers, np.ascontiguousarray(centers.T), radius, rpt,
                hd, hi, maxsz, _QB, max(1, 8 * nb.get_num_threads()), C, 1.0)
    del X32, xs
    idx_s = np.empty((m, k), dtype=np.int64)
    d2_s = np.empty((m, k))
    _heap_finish(hd, hi, idx_s, d2_s, 0)
    idx = np.empty_like(idx_s)
    d2 = np.empty_like(d2_s)
    idx[qorder] = idx_s
    d2[qorder] = d2_s
    return idx, d2


@records_seed("seed")
def knn(X, k: int, query=None, *, method: str = "auto", seed: int = 0, n_clusters: int | None = None):
    """k nearest neighbours (Euclidean): exact up to 4,096 points, approximate above (``"auto"``).

    Both methods start from a seeded k-means partition of the points (numba and BLAS, no other dependency) and
    report exact float64 distances (the plain sum of squared differences, as brute force), ordered by (distance,
    index).

    ``method="exact"``: each query scans clusters in order of their centre's distance, skipping a cluster or a
    ring of it when the triangle inequality proves it cannot hold a closer point; candidates pass a float32 filter
    with a proven rounding margin before the float64 check. The answer is the exact k-NN at any size; ``seed`` and
    ``n_clusters`` change only the speed. Deterministic; the query point itself comes first when ``query`` is None.

    ``method="approximate"`` (alias ``"approx"``): about 2 sqrt(n) clusters; each query scans every point of its
    24 nearest clusters. Distances to a cluster's points are float32 matrix products of the queries that probe it
    (grouped by nearest centre) and the cluster's points, both centred on the cluster's centre; the k + 8 best
    candidates are re-scored exactly in float64. Recall@k on real PCs: 0.96-0.996 (100k and 1M cells, k = 15 and
    30). Deterministic for a given ``seed`` and independent of the thread count. Speed against FAISS, hnswlib,
    Voyager, Annoy, pynndescent and scanpy: ``SCANPY_PARITY.md`` ("k-NN backend").

    ``method="auto"`` (default): exact when n <= 4,096 or ``query`` is given, approximate above, as
    ``scanpy.pp.neighbors``.

    Parameters
    ----------
    X : (n x d) dense embedding (e.g. PCs).
    k : neighbours per query (self included when ``query`` is None); capped at n.
    query : optional (m x d) query points; default ``X`` itself.
    method : ``"auto"``, ``"exact"`` or ``"approximate"`` (``"approx"``).
    seed : integer seed of the k-means partition.
    n_clusters : number of partitions (default about sqrt(n)).

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
        import warnings
        warnings.warn("method='pynndescent' is gone (arsc no longer depends on pynndescent); using arsc's own "
                      "approximate search", DeprecationWarning, stacklevel=2)
        method = "approximate"
    if method == "approx":
        method = "approximate"
    if method == "auto":
        method = "exact" if (query is not None or n <= AUTO_EXACT_MAX) else "approximate"
    if method not in ("exact", "approximate"):
        raise ValueError(f"unknown method {method!r}; use 'auto', 'exact' or 'approximate'")
    if method == "exact":
        nc = n_clusters or max(1, int(EXACT_CF * np.sqrt(n)))
        idx, d2 = _exact_search(X, Q, k, seed, nc)
    else:
        nc = n_clusters or max(1, int(APPROX_CF * np.sqrt(n)))
        idx, d2 = _ivf_search(X, Q, k, seed, nc, APPROX_PROBE)
        short = np.flatnonzero((idx < 0).any(1))      # a query whose probed clusters held fewer than k points
        if short.size:
            nc2 = max(1, int(EXACT_CF * np.sqrt(n)))
            idx[short], d2[short] = _exact_search(X, np.ascontiguousarray(Q[short]), k, seed, nc2)
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
    4,096 cells and approximate (arsc's clustered search) above; ``method="exact"`` forces exact search.
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
