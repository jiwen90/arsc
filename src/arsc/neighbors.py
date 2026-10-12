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

__all__ = ["knn", "snn_graph", "find_neighbors", "wnn", "WNNResult", "l2_normalize"]


# --------------------------------------------------------------------------- kNN
@nb.njit(parallel=True, cache=True)
def _topk_rows(G, qn, xn, k, out_idx, out_d2, row0):
    """For rows of G (= Q X^T), select k smallest squared distances qn + xn - 2G."""
    m, n = G.shape
    for r in nb.prange(m):
        bi = np.empty(k, dtype=np.int64)
        bd = np.empty(k)
        cnt = 0
        worst = np.inf
        q = qn[r]
        for j in range(n):
            d = q + xn[j] - 2.0 * G[r, j]
            if cnt < k or d < worst:
                # insertion into sorted buffer (ascending; ties keep earlier j first)
                pos = cnt if cnt < k else k - 1
                while pos > 0 and bd[pos - 1] > d:
                    if pos < k:
                        bd[pos] = bd[pos - 1]
                        bi[pos] = bi[pos - 1]
                    pos -= 1
                bd[pos] = d
                bi[pos] = j
                if cnt < k:
                    cnt += 1
                worst = bd[cnt - 1] if cnt == k else np.inf
        for t in range(k):
            out_idx[row0 + r, t] = bi[t]
            out_d2[row0 + r, t] = bd[t] if bd[t] > 0 else 0.0


def knn(X, k: int, query=None, *, method: str = "exact", chunk: int = 2048, seed: int = 0):
    """k nearest neighbours (Euclidean). Returns ``(idx, dist)``, each (n_query x k).

    ``method="exact"`` (default) is brute force in float64 (GEMM + numba top-k):
    deterministic, and the query point itself comes first when ``query`` is
    ``None``. ``method="pynndescent"`` is approximate (for very large data).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Q = X if query is None else np.ascontiguousarray(query, dtype=np.float64)
    n = X.shape[0]
    k = int(min(k, n))
    if method == "pynndescent":
        from pynndescent import NNDescent
        index = NNDescent(X, n_neighbors=k, random_state=seed, n_jobs=-1)
        if query is None:
            idx, dist = index.neighbor_graph
        else:
            idx, dist = index.query(Q, k=k)
        return idx.astype(np.int64), dist.astype(np.float64)
    if method != "exact":
        raise ValueError(method)
    xn = (X * X).sum(1)
    qn = (Q * Q).sum(1)
    m = Q.shape[0]
    idx = np.empty((m, k), dtype=np.int64)
    d2 = np.empty((m, k))
    buf = np.empty((min(chunk, m), n))
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        G = buf[: b - a]
        np.matmul(Q[a:b], X.T, out=G)
        _topk_rows(G, qn[a:b], xn, k, idx, d2, a)
    if query is None:
        # put self first if it tied with a duplicate point at distance 0
        rows = np.arange(m)
        self_pos = np.argmax(idx == rows[:, None], axis=1)
        has_self = (idx == rows[:, None]).any(1)
        fix = has_self & (self_pos > 0) & (d2[rows, self_pos] <= d2[:, 0])
        for r in np.flatnonzero(fix):
            p = self_pos[r]
            idx[r, 1 : p + 1] = idx[r, 0:p].copy()
            idx[r, 0] = r
    return idx, np.sqrt(d2)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay 0 (Seurat ``L2Norm``)."""
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
            touched_sorted = np.sort(touched[:nt_])
            w = 0
            base = out_ptr[i]
            for q in range(nt_):
                j = touched_sorted[q]
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

    ``idx`` is the (n x k) neighbour index matrix (include self for FindNeighbors
    semantics). Returns a symmetric CSR (n x n).
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


def find_neighbors(emb: np.ndarray, k: int = 20, prune: float = 1 / 15, l2_norm: bool = False,
                   method: str = "exact") -> dict:
    """Seurat ``FindNeighbors``: k-NN (self included) and SNN graphs from an embedding.

    Returns ``{"idx", "dist", "nn" (binary csr), "snn" (csr)}``.
    """
    X = l2_normalize(emb) if l2_norm else np.asarray(emb, dtype=np.float64)
    idx, dist = knn(X, k, method=method)
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

    ``weights`` (n x M modality weights, rows sum to 1), ``idx``/``dist``
    (n x k weighted neighbours, self excluded, Seurat ``weighted.nn``),
    ``wknn`` (binary, symmetrized, diagonal 1), ``wsnn`` (pruned Jaccard SNN),
    ``sigma`` (per-cell kernel bandwidths), ``score`` (per-modality scores).
    """

    weights: np.ndarray
    idx: np.ndarray
    dist: np.ndarray
    wknn: sp.csr_matrix
    wsnn: sp.csr_matrix
    sigma: np.ndarray
    nearest_dist: np.ndarray


def wnn(embeddings: list, k_nn: int = 20, knn_range: int = 200, prune_snn: float = 1 / 15,
        sd_scale: float = 1.0, cross_constant=1e-4, l2_norm: bool = True, smooth: bool = False,
        knn_method: str = "exact") -> WNNResult:
    """Weighted nearest neighbours (Seurat ``FindMultiModalNeighbors`` / ``FindModalityWeights``).

    Parameters
    ----------
    embeddings : list of (n x d_m) arrays, one per modality (e.g. ``[harmony_rna[:, :30],
        harmony_adt[:, :15]]``).
    k_nn, knn_range, prune_snn, sd_scale, cross_constant, l2_norm, smooth : as in Seurat
        (``cross.contant.list`` -> ``cross_constant``, scalar or per modality).

    Returns
    -------
    WNNResult
    """
    M = len(embeddings)
    if M < 2:
        raise ValueError("need at least two modalities")
    cc = np.broadcast_to(np.asarray(cross_constant, dtype=np.float64), (M,))
    E = [l2_normalize(e) if l2_norm else np.asarray(e, dtype=np.float64) for e in embeddings]
    n = E[0].shape[0]
    # -- FindModalityWeights --------------------------------------------------
    nn = [knn(e, k_nn, method=knn_method) for e in E]  # self first
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
    rng_nn = [knn(e, knn_range, method=knn_method)[0][:, 1:] for e in E]
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
