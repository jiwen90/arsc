"""UMAP (McInnes, Healy & Melville 2018) implemented in arsc: fuzzy simplicial set, spectral initialization and a
parallel layout optimizer whose result does not depend on the number of threads.

The graph construction follows umap-learn 0.5 arithmetic step for step (``smooth_knn_dist``,
``compute_membership_strengths``, the fuzzy union, the epoch pruning and ``make_epochs_per_sample``), so the fuzzy
graph agrees with umap-learn's on the same kNN (``SEURAT_PARITY.md``, UMAP section). The layout is arsc's own:

* **Deterministic parallel epochs.** Each epoch reads every other point from a snapshot taken at the start of the
  epoch. The edges are partitioned by their head point (CSR rows); one thread processes all edges of a row in
  order, moving the head point as umap-learn does (attraction to the tail, ``negative_sample_rate`` repulsions
  per sampled edge). The tail's move of each sampled edge (umap-learn's ``move_other``) is stored per edge and
  added after the epoch, per point, in edge order. A point's update therefore depends only on the snapshot and on
  its own edges, never on which thread ran it or when: the coordinates are identical at any thread count.
* **Per-edge random streams.** The negative samples of edge ``i`` at epoch ``n`` come from a counter-based
  generator (SplitMix64 finalizer of ``seed, i, n, p``, mapped to a point by a multiply-shift), not from a shared or per-thread state.

umap-learn's own parallel mode (``n_jobs > 1``, no ``random_state``) instead lets threads update the shared
coordinates in place without locks ("Hogwild"): the result depends on thread timing and differs run to run.
With a ``random_state`` umap-learn runs the layout on one thread. arsc gives a reproducible result on all threads.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import scipy.sparse as sp

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_FLOATMAX = np.finfo(np.float32).max
_N_BLOCKS = 1024  # row blocks per epoch; fixed, so the work split never depends on the thread count


# --------------------------------------------------------------------------- fuzzy simplicial set
@nb.njit(parallel=True, cache=True, locals={"psum": nb.types.float32, "lo": nb.types.float32,
                                            "mid": nb.types.float32, "hi": nb.types.float32})
def _smooth_knn_dist(distances, k, n_iter, local_connectivity, bandwidth, mean_distances):
    """umap-learn ``smooth_knn_dist`` (float32 search variables, same tolerance and floors); the global mean
    distance is passed in (computed once in a fixed order, so the result does not depend on the thread count)."""
    n, kk = distances.shape
    target = np.log2(k) * bandwidth
    rho = np.zeros(n, dtype=np.float32)
    result = np.zeros(n, dtype=np.float32)
    for i in nb.prange(n):
        lo = 0.0
        hi = NPY_FLOATMAX
        mid = 1.0
        n_nz = 0
        for j in range(kk):
            if distances[i, j] > 0.0:
                n_nz += 1
        if n_nz >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            # the index-th and (index+1)-th non-zero distances
            c = 0
            d_prev = np.float32(0.0)
            d_at = np.float32(0.0)
            for j in range(kk):
                if distances[i, j] > 0.0:
                    c += 1
                    if c == index:
                        d_prev = distances[i, j]
                    if c == index + 1:
                        d_at = distances[i, j]
                        break
            if index > 0:
                rho[i] = d_prev
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (d_at - d_prev)
            else:
                first = np.float32(0.0)
                for j in range(kk):
                    if distances[i, j] > 0.0:
                        first = distances[i, j]
                        break
                rho[i] = interpolation * first
        elif n_nz > 0:
            mx = np.float32(0.0)
            for j in range(kk):
                if distances[i, j] > 0.0 and distances[i, j] > mx:
                    mx = distances[i, j]
            rho[i] = mx
        for _ in range(n_iter):
            psum = 0.0
            for j in range(1, kk):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0
            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi >= NPY_FLOATMAX:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        if rho[i] > 0.0:
            s = 0.0
            for j in range(kk):
                s += distances[i, j]
            mean_ith = s / kk
            if result[i] < MIN_K_DIST_SCALE * mean_ith:
                result[i] = MIN_K_DIST_SCALE * mean_ith
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances
    return result, rho


@nb.njit(parallel=True, cache=True)
def _membership(knn_indices, knn_dists, sigmas, rhos, rows, cols, vals):
    n, k = knn_indices.shape
    for i in nb.prange(n):
        for j in range(k):
            t = i * k + j
            if knn_indices[i, j] == -1:
                continue
            if knn_indices[i, j] == i:
                val = np.float32(0.0)
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = np.float32(1.0)
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
            rows[t] = i
            cols[t] = knn_indices[i, j]
            vals[t] = val


def fuzzy_simplicial_set(knn_indices, knn_dists, n_neighbors: int | None = None, *, set_op_mix_ratio: float = 1.0,
                         local_connectivity: float = 1.0, return_sigmas: bool = False):
    """UMAP's fuzzy simplicial set (the weighted kNN graph the layout optimizes), from a kNN.

    Same arithmetic as umap-learn 0.5 ``fuzzy_simplicial_set`` with precomputed neighbors: distances in float32,
    per-point ``rho`` (distance to the nearest non-identical neighbor) and ``sigma`` (binary search so that the
    memberships sum to log2(k)), memberships ``exp(-(d - rho) / sigma)``, then the fuzzy union
    ``A + A^T - A * A^T``.

    Parameters
    ----------
    knn_indices : (n x k) neighbor indices, each point first (as :func:`arsc.knn` returns them).
    knn_dists : (n x k) distances.
    n_neighbors : k used for the target log2(k) (default ``knn_indices.shape[1]``).
    set_op_mix_ratio : 1 = fuzzy union (default), 0 = fuzzy intersection.
    local_connectivity : neighbors assumed connected (default 1).
    return_sigmas : also return ``(sigmas, rhos)``.

    Returns
    -------
    scipy.sparse.csr_matrix (n x n, float32), or ``(graph, sigmas, rhos)``.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> X = np.random.default_rng(0).normal(size=(300, 5))
    >>> idx, d = arsc.knn(X, 15)
    >>> G = arsc.fuzzy_simplicial_set(idx, d)
    >>> G.shape, bool(abs(G - G.T).max() == 0)
    ((300, 300), True)
    """
    idx = np.ascontiguousarray(knn_indices, dtype=np.int64)
    d = np.ascontiguousarray(knn_dists, dtype=np.float32)
    n, k = idx.shape
    if n_neighbors is None:
        n_neighbors = k
    mean_d = float(np.mean(d, dtype=np.float64))
    sig, rho = _smooth_knn_dist(d, float(n_neighbors), 64, float(local_connectivity), 1.0, mean_d)
    rows = np.zeros(idx.size, np.int32)
    cols = np.zeros(idx.size, np.int32)
    vals = np.zeros(idx.size, np.float32)
    _membership(idx, d, sig, rho, rows, cols, vals)
    result = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod = result.multiply(transpose)
    result = set_op_mix_ratio * (result + transpose - prod) + (1.0 - set_op_mix_ratio) * prod
    result = sp.csr_matrix(result)
    result.eliminate_zeros()
    if return_sigmas:
        return result, sig, rho
    return result


def find_ab_params(spread: float = 1.0, min_dist: float = 0.1):
    """Fit ``a, b`` of the low-dimensional similarity ``1 / (1 + a d^(2b))`` (umap-learn ``find_ab_params``:
    least squares on 300 points of [0, 3 spread] against the offset exponential).

    Parameters
    ----------
    spread : effective scale of embedded points.
    min_dist : minimum distance between embedded points.

    Returns
    -------
    tuple ``(a, b)`` of floats.

    Examples
    --------
    >>> from arsc._umap import find_ab_params
    >>> a, b = find_ab_params(1.0, 0.1)
    >>> round(a, 2), round(b, 2)
    (1.58, 0.9)
    """
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


# --------------------------------------------------------------------------- initialization
def _sign_fix(V):
    """Deterministic eigenvector signs: each column's largest-|value| entry positive."""
    s = np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
    s[s == 0] = 1
    return V * s


def _spectral_one(G, dim, seed):
    """Eigenvectors 2..dim+1 of the normalized Laplacian I - D^-1/2 G D^-1/2 of one connected graph.

    Computed as the largest eigenvectors of D^-1/2 G D^-1/2 (the same vectors; Lanczos converges in far fewer
    steps at the top of the spectrum than umap-learn's ``which="SM"`` at the bottom). Small graphs: dense."""
    n = G.shape[0]
    deg = np.asarray(G.sum(axis=0)).ravel()
    sqrt_deg = np.sqrt(deg)
    k = dim + 1
    Dm = sp.diags(1.0 / sqrt_deg)
    M = (Dm @ G @ Dm).tocsr()
    if n <= 2000:
        w, V = np.linalg.eigh(M.toarray())
        order = np.argsort(w)[::-1][1:k]
        return _sign_fix(V[:, order])
    from scipy.sparse.linalg import eigsh
    ncv = max(2 * k + 1, int(np.sqrt(n)))
    v0 = np.random.default_rng(seed).uniform(0.5, 1.5, n)
    w, V = eigsh(M, k, which="LA", ncv=ncv, tol=1e-4, v0=v0, maxiter=n * 5)
    order = np.argsort(w)[::-1][1:k]
    return _sign_fix(V[:, order])


def _dense_spectral(A, dim):
    """Spectral embedding of a small dense affinity matrix (scikit-learn ``SpectralEmbedding`` with a precomputed
    affinity: normalized Laplacian, eigenvectors divided by sqrt(degree), first dropped, signs fixed)."""
    deg = A.sum(1)
    dd = np.sqrt(deg)
    L = np.eye(A.shape[0]) - A / np.outer(dd, dd)
    w, V = np.linalg.eigh(L)
    V = V[:, np.argsort(w)[: dim + 1]]
    V = V / dd[:, None]
    V = V[:, 1: dim + 1]
    return _sign_fix(V)


def _multi_component(X, G, n_comp, labels, dim, rng, seed):
    """umap-learn ``multi_component_layout``: components placed by a spectral embedding of their centroids (or on
    the axes when there are at most 2 * dim of them), each laid out spectrally inside its own box."""
    result = np.empty((G.shape[0], dim), dtype=np.float32)
    if n_comp > 2 * dim and X is not None:
        cent = np.vstack([X[labels == c].mean(0) for c in range(n_comp)])
        D2 = ((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        meta = _dense_spectral(np.exp(-D2), dim)
        meta = meta / meta.max()
    elif n_comp > 2 * dim:
        meta = rng.random((n_comp, dim)) * 10.0
    else:
        k = int(np.ceil(n_comp / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta = np.vstack([base, -base])[:n_comp]
    G = G.tocsr()
    for c in range(n_comp):
        sel = np.flatnonzero(labels == c)
        dm = np.sqrt(((meta[c][None, :] - meta) ** 2).sum(1))
        data_range = dm[dm > 0.0].min() / 2.0
        if sel.size < 2 * dim or sel.size <= dim + 1:
            result[sel] = rng.uniform(-data_range, data_range, (sel.size, dim)) + meta[c]
        else:
            emb = _spectral_one(G[sel][:, sel], dim, seed + c)
            emb = emb * (data_range / np.max(np.abs(emb)))
            result[sel] = emb + meta[c]
    return result


def spectral_init(G, dim: int = 2, X=None, seed: int = 0) -> np.ndarray:
    """Spectral initialization of a UMAP layout (umap-learn ``spectral_layout``, components handled as
    ``multi_component_layout``), before scaling and noise.

    Parameters
    ----------
    G : symmetric weighted graph (n x n).
    dim : output dimensions.
    X : the input data (component centroids when the graph has more than 2 * dim components).
    seed : integer seed (eigensolver start vector, placement of tiny components).

    Returns
    -------
    numpy.ndarray (n x dim), float64 (float32 for several components).

    Examples
    --------
    >>> import numpy as np, arsc
    >>> from arsc._umap import spectral_init
    >>> X = np.random.default_rng(0).normal(size=(300, 4))
    >>> spectral_init(arsc.fuzzy_simplicial_set(*arsc.knn(X, 15)), 2).shape
    (300, 2)
    """
    from scipy.sparse.csgraph import connected_components
    n_comp, labels = connected_components(G)
    rng = np.random.default_rng(seed)
    # ARPACK / LAPACK call BLAS, whose threaded reductions change the last bits with the thread count (and through
    # the Lanczos iterations, the vectors): one BLAS thread keeps the result identical at any arsc thread count.
    with _one_blas_thread():
        if n_comp > 1:
            return _multi_component(X, G, n_comp, labels, dim, rng, seed)
        return _spectral_one(G, dim, seed)


class _one_blas_thread:
    def __enter__(self):
        try:
            from threadpoolctl import threadpool_limits
            self._c = threadpool_limits(limits=1, user_api="blas")
        except Exception:  # threadpoolctl absent: BLAS keeps its own setting
            self._c = None
        return self

    def __exit__(self, *a):
        if self._c is not None:
            self._c.restore_original_limits()


# --------------------------------------------------------------------------- layout
@nb.njit(inline="always")
def _mix64(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@nb.njit(parallel=True, fastmath=True, cache=True)
def _epoch_rows(Y, Y0, indptr, tails, eps, epns, next_s, next_neg, stamp, delta, n, alpha, a, b, gamma, key,
                n_blocks):
    nrows = indptr.size - 1
    dim = Y.shape[1]
    nv = np.uint64(Y0.shape[0])
    step = (nrows + n_blocks - 1) // n_blocks
    two_ab = np.float32(2.0 * a * b)
    a32 = np.float32(a)
    b32 = np.float32(b)
    rep = np.float32(2.0 * gamma * b)
    al = np.float32(alpha)
    fn = np.float64(n)
    un = np.uint64(n)
    for blk in nb.prange(n_blocks):
        cur = np.empty(dim, np.float32)
        for j in range(blk * step, min(nrows, (blk + 1) * step)):
            touched = False
            for i in range(indptr[j], indptr[j + 1]):
                if next_s[i] > fn:
                    continue
                if not touched:
                    for t in range(dim):
                        cur[t] = Y0[j, t]
                    touched = True
                k = tails[i]
                d2 = np.float32(0.0)
                for t in range(dim):
                    df = cur[t] - Y0[k, t]
                    d2 += df * df
                if d2 > 0.0:
                    p = d2 ** b32
                    gc = -two_ab * (p / d2) / (a32 * p + np.float32(1.0))
                else:
                    gc = np.float32(0.0)
                for t in range(dim):
                    g = gc * (cur[t] - Y0[k, t])
                    if g > 4.0:
                        g = np.float32(4.0)
                    elif g < -4.0:
                        g = np.float32(-4.0)
                    cur[t] += g * al
                    delta[i, t] = -g * al
                stamp[i] = n
                next_s[i] += eps[i]
                n_neg = int((fn - next_neg[i]) / epns[i])
                base = key ^ (np.uint64(i) * np.uint64(0x9E3779B97F4A7C15)) ^ (un * np.uint64(0xD1B54A32D192ED03))
                for q in range(n_neg):
                    h = _mix64(base + np.uint64(q) * np.uint64(0x8CB92BA72F3D8DD7)) >> np.uint64(32)
                    kk = np.int64((h * nv) >> np.uint64(32))  # uniform on 0..n-1 without a division
                    if kk == j:
                        continue
                    d2 = np.float32(0.0)
                    for t in range(dim):
                        df = cur[t] - Y0[kk, t]
                        d2 += df * df
                    if d2 > 0.0:
                        gc = rep / ((np.float32(0.001) + d2) * (a32 * d2 ** b32 + np.float32(1.0)))
                        for t in range(dim):
                            g = gc * (cur[t] - Y0[kk, t])
                            if g > 4.0:
                                g = np.float32(4.0)
                            elif g < -4.0:
                                g = np.float32(-4.0)
                            cur[t] += g * al
                next_neg[i] += n_neg * epns[i]
            if touched:
                for t in range(dim):
                    Y[j, t] = cur[t]


@nb.njit(parallel=True, cache=True)
def _apply_tail(Y, rptr, redge, stamp, delta, n, n_blocks):
    nv = rptr.size - 1
    dim = Y.shape[1]
    step = (nv + n_blocks - 1) // n_blocks
    for blk in nb.prange(n_blocks):
        for k in range(blk * step, min(nv, (blk + 1) * step)):
            for p in range(rptr[k], rptr[k + 1]):
                i = redge[p]
                if stamp[i] == n:
                    for t in range(dim):
                        Y[k, t] += delta[i, t]


def optimize_layout(Y, graph, n_epochs: int, a: float, b: float, seed: int, gamma: float = 1.0,
                    initial_alpha: float = 1.0, negative_sample_rate: float = 5.0, epochs_per_sample=None,
                    callback=None) -> np.ndarray:
    """Optimize a UMAP layout in place (deterministic parallel epochs; see the module docstring).

    Parameters
    ----------
    Y : (n x dim) float32 initial coordinates (modified in place and returned).
    graph : fuzzy graph already pruned (``coo`` or ``csr``; edges with weight 0 are ignored).
    n_epochs : epochs.
    a, b : curve parameters (:func:`find_ab_params`).
    seed : integer seed of the negative-sampling streams.
    gamma, initial_alpha, negative_sample_rate : umap-learn parameters.
    epochs_per_sample : optional precomputed array (edge order of the CSR).
    callback : optional ``callback(epoch, Y)`` after each epoch.

    Returns
    -------
    numpy.ndarray, ``Y``.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> from arsc._umap import optimize_layout, find_ab_params
    >>> X = np.random.default_rng(0).normal(size=(200, 3))
    >>> G = arsc.fuzzy_simplicial_set(*arsc.knn(X, 10))
    >>> Y = np.random.default_rng(1).uniform(0, 10, (200, 2)).astype(np.float32)
    >>> optimize_layout(Y, G, 20, *find_ab_params(1.0, 0.3), seed=0).shape
    (200, 2)
    """
    G = sp.csr_matrix(graph)
    G.eliminate_zeros()
    G.sort_indices()
    indptr = G.indptr.astype(np.int64)
    tails = G.indices.astype(np.int64)
    w = G.data.astype(np.float64)
    if epochs_per_sample is None:
        epochs_per_sample = make_epochs_per_sample(w, n_epochs)
    eps = np.asarray(epochs_per_sample, np.float64)
    epns = eps / negative_sample_rate
    next_s = eps.copy()
    next_neg = epns.copy()
    # reverse CSR: for each point, the edges whose tail it is, in edge order
    order = np.argsort(tails, kind="stable")
    rptr = np.zeros(G.shape[0] + 1, np.int64)
    np.cumsum(np.bincount(tails, minlength=G.shape[0]), out=rptr[1:])
    redge = order.astype(np.int64)
    stamp = np.full(tails.size, -1, np.int64)
    delta = np.zeros((tails.size, Y.shape[1]), np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    Y0 = np.empty_like(Y)
    key = np.uint64(_splitmix_seed(seed))
    alpha = float(initial_alpha)
    nb_ = min(_N_BLOCKS, max(1, G.shape[0]))
    for n in range(int(n_epochs)):
        Y0[:] = Y
        _epoch_rows(Y, Y0, indptr, tails, eps, epns, next_s, next_neg, stamp, delta, n, alpha, float(a),
                    float(b), float(gamma), key, nb_)
        _apply_tail(Y, rptr, redge, stamp, delta, n, nb_)
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        if callback is not None:
            callback(n, Y)
    return Y


def _splitmix_seed(seed: int) -> int:
    z = (int(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def make_epochs_per_sample(weights, n_epochs: int) -> np.ndarray:
    """umap-learn ``make_epochs_per_sample``: an edge of weight w is sampled every ``max(w) / w`` epochs.

    Parameters
    ----------
    weights : edge weights.
    n_epochs : epochs.

    Returns
    -------
    numpy.ndarray (float64), -1 for zero weights.

    Examples
    --------
    >>> from arsc._umap import make_epochs_per_sample
    >>> make_epochs_per_sample([1.0, 0.5, 0.25], 200).tolist()
    [1.0, 2.0, 4.0]
    """
    weights = np.asarray(weights, np.float64)
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


def embed(graph, n_components: int = 2, *, min_dist: float = 0.3, spread: float = 1.0, n_epochs: int | None = None,
          init="spectral", seed: int = 0, X=None, gamma: float = 1.0, initial_alpha: float = 1.0,
          negative_sample_rate: int = 5, a: float | None = None, b: float | None = None) -> np.ndarray:
    """UMAP layout of a fuzzy graph (umap-learn ``simplicial_set_embedding`` steps, arsc's layout optimizer).

    Steps: default epochs (500 up to 10,000 points, else 200); edges below ``max / n_epochs`` dropped; spectral
    (or random, or a given array) initialization scaled so the largest |coordinate| is 10, plus N(0, 1e-4) noise;
    rescaled to [0, 10] per axis; :func:`optimize_layout`.

    Parameters
    ----------
    graph : fuzzy graph (:func:`fuzzy_simplicial_set`).
    n_components : output dimensions.
    min_dist, spread : curve parameters (or pass ``a``, ``b``).
    n_epochs : epochs (default as umap-learn).
    init : ``"spectral"``, ``"random"`` or an (n x n_components) array.
    seed : integer seed.
    X : input data (only for placing several graph components).
    gamma, initial_alpha, negative_sample_rate : umap-learn parameters.
    a, b : curve parameters overriding ``min_dist`` / ``spread``.

    Returns
    -------
    numpy.ndarray (n x n_components) float32.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> from arsc._umap import embed
    >>> X = np.random.default_rng(0).normal(size=(300, 4))
    >>> embed(arsc.fuzzy_simplicial_set(*arsc.knn(X, 15)), n_epochs=50, seed=0).shape
    (300, 2)
    """
    G = sp.coo_matrix(graph)
    G.sum_duplicates()
    n = G.shape[0]
    default_epochs = 500 if n <= 10000 else 200
    if n_epochs is None:
        n_epochs = default_epochs
    n_epochs = int(n_epochs)
    G = G.copy()
    if n_epochs > 10:
        G.data[G.data < (G.data.max() / float(n_epochs))] = 0.0
    else:
        G.data[G.data < (G.data.max() / float(default_epochs))] = 0.0
    G.eliminate_zeros()
    G = G.tocsr()
    G.sort_indices()
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    rng = np.random.default_rng(seed)
    if isinstance(init, str) and init == "random":
        Y = rng.uniform(-10.0, 10.0, (n, n_components)).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        Y = spectral_init(G, n_components, X=None if X is None else np.asarray(X, np.float64), seed=seed)
        Y = (Y * (10.0 / np.abs(Y).max())).astype(np.float32)
        Y = Y + rng.normal(scale=0.0001, size=Y.shape).astype(np.float32)
    elif isinstance(init, str):
        raise ValueError(f"unknown init {init!r}; use 'spectral', 'random' or an array")
    else:
        Y = np.asarray(init, np.float32)
        if Y.shape != (n, n_components):
            raise ValueError(f"init array has shape {Y.shape}, expected {(n, n_components)}")
    mn, mx = Y.min(0), Y.max(0)
    Y = (10.0 * (Y - mn) / (mx - mn)).astype(np.float32, order="C")
    Y = optimize_layout(Y, G, n_epochs, a, b, seed, gamma, initial_alpha, negative_sample_rate)
    deg = np.asarray(sp.csr_matrix(graph).sum(1)).ravel()
    if (deg == 0).any():
        Y[deg == 0] = np.nan
    return Y
