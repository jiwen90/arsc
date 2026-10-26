"""Harmony batch integration -- a faithful port of R ``harmony`` 2.0.5 (CRAN), 2.1.0 and 1.2.0.

``run_harmony(version=...)`` reproduces one R release; the differences between them are listed in its
docstring. Common structure (``RunHarmony.default`` and the C++ ``harmony`` class: setup ->
``init_cluster_cpp`` -> repeat {``cluster_cpp``, ``moe_correct_ridge_cpp``} until ``check_convergence(1)``):

* ``theta = 2`` per variable, ``sigma = 0.1``, ``nclust = min(round(N / 30), 100)``, ``max_iter = 10``
  harmony rounds, ``epsilon_cluster = 1e-3``, ``epsilon_harmony = 1e-2``, ``block_size = 0.05``, ``tau = 0``;
  ``lambda`` 1 (1.2) or automatic ``alpha * E`` (2.x); ``max_iter_cluster`` 20 (1.2) or 4 (2.x).
* clustering: online updates of R in ``ceil(1 / block_size)`` random blocks with a diversity penalty per
  batch variable, combined over variables by ``penalty %*% Phi`` (a sum; 1.2, 2.0) or a product (2.1).
* objective = (k-means error + sigma * entropy + sigma * cross-entropy) * 2000/N, kept as float32 like the
  C++ (2.x: the serial float32 sums of ``my_accu``), so convergence decisions match.
* correction: per-cluster ridge regression on ``[1, Phi]`` with the intercept left in (``W[0] = 0``) using the
  *original* embedding (2.x: only the batch levels present in the cluster).

Differences that cannot be matched bit-for-bit: R's k-means / shuffling use R's RNG; pass ``init_centroids``
(e.g. from R) to start from identical centroids. The deterministic correction step reproduces R to ~1e-12 given
the same R matrix (see the validation table in README.md). Note ``block_size = 1`` is degenerate in harmony 1.2
(all cells are removed at once, so E = O = 0 and E / (O + E) is 0/0; R returns NaN); it is rejected for 1.2.
The R 2.x build on Windows uses double precision (no single-precision BLAS in R's Rblas), as this port does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import pandas as pd

from ._utils import check_seed
from ._detla import matmul, single_thread
from ._seedlog import records_seed

# Fixed reduction-chunk counts: per-chunk partial sums are combined in chunk order, so the float summation
# order (and hence every objective value / convergence decision) is the same for any number of threads.
# (They used to follow the thread count; 32 / min(8, threads) == these values on the 32-thread lab machine.)
_OBJ_CHUNKS = 32
_R_CHUNKS = 8

__all__ = ["run_harmony", "Harmony", "HarmonyResult"]


# --------------------------------------------------------------------------- kernels
@nb.njit(parallel=True, cache=True, error_model="numpy")
def _scale_dist(dist, sigma, out):
    """out[i, k] = exp(-dist[i, k] / sigma[k]) normalized to sum 1 per cell."""
    N, K = dist.shape
    for i in nb.prange(N):
        s = 0.0
        for k in range(K):
            v = math.exp(-dist[i, k] / sigma[k])
            out[i, k] = v
            s += v
        for k in range(K):
            out[i, k] /= s


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _block_remove_add(R, order, s, e, ET, OT, Pr_b, codes, sign, nchunk):
    """Remove (sign=-1) or add (sign=+1) a block of cells from E and O (stored B x K)."""
    K = R.shape[1]
    B = ET.shape[0]
    V = codes.shape[1]
    n = e - s
    part_O = np.zeros((nchunk, B, K))
    part_s = np.zeros((nchunk, K))
    step = (n + nchunk - 1) // nchunk
    for c in nb.prange(nchunk):
        lo = s + c * step
        hi = min(e, lo + step)
        for t in range(lo, hi):
            i = order[t]
            for k in range(K):
                part_s[c, k] += R[i, k]
            for v in range(V):
                b = codes[i, v]
                for k in range(K):
                    part_O[c, b, k] += R[i, k]
    for c in range(nchunk):
        for b in range(B):
            for k in range(K):
                OT[b, k] += sign * part_O[c, b, k]
    for k in range(K):
        rs = 0.0
        for c in range(nchunk):
            rs += part_s[c, k]
        for b in range(B):
            ET[b, k] += sign * rs * Pr_b[b]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _block_recompute(R, sd, order, s, e, PT, codes, multiply):
    """R[i] = sd[i] * penalty, L1-normalized. The per-variable penalties are summed over the batch variables
    (harmony 1.2 and 2.0: ``penalty %*% Phi``) or multiplied (harmony 2.1, ``multiply=True``)."""
    K = R.shape[1]
    V = codes.shape[1]
    for t in nb.prange(s, e):
        i = order[t]
        tot = 0.0
        for k in range(K):
            if multiply:
                acc = 1.0
                for v in range(V):
                    acc *= PT[codes[i, v], k]
            else:
                acc = 0.0
                for v in range(V):
                    acc += PT[codes[i, v], k]
            val = sd[i, k] * acc
            R[i, k] = val
            tot += val
        for k in range(K):
            R[i, k] /= tot


@nb.njit(cache=True, error_model="numpy")
def _penalty(ET, OT, theta, PT, v2):
    """Diversity penalty per (batch level, cluster): ``(E / (O + E)) ** theta`` (harmony 1.2) or the
    pseudocount form ``((2E + 1) / (O + E + 1)) ** theta`` (harmony 2.x, ``v2=True``)."""
    B, K = ET.shape
    for b in range(B):
        for k in range(K):
            if v2:
                PT[b, k] = ((2.0 * ET[b, k] + 1.0) / (OT[b, k] + ET[b, k] + 1.0)) ** theta[b]
            else:
                PT[b, k] = (ET[b, k] / (OT[b, k] + ET[b, k])) ** theta[b]


def _update_R(R, sd, order, bounds, ET, OT, Pr_b, theta, codes, nchunk, v2=False, multiply=False):
    """Online R update over blocks (harmony::update_R). Python driver: numba
    parallel regions launched from a non-parallel njit caller are much slower."""
    PT = np.empty_like(ET)
    for blk in range(bounds.size - 1):
        s, e = int(bounds[blk]), int(bounds[blk + 1])
        _block_remove_add(R, order, s, e, ET, OT, Pr_b, codes, -1.0, nchunk)
        _penalty(ET, OT, theta, PT, v2)
        _block_recompute(R, sd, order, s, e, PT, codes, multiply)
        _block_remove_add(R, order, s, e, ET, OT, Pr_b, codes, 1.0, nchunk)


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _objective_terms(R, dist, sigma, codes, B, nchunk):
    """Return (sum R*dist, sum sigma_k R log R, O_true^T (B x K)) in one pass."""
    N, K = R.shape
    V = codes.shape[1]
    step = (N + nchunk - 1) // nchunk
    km = np.zeros(nchunk)
    ent = np.zeros(nchunk)
    part_O = np.zeros((nchunk, B, K))
    for c in nb.prange(nchunk):
        lo = c * step
        hi = min(N, lo + step)
        for i in range(lo, hi):
            for k in range(K):
                r = R[i, k]
                km[c] += r * dist[i, k]
                if r > 0:
                    ent[c] += r * math.log(r) * sigma[k]
            for v in range(V):
                b = codes[i, v]
                for k in range(K):
                    part_O[c, b, k] += R[i, k]
    OT = np.zeros((B, K))
    for c in range(nchunk):
        for b in range(B):
            for k in range(K):
                OT[b, k] += part_O[c, b, k]
    return km.sum(), ent.sum(), OT


@nb.njit(cache=True, error_model="numpy")
def _objective_v2(R, dist, sigma, Lmat, codes):
    """harmony 2.x ``compute_objective``: the three K x N term matrices are each summed by ``my_accu``, a serial
    float32 accumulation in column-major order (cells outer, clusters inner); returns the three float32 sums."""
    N, K = R.shape
    V = codes.shape[1]
    km = np.float32(0.0)
    ent = np.float32(0.0)
    cross = np.float32(0.0)
    for i in range(N):
        for k in range(K):
            r = R[i, k]
            km = np.float32(np.float64(km) + r * dist[i, k])
            e = r * math.log(r) if r > 0 else 0.0
            ent = np.float32(np.float64(ent) + e * sigma[k])
            lp = 0.0
            for v in range(V):
                lp += Lmat[k, codes[i, v]]
            cross = np.float32(np.float64(cross) + (r * sigma[k]) * lp)
    return km, ent, cross


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _unkept_sums(R, Z, codes, keep):
    """Per cluster k: sum of R[i, k] and of R[i, k] * Z[i] over the cells i none of whose batch levels is kept
    for k (``keep`` K x B). harmony 2.x drops these cells from cluster k's ridge regression (and its intercept)."""
    N, K = R.shape
    d = Z.shape[1]
    V = codes.shape[1]
    rs = np.zeros(K)
    rz = np.zeros((K, d))
    for k in nb.prange(K):
        for i in range(N):
            hit = False
            for v in range(V):
                if keep[k, codes[i, v]]:
                    hit = True
                    break
            if not hit:
                r = R[i, k]
                rs[k] += r
                for j in range(d):
                    rz[k, j] += r * Z[i, j]
    return rs, rz


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ridge_gram(R, codes1, nb1):
    """M[k] = Phi_moe diag(R[:, k]) Phi_moe^T for Phi_moe = [1; Phi] (codes1 include intercept 0)."""
    N, K = R.shape
    A = codes1.shape[1]
    M = np.zeros((K, nb1, nb1))
    for k in nb.prange(K):
        for i in range(N):
            r = R[i, k]
            for a in range(A):
                ca = codes1[i, a]
                for b in range(A):
                    M[k, ca, codes1[i, b]] += r
    return M


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _rz(R, Z, codes1, nb1):
    """RZ[k, a, :] = sum_i R[i, k] * Phi_moe[a, i] * Z[i, :]."""
    N, K = R.shape
    d = Z.shape[1]
    A = codes1.shape[1]
    out = np.zeros((K, nb1, d))
    for k in nb.prange(K):
        for i in range(N):
            r = R[i, k]
            for a in range(A):
                ca = codes1[i, a]
                for j in range(d):
                    out[k, ca, j] += r * Z[i, j]
    return out


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _apply_correction(Z_orig, R, W, codes1, out):
    """out[i] = Z_orig[i] - sum_k R[i, k] * sum_{a in active(i), a > 0} W[k, a]."""
    N, K = R.shape
    d = Z_orig.shape[1]
    A = codes1.shape[1]
    for i in nb.prange(N):
        for j in range(d):
            out[i, j] = Z_orig[i, j]
        for k in range(K):
            r = R[i, k]
            for a in range(1, A):
                ca = codes1[i, a]
                for j in range(d):
                    out[i, j] -= r * W[k, ca, j]


def _l2_rows(X):
    n = np.sqrt((X * X).sum(1, keepdims=True))
    return X / n


# --------------------------------------------------------------------------- core class
@dataclass
class HarmonyResult:
    """Result of :func:`run_harmony` with ``return_object=True``: corrected embedding plus final state and traces.

    Attributes
    ----------
    Z_corr : (n_cells, d) corrected embedding (what :func:`run_harmony` returns by default).
    R : (n_cells, nclust) soft cluster assignments (cells first; R harmony stores the transpose).
    Y : (nclust, d) cluster centroids (L2-normalized rows).
    objective_harmony : objective after each harmony round.
    objective_kmeans : objective after each clustering iteration.
    kmeans_rounds : clustering iterations per harmony round.
    converged : whether the harmony objective converged before ``max_iter``.
    batch_levels : ``{variable: levels}`` in R factor order.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> res = arsc.run_harmony(pcs, ds.obs, "patient", return_object=True)
    >>> res.Z_corr.shape, list(res.batch_levels)
    ((3000, 10), ['patient'])
    """

    Z_corr: np.ndarray
    R: np.ndarray
    Y: np.ndarray
    objective_harmony: list = field(default_factory=list)
    objective_kmeans: list = field(default_factory=list)
    kmeans_rounds: list = field(default_factory=list)
    converged: bool = False
    batch_levels: dict = field(default_factory=dict)


class Harmony:
    """Stateful Harmony object (port of the C++ class of R harmony 1.2 and 2.x). Most users want :func:`run_harmony`.

    Parameters
    ----------
    Z : (n_cells, d) embedding.
    codes : (n_cells, n_variables) integer batch codes, global across variables (variable v's levels follow
        variable v-1's).
    B_vec : number of levels per variable.
    theta : per-level diversity penalty (length sum(B_vec)).
    sigma : per-cluster soft k-means width (length ``K``).
    lambda_vec : ridge penalties ``[0, per level...]``, or ``None`` for automatic lambda.
    alpha : automatic-lambda factor (default 0.2).
    max_iter_cluster : clustering iterations per round (default 20).
    epsilon_cluster, epsilon_harmony : convergence tolerances.
    K : number of clusters.
    block_size : fraction of cells per online-update block (default 0.05; must be < 1 for version 1.2).
    seed : integer seed for k-means init and block order.
    version : R harmony release to reproduce: ``"1.2"`` (default here), ``"2.0"`` or ``"2.1"`` (see :func:`run_harmony`).
    batch_prop_cutoff : 2.x: minimum share of a batch level in a cluster for it to enter that cluster's regression.

    Attributes
    ----------
    Z_corr : corrected embedding after :meth:`harmonize`; ``R``, ``Y``, objective traces as in
    :class:`HarmonyResult`.

    Examples
    --------
    >>> import numpy as np
    >>> from arsc.harmony import Harmony
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.normal(size=(200, 5)); codes = (np.arange(200) % 2)[:, None]
    >>> h = Harmony(Z, codes, [2], theta=np.full(2, 2.0), sigma=np.full(5, 0.1), lambda_vec=np.array([0, 1.0, 1.0]), K=5)
    >>> h.init_cluster()
    >>> _ = h.harmonize(3)
    >>> h.Z_corr.shape
    (200, 5)
    """

    window_size = 3

    def __init__(self, Z, codes, B_vec, *, theta, sigma, lambda_vec, alpha=0.2, max_iter_cluster=20,
                 epsilon_cluster=1e-3, epsilon_harmony=1e-2, K=100, block_size=0.05, seed=0, version="1.2",
                 batch_prop_cutoff=1e-5):
        self.version = _check_version(version)
        self.v2 = self.version != "1.2"
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        self.N, self.d = Z.shape
        if self.N < 6:
            raise ValueError("Refusing to run with less than 6 cells")
        self.Z_orig = Z
        self.Z_corr = np.zeros_like(Z)
        self.Z_cos = _l2_rows(Z)
        self.codes = np.ascontiguousarray(codes, dtype=np.int64)  # N x V, global batch columns
        self.B = int(np.sum(B_vec))
        self.codes1 = np.ascontiguousarray(np.hstack([np.zeros((self.N, 1), np.int64), self.codes + 1]))
        counts = np.zeros(self.B)
        for v in range(self.codes.shape[1]):
            counts += np.bincount(self.codes[:, v], minlength=self.B)
        self.N_b = counts
        self.Pr_b = counts / self.N
        self.K = int(K)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.lambda_estimation = lambda_vec is None
        self.lambda_vec = None if lambda_vec is None else np.asarray(lambda_vec, dtype=np.float64)
        self.alpha = np.float32(alpha)
        self.max_iter_kmeans = int(max_iter_cluster)
        self.epsilon_kmeans = np.float32(epsilon_cluster)
        self.epsilon_harmony = np.float32(epsilon_harmony)
        self.block_size = np.float32(0.2) if self.N < 40 else np.float32(block_size)
        self.batch_prop_cutoff = np.float32(batch_prop_cutoff)
        self.B_vec = [int(b) for b in B_vec]
        if not self.v2 and math.ceil(1.0 / float(self.block_size)) < 2:
            raise ValueError("block_size must be < 1: with a single block harmony's E and O become 0 "
                             "(R harmony 1.2 returns NaN in this case)")
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.objective_kmeans: list[np.float32] = []
        self.objective_harmony: list[np.float32] = []
        self.kmeans_rounds: list[int] = []
        self.R = None
        self.Y = None

    # -- helpers ---------------------------------------------------------------
    def _OE(self):
        _, _, OT = _objective_terms(self.R, self.dist, self.sigma, self.codes, self.B, _OBJ_CHUNKS)
        E = np.outer(self.R.sum(0), self.Pr_b)
        return np.ascontiguousarray(OT.T), E

    def compute_objective(self):
        f32 = np.float32
        norm_const = f32(2000) / f32(self.N)
        if self.v2:
            # harmony 2.x: pseudocount cross-entropy log((O + E + 1) / (2E + 1)); serial float32 sums (my_accu)
            Lmat = self.theta[None, :] * np.log((self.O + self.E + 1.0) / (2.0 * self.E + 1.0))
            km, ent, cross = _objective_v2(self.R, self.dist, self.sigma, np.ascontiguousarray(Lmat), self.codes)
            tot = f32(f32(km) + f32(ent)) + f32(cross)
            self.objective_kmeans.append(f32(tot * norm_const))
            return
        km, ent, OT_true = _objective_terms(self.R, self.dist, self.sigma, self.codes, self.B, _OBJ_CHUNKS)
        # cross entropy: sum((R .* sigma) .* ((theta' .* log((O+E)/E)) * Phi)) == sum(sigma * L * (R Phi^T))
        Lmat = self.theta[None, :] * np.log((self.O + self.E) / self.E)
        cross = float(np.sum(self.sigma[:, None] * Lmat * OT_true.T))
        tot = f32(f32(km) + f32(ent)) + f32(cross)
        self.objective_kmeans.append(f32(tot * norm_const))

    def check_convergence(self, kind: int) -> bool:
        f32 = np.float32
        if kind == 0:
            ok = self.objective_kmeans
            obj_old, obj_new = f32(0), f32(0)
            for i in range(self.window_size):
                obj_old = f32(obj_old + ok[-2 - i])
                obj_new = f32(obj_new + ok[-1 - i])
            diff = f32(obj_old - obj_new)
            if self.v2:  # harmony 2.x compares the absolute change
                diff = f32(abs(diff))
            return bool(diff / f32(abs(obj_old)) < self.epsilon_kmeans)
        oh = self.objective_harmony
        obj_old, obj_new = oh[-2], oh[-1]
        return bool(f32(obj_old - obj_new) / f32(abs(obj_old)) < self.epsilon_harmony)

    # -- algorithm steps ---------------------------------------------------------
    def _kmeans_v2(self, X):
        """harmony 2.x ``kmeans_centers``: its k-means++-style seeding, then 10 single Lloyd iterations
        (``arma::kmeans`` with ``keep_existing``). Seed i is redrawn with probability proportional to the cosine
        distance from the i-th uniformly drawn starting point (as in the C++, not from the nearest center)."""
        N, K = X.shape[0], self.K
        idx = np.floor(self.rng.random(K) * (N - 1)).astype(np.int64)
        Y = X[idx].copy()
        chosen: set[int] = set()
        for i in range(K):
            dist = np.abs(2.0 * (1.0 - X @ Y[i]))
            u = self.rng.random(N)
            with np.errstate(divide="ignore"):
                prob = -np.log(u) / dist
            j = int(np.argmin(prob))
            while j in chosen:
                prob[j] = prob.max()
                j = int(np.argmin(prob))
            chosen.add(j)
            Y[i] = X[j]
        x2 = (X * X).sum(1)
        for _ in range(10):
            d2 = x2[:, None] - 2.0 * matmul(X, Y.T) + (Y * Y).sum(1)[None, :]
            lab = np.argmin(d2, axis=1)
            cnt = np.bincount(lab, minlength=K)
            S = np.zeros_like(Y)
            np.add.at(S, lab, X)
            ok = cnt > 0
            Y[ok] = S[ok] / cnt[ok, None]  # an empty cluster keeps its center
        return Y

    def init_cluster(self, init_centroids=None, n_init: int = 10, kmeans_iter: int = 25):
        if init_centroids is not None:
            Y = np.asarray(init_centroids, dtype=np.float64)
            if Y.shape != (self.K, self.d):
                raise ValueError(f"init_centroids must be K x d = {self.K} x {self.d}")
        elif self.v2:
            with single_thread():
                Y = self._kmeans_v2(self.Z_cos)
        else:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.K, n_init=n_init, max_iter=kmeans_iter, random_state=self.seed,
                        algorithm="lloyd")
            with single_thread():  # k-means' OpenMP/BLAS sums depend on the thread count otherwise
                km.fit(self.Z_cos)
            Y = km.cluster_centers_
        self.Y = _l2_rows(Y)
        self.dist = 2.0 * (1.0 - matmul(self.Z_cos, self.Y.T))
        self.R = np.empty((self.N, self.K))
        _scale_dist(self.dist, self.sigma, self.R)
        self.O, self.E = self._OE()
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def _blocks(self):
        n_blocks = int(math.ceil(1.0 / float(self.block_size)))
        cpb = int(np.float32(self.N) * self.block_size)
        bounds = np.minimum(np.arange(n_blocks + 1, dtype=np.int64) * cpb, self.N)
        bounds[-1] = self.N
        return bounds

    def cluster(self):
        if self.v2:
            return self._cluster_v2()
        bounds = self._blocks()
        sd = np.empty((self.N, self.K))
        it = 0
        for it in range(self.max_iter_kmeans):
            Y = matmul(self.R.T, self.Z_cos)  # K x d (fixed-block sum over cells: thread-invariant)
            self.Y = _l2_rows(Y)
            self.dist = 2.0 * (1.0 - matmul(self.Z_cos, self.Y.T))
            _scale_dist(self.dist, self.sigma, sd)
            order = self.rng.permutation(self.N).astype(np.int64)
            ET, OT = np.ascontiguousarray(self.E.T), np.ascontiguousarray(self.O.T)
            _update_R(self.R, sd, order, bounds, ET, OT, self.Pr_b, self.theta, self.codes, _R_CHUNKS)
            self.E, self.O = np.ascontiguousarray(ET.T), np.ascontiguousarray(OT.T)
            self.compute_objective()
            if it > self.window_size and self.check_convergence(0):
                it += 1
                break
        else:
            it = self.max_iter_kmeans
        self.kmeans_rounds.append(it)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def _cluster_v2(self):
        """harmony 2.x ``cluster_cpp``: centroids stay fixed during clustering (they are updated in the
        correction step); after a correction, R, E and O are first recomputed from the corrected embedding."""
        if len(self.objective_harmony) != 1:
            self.dist = 2.0 * (1.0 - matmul(self.Z_cos, self.Y.T))
            _scale_dist(self.dist, self.sigma, self.R)
            self.O, self.E = self._OE()
        bounds = self._blocks()
        sd = np.empty((self.N, self.K))
        _scale_dist(self.dist, self.sigma, sd)  # dist is fixed within the round
        it = 0
        for it in range(self.max_iter_kmeans):
            order = self.rng.permutation(self.N).astype(np.int64)
            ET, OT = np.ascontiguousarray(self.E.T), np.ascontiguousarray(self.O.T)
            _update_R(self.R, sd, order, bounds, ET, OT, self.Pr_b, self.theta, self.codes, _R_CHUNKS,
                      v2=True, multiply=self.version == "2.1")
            self.E, self.O = np.ascontiguousarray(ET.T), np.ascontiguousarray(OT.T)
            self.compute_objective()
            if it > self.window_size and self.check_convergence(0):
                it += 1
                break
        else:
            it = self.max_iter_kmeans
        self.kmeans_rounds.append(it)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def moe_correct_ridge(self):
        if self.v2:
            return self._moe_correct_ridge_v2()
        nb1 = self.B + 1
        M = _ridge_gram(self.R, self.codes1, nb1)
        if self.lambda_estimation:
            lam = np.zeros((self.K, nb1))
            lam[:, 1:] = self.E * float(self.alpha)
        else:
            lam = np.broadcast_to(self.lambda_vec, (self.K, nb1))
        idx = np.arange(nb1)
        M[:, idx, idx] += lam
        RZ = _rz(self.R, self.Z_orig, self.codes1, nb1)
        with single_thread("blas"):
            inv = np.linalg.inv(M)
            W = inv @ RZ  # K x (B+1) x d
        W[:, 0, :] = 0.0
        _apply_correction(self.Z_orig, self.R, W, self.codes1, self.Z_corr)
        self.Z_cos = _l2_rows(self.Z_corr)

    def _moe_correct_ridge_v2(self):
        """harmony 2.x ``moe_correct_ridge_cpp``. Per cluster k, a batch level enters the regression only if
        its share of cluster k (``O[k, b] / N_b``) exceeds ``batch_prop_cutoff`` and its variable keeps at least
        two such levels; cells in no kept level leave cluster k's regression (intercept included); a cluster
        with no variable left is not corrected. The intercept (the cluster mean in the input space) becomes the
        new centroid of cluster k."""
        nb1 = self.B + 1
        K = self.K
        share = self.O / self.N_b[None, :]  # K x B
        above = share.astype(np.float32) > self.batch_prop_cutoff  # float comparison, as in the C++
        nvar = len(self.B_vec)
        var_of = np.repeat(np.arange(nvar), self.B_vec)
        nlev = np.zeros((K, nvar), dtype=np.int64)
        for v in range(nvar):
            nlev[:, v] = above[:, var_of == v].sum(1)
        keep = above & (nlev[:, var_of] > 1)  # K x B
        active = (nlev > 1).any(1)
        M = _ridge_gram(self.R, self.codes1, nb1)
        RZ = _rz(self.R, self.Z_orig, self.codes1, nb1)
        partial = active & ~keep.all(1)
        if partial.any():
            rs, rz = _unkept_sums(self.R, self.Z_orig, self.codes, np.ascontiguousarray(keep))
            M[partial, 0, 0] -= rs[partial]
            RZ[partial, 0, :] -= rz[partial]
        W = np.zeros((K, nb1, self.Z_orig.shape[1]))
        Y = self.Y.copy()
        alpha = float(self.alpha)
        with single_thread("blas"):
            for k in np.flatnonzero(active):
                kb = np.flatnonzero(keep[k])
                sel = np.concatenate([[0], kb + 1])
                Mk = M[k][np.ix_(sel, sel)]
                diag = np.arange(1, sel.size)
                Mk[diag, diag] += alpha * self.E[k, kb] if self.lambda_estimation else self.lambda_vec[kb + 1]
                Wk = np.linalg.inv(Mk) @ RZ[k, sel, :]
                Y[k] = Wk[0]
                W[k, kb + 1, :] = Wk[1:]
        _apply_correction(self.Z_orig, self.R, W, self.codes1, self.Z_corr)
        self.Y = _l2_rows(Y)
        self.Z_cos = _l2_rows(self.Z_corr)

    def harmonize(self, max_iter: int = 10, verbose: bool = False) -> bool:
        for it in range(1, max_iter + 1):
            self.cluster()
            self.moe_correct_ridge()
            if verbose:
                print(f"Harmony {it}/{max_iter}: objective {self.objective_harmony[-1]:.4f}")
            if self.check_convergence(1):
                if verbose:
                    print(f"Harmony converged after {it} iterations")
                return True
        return False


_VERSIONS = {"1.2": "1.2", "1.2.0": "1.2", "1.2.4": "1.2", "2.0": "2.0", "2.0.5": "2.0", "2.1": "2.1", "2.1.0": "2.1"}


def _check_version(version) -> str:
    v = _VERSIONS.get(str(version))
    if v is None:
        raise ValueError(f"version must be one of '1.2', '2.0' or '2.1' (R harmony 1.2.x, 2.0.5 on CRAN, 2.1.0 on "
                         f"GitHub), got {version!r}")
    return v

# --------------------------------------------------------------------------- public API
def _r_factor_levels(values) -> tuple[np.ndarray, np.ndarray]:
    """Codes/levels like R ``as.factor``: existing categorical levels kept, else sorted uniques."""
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat = pd.Categorical(s.cat.remove_unused_categories())  # Series -> Categorical (has .codes)
    else:
        cat = pd.Categorical(s)
    if (cat.codes < 0).any():
        raise ValueError("batch variable contains missing values; fill them or drop those cells before run_harmony")
    return np.asarray(cat.codes, dtype=np.int64), np.asarray(cat.categories)


@records_seed("seed")
def run_harmony(Z, meta, vars_use, *, theta=None, sigma=0.1, lambda_="default", nclust=None, max_iter=10,
                early_stop=True, alpha=0.2, tau=0.0, block_size=0.05, max_iter_cluster=None,
                epsilon_cluster=1e-3, epsilon_harmony=1e-2, batch_prop_cutoff=1e-5, version="2.0", seed=0,
                init_centroids=None, kmeans_n_init=10, n_threads=None, return_object=False, verbose=False):
    """Run Harmony batch correction on an embedding (cells x d), matching ``harmony::RunHarmony``.

    ``version`` picks the R release to reproduce: ``"2.0"`` (default; R harmony 2.0.5, the current CRAN release),
    ``"2.1"`` (2.0 with the multi-variable fix of GitHub tag v2.1.0, not yet on CRAN) or ``"1.2"`` (harmony
    1.2.x, used by Seurat-era pipelines and by :mod:`arsc.symphony`). The versions give different embeddings:

    * 2.x starts from k-means++-style seeds plus 10 Lloyd steps; 1.2 from k-means with 10 restarts.
    * 2.x keeps the centroids fixed while clustering and sets them from the regression intercept in the
      correction step; 1.2 recomputes them from R at every clustering iteration.
    * 2.x diversity penalty ``((2E + 1) / (O + E + 1)) ** theta`` (pseudocounts) vs ``(E / (O + E)) ** theta``.
    * 2.x leaves batch levels that are (almost) absent from a cluster out of that cluster's regression
      (``batch_prop_cutoff``); 1.2 regresses every level in every cluster.
    * Defaults: 2.x ``lambda = NULL`` (automatic, ``alpha * E``) and 4 clustering iterations per round; 1.2
      ``lambda = 1`` and 20.
    * With several batch variables, 1.2 and 2.0 add the per-variable penalties; 2.1 multiplies them, so a
      variable with ``theta = 0`` no longer changes the clustering.

    Each version is validated against the matching R release with one, two and three batch variables and
    distinct theta / lambda per variable (README, harmony rows): started from R's k-means centroids, per-dim r
    >= 0.98 on every dimension (median >= 0.999) and the same objective path; from arsc's own start the residual
    is the spread between two R seeds. Not the same algorithm port as ``harmonypy``.

    Parameters
    ----------
    Z : (n_cells, d) array, typically PCA cell embeddings (not rescaled).
    meta : DataFrame with the batch columns, or a 1-D array of batch labels.
    vars_use : column name or list of names (ignored if ``meta`` is an array).
    theta : diversity penalty per variable (default 2 each), in the order of ``vars_use``.
    sigma : soft k-means width (scalar or length ``nclust``).
    lambda_ : ridge penalty, scalar or one per variable (order of ``vars_use``); ``None`` = automatic
        (``alpha * E``), as ``lambda = NULL`` in R; ``"default"`` = the version's R default (1.2: 1; 2.x: None).
    nclust : number of clusters (default ``min(round(N / 30), 100)``).
    max_iter : harmony rounds (R ``max_iter``).
    early_stop : stop when the objective converges (default True; False runs all ``max_iter`` rounds).
    alpha : automatic-lambda factor (default 0.2).
    tau : protection against over-correcting small batches (default 0 = off).
    block_size : fraction of cells per online-update block (default 0.05; 2.x also accepts 1).
    max_iter_cluster : clustering iterations per round (default: 20 for 1.2, 4 for 2.x, as R's
        ``harmony_options``).
    epsilon_cluster, epsilon_harmony : convergence tolerances (defaults 1e-3, 1e-2).
    batch_prop_cutoff : 2.x only: a batch level enters cluster k's regression when its share of the cluster
        (``O[k, b] / N_b``) exceeds this (default 1e-5, R ``harmony_options(batch.prop.cutoff)``).
    version : ``"2.0"`` (default), ``"2.1"`` or ``"1.2"`` (also accepted: ``"2.0.5"``, ``"2.1.0"``, ``"1.2.0"``).
    seed : integer seed (default 0) for the k-means start (1.2: sklearn ``random_state``; 2.x: the seeding draws)
        and the per-iteration random block order (local ``np.random.default_rng``). Same input + seed ->
        bit-identical output, independent of ``n_threads`` (fixed reduction order).
    init_centroids : optional (nclust x d) starting centroids (bypasses k-means; use R's to match R exactly).
    kmeans_n_init : 1.2 only: k-means restarts (default 10).
    n_threads : numba threads (default: all).
    return_object : return :class:`HarmonyResult` (with traces) instead of the array.
    verbose : print the objective per round.

    Returns
    -------
    (n_cells, d) corrected embedding, or :class:`HarmonyResult`.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> H = arsc.run_harmony(pcs, ds.obs, "patient")
    >>> H.shape
    (3000, 10)
    """
    seed = check_seed(seed)
    version = _check_version(version)
    if isinstance(lambda_, str):
        if lambda_ != "default":
            raise ValueError("lambda_ must be 'default', None (automatic) or a positive number per variable")
        lambda_ = 1.0 if version == "1.2" else None  # R harmony 1.2: lambda = 1; 2.x: lambda = NULL
    if max_iter_cluster is None:
        max_iter_cluster = 20 if version == "1.2" else 4  # harmony_options(max.iter.cluster)
    if n_threads is not None:
        nb.set_num_threads(int(n_threads))
    Z = np.asarray(Z, dtype=np.float64)
    N = Z.shape[0]
    if Z.ndim != 2:
        raise ValueError(f"run_harmony: Z must be a cells x dims array (e.g. PCA embeddings), got shape {Z.shape}")
    if isinstance(meta, pd.DataFrame):
        vars_use = [vars_use] if isinstance(vars_use, str) else list(vars_use)
        miss = [v for v in vars_use if v not in meta.columns]
        if miss:
            raise KeyError(f"run_harmony: batch column(s) {miss} not in meta (has {list(meta.columns)[:20]}); "
                           "pass vars_use= with existing column names")
        cols = [meta[v] for v in vars_use]
    else:
        vars_use = ["batch_variable"]
        cols = [pd.Series(np.asarray(meta))]
    if len(cols[0]) != N:
        raise ValueError(f"run_harmony: {len(cols[0])} batch labels for {N} cells; meta must be row-aligned with Z "
                         "(same cells, same order)")
    if nclust is None:
        nclust = int(min(round(N / 30), 100))  # R round(): half to even, same as Python
    if theta is None:
        theta = [2.0] * len(vars_use)
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta.size != len(vars_use):
        raise ValueError("Please specify theta for each variable")
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 1:
        sigma = np.full(nclust, float(sigma))
    codes, B_vec, levels, off = [], [], {}, 0
    for v, col in zip(vars_use, cols):
        c, lev = _r_factor_levels(col)
        codes.append(c + off)
        B_vec.append(len(lev))
        levels[v] = lev
        off += len(lev)
    codes = np.stack(codes, axis=1)
    B = int(sum(B_vec))
    N_b = np.zeros(B)
    for v in range(codes.shape[1]):
        N_b += np.bincount(codes[:, v], minlength=B)
    if lambda_ is None:
        lambda_vec = None
    else:
        lam = np.atleast_1d(np.asarray(lambda_, dtype=np.float64))
        if not np.all(lam > 0):
            raise ValueError("Provided lambdas must be positive")
        if lam.size == 1:
            lambda_vec = np.concatenate([[0.0], np.full(B, lam[0])])
        elif lam.size == len(vars_use):
            lambda_vec = np.concatenate([[0.0], np.repeat(lam, B_vec)])
        else:
            raise ValueError("lambda must be a scalar or one value per variable")
    theta_vec = np.repeat(theta, B_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_vec = theta_vec * (1 - np.exp(-(N_b / (nclust * tau)) ** 2)) if tau > 0 else theta_vec
    h = Harmony(Z, codes, B_vec, theta=theta_vec, sigma=sigma, lambda_vec=lambda_vec, alpha=alpha,
                max_iter_cluster=max_iter_cluster, epsilon_cluster=epsilon_cluster,
                epsilon_harmony=epsilon_harmony if early_stop else -np.inf, K=nclust,
                block_size=block_size, seed=seed, version=version, batch_prop_cutoff=batch_prop_cutoff)
    h.init_cluster(init_centroids=init_centroids, n_init=kmeans_n_init)
    converged = h.harmonize(max_iter, verbose=verbose) if max_iter >= 1 else False
    if max_iter < 1 and h.v2:
        h.Z_corr = h.Z_cos.copy()  # harmony 2.x returns its L2-normalized input when no round runs
    if not return_object:
        return h.Z_corr
    return HarmonyResult(h.Z_corr, h.R, h.Y, [float(x) for x in h.objective_harmony],
                         [float(x) for x in h.objective_kmeans], list(h.kmeans_rounds), converged, levels)
