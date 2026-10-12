"""Harmony batch integration -- a faithful port of R ``harmony`` 1.2.0.

Mirrors ``RunHarmony.default`` and the C++ ``harmony`` class (setup ->
``init_cluster_cpp`` -> repeat {``cluster_cpp``, ``moe_correct_ridge_cpp``}
until ``check_convergence(1)``), including defaults and their quirks:

* ``theta = 2`` per variable, ``sigma = 0.1``, ``lambda = 1`` (``None`` =
  automatic lambda = ``alpha * E[k, b]``), ``nclust = min(round(N / 30), 100)``,
  ``max_iter = 10`` harmony rounds, ``max_iter_cluster = 20``,
  ``epsilon_cluster = 1e-3``, ``epsilon_harmony = 1e-2``, ``block_size = 0.05``,
  ``tau = 0``.
* init: k-means (10 starts, 25 iterations) on the L2-normalized embedding.
* clustering: online updates of R in ``ceil(1 / block_size)`` random blocks;
  diversity penalty ``(E / (O + E)) ** theta`` summed over variables
  (harmony 1.x multiplies by ``Phi``, i.e. sums across covariates).
* objective = (k-means error + sigma * entropy + sigma * cross-entropy) * 2000/N,
  stored as float32 exactly like the C++ (convergence decisions match).
* convergence: clustering stops when the 3-iteration moving window improves by
  < ``epsilon_cluster`` (checked from iteration 5); harmony stops when the
  objective improves by < ``epsilon_harmony``.
* correction: per-cluster ridge regression on ``[1, Phi]`` with the intercept
  left in (``W[0] = 0``) using the *original* embedding.

Differences that cannot be matched bit-for-bit: R's k-means / shuffling use
R's RNG; pass ``init_centroids`` (e.g. from R) to start from identical
centroids. The deterministic correction step reproduces R to ~1e-12 given the
same R matrix (see VALIDATION.md). Note ``block_size = 1`` is degenerate in
harmony 1.2 (all cells are removed at once, so E = O = 0 and E / (O + E) is
0/0; R returns NaN); it is rejected here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import pandas as pd

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
def _block_recompute(R, sd, order, s, e, PT, codes):
    K = R.shape[1]
    V = codes.shape[1]
    for t in nb.prange(s, e):
        i = order[t]
        tot = 0.0
        for k in range(K):
            acc = 0.0
            for v in range(V):
                acc += PT[codes[i, v], k]
            val = sd[i, k] * acc
            R[i, k] = val
            tot += val
        for k in range(K):
            R[i, k] /= tot


@nb.njit(cache=True, error_model="numpy")
def _penalty(ET, OT, theta, PT):
    B, K = ET.shape
    for b in range(B):
        for k in range(K):
            PT[b, k] = (ET[b, k] / (OT[b, k] + ET[b, k])) ** theta[b]


def _update_R(R, sd, order, bounds, ET, OT, Pr_b, theta, codes, nchunk):
    """Online R update over blocks (harmony::update_R). Python driver: numba
    parallel regions launched from a non-parallel njit caller are much slower."""
    PT = np.empty_like(ET)
    for blk in range(bounds.size - 1):
        s, e = int(bounds[blk]), int(bounds[blk + 1])
        _block_remove_add(R, order, s, e, ET, OT, Pr_b, codes, -1.0, nchunk)
        _penalty(ET, OT, theta, PT)
        _block_recompute(R, sd, order, s, e, PT, codes)
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
    """``Z_corr`` (cells x d) plus the final state and convergence traces."""

    Z_corr: np.ndarray
    R: np.ndarray
    Y: np.ndarray
    objective_harmony: list = field(default_factory=list)
    objective_kmeans: list = field(default_factory=list)
    kmeans_rounds: list = field(default_factory=list)
    converged: bool = False
    batch_levels: dict = field(default_factory=dict)


class Harmony:
    """Stateful Harmony object (port of the C++ class). Most users want :func:`run_harmony`."""

    window_size = 3

    def __init__(self, Z, codes, B_vec, *, theta, sigma, lambda_vec, alpha=0.2, max_iter_cluster=20,
                 epsilon_cluster=1e-3, epsilon_harmony=1e-2, K=100, block_size=0.05, seed=0):
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
        if math.ceil(1.0 / float(self.block_size)) < 2:
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
        _, _, OT = _objective_terms(self.R, self.dist, self.sigma, self.codes, self.B, nb.get_num_threads())
        E = np.outer(self.R.sum(0), self.Pr_b)
        return np.ascontiguousarray(OT.T), E

    def compute_objective(self):
        km, ent, OT_true = _objective_terms(self.R, self.dist, self.sigma, self.codes, self.B, nb.get_num_threads())
        # cross entropy: sum((R .* sigma) .* ((theta' .* log((O+E)/E)) * Phi)) == sum(sigma * L * (R Phi^T))
        Lmat = self.theta[None, :] * np.log((self.O + self.E) / self.E)
        cross = float(np.sum(self.sigma[:, None] * Lmat * OT_true.T))
        f32 = np.float32
        norm_const = f32(2000) / f32(self.N)
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
            return bool(f32(obj_old - obj_new) / f32(abs(obj_old)) < self.epsilon_kmeans)
        oh = self.objective_harmony
        obj_old, obj_new = oh[-2], oh[-1]
        return bool(f32(obj_old - obj_new) / f32(abs(obj_old)) < self.epsilon_harmony)

    # -- algorithm steps ---------------------------------------------------------
    def init_cluster(self, init_centroids=None, n_init: int = 10, kmeans_iter: int = 25):
        if init_centroids is None:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.K, n_init=n_init, max_iter=kmeans_iter, random_state=self.seed,
                        algorithm="lloyd")
            km.fit(self.Z_cos)
            Y = km.cluster_centers_
        else:
            Y = np.asarray(init_centroids, dtype=np.float64)
            if Y.shape != (self.K, self.d):
                raise ValueError(f"init_centroids must be K x d = {self.K} x {self.d}")
        self.Y = _l2_rows(Y)
        self.dist = 2.0 * (1.0 - self.Z_cos @ self.Y.T)
        self.R = np.empty((self.N, self.K))
        _scale_dist(self.dist, self.sigma, self.R)
        self.O, self.E = self._OE()
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def cluster(self):
        n_blocks = int(math.ceil(1.0 / float(self.block_size)))
        cpb = int(np.float32(self.N) * self.block_size)
        bounds = np.minimum(np.arange(n_blocks + 1, dtype=np.int64) * cpb, self.N)
        bounds[-1] = self.N
        sd = np.empty((self.N, self.K))
        it = 0
        for it in range(self.max_iter_kmeans):
            Y = self.R.T @ self.Z_cos  # K x d
            self.Y = _l2_rows(Y)
            self.dist = 2.0 * (1.0 - self.Z_cos @ self.Y.T)
            _scale_dist(self.dist, self.sigma, sd)
            order = self.rng.permutation(self.N).astype(np.int64)
            ET, OT = np.ascontiguousarray(self.E.T), np.ascontiguousarray(self.O.T)
            _update_R(self.R, sd, order, bounds, ET, OT, self.Pr_b, self.theta, self.codes, min(8, nb.get_num_threads()))
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
        nb1 = self.B + 1
        M = _ridge_gram(self.R, self.codes1, nb1)
        if self.lambda_estimation:
            lam = np.zeros((self.K, nb1))
            lam[:, 1:] = self.E * float(self.alpha)
        else:
            lam = np.broadcast_to(self.lambda_vec, (self.K, nb1))
        idx = np.arange(nb1)
        M[:, idx, idx] += lam
        inv = np.linalg.inv(M)
        RZ = _rz(self.R, self.Z_orig, self.codes1, nb1)
        W = inv @ RZ  # K x (B+1) x d
        W[:, 0, :] = 0.0
        _apply_correction(self.Z_orig, self.R, W, self.codes1, self.Z_corr)
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


# --------------------------------------------------------------------------- public API
def _r_factor_levels(values) -> tuple[np.ndarray, np.ndarray]:
    """Codes/levels like R ``as.factor``: existing categorical levels kept, else sorted uniques."""
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat = pd.Categorical(s.cat.remove_unused_categories())  # Series -> Categorical (has .codes)
    else:
        cat = pd.Categorical(s)
    if (cat.codes < 0).any():
        raise ValueError("batch variable contains missing values")
    return np.asarray(cat.codes, dtype=np.int64), np.asarray(cat.categories)


def run_harmony(Z, meta, vars_use, *, theta=None, sigma=0.1, lambda_=1.0, nclust=None, max_iter=10,
                early_stop=True, alpha=0.2, tau=0.0, block_size=0.05, max_iter_cluster=20,
                epsilon_cluster=1e-3, epsilon_harmony=1e-2, seed=0, init_centroids=None,
                kmeans_n_init=10, n_threads=None, return_object=False, verbose=False):
    """Run Harmony on an embedding (cells x d), matching ``harmony::RunHarmony`` 1.2 defaults.

    Parameters
    ----------
    Z : (n_cells, d) array, typically PCA cell embeddings (not rescaled).
    meta : DataFrame with the batch columns, or a 1-D array of batch labels.
    vars_use : column name or list of names (ignored if ``meta`` is an array).
    theta : diversity penalty per variable (default 2 each).
    sigma : soft k-means width (scalar or length ``nclust``).
    lambda_ : ridge penalty, scalar or one per variable; ``None`` = automatic
        (``alpha * E``), as ``lambda = NULL`` in R.
    nclust : default ``min(round(N / 30), 100)``.
    max_iter : harmony rounds (R ``max_iter``); ``early_stop=False`` disables the
        objective-based stop.
    seed : seeds k-means init and block shuffling.
    init_centroids : optional (nclust x d) starting centroids (bypasses k-means).
    n_threads : numba threads (default: all).
    return_object : return :class:`HarmonyResult` (with traces) instead of the array.

    Returns
    -------
    (n_cells, d) corrected embedding, or HarmonyResult.
    """
    if n_threads is not None:
        nb.set_num_threads(int(n_threads))
    Z = np.asarray(Z, dtype=np.float64)
    N = Z.shape[0]
    if isinstance(meta, pd.DataFrame):
        vars_use = [vars_use] if isinstance(vars_use, str) else list(vars_use)
        cols = [meta[v] for v in vars_use]
    else:
        vars_use = ["batch_variable"]
        cols = [pd.Series(np.asarray(meta))]
    if len(cols[0]) != N:
        raise ValueError("number of labels does not match number of cells")
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
                block_size=block_size, seed=seed)
    h.init_cluster(init_centroids=init_centroids, n_init=kmeans_n_init)
    converged = h.harmonize(max_iter, verbose=verbose) if max_iter >= 1 else False
    if not return_object:
        return h.Z_corr
    return HarmonyResult(h.Z_corr, h.R, h.Y, [float(x) for x in h.objective_harmony],
                         [float(x) for x in h.objective_kmeans], list(h.kmeans_rounds), converged, levels)
