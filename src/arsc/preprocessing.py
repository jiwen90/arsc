"""Normalization, variable features, scaling and PCA (Seurat v4 semantics).

All functions take cells x features matrices (CSR) and return new objects;
nothing is modified in place.

=====================  ==========================================================
arsc                   Seurat equivalent
=====================  ==========================================================
``log_normalize``      ``NormalizeData(normalization.method = "LogNormalize")``
``clr``                ``NormalizeData(normalization.method = "CLR", margin = 2)``
``hvg_vst``            ``FindVariableFeatures(selection.method = "vst")``
``scale``              ``ScaleData`` (center, scale, clip at ``scale.max``)
``pca``                ``RunPCA`` (cell.embeddings = U * d, stdev = d / sqrt(n - 1))
=====================  ==========================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._loess import loess_fit
from ._utils import as_csr, check_seed

__all__ = ["log_normalize", "clr", "hvg_vst", "scale", "pca", "PCAResult"]


# --------------------------------------------------------------------------- normalization
@nb.njit(parallel=True, cache=True)
def _lognorm_kernel(indptr, data, out, scale_factor):
    n = indptr.size - 1
    for i in nb.prange(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p]
        if s > 0:
            # same operation order as Seurat's C++ LogNorm: log1p(x / colSum * sf);
            # x * (sf / s) differs by 1 ulp and changes tie structure downstream
            for p in range(indptr[i], indptr[i + 1]):
                out[p] = np.log1p(data[p] / s * scale_factor)


def log_normalize(X, scale_factor: float = 1e4, dtype=np.float64) -> sp.csr_matrix:
    """``log1p(x / total_counts(cell) * scale_factor)`` per cell.

    Matches Seurat's ``LogNormalize`` (C++ ``LogNorm``) to floating-point
    precision. Returns a new CSR with the same sparsity pattern.
    """
    X = as_csr(X)
    out = np.zeros(X.data.shape, dtype=dtype)
    _lognorm_kernel(X.indptr, X.data.astype(np.float64, copy=False), out, float(scale_factor))
    return sp.csr_matrix((out, X.indices.copy(), X.indptr.copy()), shape=X.shape)


def clr(X, margin: int = 2) -> sp.csr_matrix:
    """Centered log-ratio transform as in Seurat ``NormalizeData(method = "CLR")``.

    Seurat's function is ``log1p(x / exp(sum(log1p(x[x > 0])) / length(x)))``.
    ``margin = 2`` (recommended for ADT; Seurat's margin refers to its features x
    cells layout) normalizes each *cell* across features; ``margin = 1``
    normalizes each feature across cells.
    """
    X = as_csr(X, dtype=np.float64)
    n, p = X.shape
    if margin == 2:
        lp = X.copy()
        lp.data = np.log1p(lp.data)
        gm = np.exp(np.asarray(lp.sum(axis=1)).ravel() / p)  # per cell
        scale_vec = np.repeat(1.0 / gm, np.diff(X.indptr))
    elif margin == 1:
        C = X.tocsc()
        lp = np.log1p(C.data)
        sums = np.add.reduceat(lp, C.indptr[:-1]) if C.nnz else np.zeros(p)
        sums[np.diff(C.indptr) == 0] = 0.0
        gm = np.exp(sums / n)  # per feature
        scale_vec = 1.0 / gm[X.indices]
    else:
        raise ValueError("margin must be 1 or 2")
    out = X.copy()
    out.data = np.log1p(X.data * scale_vec)
    return out


# --------------------------------------------------------------------------- HVG (vst)
@nb.njit(parallel=True, cache=True)
def _col_mean_var(indptr, indices, data, n_rows, n_cols):
    """Column mean and (n-1) variance of a CSR matrix, summing in row order (as R/Matrix)."""
    s = np.zeros(n_cols)
    for i in range(n_rows):
        for p in range(indptr[i], indptr[i + 1]):
            s[indices[p]] += data[p]
    mu = s / n_rows
    ss = np.zeros(n_cols)
    nz = np.zeros(n_cols, dtype=np.int64)
    for i in range(n_rows):
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            d = data[p] - mu[j]
            ss[j] += d * d
            nz[j] += 1
    var = np.empty(n_cols)
    for j in nb.prange(n_cols):
        var[j] = (ss[j] + mu[j] * mu[j] * (n_rows - nz[j])) / (n_rows - 1)
    return mu, var, nz


@nb.njit(cache=True)
def _var_std_clipped(indptr, indices, data, n_rows, n_cols, mu, sd, vmax):
    """Seurat ``SparseRowVarStd``: variance of standardized values, nonzeros clipped at vmax."""
    acc = np.zeros(n_cols)
    nz = np.zeros(n_cols, dtype=np.int64)
    for i in range(n_rows):
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if sd[j] == 0:
                continue
            v = (data[p] - mu[j]) / sd[j]
            if v > vmax:
                v = vmax
            acc[j] += v * v
            nz[j] += 1
    out = np.zeros(n_cols)
    for j in range(n_cols):
        if sd[j] == 0:
            continue
        z = (0.0 - mu[j]) / sd[j]
        out[j] = (acc[j] + z * z * (n_rows - nz[j])) / (n_rows - 1)
    return out


def hvg_vst(counts, nfeatures: int = 2000, features=None, loess_span: float = 0.3,
            clip_max="auto", surface: str = "interpolate") -> pd.DataFrame:
    """Seurat v3 "vst" highly-variable features on raw counts.

    Fits ``log10(variance) ~ log10(mean)`` with R-compatible loess (span 0.3),
    standardizes each feature by the fitted s.d. (clipping at ``sqrt(n_cells)``)
    and ranks by the variance of the standardized values.

    Returns
    -------
    DataFrame indexed by feature with ``mean, variance, variance_expected,
    variance_standardized, rank`` (1 = most variable; NaN for zero-mean
    features) and boolean ``highly_variable`` (top ``nfeatures``).
    ``df.index[df.highly_variable]`` is *not* ranked; use
    ``df.sort_values("rank").index[:n]`` for Seurat's ``VariableFeatures`` order.
    """
    X = as_csr(counts)
    n, p = X.shape
    data = X.data.astype(np.float64, copy=False)
    mu, var, _ = _col_mean_var(X.indptr, X.indices, data, n, p)
    if clip_max == "auto":
        clip_max = np.sqrt(n)
    expected = np.zeros(p)
    ok = var > 0
    fit = loess_fit(np.log10(mu[ok]), np.log10(var[ok]), span=loess_span, degree=2, surface=surface)
    expected[ok] = 10.0 ** fit
    vstd = _var_std_clipped(X.indptr, X.indices, data, n, p, mu, np.sqrt(expected), float(clip_max))
    idx = pd.Index(features) if features is not None else pd.RangeIndex(p)
    df = pd.DataFrame({"mean": mu, "variance": var, "variance_expected": expected,
                       "variance_standardized": vstd}, index=idx)
    # Seurat: drop zero-mean features, stable order by decreasing standardized variance
    cand = np.flatnonzero(mu != 0)
    order = cand[np.argsort(-vstd[cand], kind="stable")]
    rank = np.full(p, np.nan)
    rank[order] = np.arange(1, order.size + 1)
    df["rank"] = rank
    df["highly_variable"] = rank <= nfeatures
    return df


# --------------------------------------------------------------------------- scale
def scale(X, center: bool = True, do_scale: bool = True, scale_max: float = 10.0,
          dtype=np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seurat ``ScaleData`` (no regression): dense z-scores, s.d. with n-1, values > ``scale_max`` clipped.

    Seurat clips only the upper tail (``FastSparseRowScale``). Zero-variance
    features become 0. Returns ``(Z, mean, sd)``; select features first
    (``X[:, hvg_idx]``) as Seurat scales only ``VariableFeatures`` by default.
    """
    if sp.issparse(X):
        X = as_csr(X)
        n, p = X.shape
        mu, var, _ = _col_mean_var(X.indptr, X.indices, X.data.astype(np.float64, copy=False), n, p)
        Z = X.toarray().astype(dtype, copy=False)
    else:
        Z = np.array(X, dtype=dtype)
        n, p = Z.shape
        mu = Z.mean(0)
        var = Z.var(0, ddof=1)
    sd = np.sqrt(var)
    if center:
        Z -= mu.astype(dtype)
    if do_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            Z /= np.where(sd > 0, sd, np.nan).astype(dtype)
    if scale_max is not None and np.isfinite(scale_max):
        np.minimum(Z, scale_max, out=Z)
    Z[~np.isfinite(Z)] = 0
    return Z, mu, sd


# --------------------------------------------------------------------------- PCA
@dataclass
class PCAResult:
    """``embedding`` (cells x k, = U*d), ``loadings`` (features x k), ``stdev`` (= d/sqrt(n-1))."""

    embedding: np.ndarray
    loadings: np.ndarray
    stdev: np.ndarray
    total_variance: float

    @property
    def variance_ratio(self) -> np.ndarray:
        return self.stdev ** 2 / self.total_variance


def pca(Z: np.ndarray, n_comps: int = 50, method: str = "auto", seed: int = 42,
        n_oversamples: int | None = None, n_iter: int = 15) -> PCAResult:
    """Truncated PCA of an already-centered (scaled) matrix, Seurat ``RunPCA`` conventions.

    Seurat computes ``irlba(t(scale.data), nv = npcs)`` without re-centering, so
    neither do we. ``method``:

    * ``"gram"`` -- exact: eigendecomposition of the p x p Gram matrix (fast when
      p, the number of HVGs, is a few thousand). Default for p <= 6000.
    * ``"arpack"`` -- exact truncated SVD via ARPACK (``scipy.sparse.linalg.svds``),
      default for p > 6000.
    * ``"randomized"`` -- Halko randomized SVD (``n_iter`` power iterations,
      ``n_oversamples`` defaults to max(20, k)). Fast but approximate on the flat
      tail of the spectrum; leading PCs are exact.

    Component signs are fixed so the largest-|loading| gene of each PC is positive
    (Seurat's signs are arbitrary; compare with ``abs(corr)``).

    ``seed`` (default 42 = Seurat ``RunPCA(seed.use = 42)``): ARPACK start vector (``"arpack"``)
    or the Gaussian test matrix (``"randomized"``); ``"gram"`` is deterministic and ignores it.
    """
    seed = check_seed(seed)
    Z = np.asarray(Z)
    n, p = Z.shape
    k = min(n_comps, p - 1, n - 1)
    if method == "auto":
        method = "gram" if p <= 6000 else "arpack"
    total_var = float(((Z.astype(np.float64) ** 2).sum(0) / (n - 1)).sum()) if Z.size else 0.0
    if method == "gram":
        G = Z.T @ Z
        w, V = np.linalg.eigh(G)
        o = np.argsort(w)[::-1][:k]
        d = np.sqrt(np.maximum(w[o], 0))
        V = V[:, o]
    elif method == "arpack":
        from scipy.sparse.linalg import svds
        v0 = np.random.default_rng(seed).standard_normal(min(n, p))
        _, d, Vt = svds(Z, k=k, v0=v0)
        o = np.argsort(d)[::-1]
        d, V = d[o], Vt[o].T
    elif method == "randomized":
        rng = np.random.default_rng(seed)
        ov = max(20, k) if n_oversamples is None else n_oversamples
        Q = Z @ rng.standard_normal((p, k + ov))
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(Q)
            Q, _ = np.linalg.qr(Z.T @ Q)
            Q = Z @ Q
        Q, _ = np.linalg.qr(Q)
        B = Q.T @ Z
        _, d, Vt = np.linalg.svd(B, full_matrices=False)
        d, V = d[:k], Vt[:k].T
    else:
        raise ValueError(f"unknown method {method!r}")
    sign = np.sign(V[np.abs(V).argmax(0), np.arange(k)])
    sign[sign == 0] = 1
    V = V * sign
    emb = Z @ V
    return PCAResult(emb, V, d / np.sqrt(max(1, n - 1)), total_var)
