"""All-groups Wilcoxon rank-sum markers -- port of ``presto::wilcoxauc``.

One pass over each gene's nonzeros (zeros form a single tied block, so only
nonzero values are sorted), yielding for every group vs. rest: Mann-Whitney U,
AUC, tie-corrected normal-approximation p-value (with continuity correction),
BH-adjusted p (per group, across genes), mean expression, logFC (difference
of means, i.e. log-scale for log-normalized input) and percent expressing.

Output columns and semantics are identical to presto:
``feature, group, avgExpr, logFC, statistic, auc, pval, padj, pct_in, pct_out``.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtr

from ._utils import as_csr, factorize_groups, p_adjust_bh

__all__ = ["wilcoxauc", "top_markers"]


@nb.njit(parallel=True, cache=True)
def _rank_stats(indptr, indices, data, groups, n_groups, n_rows):
    """Per gene (CSC column): group rank sums of nonzeros, nnz, value sums, tie term."""
    n_genes = indptr.size - 1
    rs = np.zeros((n_genes, n_groups))
    nnz = np.zeros((n_genes, n_groups))
    sums = np.zeros((n_genes, n_groups))
    ties = np.zeros(n_genes)
    for j in nb.prange(n_genes):
        p0, p1 = indptr[j], indptr[j + 1]
        m = p1 - p0
        nz = n_rows - m
        tie = float(nz) ** 3 - float(nz)
        if m > 0:
            vals = data[p0:p1]
            o = np.argsort(vals, kind="mergesort")
            a = 0
            while a < m:
                b = a + 1
                v = vals[o[a]]
                while b < m and vals[o[b]] == v:
                    b += 1
                t = b - a
                rank = nz + (a + 1 + b) / 2.0  # mean of 1-based positions a+1..b, offset by zeros
                if t > 1:
                    tie += float(t) ** 3 - float(t)
                for q in range(a, b):
                    r = indices[p0 + o[q]]
                    g = groups[r]
                    rs[j, g] += rank
                    nnz[j, g] += 1.0
                    sums[j, g] += vals[o[q]]
                a = b
        ties[j] = tie
    return rs, nnz, sums, ties


def wilcoxauc(X, groups, *, features=None, groups_use=None, obs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wilcoxon rank-sum test of every group vs. all other cells, for every feature.

    Parameters
    ----------
    X : cells x features matrix (sparse CSR/CSC or dense), e.g. log-normalized data.
        Values are assumed non-negative (zeros are ranked as one tied block, as presto).
    groups : labels per cell, or a column name of ``obs``. NA cells are dropped.
    features : feature names (default ``0..p-1``).
    groups_use : restrict to cells in these groups (presto ``groups_use``).

    Returns
    -------
    Tidy DataFrame (n_groups * n_features rows, group-major like presto):
    ``feature, group, avgExpr, logFC, statistic, auc, pval, padj, pct_in, pct_out``.
    """
    if isinstance(groups, str):
        groups = obs[groups]
    labels = groups if isinstance(groups, pd.Series) else pd.Series(np.asarray(groups))
    labels = labels.reset_index(drop=True)
    if sp.issparse(X):
        X = as_csr(X)
    else:
        X = sp.csr_matrix(np.asarray(X))
    if groups_use is not None:
        keep = labels.isin(list(groups_use)).to_numpy()
        X, labels = X[keep], labels[keep].reset_index(drop=True)
    codes, glev = factorize_groups(None, labels, X.shape[0])
    keep = codes >= 0
    if not keep.all():
        X, codes = X[keep], codes[keep]
    if X.nnz and X.data.min() < 0:
        raise ValueError("wilcoxauc assumes non-negative values (zeros ranked as one tied block)")
    X = X.copy()
    X.eliminate_zeros()
    N, P = X.shape
    G = len(glev)
    if G < 2:
        raise ValueError("Must have at least 2 groups defined.")
    C = X.tocsc()
    C.sort_indices()
    rs, nnz, sums, ties = _rank_stats(C.indptr.astype(np.int64), C.indices.astype(np.int64),
                                      C.data.astype(np.float64), codes.astype(np.int64), G, N)
    n_g = np.bincount(codes, minlength=G).astype(np.float64)  # (G,)
    nz = (N - np.diff(C.indptr)).astype(np.float64)  # zeros per gene (P,)
    zero_rank = (nz + 1.0) / 2.0
    # U statistic (P x G)
    ustat = rs + (n_g[None, :] - nnz) * zero_rank[:, None] - n_g * (n_g + 1) / 2.0
    n1n2 = n_g * (N - n_g)
    auc = ustat / n1n2
    z = ustat - 0.5 * n1n2
    z = z - np.sign(z) * 0.5
    x1 = float(N) ** 3 - N
    x2 = 1.0 / (12.0 * (float(N) ** 2 - N))
    rhs = (x1 - ties) * x2  # (P,)
    usigma = np.sqrt(np.outer(rhs, n1n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        pval = 2.0 * ndtr(-np.abs(z / usigma))
    padj = p_adjust_bh(pval, axis=0)
    means = sums / n_g
    tot = sums.sum(1, keepdims=True)
    lfc = means - (tot - sums) / (N - n_g)
    pct_in = 100.0 * nnz / n_g
    pct_out = 100.0 * (nnz.sum(1, keepdims=True) - nnz) / (N - n_g)
    feats = np.asarray(pd.Index(features) if features is not None else pd.RangeIndex(P))
    gl = glev.index.to_numpy()

    def flat(a):  # (P x G) -> group-major vector
        return np.asarray(a).T.ravel()

    return pd.DataFrame({
        "feature": np.tile(feats, G),
        "group": np.repeat(gl, P),
        "avgExpr": flat(means), "logFC": flat(lfc), "statistic": flat(ustat), "auc": flat(auc),
        "pval": flat(pval), "padj": flat(padj), "pct_in": flat(pct_in), "pct_out": flat(pct_out),
    })


def top_markers(res: pd.DataFrame, n: int = 10, auc_min: float = 0.0, padj_max: float = 1.0,
                pct_in_min: float = 0.0, by: str = "auc") -> pd.DataFrame:
    """Top ``n`` features per group from :func:`wilcoxauc` output (filters, then sort by ``by``)."""
    r = res[(res.auc >= auc_min) & (res.padj <= padj_max) & (res.pct_in >= pct_in_min)]
    return (r.sort_values(["group", by], ascending=[True, False]).groupby("group", sort=False).head(n)
            .reset_index(drop=True))
