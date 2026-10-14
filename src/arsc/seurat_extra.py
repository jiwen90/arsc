"""Seurat functions used by the research scripts that are not covered by the other arsc modules.

Every function takes plain matrices / DataFrames (cells x features, arsc convention) and mirrors the
numerics of Seurat 4.3 / SeuratObject 5.0 (validated in ``tests/test_seurat_extra.py``; see SEURAT_PARITY.md).

=====================================  ==========================================================
Python                                 Seurat
=====================================  ==========================================================
:func:`find_markers`                   ``FindMarkers`` (wilcox [limma rank-sum], LR, t; ident.1/ident.2,
                                       min.pct, min.diff.pct, logfc.threshold, only.pos, max.cells.per.ident,
                                       latent.vars for LR, pseudocount, base, norm.method)
:func:`find_all_markers`               ``FindAllMarkers`` (one-vs-rest, return.thresh, only.pos); wilcox path
                                       is a single pass over all clusters
:func:`fold_change`                    ``FoldChange`` (avg_log2FC, pct.1, pct.2)
:func:`average_expression`             ``AverageExpression`` / ``AggregateExpression`` (group.by vectors)
:func:`cell_cycle_scoring`             ``CellCycleScoring`` (+ ``cc_genes`` = ``Seurat::cc.genes``)
:func:`scale_data`                     ``ScaleData`` incl. ``vars.to.regress`` (linear), ``split.by``
:func:`normalize_data`                 ``NormalizeData`` (LogNormalize / CLR margin 1|2 / RC)
:func:`add_metadata`, :func:`which_cells`, :func:`split_object`, :func:`rename_idents`
                                       ``AddMetaData`` / ``WhichCells`` / ``subset`` / ``SplitObject`` / ``RenameIdents``
=====================================  ==========================================================

Note on the Wilcoxon test: Seurat 4.3 ``FindMarkers`` uses ``limma::rankSumTestWithCorrelation`` (normal
approximation with tie correction and continuity correction), which gives the same p-values as
``presto::wilcoxauc`` (``arsc.wilcoxauc``) up to rounding; this module reproduces the limma formula.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.special import ndtr

from ._cc_genes import CC_GENES, CC_GENES_UPDATED_2019
from ._rrng import RRandom
from ._utils import as_csr, check_seed, factorize_groups
from .markers import _rank_stats
from . import preprocessing as pp

__all__ = ["find_markers", "find_all_markers", "fold_change", "average_expression", "cell_cycle_scoring",
           "scale_data", "normalize_data", "add_metadata", "which_cells", "split_object", "rename_idents",
           "cc_genes", "cc_genes_updated_2019"]

cc_genes = CC_GENES  # Seurat::cc.genes (what the research scripts use)
cc_genes_updated_2019 = CC_GENES_UPDATED_2019  # Seurat::cc.genes.updated.2019


# ============================================================================ helpers
def _names(features, p):
    return pd.Index(features) if features is not None else pd.RangeIndex(p)


def _mask(groups, ident):
    g = groups if isinstance(groups, pd.Series) else pd.Series(np.asarray(groups))
    idents = list(ident) if isinstance(ident, (list, tuple, set, np.ndarray, pd.Index)) else [ident]
    return g.isin(idents).to_numpy()


def _col_nnz_sum(X: sp.csr_matrix, fn=None):
    """Column sums of fn(data) and nnz of a CSR block."""
    d = X.data if fn is None else fn(X.data)
    out = np.bincount(X.indices, weights=d, minlength=X.shape[1])
    nnz = np.bincount(X.indices[X.data > 0], minlength=X.shape[1])
    return out, nnz


# ============================================================================ FoldChange
def fold_change(X, mask1, mask2, *, norm_method: str | None = "LogNormalize", slot: str = "data",
                pseudocount: float = 1.0, base: float = 2.0, features=None) -> pd.DataFrame:
    """Seurat ``FoldChange``: ``avg_log{base}FC`` (group 1 - group 2), ``pct.1``, ``pct.2``.

    ``slot='data'`` with ``norm_method='LogNormalize'`` averages ``expm1(x)`` before the log (Seurat's default
    for log-normalized data); any other ``norm_method`` (e.g. ``'CLR'``) or ``slot='counts'`` averages the raw
    values; ``slot='scale.data'`` returns the plain mean difference (``avg_diff``). ``pct`` is rounded to 3
    digits like Seurat.
    """
    X = as_csr(X)
    n1, n2 = int(np.sum(mask1)), int(np.sum(mask2))
    A, B = X[np.asarray(mask1)], X[np.asarray(mask2)]
    expm = (slot == "data" and norm_method == "LogNormalize")
    fn = np.expm1 if expm else None
    s1, c1 = _col_nnz_sum(A, fn)
    s2, c2 = _col_nnz_sum(B, fn)
    m1, m2 = s1 / n1, s2 / n2
    if slot == "scale.data":
        fc, name = m1 - m2, "avg_diff"
    else:
        lg = (lambda v: np.log(v + pseudocount) / np.log(base))
        fc = lg(m1) - lg(m2)
        name = "avg_log2FC" if base == 2 else ("avg_logFC" if base == np.e else f"avg_log{base}FC")
    return pd.DataFrame({name: fc, "pct.1": np.round(c1 / n1, 3), "pct.2": np.round(c2 / n2, 3)},
                        index=_names(features, X.shape[1]))


# ============================================================================ tests
def _limma_ranksum_p(ustat1, n1, n2, ties):
    """limma::rankSumTestWithCorrelation (corr = 0), two-sided as Seurat: min(2*min(p.less, p.greater), 1).

    ``ustat1`` = Mann-Whitney U of group 1 in presto's orientation (rank sum - n1(n1+1)/2);
    limma's U = n1*n2 + n1(n1+1)/2 - sum(r1) = n1*n2 - ustat1. ``ties`` = sum(t^3 - t) over tie blocks; the
    variance n1*n2*(n+1)/12 is multiplied by ``1 - ties/(n^3 - n)`` (limma >= 3.5x; identical to presto).
    """
    n = n1 + n2
    U = n1 * n2 - ustat1
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n + 1) / 12.0 * (1.0 - ties / (float(n) ** 3 - n)))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_less = ndtr((U + 0.5 - mu) / sd)
        p_greater = ndtr(-(U - 0.5 - mu) / sd)
    return np.minimum(2.0 * np.minimum(p_less, p_greater), 1.0)


def _wilcox_p(Xs, codes):
    """limma Wilcoxon p-values (group code 0 vs 1) for every column of ``Xs`` (cells x genes)."""
    N, P = Xs.shape
    n1 = float((codes == 0).sum())
    n2 = N - n1
    if sp.issparse(Xs) and (Xs.nnz == 0 or Xs.data.min() >= 0):
        C = Xs.tocsc(copy=True)
        C.eliminate_zeros()
        C.sort_indices()
        rs, nnz, _, ties = _rank_stats(C.indptr.astype(np.int64), C.indices.astype(np.int64),
                                       C.data.astype(np.float64), codes.astype(np.int64), 2, N)
        nz = (N - np.diff(C.indptr)).astype(np.float64)
        zero_rank = (nz + 1.0) / 2.0
        ustat = rs[:, 0] + (n1 - nnz[:, 0]) * zero_rank - n1 * (n1 + 1) / 2.0
        ties = np.where(np.diff(C.indptr) == 0, float(N) ** 3 - N, ties)  # all-tied column
    else:  # dense / signed data: rank directly
        D = Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs, dtype=np.float64)
        ustat = np.empty(P)
        ties = np.empty(P)
        for a in range(0, P, 512):
            r = stats.rankdata(D[:, a:a + 512], axis=0)
            ustat[a:a + 512] = r[codes == 0].sum(0) - n1 * (n1 + 1) / 2.0
        for j in range(P):
            t = np.unique(D[:, j], return_counts=True)[1].astype(np.float64)
            ties[j] = (t ** 3 - t).sum()
    return _limma_ranksum_p(ustat, n1, n2, ties)


def _t_p(Xs, mask1):
    """Welch two-sample t-test (``t.test(x, y)``) per column; Xs = cells x genes (group1 rows = mask1)."""
    Xs = as_csr(Xs)

    def mv(M):
        n = M.shape[0]
        s = np.asarray(M.sum(0)).ravel()
        m = s / n
        nnz = np.bincount(M.indices, minlength=M.shape[1])
        ss = np.bincount(M.indices, weights=(M.data - m[M.indices]) ** 2, minlength=M.shape[1])
        ss += (n - nnz) * m ** 2
        return m, ss / (n - 1), n

    m1, v1, n1 = mv(Xs[mask1])
    m2, v2, n2 = mv(Xs[~mask1])
    se2 = v1 / n1 + v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m2) / np.sqrt(se2)
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        p = 2.0 * stats.t.sf(np.abs(t), df)
    p[~np.isfinite(p)] = np.nan
    return p


@nb.njit(cache=True)
def _glm_binom_dev(Xm, y):
    """Deviance of a logistic GLM fitted like R's ``glm.fit`` (IRLS, eps 1e-8, 25 iterations).

    The weighted least-squares step uses re-orthogonalized Gram-Schmidt QR with R's rank rule
    (``dqrdc2``, tol 1e-11: a column whose residual norm falls below ``tol * norm`` is dropped, i.e.
    gets coefficient 0), so quasi-separated / collinear designs follow the same path as R.
    """
    n, p = Xm.shape
    eps = 2.220446049250313e-16
    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1.0 - mu))
    devold = 0.0
    for i in range(n):
        devold += -2.0 * (y[i] * np.log(mu[i]) + (1 - y[i]) * np.log(1 - mu[i]))
    dev = devold
    Q = np.empty((n, p))
    Rm = np.zeros((p, p))
    keep = np.zeros(p, dtype=np.bool_)
    v = np.empty(n)
    rhs = np.empty(n)
    beta = np.zeros(p)
    for it in range(25):
        for i in range(n):
            w = mu[i] * (1.0 - mu[i])
            rhs[i] = np.sqrt(w) * (eta[i] + (y[i] - mu[i]) / w)
        sw = np.sqrt(mu * (1.0 - mu))
        Rm[:, :] = 0.0
        for j in range(p):
            keep[j] = False
        nk = 0
        for j in range(p):
            nrm0 = 0.0
            for i in range(n):
                v[i] = Xm[i, j] * sw[i]
                nrm0 += v[i] * v[i]
            nrm0 = np.sqrt(nrm0)
            for rep in range(2):
                for k in range(j):
                    if keep[k]:
                        d = 0.0
                        for i in range(n):
                            d += Q[i, k] * v[i]
                        Rm[k, j] += d
                        for i in range(n):
                            v[i] -= d * Q[i, k]
            nrm = 0.0
            for i in range(n):
                nrm += v[i] * v[i]
            nrm = np.sqrt(nrm)
            if nrm0 > 0 and nrm > 1e-11 * nrm0:
                keep[j] = True
                Rm[j, j] = nrm
                for i in range(n):
                    Q[i, j] = v[i] / nrm
                nk += 1
        # back-substitution on kept columns
        qtr = np.zeros(p)
        for j in range(p):
            if keep[j]:
                d = 0.0
                for i in range(n):
                    d += Q[i, j] * rhs[i]
                qtr[j] = d
        beta[:] = 0.0
        for j in range(p - 1, -1, -1):
            if keep[j]:
                s_ = qtr[j]
                for k in range(j + 1, p):
                    if keep[k]:
                        s_ -= Rm[j, k] * beta[k]
                beta[j] = s_ / Rm[j, j]
        dev = 0.0
        for i in range(n):
            e = 0.0
            for a in range(p):
                e += Xm[i, a] * beta[a]
            eta[i] = e
            m = 1.0 / (1.0 + np.exp(-e))
            if m < eps:
                m = eps
            elif m > 1.0 - eps:
                m = 1.0 - eps
            mu[i] = m
            dev += -2.0 * (y[i] * np.log(m) + (1 - y[i]) * np.log(1 - m))
        if abs(dev - devold) / (abs(dev) + 0.1) < 1e-8:
            break
        devold = dev
    return dev


@nb.njit(parallel=True, cache=True)
def _lr_devs(indptr, indices, data, L, y):
    """Per gene (CSC column) deviance of ``y ~ gene + L``."""
    n, q = L.shape
    P = indptr.size - 1
    out = np.empty(P)
    for j in nb.prange(P):
        Xm = np.empty((n, q + 1))  # R's model.matrix order: intercept, GENE, latent...
        Xm[:, 1] = 0.0
        Xm[:, 0] = L[:, 0]
        for a in range(1, q):
            Xm[:, a + 1] = L[:, a]
        for k in range(indptr[j], indptr[j + 1]):
            Xm[indices[k], 1] = data[k]
        out[j] = _glm_binom_dev(Xm, y)
    return out


def _design(latent: pd.DataFrame | None, n: int) -> np.ndarray:
    """R ``model.matrix(~ latent)``: intercept + numeric columns + treatment-coded factors."""
    cols = [np.ones(n)]
    if latent is not None:
        for c in latent.columns:
            s = latent[c]
            if s.dtype == bool:
                cols.append(s.to_numpy(float))
            elif pd.api.types.is_numeric_dtype(s):
                cols.append(s.to_numpy(float))
            else:
                cat = pd.Categorical(s)
                for k in range(1, len(cat.categories)):
                    cols.append((cat.codes == k).astype(float))
    return np.column_stack(cols)


def _lr_p(Xs, mask1, latent):
    """``LRDETest``: LRT of ``group ~ gene + latent`` vs ``group ~ latent`` (binomial GLMs)."""
    C = as_csr(Xs).tocsc()
    C.sort_indices()
    n = C.shape[0]
    y = (~mask1).astype(np.float64)  # factor(Group1, Group2): modelled response is Group2
    L = _design(latent, n)
    dev0 = _glm_binom_dev(L, y)
    dev1 = _lr_devs(C.indptr.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.float64), L, y)
    return stats.chi2.sf(np.maximum(dev0 - dev1, 0.0), 1)


# ============================================================================ FindMarkers
def find_markers(X, groups=None, ident_1=None, ident_2=None, *, features=None, cells_1=None, cells_2=None,
                 test_features=None, test_use: str = "wilcox", logfc_threshold: float = 0.25,
                 min_pct: float = 0.1, min_diff_pct: float = -np.inf, only_pos: bool = False,
                 max_cells_per_ident: float = np.inf, random_seed: int = 1, latent_vars: pd.DataFrame | None = None,
                 min_cells_group: int = 3, pseudocount_use: float = 1.0, base: float = 2.0,
                 norm_method: str | None = "LogNormalize", slot: str = "data",
                 n_features_total: int | None = None) -> pd.DataFrame:
    """Seurat ``FindMarkers`` (``ident.1`` vs ``ident.2`` or vs all other cells).

    Parameters
    ----------
    X : cells x features matrix (Seurat ``data`` slot: log-normalized RNA, CLR ADT, ...).
    groups, ident_1, ident_2 : identity per cell and the two identities (``ident_1`` may be a list;
        ``ident_2=None`` = all other cells). Alternatively pass boolean masks ``cells_1`` / ``cells_2``.
    features : feature names for ``X`` columns. ``test_features`` restricts the tested set (Seurat ``features=``).
    test_use : ``"wilcox"`` (limma rank-sum, as Seurat), ``"LR"`` (needs ``latent_vars``), ``"t"``.
        (``MAST``, ``DESeq2``, ``negbinom``, ``poisson``, ``roc``, ``bimod`` are not ported; use ``arsc.pseudobulk``
        + an external DE tool for sample-level DESeq2.)
    norm_method : ``"LogNormalize"`` => fold change from ``log(mean(expm1(x)) + 1)``; anything else (``"CLR"``)
        => ``log(mean(x) + 1)`` (what Seurat does for ADT).
    n_features_total : number of features in the assay (Bonferroni multiplier; default ``X.shape[1]``).
    random_seed : R ``set.seed`` value (default 1 = Seurat ``random.seed = 1``) for the
        ``max_cells_per_ident`` downsampling, drawn with the R-compatible RNG (identical cells to R).
        Unused when ``max_cells_per_ident`` is infinite (the default).

    Returns
    -------
    DataFrame indexed by feature: ``p_val, avg_log2FC, pct.1, pct.2, p_val_adj``, ordered by
    ``(p_val, -avg_log2FC)`` like Seurat.
    """
    random_seed = check_seed(random_seed, "random_seed", r_seed=True)
    X = as_csr(X)
    N, P = X.shape
    names = _names(features, P)
    if cells_1 is None:
        g = groups
        m1 = _mask(g, ident_1)
        m2 = ~m1 if ident_2 is None else _mask(g, ident_2)
        if ident_2 is None:
            m2 = ~m1
    else:
        m1 = np.asarray(cells_1, dtype=bool)
        m2 = ~m1 if cells_2 is None else np.asarray(cells_2, dtype=bool)
    if m1.sum() < min_cells_group or m2.sum() < min_cells_group:
        raise ValueError(f"Cell group 1 ({m1.sum()}) or 2 ({m2.sum()}) has fewer than {min_cells_group} cells")
    if test_use not in ("wilcox", "LR", "t"):
        raise NotImplementedError(f"test_use={test_use!r} is not ported (wilcox, LR, t are)")
    if latent_vars is not None and test_use != "LR":
        latent_vars = None
    if test_use == "LR" and latent_vars is not None and len(latent_vars) != N:
        raise ValueError("latent_vars must have one row per cell")
    fc = fold_change(X, m1, m2, norm_method=norm_method, slot=slot, pseudocount=pseudocount_use, base=base,
                     features=names)
    fcname = fc.columns[0]
    pct1, pct2 = fc["pct.1"].to_numpy(), fc["pct.2"].to_numpy()
    amin = np.maximum(pct1, pct2)
    keep = amin >= min_pct
    keep &= (amin - np.minimum(pct1, pct2)) >= min_diff_pct
    if slot != "scale.data":
        tot = fc[fcname].to_numpy()
        keep &= (tot >= logfc_threshold) if only_pos else (np.abs(tot) >= logfc_threshold)
    if test_features is not None:
        keep &= names.isin(list(test_features))
    j = np.flatnonzero(keep)
    if j.size == 0:
        return pd.DataFrame(columns=["p_val", fcname, "pct.1", "pct.2", "p_val_adj"])
    i1, i2 = np.flatnonzero(m1), np.flatnonzero(m2)
    if max_cells_per_ident < np.inf:
        rr = RRandom(random_seed)
        k = int(max_cells_per_ident)
        if i1.size > k:
            i1 = i1[rr.sample_int(i1.size, k)]
        if i2.size > k:
            i2 = i2[rr.sample_int(i2.size, k)]
        if latent_vars is not None:
            pass  # latent rows are taken below by the same cell index
    rows = np.concatenate([i1, i2])
    codes = np.concatenate([np.zeros(i1.size, int), np.ones(i2.size, int)])
    Xs = X[rows][:, j]
    if test_use == "wilcox":
        p = _wilcox_p(Xs, codes)
    elif test_use == "t":
        p = _t_p(Xs, codes == 0)
    else:
        lat = latent_vars.iloc[rows].reset_index(drop=True) if latent_vars is not None else None
        p = _lr_p(Xs, codes == 0, lat)
    res = pd.DataFrame({"p_val": p}, index=names[j]).join(fc)
    if only_pos:
        res = res[res[fcname] > 0]
    order = np.lexsort((-res[fcname].to_numpy(), res["p_val"].to_numpy()))  # order(p_val, -fc), stable
    res = res.iloc[order]
    ntot = P if n_features_total is None else n_features_total
    res["p_val_adj"] = np.minimum(res["p_val"].to_numpy() * ntot, 1.0)
    return res[["p_val", fcname, "pct.1", "pct.2", "p_val_adj"]]


def find_all_markers(X, groups, *, features=None, test_use: str = "wilcox", logfc_threshold: float = 0.25,
                     min_pct: float = 0.1, min_diff_pct: float = -np.inf, only_pos: bool = False,
                     return_thresh: float = 0.01, max_cells_per_ident: float = np.inf, random_seed: int = 1,
                     latent_vars=None, min_cells_group: int = 3, pseudocount_use: float = 1.0, base: float = 2.0,
                     norm_method: str | None = "LogNormalize", slot: str = "data",
                     levels=None) -> pd.DataFrame:
    """Seurat ``FindAllMarkers``: each identity vs all other cells.

    Clusters are visited in ``levels`` order (default: categorical order, else sorted). Note that
    ``Idents(obj) <- "meta_column"`` in Seurat creates the factor with levels in order of FIRST APPEARANCE in the
    object, so pass ``levels=pd.unique(labels)`` to reproduce the row order of such an R run.

    Returns the concatenation of per-cluster :func:`find_markers` tables, keeping ``p_val < return_thresh``
    (raw p-value, as Seurat), with ``cluster`` and ``gene`` columns. Clusters with fewer than
    ``min_cells_group`` cells are skipped (Seurat reports an error message and continues).
    ``random_seed`` is passed unchanged to every per-cluster :func:`find_markers` call (as Seurat).
    For ``test_use="wilcox"`` without cell downsampling every cluster is tested from a single rank pass.
    """
    random_seed = check_seed(random_seed, "random_seed", r_seed=True)
    X = as_csr(X)
    P = X.shape[1]
    names = _names(features, P)
    g = groups if isinstance(groups, pd.Series) else pd.Series(np.asarray(groups))
    cat = pd.Series(pd.Categorical(g, categories=None if levels is None else list(levels)))         if (levels is not None or not isinstance(g.dtype, pd.CategoricalDtype)) else g
    levels = [l for l in cat.cat.categories if (cat == l).any()]  # sort(unique(Idents))
    fast = test_use == "wilcox" and max_cells_per_ident == np.inf
    allp = None
    if fast:  # one pass: U statistic of every cluster vs rest for every gene
        C = X.tocsc(copy=True)
        C.eliminate_zeros()
        C.sort_indices()
        if C.nnz and C.data.min() < 0:
            fast = False
    if fast:
        codes = pd.Categorical(cat, categories=levels).codes.astype(np.int64)
        N = X.shape[0]
        G = len(levels)
        rs, nnz, _, ties_all = _rank_stats(C.indptr.astype(np.int64), C.indices.astype(np.int64),
                                           C.data.astype(np.float64), codes, G, N)
        ties_all = np.where(np.diff(C.indptr) == 0, float(N) ** 3 - N, ties_all)
        n_g = np.bincount(codes, minlength=G).astype(np.float64)
        nz = (N - np.diff(C.indptr)).astype(np.float64)
        ust = rs + (n_g[None, :] - nnz) * ((nz + 1.0) / 2.0)[:, None] - n_g * (n_g + 1) / 2.0
        allp = np.column_stack([_limma_ranksum_p(ust[:, k], n_g[k], N - n_g[k], ties_all) for k in range(G)])
    out = []
    for gi, lev in enumerate(levels):
        m1 = (cat == lev).to_numpy()
        if m1.sum() < min_cells_group or (~m1).sum() < min_cells_group:
            continue
        if fast:
            fc = fold_change(X, m1, ~m1, norm_method=norm_method, slot=slot, pseudocount=pseudocount_use,
                             base=base, features=names)
            fcn = fc.columns[0]
            pa = np.maximum(fc["pct.1"].to_numpy(), fc["pct.2"].to_numpy())
            keep = (pa >= min_pct) & (pa - np.minimum(fc["pct.1"], fc["pct.2"]).to_numpy() >= min_diff_pct)
            tot = fc[fcn].to_numpy()
            keep &= (tot >= logfc_threshold) if only_pos else (np.abs(tot) >= logfc_threshold)
            res = pd.DataFrame({"p_val": allp[keep, gi]}, index=names[keep]).join(fc)
            if only_pos:
                res = res[res[fcn] > 0]
            res = res.iloc[np.lexsort((-res[fcn].to_numpy(), res["p_val"].to_numpy()))]
            res["p_val_adj"] = np.minimum(res["p_val"].to_numpy() * P, 1.0)
            res = res[["p_val", fcn, "pct.1", "pct.2", "p_val_adj"]]
        else:
            res = find_markers(X, features=names, cells_1=m1, test_use=test_use, logfc_threshold=logfc_threshold,
                               min_pct=min_pct, min_diff_pct=min_diff_pct, only_pos=only_pos,
                               max_cells_per_ident=max_cells_per_ident, random_seed=random_seed,
                               latent_vars=latent_vars, min_cells_group=min_cells_group,
                               pseudocount_use=pseudocount_use, base=base, norm_method=norm_method, slot=slot)
        if len(res) == 0:
            continue
        res = res.iloc[np.lexsort((-res.iloc[:, 1].to_numpy(), res["p_val"].to_numpy()))]
        res = res[res["p_val"] < return_thresh]
        if len(res):
            res = res.assign(cluster=lev, gene=res.index.to_numpy())
            out.append(res)
    if not out:
        return pd.DataFrame(columns=["p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj", "cluster", "gene"])
    res = pd.concat(out)
    if only_pos:
        res = res[res.iloc[:, 1] > 0]
    return res


# ============================================================================ AverageExpression
def average_expression(X, obs: pd.DataFrame | None = None, group_by=None, *, features=None, layer: str = "data",
                       method: str = "average", transpose: bool = False, sep: str = "_") -> pd.DataFrame:
    """Seurat ``AverageExpression`` (``method='average'``) / ``AggregateExpression``-style sums (``'aggregate'``).

    ``layer='data'`` averages ``expm1(data)`` (non-log values, as Seurat); ``'counts'`` / ``'scale.data'`` use
    the values as given. ``group_by`` is a column name, list of columns (labels joined with ``sep``, first
    column varying slowest -- Seurat's ``"a_b"`` naming) or a per-cell label vector. Cells with NA in a
    grouping variable are dropped; grouping variables with a single level are ignored (Seurat); if none
    remain the single group ``"all"`` is returned.

    Returns features x groups (Seurat orientation); ``transpose=True`` gives groups x features.
    """
    X = as_csr(X) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    N, P = X.shape
    names = _names(features, P)
    cols = None
    if isinstance(group_by, (list, tuple)) and obs is not None and all(isinstance(b, str) for b in group_by):
        cols = [b for b in group_by if obs[b].dropna().nunique() > 1]
        if not cols:
            codes, groups = np.zeros(N, dtype=np.int64), pd.DataFrame(index=["all"])
        else:
            codes, groups = factorize_groups(obs, cols, N, sep=sep)
    else:
        codes, groups = factorize_groups(obs, group_by, N, sep=sep)
    keep = codes >= 0
    G = len(groups)
    cnt = np.bincount(codes[keep], minlength=G).astype(np.float64)
    w = (1.0 / cnt[codes[keep]]) if method == "average" else np.ones(keep.sum())
    S = sp.csr_matrix((w, (np.flatnonzero(keep), codes[keep])), shape=(N, G))
    M = as_csr(X) if sp.issparse(X) else X
    if layer == "data":
        if sp.issparse(M):
            M = M.copy()
            M.data = np.expm1(M.data)
        else:
            M = np.expm1(M)
    R = (M.T @ S)
    R = R.toarray() if sp.issparse(R) else np.asarray(R)
    df = pd.DataFrame(R, index=names, columns=groups.index)
    return df.T if transpose else df


# ============================================================================ CellCycleScoring
def cell_cycle_scoring(X, features, s_features=None, g2m_features=None, *, ctrl: int | None = None,
                       nbin: int = 24, seed: int = 1, cells=None, pool=None) -> pd.DataFrame:
    """Seurat ``CellCycleScoring``: ``S.Score`` / ``G2M.Score`` (``AddModuleScore``) and ``Phase``.

    ``X`` is the log-normalized (cells x features) matrix; default gene sets are ``Seurat::cc.genes``.
    ``ctrl`` defaults to the size of the smaller input set (as Seurat). Phase: both scores < 0 -> ``G1``;
    equal maxima -> ``Undecided``; else the larger of ``S`` / ``G2M``.
    ``seed`` (default 1, as Seurat) is the ``AddModuleScore`` R seed; both scores share one RNG stream.
    """
    from .scores import add_module_score

    s_features = list(CC_GENES["s.genes"] if s_features is None else s_features)
    g2m_features = list(CC_GENES["g2m.genes"] if g2m_features is None else g2m_features)
    if ctrl is None:
        ctrl = min(len(s_features), len(g2m_features))
    sc = add_module_score(X, features, {"S.Score": s_features, "G2M.Score": g2m_features}, pool=pool, nbin=nbin,
                          ctrl=ctrl, seed=seed, cells=cells)
    s, g = sc["S.Score"].to_numpy(), sc["G2M.Score"].to_numpy()
    phase = np.where((s < 0) & (g < 0), "G1", np.where(s == g, "Undecided", np.where(s > g, "S", "G2M")))
    sc["Phase"] = phase
    return sc


# ============================================================================ ScaleData
@nb.njit(parallel=True, cache=True)
def _scale_dense_inplace(Z, center, do_scale, scale_max):
    """FastRowScale on a cells x genes dense block (columns = genes)."""
    n, p = Z.shape
    for j in nb.prange(p):
        m = 0.0
        for i in range(n):
            m += Z[i, j]
        m /= n
        v = 0.0
        for i in range(n):
            d = Z[i, j] - m if center else Z[i, j]  # base::scale: root-mean-square when not centering
            v += d * d
        sd = np.sqrt(v / (n - 1)) if n > 1 else 0.0
        for i in range(n):
            x = Z[i, j]
            if center:
                x -= m
            if do_scale:
                x = x / sd if sd > 0 else np.nan
            if x > scale_max:
                x = scale_max
            if not np.isfinite(x):
                x = 0.0
            Z[i, j] = x


def scale_data(X, features=None, *, select=None, vars_to_regress=None, latent_data: pd.DataFrame | None = None,
               split_by=None, model_use: str = "linear", do_scale: bool = True, do_center: bool = True,
               scale_max: float = 10.0, block_size: int = 1000, dtype=np.float64):
    """Seurat ``ScaleData`` for dense output (cells x selected features).

    Parameters
    ----------
    X : cells x features log-normalized matrix. ``features`` = its feature names; ``select`` = names to scale
        (Seurat ``features=``; default all).
    vars_to_regress : columns of ``latent_data`` and / or feature names in ``X`` regressed out per gene with
        ``lm`` (``model_use='linear'``; residuals are then centered/scaled). ``poisson`` / ``negbinom`` are
        not ported.
    split_by : per-cell labels; regression and scaling are done separately within each split.

    Returns ``(Z, feature_names)``; clipping at ``scale_max`` is upper-tail only, like Seurat.
    """
    if model_use != "linear":
        raise NotImplementedError("only model_use='linear' regression is ported")
    X = as_csr(X) if sp.issparse(X) else np.asarray(X)
    N, P = X.shape
    names = _names(features, P)
    sel = np.arange(P) if select is None else names.get_indexer([f for f in select if f in names])
    if (sel < 0).any():
        raise KeyError("select contains unknown features")
    out_names = names[sel]
    lat = None
    if vars_to_regress is not None:
        vars_to_regress = list(vars_to_regress)
        parts = {}
        if latent_data is not None:
            for v in vars_to_regress:
                if v in latent_data.columns:
                    parts[v] = latent_data[v].reset_index(drop=True)
        for v in vars_to_regress:
            if v not in parts and v in names:
                col = X[:, names.get_loc(v)]
                parts[v] = pd.Series(col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel())
        if not parts:
            raise ValueError("None of the requested variables to regress are present")
        lat = pd.DataFrame(parts)
    splits = [np.arange(N)] if split_by is None else [np.flatnonzero(np.asarray(split_by) == s)
                                                     for s in pd.unique(pd.Series(np.asarray(split_by)))]
    from scipy.linalg import qr as _qr

    Z = np.zeros((N, sel.size), dtype=dtype)
    for rows in splits:
        Xr = X[rows] if not sp.issparse(X) else X[rows]
        Dl = _design(lat.iloc[rows] if lat is not None else None, rows.size) if lat is not None else None
        if Dl is not None:
            Q, R, _ = _qr(Dl, mode="economic", pivoting=True)
            rank = int((np.abs(np.diag(R)) > np.abs(R[0, 0]) * 1e-7).sum())
            Q = Q[:, :rank]
        for a in range(0, sel.size, block_size):
            cols = sel[a:a + block_size]
            B = Xr[:, cols]
            B = B.toarray() if sp.issparse(B) else np.array(B, dtype=np.float64)
            B = B.astype(np.float64, copy=False)
            if Dl is not None:
                B = B - Q @ (Q.T @ B)
            if do_scale or do_center or np.isfinite(scale_max):
                B = np.ascontiguousarray(B)
                _scale_dense_inplace(B, do_center, do_scale, scale_max if np.isfinite(scale_max) else np.inf)
            Z[np.ix_(rows, np.arange(a, a + cols.size))] = B
    return Z, out_names


# ============================================================================ NormalizeData
def normalize_data(X, method: str = "LogNormalize", scale_factor: float = 1e4, margin: int = 1):
    """Seurat ``NormalizeData``: ``LogNormalize``, ``CLR`` or ``RC`` (relative counts * ``scale_factor``).

    ``margin`` follows Seurat's CLR convention: ``margin=1`` normalizes each *feature* across cells (Seurat's
    default), ``margin=2`` each *cell* across features (the recommended setting for ADT / HTO).
    """
    if method == "LogNormalize":
        return pp.log_normalize(X, scale_factor=scale_factor)
    if method == "CLR":
        return pp.clr(X, margin=margin)
    if method == "RC":
        Xc = as_csr(X, dtype=np.float64)
        tot = np.asarray(Xc.sum(1)).ravel()
        out = Xc.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out.data = out.data / np.repeat(tot, np.diff(Xc.indptr)) * scale_factor
        return out
    raise ValueError("method must be LogNormalize, CLR or RC")


# ============================================================================ metadata / subsetting helpers
def add_metadata(obs: pd.DataFrame, metadata, col_name: str | None = None) -> pd.DataFrame:
    """Seurat ``AddMetaData``: align ``metadata`` (DataFrame / Series indexed by cell barcode, or a dict) to
    ``obs`` by index and return a new ``obs`` with the columns added (missing cells -> NA). Existing columns are
    overwritten, like Seurat.
    """
    if isinstance(metadata, dict):
        metadata = pd.DataFrame(metadata)
    if isinstance(metadata, pd.Series):
        metadata = metadata.to_frame(col_name or metadata.name or "metadata")
    elif col_name is not None and metadata.shape[1] == 1:
        metadata = metadata.set_axis([col_name], axis=1)
    out = obs.drop(columns=[c for c in metadata.columns if c in obs.columns])
    return out.join(metadata.reindex(out.index))


def which_cells(obs: pd.DataFrame, expression: str | None = None, *, idents=None, ident_col: str | None = None,
                cells=None, invert: bool = False) -> np.ndarray:
    """Seurat ``WhichCells`` / the row selection of ``subset``: boolean mask over ``obs``.

    ``expression`` is a pandas ``eval`` string on ``obs`` columns (e.g. ``"timepoint == 'Baseline' and
    nCount_RNA > 1000"``; use backticks for names with dots, e.g. ``"`percent.mt` < 10"``). ``idents`` selects
    values of ``ident_col``; ``cells`` restricts to barcodes. Conditions are AND-ed.
    """
    m = np.ones(len(obs), dtype=bool)
    if expression is not None:
        m &= obs.eval(expression).fillna(False).to_numpy(dtype=bool)
    if idents is not None:
        m &= obs[ident_col].isin(list(idents) if not isinstance(idents, str) else [idents]).to_numpy()
    if cells is not None:
        m &= obs.index.isin(list(cells))
    return ~m if invert else m


def split_object(ds, by: str) -> dict:
    """Seurat ``SplitObject``: ``{level: Dataset}`` split by an ``obs`` column (uses ``Dataset.subset``)."""
    s = ds.obs[by]
    levels = s.cat.categories if isinstance(s.dtype, pd.CategoricalDtype) else sorted(s.dropna().unique())
    return {l: ds.subset((s == l).to_numpy()) for l in levels if (s == l).any()}


def rename_idents(labels, mapping: dict) -> pd.Series:
    """Seurat ``RenameIdents``: map identity labels (unmapped labels kept; several old levels may merge)."""
    s = labels if isinstance(labels, pd.Series) else pd.Series(np.asarray(labels))
    return s.astype(object).map(lambda v: mapping.get(v, v)).astype("category")
