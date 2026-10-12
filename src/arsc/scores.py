"""Gene-signature scores: UCell and Seurat ``AddModuleScore``.

``score_ucell``      -- port of ``UCell::ScoreSignatures_UCell`` (Mann-Whitney U on
                        per-cell descending ranks, ranks above ``max_rank`` capped).
``add_module_score`` -- port of ``Seurat::AddModuleScore`` including its random
                        control-gene selection; with ``seed`` set it reproduces R's
                        ``set.seed(seed)`` draws exactly (R-compatible RNG).
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from ._rrng import RRandom
from ._utils import as_csr

__all__ = ["score_ucell", "add_module_score", "parse_signature"]


# --------------------------------------------------------------------------- UCell
def parse_signature(genes) -> tuple[list, list]:
    """Split UCell-style genes into (positive, negative): trailing ``-`` = negative, ``+`` stripped."""
    genes = list(genes)
    neg = [g[:-1] for g in genes if g.endswith("-")]
    pos = [g[:-1] if g.endswith("+") else g for g in genes if not g.endswith("-")]
    return pos, neg


@nb.njit(parallel=True, cache=True)
def _ucell_ranks(indptr, indices, data, cols, n_total, n_missing, fill):
    """Descending average ranks of genes ``cols`` (-1 = missing gene) in every cell.

    The rank universe per cell is all genes plus ``n_missing`` appended rows
    holding ``fill`` (UCell's check_genes adds missing genes with min(matrix)).
    """
    n_cells = indptr.size - 1
    G = cols.size
    out = np.empty((n_cells, G))
    for i in nb.prange(n_cells):
        s, e = indptr[i], indptr[i + 1]
        nnz = e - s
        vals = np.sort(data[s:e])  # ascending
        n_zero = n_total - n_missing - nnz  # implicit zeros among real genes
        for g in range(G):
            c = cols[g]
            if c < 0:
                x = fill
            else:
                # binary search column c in this row's sorted indices
                lo, hi = s, e
                while lo < hi:
                    mid = (lo + hi) // 2
                    if indices[mid] < c:
                        lo = mid + 1
                    else:
                        hi = mid
                x = data[lo] if (lo < e and indices[lo] == c) else 0.0
            # counts among stored values: greater / equal
            gt = nnz - np.searchsorted(vals, x, side="right")
            eq = np.searchsorted(vals, x, side="right") - np.searchsorted(vals, x, side="left")
            if 0.0 > x:
                gt += n_zero
            elif 0.0 == x:
                eq += n_zero
            if fill > x:
                gt += n_missing
            elif fill == x:
                eq += n_missing
            out[i, g] = gt + (eq + 1) / 2.0
    return out


def _u_stat(ranks: np.ndarray, max_rank: int) -> np.ndarray:
    """UCell ``u_stat`` applied to each row of a cells x L rank matrix."""
    L = ranks.shape[1]
    if L == 0:
        return np.zeros(ranks.shape[0])
    insig = ranks > max_rank
    r = np.where(insig, max_rank + 1, ranks)
    u = r.sum(1) - L * (L + 1) / 2.0
    auc = 1.0 - u / (L * max_rank)
    auc[insig.all(1)] = 0.0
    return auc


def score_ucell(X, features, signatures: dict, *, max_rank: int = 1500, w_neg: float = 1.0,
                suffix: str = "_UCell", cells=None) -> pd.DataFrame:
    """UCell scores (``ScoreSignatures_UCell`` defaults; use raw counts like UCell's default).

    Parameters
    ----------
    X : cells x genes sparse matrix (non-negative).
    features : gene names (columns of ``X``).
    signatures : ``{name: [genes]}``; ``"GENE-"`` marks a negative gene, ``"GENE+"``
        is the same as ``"GENE"``. Score = max(0, U_pos - w_neg * U_neg).
    max_rank : rank cap (``maxRank``); lowered to the number of genes if larger.

    Notes
    -----
    Faithful to UCell 2.6 quirks: genes missing from the data (union across all
    signatures) are appended as rows of ``min(X)`` before ranking, which enlarges
    the zero-tie block; duplicated genes within a signature count twice.
    """
    X = as_csr(X)
    feats = pd.Index(features)
    sigs = {k: parse_signature(v) for k, v in signatures.items()}
    all_genes = []
    for pos, neg in sigs.values():
        all_genes += pos + neg
    missing = [g for g in dict.fromkeys(all_genes) if g not in feats]
    n_missing = len(missing)
    n_total = X.shape[1] + n_missing
    max_rank = int(min(max_rank, n_total))
    for name, (pos, neg) in sigs.items():
        if len(pos) + len(neg) > max_rank:
            raise ValueError(f"signature {name} has more genes than maxRank")
    fill = 0.0
    if n_missing:
        has_zero = X.nnz < X.shape[0] * X.shape[1]
        fill = float(min(X.data.min() if X.nnz else 0.0, 0.0 if has_zero else np.inf))
    needed = list(dict.fromkeys(all_genes))
    cols = np.array([feats.get_loc(g) if g in feats else -1 for g in needed], dtype=np.int64)
    ranks = _ucell_ranks(X.indptr, X.indices, X.data.astype(np.float64, copy=False), cols,
                         n_total, n_missing, fill)
    pos_of = {g: i for i, g in enumerate(needed)}
    out = {}
    for name, (pos, neg) in sigs.items():
        up = _u_stat(ranks[:, [pos_of[g] for g in pos]], max_rank) if pos else 0.0
        un = _u_stat(ranks[:, [pos_of[g] for g in neg]], max_rank) if neg else 0.0
        out[f"{name}{suffix}"] = np.maximum(up - w_neg * un, 0.0)
    return pd.DataFrame(out, index=cells)


# --------------------------------------------------------------------------- AddModuleScore
def _quantile7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R ``quantile(x, probs, type = 7)``."""
    xs = np.sort(x)
    n = xs.size
    index = 1 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(np.int64)
    hi = np.ceil(index).astype(np.int64)
    qs = xs[lo - 1].copy()
    xhi = xs[hi - 1]
    i = (index > lo) & (xhi != qs)
    h = (index - lo)[i]
    qs[i] = (1 - h) * qs[i] + h * xhi[i]
    return qs


def _cut_number(x: np.ndarray, n: int) -> np.ndarray:
    """ggplot2 ``cut_number(x, n, labels = FALSE, right = FALSE)`` -> 1-based bin codes."""
    probs = np.concatenate([[0.0], np.arange(1, n) * (1.0 / n), [1.0]])  # seq(0, 1, length.out = n + 1)
    brk = _quantile7(x, probs)
    if np.unique(brk).size != brk.size:
        raise ValueError(f"Insufficient data values to produce {n} bins.")
    code = np.searchsorted(brk, x, side="right")  # right = FALSE: [b_i, b_i+1)
    return np.clip(code, 1, n)  # include.lowest: max value goes to the last bin


def add_module_score(X, features, signatures, *, pool=None, nbin: int = 24, ctrl: int = 100,
                     seed: int | None = 1, name: str = "Cluster", cells=None,
                     return_controls: bool = False):
    """Seurat ``AddModuleScore``: mean expression of each gene set minus a binned control set.

    Parameters
    ----------
    X : cells x genes matrix of normalized expression (Seurat's ``data`` slot).
    features : gene names.
    signatures : list of gene lists, or dict ``{name: genes}`` (dict keys become
        column names; otherwise ``f"{name}{i}"`` like Seurat).
    pool, nbin, ctrl : as in Seurat (genes binned by mean into ``nbin`` bins;
        ``ctrl`` random genes drawn from each signature gene's bin).
    seed : R seed; ``seed=1`` (Seurat's default) reproduces R exactly.
        ``None`` draws with numpy instead (not R-reproducible).

    Returns
    -------
    DataFrame (cells x signatures); with ``return_controls`` also the list of
    control-gene lists.
    """
    import scipy.sparse as sp

    feats = pd.Index(features)
    if isinstance(signatures, dict):
        names, gene_lists = list(signatures), list(signatures.values())
    else:
        gene_lists = list(signatures)
        names = [f"{name}{i + 1}" for i in range(len(gene_lists))]
    gene_lists = [list(dict.fromkeys(g for g in gl if g in feats)) for gl in gene_lists]
    for nm, gl in zip(names, gene_lists):
        if not gl:
            raise ValueError(f"no features of {nm} present in the data")
    rr = RRandom(seed) if seed is not None else None
    rng = np.random.default_rng() if seed is None else None
    Xc = as_csr(X) if sp.issparse(X) else np.asarray(X)
    pool_idx = np.arange(len(feats)) if pool is None else feats.get_indexer(list(pool))
    n_cells = Xc.shape[0]
    if sp.issparse(Xc):  # sum in cell order, like Matrix::rowMeans on the dgCMatrix
        from .preprocessing import _col_mean_var
        mu = _col_mean_var(Xc.indptr, Xc.indices, Xc.data.astype(np.float64, copy=False), n_cells, Xc.shape[1])[0]
    else:
        mu = Xc.sum(0) / n_cells
    data_avg = mu[pool_idx]
    order = np.argsort(data_avg, kind="stable")
    avg_sorted = data_avg[order]
    noise = rr.rnorm(avg_sorted.size) if rr is not None else rng.standard_normal(avg_sorted.size)
    cut = _cut_number(avg_sorted + noise / 1e30, nbin)
    sorted_genes = feats[pool_idx][order]
    bin_of = pd.Series(cut, index=sorted_genes)
    ctrl_lists = []
    for gl in gene_lists:
        picked = []
        for g in gl:
            b = bin_of.loc[g]
            cand = np.flatnonzero(cut == b)  # in sorted order, like which(data.cut == ...)
            idx = rr.sample_int(cand.size, ctrl) if rr is not None else rng.choice(cand.size, ctrl, replace=False)
            picked += list(sorted_genes[cand[idx]])
        ctrl_lists.append(list(dict.fromkeys(picked)))

    def colmeans(genes):
        j = feats.get_indexer(genes)
        sub = Xc[:, j]
        return np.asarray(sub.mean(axis=1)).ravel()

    out = {nm: colmeans(gl) - colmeans(cl) for nm, gl, cl in zip(names, gene_lists, ctrl_lists)}
    df = pd.DataFrame(out, index=cells)
    return (df, ctrl_lists) if return_controls else df
