"""CloneTrace: clonal fate modeling on single-cell embeddings (numpy/numba port).

Port of the Clonotrace R package (Fu, Mathew, Wang, Chen, Lin, Schaff, Shaffer, Pardoll, Jackson,
Zhang. *Deciphering Cell Fate and Clonal Dynamics via Integrative Single-Cell Lineage Modeling*.
bioRxiv 2025.09.01.673503; github.com/yuntianf/Clonotrace, MIT, v1.2.1). The inputs are a cell
embedding (e.g. PCA) and one clone label per cell. A clone is a lineage barcode, or a TCR clonotype,
optionally crossed with timepoint. The steps:

1. :func:`cell_transition` -- kNN Gaussian-kernel graph, double-normalized transition (``embedding2knn``
   + ``compute_transition``).
2. :func:`propagate_clones` -- clone-label spreading over the cell graph, with a bootstrap deviance
   (``label_spreading_bootstrap``, blocked backend). :func:`filter_and_sparsify` applies the
   vignette's deviance cut plus :func:`sparsify` (``mat_sparsify``). The result is ``Q``: cells x
   expanded clones, rows summing to 1.
3. :func:`clone_distance` -- clone x clone distance on a shared-nearest-neighbor cost graph
   (``exp(-shared)``): the fast nearest-neighbor approximation (``graph_clone_nn``) or exact optimal
   transport (``clone_2_ot``). :func:`revise_distance` blends in a centroid distance
   (``revise_clone_distance``).
4. :func:`cluster_clones` (``leiden_dis``), :func:`cmdscale` / :func:`isomds` (clone embedding),
   :func:`clone_dpt` (clone diffusion pseudotime, root picked by :func:`clone_root`).
5. :func:`cell_profiles` (Q x clone->profile), :func:`profile_enrichment` (``cluster_profile_enrich``),
   :func:`fate_genes` (``profile_cluster_DEG``: ridge natural-spline F test, permutation-calibrated),
   and :func:`coembed_graph` (clone-weighted cell kNN distances, ``cell_clone_coembed``).

:func:`run` chains steps 1-4. :func:`clone_table` builds one row per clone, and can join a
:func:`arsc.clones.cyclone` result, so that temporal expansion trajectories (Cyclone) and fate
profiles (CloneTrace) of the same clones sit in one table.

Parity with the R code is tested in ``tests/test_clonotrace.py`` (``CLONOTRACE_PARITY.md``). The
deterministic steps reproduce R to floating-point precision. The random steps (bootstrap,
permutations, Leiden) use numpy/igraph RNGs, so they agree statistically, not bitwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as sps

from ._utils import check_seed
from ._seedlog import records_seed

__all__ = [
    "knn_connectivity", "cell_transition", "label_spreading", "propagate_clones", "sparsify", "filter_and_sparsify",
    "snn_cost_graph", "shortest_paths", "clone_distance", "revise_distance", "cluster_clones", "cmdscale",
    "isomds", "clone_root", "clone_dpt", "cell_profiles", "profile_mass", "profile_enrichment", "ns_basis",
    "soft_cluster_fit", "bin_filter", "fate_genes", "coembed_graph", "CloneProb", "CloneTrace", "run",
    "clone_table", "encode_clones",
]


# =========================================================================== helpers
def _r_sort(values) -> list:
    """Sorted unique non-missing labels (R ``levels(factor(x))`` in the C locale)."""
    return sorted({str(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))})


def encode_clones(clones, levels: Sequence[str] | None = None) -> tuple[np.ndarray, list]:
    """Clone labels -> integer codes (``-1`` = missing) and the level order.

    R's ``as.factor`` orders levels in the session locale. Pass ``levels`` to reproduce a specific R run
    exactly; the default is code-point (C locale) order.

    Parameters
    ----------
    clones : one label per cell (``None`` / NaN = no clone).
    levels : explicit level order (labels not in it become ``-1``).

    Returns
    -------
    tuple ``(codes, levels)``: int64 array per cell and the list of levels.

    Examples
    --------
    >>> from arsc import clonotrace as ct
    >>> codes, levels = ct.encode_clones(["b", None, "a", "b"])
    >>> codes.tolist(), levels
    ([1, -1, 0, 1], ['a', 'b'])
    """
    s = pd.Series(list(clones), dtype=object)
    miss = s.isna().to_numpy()
    lv = list(levels) if levels is not None else _r_sort(s[~miss])
    pos = {v: i for i, v in enumerate(lv)}
    codes = np.array([-1 if m else pos.get(str(v), -1) for v, m in zip(s, miss)], dtype=np.int64)
    if levels is not None and ((codes < 0) & ~miss).any():
        raise KeyError("some clone labels are not in `levels`")
    return codes, lv


def _knn_excl_self(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbors excluding the point itself (like ``dbscan::kNN``); exact Euclidean."""
    from .neighbors import knn
    idx, _ = knn(X, k + 1, method="exact")
    n = X.shape[0]
    rows = np.arange(n)[:, None]
    out = np.empty((n, k), dtype=np.int64)
    for r in range(n):  # drop self (normally column 0; duplicates can move it)
        row = idx[r]
        row = row[row != r][:k]
        out[r] = row
    d = np.sqrt(((X[rows, :] - X[out]) ** 2).sum(-1)) if n * k * X.shape[1] < 5e8 else _pair_dist(X, rows, out)
    return out, d


def _pair_dist(X, rows, out):
    d = np.empty(out.shape)
    for c in range(out.shape[1]):
        d[:, c] = np.sqrt(((X - X[out[:, c]]) ** 2).sum(1))
    return d


def _flat_knn(idx: np.ndarray, d: np.ndarray):
    """R ``knn_flat`` order: column-major over the neighbor rank (all 1st neighbors, then 2nd, ...)."""
    n, k = idx.shape
    node1 = np.tile(np.arange(n), k)
    node2 = idx.T.ravel()
    dist = d.T.ravel()
    return node1, node2, dist


def _symmetrize_long(n1, n2, val, *extra):
    """R ``long_symmetry``: append reversed pairs, drop exact duplicate rows, sort by (node1, node2)."""
    a1 = np.concatenate([n1, n2])
    a2 = np.concatenate([n2, n1])
    av = np.concatenate([val, val])
    ex = [np.concatenate([e, e]) for e in extra]
    keys = [a1, a2, av] + ex
    order = np.lexsort(tuple(reversed(keys)))
    keep = np.ones(order.size, bool)
    for kk in keys:
        s = kk[order]
        if kk is keys[0]:
            same = np.r_[False, s[1:] == s[:-1]]
        else:
            same &= np.r_[False, s[1:] == s[:-1]]
    keep = ~same
    o = order[keep]
    # R keeps the first occurrence then sorts by (node1, node2) (stable on the original order)
    o2 = o[np.lexsort((a2[o], a1[o]))]
    return (a1[o2], a2[o2], av[o2]) + tuple(e[o2] for e in ex)


# =========================================================================== 1. cell graph
def knn_connectivity(embedding, k: int = 30, *, self_loops: bool = False, mode: str = "connectivity") -> sp.csr_matrix:
    """Symmetrized kNN graph with Gaussian kernel weights (R ``embedding2knn``).

    Edges are the union of the kNN relations. With ``sigma_i`` the mean distance over node *i*'s edges, the
    weight is ``K_ij = exp(-d_ij^2 / (sigma_i sigma_j))``.

    Parameters
    ----------
    embedding : cells x dims array.
    k : neighbors per cell (default 30).
    self_loops : add ``(i, i, 0)`` edges after symmetrization (R ``if_self``).
    mode : ``"connectivity"`` (kernel weights, default) or ``"dist"`` (the distances).

    Returns
    -------
    scipy.sparse.csr_matrix, cells x cells, symmetric.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> ct.knn_connectivity(pcs, k=15).shape
    (1597, 1597)
    """
    X = np.ascontiguousarray(embedding, dtype=np.float64)
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, n_cells): got {k} for {n} cells")
    idx, d = _knn_excl_self(X, k)
    n1, n2, dist = _symmetrize_long(*_flat_knn(idx, d))
    if self_loops:
        n1 = np.r_[n1, np.arange(n)]
        n2 = np.r_[n2, np.arange(n)]
        dist = np.r_[dist, np.zeros(n)]
    if mode == "dist":
        return sp.csr_matrix(sp.coo_matrix((dist, (n1, n2)), shape=(n, n)))
    if mode != "connectivity":
        raise ValueError("mode must be 'connectivity' or 'dist'")
    sums = np.bincount(n1, weights=dist, minlength=n)
    cnt = np.bincount(n1, minlength=n)
    sigma = sums / np.maximum(cnt, 1)
    w = np.exp(-dist ** 2 / (sigma[n1] * sigma[n2]))
    return sp.csr_matrix(sp.coo_matrix((w, (n1, n2)), shape=(n, n)))


def cell_transition(embedding, k: int = 30) -> sp.csr_matrix:
    """Double-normalized transition matrix ``T = Z D^-1 K D^-1 Z`` of the kNN kernel (R ``compute_transition``).

    Parameters
    ----------
    embedding : cells x dims array (e.g. 15 PCs).
    k : neighbors per cell for :func:`knn_connectivity` (default 30).

    Returns
    -------
    scipy.sparse.csr_matrix, cells x cells (CloneTrace step 1).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> T = ct.cell_transition(pcs, k=15)
    >>> T.shape
    (1597, 1597)
    """
    K = knn_connectivity(embedding, k)
    return _compute_transition(K)


def _compute_transition(K: sp.csr_matrix) -> sp.csr_matrix:
    rs = np.asarray(K.sum(1)).ravel()
    Q = sp.diags(1.0 / rs)
    T = Q @ K @ Q
    rs2 = np.asarray(T.sum(1)).ravel()
    Z = sp.diags(1.0 / np.sqrt(rs2))
    return sp.csr_matrix(Z @ T @ Z)


# =========================================================================== 2. label spreading
@nb.njit(parallel=True, cache=True)
def _spmm_step(indptr, indices, data, F, Y, alpha, out, rowdiff):
    n, B = F.shape
    for i in nb.prange(n):
        md = 0.0
        for b in range(B):
            acc = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += data[p] * F[indices[p], b]
            v = alpha * acc + (1.0 - alpha) * Y[i, b]
            out[i, b] = v
            dd = abs(v - F[i, b])
            if dd > md:
                md = dd
        rowdiff[i] = md


def _row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    A = sp.csr_matrix(adj, dtype=np.float64, copy=True)
    A.sort_indices()
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg < 1e-12] = 1e-12
    A.data /= np.repeat(deg, np.diff(A.indptr))
    return A


def label_spreading(adj, labels: np.ndarray, n_classes: int | None = None, *, alpha: float = 0.9,
                    max_iter: int = 100, tol: float = 1e-3, block_size: int = 128, epsilon: float = 0.0,
                    _P: sp.csr_matrix | None = None) -> np.ndarray:
    """Label spreading ``F <- alpha P F + (1 - alpha) Y`` from ``F = 0`` (R ``label_spreading_blocked``).

    ``P`` is ``adj`` row-normalized. ``Y[i, c] = log2(n_c) / n_c`` for labeled cells, ``epsilon / C`` for
    unlabeled cells. Classes are processed in blocks of ``block_size``; each block iterates until the max-abs
    change is ``< tol``, exactly as in R (so the block size changes the result slightly).

    Parameters
    ----------
    adj : cells x cells graph (ignored when ``_P`` is given).
    labels : integer class codes per cell, ``-1`` = unlabeled (see :func:`encode_clones`).
    n_classes : number of classes (default ``labels.max() + 1``).
    alpha : propagation weight (default 0.9; CloneTrace uses 0.6).
    max_iter : iterations per block (default 100).
    tol : convergence tolerance on the max-abs change (default 1e-3).
    block_size : classes per block (default 128).
    epsilon : prior mass of unlabeled cells (default 0).
    _P : internal: a precomputed row-normalized ``adj``.

    Returns
    -------
    dense numpy array, cells x classes (unnormalized).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> codes, levels = ct.encode_clones(clones)
    >>> F = ct.label_spreading(ct.cell_transition(pcs, 15), codes, alpha=0.6)
    >>> F.shape == (len(clones), len(levels))
    True
    """
    labs = np.asarray(labels, dtype=np.int64)
    C = int(n_classes if n_classes is not None else labs.max() + 1)
    N = labs.size
    P = _P if _P is not None else _row_normalize(adj)
    if P.shape != (N, N):
        raise ValueError(f"adjacency is {P.shape}, labels have {N} cells")
    cnt = np.bincount(labs[labs >= 0], minlength=C).astype(np.float64)
    wts = np.zeros(C)
    nz = cnt > 0
    wts[nz] = np.log2(cnt[nz]) / cnt[nz]
    labeled = labs >= 0
    prob = np.empty((N, C))
    ind, ptr, dat = P.indices.astype(np.int64), P.indptr.astype(np.int64), P.data
    for s in range(0, C, block_size):
        cols = np.arange(s, min(s + block_size, C))
        B = cols.size
        Y = np.full((N, B), epsilon / C)
        Y[labeled] = 0.0
        inb = labeled & (labs >= s) & (labs < s + B)
        Y[np.flatnonzero(inb), labs[inb] - s] = wts[labs[inb]]
        F = np.zeros((N, B))
        Fn = np.empty_like(F)
        rd = np.empty(N)
        for _ in range(max_iter):
            _spmm_step(ptr, ind, dat, F, Y, float(alpha), Fn, rd)
            F, Fn = Fn, F
            if rd.max() < tol:
                break
        prob[:, cols] = F
    return prob


def _normalize_rows_eps(F: np.ndarray) -> np.ndarray:
    R = F + 1e-12
    rs = R.sum(1)
    rs[rs == 0] = 1
    return R / rs[:, None]


@dataclass
class CloneProb:
    """Output of :func:`propagate_clones`.

    Attributes
    ----------
    prob : cells x clones array of propagated clone probabilities (rows sum to 1).
    deviance : per-cell bootstrap deviance (mean L1 distance from the full-label reference; NaN without
        bootstraps).
    clones : clone names (columns of ``prob``).
    cells : cell names (rows of ``prob``).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> cp.frame().shape == (len(cp.cells), len(cp.clones))
    True
    """
    prob: np.ndarray          # cells x clones, rows normalized (reference spreading with all labels)
    deviance: np.ndarray      # per-cell bootstrap deviance (nan if n_boot == 0)
    clones: list              # column names (clone ids)
    cells: pd.Index           # row names

    def frame(self) -> pd.DataFrame:
        """Probabilities as a cells x clones DataFrame.

        Returns
        -------
        pandas.DataFrame indexed by ``cells`` with one column per clone.

        Examples
        --------
        >>> import arsc, numpy as np
        >>> from arsc import clonotrace as ct
        >>> ds = arsc.datasets.toy()
        >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
        >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
        >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
        >>> clones = t.obs["clone_id"].to_numpy()
        >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
        >>> list(cp.frame().index[:2]) == list(t.obs.index[:2])
        True
        """
        return pd.DataFrame(self.prob, index=self.cells, columns=self.clones)


@records_seed("seed")
def propagate_clones(adj, clones, *, min_size: int = 10, alpha: float = 0.6, sample_rate: float = 0.8,
                     n_boot: int = 48, block_size: int = 128, tol: float = 1e-3, max_iter: int = 100,
                     epsilon: float = 0.0, seed: int = 0, levels: Sequence[str] | None = None,
                     cells=None) -> CloneProb:
    """Spread expanded-clone labels over the cell graph, with a bootstrap deviance (CloneTrace step 2).

    Clones with fewer than ``min_size`` cells are treated as unlabeled. Defaults follow the published vignette:
    ``alpha=0.6``, 48 bootstraps at ``sample_rate=0.8``. For each bootstrap, a random ``round(0.8 n_labeled)``
    subset keeps its labels and the spreading is rerun. The per-cell deviance is the mean L1 distance from the
    full-label reference, over the bootstraps where the cell's labeled/unlabeled status is unchanged. Bootstrap
    draws use numpy's RNG, not R's, so the deviances agree with R only statistically; the spreading itself is
    deterministic, so same input + seed -> identical deviances.

    Parameters
    ----------
    adj : cell graph, normally :func:`cell_transition`.
    clones : one label per cell (``None`` / NaN = no clone).
    min_size : minimum cells of a clone to be propagated (default 10, as the vignette).
    alpha : propagation weight (default 0.6).
    sample_rate : fraction of labeled cells kept per bootstrap (default 0.8).
    n_boot : number of bootstraps (default 48; 0 = no deviance).
    block_size, tol, max_iter, epsilon : passed to :func:`label_spreading`.
    seed : integer seed of a local ``np.random.default_rng`` (default 0).
    levels : explicit clone level order (see :func:`encode_clones`).
    cells : cell names for the rows (default ``0..n-1``); pass them to join results back to ``obs``.

    Returns
    -------
    :class:`CloneProb`.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> len(cp.clones)
    21
    """
    seed = check_seed(seed)
    s = pd.Series(list(clones), dtype=object)
    if adj is not None and len(s) != adj.shape[0]:
        raise ValueError(f"propagate_clones: {len(s)} clone labels for a {adj.shape[0]}-cell graph; pass one label "
                         "per graph node (same cell order as the embedding)")
    vc = s.dropna().astype(str).value_counts()
    big = set(vc.index[vc >= min_size])
    lab = s.where(s.astype(str).isin(big) & s.notna(), None)
    codes, lv = encode_clones(lab, levels)
    if not lv:
        raise ValueError(f"no clone has >= {min_size} cells (largest: {int(vc.max()) if len(vc) else 0}); "
                         "lower min_size or check the clone column")
    P = _row_normalize(adj)
    C = len(lv)
    kw = dict(alpha=alpha, max_iter=max_iter, tol=tol, block_size=block_size, epsilon=epsilon, _P=P)
    ref = _normalize_rows_eps(label_spreading(None, codes, C, **kw))
    dev = np.full(codes.size, np.nan)
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        labeled = np.flatnonzero(codes >= 0)
        take = max(1, int(np.round(labeled.size * sample_rate)))
        full_flag = codes >= 0
        dsum = np.zeros(codes.size)
        fsum = np.zeros(codes.size)
        for _ in range(n_boot):
            sub = np.full(codes.size, -1, dtype=np.int64)
            pick = rng.choice(labeled, size=take, replace=False)
            sub[pick] = codes[pick]
            same = (full_flag == (sub >= 0)).astype(np.float64)
            Fb = _normalize_rows_eps(label_spreading(None, sub, C, **kw))
            dsum += np.abs(Fb - ref).sum(1) * same
            fsum += same
        dev = dsum / np.maximum(1, fsum)
    idx = pd.Index(cells) if cells is not None else pd.RangeIndex(codes.size)
    return CloneProb(ref, dev, lv, idx)


# =========================================================================== sparsify
def _mass_filter_rows(M: np.ndarray, thresh: float) -> np.ndarray:
    M = M.copy()
    tot = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = M / tot[:, None]
    srt = -np.sort(-norm, axis=1)
    cs = np.cumsum(srt, axis=1)
    flag = cs < thresh
    nflag = flag.sum(1)
    lwr = np.where(nflag > 0, srt[np.arange(M.shape[0]), np.maximum(nflag - 1, 0)], thresh)
    with np.errstate(invalid="ignore"):
        drop = norm < lwr[:, None]
    M[drop] = 0.0
    return M


def sparsify(mat, row_mass: float = 0.9, col_mass: float = 0.9) -> np.ndarray:
    """Keep the top ``row_mass`` of each row, then of each column (R ``mat_sparsify``; dense in, dense out).

    Per vector: sort the normalized masses in decreasing order. ``lwr`` is the smallest mass whose cumulative
    sum is still ``< thresh``. Entries below ``lwr`` are zeroed, so ties are kept and the entry that crosses
    ``thresh`` is dropped. Rows are filtered first; columns are then filtered on the result.

    Parameters
    ----------
    mat : dense cells x clones array.
    row_mass : mass kept per row (default 0.9).
    col_mass : mass kept per column (default 0.9).

    Returns
    -------
    dense numpy array of the same shape (not renormalized).

    Examples
    --------
    >>> import numpy as np
    >>> from arsc import clonotrace as ct
    >>> ct.sparsify(np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])).tolist()
    [[0.6, 0.0, 0.0], [0.0, 0.2, 0.6]]
    """
    M = np.asarray(mat.toarray() if sp.issparse(mat) else mat, dtype=np.float64)
    M = _mass_filter_rows(M, row_mass)
    return _mass_filter_rows(M.T, col_mass).T


def filter_and_sparsify(cp: CloneProb, max_deviance: float = 0.3, row_mass: float = 0.9,
                        col_mass: float = 0.9) -> pd.DataFrame:
    """Vignette post-processing of :func:`propagate_clones`: deviance filter, renormalize, sparsify, renormalize.

    Parameters
    ----------
    cp : a :class:`CloneProb`.
    max_deviance : drop cells with deviance ``>= max_deviance`` (default 0.3).
    row_mass, col_mass : passed to :func:`sparsify` (default 0.9 each).

    Returns
    -------
    ``Q``: cells x clones DataFrame, rows summing to 1; empty rows and columns are dropped.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> bool(np.allclose(Q.sum(1), 1))
    True
    """
    keep = cp.deviance < max_deviance if np.isfinite(cp.deviance).any() else np.ones(len(cp.cells), bool)
    Q = cp.prob[keep]
    Q = Q / Q.sum(1, keepdims=True)
    Q = sparsify(Q, row_mass, col_mass)
    with np.errstate(invalid="ignore"):
        Q = Q / Q.sum(1, keepdims=True)
    Q = np.nan_to_num(Q)
    df = pd.DataFrame(Q, index=cp.cells[keep], columns=cp.clones)
    return df.loc[df.sum(1) > 0, df.sum(0) > 0]


# =========================================================================== 3. clone distances
def snn_cost_graph(embedding, k: int = 10) -> sp.csr_matrix:
    """Shared-nearest-neighbor cost graph (``bluster::makeSNNGraph(type="number")`` + ``exp(-w)``).

    ``w_ij = |({i} + kNN_i) & ({j} + kNN_j)|``, i.e. the off-diagonal of ``(I + A)(I + A)^T``. There is an edge
    for every pair with ``w >= 1``, and its cost is ``exp(-w_ij)``.

    Parameters
    ----------
    embedding : cells x dims array.
    k : neighbors per cell (default 10).

    Returns
    -------
    scipy.sparse.csr_matrix of edge costs, cells x cells.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> ct.snn_cost_graph(pcs, k=10).shape
    (1597, 1597)
    """
    X = np.ascontiguousarray(embedding, dtype=np.float64)
    n = X.shape[0]
    idx, _ = _knn_excl_self(X, k)
    rows = np.repeat(np.arange(n), k + 1)
    cols = np.concatenate([np.arange(n)[:, None], idx], axis=1).ravel()
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    S = (A @ A.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    S.data = np.exp(-S.data)
    S.sort_indices()
    return S


@nb.njit(cache=True)
def _dijkstra(indptr, indices, w, src, dist, heap, pos):
    n = dist.size
    for v in range(n):
        dist[v] = np.inf
        pos[v] = -1
    size = 0
    dist[src] = 0.0
    heap[0] = src
    pos[src] = 0
    size = 1
    while size > 0:
        u = heap[0]
        size -= 1
        pos[u] = -2  # settled
        if size > 0:
            last = heap[size]
            # sift down
            i = 0
            while True:
                l = 2 * i + 1
                if l >= size:
                    break
                r = l + 1
                c = l
                if r < size and dist[heap[r]] < dist[heap[l]]:
                    c = r
                if dist[heap[c]] < dist[last]:
                    heap[i] = heap[c]
                    pos[heap[i]] = i
                    i = c
                else:
                    break
            heap[i] = last
            pos[last] = i
        du = dist[u]
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if pos[v] == -2:
                continue
            nd = du + w[p]
            if nd < dist[v]:
                dist[v] = nd
                if pos[v] == -1:
                    i = size
                    size += 1
                else:
                    i = pos[v]
                # sift up
                while i > 0:
                    par = (i - 1) // 2
                    if dist[heap[par]] > nd:
                        heap[i] = heap[par]
                        pos[heap[i]] = i
                        i = par
                    else:
                        break
                heap[i] = v
                pos[v] = i


@nb.njit(parallel=True, cache=True)
def _dijkstra_rows(indptr, indices, w, sources, out):
    n = out.shape[1]
    for t in nb.prange(sources.size):
        heap = np.empty(n, np.int64)
        pos = np.empty(n, np.int64)
        d = np.empty(n)
        _dijkstra(indptr, indices, w, sources[t], d, heap, pos)
        out[t, :] = d


def shortest_paths(graph: sp.csr_matrix, sources) -> np.ndarray:
    """Weighted shortest-path distances from ``sources`` to all nodes (numba Dijkstra, parallel).

    Parameters
    ----------
    graph : cells x cells sparse graph of non-negative edge costs (e.g. :func:`snn_cost_graph`).
    sources : node indices to start from.

    Returns
    -------
    numpy array, len(sources) x n_nodes (``inf`` = unreachable).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> d = ct.shortest_paths(ct.snn_cost_graph(pcs, 10), [0, 1])
    >>> d.shape, float(d[0, 0])
    ((2, 1597), 0.0)
    """
    G = sp.csr_matrix(graph)
    src = np.asarray(sources, dtype=np.int64)
    out = np.empty((src.size, G.shape[0]))
    _dijkstra_rows(G.indptr.astype(np.int64), G.indices.astype(np.int64), G.data.astype(np.float64), src, out)
    return out


@nb.njit(parallel=True, cache=True)
def _nn_stats(indptr, indices, w, sources, mptr, midx, k, S, M):
    """For each source cell: mean of the k smallest and the mean distance to each clone's cells."""
    n = indptr.size - 1
    C = mptr.size - 1
    for t in nb.prange(sources.size):
        heap = np.empty(n, np.int64)
        pos = np.empty(n, np.int64)
        d = np.empty(n)
        _dijkstra(indptr, indices, w, sources[t], d, heap, pos)
        best = np.empty(k)
        for c in range(C):
            m = mptr[c + 1] - mptr[c]
            tot = 0.0
            for q in range(k):
                best[q] = np.inf
            for p in range(mptr[c], mptr[c + 1]):
                v = d[midx[p]]
                tot += v
                if v < best[k - 1]:
                    q = k - 1
                    while q > 0 and best[q - 1] > v:
                        best[q] = best[q - 1]
                        q -= 1
                    best[q] = v
            M[t, c] = tot / m
            if m < k:
                S[t, c] = np.nan
            else:
                s = 0.0
                for q in range(k):
                    s += best[q]
                S[t, c] = s / k


def _as_Q(Q) -> tuple[sp.csc_matrix, list, pd.Index | None]:
    if isinstance(Q, pd.DataFrame):
        return sp.csc_matrix(Q.to_numpy(dtype=np.float64)), [str(c) for c in Q.columns], Q.index
    M = sp.csc_matrix(Q, dtype=np.float64)
    return M, [str(i) for i in range(M.shape[1])], None


def clone_distance(embedding, Q, *, method: str = "nn", graph_k: int = 10, prob_thresh: float | None = None,
                   k: int = 2, graph: sp.csr_matrix | None = None, square: bool = True):
    """Clone x clone distance on the SNN cost graph (R ``clone_disance`` with ``revised=FALSE``).

    The graph is :func:`snn_cost_graph` ``(embedding, graph_k)``. ``method="nn"`` (default; ``graph_clone_nn``):
    cells with ``Q >= prob_thresh`` (default 0.1) belong to the clone. The distance between clones *i* and *j*
    is the average of (mean over *i*'s cells of the mean of the ``k`` smallest path lengths to *j*'s cells)
    and the same from *j* to *i*; a single-cell clone uses the plain mean. ``method="exact"``: the
    Wasserstein-1 distance between the thresholded (default 0.05), renormalized clone distributions, with
    shortest-path ground costs (``clone_2_ot``); solved with POT (``ot.emd2``) when installed, else HiGHS. It is
    slow for many clones.

    Parameters
    ----------
    embedding : cells x dims, rows aligned to ``Q``.
    Q : cells x clones (DataFrame keeps the clone names), e.g. :func:`filter_and_sparsify`.
    method : ``"nn"`` (default) or ``"exact"``.
    graph_k : neighbors of the SNN cost graph (default 10).
    prob_thresh : membership threshold (default 0.1 for ``"nn"``, 0.05 for ``"exact"``).
    k : number of smallest path lengths averaged (``"nn"``; default 2).
    graph : a precomputed cost graph (skips :func:`snn_cost_graph`).
    square : return a square DataFrame (default) or the long table.

    Returns
    -------
    square clone x clone DataFrame, or the long table ``group1, group2, dis`` with 1-based clone indices
    like R.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> D.shape
    (21, 21)
    """
    X = np.ascontiguousarray(embedding, dtype=np.float64)
    Qc, names, _ = _as_Q(Q)
    if Qc.shape[0] != X.shape[0]:
        raise ValueError(f"Q has {Qc.shape[0]} cells, embedding {X.shape[0]}")
    G = graph if graph is not None else snn_cost_graph(X, graph_k)
    C = Qc.shape[1]
    if C < 2:
        raise ValueError("need at least two clones")
    if method == "nn":
        th = 0.1 if prob_thresh is None else prob_thresh
        B = Qc.copy()
        B.data = (B.data >= th).astype(np.float64)
        B.eliminate_zeros()
        sizes = np.diff(B.indptr)
        if (sizes == 0).any():
            raise ValueError(f"clones without cells at prob_thresh={th}: {[names[i] for i in np.flatnonzero(sizes == 0)]}")
        B.sort_indices()
        src = np.unique(B.indices)
        S = np.empty((src.size, C))
        Mn = np.empty((src.size, C))
        _nn_stats(G.indptr.astype(np.int64), G.indices.astype(np.int64), G.data.astype(np.float64), src.astype(np.int64),
                  B.indptr.astype(np.int64), B.indices.astype(np.int64), int(k), S, Mn)
        row_of = np.full(X.shape[0], -1, dtype=np.int64)
        row_of[src] = np.arange(src.size)
        D = np.zeros((C, C))
        for i in range(C):
            ri = row_of[B.indices[B.indptr[i]:B.indptr[i + 1]]]
            mS = S[ri].mean(0)    # mean over i's cells of k-min to each clone
            mM = Mn[ri].mean(0)
            for j in range(i + 1, C):
                if sizes[i] == 1 or sizes[j] == 1:
                    v = mM[j]
                else:
                    rj = row_of[B.indices[B.indptr[j]:B.indptr[j + 1]]]
                    v = 0.5 * (mS[j] + S[rj, i].mean())
                D[i, j] = D[j, i] = v
    elif method == "exact":
        th = 0.05 if prob_thresh is None else prob_thresh
        T = Qc.copy()
        T.data[T.data < th] = 0.0
        T.eliminate_zeros()
        T.sort_indices()
        D = np.zeros((C, C))
        emd = _emd_solver()
        for i in range(C - 1):
            ci = T.indices[T.indptr[i]:T.indptr[i + 1]]
            mi = T.data[T.indptr[i]:T.indptr[i + 1]]
            Di = shortest_paths(G, ci)
            for j in range(i + 1, C):
                cj = T.indices[T.indptr[j]:T.indptr[j + 1]]
                mj = T.data[T.indptr[j]:T.indptr[j + 1]]
                D[i, j] = D[j, i] = emd(mi / mi.sum(), mj / mj.sum(), np.ascontiguousarray(Di[:, cj]))
    else:
        raise ValueError("method must be 'nn' or 'exact'")
    if not np.isfinite(D).all():
        import warnings
        warnings.warn("some clone distances are infinite: the SNN graph is disconnected between those clones "
                      "(increase graph_k or check the embedding)", RuntimeWarning, stacklevel=2)
    if square:
        return pd.DataFrame(D, index=names, columns=names)
    iu = np.triu_indices(C, 1)
    return pd.DataFrame({"group1": iu[0] + 1, "group2": iu[1] + 1, "dis": D[iu]})


def _emd_solver():
    try:
        import ot  # POT

        def solve(a, b, M):
            return float(ot.emd2(a, b, M, numItermax=10_000_000))
        return solve
    except ImportError:
        import warnings
        warnings.warn("POT is not installed: exact OT falls back to scipy's HiGHS linear program, about 40x slower "
                      "(pip install pot, or arsc[clonotrace])", RuntimeWarning, stacklevel=3)
        return _emd_highs


def _emd_highs(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
    """Exact transport cost by linear programming (HiGHS)."""
    from scipy.optimize import linprog
    m, n = M.shape
    if m == 1 or n == 1:
        return float((M * (a[:, None] * b[None, :])).sum())
    rows = np.repeat(np.arange(m), n)
    cols = np.arange(m * n)
    A1 = sp.csr_matrix((np.ones(m * n), (rows, cols)), shape=(m, m * n))
    A2 = sp.csr_matrix((np.ones(m * n), (np.tile(np.arange(n), m), cols)), shape=(n, m * n))
    A = sp.vstack([A1, A2[:-1]]).tocsr()  # one constraint is redundant
    beq = np.concatenate([a, b[:-1]])
    res = linprog(M.ravel(), A_eq=A, b_eq=beq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"OT linear program failed: {res.message}")
    return float(res.fun)


def _square(D) -> tuple[np.ndarray, list]:
    if isinstance(D, pd.DataFrame):
        Dm, names = D.to_numpy(dtype=np.float64), [str(c) for c in D.columns]
    else:
        Dm = np.asarray(D, dtype=np.float64)
        names = [str(i) for i in range(Dm.shape[0])]
    if Dm.ndim != 2 or Dm.shape[0] != Dm.shape[1]:
        raise ValueError(f"expected a square distance matrix, got shape {Dm.shape}")
    if not np.isfinite(Dm).all():
        raise ValueError("distance matrix has non-finite entries (disconnected clones?)")
    return Dm, names


@records_seed("seed")
def revise_distance(D, embedding, Q, *, clusters: pd.DataFrame | np.ndarray | None = None, sample=None,
                    local_rescale: bool = False, local_k: int = 7, resolutions=(0.3, 0.5, 0.8, 1.2),
                    cluster_k: int = 15, mean_weight: float | None = None, s_grid=None, seed: int = 0):
    """Blend the CloneTrace distance with a centroid distance (R ``revise_clone_distance``; default in 1.2.x).

    Both distances are divided by their median over clone pairs (only same-``sample`` pairs when ``sample``
    gives a group per clone). The blend weight ``s`` maximizes the Spearman correlation between blended pair
    distances and the total-variation distance of the clones' cell-cluster compositions. It is chosen per
    clustering, and the median over clusterings is used.

    Parameters
    ----------
    D : clone x clone distance from :func:`clone_distance`.
    embedding : cells x dims, rows aligned to ``Q``.
    Q : cells x clones probabilities.
    clusters : cells x resolutions labels (give it for reproducible results). Default: a Leiden ladder on a
        UMAP kNN graph at ``resolutions``, like R (partitions are not identical to R's).
    sample : optional group per clone; only same-group pairs set the medians.
    local_rescale : rescale distances by local density (R ``local_rescale``).
    local_k : neighbors for the local rescaling (default 7).
    resolutions : Leiden resolutions of the default ladder.
    cluster_k : neighbors of the default ladder's kNN graph (default 15).
    mean_weight : fix the blend weight instead of selecting it.
    s_grid : grid of candidate blend weights.
    seed : leidenalg seed of the default ladder (default 0; unused with ``clusters`` or ``mean_weight``).

    Returns
    -------
    tuple ``(D_revised DataFrame, info dict)`` with ``mean_weight``, ``picks``, ``selection_correlations``.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> Dr, info = ct.revise_distance(D, Xq, Q, mean_weight=0.5)
    >>> Dr.shape, sorted(info)
    ((21, 21), ['mean_weight', 'picks', 'selection_correlations'])
    """
    seed = check_seed(seed)
    Dm, names = _square(D)
    X = np.ascontiguousarray(embedding, dtype=np.float64)
    Qc, qnames, _ = _as_Q(Q)
    if isinstance(D, pd.DataFrame) and isinstance(Q, pd.DataFrame) and list(D.columns) != list(Q.columns):
        raise ValueError("D and Q clone order differ")
    W = sp.csc_matrix(Qc)
    mass = np.asarray(W.sum(0)).ravel()
    if (mass <= 0).any():
        raise ValueError("clones without mass")
    cent = np.asarray((W.T @ X)) / mass[:, None]
    r2 = (cent ** 2).sum(1)
    cdist = np.sqrt(np.maximum(r2[:, None] + r2[None, :] - 2 * cent @ cent.T, 0))
    n = Dm.shape[0]
    pi, pj = np.triu_indices(n, 1)
    order = np.lexsort((pi, pj))  # R which(upper.tri, arr.ind=TRUE): column-major
    pi, pj = pi[order], pj[order]
    if sample is not None:
        sm = np.asarray(sample)
        ok = sm[pi] == sm[pj]
        pi, pj = pi[ok], pj[ok]
    if pi.size == 0:
        raise ValueError("no clone pairs available")
    if local_rescale:
        ls = np.array([np.sort(np.delete(Dm[i], i))[local_k - 1] for i in range(n)])
        ls = np.maximum(ls, np.finfo(float).eps)
        L = Dm / np.sqrt(np.outer(ls, ls))
    else:
        L = Dm
    Ln = L / np.median(L[pi, pj])
    Cn = cdist / np.median(cdist[pi, pj])
    picks, curves = None, None
    if mean_weight is None:
        grid = np.unique(np.asarray(s_grid if s_grid is not None else np.round(np.arange(21) * 0.05, 10), float))
        if clusters is None:
            clusters = _leiden_ladder(X, resolutions, cluster_k, seed)
        cl = pd.DataFrame(clusters)
        if len(cl) != X.shape[0]:
            raise ValueError("clusters must have one row per cell")
        lp, cp = Ln[pi, pj], Cn[pi, pj]
        picks, curves = [], np.full((grid.size, cl.shape[1]), np.nan)
        integer_w = bool(np.all(np.abs(W.data - np.round(W.data)) < 1e-12)) if W.nnz else True
        for c in range(cl.shape[1]):
            lab = cl.iloc[:, c]
            obs = lab.notna().to_numpy()
            if lab[obs].nunique() < 2:
                picks.append(0.0)
                continue
            codes = pd.factorize(lab[obs].astype(str), sort=True)[0]
            ind = sp.csr_matrix((np.ones(obs.sum()), (np.flatnonzero(obs), codes)), shape=(X.shape[0], codes.max() + 1))
            comp = np.asarray((W.T @ ind).todense())
            keep = comp.sum(1) > 0
            el = keep[pi] & keep[pj]
            if el.sum() < 3:
                picks.append(0.0)
                continue
            tv = _composition_tv(comp, pi[el], pj[el], integer_w)
            tr = sps.rankdata(tv)
            curve = np.array([_pearson(sps.rankdata((1 - s) * lp[el] + s * cp[el]), tr) for s in grid])
            curves[:, c] = curve
            fin = np.flatnonzero(np.isfinite(curve))
            picks.append(float(grid[fin[np.argmax(curve[fin])]]) if fin.size else 0.0)
        mean_weight = float(np.median(picks))
    R = (1 - mean_weight) * Ln + mean_weight * Cn
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0)
    info = {"mean_weight": mean_weight, "picks": picks, "selection_correlations": curves}
    return pd.DataFrame(R, index=names, columns=names), info


def _pearson(a, b) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _composition_tv(comp, a, b, integer):
    if integer:
        comp = np.round(comp)
    m = comp.sum(1)
    res = np.abs(comp[a] * m[b, None] - comp[b] * m[a, None]).sum(1) / (2 * m[a] * m[b])
    return res if integer else np.round(res, 12)


def _leiden_ladder(X, resolutions, k, seed):
    from .cluster import cluster
    from .scanpy_extra import neighbors
    nb_ = neighbors(X, n_neighbors=int(k), random_state=seed)
    G = nb_["connectivities"] if isinstance(nb_, dict) else nb_[1]
    return pd.DataFrame({f"leiden_r{r}": cluster(G, method="leiden", resolution=r, seed=seed,
                                                   n_iterations=-1, group_singletons=False, engine="leidenalg")
                         for r in resolutions})


# =========================================================================== 4. clone clusters / embedding / dpt
@records_seed("seed")
def cluster_clones(D, k: int = 10, resolution: float = 1.0, *, prune: float = 0.0, weight: str = "jaccard",
                   seed: int = 0, n_iterations: int = 2) -> pd.Series:
    """Leiden communities of clones from their distance matrix (R ``leiden_dis``, ``if_umap=FALSE``).

    Each clone gets its ``k`` nearest clones (ties go to the lower index). Edge weight = Jaccard of the two kNN
    sets, ``shared / (2k - shared)``, or the distance with ``weight="dis"``. The graph is undirected and keeps
    both directions as parallel edges, as igraph does in R. Leiden optimizes modularity (igraph
    ``community_leiden``, beta 0.01). The RNG differs from R's, so expect a high ARI, not identical labels.
    A private ``random.Random(seed)`` is installed as igraph's RNG for the call and igraph's default restored
    afterwards (Python's global ``random`` state is not touched).

    Parameters
    ----------
    D : clone x clone distance DataFrame.
    k : nearest clones per clone (default 10).
    resolution : Leiden resolution (default 1.0).
    prune : drop edges with weight below this (default 0).
    weight : ``"jaccard"`` (default) or ``"dis"``.
    seed : integer seed (default 0).
    n_iterations : Leiden iterations (default 2).

    Returns
    -------
    pandas.Series of 1-based cluster labels indexed by clone (the clone "fate profiles").

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> prof = ct.cluster_clones(D, k=5)
    >>> int(prof.min())
    1
    """
    import random

    import igraph as ig
    seed = check_seed(seed)
    Dm, names = _square(D)
    n = Dm.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be in [1, n_clones)")
    nbr = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        cand = np.delete(np.arange(n), i)
        nbr[i] = cand[np.argsort(Dm[i, cand], kind="stable")[:k]]
    sets = [set(r) for r in nbr]
    shared = np.array([[len(sets[i] & sets[nbr[i, j]]) for j in range(k)] for i in range(n)], dtype=float)
    jac = shared / (2 * k - shared)
    s_, e_ = np.repeat(np.arange(n), k), nbr.ravel()
    wv = jac.ravel() if weight == "jaccard" else Dm[s_, e_]
    keep = jac.ravel() > prune
    g = ig.Graph(n=n, edges=list(zip(s_[keep].tolist(), e_[keep].tolist())), directed=False)
    ig.set_random_number_generator(random.Random(seed))
    try:
        part = g.community_leiden(objective_function="modularity", weights=wv[keep].tolist(), resolution=resolution,
                                  beta=0.01, n_iterations=n_iterations)
    finally:
        ig.set_random_number_generator(random)  # igraph's default
    mem = np.asarray(part.membership) + 1
    return pd.Series(mem, index=names, name="cluster")


def cmdscale(D, k: int = 2) -> pd.DataFrame:
    """Classical (Torgerson) MDS, as R ``cmdscale`` (column signs may differ).

    Parameters
    ----------
    D : square distance DataFrame (or array).
    k : number of dimensions (default 2).

    Returns
    -------
    DataFrame, clones x k, columns ``mds_1..mds_k``.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> list(ct.cmdscale(D, 2).columns)
    ['mds_1', 'mds_2']
    """
    Dm, names = _square(D)
    n = Dm.shape[0]
    x = Dm ** 2
    J = np.eye(n) - 1.0 / n
    Bm = -0.5 * J @ x @ J
    ev, V = np.linalg.eigh(Bm)
    o = np.argsort(ev)[::-1][:k]
    ev, V = ev[o], V[:, o]
    if (ev <= 0).any():
        keep = ev > 0
        ev, V = ev[keep], V[:, keep]
    pts = V * np.sqrt(ev)[None, :]
    return pd.DataFrame(pts, index=names, columns=[f"mds_{i + 1}" for i in range(pts.shape[1])])


@nb.njit(cache=True)
def _pava(y: np.ndarray) -> np.ndarray:
    """Isotonic (non-decreasing) least-squares fit, pool-adjacent-violators."""
    n = y.size
    val = np.empty(n)
    wt = np.empty(n)
    cnt = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        val[m] = y[i]
        wt[m] = 1.0
        cnt[m] = 1
        while m > 0 and val[m - 1] > val[m]:
            tw = wt[m - 1] + wt[m]
            val[m - 1] = (val[m - 1] * wt[m - 1] + val[m] * wt[m]) / tw
            wt[m - 1] = tw
            cnt[m - 1] += cnt[m]
            m -= 1
        m += 1
    out = np.empty(n)
    p = 0
    for b in range(m):
        for _ in range(cnt[b]):
            out[p] = val[b]
            p += 1
    return out


def isomds(D, k: int = 2, *, maxit: int = 200, tol: float = 1e-6, init=None) -> tuple[pd.DataFrame, float]:
    """Kruskal non-metric MDS initialized at :func:`cmdscale` (like ``MASS::isoMDS``).

    Minimizes Kruskal stress-1 with L-BFGS, taking the isotonic regression of configuration distances on the
    dissimilarity order. The optimizer is not the one ``MASS`` uses, so the configurations differ slightly from
    R; the stress (in percent, as R reports it) is close.

    Parameters
    ----------
    D : square distance DataFrame (or array).
    k : number of dimensions (default 2).
    maxit : maximum optimizer iterations (default 200).
    tol : convergence tolerance (default 1e-6).
    init : starting configuration (default :func:`cmdscale`).

    Returns
    -------
    tuple ``(points DataFrame, stress in percent)``.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> pts, stress = ct.isomds(D, 2)
    >>> pts.shape
    (21, 2)
    """
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    Dm, names = _square(D)
    n = Dm.shape[0]
    iu = np.triu_indices(n, 1)
    dis = Dm[iu]
    order = np.argsort(dis, kind="stable")
    X0 = cmdscale(Dm, k).to_numpy() if init is None else np.asarray(init, float)

    def f(xf):
        X = xf.reshape(n, -1)
        d = pdist(X)
        ds = d[order]
        fit = np.empty_like(d)
        fit[order] = _pava(ds)
        num = ((d - fit) ** 2).sum()
        den = (d ** 2).sum()
        s = math.sqrt(num / den)
        # gradient of sqrt(num/den) wrt X (fit treated as constant: standard Kruskal step)
        with np.errstate(invalid="ignore", divide="ignore"):
            g_d = ((d - fit) / den - num * d / den ** 2) / max(s, 1e-300)
            g_d = np.where(d > 0, g_d / d, 0.0)
        # G_i = sum_j w_ij (x_i - x_j) with W the symmetric matrix of g_d: a dense product, not np.add.at
        W = squareform(g_d)
        G = W.sum(1)[:, None] * X - W @ X
        return s, G.ravel()

    res = minimize(f, X0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxit, "gtol": tol})
    X = res.x.reshape(n, -1)
    return pd.DataFrame(X, index=names, columns=[f"mds_{i + 1}" for i in range(X.shape[1])]), float(res.fun * 100)


def clone_root(clone_names: Sequence[str], cell_clones, cell_clusters, start_cluster) -> str:
    """Root clone for pseudotime: highest fraction of its cells in ``start_cluster``, ties broken by size.

    Port of R ``clone_root``.

    Parameters
    ----------
    clone_names : candidate clones (e.g. ``D.index``).
    cell_clones : clone label per cell.
    cell_clusters : cell cluster / type per cell.
    start_cluster : the cluster where trajectories start (e.g. naive cells).

    Returns
    -------
    str, the root clone name.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> root = ct.clone_root(list(D.index), clones, t.obs["cell_type"].astype(str).to_numpy(), "CD8 T")
    >>> root in D.index
    True
    """
    df = pd.DataFrame({"clone": pd.Series(list(cell_clones), dtype=object), "cl": pd.Series(list(cell_clusters))})
    df = df[df.clone.isin(set(map(str, clone_names)))]
    df["clone"] = df.clone.astype(str)
    g = df.groupby("clone", sort=True)
    tab = pd.DataFrame({"ratio": g.cl.apply(lambda s: (s.astype(str) == str(start_cluster)).sum() / len(s)),
                        "count": g.size()})
    tab = tab.reset_index().sort_values(["ratio", "count"], ascending=[False, False], kind="stable")
    return str(tab.clone.iloc[0])


def clone_dpt(clone_embedding, root, *, k: int = 10, dpt_k: int = 30) -> pd.Series:
    """Diffusion pseudotime of clones (R ``clone_dpt`` / ``dpt``) on a kNN graph of the clone embedding.

    ``T`` = row-normalized :func:`knn_connectivity` ``(clone_embedding, k)``. The top ``dpt_k`` eigenpairs by
    magnitude are taken, and those with ``lambda >= 0.9999`` dropped. The kept eigenvectors are scaled by
    ``lambda / (1 - lambda)``; pseudotime is the Euclidean distance to the root, divided by its maximum.

    Parameters
    ----------
    clone_embedding : clones x dims DataFrame (e.g. :func:`cmdscale` / :func:`isomds`).
    root : clone name or row position of the root.
    k : neighbors of the clone kNN graph (default 10).
    dpt_k : number of eigenpairs (default 30; capped by the number of clones).

    Returns
    -------
    pandas.Series of pseudotime in [0, 1] indexed by clone (0 at the root).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> E = ct.cmdscale(D, 4)
    >>> tt = ct.clone_dpt(E, E.index[0], k=5, dpt_k=4)
    >>> float(tt.iloc[0])
    0.0
    """
    E = clone_embedding
    names = [str(x) for x in E.index] if isinstance(E, pd.DataFrame) else None
    X = np.ascontiguousarray(np.asarray(E), dtype=np.float64)
    n = X.shape[0]
    K = knn_connectivity(X, k).toarray()
    d = K.sum(1)
    Ssym = K / np.sqrt(np.outer(d, d))
    ev, V = np.linalg.eigh((Ssym + Ssym.T) / 2)
    o = np.argsort(-np.abs(ev), kind="stable")[: min(dpt_k, n)]
    ev, V = ev[o], V[:, o]
    R = V / np.sqrt(d)[:, None]
    R /= np.linalg.norm(R, axis=0)[None, :]
    keep = ev < 0.9999
    ev, R = ev[keep], R[:, keep]
    R = R * (ev / (1 - ev))[None, :]
    r = names.index(str(root)) if (names is not None and not isinstance(root, (int, np.integer))) else int(root)
    t = np.sqrt(((R - R[r]) ** 2).sum(1))
    t = t / t.max()
    return pd.Series(t, index=names, name="dpt")


# =========================================================================== 5. profiles, enrichment, genes
def cell_profiles(Q: pd.DataFrame, clone_cluster: pd.Series, min_prob: float = 0.5):
    """Cell x profile probabilities ``Q P`` (P = one-hot clone -> profile) and a hard label per cell.

    Parameters
    ----------
    Q : cells x clones probabilities.
    clone_cluster : profile per clone (Series indexed by clone name, e.g. :func:`cluster_clones`).
    min_prob : the hard label is the argmax profile when its probability is ``> min_prob``, else NA
        (default 0.5).

    Returns
    -------
    tuple ``(prob DataFrame cells x profiles, label Series)``.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
    >>> prob.shape[0] == Q.shape[0]
    True
    """
    cc = clone_cluster.reindex([str(c) for c in Q.columns])
    if cc.isna().any():
        miss = [str(c) for c, v in zip(Q.columns, cc.isna()) if v][:10]
        raise KeyError(f"clone_cluster misses clones of Q, e.g. {miss}; pass a Series indexed by clone name "
                       "(e.g. cluster_clones(D)) covering every column of Q")
    prof = pd.Index(sorted(cc.unique(), key=lambda x: (str(type(x)), x)))
    Pm = sp.csr_matrix((np.ones(len(cc)), (np.arange(len(cc)), prof.get_indexer(cc))), shape=(len(cc), len(prof)))
    prob = np.asarray(sp.csr_matrix(Q.to_numpy()) @ Pm.toarray())
    df = pd.DataFrame(prob, index=Q.index, columns=prof)
    mx = prob.max(1)
    lab = pd.Series(np.where(mx > min_prob, prof.to_numpy()[prob.argmax(1)], None), index=Q.index, name="profile")
    return df, lab


def _check_cell_labels(lab, n: int, fn: str) -> None:
    lab = pd.Series(np.asarray(lab, dtype=object))
    if len(lab) != n:
        raise ValueError(f"{fn}: {len(lab)} cluster labels for {n} cells; pass one label per row of "
                         "cell_profile_prob (same order, e.g. obs['cluster'].reindex(prob.index))")
    if lab.isna().any():
        raise ValueError(f"{fn}: {int(lab.isna().sum())} cluster labels are missing; if you reindexed by cell "
                         "names, check that cell_profile_prob.index holds the same names (pass cells= upstream)")


def profile_mass(cell_profile_prob: pd.DataFrame, cluster_label) -> pd.DataFrame:
    """Summed profile probability per cell cluster (R ``cluster_profile_mass``; rows sorted).

    Parameters
    ----------
    cell_profile_prob : cells x profiles DataFrame (from :func:`cell_profiles`).
    cluster_label : cell cluster per row of ``cell_profile_prob`` (same order; no missing values).

    Returns
    -------
    DataFrame, clusters x profiles.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
    >>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
    >>> sorted(ct.profile_mass(prob, cell_type).index)
    ['CD4 T', 'CD8 T']
    """
    df = pd.DataFrame(np.asarray(cell_profile_prob), columns=cell_profile_prob.columns)
    _check_cell_labels(cluster_label, len(df), "profile_mass")
    df["__cl"] = np.asarray(cluster_label)
    return df.groupby("__cl", sort=True).sum()


@records_seed("seed")
def profile_enrichment(cell_profile_prob: pd.DataFrame, cluster_label, n_perm: int = 300, seed: int = 0):
    """Cluster x profile mass and a one-sided permutation p-value (R ``cluster_profile_enrich``).

    The cluster labels are permuted ``n_perm`` times; ``p = mean(perm >= observed)``, with no ``+1``, as in R.

    Parameters
    ----------
    cell_profile_prob : cells x profiles DataFrame.
    cluster_label : cell cluster per row (same order; no missing values).
    n_perm : number of permutations (default 300).
    seed : integer seed of a local ``np.random.default_rng`` (default 0).

    Returns
    -------
    tuple ``(mass, pval)`` DataFrames, clusters x profiles.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
    >>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
    >>> mass, pval = ct.profile_enrichment(prob, cell_type, n_perm=50)
    >>> mass.shape == pval.shape
    True
    """
    seed = check_seed(seed)
    lab = np.asarray(cluster_label)
    _check_cell_labels(lab, len(cell_profile_prob), "profile_enrichment")
    truth = profile_mass(cell_profile_prob, lab)
    codes, uniq = pd.factorize(pd.Series(lab), sort=True)
    X = np.asarray(cell_profile_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ge = np.zeros(truth.shape)
    for _ in range(n_perm):
        pc = rng.permutation(codes)
        m = np.zeros(truth.shape)
        np.add.at(m, pc, X)
        ge += m >= truth.to_numpy()
    return truth, pd.DataFrame(ge / n_perm, index=truth.index, columns=truth.columns)


def ns_basis(x, df: int) -> np.ndarray:
    """Natural cubic spline basis without intercept (R ``splines::ns(x, df)``; columns up to sign/rotation).

    The ridge fits in :func:`soft_cluster_fit` are invariant to the orthogonal ambiguity of the constraint
    projection.

    Parameters
    ----------
    x : 1-D array (e.g. pseudotime).
    df : degrees of freedom (number of columns).

    Returns
    -------
    numpy array, len(x) x df.

    Examples
    --------
    >>> import numpy as np
    >>> from arsc import clonotrace as ct
    >>> ct.ns_basis(np.linspace(0, 1, 50), 3).shape
    (50, 3)
    """
    from scipy.interpolate import BSpline
    x = np.asarray(x, dtype=np.float64)
    nik = df - 1
    lo, hi = float(x.min()), float(x.max())
    knots = np.quantile(x, np.linspace(0, 1, nik + 2)[1:-1]) if nik > 0 else np.zeros(0)
    t = np.r_[[lo] * 4, np.sort(knots), [hi] * 4]
    nb_ = t.size - 4

    def design(v, der=0):
        out = np.empty((v.size, nb_))
        for j in range(nb_):
            c = np.zeros(nb_)
            c[j] = 1.0
            s = BSpline(t, c, 3, extrapolate=True)
            out[:, j] = (s.derivative(der) if der else s)(v)
        return out

    basis = design(x)[:, 1:]
    const = design(np.array([lo, hi]), 2)[:, 1:]
    Qc, _ = np.linalg.qr(const.T, mode="complete")
    return (Qc.T @ basis.T)[2:].T


def _ridge(X, G, lam):
    d = X.shape[1]
    beta = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ G)
    rss = ((G - X @ beta) ** 2).sum(0)
    return beta, rss


def soft_cluster_fit(G, t, P, *, df: int = 5, lam: float = 1e-4, test: str = "F") -> pd.DataFrame:
    """Test whether expression along pseudotime differs between soft groups (R ``soft_cluster_gam_fit``).

    The null is a shared ``[1, ns(t, df)]`` curve; the full model has one curve per group (design
    ``B * P[:, k]``). Both are ridge fits (``lam``). ``test="F"`` gives
    ``F = ((RSS0 - RSS1) / df1) / (RSS1 / df2)``; ``"LRT"`` gives the Gaussian likelihood ratio. Cells are the
    observations here: use it to rank genes, and confirm across patients before claiming an effect.

    Parameters
    ----------
    G : cells x genes expression (dense or sparse).
    t : pseudotime per cell.
    P : cells x m soft assignment (two columns, target and control, for Cohen's d).
    df : spline degrees of freedom (default 5).
    lam : ridge penalty (default 1e-4).
    test : ``"F"`` (default) or ``"LRT"``.

    Returns
    -------
    DataFrame per gene: ``stat``, ``pval``, ``mean_diff``, ``cohen``.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
    >>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
    >>> pos = t.obs.index.get_indexer(Q.index)
    >>> w = prob.iloc[:, 0].to_numpy()
    >>> fit = ct.soft_cluster_fit(L[pos][:, :10], np.linspace(0, 1, len(pos)), np.c_[w, 1 - w], df=3)
    >>> list(fit.columns)
    ['stat', 'pval', 'mean_diff', 'cohen']
    """
    Gd = np.asarray(G.toarray() if sp.issparse(G) else G, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    n, m = P.shape
    Bm = np.column_stack([np.ones(n), ns_basis(t, df)])
    d = Bm.shape[1]
    Xf = np.column_stack([Bm * P[:, [k]] for k in range(m)])
    _, rss0 = _ridge(Bm, Gd, lam)
    bf, rss1 = _ridge(Xf, Gd, lam)
    df1, df2 = m * d - d, n - m * d
    if test == "F":
        stat = ((rss0 - rss1) / df1) / (rss1 / df2)
        pv = sps.f.sf(stat, df1, df2)
    elif test == "LRT":
        stat = 2 * (-0.5 * n * np.log(rss1) + 0.5 * n * np.log(rss0))
        pv = sps.chi2.sf(stat, df1)
    else:
        raise ValueError("test must be 'F' or 'LRT'")
    tp = Bm @ bf[:d]
    cp = Bm @ bf[d:2 * d]
    sigma = np.sqrt(rss0 / n)
    md = np.nanmean(tp - cp, axis=0)
    return pd.DataFrame({"stat": stat, "pval": pv, "mean_diff": md, "cohen": md / sigma})


def bin_filter(mass, time, thresh: float = 5, binsize: float = 0.005) -> np.ndarray:
    """Cells in pseudotime bins where both target and control mass exceed ``thresh`` (R ``bin_filter_profile_mass``).

    Mirrors R's ``cut(time, seq(min, max, by))``: right-closed bins. Times at the minimum, or above the last
    break, fall into an NA bin, which is kept only if that NA bin itself passes.

    Parameters
    ----------
    mass : cells x 2 array (target mass, control mass).
    time : pseudotime per cell.
    thresh : minimum summed mass of each column per bin (default 5).
    binsize : bin width in pseudotime units (default 0.005).

    Returns
    -------
    boolean numpy array, one per cell (True = kept).

    Examples
    --------
    >>> import numpy as np
    >>> from arsc import clonotrace as ct
    >>> keep = ct.bin_filter(np.ones((100, 2)), np.linspace(0, 1, 100), thresh=2, binsize=0.1)
    >>> keep.dtype.name, int(keep.sum())
    ('bool', 99)
    """
    M = np.asarray(mass, dtype=np.float64)
    tt = np.asarray(time, dtype=np.float64)
    lo, hi = tt.min(), tt.max()
    nb_ = int(math.floor((hi - lo) / binsize + 1e-10))
    br = lo + np.arange(nb_ + 1) * binsize
    code = np.searchsorted(br, tt, side="left") - 1  # (br[k], br[k+1]] -> k
    code[(tt <= br[0]) | (tt > br[-1])] = -1
    if br.size < 2:
        code[:] = -1
    keys = np.unique(code)
    ok = set()
    for kk in keys:
        sel = code == kk
        if M[sel, 0].sum() > thresh and M[sel, 1].sum() > thresh:
            ok.add(kk)
    return np.isin(code, list(ok))


@records_seed("seed")
def fate_genes(profile, cluster, X, genes: Sequence[str], cell_cluster, cell_t, cell_profile_prob: pd.DataFrame,
               *, n_perm: int = 50, seed: int = 0, min_cells: int = 30, mass_thresh: float = 3,
               binsize: float = 0.005) -> pd.DataFrame | None:
    """Fate-associated genes of ``profile`` inside cell ``cluster`` (R ``profile_cluster_DEG``).

    Target mass = summed probability of ``profile`` (one or several); control = the rest. Steps: bin filter
    (:func:`bin_filter`); genes with ``> min_cells`` non-zero cells; :func:`soft_cluster_fit` with ``df=1``;
    calibrate each p-value against all genes' p-values under ``n_perm`` label permutations (rows shuffled,
    then shuffled within each row); then BH.

    Parameters
    ----------
    profile : profile label (or list of labels) of interest.
    cluster : cell cluster in which genes are tested.
    X : cells x genes expression aligned to ``cell_profile_prob`` rows.
    genes : gene names (columns of ``X``).
    cell_cluster : cluster per cell (same order).
    cell_t : pseudotime per cell (same order).
    cell_profile_prob : cells x profiles DataFrame (:func:`cell_profiles`).
    n_perm : number of permutations (default 50).
    seed : integer seed of a local ``np.random.default_rng`` (default 0).
    min_cells : minimum non-zero cells per gene (default 30).
    mass_thresh : ``thresh`` of :func:`bin_filter` (default 3).
    binsize : ``binsize`` of :func:`bin_filter` (default 0.005).

    Returns
    -------
    DataFrame per gene (``stat, pval, mean_diff, cohen, pcali, padj``), or None if fewer than 5 cells pass, as
    R does.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
    >>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
    >>> pos = t.obs.index.get_indexer(Q.index)
    >>> fg = ct.fate_genes(prob.columns[0], "CD8 T", L[pos], list(t.features("RNA")), cell_type,
    ...                    np.linspace(0, 1, len(pos)), prob, n_perm=5, mass_thresh=1, binsize=0.1)
    >>> list(fg.columns)
    ['stat', 'pval', 'mean_diff', 'cohen', 'pcali', 'padj']
    """
    seed = check_seed(seed)
    from .clones import p_adjust_bh
    profs = [profile] if not isinstance(profile, (list, tuple)) else list(profile)
    sel = np.asarray(cell_cluster).astype(str) == str(cluster)
    Pf = cell_profile_prob.to_numpy()[sel]
    tgt = cell_profile_prob.loc[:, profs].to_numpy()[sel].sum(1)
    P2 = np.column_stack([tgt, Pf.sum(1) - tgt])
    tt = np.asarray(cell_t, dtype=np.float64)[sel]
    flag = bin_filter(P2, tt, thresh=mass_thresh, binsize=binsize)
    if flag.sum() < 5:
        return None
    P2, tt = P2[flag], tt[flag]
    rows = np.flatnonzero(sel)[flag]
    Xs = X[rows]
    Xs = Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs, dtype=np.float64)
    gk = (Xs != 0).sum(0) > min_cells
    Gm = Xs[:, gk]
    gn = np.asarray(genes)[gk]
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        Pp = P2[rng.permutation(P2.shape[0])]
        Pp = np.array([r[rng.permutation(r.size)] for r in Pp])
        nulls.append(soft_cluster_fit(Gm, tt, Pp, df=1).pval.to_numpy())
    pnull = np.sort(np.concatenate(nulls))
    res = soft_cluster_fit(Gm, tt, P2, df=1)
    res.index = gn
    res["pcali"] = np.searchsorted(pnull, res.pval.to_numpy(), side="right") / pnull.size
    res["padj"] = p_adjust_bh(res.pcali.to_numpy())
    return res


# =========================================================================== co-embedding
def coembed_graph(cell_embedding, clone_embedding, Q, *, cell_k: int = 30, clone_k: int = 15,
                  min_weight: float = 0.1, n_keep: int = 20) -> sp.csr_matrix:
    """Clone-weighted cell kNN distances (R ``cell_clone_coembed``), for UMAP with a precomputed metric.

    For each cell kNN pair (self included), ``w_ij = Q_i V Q_j^T``, where V is the binary clone kNN graph
    (``clone_k``, self included). Pairs with ``w > min_weight`` get distance ``d_ij / w_ij``. Each cell keeps
    its ``n_keep`` smallest (ties kept), then the graph is symmetrized and the diagonal dropped.

    Parameters
    ----------
    cell_embedding : cells x dims, rows aligned to ``Q``.
    clone_embedding : clones x dims, rows aligned to ``Q``'s columns.
    Q : cells x clones probabilities.
    cell_k : cell neighbors (default 30).
    clone_k : clone neighbors (default 15).
    min_weight : minimum clone weight of a kept pair (default 0.1).
    n_keep : distances kept per cell (default 20).

    Returns
    -------
    scipy.sparse.csr_matrix of distances, cells x cells.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
    >>> Q = ct.filter_and_sparsify(cp)
    >>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
    >>> D = ct.clone_distance(Xq, Q)
    >>> G = ct.coembed_graph(Xq, ct.cmdscale(D, 2), Q, cell_k=10, clone_k=3)
    >>> G.shape == (len(Q), len(Q))
    True
    """
    Xc = np.ascontiguousarray(cell_embedding, dtype=np.float64)
    Xk = np.ascontiguousarray(clone_embedding, dtype=np.float64)
    Qm = sp.csr_matrix(Q.to_numpy() if isinstance(Q, pd.DataFrame) else Q, dtype=np.float64)
    n = Xc.shape[0]
    idx, d = _knn_excl_self(Xc, cell_k)
    n1, n2, dist = _flat_knn(idx, d)
    a, b = np.minimum(n1, n2), np.maximum(n1, n2)
    key = a * n + b
    _, first = np.unique(key, return_index=True)
    keep = np.zeros(key.size, bool)
    keep[first] = True
    n1, n2, dist = n1[keep], n2[keep], dist[keep]
    n1 = np.r_[n1, np.arange(n)]
    n2 = np.r_[n2, np.arange(n)]
    dist = np.r_[dist, np.zeros(n)]
    V = knn_connectivity(Xk, clone_k, self_loops=True)
    V.data[:] = 1.0
    QV = (Qm @ V).tocsr()
    wgt = np.asarray(QV[n1].multiply(Qm[n2]).sum(1)).ravel()
    s1, s2, sd, sw = _symmetrize_long(n1, n2, dist, wgt)
    ok = sw > min_weight
    s1, s2, dis = s1[ok], s2[ok], sd[ok] / sw[ok]
    # slice_min(dis, n = n_keep) per node1, ties kept
    o = np.lexsort((dis, s1))
    s1, s2, dis = s1[o], s2[o], dis[o]
    starts = np.r_[0, np.flatnonzero(np.diff(s1)) + 1]
    sel = np.zeros(s1.size, bool)
    for st, en in zip(starts, np.r_[starts[1:], s1.size]):
        if en - st <= n_keep:
            sel[st:en] = True
        else:
            cut = dis[st + n_keep - 1]
            sel[st:en] = dis[st:en] <= cut
    f1, f2, fd = _symmetrize_long(s1[sel], s2[sel], dis[sel])
    M = sp.csr_matrix(sp.coo_matrix((fd, (f1, f2)), shape=(n, n)))
    M.setdiag(0)
    M.eliminate_zeros()
    return M


# =========================================================================== pipeline + tables
@dataclass
class CloneTrace:
    """Result of :func:`run`.

    Attributes
    ----------
    Q : cells x clones propagated probabilities (after the deviance filter and sparsification).
    deviance : per-cell bootstrap deviance.
    distance : clone x clone distance used downstream (revised when ``revised=True``).
    raw_distance : the unrevised :func:`clone_distance`.
    clusters : fate profile per clone (:func:`cluster_clones`).
    embedding : clone MDS embedding.
    clone_t : pseudotime per clone (:func:`clone_dpt`).
    cell_t : pseudotime per cell (Q-weighted clone pseudotime).
    cell_profile_prob : cells x profiles probabilities; cell_profile : hard profile per cell.
    params : the arguments used.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
    ...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
    >>> res.Q.shape[1] == res.distance.shape[0]
    True
    """
    Q: pd.DataFrame                      # cells x clones (filtered, sparsified, rows sum to 1)
    deviance: pd.Series                  # per input cell
    distance: pd.DataFrame               # clone x clone (revised if requested)
    raw_distance: pd.DataFrame           # clone x clone before revision
    clusters: pd.Series                  # clone -> profile (1-based)
    embedding: pd.DataFrame              # clone MDS embedding
    clone_t: pd.Series                   # clone pseudotime
    cell_t: pd.Series                    # cell pseudotime (Q @ clone_t)
    cell_profile_prob: pd.DataFrame
    cell_profile: pd.Series
    params: dict = field(default_factory=dict)

    def table(self, cyclone=None, clone_col: str | None = None) -> pd.DataFrame:
        """One row per clone; see :func:`clone_table`.

        Parameters
        ----------
        cyclone : optional :class:`arsc.clones.Cyclone` result to join (trajectory ``k``, patient).
        clone_col : Cyclone clone column (default: the Cyclone ``clone_col``).

        Returns
        -------
        DataFrame, one row per clone.

        Examples
        --------
        >>> import arsc, numpy as np
        >>> from arsc import clonotrace as ct
        >>> ds = arsc.datasets.toy()
        >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
        >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
        >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
        >>> clones = t.obs["clone_id"].to_numpy()
        >>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
        ...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
        >>> list(res.table().columns)
        ['clone', 'mass', 'n_cells', 'profile', 'clone_t']
        """
        return clone_table(self, cyclone, clone_col=clone_col)

    def __repr__(self) -> str:
        return (f"<CloneTrace> {self.Q.shape[1]} clones, {self.Q.shape[0]} cells, "
                f"{self.clusters.nunique()} profiles")


@records_seed("seed")
def run(embedding, clones, *, cells=None, cell_clusters=None, start_cluster=None, k: int = 30,
        min_size: int = 10, alpha: float = 0.6, n_boot: int = 48, max_deviance: float = 0.3,
        distance: str = "nn", graph_k: int = 10, revised: bool = False, revise_clusters=None,
        n_mds: int = 30, cluster_k: int = 20, resolution: float = 0.5, dpt_k: int = 30, dpt_nn: int = 10,
        seed: int = 0) -> CloneTrace:
    """Run CloneTrace steps 1-4 with the published vignette defaults.

    ``revised=False`` reproduces the published figures; ``True`` is the package's 1.2.x default. ``seed`` is the
    single seed of the whole pipeline: it is passed unchanged to every stochastic step (bootstrap of
    :func:`propagate_clones`, the Leiden ladder of :func:`revise_distance`, :func:`cluster_clones`); the other
    steps are deterministic.

    Parameters
    ----------
    embedding : cells x dims (e.g. 15 PCs).
    clones : clone label per cell (NaN = none).
    cells : cell names (default: the DataFrame index or ``0..n-1``).
    cell_clusters : cell cluster per cell; with ``start_cluster`` it picks the pseudotime root (default: the
        first clone).
    start_cluster : the cluster where trajectories start.
    k : neighbors of :func:`cell_transition` (default 30).
    min_size : minimum cells per propagated clone (default 10).
    alpha : propagation weight (default 0.6).
    n_boot : bootstraps for the deviance (default 48).
    max_deviance : deviance filter (default 0.3).
    distance : ``"nn"`` (default) or ``"exact"`` (:func:`clone_distance`).
    graph_k : neighbors of the SNN cost graph (default 10).
    revised : apply :func:`revise_distance` (default False).
    revise_clusters : cells x resolutions labels for :func:`revise_distance`.
    n_mds : dimensions of the clone MDS (default 30; capped by the number of clones).
    cluster_k : neighbors for :func:`cluster_clones` (default 20).
    resolution : Leiden resolution for :func:`cluster_clones` (default 0.5).
    dpt_k : eigenpairs for :func:`clone_dpt` (default 30).
    dpt_nn : neighbors for :func:`clone_dpt` (default 10).
    seed : integer seed of the whole pipeline (default 0).

    Returns
    -------
    :class:`CloneTrace`.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
    ...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
    >>> res.clusters.index.equals(res.distance.index)
    True
    """
    seed = check_seed(seed)
    X = np.ascontiguousarray(embedding, dtype=np.float64)
    if X.ndim != 2 or len(clones) != X.shape[0]:
        raise ValueError(f"clonotrace.run: embedding has shape {X.shape} but there are {len(clones)} clone labels; "
                         "pass a cells x dims embedding and one clone label per row")
    cells = pd.Index(cells if cells is not None else (embedding.index if isinstance(embedding, pd.DataFrame)
                                                       else range(X.shape[0])))
    T = cell_transition(X, k)
    cp = propagate_clones(T, clones, min_size=min_size, alpha=alpha, n_boot=n_boot, seed=seed, cells=cells)
    Q = filter_and_sparsify(cp, max_deviance)
    pos = cells.get_indexer(Q.index)
    Xq = X[pos]
    raw = clone_distance(Xq, Q, method=distance, graph_k=graph_k)
    D = raw
    info = {}
    if revised:
        rc = None if revise_clusters is None else pd.DataFrame(revise_clusters).iloc[pos]
        D, info = revise_distance(raw, Xq, Q, clusters=rc, seed=seed)
    cl = cluster_clones(D, k=min(cluster_k, D.shape[0] - 1), resolution=resolution, seed=seed)
    emb, _ = isomds(D, k=min(n_mds, D.shape[0] - 2))
    if cell_clusters is not None and start_cluster is not None:
        lab = pd.Series(list(clones), index=cells)
        root = clone_root(Q.columns, lab.to_numpy(), np.asarray(cell_clusters), start_cluster)
    else:
        root = Q.columns[0]
    ct = clone_dpt(emb, root, k=min(dpt_nn, emb.shape[0] - 1), dpt_k=dpt_k)
    cell_t = pd.Series(Q.to_numpy() @ ct.reindex(Q.columns).to_numpy(), index=Q.index, name="cell_t")
    cpp, clab = cell_profiles(Q, cl)
    return CloneTrace(Q, pd.Series(cp.deviance, index=cells, name="deviance"), D, raw, cl, emb, ct, cell_t, cpp,
                      clab, {"root": root, "revise": info, "k": k, "alpha": alpha, "n_boot": n_boot,
                             "distance": distance, "seed": seed})


def clone_table(ct: CloneTrace, cyclone=None, *, clone_col: str | None = None) -> pd.DataFrame:
    """One row per CloneTrace clone: mass, cell count, profile, pseudotime (+ Cyclone trajectory).

    Pass a :class:`arsc.clones.Cyclone` result to add its trajectory assignment (``cyclone_k``) and patient for
    clones whose id matches. Typically the CloneTrace clone is the TCR clonotype; when it is clonotype x
    timepoint, map it to the clonotype before joining.

    Parameters
    ----------
    ct : a :class:`CloneTrace` result.
    cyclone : optional :class:`arsc.clones.Cyclone` result.
    clone_col : Cyclone clone column (default: the Cyclone ``clone_col``).

    Returns
    -------
    DataFrame: ``clone, mass, n_cells, profile, clone_t`` (+ ``cyclone_k`` and the patient column).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> from arsc import clonotrace as ct
    >>> ds = arsc.datasets.toy()
    >>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
    >>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
    >>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
    >>> clones = t.obs["clone_id"].to_numpy()
    >>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
    ...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
    >>> cy = arsc.clones.cyclone(ds.obs, patient_col="patient", n_clust=2, n_max_min=3, verbose=False)
    >>> list(ct.clone_table(res, cy).columns)
    ['clone', 'mass', 'n_cells', 'profile', 'clone_t', 'cyclone_k', 'patient']
    """
    Q = ct.Q
    out = pd.DataFrame({"clone": list(Q.columns), "mass": Q.sum(0).to_numpy(), "n_cells": (Q > 0).sum(0).to_numpy(),
                        "profile": ct.clusters.reindex(Q.columns).to_numpy(),
                        "clone_t": ct.clone_t.reindex(Q.columns).to_numpy()})
    if cyclone is not None:
        cc = clone_col or cyclone.params.get("clone_col", "clone_id")
        a = cyclone.assignment.copy()
        a[cc] = a[cc].astype(str)
        keep = [c for c in (cc, "k", cyclone.params.get("patient_col")) if c and c in a.columns]
        out = out.merge(a[keep].rename(columns={cc: "clone", "k": "cyclone_k"}), on="clone", how="left")
    return out
