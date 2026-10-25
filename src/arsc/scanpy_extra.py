"""scanpy functions (beyond the other arsc modules) needed to reproduce scanpy-based analyses.

Conventions follow arsc: plain matrices / DataFrames, cells x features, no AnnData, no plotting.
Each function reproduces the numerics of scanpy 1.11.5 (validated in ``tests/test_scanpy_extra.py``;
see ``SCANPY_PARITY.md`` for the function-by-function map).

====================================  =========================================================
Python                                scanpy
====================================  =========================================================
:func:`score_genes`                   ``sc.tl.score_genes`` (exact: same RNG stream, same control set)
:func:`rank_genes_groups`             ``sc.tl.rank_genes_groups`` (wilcoxon [+tie_correct], t-test,
                                      t-test_overestim_var; reference ``"rest"`` or a group; pts)
:func:`highly_variable_genes`         ``sc.pp.highly_variable_genes`` (seurat, cell_ranger, seurat_v3)
:func:`dendrogram`                    ``sc.tl.dendrogram`` (linkage on group-mean correlation)
:func:`regress_out`                   ``sc.pp.regress_out`` (numeric or one categorical key)
:func:`qc_metrics`                    ``sc.pp.calculate_qc_metrics`` (n_genes_by_counts, total_counts,
                                      pct_counts_in_top_N_genes, per-qc_var totals/pct, log1p_*)
:func:`neighbors`                     ``sc.pp.neighbors`` (umap-style connectivities, exact kNN)
:func:`leiden`                        ``sc.tl.leiden`` (flavor leidenalg: same graph + seed -> same labels)
:func:`umap_from_connectivities`      ``sc.tl.umap`` (min_dist/spread, spectral init, scanpy a/b fit)
:func:`diffmap`, :func:`dpt`          ``sc.tl.diffmap`` / ``sc.tl.dpt`` (the scanpy pseudotime route)
====================================  =========================================================

``sc.pp.normalize_total`` + ``log1p`` = :func:`arsc.pp.log_normalize`; ``sc.pp.scale`` = :func:`arsc.pp.scale`;
``sc.tl.pca`` = :func:`arsc.pp.pca`; ``sc.pp.harmony_integrate`` ~ :func:`arsc.run_harmony`;
``sc.tl.embedding_density``/``paga`` are not implemented.
"""
from __future__ import annotations

import warnings

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as _st

from ._utils import as_csr, check_seed, p_adjust_bh
from ._seedlog import records_seed
from .markers import _rank_stats
from .preprocessing import _col_mean_var, hvg_vst

__all__ = ["normalize_total_log1p", "score_genes", "rank_genes_groups", "highly_variable_genes", "dendrogram", "regress_out",
           "qc_metrics", "neighbors", "leiden", "umap_from_connectivities", "diffmap", "dpt",
           "palantir_diffusion_maps", "palantir_multiscale_space", "magic_impute"]


# --------------------------------------------------------------------------- helpers
def _col_means(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
    return np.nanmean(np.asarray(X), axis=0, dtype=np.float64)


def _mean_var_cols(X):
    """Column mean and unbiased variance (scanpy ``_get_mean_var``)."""
    n = X.shape[0]
    if sp.issparse(X):
        C = as_csr(X)
        mu, var, _ = _col_mean_var(C.indptr, C.indices, C.data.astype(np.float64, copy=False), C.shape[0], C.shape[1])
        return mu, var
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    var = (X * X).mean(0) - mu ** 2
    var = var * (n / (n - 1)) if n > 1 else var
    return mu, var


# --------------------------------------------------------------------------- normalize_total + log1p
@nb.njit(parallel=True, cache=True)
def _scanpy_norm_kernel(indptr, data, out, target):
    for i in nb.prange(indptr.size - 1):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p]
        if s > 0:
            f = s / target  # scanpy: counts = counts_per_cell / after ; X /= counts
            for p in range(indptr[i], indptr[i + 1]):
                out[p] = np.log1p(data[p] / f)


def normalize_total_log1p(counts, target_sum: float = 1e4, dtype=np.float64) -> sp.csr_matrix:
    """``sc.pp.normalize_total(target_sum)`` followed by ``sc.pp.log1p`` (bitwise equal to scanpy).

    Differs from :func:`arsc.pp.log_normalize` (Seurat: ``x / s * sf``) only at the 1-ulp level, but that
    changes tie structure, which matters for rank tests (Wilcoxon): use this one to reproduce scanpy
    ``rank_genes_groups`` p-values exactly. Also equals ``sc.pp.normalize_per_cell(counts_per_cell_after=)``
    + log1p. Cells with zero counts stay zero.

    Parameters
    ----------
    counts : cells x genes raw counts (sparse or dense).
    target_sum : counts per cell after scaling (default 1e4).
    dtype : output dtype (default float64).

    Returns
    -------
    scipy.sparse.csr_matrix of log1p-normalized values, cells x genes.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> L.shape
    (600, 300)
    """
    X = as_csr(counts)
    out = np.zeros(X.data.shape, dtype=dtype)
    _scanpy_norm_kernel(X.indptr, X.data.astype(np.float64, copy=False), out, float(target_sum))
    return sp.csr_matrix((out, X.indices.copy(), X.indptr.copy()), shape=X.shape)


# --------------------------------------------------------------------------- score_genes
@records_seed("random_state")
def score_genes(X, features, gene_list, *, ctrl_size: int = 50, gene_pool=None, n_bins: int = 25,
                random_state: int = 0, ctrl_as_ref: bool = True, return_controls: bool = False):
    """``sc.tl.score_genes``: module score with expression-matched control genes (scanpy-exact).

    ``random_state`` (default 0, as scanpy) seeds the control-gene sampling. scanpy calls
    ``np.random.seed(random_state)`` and then ``Series.sample`` on the global RNG; arsc draws the identical
    stream from a *local* ``np.random.RandomState(random_state)`` (same MT19937 sequence), so the control set,
    and hence the score, equals scanpy's, without touching numpy's global RNG. ``random_state=None`` (scanpy:
    "do not reseed") is rejected.

    Parameters
    ----------
    X : cells x genes, whatever ``adata.X`` / ``.raw.X`` would be (log-normalized or scaled; sparse or dense).
    features : gene names of the columns of ``X``.
    gene_list : genes of the signature (genes absent from ``features`` are ignored, as scanpy).
    ctrl_size : control genes drawn per expression bin (default 50).
    gene_pool : genes to draw controls from (default all).
    n_bins : number of expression bins (default 25).
    random_state : integer seed (default 0).
    ctrl_as_ref : include the signature genes in the control pool (scanpy >= 1.10 default True).
    return_controls : also return the control genes.

    Returns
    -------
    numpy array, one score per cell (plus the control gene Index when ``return_controls``).

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> s = sx.score_genes(L, genes, ["CD8A", "CD8B", "GZMK"])
    >>> s.shape
    (600,)
    """
    rs = np.random.RandomState(check_seed(random_state, "random_state"))
    var_names = pd.Index(features)
    gl = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
    gl = gl.intersection(var_names)
    if len(gl) == 0:
        raise ValueError("No valid genes were passed for scoring: none of "
                         f"{list(pd.Index([gene_list] if isinstance(gene_list, str) else gene_list))[:10]} is in "
                         "`features` (check the gene symbols / species, and that `features` matches X's columns).")
    if gene_pool is None:
        pool = var_names.astype("string")
    else:
        pool = pd.Index(gene_pool, dtype="string").intersection(var_names)
    pos = pd.Series(np.arange(len(var_names)), index=var_names.astype("string"))
    mu = _col_means(X)
    obs_avg = pd.Series(mu[pos.loc[pool].to_numpy()], index=pool)
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    keep_ctrl_in_obs_cut = False if ctrl_as_ref else obs_cut.index.isin(gl)
    control = pd.Index([], dtype="string")
    for cut in np.unique(obs_cut.loc[gl]):
        r_genes = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size, random_state=rs).index
        if ctrl_as_ref:
            r_genes = r_genes.difference(gl)
        control = control.union(r_genes)
    if len(control) == 0:
        raise RuntimeError("No control genes found in any cut.")
    w = np.zeros(len(var_names))
    w[pos.loc[gl].to_numpy()] += 1.0 / len(gl)
    w[pos.loc[control].to_numpy()] -= 1.0 / len(control)
    score = np.asarray(X @ w, dtype=np.float64).ravel()
    return (score, control) if return_controls else score


# --------------------------------------------------------------------------- rank_genes_groups
def _tiecorr_dense(ranks_sorted_counts):  # not used (kept simple)
    raise NotImplementedError


def _wilcox_rest_sparse(C, codes, G):
    """Rank sums of each group among ALL cells for a CSC (sorted, non-negative) matrix.

    Returns ``R`` (G x P) rank sums with the zero block averaged, and ``tc`` (P,) = sum(t^3 - t) over ties.
    """
    N, P = C.shape
    rs, nnz, _sums, ties = _rank_stats(C.indptr.astype(np.int64), C.indices.astype(np.int64),
                                       C.data.astype(np.float64), codes.astype(np.int64), G, N)
    n_g = np.bincount(codes, minlength=G).astype(np.float64)
    nz = (N - np.diff(C.indptr)).astype(np.float64)
    R = rs + (n_g[None, :] - nnz) * ((nz + 1.0) / 2.0)[:, None]
    allzero = np.diff(C.indptr) == 0
    ties = np.where(allzero, float(N) ** 3 - N, ties)  # an all-zero gene is one tie block
    return R.T, ties


def _wilcox_dense(Xd, codes, G, chunk=2000):
    """Generic path (negative values / dense): scipy rankdata per gene chunk."""
    N, P = Xd.shape
    R = np.zeros((G, P))
    ties = np.zeros(P)
    onehot = np.zeros((N, G))
    ok = codes >= 0
    onehot[np.flatnonzero(ok), codes[ok]] = 1.0
    for a in range(0, P, chunk):
        blk = np.asarray(Xd[:, a:a + chunk], dtype=np.float64)
        rk = _st.rankdata(blk, axis=0)
        R[:, a:a + blk.shape[1]] = onehot.T @ rk
        for j in range(blk.shape[1]):
            _, t = np.unique(blk[:, j], return_counts=True)
            ties[a + j] = float((t.astype(np.float64) ** 3 - t).sum())
    return R, ties


def _wilcox_scores_rest(X, codes, G, tie_correct):
    N = X.shape[0]
    if sp.issparse(X):
        C = as_csr(X)
        nonneg = C.nnz == 0 or C.data.min() >= 0
        if nonneg:
            Cc = C.tocsc()
            Cc.sort_indices()
            R, ties = _wilcox_rest_sparse(Cc, codes, G)
        else:
            R, ties = _wilcox_dense(C.toarray(), codes, G)
    else:
        Xd = np.asarray(X)
        if Xd.min() >= 0:
            Cc = sp.csc_matrix(Xd)
            Cc.sort_indices()
            R, ties = _wilcox_rest_sparse(Cc, codes, G)
        else:
            R, ties = _wilcox_dense(Xd, codes, G)
    n_g = np.bincount(codes, minlength=G).astype(np.float64)
    coef = (1.0 - ties / (float(N) ** 3 - N)) if tie_correct else np.ones_like(ties)
    sd = np.sqrt(coef[None, :] * (n_g * (N - n_g) * (N + 1) / 12.0)[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (R - (n_g * (N + 1) / 2.0)[:, None]) / sd
    z[np.isnan(z)] = 0.0
    return z


def rank_genes_groups(X, groups, features=None, *, method: str = "wilcoxon", reference: str = "rest",
                      groups_subset=None, tie_correct: bool = False, corr_method: str = "benjamini-hochberg",
                      pts: bool = False, n_genes: int | None = None, rankby_abs: bool = False,
                      log1p_base: float | None = None) -> pd.DataFrame:
    """``sc.tl.rank_genes_groups`` as a tidy table (equal to ``sc.get.rank_genes_groups_df(adata, group=None)``).

    Cell-level marker ranking: use it to describe clusters, not as evidence for differences between
    conditions (cells are not independent replicates; for conditions use :func:`arsc.pseudobulk` + DE).

    Parameters
    ----------
    X : cells x genes; expected log1p-normalized (scanpy's ``logfoldchanges`` use ``expm1``).
    groups : label per cell (e.g. ``ds.obs["leiden"]``).
    features : gene names (default ``0..p-1``).
    method : ``"wilcoxon"`` (normal approximation, no continuity correction), ``"t-test"`` (Welch) or
        ``"t-test_overestim_var"``. ``logreg`` is not implemented.
    reference : ``"rest"`` or one group label (ranks are then computed within group + reference only).
    groups_subset : restrict the output to these groups (scanpy ``groups=``).
    tie_correct : tie correction for wilcoxon (scanpy default False).
    corr_method : ``"benjamini-hochberg"`` or ``"bonferroni"``.
    pts : add ``pct_nz_group`` / ``pct_nz_reference``.
    n_genes : keep the top n genes per group (by |score| when ``rankby_abs``).
    rankby_abs : rank by absolute score.
    log1p_base : base used for log1p, if not natural (as ``adata.uns["log1p"]["base"]``).

    Returns
    -------
    DataFrame with ``group, names, scores, logfoldchanges, pvals, pvals_adj`` (+ ``pct_nz_*`` with ``pts``),
    sorted by decreasing score within group.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> rg = sx.rank_genes_groups(L, ds.obs["cell_type"], genes, n_genes=3)
    >>> sorted(rg[rg.group == "B"]["names"])
    ['CD19', 'CD79A', 'MS4A1']
    """
    if method not in ("wilcoxon", "t-test", "t-test_overestim_var"):
        raise NotImplementedError(f"rank_genes_groups: method {method!r} is not implemented; use 'wilcoxon', "
                                  "'t-test' or 't-test_overestim_var' (logreg is not ported)")
    if corr_method not in ("benjamini-hochberg", "bonferroni"):
        raise ValueError(f"corr_method must be 'benjamini-hochberg' or 'bonferroni', got {corr_method!r}")
    if len(groups) != X.shape[0]:
        raise ValueError(f"rank_genes_groups: {len(groups)} group labels for {X.shape[0]} cells; pass one label "
                         "per row of X (same order)")
    if features is not None and len(features) != X.shape[1]:
        raise ValueError(f"rank_genes_groups: {len(features)} feature names for {X.shape[1]} columns of X")
    lab = pd.Series(np.asarray(groups)).reset_index(drop=True)
    cat = lab.astype("category")
    levels = list(cat.cat.categories)
    if groups_subset is not None:
        sel = [g for g in levels if g in set(groups_subset)] if not isinstance(groups_subset, str) else [groups_subset]
    else:
        sel = levels
    codes_all = cat.cat.codes.to_numpy().astype(np.int64)
    N, P = X.shape
    names = np.asarray(pd.Index(features) if features is not None else pd.RangeIndex(P))
    if reference != "rest" and reference not in levels:
        raise ValueError(f"reference {reference!r} not found among the group labels {levels[:20]}; "
                         "use 'rest' or one of these labels")
    sel = [g for g in sel if g != reference] if reference != "rest" else sel
    expm1 = (lambda x: np.expm1(x * np.log(log1p_base))) if log1p_base else np.expm1
    Xs = as_csr(X) if sp.issparse(X) else np.asarray(X)

    # per-level sums (one sparse product each): n, sum, sum of squares, number of nonzeros
    Lg = len(levels)
    ok_c = codes_all >= 0
    OH = sp.csr_matrix((np.ones(ok_c.sum()), (codes_all[ok_c], np.flatnonzero(ok_c))), shape=(Lg, N))
    if sp.issparse(Xs):
        Xsq = Xs.copy()
        Xsq.data = Xsq.data.astype(np.float64) ** 2
        S1 = np.asarray((OH @ Xs).todense(), dtype=np.float64)
        S2 = np.asarray((OH @ Xsq).todense(), dtype=np.float64)
        Xnz = Xs.copy()
        Xnz.data = (Xnz.data != 0).astype(np.float64)
        NZ = np.asarray((OH @ Xnz).todense()) if pts else None
    else:
        Xf = Xs.astype(np.float64)
        S1, S2 = OH @ Xf, OH @ (Xf * Xf)
        NZ = OH @ (Xf != 0) if pts else None
    ng = np.asarray(OH.sum(1)).ravel()

    def stats_for(sel_levels):
        n = ng[sel_levels].sum()
        s1, s2 = S1[sel_levels].sum(0), S2[sel_levels].sum(0)
        mu = s1 / n
        var = (s2 / n - mu ** 2) * (n / (n - 1))
        return mu, var, (NZ[sel_levels].sum(0) / n if pts else None), int(n)

    out = []
    if method == "wilcoxon" and reference == "rest":
        keep_codes = np.array([levels.index(g) for g in sel])
        remap = -np.ones(len(levels), dtype=np.int64)
        remap[keep_codes] = np.arange(len(sel))
        # ranks are among ALL cells; non-selected groups only act as "rest" (code -1 -> not summed)
        Z = _wilcox_scores_rest(Xs, remap[codes_all], len(sel), tie_correct)
    for gi, g in enumerate(sel):
        ig = levels.index(g)
        mask = codes_all == ig
        ref_levels = [i for i in range(Lg) if i != ig] if reference == "rest" else [levels.index(reference)]
        m1, v1, p1, n1 = stats_for([ig])
        m2, v2, p2, n2 = stats_for(ref_levels)
        if n1 < 2:
            raise ValueError(f"Could not calculate statistics for group {g} since it only contains one sample.")
        mask_ref = ~mask if reference == "rest" else codes_all == levels.index(reference)
        if method == "wilcoxon":
            if reference == "rest":
                z = Z[gi]
            else:
                both = mask | mask_ref
                sub = Xs[both]
                c = np.where(mask[both], 0, 1).astype(np.int64)
                z = _wilcox_scores_rest(sub, c, 2, tie_correct)[0]
            pv = 2 * _st.norm.sf(np.abs(z))
        else:
            ns_rest = n2 if method == "t-test" else n1
            with np.errstate(invalid="ignore"):
                z, pv = _st.ttest_ind_from_stats(m1, np.sqrt(v1), n1, m2, np.sqrt(v2), ns_rest, equal_var=False)
            z = np.asarray(z, dtype=np.float64)
            pv = np.asarray(pv, dtype=np.float64)
            z[np.isnan(z)] = 0
            pv[np.isnan(pv)] = 1
        if corr_method == "benjamini-hochberg":
            padj = p_adjust_bh(np.where(np.isnan(pv), 1.0, pv))
        elif corr_method == "bonferroni":
            padj = np.minimum(pv * P, 1.0)
        else:
            raise ValueError(corr_method)
        lfc = np.log2((expm1(m1) + 1e-9) / (expm1(m2) + 1e-9))
        df = pd.DataFrame({"group": str(g), "names": names, "scores": z, "logfoldchanges": lfc,
                           "pvals": pv, "pvals_adj": padj})
        if pts:
            df["pct_nz_group"] = p1
            df["pct_nz_reference"] = p2
        key = np.abs(z) if rankby_abs else z
        o = np.argsort(-key, kind="stable")
        df = df.iloc[o]
        if n_genes is not None:
            df = df.iloc[:n_genes]
        out.append(df)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- highly_variable_genes
def highly_variable_genes(X, features=None, *, flavor: str = "seurat", n_top_genes: int | None = None,
                          min_mean: float = 0.0125, max_mean: float = 3.0, min_disp: float = 0.5,
                          max_disp: float = np.inf, n_bins: int = 20, span: float = 0.3,
                          log1p_base: float | None = None, batch_key=None) -> pd.DataFrame:
    """``sc.pp.highly_variable_genes`` for ``flavor`` in ``seurat`` | ``cell_ranger`` | ``seurat_v3``.

    seurat / cell_ranger: ``X`` = log1p-normalized cells x genes; columns ``means, dispersions,
    dispersions_norm, highly_variable`` (exactly scanpy's). With ``n_top_genes`` the mean/dispersion cutoffs
    are ignored. seurat_v3: ``X`` = RAW counts; delegates to :func:`arsc.pp.hvg_vst` (Seurat vst, R-validated;
    scanpy's skmisc loess is the same model); columns ``means, variances, variances_norm,
    highly_variable_rank, highly_variable_nbatches, highly_variable``.

    Parameters
    ----------
    X : cells x genes (log1p-normalized for seurat/cell_ranger; raw counts for seurat_v3).
    features : gene names (index of the result).
    flavor : ``"seurat"`` (default), ``"cell_ranger"``, ``"seurat_v3"`` (or ``"seurat_v3_paper"``).
    n_top_genes : number of genes to flag (required for seurat_v3).
    min_mean, max_mean, min_disp, max_disp : cutoffs for seurat/cell_ranger when ``n_top_genes`` is None.
    n_bins : number of mean bins for dispersion normalisation (default 20).
    span : loess span for seurat_v3 (default 0.3).
    log1p_base : base used for log1p, if not natural.
    batch_key : array of batch labels (seurat_v3 only): per-batch ranking combined by median rank, as scanpy.

    Returns
    -------
    DataFrame indexed by gene with the scanpy columns listed above.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> hv = sx.highly_variable_genes(L, genes, n_top_genes=50)
    >>> int(hv["highly_variable"].sum())
    50
    >>> v3 = sx.highly_variable_genes(ds.X("RNA", "counts"), genes, flavor="seurat_v3", n_top_genes=50)
    >>> "variances_norm" in v3.columns
    True
    """
    names = pd.Index(features) if features is not None else pd.RangeIndex(X.shape[1])
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        return _hvg_v3(X, names, n_top_genes, span, flavor, batch_key)
    if flavor not in ("seurat", "cell_ranger"):
        raise ValueError(flavor)
    if batch_key is not None:
        raise NotImplementedError("batch_key for seurat/cell_ranger flavors")
    if sp.issparse(X):
        x = as_csr(X).copy()
        x = x.astype(np.float64) if x.dtype != np.float64 else x
        if flavor == "seurat":
            if log1p_base:
                x.data *= np.log(log1p_base)
            np.expm1(x.data, out=x.data)
    else:
        x = np.array(X, dtype=np.float64)
        if flavor == "seurat":
            if log1p_base:
                x *= np.log(log1p_base)
            np.expm1(x, out=x)
    mean, var = _mean_var_cols(x)
    mean = mean.copy()
    mean[mean == 0] = 1e-12
    disp = var / mean
    if flavor == "seurat":
        disp[disp == 0] = np.nan
        disp = np.log(disp)
        mean = np.log1p(mean)
    df = pd.DataFrame({"means": mean, "dispersions": disp})
    bins = n_bins if flavor == "seurat" else np.r_[-np.inf, np.percentile(mean, np.arange(10, 105, 5)), np.inf]
    df["mean_bin"] = pd.cut(df["means"], bins=bins)
    g = df.groupby("mean_bin", observed=True)["dispersions"]
    if flavor == "seurat":
        st = g.agg(avg="mean", dev="std")
        one = st["dev"].isnull()
        if one.any():
            st.loc[one, "dev"] = st.loc[one, "avg"]
            st.loc[one, "avg"] = 0
    else:
        from statsmodels.robust import mad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            st = g.agg(avg="median", dev=mad)
    st = st.loc[df["mean_bin"]].set_index(df.index)
    df["dispersions_norm"] = (df["dispersions"] - st["avg"]) / st["dev"]
    dn = df["dispersions_norm"].to_numpy()
    if n_top_genes is not None:
        n = min(n_top_genes, len(df))
        v = dn[~np.isnan(dn)].copy()
        n = min(n, v.size)
        v[::-1].sort()
        cut = v[n - 1]
        hv = np.nan_to_num(dn, nan=-np.inf) >= cut
    else:
        d2 = np.nan_to_num(dn)
        hv = (mean > min_mean) & (mean < max_mean) & (d2 > min_disp) & (d2 < max_disp)
    df["highly_variable"] = hv
    df = df.drop(columns="mean_bin")
    df.index = names
    return df


def _hvg_v3(X, names, n_top_genes, span, flavor, batch_key):
    if n_top_genes is None:
        raise ValueError("seurat_v3 requires n_top_genes")
    N = X.shape[0]
    Xc = as_csr(X)
    mu_all, var_all = _mean_var_cols(Xc)
    batches = np.zeros(N, dtype=int) if batch_key is None else pd.Categorical(np.asarray(batch_key)).codes
    nv = []
    for b in np.unique(batches):
        sub = Xc[batches == b]
        r = hvg_vst(sub, nfeatures=n_top_genes, loess_span=span)
        v = r["variance_standardized"].to_numpy().copy()
        v[~np.isfinite(v)] = 0.0  # constant genes: scanpy gives 0 (reg_std = 1, no variance)
        nv.append(v)
    nv = np.vstack(nv)

    rk = np.argsort(np.argsort(-nv, axis=1), axis=1).astype(np.float32)  # same (unstable) tie order as scanpy
    nb_hv = (rk < n_top_genes).sum(0)
    rk[rk >= n_top_genes] = np.nan
    med = np.nanmedian(np.where(np.isnan(rk), np.nan, rk), axis=0) if rk.shape[0] > 1 else rk[0]
    df = pd.DataFrame({"means": mu_all, "variances": var_all}, index=names)
    df["highly_variable_nbatches"] = nb_hv
    df["highly_variable_rank"] = med
    df["variances_norm"] = np.where(np.isfinite(nv), nv, 0).mean(0)
    cols = ["highly_variable_rank", "highly_variable_nbatches"]
    asc = [True, False] if flavor == "seurat_v3" else [False, True]
    if flavor == "seurat_v3_paper":
        cols = cols[::-1]
    order = df[cols].sort_values(cols, ascending=asc, na_position="last").index
    df["highly_variable"] = False
    df.loc[order[:int(n_top_genes)], "highly_variable"] = True
    return df


# --------------------------------------------------------------------------- dendrogram
def dendrogram(rep, groups, *, cor_method: str = "pearson", linkage_method: str = "complete",
               optimal_ordering: bool = False) -> dict:
    """``sc.tl.dendrogram``: hierarchical clustering of group means of a representation (PCA by default).

    Parameters
    ----------
    rep : cells x d array (e.g. ``X_pca``) or a cells x genes DataFrame (``var_names`` mode).
    groups : label per cell.
    cor_method : correlation between group means: ``"pearson"`` (default), ``"spearman"``, ``"kendall"``.
    linkage_method : scipy linkage method (default ``"complete"``).
    optimal_ordering : scipy optimal leaf ordering.

    Returns
    -------
    dict with ``linkage``, ``categories_ordered``, ``categories_idx_ordered``, ``correlation_matrix`` and
    ``dendrogram_info`` (scipy ``dendrogram`` output, no plotting).

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> dn = sx.dendrogram(pcs, ds.obs["cell_type"])
    >>> sorted(dn["categories_ordered"])
    ['B', 'CD4 T', 'CD8 T', 'Mono', 'NK']
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial import distance

    cat = pd.Series(np.asarray(groups)).astype("category")
    df = pd.DataFrame(np.asarray(rep, dtype=np.float64))
    df.index = cat.to_numpy()
    cats = cat.cat.categories
    mean_df = df.groupby(level=0, observed=True).mean().loc[cats]
    corr = mean_df.T.corr(method=cor_method).clip(-1, 1)
    z = sch.linkage(distance.squareform(1 - corr), method=linkage_method, optimal_ordering=optimal_ordering)
    info = sch.dendrogram(z, labels=list(cats), no_plot=True)
    return dict(linkage=z, categories_ordered=info["ivl"], categories_idx_ordered=info["leaves"],
                correlation_matrix=corr.to_numpy(), dendrogram_info=info)


# --------------------------------------------------------------------------- regress_out
def regress_out(X, covariates, *, categorical: bool = False) -> np.ndarray:
    """``sc.pp.regress_out``: residuals of every gene after OLS on ``covariates`` (+ intercept).

    Parameters
    ----------
    X : cells x genes (sparse is densified, as scanpy; subset genes first for large data).
    covariates : cells x k numeric array / DataFrame, or (``categorical=True``) one label vector.
    categorical : treat ``covariates`` as a single categorical key: each gene is centered per category.

    Returns
    -------
    dense numpy array of residuals, cells x genes.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> import numpy as np
    >>> depth = np.asarray(ds.X("RNA", "counts").sum(1)).ravel()
    >>> sx.regress_out(L[:, :20], depth[:, None]).shape
    (600, 20)
    """
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    Xd = Xd.astype(np.float64 if Xd.dtype.itemsize > 4 or Xd.dtype.kind == "i" else Xd.dtype, copy=True)
    if categorical:
        c = pd.Series(np.asarray(covariates)).astype("category").cat.codes.to_numpy()
        for k in np.unique(c):
            m = c == k
            Xd[m] -= Xd[m].mean(0)
        return Xd
    R = np.asarray(covariates, dtype=np.float64)
    if R.ndim == 1:
        R = R[:, None]
    R = np.c_[np.ones(R.shape[0]), R]
    beta = np.linalg.lstsq(R, Xd.astype(np.float64), rcond=None)[0]
    return (Xd - R @ beta).astype(Xd.dtype, copy=False)


# --------------------------------------------------------------------------- qc_metrics
@nb.njit(parallel=True, cache=True)
def _top_pct(indptr, data, tops, total):
    n = indptr.size - 1
    out = np.zeros((n, tops.size))
    for i in nb.prange(n):
        v = np.sort(data[indptr[i]:indptr[i + 1]])[::-1]
        c = np.cumsum(v)
        for t in range(tops.size):
            k = min(tops[t], v.size)
            out[i, t] = 100.0 * c[k - 1] / total[i] if (k > 0 and total[i] > 0) else np.nan
    return out


def qc_metrics(counts, features=None, qc_vars: dict | None = None, percent_top=(20,), log1p: bool = True) -> pd.DataFrame:
    """``sc.pp.calculate_qc_metrics`` (cell-level columns) on a raw-counts matrix.

    Parameters
    ----------
    counts : cells x genes raw counts.
    features : gene names (needed when ``qc_vars`` values are gene lists).
    qc_vars : ``{"mt": boolean mask or list of genes, ...}``.
    percent_top : top-N gene counts for ``pct_counts_in_top_<N>_genes`` (default ``(20,)``).
    log1p : add ``log1p_`` versions (default True).

    Returns
    -------
    DataFrame, one row per cell: ``n_genes_by_counts, total_counts, pct_counts_in_top_<n>_genes,
    total_counts_<v>, pct_counts_<v>`` (+ ``log1p_*``), as scanpy.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> qc = sx.qc_metrics(ds.X("RNA", "counts"), genes, qc_vars={"mt": genes.str.startswith("MT-")})
    >>> keep = (qc["n_genes_by_counts"] >= 20) & (qc["pct_counts_mt"] < 20)
    >>> "pct_counts_mt" in qc.columns
    True
    """
    C = as_csr(counts)
    n_genes = np.diff(C.indptr).astype(np.int64)
    total = np.asarray(C.sum(1)).ravel().astype(np.float64)
    out = {"n_genes_by_counts": n_genes, "total_counts": total}
    if log1p:
        out["log1p_n_genes_by_counts"] = np.log1p(n_genes)
        out["log1p_total_counts"] = np.log1p(total)
    if percent_top:
        tops = np.asarray(percent_top, dtype=np.int64)
        pt = _top_pct(C.indptr.astype(np.int64), C.data.astype(np.float64), tops, total)
        for j, t in enumerate(tops):
            out[f"pct_counts_in_top_{t}_genes"] = pt[:, j]
    names = pd.Index(features) if features is not None else None
    for k, m in (qc_vars or {}).items():
        m = np.asarray(m)
        if m.dtype != bool:
            m = names.isin(list(m))
        tv = np.asarray(C[:, np.flatnonzero(m)].sum(1)).ravel().astype(np.float64)
        out[f"total_counts_{k}"] = tv
        if log1p:
            out[f"log1p_total_counts_{k}"] = np.log1p(tv)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"pct_counts_{k}"] = tv / total * 100
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- neighbors / leiden / umap
@records_seed("random_state")
def neighbors(emb, n_neighbors: int = 15, *, metric: str = "euclidean", method: str = "auto",
              random_state: int = 0) -> dict:
    """``sc.pp.neighbors`` (``method="umap"``): kNN + UMAP fuzzy-simplicial-set connectivities.

    Uses :func:`arsc.knn` with scanpy's switch (``method="auto"``: exact up to 4,096 cells, pynndescent with
    scanpy's settings above; ``"exact"`` forces exact search). As
    scanpy, ``n_neighbors`` counts the cell itself (n_neighbors - 1 real neighbours).

    Parameters
    ----------
    emb : cells x d embedding (e.g. PCA or Harmony).
    n_neighbors : neighbourhood size including the cell itself (default 15).
    metric : ``"euclidean"`` or ``"cosine"``.
    method : ``"auto"`` (default), ``"exact"`` or ``"approximate"`` (see :func:`arsc.knn`).
    random_state : integer seed (default 0): pynndescent's ``random_state`` (unused by exact search).

    Returns
    -------
    dict ``{"connectivities", "distances", "indices"}`` (sparse cells x cells graphs and the kNN index array).

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> sorted(nn)
    ['connectivities', 'distances', 'indices']
    """
    random_state = check_seed(random_state, "random_state")
    from umap.umap_ import fuzzy_simplicial_set

    from .neighbors import knn as _knn, l2_normalize

    X = np.asarray(emb, dtype=np.float64)
    if metric == "cosine":
        idx, d = _knn(l2_normalize(X), n_neighbors, method=method, seed=random_state)
        d = 1.0 - (1.0 - d ** 2 / 2.0)  # unit-vector euclid^2 / 2 = cosine distance
    elif metric == "euclidean":
        idx, d = _knn(X, n_neighbors, method=method, seed=random_state)
    else:
        raise ValueError(metric)
    n = X.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn, _s, _r = fuzzy_simplicial_set(sp.coo_matrix((n, 1)), n_neighbors, None, None,
                                            knn_indices=idx, knn_dists=d, set_op_mix_ratio=1.0,
                                            local_connectivity=1.0)
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    dist = sp.csr_matrix((d[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return {"connectivities": conn.tocsr(), "distances": dist, "indices": idx}


@records_seed("random_state")
def leiden(connectivities, resolution: float = 1.0, *, random_state: int = 0, n_iterations: int = -1,
           directed: bool = True, use_weights: bool = True, flavor: str = "leidenalg") -> np.ndarray:
    """``sc.tl.leiden`` on a connectivities graph (scanpy 1.11 defaults); same graph + seed -> identical labels.

    Default settings: leidenalg, directed graph, ``n_iterations=-1``, ``RBConfigurationVertexPartition``.
    Labels are integers exactly as leidenalg numbers them (scanpy stores them as strings). For
    ``flavor="igraph"`` a private ``random.Random(random_state)`` is installed as igraph's RNG for the call and
    igraph's default RNG restored afterwards (Python's global ``random`` state is never touched).

    Parameters
    ----------
    connectivities : cells x cells weighted graph (``neighbors(...)["connectivities"]``).
    resolution : resolution parameter (default 1.0; higher = more clusters).
    random_state : integer seed (default 0).
    n_iterations : leidenalg iterations (-1 = until stable).
    directed : build a directed igraph graph (scanpy default True for leidenalg).
    use_weights : use edge weights.
    flavor : ``"leidenalg"`` (default) or ``"igraph"`` (modularity, undirected).

    Returns
    -------
    numpy integer array of cluster labels, one per cell.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> labels = sx.leiden(nn["connectivities"], resolution=0.5)
    >>> labels.shape
    (600,)
    """
    import igraph as ig

    random_state = check_seed(random_state, "random_state")

    A = sp.csr_matrix(connectivities)
    src, tgt = A.nonzero()
    w = np.asarray(A[src, tgt]).ravel()
    if flavor == "leidenalg":
        import leidenalg
        g = ig.Graph(directed=directed)
        g.add_vertices(A.shape[0])
        g.add_edges(list(zip(src, tgt)))
        g.es["weight"] = w
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        weights=np.array(g.es["weight"]).astype(np.float64) if use_weights else None,
                                        resolution_parameter=resolution, seed=random_state,
                                        n_iterations=n_iterations)
        return np.asarray(part.membership)
    g = ig.Graph(directed=False)
    g.add_vertices(A.shape[0])
    g.add_edges(list(zip(src, tgt)))
    g.es["weight"] = w
    import random
    ig.set_random_number_generator(random.Random(random_state))
    try:
        part = g.community_leiden(objective_function="modularity", weights="weight" if use_weights else None,
                                  resolution=resolution, n_iterations=2 if n_iterations == -1 else n_iterations)
    finally:
        ig.set_random_number_generator(random)  # igraph's default
    return np.asarray(part.membership)


@records_seed("random_state")
def umap_from_connectivities(connectivities, *, min_dist: float = 0.5, spread: float = 1.0, n_components: int = 2,
                             maxiter: int | None = None, alpha: float = 1.0, gamma: float = 1.0,
                             negative_sample_rate: int = 5, init_pos: str = "spectral", random_state: int = 0) -> np.ndarray:
    """``sc.tl.umap``: UMAP layout optimisation on the scanpy connectivities graph (not on raw X).

    Same call path as scanpy (``umap.umap_.simplicial_set_embedding`` with ``a, b = find_ab_params``).
    ``random_state`` seeds the spectral-init noise and the SGD layout through a local ``RandomState``;
    umap-learn then runs the layout single-threaded, so same graph + seed -> bit-identical coordinates.

    Parameters
    ----------
    connectivities : cells x cells graph from :func:`neighbors`.
    min_dist : minimum distance (scanpy default 0.5).
    spread : effective scale of embedded points (default 1.0).
    n_components : output dimensions (default 2).
    maxiter : number of epochs (default: 500 up to 10,000 cells, else 200, as scanpy).
    alpha : initial learning rate.
    gamma : repulsion strength.
    negative_sample_rate : negative samples per positive edge.
    init_pos : ``"spectral"`` (default) or ``"random"``.
    random_state : integer seed (default 0); ``None`` is rejected.

    Returns
    -------
    numpy array, cells x ``n_components`` coordinates.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> sx.umap_from_connectivities(nn["connectivities"], maxiter=50).shape
    (600, 2)
    """
    from sklearn.utils import check_random_state
    from umap.umap_ import find_ab_params, simplicial_set_embedding

    A = sp.csr_matrix(connectivities)
    a, b = find_ab_params(spread, min_dist)
    n = A.shape[0]
    if maxiter is None:
        maxiter = 500 if n <= 10000 else 200
    rs = check_random_state(check_seed(random_state, "random_state"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb, _ = simplicial_set_embedding(
            data=None, graph=A, n_components=n_components, initial_alpha=alpha, a=a, b=b, gamma=gamma,
            negative_sample_rate=negative_sample_rate, n_epochs=maxiter, init=init_pos, random_state=rs,
            metric="euclidean", metric_kwds={}, densmap=False, densmap_kwds={}, output_dens=False,
            verbose=False)
    return np.asarray(emb)


# --------------------------------------------------------------------------- diffmap / dpt
@records_seed("random_state")
def diffmap(connectivities, n_comps: int = 15, *, random_state: int = 0) -> dict:
    """``sc.tl.diffmap`` on a scanpy connectivities graph (symmetric ``transitions_sym`` eigendecomposition).

    Parameters
    ----------
    connectivities : cells x cells graph from :func:`neighbors`.
    n_comps : number of diffusion components (default 15).
    random_state : integer seed (default 0) for the ARPACK start vector (local ``RandomState``).

    Returns
    -------
    dict with ``evals`` (n_comps), ``X_diffmap`` (cells x n_comps; column 0 is the stationary component, as
    scanpy) and the transition matrix pieces (``Tsym``, ``q``) needed by :func:`dpt`.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> dm = sx.diffmap(nn["connectivities"], n_comps=10)
    >>> dm["X_diffmap"].shape
    (600, 10)
    """
    random_state = check_seed(random_state, "random_state")
    from scipy.sparse.linalg import eigsh

    W = sp.csr_matrix(connectivities).astype(np.float64)
    q = np.asarray(W.sum(1)).ravel()
    Z = sp.diags(1.0 / q)                       # scanpy density normalization (Coifman & Lafon)
    Wn = Z @ W @ Z
    q2 = np.asarray(Wn.sum(1)).ravel()
    Tsym = sp.diags(1.0 / np.sqrt(q2)) @ Wn @ sp.diags(1.0 / np.sqrt(q2))
    rng = np.random.RandomState(random_state)
    v0 = rng.standard_normal(W.shape[0])
    w, V = eigsh(Tsym, k=n_comps, which="LM", v0=v0)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    return {"evals": w, "X_diffmap": V, "Tsym": Tsym, "q": q2}


def dpt(dm: dict, root: int, n_dcs: int = 10, *, thresh: float = 0.9994) -> np.ndarray:
    """``sc.tl.dpt`` diffusion pseudotime (``n_branchings=0``) from :func:`diffmap` output.

    Distance to the root in diffusion space (Haghverdi 2016 with the Wolf 2019 stationary-state term): sum over
    the first ``n_dcs`` components of ``(l / (1 - l) (psi_i - psi_root))^2`` for l < ``thresh``, plus the plain
    squared difference for l >= ``thresh``; then sqrt and divide by the maximum. A connected graph is assumed.

    Parameters
    ----------
    dm : output of :func:`diffmap`.
    root : index (position) of the root cell.
    n_dcs : number of diffusion components used (default 10).
    thresh : eigenvalue threshold of the stationary-state term (default 0.9994).

    Returns
    -------
    numpy array, pseudotime per cell in [0, 1] (0 at the root).

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> dm = sx.diffmap(nn["connectivities"], n_comps=10)
    >>> t = sx.dpt(dm, root=0)
    >>> float(t[0])
    0.0
    """
    w = np.asarray(dm["evals"][:n_dcs], dtype=np.float32).astype(np.float64)
    V = np.asarray(dm["X_diffmap"][:, :n_dcs], dtype=np.float32).astype(np.float64)
    diff = V[root][None, :] - V
    lo = w < thresh
    row = ((w[lo] / (1 - w[lo]))[None, :] * diff[:, lo]) ** 2
    row = row.sum(1) + (diff[:, ~lo] ** 2).sum(1)
    t = np.sqrt(row)
    return t / t.max()


# --------------------------------------------------------------------------- palantir pieces (deterministic parts)
@records_seed("seed")
def palantir_diffusion_maps(emb, n_components: int = 10, knn: int = 30, alpha: float = 0.0, seed: int = 0,
                            method: str = "auto") -> dict:
    """``palantir.utils.run_diffusion_maps`` (adaptive anisotropic kernel, Setty 2019).

    kNN (self excluded, ``knn - 1`` neighbours as scanpy's distance matrix), adaptive bandwidth = distance to
    the ``floor(knn / 3)``-th neighbour, kernel ``exp(-d / sigma_i)`` symmetrized as ``W + W^T``, row-normalized
    transition ``T``, ``eigs`` with palantir's tolerance and start vector (``default_rng(seed).random``). Exact
    kNN here vs scanpy's approximate kNN for > 4,096 cells in palantir: identical below that size.

    Parameters
    ----------
    emb : cells x d embedding (e.g. PCA / Harmony).
    n_components : number of eigenvectors (default 10).
    knn : neighbourhood size (default 30).
    alpha : anisotropy (default 0).
    seed : integer seed (default 0, as palantir) for the ARPACK start vector and the kNN (pynndescent).
    method : kNN method, ``"auto"`` (default: exact up to 4,096 cells), ``"exact"`` or ``"approximate"``.

    Returns
    -------
    dict with ``kernel``, ``T`` (sparse cells x cells), ``EigenVectors`` (cells x n_components, unit norm) and
    ``EigenValues`` (descending).

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> dm = sx.palantir_diffusion_maps(pcs, n_components=10)
    >>> dm["EigenVectors"].shape
    (600, 10)
    """
    seed = check_seed(seed)
    from scipy.sparse.linalg import eigs

    from .neighbors import knn as _knn

    X = np.asarray(emb, dtype=np.float64)
    N = X.shape[0]
    idx, d = _knn(X, knn, method=method, seed=seed)
    idx, d = idx[:, 1:], d[:, 1:]
    ak = int(np.floor(knn / 3))
    sig = np.partition(d, ak - 1, axis=1)[:, ak - 1]
    rows = np.repeat(np.arange(N), idx.shape[1])
    W = sp.csr_matrix((np.exp(-(d / sig[:, None]).ravel()), (rows, idx.ravel())), shape=(N, N))
    K = W + W.T
    if alpha > 0:
        D = np.ravel(K.sum(1))
        D[D != 0] = D[D != 0] ** (-alpha)
        M = sp.diags(D)
        K = M @ K @ M
    Dr = np.ravel(K.sum(1))
    Dr[Dr != 0] = 1.0 / Dr[Dr != 0]
    T = sp.diags(Dr) @ K
    v0 = np.random.default_rng(seed).random(N)
    w, V = eigs(T.tocsr(), n_components, tol=1e-4, maxiter=1000, v0=v0)
    w, V = np.real(w), np.real(V)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    return {"kernel": K.tocsr(), "T": T.tocsr(), "EigenVectors": V, "EigenValues": w}


def palantir_multiscale_space(dm: dict, n_eigs: int | None = None) -> np.ndarray:
    """``palantir.utils.determine_multiscale_space``: eigenvectors 1..n_eigs-1 scaled by ``l / (1 - l)``.

    Parameters
    ----------
    dm : output of :func:`palantir_diffusion_maps`.
    n_eigs : number of eigenvectors to use (default: palantir's eigen-gap rule, at least 3).

    Returns
    -------
    numpy array, cells x (n_eigs - 1) multiscale space.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> dm = sx.palantir_diffusion_maps(pcs, n_components=10)
    >>> sx.palantir_multiscale_space(dm, n_eigs=4).shape
    (600, 3)
    """
    vals = np.ravel(dm["EigenValues"])
    if n_eigs is None:
        gap = vals[:-1] - vals[1:]
        n_eigs = np.argsort(gap)[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(gap)[-2] + 1
        if n_eigs < 3:
            n_eigs = 3
    use = list(range(1, n_eigs))
    ev = vals[use]
    return dm["EigenVectors"][:, use] * (ev / (1 - ev))


def magic_impute(X, T, n_steps: int = 3, clip_threshold: float = 1e-2, dtype=np.float32, chunk: int = 500):
    """``palantir.utils.run_magic_imputation``: ``T^n_steps @ X``, values below ``clip_threshold`` set to 0.

    Applies ``T`` ``n_steps`` times to dense gene chunks instead of forming the denser ``T^n_steps`` (same
    result up to float32 rounding, ~1e-6 relative). Imputed values are smoothed over neighbours: use them for
    visualisation, not as independent observations in tests.

    Parameters
    ----------
    X : cells x genes (sparse or dense), e.g. log-normalized data of the genes of interest.
    T : cells x cells row-stochastic transition matrix (``palantir_diffusion_maps(...)["T"]``).
    n_steps : diffusion steps (default 3).
    clip_threshold : values below this are set to 0 (default 0.01).
    dtype : computation dtype (default float32).
    chunk : genes per dense chunk (default 500).

    Returns
    -------
    scipy.sparse.csr_matrix, cells x genes.

    Examples
    --------
    >>> import arsc
    >>> from arsc import scanpy_extra as sx
    >>> ds = arsc.datasets.toy(n_cells=600)
    >>> genes = ds.features("RNA")
    >>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
    >>> Z, _, _ = arsc.pp.scale(L.toarray())
    >>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
    >>> nn = sx.neighbors(pcs, n_neighbors=15)
    >>> dm = sx.palantir_diffusion_maps(pcs, n_components=10)
    >>> sx.magic_impute(L[:, :20], dm["T"]).shape
    (600, 20)
    """
    Tn = sp.csr_matrix(T).astype(dtype)
    out = []
    for a in range(0, X.shape[1], chunk):
        Y = X[:, a:a + chunk]
        Y = (Y.toarray() if sp.issparse(Y) else np.asarray(Y)).astype(dtype, copy=False)
        for _ in range(n_steps):
            Y = Tn @ Y
        Y[Y < clip_threshold] = 0
        out.append(sp.csr_matrix(Y))
    return sp.hstack(out, format="csr")
