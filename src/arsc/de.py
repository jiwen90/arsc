"""Pseudobulk differential expression + gene-set enrichment (Python port of an R DESeq2/limma/fgsea layer).

    m  = (obs.cell_type == "CD8 T").to_numpy()
    pb = arsc.pseudobulk(C[m], obs[m], ["patient", "timepoint"], features=feats, min_cells=20)
    counts, cd = pb.to_frame(), pb.groups                     # genes x samples, sample table
    cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])                               # reference first
    r1 = de.deseq2(counts, cd, "~ patient + timepoint", ("timepoint", "post", "pre"))                  # DESeq2 (pydeseq2)
    r2 = de.voom(counts, cd, "~ timepoint", block="patient", coef=("timepoint", "post"))               # voom + dupCor
    gs = de.msigdb("H")
    fg = de.fgsea(r1.set_index("gene")["stat"], gs)                                                    # fgsea (simple)

* ``deseq2``   -> pydeseq2 (independent re-implementation of DESeq2 1.x; no LFC shrinkage, Wald test, alpha = 0.1)
* ``voom``     -> own port of edgeR filterByExpr/calcNormFactors(TMM) + limma voom, duplicateCorrelation (REML),
                  gls lmFit, eBayes (fitFDist/squeezeVar) and topTable
* ``fgsea``    -> own port of fgsea's ES / NES / leading edge with sampled nulls (fgseaSimple); numba kernels
* ``msigdb``   -> msigdbr export (Rscript, once) cached under the arsc cache directory (``ARSC_CACHE``, else
                  ``<AR_ROOT>/.cache`` when ``AR_ROOT`` is set, else ``~/.cache/arsc``), subfolder ``genesets/``

Numerical agreement with R is documented in STATS_PARITY.md.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from scipy import special, stats as sps
from scipy.optimize import minimize_scalar

from ._utils import check_seed
from ._seedlog import records_seed
from .stats import bh

__all__ = ["deseq2", "voom", "fgsea", "msigdb", "filter_by_expr", "tmm_factors", "duplicate_correlation"]

def _cache_root() -> Path:
    """arsc cache directory: ``$ARSC_CACHE``, else ``$AR_ROOT/.cache``, else ``~/.cache/arsc``."""
    if os.environ.get("ARSC_CACHE"):
        return Path(os.environ["ARSC_CACHE"])
    if os.environ.get("AR_ROOT"):
        return Path(os.environ["AR_ROOT"]) / ".cache"
    return Path.home() / ".cache" / "arsc"


GENESET_CACHE = _cache_root() / "genesets"
_RSCRIPT = os.environ.get("RSCRIPT", "Rscript")


# ------------------------------------------------------------------ helpers
def _make_names(s: str) -> str:
    """R make.names() for factor levels."""
    import re
    s = re.sub(r"[^A-Za-z0-9._]", ".", str(s))
    if not re.match(r"^([A-Za-z]|\.(?![0-9]))", s):
        s = "X" + s
    return s


def _vars_in(formula: str) -> list[str]:
    import re
    f = formula.replace("~", " ")
    return [v for v in re.split(r"[\s+*:()|]+", f) if v and not re.fullmatch(r"[0-9.]+", v)]


def _align_coldata(counts: pd.DataFrame, coldata: pd.DataFrame) -> pd.DataFrame:
    """coldata rows in the order of counts' columns, with a clear error when samples are missing."""
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("counts must be a genes x samples DataFrame (Pseudobulk.to_frame())")
    missing = [c for c in counts.columns if c not in coldata.index]
    if missing:
        raise KeyError(f"{len(missing)} sample(s) of counts are not in coldata.index (e.g. {missing[:3]}); "
                       "coldata must be indexed by the sample names = counts.columns (use Pseudobulk.groups). "
                       "Did you pass the cells x genes matrix instead of pb.to_frame() (genes x samples)?")
    return coldata.loc[counts.columns].copy()


def _check_raw_counts(counts: pd.DataFrame, fn: str, strict: bool) -> None:
    """Raw pseudobulk counts are non-negative integers; log-normalised input is a common silent mistake."""
    v = counts.to_numpy(dtype=float, na_value=np.nan)
    bad = bool(np.isnan(v).any() or (v < 0).any() or (v != np.round(v)).any())
    if bad:
        msg = (f"{fn}: counts are not non-negative integers; pass raw counts summed per sample "
               "(arsc.pseudobulk on the counts layer, then .to_frame()), not log-normalised or scaled values")
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg, stacklevel=3)


# ------------------------------------------------------------------ DESeq2 (pydeseq2)

_DDS_CLS = None


def _desq2_dataset():
    """pydeseq2 DeseqDataSet whose parametric dispersion trend follows DESeq2 exactly (parametricDispersionFit:
    Gamma-identity GLM on genes with gene-wise dispersion > 100*min_disp, iteratively dropping genes with
    disp/fit outside (1e-4, 15)).  pydeseq2 0.5's own L-BFGS fit fails to converge on small designs and silently
    falls back to a flat mean trend, which shifts dispersions, SEs and Wald statistics."""
    global _DDS_CLS
    if _DDS_CLS is not None:
        return _DDS_CLS
    import warnings
    from pydeseq2.dds import DeseqDataSet

    class _DDS(DeseqDataSet):
        def _fit_parametric_dispersion_trend(self, vst: bool = False):
            if vst:
                return super()._fit_parametric_dispersion_trend(vst)
            import statsmodels.api as sm
            nz = self.var["non_zero"].values
            means = self.var["_normed_means"].values[nz]
            disps = self.var["genewise_dispersions"].values[nz]
            use = disps > 100 * self.min_disp
            means, disps = means[use], disps[use]
            coefs = np.array([0.1, 1.0])
            try:
                for _ in range(11):
                    resid = disps / (coefs[0] + coefs[1] / means)
                    good = (resid > 1e-4) & (resid < 15)
                    X = np.column_stack([np.ones(good.sum()), 1 / means[good]])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = sm.GLM(disps[good], X, family=sm.families.Gamma(sm.families.links.Identity())).fit(
                            start_params=coefs, maxiter=100)
                    old, coefs = coefs, np.asarray(fit.params)
                    if not (coefs > 0).all():
                        raise RuntimeError("parametric dispersion fit failed")
                    if np.sum(np.log(coefs / old) ** 2) < 1e-6:
                        break
                else:
                    raise RuntimeError("parametric dispersion fit did not converge")
            except Exception:
                warnings.warn("DESeq2-style parametric dispersion fit failed; using pydeseq2's fallback")
                return super()._fit_parametric_dispersion_trend(vst)
            self.uns["trend_coeffs"] = pd.Series(coefs, index=["a0", "a1"])
            self.uns["disp_function_type"] = "parametric"
            self.var["fitted_dispersions"] = np.full(self.n_vars, np.nan)
            self.var.loc[self.var["non_zero"], "fitted_dispersions"] = self.disp_function(
                self.var.loc[self.var["non_zero"], "_normed_means"])

    _DDS_CLS = _DDS
    return _DDS

def deseq2(counts: pd.DataFrame, coldata: pd.DataFrame, design: str, contrast, min_count: int = 10,
           min_samples: int = 3, alpha: float = 0.1, quiet: bool = True, n_cpus: int = 1) -> pd.DataFrame:
    """DESeq2 Wald-test DE on pseudobulk counts (pydeseq2 with DESeq2's parametric dispersion trend).

    Parity with R DESeq2: log2FC r 0.9998, Wald stat r 0.9993-0.9998, same padj < 0.1 calls (STATS_PARITY.md).
    No LFC shrinkage.

    Parameters
    ----------
    counts : genes x samples integer table (``Pseudobulk.to_frame()``).
    coldata : sample table indexed by sample name (the columns of ``counts``), e.g. ``Pseudobulk.groups``.
    design : formula, e.g. ``"~ patient + timepoint"`` (paired: patient as a fixed block).
    contrast : ``(variable, level, reference)``, e.g. ``("timepoint", "post", "pre")``.
    min_count, min_samples : genes with fewer than ``min_samples`` samples at count >= ``min_count`` are
        removed first.
    alpha : independent-filtering / padj target (DESeq2 ``results()`` default 0.1).
    quiet : silence pydeseq2.
    n_cpus : pydeseq2 worker processes.

    Returns
    -------
    DataFrame with R's columns ``baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, gene``, sorted by
    pvalue; ``.attrs`` holds ``n_samples`` and ``n_genes_tested``. Needs ``pip install pydeseq2``.

    Examples
    --------
    >>> import arsc, pandas as pd
    >>> ds = arsc.datasets.toy()
    >>> m = ((ds.obs.cell_type == "CD8 T") & (ds.obs.response == "R")).to_numpy()
    >>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], ds.obs[m], ["patient", "timepoint"],
    ...                      features=ds.features("RNA"), min_cells=10)
    >>> cd = pb.groups.copy()
    >>> cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])   # reference level first
    >>> res = arsc.de.deseq2(pb.to_frame(), cd, "~ patient + timepoint",
    ...                      ("timepoint", "post", "pre"))                   # doctest: +SKIP
    >>> res.head(5).gene.tolist()                                            # doctest: +SKIP
    """
    try:
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("arsc.de.deseq2 needs pydeseq2: pip install pydeseq2 "
                          "(or use arsc.de.voom, which has no extra dependency)") from e

    cd = _align_coldata(counts, coldata)
    var_, lvl_, ref_ = contrast if len(contrast) == 3 else (None, None, None)
    if var_ is None:
        raise ValueError(f"contrast must be (variable, level, reference), e.g. ('timepoint', 'post', 'pre'); got {contrast!r}")
    if var_ not in cd.columns:
        raise KeyError(f"contrast variable {var_!r} not in coldata; available: {list(cd.columns)}")
    lv = set(cd[var_].astype(str))
    bad = [x for x in (lvl_, ref_) if str(x) not in lv]
    if bad:
        raise ValueError(f"contrast level(s) {bad} not in coldata[{var_!r}]; levels are {sorted(lv)}")
    miss = [v for v in _vars_in(design) if v not in cd.columns]
    if miss:
        raise KeyError(f"design variable(s) {miss} not in coldata; available: {list(cd.columns)}")
    _check_raw_counts(counts, "de.deseq2", strict=True)
    keep = (counts.values >= min_count).sum(axis=1) >= min_samples
    if not keep.any():
        raise ValueError(f"de.deseq2: no gene has >= {min_count} counts in >= {min_samples} samples; "
                         "check that counts are raw pseudobulk sums, or lower min_count / min_samples")
    cnt = counts.loc[keep].round().astype(int)
    for v in _vars_in(design):
        if v in cd and (cd[v].dtype == object or str(cd[v].dtype) in ("category", "string")):
            cd[v] = cd[v].astype(str).map(_make_names).astype("category")
    var, lvl, ref = contrast
    contrast_m = [var, _make_names(lvl), _make_names(ref)]
    cd = cd[[c for c in cd.columns if c in _vars_in(design)]]
    inf = DefaultInference(n_cpus=n_cpus)
    # counts go in as an AnnData with an integer array: pydeseq2 then validates them with numpy instead of a
    # per-gene pandas apply (2 s at 10,000 genes); same values and checks (raw counts were already checked above)
    import anndata as ad
    adata = ad.AnnData(X=cnt.T.astype(int).to_numpy(), obs=cd.copy(),  # same array (and layout) as counts=
                       var=pd.DataFrame(index=pd.Index(cnt.index.astype(str), name=None)))
    dds = _desq2_dataset()(adata=adata, design=design, refit_cooks=True, inference=inf, quiet=quiet)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast_m, alpha=alpha, inference=inf, quiet=quiet)
    st.summary()
    r = st.results_df.copy()
    r["gene"] = r.index
    r = r.sort_values("pvalue", na_position="last")
    r.attrs["n_samples"] = cnt.shape[1]
    r.attrs["n_genes_tested"] = cnt.shape[0]
    return r


# ------------------------------------------------------------------ edgeR bits
def filter_by_expr(counts: np.ndarray, design: np.ndarray, min_count: float = 10, min_total_count: float = 15,
                   large_n: int = 10, min_prop: float = 0.7) -> np.ndarray:
    """edgeR ``filterByExpr(DGEList(counts), design = ...)`` (group = NULL branch).

    Parameters
    ----------
    counts : genes x samples count array.
    design : samples x p design matrix.
    min_count : minimum count (scaled to CPM at the median library size).
    min_total_count : minimum total count per gene.
    large_n : sample-size threshold for the ``min_prop`` rule.
    min_prop : proportion rule for large groups.

    Returns
    -------
    numpy.ndarray of bool, one per gene (True = keep).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> counts = np.array([[0, 0, 1, 0], [50, 60, 40, 55], [100, 90, 120, 80]])
    >>> design = np.column_stack([np.ones(4), [0, 0, 1, 1]])
    >>> arsc.de.filter_by_expr(counts, design).tolist()
    [False, True, True]
    """
    lib = counts.sum(axis=0).astype(float)
    Q, _ = np.linalg.qr(design)
    rank = np.linalg.matrix_rank(design)
    U, s, _ = np.linalg.svd(design, full_matrices=False)
    U = U[:, s > s.max() * 1e-10][:, :rank]
    h = (U ** 2).sum(axis=1)
    mss = 1 / h.max()
    if mss > large_n:
        mss = large_n + (mss - large_n) * min_prop
    cutoff = min_count / np.median(lib) * 1e6
    cpm = counts / lib * 1e6
    return ((cpm >= cutoff).sum(axis=1) >= mss - 1e-14) & (counts.sum(axis=1) >= min_total_count)


def tmm_factors(counts: np.ndarray, logratio_trim: float = 0.3, sum_trim: float = 0.05, acutoff: float = -1e10):
    """edgeR ``calcNormFactors(method = "TMM")``: weighted trimmed mean of M-values, upper-quartile reference.

    Parameters
    ----------
    counts : genes x samples count array.
    logratio_trim : trim fraction for M values.
    sum_trim : trim fraction for A values.
    acutoff : cutoff on A values.

    Returns
    -------
    numpy.ndarray of normalisation factors, one per sample (geometric mean 1).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = arsc.de.tmm_factors(rng.poisson(20, size=(200, 4)))
    >>> f.shape
    (4,)
    """
    x = counts[counts.sum(axis=1) > 0].astype(float)
    lib = x.sum(axis=0)
    f75 = np.quantile(x / lib, 0.75, axis=0)  # R type-7 == numpy default
    ref = int(np.argmin(np.abs(f75 - f75.mean())))
    f = np.ones(x.shape[1])
    for j in range(x.shape[1]):
        if j == ref:
            continue
        obs, rf = x[:, j], x[:, ref]
        nO, nR = lib[j], lib[ref]
        with np.errstate(divide="ignore", invalid="ignore"):
            logR = np.log2((obs / nO) / (rf / nR))
            absE = (np.log2(obs / nO) + np.log2(rf / nR)) / 2
            v = (nO - obs) / nO / obs + (nR - rf) / nR / rf
        fin = np.isfinite(logR) & np.isfinite(absE) & (absE > acutoff)
        logR, absE, v = logR[fin], absE[fin], v[fin]
        if np.max(np.abs(logR)) < 1e-6:
            f[j] = 1.0
            continue
        n = len(logR)
        loL = int(np.floor(n * logratio_trim)) + 1
        hiL = n + 1 - loL
        loS = int(np.floor(n * sum_trim)) + 1
        hiS = n + 1 - loS
        rl, ra = sps.rankdata(logR), sps.rankdata(absE)
        k = (rl >= loL) & (rl <= hiL) & (ra >= loS) & (ra <= hiS)
        val = np.sum(logR[k] / v[k]) / np.sum(1 / v[k])
        f[j] = 2 ** (0 if np.isnan(val) else val)
    return f / np.exp(np.mean(np.log(f)))


# ------------------------------------------------------------------ limma bits
def _lowess(x, y, f=0.5, iters=3):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    delta = 0.01 * (x.max() - x.min())
    return lowess(y, x, frac=f, it=iters, delta=delta, return_sorted=True)


def _lmfit_ols(Y, X):
    """Y genes x samples; returns coef (G x p), sigma (G), stdev_unscaled (G x p), df, Amean."""
    n, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    B = Y @ X @ XtXi
    res = Y - B @ X.T
    df = n - np.linalg.matrix_rank(X)
    sigma = np.sqrt((res ** 2).sum(axis=1) / df)
    su = np.tile(np.sqrt(np.diag(XtXi)), (Y.shape[0], 1))
    return B, sigma, su, df


def _lmfit_gls(Y, X, block, rho, W=None):
    """limma gls.series: per-gene whitening with V = W^-1/2 C W^-1/2, C = block-exchangeable correlation."""
    n, p = X.shape
    b = np.asarray(block)
    C = (b[:, None] == b[None, :]) * rho
    np.fill_diagonal(C, 1.0)
    G = Y.shape[0]
    B = np.zeros((G, p)); sigma = np.zeros(G); su = np.zeros((G, p))
    df = n - np.linalg.matrix_rank(X)
    if W is None:
        L = np.linalg.cholesky(C)  # C = L L'
        Xw = np.linalg.solve(L, X)
        Yw = np.linalg.solve(L, Y.T).T
        XtXi = np.linalg.pinv(Xw.T @ Xw)
        B = Yw @ Xw @ XtXi
        res = Yw - B @ Xw.T
        return B, np.sqrt((res ** 2).sum(axis=1) / df), np.tile(np.sqrt(np.diag(XtXi)), (G, 1)), df
    for g in range(G):
        wr = 1 / np.sqrt(W[g])
        V = wr[:, None] * C * wr[None, :]
        L = np.linalg.cholesky(V)
        Xw = np.linalg.solve(L, X)
        yw = np.linalg.solve(L, Y[g])
        XtXi = np.linalg.pinv(Xw.T @ Xw)
        beta = XtXi @ Xw.T @ yw
        r = yw - Xw @ beta
        B[g] = beta
        sigma[g] = np.sqrt(r @ r / df)
        su[g] = np.sqrt(np.diag(XtXi))
    return B, sigma, su, df


def _voom_weights(counts, X, lib_size, span=0.5, block=None, correlation=None):
    y = np.log2((counts + 0.5) / (lib_size + 1) * 1e6)
    if block is None:
        B, sigma, _, df = _lmfit_ols(y, X)
    else:
        B, sigma, _, df = _lmfit_gls(y, X, block, correlation)
    amean = y.mean(axis=1)
    sx = amean + np.mean(np.log2(lib_size + 1)) - np.log2(1e6)
    sy = np.sqrt(sigma)
    allzero = counts.sum(axis=1) == 0
    if allzero.any():
        sx, sy = sx[~allzero], sy[~allzero]
    l = _lowess(sx, sy, span)
    xs, ys = l[:, 0], l[:, 1]
    # approxfun(l, rule = 2, ties = mean)
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.bincount(inv, weights=ys) / np.bincount(inv)
    fitted = B @ X.T
    fitted_count = 1e-6 * 2 ** fitted * (lib_size + 1)
    fl = np.log2(fitted_count)
    w = 1 / np.interp(fl, ux, uy) ** 4
    return y, w


def duplicate_correlation(y, weights, design, block, trim: float = 0.15) -> float:
    """limma::duplicateCorrelation(..., block) consensus correlation via per-gene REML (statmod mixedModel2Fit model:
    whitened var = s_e^2 I + s_a^2 ZZ').  The REML objective is profiled over rho = s_a^2/(s_e^2+s_a^2) and maximised
    over (1/(1-maxblock), 0.999); per-gene values are then clipped to [1/(1-maxblock)+0.01, 0.99] as in limma.

    Used inside :func:`voom` (``block=``); call it directly only to inspect the correlation.

    Parameters
    ----------
    y : genes x samples log-expression (voom ``E``).
    weights : genes x samples precision weights (voom weights).
    design : samples x p design matrix.
    block : per-sample block labels (e.g. patient).
    trim : trimmed-mean fraction for the consensus.

    Returns
    -------
    float consensus within-block correlation (0.0 when no block has more than one sample).

    Examples
    --------
    >>> import arsc, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> pat = rng.normal(0, 1, size=(100, 4))
    >>> y = np.repeat(pat, 2, axis=1) + rng.normal(0, 1, size=(100, 8))
    >>> design = np.column_stack([np.ones(8), np.tile([0, 1], 4)])
    >>> rho = arsc.de.duplicate_correlation(y, np.ones_like(y), design, np.repeat(["a", "b", "c", "d"], 2))
    >>> 0.2 < rho < 0.8
    True
    """
    block = np.asarray(block)
    lv, codes = np.unique(block, return_inverse=True)
    n, p = design.shape
    Z = np.eye(len(lv))[codes]
    ZZ = Z @ Z.T
    maxb = np.bincount(codes).max()
    if maxb == 1:
        return 0.0
    rhomin, rhomax = 1 / (1 - maxb) + 0.01, 0.99
    G = y.shape[0]
    rho = np.full(G, np.nan)
    if len(lv) <= 1 or len(lv) >= n - 1 or n <= p + 2:
        return 0.0

    # statmod::mixedModel2Fit: y and X are scaled by sqrt(w) but Z is not, i.e. in the whitened scale
    # var = s_e^2 I + s_a^2 Z Z'  (the same V = W^-1/2 C W^-1/2 that gls.series uses).
    I = np.eye(n)
    lo = 1 / (1 - maxb) + 1e-4   # negative correlations are allowed (statmod does not constrain the block variance >= 0)

    def nll(r, yg, Xg):
        V = (1 - r) * I + r * ZZ
        Lc = np.linalg.cholesky(V)
        XtViX = Xg.T @ np.linalg.solve(V, Xg)
        b = np.linalg.solve(XtViX, Xg.T @ np.linalg.solve(V, yg))
        rr = yg - Xg @ b
        q = rr @ np.linalg.solve(V, rr)
        return 2 * np.log(np.diag(Lc)).sum() + np.linalg.slogdet(XtViX)[1] + (n - p) * np.log(q)

    for g in range(G):
        sw = np.sqrt(weights[g])
        yg, Xg = sw * y[g], sw[:, None] * design
        try:
            res = minimize_scalar(nll, bounds=(lo, 0.999), args=(yg, Xg), method="bounded",
                                  options=dict(xatol=1e-7))
            rho[g] = res.x
        except np.linalg.LinAlgError:
            pass
    rho = np.where(np.isfinite(rho), np.clip(rho, rhomin, rhomax), np.nan)
    arho = np.arctanh(rho[np.isfinite(rho)])
    return float(np.tanh(sps.trim_mean(arho, trim)))


def _trigamma_inverse(x):
    x = float(x)
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if -dif / y < 1e-8:
            break
    return y


def _fit_fdist(x, df1):
    """limma::fitFDist (no covariate, non-robust): returns (scale s0^2, df2)."""
    ok = np.isfinite(x) & np.isfinite(df1) & (df1 > 1e-15)
    x, df1 = x[ok], df1[ok]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), 0.0
    m = np.median(x)
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)
    z = np.log(x)
    e = z - special.digamma(df1 / 2) + np.log(df1 / 2)
    emean = e.mean()
    evar = ((e - emean) ** 2).sum() / (n - 1) - np.mean(special.polygamma(1, df1 / 2))
    if evar > 0:
        df2 = 2 * _trigamma_inverse(evar)
        s20 = float(np.exp(emean + special.digamma(df2 / 2) - np.log(df2 / 2)))
    else:
        df2, s20 = np.inf, float(np.exp(emean))
    return s20, df2


def voom(counts: pd.DataFrame, coldata: pd.DataFrame, formula: str, coef, block: str | None = None,
         min_count: float = 10, min_total_count: float = 15, filter: bool = True) -> pd.DataFrame:
    """limma-voom DE on pseudobulk counts, with optional duplicateCorrelation for repeated patients.

    Pipeline (as in R): filterByExpr -> TMM -> voom -> duplicateCorrelation -> voom(block, cor)
    -> lmFit(block, cor) -> eBayes -> topTable. Parity with R limma: logFC / t r = 1.00000, max |diff| < 1e-3
    (STATS_PARITY.md).

    Parameters
    ----------
    counts : genes x samples counts (``Pseudobulk.to_frame()``).
    coldata : sample table indexed by sample name (the columns of ``counts``), e.g. ``Pseudobulk.groups``.
        The reference level of a factor is its first level: string columns sort alphabetically
        (``"post"`` < ``"pre"``), so set ``coldata[var] = pd.Categorical(coldata[var], [reference, level])``.
    formula : patsy / R model.matrix formula, e.g. ``"~ timepoint"`` or ``"~ patient + timepoint"`` (paired).
    coef : design column name, or ``(variable, level)`` -> ``"variable[T.level]"``.
    block : column with the repeated-measure unit (patient), for duplicateCorrelation. As in R, the block
        factor must NOT also be in the design (else the correlation is 0).
    min_count, min_total_count : filterByExpr thresholds.
    filter : apply filterByExpr.

    Returns
    -------
    DataFrame ``logFC, AveExpr, t, P.Value, adj.P.Val, gene`` sorted by P.Value; ``.attrs`` holds
    ``consensus_cor``, ``df_prior``, ``s2_prior``, ``df_residual``, ``design_columns``, ``n_genes``.

    Examples
    --------
    >>> import arsc, pandas as pd
    >>> ds = arsc.datasets.toy()
    >>> m = ((ds.obs.cell_type == "CD8 T") & (ds.obs.response == "R")).to_numpy()
    >>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], ds.obs[m], ["patient", "timepoint"],
    ...                      features=ds.features("RNA"), min_cells=10)
    >>> cd = pb.groups.copy()
    >>> cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])   # reference level first
    >>> res = arsc.de.voom(pb.to_frame(), cd, "~ timepoint", coef=("timepoint", "post"), block="patient")
    >>> sorted(res.gene[:5])                    # the planted responder genes
    ['CXCL13', 'GZMB', 'GZMK', 'IFNG', 'PRF1']
    """
    import patsy
    cd = _align_coldata(counts, coldata)
    _check_raw_counts(counts, "de.voom", strict=False)
    miss = [v for v in _vars_in(formula) if v not in cd.columns]
    if miss:
        raise KeyError(f"formula variable(s) {miss} not in coldata; available: {list(cd.columns)}")
    if block is not None and block not in cd.columns:
        raise KeyError(f"block column {block!r} not in coldata; available: {list(cd.columns)}")
    dm = patsy.dmatrix(formula, cd, return_type="dataframe")
    X = dm.values
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(f"design {formula!r} is not full rank (columns {list(dm.columns)}): some terms are "
                         "confounded, e.g. a patient-level variable (response) together with patient. Drop one "
                         "of them, or use block= for the patient instead of a fixed effect")
    cnt = counts.values.astype(float)
    genes = counts.index.to_numpy()
    if filter:
        keep = filter_by_expr(cnt, X, min_count, min_total_count)
        cnt, genes = cnt[keep], genes[keep]
    lib = cnt.sum(axis=0)
    nf = tmm_factors(cnt)
    eff = lib * nf
    y, w = _voom_weights(cnt, X, eff)
    if block is None:
        rho, bl = 0.0, None
        cor_attr = None
    else:
        bl = cd[block].astype(str).values
        rho = duplicate_correlation(y, w, X, bl)
        cor_attr = rho
        y, w = _voom_weights(cnt, X, eff, block=bl, correlation=rho)
    if bl is None:
        # weighted OLS == gls with C = I
        B, sigma, su, df = _lmfit_gls(y, X, np.arange(X.shape[0]), 0.0, w)
    else:
        B, sigma, su, df = _lmfit_gls(y, X, bl, rho, w)
    # eBayes (squeezeVar, no trend / robust)
    s2 = sigma ** 2
    s20, d0 = _fit_fdist(s2, np.full(len(s2), float(df)))
    if np.isinf(d0):
        s2post = np.full_like(s2, s20)
    else:
        s2post = (d0 * s20 + df * s2) / (d0 + df)
    if isinstance(coef, tuple):
        coef = f"{coef[0]}[T.{coef[1]}]"
    if coef not in dm.columns:
        raise ValueError(f"coef {coef!r} is not a design column; design columns are {list(dm.columns)}. "
                         "The reference level is the first level of the factor (strings sort alphabetically); "
                         "set coldata[var] = pd.Categorical(coldata[var], categories=[reference, level])")
    j = list(dm.columns).index(coef)
    dtot = min(df + d0, len(s2) * df)
    t = B[:, j] / (np.sqrt(s2post) * su[:, j])
    p = 2 * sps.t.sf(np.abs(t), dtot)
    tt = pd.DataFrame({"logFC": B[:, j], "AveExpr": y.mean(axis=1), "t": t, "P.Value": p,
                       "adj.P.Val": bh(p), "gene": genes})
    tt = tt.sort_values("P.Value").reset_index(drop=True)
    tt.attrs.update(consensus_cor=cor_attr, df_prior=float(d0), s2_prior=float(s20), df_residual=int(df),
                    design_columns=list(dm.columns), n_genes=len(genes))
    return tt


# ------------------------------------------------------------------ fgsea (simple, sampled null)
@nb.njit(cache=True)
def _splitmix(state):
    state = (state + np.uint64(0x9E3779B97F4A7C15))
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return state, z ^ (z >> np.uint64(31))


@nb.njit(cache=True)
def _es(pos, absw, N, k):
    """fgsea calcGseaStat: running-sum maximum deviation for sorted hit positions (0-based)."""
    nr = 0.0
    for i in range(k):
        nr += absw[pos[i]]
    cum = 0.0
    top = -np.inf
    bot = np.inf
    ti = 0
    bi = 0
    for i in range(k):
        miss = (pos[i] - i) / (N - k)
        b = cum / nr - miss
        cum += absw[pos[i]]
        t = cum / nr - miss
        if t > top:
            top = t
            ti = i
        if b < bot:
            bot = b
            bi = i
    if top > -bot:
        return top, ti
    return bot, bi


@nb.njit(parallel=True, cache=True)
def _null_es(absw, N, k, nperm, seed):
    out = np.empty(nperm)
    for it in nb.prange(nperm):
        st = np.uint64(seed) ^ (np.uint64(it + 1) * np.uint64(0x2545F4914F6CDD1D))
        mark = np.zeros(N, np.uint8)
        pos = np.empty(k, np.int64)
        c = 0
        while c < k:
            st, r = _splitmix(st)
            j = np.int64(r % np.uint64(N))
            if mark[j] == 0:
                mark[j] = 1
                pos[c] = j
                c += 1
        pos.sort()
        out[it] = _es(pos, absw, N, k)[0]
    return out


@records_seed("seed")
def fgsea(stats: pd.Series, genesets: dict, min_size: int = 10, max_size: int = 500, nperm: int = 10000,
          gsea_param: float = 1.0, seed: int = 1) -> pd.DataFrame:
    """Preranked GSEA following fgsea.  ES, leading edge and the NES definition (ES / mean of the
    same-sign null) follow fgsea exactly; p-values come from ``nperm`` sampled gene sets per set size
    (fgseaSimple), so the smallest attainable p is 1/(nperm+1) -- fgsea >= 1.14 default (multilevel) resolves
    far smaller p.  Columns: pathway, pval, padj, ES, NES, nMoreExtreme, size, leadingEdge (comma-joined).
    ``seed`` (default 1): the null for set size k uses permutation streams seeded from ``(seed, k, i)``
    (splitmix64, one independent stream per permutation), so results are bit-identical regardless of the
    number of threads.

    Parameters
    ----------
    stats : gene-level ranking statistic, a Series indexed by gene (e.g. voom ``t`` or DESeq2 ``stat``).
    genesets : ``{name: [genes]}`` (e.g. :func:`msigdb`, :func:`arsc.genesets.read_gmt`).
    min_size, max_size : gene-set size limits after intersecting with ``stats``.
    nperm : permutations per set size.
    gsea_param : weight exponent (fgsea ``gseaParam``).
    seed : integer seed.

    Returns
    -------
    DataFrame ``pathway, pval, padj, ES, NES, nMoreExtreme, size, leadingEdge`` sorted by pval
    (empty if no set passes the size limits).

    Examples
    --------
    >>> import arsc, pandas as pd
    >>> ds = arsc.datasets.toy()
    >>> m = ((ds.obs.cell_type == "CD8 T") & (ds.obs.response == "R")).to_numpy()
    >>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], ds.obs[m], ["patient", "timepoint"],
    ...                      features=ds.features("RNA"), min_cells=10)
    >>> cd = pb.groups.copy()
    >>> cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])   # reference level first
    >>> res = arsc.de.voom(pb.to_frame(), cd, "~ timepoint", coef=("timepoint", "post"), block="patient")
    >>> sets = {"effect": arsc.datasets.TOY_EFFECT_GENES, "B": ["MS4A1", "CD79A", "CD19"]}
    >>> fg = arsc.de.fgsea(res.set_index("gene")["t"], sets, min_size=3)
    >>> fg.pathway.tolist()[0], bool(fg.NES.iloc[0] > 0)
    ('effect', True)
    """
    if not isinstance(stats, pd.Series):
        raise TypeError("stats must be a pandas Series indexed by gene, e.g. res.set_index('gene')['t']")
    seed = check_seed(seed)
    s = stats[np.isfinite(stats)].sort_values(ascending=False, kind="stable")
    names = s.index.to_numpy()
    idx = {g: i for i, g in enumerate(names)}
    absw = np.abs(s.values) ** gsea_param
    N = len(s)
    rows, nulls = [], {}
    for pw, genes in genesets.items():
        pos = np.sort(np.array([idx[g] for g in set(genes) if g in idx], dtype=np.int64))
        k = len(pos)
        if k < min_size or k > max_size or k >= N:
            continue
        es, ei = _es(pos, absw, N, k)
        if es >= 0:
            le = pos[: ei + 1]
        else:
            le = pos[ei:]
        if k not in nulls:
            nulls[k] = _null_es(absw, N, k, nperm, seed * 1000003 + k)
        nu = nulls[k]
        if es > 0:
            m = (nu >= es).sum()
            pval = (m + 1) / ((nu > 0).sum() + 1)
            nes = es / nu[nu > 0].mean()
        else:
            m = (nu <= es).sum()
            pval = (m + 1) / ((nu < 0).sum() + 1)
            nes = -es / nu[nu < 0].mean()
        rows.append(dict(pathway=pw, pval=min(pval, 1.0), ES=es, NES=nes, nMoreExtreme=int(m), size=k,
                         leadingEdge=",".join(names[le])))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["padj"] = bh(out["pval"].values)
    out = out[["pathway", "pval", "padj", "ES", "NES", "nMoreExtreme", "size", "leadingEdge"]]
    return out.sort_values("pval").reset_index(drop=True)


# ------------------------------------------------------------------ MSigDB (msigdbr via Rscript, cached)
def msigdb(collection: str = "H", subcollection: str | None = None, species: str = "Homo sapiens",
           as_frame: bool = False, refresh: bool = False):
    """MSigDB gene sets as ``{gs_name: [symbols]}``, exported once from R msigdbr and cached.

    The first call runs msigdbr (Rscript, ``arsc/R/msigdb_export.R``) and writes
    ``<cache>/genesets/<species>__<coll>__<sub>.csv.gz``, where ``<cache>`` is ``$ARSC_CACHE``, else
    ``$AR_ROOT/.cache``, else ``~/.cache/arsc``; later calls read the cache (no R needed).

    Parameters
    ----------
    collection : MSigDB collection, e.g. ``"H"`` (hallmark), ``"C2"``, ``"C7"``.
    subcollection : e.g. ``"CP:REACTOME"``.
    species : msigdbr species name.
    as_frame : return the long table instead of a dict.
    refresh : re-export even if cached.

    Returns
    -------
    dict ``{gs_name: [gene symbols]}`` (or a DataFrame with ``as_frame=True``).

    Examples
    --------
    >>> import arsc
    >>> hallmark = arsc.de.msigdb("H")                       # doctest: +SKIP
    >>> len(hallmark)                                        # doctest: +SKIP
    50
    """
    GENESET_CACHE.mkdir(parents=True, exist_ok=True)
    tag = "__".join([species.replace(" ", "_"), collection, (subcollection or "all").replace(":", "-")])
    f = GENESET_CACHE / f"{tag}.csv.gz"
    if refresh or not f.exists():
        script = Path(__file__).with_name("R") / "msigdb_export.R"
        cmd = [_RSCRIPT, str(script), str(f), species, collection, subcollection or ""]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not f.exists():
            raise RuntimeError(f"msigdbr export failed:\n{r.stderr[-2000:]}")
    d = pd.read_csv(f)
    if as_frame:
        return d
    return {k: g["gene_symbol"].tolist() for k, g in d.groupby("gs_name", sort=False)}
