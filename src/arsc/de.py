"""Pseudobulk differential expression + gene-set enrichment (port of tools/R/ar.R DE layer).

    pb = arsc.pseudobulk(C, obs[obs.cell_type == "gd"], ["patient_alias", "timepoint"], features=feats, min_cells=20)
    counts, cd = pb.to_frame(), pb.groups                     # genes x samples, sample table
    r1 = de.deseq2(counts, cd, "~ patient_alias + timepoint", ("timepoint", "Follow Up 1", "Baseline"))   # DESeq2 (pydeseq2)
    r2 = de.voom(counts, cd, "~ timepoint", block="patient_alias", coef=("timepoint", "Follow Up 1"))       # voom + dupCor
    gs = de.msigdb("H")
    fg = de.fgsea(r1.set_index("gene")["stat"], gs)                                                       # fgsea (simple)

* ``deseq2``   -> pydeseq2 (independent re-implementation of DESeq2 1.x; no LFC shrinkage, Wald test, alpha = 0.1)
* ``voom``     -> own port of edgeR filterByExpr/calcNormFactors(TMM) + limma voom, duplicateCorrelation (REML),
                  gls lmFit, eBayes (fitFDist/squeezeVar) and topTable
* ``fgsea``    -> own port of fgsea's ES / NES / leading edge with sampled nulls (fgseaSimple); numba kernels
* ``msigdb``   -> msigdbr export (Rscript, once) cached under <AR_ROOT>/.cache/genesets/

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

from .stats import bh

__all__ = ["deseq2", "voom", "fgsea", "msigdb", "filter_by_expr", "tmm_factors", "duplicate_correlation"]

AR_ROOT = Path(os.environ.get("AR_ROOT", "E:/autoresearch"))
GENESET_CACHE = AR_ROOT / ".cache" / "genesets"
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
    """Wald-test DE on pseudobulk (== ar_de_deseq2).

    counts   genes x samples integer matrix (as produced by ``Pseudobulk.to_frame()``).
    coldata  sample table, index = sample names (columns of ``counts``).
    design   e.g. ``"~ patient_alias + timepoint"`` (paired).
    contrast ``(variable, level, reference)``.
    Genes with fewer than ``min_samples`` samples at count >= ``min_count`` are removed first (as the R helper).
    ``alpha`` is the independent-filtering / padj target (DESeq2 ``results()`` default 0.1).
    Returns a frame with R's columns: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, gene (sorted by pvalue).
    """
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    cd = coldata.loc[counts.columns].copy()
    keep = (counts.values >= min_count).sum(axis=1) >= min_samples
    cnt = counts.loc[keep].round().astype(int)
    for v in _vars_in(design):
        if v in cd and (cd[v].dtype == object or str(cd[v].dtype) in ("category", "string")):
            cd[v] = cd[v].astype(str).map(_make_names).astype("category")
    var, lvl, ref = contrast
    contrast_m = [var, _make_names(lvl), _make_names(ref)]
    cd = cd[[c for c in cd.columns if c in _vars_in(design)]]
    inf = DefaultInference(n_cpus=n_cpus)
    dds = _desq2_dataset()(counts=cnt.T, metadata=cd, design=design, refit_cooks=True, inference=inf, quiet=quiet)
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
    """edgeR::filterByExpr(DGEList(counts), design = ...) (group = NULL branch)."""
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
    """edgeR::calcNormFactors(method = 'TMM') (weighted trimmed mean of M-values, upper-quartile reference)."""
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
    over (1/(1-maxblock), 0.999); per-gene values are then clipped to [1/(1-maxblock)+0.01, 0.99] as in limma."""
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
    """limma-voom with optional duplicateCorrelation (== ar_de_voom).

    counts   genes x samples; coldata indexed by sample.  ``formula`` e.g. ``"~ timepoint"`` (patsy / R model.matrix
    treatment coding).  ``block``: column with the repeated-measure unit (patient); note that, as in R, the block
    factor must NOT be fully encoded in the design (else the correlation is 0).
    ``coef``: column name of the design, or ``(variable, level)`` -> ``variable[T.level]``.
    Pipeline (identical to the R helper): filterByExpr -> TMM -> voom -> duplicateCorrelation -> voom(block, cor)
    -> lmFit(block, cor) -> eBayes -> topTable.
    Returns logFC, AveExpr, t, P.Value, adj.P.Val, gene (sorted by P.Value); ``.attrs['consensus_cor']``.
    """
    import patsy
    cd = coldata.loc[counts.columns]
    dm = patsy.dmatrix(formula, cd, return_type="dataframe")
    X = dm.values
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


def fgsea(stats: pd.Series, genesets: dict, min_size: int = 10, max_size: int = 500, nperm: int = 10000,
          gsea_param: float = 1.0, seed: int = 1) -> pd.DataFrame:
    """Preranked GSEA following fgsea (== ar_fgsea).  ES, leading edge and the NES definition (ES / mean of the
    same-sign null) follow fgsea exactly; p-values come from ``nperm`` sampled gene sets per set size
    (fgseaSimple), so the smallest attainable p is 1/(nperm+1) -- fgsea >= 1.14 default (multilevel) resolves
    far smaller p.  Columns: pathway, pval, padj, ES, NES, nMoreExtreme, size, leadingEdge (comma-joined).
    """
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
    """MSigDB gene sets as ``{gs_name: [symbols]}`` (== ar_msigdb).  The first call exports the collection with
    msigdbr (Rscript, ``arsc/R/msigdb_export.R``) to ``<AR_ROOT>/.cache/genesets/<species>__<coll>__<sub>.csv.gz``;
    later calls read the cache (no R needed)."""
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
