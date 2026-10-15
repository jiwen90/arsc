"""Patient-level statistics for small-n clinical single-cell data (Python port of R helper functions).

The unit of inference is the patient (sample).  Everything here operates on per-sample tables, never on cells.

    comp = stats.composition(obs, "cell_type", ["patient", "timepoint"])
    stats.paired_test(comp, "prop", t0="pre", t1="post", patient="patient", by="celltype", transform="logit")
    stats.glmm_prop(comp[comp.celltype == "CD8 T"], fixed="timepoint", groups=("patient", "sample_id"))
    #   == glmer(cbind(k, n-k) ~ timepoint + (1|patient) + (1|sample))

Several functions default to a patient column named ``"patient_alias"``; pass your own column name
(``patient=``, ``sample_cols=``, ``groups=``, ``group=``).

Differences from R are listed in STATS_PARITY.md.  Highlights:
  * ``wilcoxon_signed_rank`` reproduces ``stats::wilcox.test`` (exact for n < 50 without ties/zeros, otherwise the
    normal approximation with tie + continuity correction); scipy switches to the exact test only for n <= 25.
  * ``glmm_prop`` is an own Laplace (nAGQ = 1) ML implementation of lme4::glmer for binomial random-intercept models.
  * bootstrap CIs use numpy's RNG, so they agree with R only up to Monte-Carlo error.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps

from ._utils import check_seed

__all__ = ["composition", "sample_means", "bh", "boot_ci", "wilcoxon_signed_rank", "paired_test", "perm_test",
           "arm_contrast", "glmm_prop", "gee_prop", "lmm_score", "mde", "feasibility", "count_logit_delta",
           "loo_effect"]


def _need_cols(df, cols, what: str, args: str) -> None:
    """Raise a KeyError naming the missing columns and the available ones."""
    missing = [c for c in dict.fromkeys(cols) if c not in df.columns]
    if missing:
        raise KeyError(f"column(s) {missing} not in {what}; available: {list(df.columns)}. "
                       f"Check the {args} arguments (several default to 'patient_alias').")


# ------------------------------------------------------------------ aggregation
def composition(obs: pd.DataFrame, celltype: str, sample_cols=("patient_alias", "timepoint"),
                min_parent: int = 50, extra_cols=None) -> pd.DataFrame:
    """Per-sample counts and proportions of every level of a cell-type column.

    Zeros are filled (a cell type absent from a sample gets ``k = 0``); samples whose parent total ``n``
    is below ``min_parent`` are dropped (a message says how many).

    Parameters
    ----------
    obs : cell metadata (one row per cell).
    celltype : column with the cell-type labels.
    sample_cols : columns defining a sample (default ``("patient_alias", "timepoint")``; pass yours).
    min_parent : minimum cells per sample.
    extra_cols : sample-level columns carried along (e.g. ``["response"]``).

    Returns
    -------
    DataFrame with the key columns, ``celltype``, ``k`` (cells of that type), ``n`` (cells in the
    sample) and ``prop`` = k / n; one row per sample x cell type.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
    >>> cd8 = comp[comp.celltype == "CD8 T"]
    >>> list(comp.columns)
    ['patient', 'timepoint', 'response', 'celltype', 'k', 'n', 'prop']
    >>> len(cd8)                                     # 6 patients x 2 timepoints
    12
    """
    keys = list(sample_cols) + list(extra_cols or [])
    _need_cols(obs, keys + [celltype], "obs", "sample_cols / extra_cols / celltype")
    o = obs[keys + [celltype]].copy()
    o[celltype] = o[celltype].astype(str)
    o = o.dropna(subset=keys)
    tot = o.groupby(keys, observed=True, sort=True).size().rename("n").reset_index()
    k = o.groupby(keys + [celltype], observed=True).size().rename("k")
    k = k.unstack(celltype, fill_value=0).stack().rename("k").reset_index()
    out = k.merge(tot, on=keys)
    out["prop"] = out["k"] / out["n"]
    dropped = int((tot["n"] < min_parent).sum())
    if dropped:
        print(f"[arsc] composition: {dropped} samples below parent floor ({min_parent} cells) dropped")
    out = out[out["n"] >= min_parent].rename(columns={celltype: "celltype"})
    return out.sort_values(keys + ["celltype"]).reset_index(drop=True)


def sample_means(obs: pd.DataFrame, cols, sample_cols=("patient_alias", "timepoint"), by=None, min_cells: int = 10):
    """Per-sample means of numeric per-cell columns (e.g. a module score), with a cell floor.

    Parameters
    ----------
    obs : cell metadata with the numeric columns.
    cols : numeric columns to average.
    sample_cols : columns defining a sample (default ``("patient_alias", "timepoint")``; pass yours).
    by : optional extra grouping column(s) (e.g. ``"cell_type"``).
    min_cells : drop samples with fewer cells.

    Returns
    -------
    DataFrame, one row per sample (x ``by``): keys, the means of ``cols`` and ``n_cells``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> obs = ds.obs.assign(nUMI=ds.X("RNA", "counts").sum(axis=1).A1)
    >>> sm = arsc.stats.sample_means(obs, ["nUMI"], ["patient", "timepoint"], by="cell_type")
    >>> list(sm.columns)
    ['patient', 'timepoint', 'cell_type', 'nUMI', 'n_cells']
    """
    keys = list(sample_cols) + ([by] if isinstance(by, str) else list(by or []))
    _need_cols(obs, keys + list(cols), "obs", "cols / sample_cols / by")
    g = obs.groupby(keys, observed=True)
    df = g[list(cols)].mean()
    df["n_cells"] = g.size()
    return df[df["n_cells"] >= min_cells].reset_index()


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN preserved (== R ``p.adjust(method = "BH")``).

    Parameters
    ----------
    p : array-like of p-values.

    Returns
    -------
    numpy.ndarray of adjusted p-values (same shape; NaN where ``p`` is NaN).

    Examples
    --------
    >>> import arsc
    >>> arsc.stats.bh([0.01, 0.04, 0.03, 0.5]).round(4).tolist()
    [0.04, 0.0533, 0.0533, 0.5]
    """
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    n = len(q)
    o = np.argsort(q)
    adj = np.minimum.accumulate((q[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    r = np.empty(n)
    r[o] = np.minimum(adj, 1)
    out[ok] = r
    return out


# ------------------------------------------------------------------ paired / group tests
def _tf(x, transform, eps=1e-3):
    x = np.asarray(x, float)
    if transform == "logit":
        x = np.clip(x, eps, 1 - eps)
        return np.log(x / (1 - x))
    if transform == "log2ratio":
        return np.log2(x + eps)
    if transform in ("none", None):
        return x
    raise ValueError(f"transform must be 'none', 'logit' or 'log2ratio'; got {transform!r}")


def boot_ci(x, f=np.median, B: int = 4000, seed: int = 1):
    """Percentile bootstrap 95% CI of a statistic of per-patient values (NaN when < 3 finite values).

    Parameters
    ----------
    x : per-patient values.
    f : statistic (default ``np.median``).
    B : bootstrap resamples.
    seed : seeds a local ``np.random.default_rng`` for the ``B`` resamples (default 1).

    Returns
    -------
    tuple ``(lo, hi)``.

    Examples
    --------
    >>> import arsc
    >>> lo, hi = arsc.stats.boot_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> float(lo), float(hi)
    (1.0, 5.0)
    """
    seed = check_seed(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    b = f(x[idx], axis=1) if f in (np.median, np.mean) else np.array([f(x[i]) for i in idx])
    return tuple(np.quantile(b, [0.025, 0.975]))


def _psignrank_counts(n: int) -> np.ndarray:
    """Number of subsets of {1..n} with each rank-sum 0..n(n+1)/2 (null distribution of the signed-rank statistic)."""
    top = n * (n + 1) // 2
    c = np.zeros(top + 1)
    c[0] = 1.0
    for i in range(1, n + 1):
        c[i:] = c[i:] + c[:top + 1 - i].copy()
    return c


def wilcoxon_signed_rank(x, exact: bool | None = None, correct: bool = True) -> float:
    """Two-sided one-sample / paired Wilcoxon signed-rank p-value replicating R ``wilcox.test(x, exact=...)``.

    Zeros are dropped; ``exact`` defaults to n < 50 and is only honoured without zeros/ties (R falls back to the
    normal approximation with continuity correction and tie-corrected variance otherwise).

    Parameters
    ----------
    x : per-patient paired differences (post - pre).
    exact : exact null distribution (default: n < 50).
    correct : continuity correction for the normal approximation.

    Returns
    -------
    float p-value (NaN if no non-zero observations remain). With n patients the smallest possible
    exact p is 2 / 2**n (0.03125 for n = 6).

    Examples
    --------
    >>> import arsc
    >>> arsc.stats.wilcoxon_signed_rank([0.5, 1.2, 0.8, 2.0, 1.1, 0.3])
    0.03125
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    zeros = bool(np.any(x == 0))
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return float("nan")
    if exact is None:
        exact = n < 50
    r = sps.rankdata(np.abs(x))
    ties = len(np.unique(r)) != len(r)
    stat = float(r[x > 0].sum())
    if exact and not ties and not zeros:
        c = _psignrank_counts(n)
        tot = c.sum()
        cdf = np.cumsum(c) / tot
        s = int(round(stat))
        lower = cdf[s]
        upper = 1.0 - (cdf[s - 1] if s > 0 else 0.0)
        return float(min(2 * min(lower, upper), 1.0))
    z = stat - n * (n + 1) / 4
    _, tc = np.unique(r, return_counts=True)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(tc ** 3 - tc) / 48)
    corr = (np.sign(z) * 0.5) if correct else 0.0
    z = (z - corr) / sigma
    return float(min(2 * min(sps.norm.cdf(z), sps.norm.sf(z)), 1.0))


def paired_test(df: pd.DataFrame, value: str = "value", t0=None, t1=None, patient: str = "patient_alias",
                time: str = "timepoint", by=None, transform: str = "none", eps: float = 1e-3, B: int = 4000,
                seed: int = 1) -> pd.DataFrame:
    """Within-patient change from ``t0`` to ``t1`` on a per-sample table (the patient is the unit).

    Per group: number of patients with both timepoints, sign counts, median delta, bootstrap CI, exact
    Wilcoxon signed-rank p, leave-one-patient-out range of the median, minimum attainable p, BH fdr.

    Parameters
    ----------
    df : per-sample table (one row per patient x timepoint [x group]), e.g. from :func:`composition`.
    value : column with the per-sample value (e.g. ``"prop"``).
    t0, t1 : the two timepoint levels (values of ``time``), e.g. ``"pre"``, ``"post"``.
    patient : patient column (default ``"patient_alias"``; pass yours).
    time : timepoint column.
    by : optional grouping column(s); one result row per group (e.g. ``"celltype"``).
    transform : ``"none"``, ``"logit"`` (proportions) or ``"log2ratio"``, applied before differencing.
    eps : clipping / pseudo-count for the transforms.
    B : bootstrap resamples.
    seed : seeds the bootstrap CI; every group uses the same seed (default 1).

    Returns
    -------
    DataFrame, one row per group: ``group, n_patients, k_up, k_down, median_delta, ci_lo, ci_hi,
    loo_min, loo_max, p_wilcox_exact, min_possible_p, transform, fdr``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
    >>> cd8 = comp[comp.celltype == "CD8 T"]
    >>> res = arsc.stats.paired_test(cd8, "prop", t0="pre", t1="post", patient="patient",
    ...                              by="response", transform="logit")
    >>> res[["group", "n_patients", "min_possible_p"]].values.tolist()
    [['R', 3, 0.25], ['NR', 3, 0.25]]
    """
    seed = check_seed(seed)
    _need_cols(df, [value, patient, time] + ([by] if isinstance(by, str) else list(by or [])), "df",
               "value / patient / time / by")
    present = set(df[time].astype(str).unique())
    missing_tp = [t for t in (t0, t1) if str(t) not in present]
    if missing_tp:
        raise ValueError(f"timepoint level(s) {missing_tp} not found in column {time!r}; levels are "
                         f"{sorted(present)} (pass t0= and t1= as two of these)")
    groups = [("all", df)] if by is None else list(df.groupby(by, observed=True))
    rows = []
    for g, d in groups:
        a = d[d[time] == t0].set_index(patient)[value]
        b = d[d[time] == t1].set_index(patient)[value]
        common = a.index.intersection(b.index)
        delta = _tf(b.loc[common].values, transform, eps) - _tf(a.loc[common].values, transform, eps)
        delta = delta[np.isfinite(delta)]
        n = len(delta)
        p = np.nan
        if n >= 2 and np.any(delta != 0):
            p = wilcoxon_signed_rank(delta)
        loo = [np.median(np.delete(delta, i)) for i in range(n)] if n >= 3 else [np.nan]
        lo, hi = boot_ci(delta, B=B, seed=seed)
        rows.append(dict(group=g if not isinstance(g, tuple) else "|".join(map(str, g)), n_patients=n,
                         k_up=int((delta > 0).sum()), k_down=int((delta < 0).sum()),
                         median_delta=float(np.median(delta)) if n else np.nan, ci_lo=lo, ci_hi=hi,
                         loo_min=float(np.min(loo)), loo_max=float(np.max(loo)), p_wilcox_exact=p,
                         min_possible_p=2 / 2 ** n if n else np.nan, transform=transform))
    res = pd.DataFrame(rows)
    res["fdr"] = bh(res["p_wilcox_exact"].values)
    return res


def perm_test(x, y, stat=None, B: int = 20000, seed: int = 1) -> dict:
    """Two-group permutation test on per-patient values (e.g. responders vs non-responders).

    Exact enumeration when ``choose(n_x + n_y, n_x) <= B``, otherwise ``B`` random permutations.
    Two-sided: p = fraction of permutations with ``|stat| >= |observed|``.

    Parameters
    ----------
    x, y : per-patient values of the two groups (one value per patient, not per cell).
    stat : ``stat(a, b) -> float``; default difference of medians.
    B : enumeration limit / number of sampled permutations.
    seed : seeds a local ``np.random.default_rng`` for sampled permutations (unused when exact).

    Returns
    -------
    dict with ``n_x``, ``n_y``, ``stat`` (observed) and ``p_perm``. With 3 vs 3 patients the smallest
    possible p is 2/20 = 0.1 (ties of the median statistic can make it larger, as below).

    Examples
    --------
    >>> import arsc
    >>> r = arsc.stats.perm_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    >>> r["stat"], r["p_perm"]
    (-3.0, 0.2)
    """
    seed = check_seed(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    z = np.concatenate([x, y]); nx = len(x)
    default = stat is None
    if default:
        stat = lambda a, b: np.median(a) - np.median(b)  # noqa: E731
    obs = stat(x, y)
    if math.comb(len(z), nx) <= B:
        idxs = np.array(list(itertools.combinations(range(len(z)), nx)))
    else:
        rng = np.random.default_rng(seed)
        idxs = np.argsort(rng.random((B, len(z))), axis=1)[:, :nx]
    mask = np.zeros((len(idxs), len(z)), bool)
    np.put_along_axis(mask, idxs, True, axis=1)
    if default:
        zz = np.broadcast_to(z, mask.shape)
        a = np.where(mask, zz, np.nan); b = np.where(mask, np.nan, zz)
        null = np.nanmedian(a, axis=1) - np.nanmedian(b, axis=1)
    else:
        null = np.array([stat(z[m], z[~m]) for m in mask])
    return dict(n_x=nx, n_y=len(y), stat=float(obs), p_perm=float(np.mean(np.abs(null) >= abs(obs) - 1e-12)))


def arm_contrast(a, b, B: int = 20000, seed: int = 1) -> dict:
    """Between-arm comparison of per-patient deltas (e.g. post - pre in responders vs non-responders).

    Exact permutation on the median difference, Hodges-Lehmann shift and leave-one-patient-out range /
    sign reversals.

    Parameters
    ----------
    a, b : per-patient values (deltas) of the two arms.
    B : passed to :func:`perm_test`.
    seed : passed to :func:`perm_test` (used only when the permutation is sampled).

    Returns
    -------
    dict with ``n_a, n_b, median_a, median_b, diff_medians, hodges_lehmann, p_perm, loo_min, loo_max,
    loo_sign_reversals``.

    Examples
    --------
    >>> import arsc
    >>> r = arsc.stats.arm_contrast([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    >>> r["hodges_lehmann"], r["loo_sign_reversals"]
    (-3.0, 0)
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    pt = perm_test(a, b, B=B, seed=seed)
    hl = float(np.median(np.subtract.outer(a, b)))
    loo = [np.median(np.delete(a, i)) - np.median(b) for i in range(len(a))] + \
          [np.median(a) - np.median(np.delete(b, i)) for i in range(len(b))]
    return dict(n_a=len(a), n_b=len(b), median_a=float(np.median(a)), median_b=float(np.median(b)),
                diff_medians=pt["stat"], hodges_lehmann=hl, p_perm=pt["p_perm"],
                loo_min=float(min(loo)), loo_max=float(max(loo)),
                loo_sign_reversals=int(sum(np.sign(v) != np.sign(pt["stat"]) for v in loo)))


def loo_effect(df: pd.DataFrame, patient: str, fn) -> pd.Series:
    """Leave-one-patient-out robustness: ``fn(df without patient p)`` for each patient p.

    Parameters
    ----------
    df : per-sample table.
    patient : patient column.
    fn : ``fn(df_subset) -> float``, the effect estimate.

    Returns
    -------
    pandas.Series indexed by the left-out patient, named ``effect_without``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
    >>> cd8 = comp[comp.celltype == "CD8 T"]
    >>> loo = arsc.stats.loo_effect(cd8, "patient", lambda d: d.prop.mean())
    >>> loo.index.tolist()
    ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    """
    _need_cols(df, [patient], "df", "patient")
    return pd.Series({p: fn(df[df[patient] != p]) for p in df[patient].unique()}, name="effect_without")


# ------------------------------------------------------------------ binomial GLMM (glmer parity)
def _design(fixed: str, data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    import patsy
    X = patsy.dmatrix(fixed, data, return_type="dataframe")
    return X.values, list(X.columns)


def _laplace_fit(X, Z_list, y, n, vcov_method="hessian"):
    """ML fit of a binomial GLMM with independent random intercepts (Laplace, nAGQ = 1) -- same objective as
    lme4::glmer.  Z_list: list of (N x q_j) 0/1 indicator matrices.  Returns dict."""
    N, p = X.shape
    Z = np.hstack(Z_list)
    qs = [z.shape[1] for z in Z_list]
    col_grp = np.repeat(np.arange(len(qs)), qs)

    def inner(beta, sig):
        lam = sig[col_grp]
        ZL = Z * lam
        b = np.zeros(Z.shape[1])
        eta0 = X @ beta
        for _ in range(100):
            eta = eta0 + ZL @ b
            mu = 1 / (1 + np.exp(-eta))
            w = n * mu * (1 - mu)
            g = ZL.T @ (y - n * mu) - b
            H = ZL.T @ (w[:, None] * ZL) + np.eye(len(b))
            step = np.linalg.solve(H, g)
            # step-halving on the penalised log-likelihood
            f0 = _pen(eta, b, y, n)
            t = 1.0
            while t > 1e-6:
                bn = b + t * step
                if _pen(eta0 + ZL @ bn, bn, y, n) >= f0 - 1e-12:
                    break
                t /= 2
            b = b + t * step
            if np.max(np.abs(t * step)) < 1e-10:
                break
        eta = eta0 + ZL @ b
        mu = 1 / (1 + np.exp(-eta))
        w = n * mu * (1 - mu)
        H = ZL.T @ (w[:, None] * ZL) + np.eye(len(b))
        return b, eta, mu, w, H, ZL

    def negll(par):
        beta, sig = par[:p], np.abs(par[p:])
        b, eta, mu, w, H, _ = inner(beta, sig)
        ll = _binll(eta, y, n) - 0.5 * b @ b
        return -(ll - 0.5 * np.linalg.slogdet(H)[1])

    # start: plain GLM
    import statsmodels.api as sm
    glm = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit()
    x0 = np.concatenate([glm.params, np.full(len(qs), 0.5)])
    r = optimize.minimize(negll, x0, method="BFGS", options=dict(gtol=1e-6))
    r = optimize.minimize(negll, r.x, method="Nelder-Mead",
                          options=dict(xatol=1e-7, fatol=1e-11, maxiter=800, maxfev=800,
                                       initial_simplex=r.x + np.vstack([np.zeros(len(r.x)), 1e-3 * np.eye(len(r.x))])))
    beta, sig = r.x[:p], np.abs(r.x[p:])
    b, eta, mu, w, H, ZL = inner(beta, sig)
    # lme4-style conditional covariance of beta: (X' (W^-1 + Z L L' Z')^-1 X)^-1
    XtWX = X.T @ (w[:, None] * X)
    XtWZ = X.T @ (w[:, None] * ZL)
    M = XtWX - XtWZ @ np.linalg.solve(H, XtWZ.T)
    vcov = np.linalg.inv(M)
    vcov_cond = vcov
    # lme4 (>= 1.1) reports vcov(beta) from the numerical Hessian of the Laplace deviance in (theta, beta) for GLMMs:
    # it includes variance-component uncertainty, hence is a little larger than the conditional RX^-1 version.
    if vcov_method == "hessian" and np.all(sig > 1e-6):
        x = np.concatenate([beta, sig])
        k = len(x)
        h = 1e-3 * np.maximum(1.0, np.abs(x)) * 0.1
        Hm = np.zeros((k, k))
        f0 = negll(x)
        for i in range(k):
            for j in range(i, k):
                ei, ej = np.zeros(k), np.zeros(k)
                ei[i], ej[j] = h[i], h[j]
                Hm[i, j] = Hm[j, i] = (negll(x + ei + ej) - negll(x + ei - ej) - negll(x - ei + ej) + negll(x - ei - ej)) / (4 * h[i] * h[j])
        try:
            Hm_inv = np.linalg.inv(2 * Hm)  # deviance = 2 * negll
            vh = 2 * Hm_inv[:p, :p]
            if np.all(np.diag(vh) > 0):
                vcov = vh
        except np.linalg.LinAlgError:
            pass
    from scipy.special import gammaln
    lchoose = float(np.sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)))  # binomial coefficient (R's logLik includes it)
    return dict(beta=beta, sigma=sig, vcov=vcov, vcov_cond=vcov_cond, logLik=-r.fun + lchoose, converged=bool(r.success), b=b)


def _binll(eta, y, n):
    # sum y*eta - n*log(1+exp(eta)) (binomial coefficient omitted: constant)
    return float(np.sum(y * eta - n * np.logaddexp(0, eta)))


def _pen(eta, b, y, n):
    return _binll(eta, y, n) - 0.5 * b @ b


def glmm_prop(comp: pd.DataFrame, fixed: str = "timepoint", groups=("patient_alias", "sample_id"),
              k: str = "k", n: str = "n", vcov_method: str = "hessian") -> pd.DataFrame:
    """Binomial GLMM  ``cbind(k, n-k) ~ <fixed> + (1|g1) + (1|g2) ...``  (== lme4::glmer, nAGQ = 1).

    ``groups`` are independent random-intercept factors; a missing ``sample_id`` is created as observation-level
    (one level per row), which is the overdispersion term of the R default.  Estimated by Laplace-approximated ML
    over (beta, sigma), so estimates / variance components match glmer to optimiser tolerance; Std. Error uses
    lme4's default Vcov (numerical Hessian of the deviance in (sigma, beta); ``vcov_method="conditional"`` gives
    the RX^-1 version).

    Parameters
    ----------
    comp : per-sample count table with ``k`` and ``n`` columns, e.g. one cell type of :func:`composition`.
    fixed : fixed-effect formula right-hand side (patsy), e.g. ``"timepoint"`` or ``"timepoint * response"``.
    groups : random-intercept columns (default ``("patient_alias", "sample_id")``; pass your patient column).
    k, n : count columns (successes, totals).
    vcov_method : ``"hessian"`` (lme4 default) or ``"conditional"``.

    Returns
    -------
    DataFrame of coefficients (``Estimate, Std. Error, z value, Pr(>|z|), term, OR``); ``.attrs`` carries
    ``sigma`` (dict of random-effect SDs), ``logLik`` and ``converged``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
    >>> cd8 = comp[comp.celltype == "CD8 T"]
    >>> fit = arsc.stats.glmm_prop(cd8, fixed="timepoint", groups=("patient", "sample_id"))
    >>> fit.term.tolist()
    ['Intercept', 'timepoint[T.post]']
    """
    _need_cols(comp, [k, n] + [g for g in groups if g != "sample_id"], "comp", "k / n / groups")
    d = comp.reset_index(drop=True).copy()
    if "sample_id" in groups and "sample_id" not in d:
        d["sample_id"] = np.arange(len(d))
    X, names = _design(fixed, d)
    Zs = [pd.get_dummies(d[g].astype(str)).values.astype(float) for g in groups]
    fit = _laplace_fit(X, Zs, d[k].values.astype(float), d[n].values.astype(float), vcov_method)
    se = np.sqrt(np.diag(fit["vcov"]))
    z = fit["beta"] / se
    out = pd.DataFrame({"Estimate": fit["beta"], "Std. Error": se, "z value": z,
                        "Pr(>|z|)": 2 * sps.norm.sf(np.abs(z)), "term": names})
    out["OR"] = np.exp(out["Estimate"])
    out.attrs["sigma"] = dict(zip(groups, fit["sigma"]))
    out.attrs["logLik"] = fit["logLik"]
    out.attrs["converged"] = fit["converged"]
    return out


def gee_prop(comp: pd.DataFrame, fixed: str = "timepoint", group: str = "patient_alias", k: str = "k",
             n: str = "n", cov_struct: str = "exchangeable") -> pd.DataFrame:
    """Marginal (population-averaged) alternative: binomial GEE with exchangeable working correlation and robust
    SEs.  Coefficients are marginal log-ORs, NOT the conditional glmer ones (they attenuate toward 0 with
    between-patient variance); use for few-cluster sanity checks only (sandwich SEs are anti-conservative at
    < ~10 clusters).

    Parameters
    ----------
    comp : per-sample count table with ``k`` and ``n`` columns.
    fixed : fixed-effect formula right-hand side (patsy).
    group : cluster (patient) column.
    k, n : count columns.
    cov_struct : statsmodels working correlation (``"exchangeable"``, ``"independence"``).

    Returns
    -------
    DataFrame of coefficients (``Estimate, Std. Error, z value, Pr(>|z|), term, OR``).

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
    >>> cd8 = comp[comp.celltype == "CD8 T"]
    >>> arsc.stats.gee_prop(cd8, fixed="timepoint", group="patient").term.tolist()
    ['Intercept', 'timepoint[T.post]']
    """
    _need_cols(comp, [k, n, group], "comp", "k / n / group")
    import statsmodels.api as sm
    d = comp.reset_index(drop=True)
    X, names = _design(fixed, d)
    m = sm.GEE(d[k].values / d[n].values, X, groups=d[group].values, family=sm.families.Binomial(),
               cov_struct=getattr(sm.cov_struct, cov_struct.capitalize())(), weights=d[n].values).fit()
    out = pd.DataFrame({"Estimate": m.params, "Std. Error": m.bse, "z value": m.tvalues,
                        "Pr(>|z|)": m.pvalues, "term": names})
    out["OR"] = np.exp(out["Estimate"])
    return out


def lmm_score(df: pd.DataFrame, formula: str, group: str = "patient_alias", reml: bool = True) -> pd.DataFrame:
    """Linear mixed model ``score ~ time + (1|patient)`` (== lme4::lmer; statsmodels MixedLM REML).  Coefficients,
    SEs and variance components agree with lmer; statsmodels reports Wald z (lmerTest's Satterthwaite df/p are not
    reproduced).

    Parameters
    ----------
    df : per-sample (or per-cell) table.
    formula : statsmodels formula, e.g. ``"score ~ timepoint"``.
    group : random-intercept column (default ``"patient_alias"``; pass yours).
    reml : REML (True, lmer default) or ML.

    Returns
    -------
    DataFrame of fixed effects (``Estimate, Std. Error, z value, Pr(>|z|), term``); ``.attrs`` carries
    ``sigma_group`` and ``sigma_resid``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> obs = ds.obs.assign(nUMI=ds.X("RNA", "counts").sum(axis=1).A1)
    >>> sm = arsc.stats.sample_means(obs, ["nUMI"], ["patient", "timepoint"])
    >>> arsc.stats.lmm_score(sm, "nUMI ~ timepoint", group="patient").term.tolist()
    ['Intercept', 'timepoint[T.post]']
    """
    _need_cols(df, [group], "df", "group")
    import statsmodels.formula.api as smf
    m = smf.mixedlm(formula, df, groups=df[group]).fit(reml=reml)
    k = len(m.fe_params)
    out = pd.DataFrame({"Estimate": m.fe_params.values, "Std. Error": m.bse_fe.values,
                        "z value": m.tvalues[:k].values, "Pr(>|z|)": m.pvalues[:k].values,
                        "term": list(m.fe_params.index)})
    out.attrs["sigma_group"] = float(np.sqrt(m.cov_re.iloc[0, 0]))
    out.attrs["sigma_resid"] = float(np.sqrt(m.scale))
    return out


# ------------------------------------------------------------------ planning helpers
def feasibility(sample_ids, parent, positive=None, parent_floor: int = 50, pos_floor: int = 10):
    """Per-sample cell counts for a planned readout: run BEFORE committing to an analysis plan.

    Counts parent cells (e.g. CD8 T) and positive cells (e.g. GZMK+) per sample, and says whether
    enough samples reach the floors to support a patient-level test. Uses no outcome data.

    Parameters
    ----------
    sample_ids : per-cell sample labels.
    parent : per-cell boolean, cell belongs to the parent population.
    positive : optional per-cell boolean, cell is positive for the readout.
    parent_floor : minimum parent cells per usable sample.
    pos_floor : minimum median positive cells per sample.

    Returns
    -------
    tuple ``(table, summary)``: per-sample counts DataFrame and a dict with ``n_samples``,
    ``median_parent``, ``usable`` (and ``verdict`` when ``positive`` is given).

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> tab, summ = arsc.stats.feasibility(ds.obs["sample"], ds.obs["cell_type"] == "CD8 T")
    >>> summ["n_samples"], summ["usable"]
    (12, True)
    """
    df = pd.DataFrame({"sample": np.asarray(sample_ids), "parent": np.asarray(parent, bool)})
    if positive is not None:
        df["pos"] = np.asarray(positive, bool) & df["parent"]
    g = df.groupby("sample").agg(n_parent=("parent", "sum"), n_cells=("parent", "size"))
    if positive is not None:
        g["k_pos"] = df.groupby("sample")["pos"].sum()
        g["frac_pos"] = g["k_pos"] / g["n_parent"].replace(0, np.nan)
    g = g.reset_index()
    ok_parent = int((g.n_parent >= parent_floor).sum())
    summ = dict(n_samples=len(g), median_parent=float(g.n_parent.median()), min_parent=int(g.n_parent.min()),
                n_samples_ge_parent_floor=ok_parent, parent_floor=parent_floor)
    if positive is not None:
        sub = g[g.n_parent >= parent_floor]
        summ.update(median_positive=float(sub.k_pos.median()) if len(sub) else np.nan,
                    n_samples_ge_pos_floor=int((sub.k_pos >= pos_floor).sum()), pos_floor=pos_floor,
                    pooled_rate=float(sub.k_pos.sum() / sub.n_parent.sum()) if len(sub) else np.nan)
        summ["usable"] = bool(summ["median_positive"] >= pos_floor and ok_parent >= 6)
        summ["verdict"] = ("ok" if summ["usable"] else
                           f"SPARSE: median {summ['median_positive']:.0f} positive cells per sample "
                           f"(floor {pos_floor}) - use counts, a bigger parent, or another readout")
    else:
        summ["usable"] = ok_parent >= 6
    return g, summ


def count_logit_delta(k0, n0, k1, n1):
    """Within-patient logit change from counts with a (k+0.5)/(n+1) correction; flags sparse (< 10) numerators.

    Parameters
    ----------
    k0, n0 : positive and total counts at the first timepoint (per patient).
    k1, n1 : the same at the second timepoint.

    Returns
    -------
    tuple ``(delta, sparse)``: logit(k1/n1) - logit(k0/n0) and a boolean flag ``min(k0, k1) < 10``.

    Examples
    --------
    >>> import arsc
    >>> d, sparse = arsc.stats.count_logit_delta([5], [100], [20], [100])
    >>> round(float(d[0]), 3), bool(sparse[0])
    (1.487, True)
    """
    k0, n0, k1, n1 = (np.asarray(v, float) for v in (k0, n0, k1, n1))
    lg = lambda k, n: np.log((k + 0.5) / (n - k + 0.5))  # noqa: E731
    return lg(k1, n1) - lg(k0, n0), (np.minimum(k0, k1) < 10)


def mde(n, n2=None, sd: float = 1.0, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Minimum detectable effect (in units of ``sd``) for a paired (n) or two-group (n, n2) design.

    Checks the exact-test floor first (min p = 2/2^n paired, 2/C(n+n2, n) two-group): if that floor exceeds
    ``alpha`` no effect is detectable. Otherwise a t-based MDE with 1.15 rank-test inflation.

    Parameters
    ----------
    n : patients (paired design) or patients in group 1.
    n2 : patients in group 2 (two-group design); ``None`` = paired.
    sd : standard deviation of the per-patient value (the MDE is also returned in these units).
    alpha : two-sided significance level.
    power : target power.

    Returns
    -------
    dict with ``design``, ``n`` (``n2``), ``min_p``, ``mde_sd``, ``mde`` and ``feasible``.

    Examples
    --------
    >>> import arsc
    >>> arsc.stats.mde(3, 3)["feasible"]          # 3 vs 3 patients: min p = 0.1 > 0.05
    False
    >>> round(float(arsc.stats.mde(6)["mde_sd"]), 2)  # 6 paired patients
    1.64
    """
    if n2 is None:
        min_p = 2.0 / 2 ** n
        if min_p > alpha or n < 2:
            return dict(design="paired", n=n, min_p=min_p, mde_sd=float("inf"), mde=float("inf"), feasible=False)
        d = (sps.t.ppf(1 - alpha / 2, n - 1) + sps.t.ppf(power, n - 1)) / math.sqrt(n) * 1.15
        return dict(design="paired", n=n, min_p=min_p, mde_sd=d, mde=d * sd, feasible=True)
    min_p = 2.0 / math.comb(n + n2, n)
    if min_p > alpha:
        return dict(design="two-group", n=n, n2=n2, min_p=min_p, mde_sd=float("inf"), mde=float("inf"), feasible=False)
    df = n + n2 - 2
    d = (sps.t.ppf(1 - alpha / 2, df) + sps.t.ppf(power, df)) * math.sqrt(1 / n + 1 / n2) * 1.15
    return dict(design="two-group", n=n, n2=n2, min_p=min_p, mde_sd=d, mde=d * sd, feasible=True)
