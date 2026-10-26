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
from scipy.special import gammaln as _gammaln

from ._utils import as_list, check_seed
from ._seedlog import records_seed

__all__ = ["composition", "sample_means", "bh", "boot_ci", "wilcoxon_signed_rank", "paired_test", "perm_test",
           "arm_contrast", "glmm_prop", "gee_prop", "lmm_score", "mde", "feasibility", "count_logit_delta",
           "loo_effect", "signflip_test", "signflip_maxt", "hodges_lehmann", "mantel_haenszel", "icc_oneway",
           "freedman_lane", "downsample_counts", "rarefy_richness", "poisson_binomial", "perm_pvalue", "strat_perm_test",
           "boot_ci_diff"]


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
    keys = as_list(sample_cols) + as_list(extra_cols)
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
    keys = as_list(sample_cols) + as_list(by)
    cols = as_list(cols)
    _need_cols(obs, keys + cols, "obs", "cols / sample_cols / by")
    g = obs.groupby(keys, observed=True)
    df = g[cols].mean()
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


@records_seed("seed")
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

    Zeros are dropped; ``exact`` defaults to n < 50 and is only honored without zeros/ties (R falls back to the
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


@records_seed("seed")
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


@records_seed("seed")
def perm_test(x, y, stat=None, B: int = 20000, seed: int = 1) -> dict:
    """Two-group permutation test on per-patient values (e.g. responders vs non-responders).

    Exact enumeration when ``choose(n_x + n_y, n_x) <= B``, otherwise ``B`` random permutations.
    Two-sided: exact p = fraction of permutations with ``|stat| >= |observed|``; sampled p =
    ``(1 + #extreme) / (B + 1)``, which is never 0 (the observed labeling counts as one permutation).
    Non-finite values are dropped first; an empty group or a non-finite observed statistic gives p = NaN.

    Parameters
    ----------
    x, y : per-patient values of the two groups (one value per patient, not per cell).
    stat : ``stat(a, b) -> float``; default difference of medians. Medians tie across permutations, so with
        complete separation the median p can exceed the exact floor of :func:`mde` (4 vs 4: 0.057, not 0.029);
        pass ``stat=lambda a, b: a.mean() - b.mean()`` to reach that floor.
    B : enumeration limit / number of sampled permutations.
    seed : seeds a local ``np.random.default_rng`` for sampled permutations (unused when exact).

    Returns
    -------
    dict with ``n_x``, ``n_y``, ``stat`` (observed), ``p_perm`` and ``exact`` (True when enumerated). With 3 vs 3
    patients the smallest possible p is 2/20 = 0.1 (ties of the median statistic can make it larger, as below).

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
    if nx == 0 or len(y) == 0:
        return dict(n_x=nx, n_y=len(y), stat=np.nan, p_perm=np.nan, exact=True)
    obs = stat(x, y)
    if not np.isfinite(obs):
        return dict(n_x=nx, n_y=len(y), stat=float(obs), p_perm=np.nan, exact=True)
    exact = math.comb(len(z), nx) <= B
    if exact:
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
    hits = int(np.sum(np.abs(null) >= abs(obs) - 1e-12))
    p = hits / len(null) if exact else (hits + 1) / (len(null) + 1)
    return dict(n_x=nx, n_y=len(y), stat=float(obs), p_perm=float(p), exact=bool(exact))


@records_seed("seed")
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
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]  # same patients as perm_test uses (a NaN made HL and LOO NaN)
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
            # step-halving on the penalized log-likelihood
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
    over (beta, sigma), so estimates / variance components match glmer to optimizer tolerance; Std. Error uses
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


# ------------------------------------------------------------------ randomization tests on per-patient values
_ALTS = ("two-sided", "greater", "less")


def _tol(obs):
    """Tie tolerance for 'as or more extreme': absolute 1e-12 plus 1e-10 relative (sums of floats round)."""
    return 1e-12 + 1e-10 * abs(obs)


def _row_stat(M, stat):
    """Statistic of each row of a patterns x n matrix (``stat`` = 'mean' | 'sum' | 'median' | callable(1-D))."""
    if stat == "mean":
        return M.mean(axis=1)
    if stat == "sum":
        return M.sum(axis=1)
    if stat == "median":
        return np.median(M, axis=1)
    if callable(stat):
        return np.apply_along_axis(stat, 1, M)
    raise ValueError("stat must be 'mean', 'sum', 'median' or a callable on a 1-D array")


def _sign_patterns(n, max_exact, B, rng, chunk=1 << 16):
    """Yield blocks of sign patterns: all 2**n (exact) or B random ones."""
    if n <= max_exact:
        total = 1 << n
        bits = np.arange(n)
        for s in range(0, total, chunk):
            idx = np.arange(s, min(total, s + chunk))
            yield 1.0 - 2.0 * ((idx[:, None] >> bits) & 1)
    else:
        done = 0
        while done < B:
            m = min(chunk, B - done)
            yield rng.choice(np.array([-1.0, 1.0]), size=(m, n))
            done += m


def _extreme(null, obs, alternative):
    t = _tol(obs)
    if alternative == "two-sided":
        return np.abs(null) >= abs(obs) - t
    if alternative == "greater":
        return null >= obs - t
    return null <= obs + t


@records_seed("seed")
def signflip_test(d, stat="mean", alternative: str = "two-sided", max_exact: int = 20, B: int = 100_000,
                  seed: int = 1) -> dict:
    """One-sample sign-flip (randomization) test of per-patient differences: H0 = each difference is symmetric about 0.

    Exact over all ``2**n`` sign patterns for ``n <= max_exact``; otherwise ``B`` random patterns with the
    never-zero Monte Carlo p-value ``(1 + #extreme) / (B + 1)``. Non-finite values are dropped first (a NaN
    left in a hand-written version makes every comparison False and returns p = 0). "As or more extreme"
    uses a tolerance of 1e-12 + 1e-10 * |observed| so floating-point rounding of equal sums counts as a tie.

    Parameters
    ----------
    d : per-patient values (e.g. post - pre deltas), one per patient.
    stat : ``"mean"`` (default), ``"sum"`` (same p as the mean), ``"median"`` or a callable on a 1-D array.
    alternative : ``"two-sided"`` (``|null| >= |obs|``), ``"greater"`` or ``"less"``.
    max_exact : enumerate all patterns up to this many patients (2**20 = 1,048,576 patterns, chunked).
    B : number of random patterns beyond ``max_exact``.
    seed : seeds a local ``np.random.default_rng`` for the random patterns (unused when exact).

    Returns
    -------
    dict with ``n``, ``stat`` (observed), ``p``, ``exact``, ``n_patterns`` and ``min_p`` (the smallest p the
    design can give: ``2 / 2**n`` two-sided, ``1 / 2**n`` one-sided, exact case; ``1 / (B + 1)`` otherwise).

    Examples
    --------
    >>> import arsc
    >>> r = arsc.stats.signflip_test([0.5, 1.2, 0.8, 2.0, 1.1, 0.3])
    >>> r["p"], r["exact"], r["min_p"]
    (0.03125, True, 0.03125)
    """
    seed = check_seed(seed)
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")
    d = np.asarray(d, float).ravel()
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return dict(n=0, stat=np.nan, p=np.nan, exact=True, n_patterns=0, min_p=np.nan)
    obs = float(_row_stat(d[None, :], stat)[0])
    exact = n <= max_exact
    rng = np.random.default_rng(seed)
    hits = tot = 0
    for S in _sign_patterns(n, max_exact, B, rng):
        hits += int(_extreme(_row_stat(S * d, stat), obs, alternative).sum())
        tot += len(S)
    if exact:
        p = hits / tot
        min_p = (2.0 if alternative == "two-sided" else 1.0) / tot
    else:
        p = (hits + 1) / (tot + 1)
        min_p = 1.0 / (tot + 1)
    return dict(n=n, stat=obs, p=float(min(p, 1.0)), exact=exact, n_patterns=tot, min_p=float(min(min_p, 1.0)))


def _col_stats(M3, stat):
    """M3: patterns x n x K (NaN = missing). Column statistic per pattern -> patterns x K."""
    if stat == "mean":
        return np.nanmean(M3, axis=1)
    if stat == "t":
        k = np.sum(np.isfinite(M3), axis=1)
        m = np.nanmean(M3, axis=1)
        s = np.nanstd(M3, axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return m / (s / np.sqrt(k))
    raise ValueError("stat must be 'mean' or 't'")


@records_seed("seed")
def signflip_maxt(D, stat: str = "mean", method: str = "step-down", max_exact: int = 16, B: int = 20_000,
                  seed: int = 1) -> pd.DataFrame:
    """Family-wise sign-flip test over K readouts measured on the same patients (Westfall-Young maxT).

    Each permutation flips the sign of a whole patient row (all K readouts together), so the correlation
    between readouts is kept. The family-wise p of readout k is the share of patterns whose maximum
    ``|T*|`` over the family (single-step) or over the readouts not yet rejected (free step-down, then
    monotone) reaches ``|t_k|``. Use when a kill rule asks whether "the largest of K channels" moves.
    Missing values (NaN) are allowed: each readout uses its finite patients (``np.nanmean``).

    Parameters
    ----------
    D : patients x readouts array or DataFrame of per-patient differences (columns = readouts).
    stat : ``"mean"`` (default) or ``"t"`` (studentised mean; use when readouts have different scales).
    method : ``"step-down"`` (default; more powerful, same FWER control) or ``"single-step"``.
    max_exact : enumerate all ``2**n`` patterns up to this many patients.
    B : random patterns beyond ``max_exact`` (Monte Carlo p = (1 + #) / (B + 1)).
    seed : seeds a local ``np.random.default_rng`` for the random patterns (unused when exact).

    Returns
    -------
    DataFrame, one row per readout: ``readout, n, stat, p_raw, p_fwer, exact, n_patterns``.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> D = np.array([[1.0, 0.2], [0.8, -0.1], [1.3, 0.3], [0.9, 0.1], [1.1, -0.2], [0.7, 0.0]])
    >>> r = arsc.stats.signflip_maxt(D)
    >>> r[["p_raw", "p_fwer"]].round(4).values.tolist()
    [[0.0312, 0.0312], [0.6875, 0.6875]]
    """
    seed = check_seed(seed)
    if method not in ("step-down", "single-step"):
        raise ValueError("method must be 'step-down' or 'single-step'")
    names = list(D.columns) if isinstance(D, pd.DataFrame) else list(range(np.shape(D)[1]))
    X = np.asarray(D, float)
    if X.ndim != 2:
        raise ValueError("D must be 2-D (patients x readouts)")
    X = np.where(np.isfinite(X), X, np.nan)
    X = X[np.isfinite(X).any(axis=1)]
    n, K = X.shape
    obs = _col_stats(X[None], stat)[0]
    a = np.abs(obs)
    order = np.argsort(-np.where(np.isfinite(a), a, -np.inf), kind="stable")
    exact = n <= max_exact
    rng = np.random.default_rng(seed)
    raw = np.zeros(K); fw = np.zeros(K); tot = 0
    for S in _sign_patterns(n, max_exact, B, rng, chunk=1 << 14):
        T = np.abs(_col_stats(S[:, :, None] * X[None], stat))
        T = np.where(np.isfinite(T), T, -np.inf)
        tol = 1e-12 + 1e-10 * a
        raw += (T >= a - tol).sum(axis=0)
        if method == "single-step":
            fw += (T.max(axis=1)[:, None] >= a - tol).sum(axis=0)
        else:
            # successive maxima over readouts ranked at or below each one (in |obs| order)
            Ts = T[:, order]
            succ = np.maximum.accumulate(Ts[:, ::-1], axis=1)[:, ::-1]
            fw[order] += (succ >= (a - tol)[order]).sum(axis=0)
        tot += len(S)
    if exact:
        p_raw, p_fw = raw / tot, fw / tot
    else:
        p_raw, p_fw = (raw + 1) / (tot + 1), (fw + 1) / (tot + 1)
    if method == "step-down":
        p_fw[order] = np.maximum.accumulate(p_fw[order])
    p_fw = np.maximum(p_fw, p_raw)
    bad = ~np.isfinite(obs)
    p_raw[bad] = np.nan; p_fw[bad] = np.nan
    return pd.DataFrame(dict(readout=names, n=np.isfinite(X).sum(axis=0), stat=obs, p_raw=np.minimum(p_raw, 1),
                             p_fwer=np.minimum(p_fw, 1), exact=exact, n_patterns=tot))


# ------------------------------------------------------------------ Hodges-Lehmann estimate + CI (R wilcox.test)
class _RankNull:
    """Exact null of the signed-rank V (one sample) or Mann-Whitney W (two samples), with R's p/q arithmetic
    (psignrank / pwilcox sum count * f over the shorter tail; q* use R's 10 * DBL_EPSILON fuzz), so achieved
    confidence levels and p-values agree with R to the last bit."""

    def __init__(self, n, m=None):
        if m is None:
            self.c = _psignrank_counts(n)
            self.f = math.exp(-n * math.log(2))
            self.top = n * (n + 1) // 2
            self.half = n * (n + 1) / 4
            self.signrank = True
        else:
            self.c = _ranksum_counts(n, m)
            self.f = 1.0 / float(math.comb(n + m, m))
            self.top = n * m
            self.half = n * m / 2
            self.signrank = False

    def p(self, q, lower=True):
        q = float(np.rint(q + 1e-7)) if self.signrank else math.floor(q + 1e-7)
        if q < 0:
            return 0.0 if lower else 1.0
        if q >= self.top:
            return 1.0 if lower else 0.0
        acc = 0.0
        if q <= self.half:
            for i in range(int(q) + 1):
                acc += self.c[i] * self.f
        else:
            q = self.top - q
            for i in range(int(q)):
                acc += self.c[i] * self.f
            lower = not lower
        return acc if lower else (0.5 - acc + 0.5)

    def q(self, x):
        eps = np.finfo(float).eps
        acc = 0.0
        if x <= 0.5:
            x = x - 10 * eps
            for q in range(self.top + 1):
                acc += self.c[q] * self.f
                if acc >= x:
                    return q
            return self.top
        x = 1 - x + 10 * eps
        for q in range(self.top + 1):
            acc += self.c[q] * self.f
            if acc > x:
                return self.top - q
        return 0


def _ranksum_counts(m: int, n: int) -> np.ndarray:
    """Counts of the Mann-Whitney W = rank sum of x - m(m+1)/2 (values 0..m*n) over all C(m+n, m) splits."""
    # Gaussian-binomial recursion c(k; m, n) = c(k - n; m - 1, n) + c(k; m, n - 1) (R's cwilcox)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def cw(mm, nn):
        if mm == 0 or nn == 0:
            out = np.zeros(mm * nn + 1); out[0] = 1.0
            return out
        a = cw(mm - 1, nn)          # length (mm-1)*nn + 1, shifted by nn
        b = cw(mm, nn - 1)          # length mm*(nn-1) + 1
        out = np.zeros(mm * nn + 1)
        out[nn:nn + len(a)] += a
        out[:len(b)] += b
        return out
    return cw(m, n).copy()


def _brent(f, a, b, fa, fb, tol):
    """R's uniroot (Brent's zeroin, R_zeroin2) - same iterates, so step-function roots land where R's do."""
    EPS = np.finfo(float).eps
    if fa == 0:
        return a
    if fb == 0:
        return b
    c, fc = a, fa
    maxit = 1000
    for _ in range(maxit + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * EPS * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc; t1 = fb / fc; t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b


def _signif(v, digits):
    """R ``signif`` (decimal rounding of the shortest repr, half to even as R 4 does for representable halves)."""
    if v == 0 or not np.isfinite(v):
        return v
    from decimal import Decimal, ROUND_HALF_EVEN
    e = math.floor(math.log10(abs(v)))
    q = Decimal(1).scaleb(e - digits + 1)
    return float(Decimal(v).quantize(q, rounding=ROUND_HALF_EVEN))


def _ties_term(r):
    _, t = np.unique(r, return_counts=True)
    return float(np.sum(t.astype(float) ** 3 - t))


def hodges_lehmann(x, y=None, paired: bool = False, conf_level: float = 0.95, alternative: str = "two-sided",
                   mu: float = 0.0, exact=None, correct: bool = True, tol_root: float = 1e-4) -> dict:
    """Hodges-Lehmann location estimate with its Wilcoxon confidence interval and p, replicating R ``wilcox.test(conf.int = TRUE)``.

    One sample (or ``paired=True``): the pseudomedian (median of the Walsh averages) of ``x`` (or ``x - y``).
    Two samples: the median of all ``x_i - y_j`` differences. Exact (n < 50 and no ties or zeros, as R): the CI
    is read off the sorted Walsh averages / differences at the ``qsignrank`` / ``qwilcox`` quantiles; when the
    requested level cannot be reached (tiny n) the achieved level is reported, as R does. Otherwise R's normal
    approximation: the CI inverts the continuity-corrected z by Brent root finding (R's uniroot, ``tol_root``),
    and the estimate is the root without correction. Use this instead of bootstrap CIs of the HL estimate,
    which are degenerate with 4-8 patients (they collapse onto the few distinct resampled medians).

    Parameters
    ----------
    x : values (one per patient) of group 1, or the paired differences.
    y : values of group 2 (two-sample), or the second member of each pair with ``paired=True``.
    paired : treat ``x`` and ``y`` as pairs (``x - y`` on complete pairs).
    conf_level : requested confidence level.
    alternative : ``"two-sided"``, ``"greater"`` or ``"less"`` (one-sided intervals are half-infinite).
    mu : null location (shift).
    exact : exact distribution (default: n < 50, both groups for two samples; ignored with ties / zeros).
    correct : continuity correction in the normal approximation.
    tol_root : root-finding tolerance (R's ``tol.root``).

    Returns
    -------
    dict with ``estimate``, ``ci_lo``, ``ci_hi``, ``conf_level`` (achieved), ``statistic`` (R's V or W),
    ``p``, ``method`` (``"exact"`` or ``"normal"``), ``n_x``, ``n_y``.

    Examples
    --------
    >>> import arsc
    >>> r = arsc.stats.hodges_lehmann([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30],
    ...                               [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29], paired=True)
    >>> round(r["estimate"], 3), round(r["ci_lo"], 3), round(r["ci_hi"], 3), round(r["p"], 5)
    (0.46, 0.01, 0.786, 0.03906)
    """
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")
    if not (0 < conf_level < 1):
        raise ValueError("conf_level must be between 0 and 1")
    x = np.asarray(x, float).ravel()
    if y is not None:
        y = np.asarray(y, float).ravel()
        if paired:
            if len(x) != len(y):
                raise ValueError("x and y must have the same length for paired=True")
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok] - y[ok]
            y = None
        else:
            y = y[np.isfinite(y)]
    elif paired:
        raise ValueError("y is missing for paired=True")
    x = x[np.isfinite(x)]
    if len(x) < 1:
        raise ValueError("not enough (non-missing) x observations")
    alpha = 1 - conf_level
    nq = sps.norm.ppf
    if y is None:
        xs = x - mu
        zeros = bool(np.any(xs == 0))
        xs = xs[xs != 0]
        n = len(xs)
        ex = (n < 50) if exact is None else bool(exact)
        r = sps.rankdata(np.abs(xs))
        V = float(r[xs > 0].sum())
        ties = len(np.unique(r)) != len(r)
        out = dict(n_x=len(x), n_y=0, statistic=V)
        if ex and not ties and not zeros:
            nd = _RankNull(n)
            top = nd.top
            if alternative == "two-sided":
                p = nd.p(V - 1, lower=False) if V > n * (n + 1) / 4 else nd.p(V)
                p = min(2 * p, 1.0)
            elif alternative == "greater":
                p = nd.p(V - 1, lower=False)
            else:
                p = nd.p(V)
            xx = xs + mu
            W = (xx[:, None] + xx[None, :])[np.triu_indices(n)]
            diffs = np.sort(W) / 2
            if alternative == "two-sided":
                qu = nd.q(alpha / 2) or 1
                ql = top - qu
                ach = 2 * nd.p(qu - 1)
                lo, hi = diffs[qu - 1], diffs[ql]
            elif alternative == "greater":
                qu = nd.q(alpha) or 1
                ach = nd.p(qu - 1)
                lo, hi = diffs[qu - 1], np.inf
            else:
                qu = nd.q(alpha) or 1
                ql = top - qu
                ach = nd.p(qu - 1)
                lo, hi = -np.inf, diffs[ql]
            cl = conf_level
            if ach - alpha > alpha / 2:
                cl = 1 - _signif(ach, 2)
            out.update(estimate=float(np.median(diffs)), ci_lo=float(lo), ci_hi=float(hi), conf_level=cl, p=float(p),
                       method="exact")
            return out
        z = V - n * (n + 1) / 4
        sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - _ties_term(r) / 48) if n else 0.0
        corr = ({"two-sided": np.sign(z) * 0.5, "greater": 0.5, "less": -0.5}[alternative]) if correct else 0.0
        zz = (z - corr) / sig if sig > 0 else np.nan
        p = {"less": sps.norm.cdf(zz), "greater": sps.norm.sf(zz),
             "two-sided": 2 * min(sps.norm.cdf(zz), sps.norm.sf(zz))}[alternative] if sig > 0 else np.nan
        xx = xs + mu

        def Wfun(dd, use_corr):
            xd = xx - dd
            xd = xd[xd != 0]
            nx = len(xd)
            dr = sps.rankdata(np.abs(xd))
            zd = dr[xd > 0].sum() - nx * (nx + 1) / 4
            s = math.sqrt(nx * (nx + 1) * (2 * nx + 1) / 24 - _ties_term(dr) / 48) if nx else 0.0
            c = ({"two-sided": np.sign(zd) * 0.5, "greater": 0.5, "less": -0.5}[alternative]) if use_corr else 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                return (zd - c) / s if s > 0 else (np.nan if zd - c == 0 else np.sign(zd - c) * np.inf)
        return _hl_normal(out, xx, Wfun, float(np.min(xx)) if n else np.nan, float(np.max(xx)) if n else np.nan,
                          n, alpha, alternative, conf_level, correct, p, tol_root, midpoint=float(np.median(xx)) if n else np.nan)
    if len(y) < 1:
        raise ValueError("not enough y observations")
    nx, ny = len(x), len(y)
    rr = sps.rankdata(np.concatenate([x - mu, y]))
    Wst = float(rr[:nx].sum() - nx * (nx + 1) / 2)
    ex = (nx < 50 and ny < 50) if exact is None else bool(exact)
    ties = len(np.unique(rr)) != len(rr)
    out = dict(n_x=nx, n_y=ny, statistic=Wst)
    if ex and not ties:
        nd = _RankNull(nx, ny)
        top = nd.top
        if alternative == "two-sided":
            p = nd.p(Wst - 1, lower=False) if Wst > nx * ny / 2 else nd.p(Wst)
            p = min(2 * p, 1.0)
        elif alternative == "greater":
            p = nd.p(Wst - 1, lower=False)
        else:
            p = nd.p(Wst)
        diffs = np.sort(np.subtract.outer(x, y).ravel())
        if alternative == "two-sided":
            qu = nd.q(alpha / 2) or 1
            ql = top - qu
            ach = 2 * nd.p(qu - 1)
            lo, hi = diffs[qu - 1], diffs[ql]
        elif alternative == "greater":
            qu = nd.q(alpha) or 1
            ach = nd.p(qu - 1)
            lo, hi = diffs[qu - 1], np.inf
        else:
            qu = nd.q(alpha) or 1
            ql = top - qu
            ach = nd.p(qu - 1)
            lo, hi = -np.inf, diffs[ql]
        cl = conf_level
        if ach - alpha > alpha / 2:
            cl = 1 - ach
        out.update(estimate=float(np.median(diffs)), ci_lo=float(lo), ci_hi=float(hi), conf_level=float(cl), p=float(p),
                   method="exact")
        return out
    z = Wst - nx * ny / 2
    N = nx + ny
    sig = math.sqrt((nx * ny / 12) * ((N + 1) - _ties_term(rr) / (N * (N - 1))))
    corr = ({"two-sided": np.sign(z) * 0.5, "greater": 0.5, "less": -0.5}[alternative]) if correct else 0.0
    zz = (z - corr) / sig if sig > 0 else np.nan
    p = {"less": sps.norm.cdf(zz), "greater": sps.norm.sf(zz),
         "two-sided": 2 * min(sps.norm.cdf(zz), sps.norm.sf(zz))}[alternative] if sig > 0 else np.nan

    def Wfun(dd, use_corr):
        dr = sps.rankdata(np.concatenate([x - dd, y]))
        dz = dr[:nx].sum() - nx * (nx + 1) / 2 - nx * ny / 2
        c = ({"two-sided": np.sign(dz) * 0.5, "greater": 0.5, "less": -0.5}[alternative]) if use_corr else 0.0
        s = math.sqrt((nx * ny / 12) * ((N + 1) - _ties_term(dr) / (N * (N - 1))))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (dz - c) / s if s > 0 else (np.nan if dz - c == 0 else np.sign(dz - c) * np.inf)
    return _hl_normal(out, None, Wfun, float(x.min() - y.max()), float(x.max() - y.min()), N, alpha, alternative,
                      conf_level, correct, p, tol_root, midpoint=None, two_sample=True)


def _hl_normal(out, xx, Wfun, mumin, mumax, n, alpha, alternative, conf_level, correct, p, tol_root, midpoint=None,
               two_sample=False):
    """Normal-approximation CI / estimate branch of R wilcox.test (one- and two-sample)."""
    nq = sps.norm.ppf
    W = lambda dd: Wfun(dd, correct)  # noqa: E731
    cl = conf_level
    if not two_sample:
        Wmin = W(mumin) if n > 0 else np.nan
        Wmax = W(mumax) if (n > 0 and np.isfinite(Wmin)) else np.nan
        if n == 0 or not np.isfinite(Wmax):
            out.update(estimate=(mumin + mumax) / 2 if n > 0 else np.nan,
                       ci_lo=-np.inf if alternative == "less" else np.nan,
                       ci_hi=np.inf if alternative == "greater" else np.nan, conf_level=0.0, p=float(p), method="normal")
            return out

        def root(zq):
            return _brent(lambda d: W(d) - zq, mumin, mumax, Wmin - zq, Wmax - zq, tol_root)
        a = alpha
        if alternative == "two-sided":
            while Wmin - nq(1 - a / 2) < 0 or Wmax - nq(a / 2) > 0:
                a *= 2
            if a >= 1 or 1 - conf_level < a * 0.75:
                cl = 1 - min(1, a)
            if a < 1:
                lo, hi = root(nq(1 - a / 2)), root(nq(a / 2))
            else:
                lo = hi = midpoint
        elif alternative == "greater":
            while Wmin - nq(1 - a) < 0:
                a *= 2
            if a >= 1 or 1 - conf_level < a * 0.75:
                cl = 1 - min(1, a)
            lo, hi = (root(nq(1 - a)) if a < 1 else midpoint), np.inf
        else:
            while Wmax - nq(a / 2) > 0:
                a *= 2
            if a >= 1 or 1 - conf_level < a * 0.75:
                cl = 1 - min(1, a)
            lo, hi = -np.inf, (root(nq(a)) if a < 1 else midpoint)
        W0 = lambda dd: Wfun(dd, False)  # noqa: E731
        est = _brent(W0, mumin, mumax, W0(mumin), W0(mumax), tol_root)
    else:
        Wmin, Wmax = W(mumin), W(mumax)

        def root(zq):
            fl = Wmin - zq
            if fl <= 0:
                return mumin
            fu = Wmax - zq
            if fu >= 0:
                return mumax
            return _brent(lambda d: W(d) - zq, mumin, mumax, fl, fu, tol_root)
        if alternative == "two-sided":
            lo, hi = root(nq(1 - alpha / 2)), root(nq(alpha / 2))
        elif alternative == "greater":
            lo, hi = root(nq(1 - alpha)), np.inf
        else:
            lo, hi = -np.inf, root(nq(alpha))
        W0 = lambda dd: Wfun(dd, False)  # noqa: E731
        est = _brent(W0, mumin, mumax, W0(mumin), W0(mumax), tol_root)
    out.update(estimate=float(est), ci_lo=float(lo), ci_hi=float(hi), conf_level=float(cl), p=float(p), method="normal")
    return out


# ------------------------------------------------------------------ stratified 2x2 tables
def mantel_haenszel(a, b=None, c=None, d=None, conf_level: float = 0.95, correct: bool = True) -> dict:
    """Mantel-Haenszel common odds ratio over strata (e.g. patients) with its CI and the CMH test, as R ``mantelhaen.test``.

    Stratum k is the 2 x 2 table ``[[a_k, b_k], [c_k, d_k]]`` (rows = exposure yes/no, columns = outcome yes/no).
    Odds ratio = sum(a d / n) / sum(b c / n); CI from the Robins-Breslow-Greenland variance; the chi-square
    statistic with continuity correction (R's default ``correct = TRUE``, applied when |sum(a - E a)| >= 0.5).
    No pseudo-counts are added (adding 0.5 to every cell, as hand-written versions did, pulls the odds ratio
    towards 1). Strata with fewer than 2 observations carry no information and are dropped (R stops on them).

    Parameters
    ----------
    a : cell counts a_k, or a K x 2 x 2 array of the tables (then ``b``, ``c``, ``d`` are omitted).
    b, c, d : the other cell counts per stratum.
    conf_level : CI level.
    correct : continuity correction of the CMH statistic.

    Returns
    -------
    dict with ``or_mh``, ``ci_lo``, ``ci_hi``, ``log_or``, ``se_log_or``, ``statistic`` (CMH chi-square, 1 df),
    ``p``, ``n_strata`` (used), ``n_dropped``.

    Examples
    --------
    >>> import arsc
    >>> r = arsc.stats.mantel_haenszel([10, 6, 8], [5, 7, 4], [4, 3, 5], [12, 9, 10])
    >>> round(r["or_mh"], 3), round(r["p"], 4)
    (4.041, 0.006)
    """
    if b is None:
        T = np.asarray(a, float)
        if T.ndim != 3 or T.shape[1:] != (2, 2):
            raise ValueError("pass a K x 2 x 2 array or the four count vectors a, b, c, d")
        A, Bv, Cv, Dv = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    else:
        A, Bv, Cv, Dv = (np.asarray(v, float).ravel() for v in (a, b, c, d))
    if np.any(np.array([A, Bv, Cv, Dv]) < 0):
        raise ValueError("counts must be non-negative")
    n = A + Bv + Cv + Dv
    keep = n >= 2
    A, Bv, Cv, Dv, n = A[keep], Bv[keep], Cv[keep], Dv[keep], n[keep]
    out = dict(n_strata=int(keep.sum()), n_dropped=int((~keep).sum()))
    if keep.sum() == 0:
        out.update(or_mh=np.nan, ci_lo=np.nan, ci_hi=np.nan, log_or=np.nan, se_log_or=np.nan, statistic=np.nan, p=np.nan)
        return out
    r1, r2, c1, c2 = A + Bv, Cv + Dv, A + Cv, Bv + Dv
    delta = float(np.sum(A - r1 * c1 / n))
    yates = 0.5 if (correct and abs(delta) >= 0.5) else 0.0
    var = float(np.sum(r1 * r2 * c1 * c2 / (n ** 2 * (n - 1))))
    stat = (abs(delta) - yates) ** 2 / var if var > 0 else np.nan
    p = float(sps.chi2.sf(stat, 1)) if var > 0 else np.nan
    sd_ = float(np.sum(A * Dv / n)); so = float(np.sum(Bv * Cv / n))
    with np.errstate(divide="ignore", invalid="ignore"):
        est = sd_ / so
        se = math.sqrt(np.sum((A + Dv) * A * Dv / n ** 2) / (2 * sd_ ** 2)
                       + np.sum(((A + Dv) * Bv * Cv + (Bv + Cv) * A * Dv) / n ** 2) / (2 * sd_ * so)
                       + np.sum((Bv + Cv) * Bv * Cv / n ** 2) / (2 * so ** 2)) if sd_ > 0 and so > 0 else np.nan
    zq = sps.norm.ppf(1 - (1 - conf_level) / 2)
    lo, hi = (est * math.exp(-zq * se), est * math.exp(zq * se)) if np.isfinite(se) else (np.nan, np.nan)
    out.update(or_mh=float(est), ci_lo=float(lo), ci_hi=float(hi), log_or=float(np.log(est)) if est > 0 else np.nan,
               se_log_or=float(se), statistic=float(stat), p=p)
    return out


# ------------------------------------------------------------------ intraclass correlation
def icc_oneway(y, groups, strata=None, conf_level: float = 0.95, clamp: bool = False) -> dict:
    """One-way random-effects intraclass correlation ICC(1) by ANOVA (unbalanced groups allowed), with its F test and CI.

    ``ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW)`` with Searle's ``n0 = (N - sum n_i^2 / N) / (k - 1)``
    (e.g. y = a per-cell score, groups = clone: how much of the variance is between clones). With ``strata``
    (e.g. patient) the groups are nested in strata and the strata are removed as fixed effects: sums of squares
    and degrees of freedom are pooled over strata, and ``n0 = sum_s (N_s - sum n_si^2 / N_s) / sum_s (k_s - 1)``,
    the coefficient of the between-group variance in E[MSB]. (A plain mean of per-stratum n0 values, used by
    hand-written versions, is a different and biased weighting when strata differ in size.) The CI is the F-based
    Shrout-Fleiss interval, exact for balanced groups and approximate otherwise. Groups of size 1 add nothing
    to MSW but count in MSB, as in the ANOVA.

    Parameters
    ----------
    y : per-observation values.
    groups : group label per observation (e.g. clone id).
    strata : optional stratum label per observation (e.g. patient) within which groups are nested.
    conf_level : CI level.
    clamp : truncate a negative estimate (and CI bounds) at 0.

    Returns
    -------
    dict with ``icc``, ``ci_lo``, ``ci_hi``, ``F``, ``df_b``, ``df_w``, ``p`` (F test of no between-group
    variance), ``msb``, ``msw``, ``n0``, ``k`` (groups), ``n`` (observations).

    Examples
    --------
    >>> import arsc
    >>> r = arsc.stats.icc_oneway([1.0, 1.2, 0.9, 3.0, 3.3, 2.8, 2.0, 2.1, 1.7], list("aaabbbccc"))
    >>> round(r["icc"], 3), round(r["p"], 5)
    (0.958, 7e-05)
    """
    y = np.asarray(y, float).ravel()
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    s = np.zeros(len(y), int) if strata is None else pd.factorize(pd.Series(np.asarray(strata)).astype(str))[0]
    if not (len(g) == len(y) == len(s)):
        raise ValueError("y, groups and strata must have the same length")
    ok = np.isfinite(y)
    y, g, s = y[ok], g[ok], s[ok]
    ssb = ssw = 0.0; dfb = dfw = 0; n0num = 0.0; k_tot = 0
    for st_ in np.unique(s):
        m = s == st_
        ys, gs = y[m], g[m]
        codes, inv = np.unique(gs, return_inverse=True)
        k = len(codes)
        if k < 1:
            continue
        cnt = np.bincount(inv).astype(float)
        mu = np.bincount(inv, ys) / cnt
        gm = ys.mean()
        ssb += float((cnt * (mu - gm) ** 2).sum()); dfb += k - 1
        ssw += float(((ys - mu[inv]) ** 2).sum()); dfw += len(ys) - k
        n0num += len(ys) - (cnt ** 2).sum() / len(ys)
        k_tot += k
    out = dict(k=int(k_tot), n=int(len(y)), df_b=int(dfb), df_w=int(dfw))
    if dfb < 1 or dfw < 1:
        out.update(icc=np.nan, ci_lo=np.nan, ci_hi=np.nan, F=np.nan, p=np.nan, msb=np.nan, msw=np.nan, n0=np.nan)
        return out
    msb, msw = ssb / dfb, ssw / dfw
    n0 = n0num / dfb
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = msb / msw
    p = float(sps.f.sf(F, dfb, dfw)) if np.isfinite(F) else np.nan
    q = 1 - (1 - conf_level) / 2
    FL = F / sps.f.ppf(q, dfb, dfw)
    FU = F * sps.f.ppf(q, dfw, dfb)
    lo = (FL - 1) / (FL + n0 - 1)
    hi = (FU - 1) / (FU + n0 - 1)
    if clamp:
        icc, lo, hi = max(icc, 0.0), max(lo, 0.0), max(hi, 0.0)
    out.update(icc=float(icc), ci_lo=float(lo), ci_hi=float(hi), F=float(F), p=p, msb=float(msb), msw=float(msw),
               n0=float(n0))
    return out


# ------------------------------------------------------------------ covariate-adjusted permutation (Freedman-Lane)
def _ols_t(Xf, Y):
    """t of the last column of Xf for each column of Y (n x P)."""
    n, q = Xf.shape
    XtX_inv = np.linalg.pinv(Xf.T @ Xf)
    beta = XtX_inv @ (Xf.T @ Y)
    res = Y - Xf @ beta
    s2 = (res ** 2).sum(axis=0) / (n - q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return beta[-1] / np.sqrt(s2 * XtX_inv[-1, -1]), beta[-1]


@records_seed("seed")
def freedman_lane(y, x, covariates=None, B: int = 10_000, max_exact: int = 8, alternative: str = "two-sided",
                  seed: int = 1) -> dict:
    """Permutation test of one regression coefficient adjusted for covariates (Freedman & Lane 1983).

    Tests ``x`` in ``y ~ 1 + covariates + x``: fit the reduced model ``y ~ 1 + covariates``, permute its residuals,
    add them back to its fitted values, refit the full model and compare the t statistic of ``x``. Permuting
    ``y`` itself (or ``x``) ignores the covariates and is not a valid adjusted test when ``x`` and the
    covariates are correlated. All ``n!`` permutations for ``n <= max_exact`` (8! = 40,320), else ``B`` random ones
    with p = (1 + #) / (B + 1). Rows with any missing value are dropped.

    Parameters
    ----------
    y : outcome per unit (e.g. per patient).
    x : the predictor tested (one column).
    covariates : optional array / DataFrame of nuisance covariates (an intercept is always added).
    B : random permutations beyond ``max_exact``.
    max_exact : enumerate every permutation up to this many units.
    alternative : ``"two-sided"`` (``|t|``), ``"greater"`` or ``"less"``.
    seed : seeds a local ``np.random.default_rng`` (unused when exact).

    Returns
    -------
    dict with ``beta`` and ``t`` of ``x`` (OLS), ``p_param`` (the OLS t-test), ``p_perm``, ``n``, ``exact``,
    ``n_perm``.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(size=30); x = z + rng.normal(scale=0.5, size=30); y = 2 * z + rng.normal(size=30)
    >>> r = arsc.stats.freedman_lane(y, x, covariates=z, B=2000)
    >>> r["p_perm"] > 0.05, bool(r["exact"])
    (True, False)
    """
    seed = check_seed(seed)
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")
    y = np.asarray(y, float).ravel(); x = np.asarray(x, float).ravel()
    Z = np.empty((len(y), 0)) if covariates is None else np.asarray(covariates, float).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    y, x, Z = y[ok], x[ok], Z[ok]
    n = len(y)
    R = np.column_stack([np.ones(n), Z])
    Xf = np.column_stack([R, x])
    if n <= Xf.shape[1] + 1 or np.linalg.matrix_rank(Xf) < Xf.shape[1]:
        return dict(beta=np.nan, t=np.nan, p_param=np.nan, p_perm=np.nan, n=n, exact=False, n_perm=0)
    t_obs, b_obs = _ols_t(Xf, y[:, None])
    t_obs, b_obs = float(t_obs[0]), float(b_obs[0])
    dfr = n - Xf.shape[1]
    p_param = {"two-sided": 2 * sps.t.sf(abs(t_obs), dfr), "greater": sps.t.sf(t_obs, dfr),
               "less": sps.t.cdf(t_obs, dfr)}[alternative]
    g = np.linalg.lstsq(R, y, rcond=None)[0]
    fit = R @ g; e = y - fit
    exact = n <= max_exact
    rng = np.random.default_rng(seed)
    hits = tot = 0
    if exact:
        perms = np.array(list(itertools.permutations(range(n))))
        blocks = [perms[i:i + 8192] for i in range(0, len(perms), 8192)]
    else:
        blocks = []
        done = 0
        while done < B:
            m = min(8192, B - done)
            blocks.append(np.argsort(rng.random((m, n)), axis=1))
            done += m
    for P in blocks:
        Ys = fit[:, None] + e[P].T
        ts, _ = _ols_t(Xf, Ys)
        hits += int(_extreme(ts, t_obs, alternative).sum()); tot += len(P)
    p = hits / tot if exact else (hits + 1) / (tot + 1)
    return dict(beta=b_obs, t=t_obs, p_param=float(p_param), p_perm=float(p), n=n, exact=exact, n_perm=tot)


# ------------------------------------------------------------------ depth matching: thinning and rarefaction
@records_seed("seed")
def downsample_counts(X, target, method: str = "hypergeometric", seed: int = 1):
    """Thin a cells x genes count matrix to a common depth per cell (removes depth differences before a comparison).

    ``method="hypergeometric"`` draws exactly ``target`` UMIs without replacement from each cell (scanpy
    ``pp.downsample_counts``, Seurat ``SampleUMI``, vegan ``rrarefy``): every thinned cell has total ``target``.
    ``method="binomial"`` keeps each UMI with probability ``target / total`` (expected total ``target``; the
    variance is larger and totals vary). Cells whose total is at or below ``target`` are returned unchanged:
    drop them (``totals < target``) when an exact common depth matters. Normalize thinned cells by their own
    new total (or by ``target`` for the hypergeometric method, where they are equal).

    Parameters
    ----------
    X : cells x genes integer counts (CSR or dense).
    target : common depth (int) or one target per cell.
    method : ``"hypergeometric"`` (exact depth) or ``"binomial"``.
    seed : seeds a local ``np.random.default_rng``.

    Returns
    -------
    scipy.sparse.csr_matrix of the same shape and dtype float64.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> ds = arsc.datasets.toy()
    >>> X = ds.X()
    >>> Y = arsc.stats.downsample_counts(X, 200)
    >>> tot = np.asarray(X.sum(1)).ravel(); new = np.asarray(Y.sum(1)).ravel()
    >>> bool(np.all(new[tot >= 200] == 200)), bool(np.all(new[tot < 200] == tot[tot < 200]))
    (True, True)
    """
    import scipy.sparse as _sp
    seed = check_seed(seed)
    if method not in ("hypergeometric", "binomial"):
        raise ValueError("method must be 'hypergeometric' or 'binomial'")
    M = _sp.csr_matrix(X, dtype=np.float64, copy=True)
    M.sum_duplicates()
    data = M.data
    if np.any(data < 0) or not np.allclose(data, np.round(data)):
        raise ValueError("X must hold non-negative integer counts")
    data = np.round(data).astype(np.int64)
    nr = M.shape[0]
    tgt = np.broadcast_to(np.asarray(target, np.int64), (nr,)).copy()
    rows = np.repeat(np.arange(nr), np.diff(M.indptr))
    tot = np.bincount(rows, weights=data, minlength=nr).astype(np.int64)
    rng = np.random.default_rng(seed)
    thin = tot > tgt
    out = data.copy()
    if method == "binomial":
        pr = np.where(thin, tgt / np.maximum(tot, 1), 1.0)
        sel = thin[rows]
        out[sel] = rng.binomial(data[sel], pr[rows][sel])
    else:
        rem_n = tot.astype(np.int64).copy(); rem_d = tgt.copy()
        nnz = np.diff(M.indptr)
        start = M.indptr[:-1]
        trows = np.flatnonzero(thin)
        maxk = int(nnz[trows].max()) if len(trows) else 0
        for k in range(maxk):
            r = trows[nnz[trows] > k]
            if len(r) == 0:
                break
            pos = start[r] + k
            c = data[pos]
            h = rng.hypergeometric(c, rem_n[r] - c, rem_d[r])
            out[pos] = h
            rem_n[r] -= c; rem_d[r] -= h
    M.data = out.astype(np.float64)
    M.eliminate_zeros()
    return M


def rarefy_richness(counts, depth, se: bool = False):
    """Expected number of distinct clones (or genes) in a random subsample of ``depth`` cells, as vegan ``rarefy``.

    Closed form (Hurlbert 1971; variance Heck et al. 1975): ``E[S_d] = sum_i 1 - C(N - n_i, d) / C(N, d)``. No
    Monte Carlo is needed (repeated random subsampling estimates the same number with noise). A depth above
    the sample total returns the observed richness, as vegan does: check ``depth <= N``.

    Parameters
    ----------
    counts : clone sizes of one sample (1-D), or samples x clones (2-D; one result per row).
    depth : subsample size (int).
    se : also return the standard error.

    Returns
    -------
    float (1-D input) or array (2-D); with ``se=True`` a tuple ``(expected, se)``.

    Examples
    --------
    >>> import arsc
    >>> round(float(arsc.stats.rarefy_richness([10, 5, 3, 1, 1], 5)), 4)
    2.8909
    """
    C = np.asarray(counts, float)
    one = C.ndim == 1
    C = np.atleast_2d(C)
    if np.any(C < 0) or not np.allclose(C, np.round(C)):
        raise ValueError("counts must be non-negative integers")
    lg = lambda a, b: _gammaln(a + 1) - _gammaln(b + 1) - _gammaln(a - b + 1)  # noqa: E731
    E = np.empty(len(C)); S = np.empty(len(C))
    for i, row in enumerate(C):
        x = row[row > 0]
        J = x.sum()
        ldiv = lg(J, depth) if depth <= J else -np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            p1 = np.where(J - x < depth, 0.0, np.exp(lg(J - x, np.minimum(depth, J - x)) - ldiv))
        E[i] = np.sum(1 - p1)
        if se:
            V = np.sum(p1 * (1 - p1))
            Jxx = J - np.add.outer(x, x)
            iu = np.tril_indices(len(x), -1)
            jj = Jxx[iu]
            with np.errstate(over="ignore", invalid="ignore"):
                t2 = np.where(jj < depth, 0.0, np.exp(lg(jj, np.minimum(depth, np.maximum(jj, 0))) - ldiv))
            V = V + 2 * np.sum(t2 - np.outer(p1, p1)[iu])
            S[i] = math.sqrt(max(V, 0.0))
    if one:
        return (float(E[0]), float(S[0])) if se else float(E[0])
    return (E, S) if se else E


# ------------------------------------------------------------------ Poisson-binomial ("k of n patients" rules)
def poisson_binomial(p) -> np.ndarray:
    """Distribution of the number of successes among independent trials with different probabilities.

    E.g. the chance that a "at least k of n patients pass" kill rule passes when patient i passes with
    probability ``p_i`` (from a per-patient planted-effect calibration): ``pmf[k:].sum()``. Exact dynamic
    programming over the trials (no normal approximation).

    Parameters
    ----------
    p : success probability per trial (patient), each in [0, 1].

    Returns
    -------
    numpy array ``pmf`` of length ``len(p) + 1``: ``pmf[k] = P(X = k)``.

    Examples
    --------
    >>> import arsc
    >>> pmf = arsc.stats.poisson_binomial([0.9, 0.8, 0.7, 0.6])
    >>> round(float(pmf[3:].sum()), 4)          # P(at least 3 of 4 pass)
    0.7428
    """
    p = np.asarray(p, float).ravel()
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for i, q in enumerate(p):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - q) + pmf[0:i + 1] * q
        pmf[0] *= (1 - q)
    return pmf


def perm_pvalue(null, obs, alternative: str = "two-sided", exact: bool = False) -> float:
    """p-value of an observed statistic against a null distribution you built yourself (random gene sets, a custom
    permutation scheme, shuffled clone labels).

    Use it for the p whenever the null is not one of the standard schemes (:func:`perm_test`,
    :func:`strat_perm_test`, :func:`signflip_test`, :func:`freedman_lane`): it does the three things hand-written
    versions got wrong. The Monte Carlo p is ``(1 + #extreme) / (B + 1)`` and never 0 (the observed labeling counts
    as one draw); non-finite null values are dropped before counting (a NaN makes every comparison False, so a
    hand count returns p = 0); "as or more extreme" uses a tie tolerance of 1e-12 + 1e-10 * |obs|, so equal sums
    that differ by floating-point rounding count as ties.

    Parameters
    ----------
    null : statistics of the null draws (not including the observed value).
    obs : the observed statistic.
    alternative : ``"two-sided"`` (``|null| >= |obs|``: the null must be centered at 0, e.g. a difference or a
        centered sum), ``"greater"`` (``null >= obs``) or ``"less"`` (``null <= obs``).
    exact : True when ``null`` is the complete enumeration of the randomization distribution (every
        permutation, observed labeling included): then p = ``#extreme / len(null)``.

    Returns
    -------
    float p-value; NaN when ``obs`` is not finite or no null value is finite.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> null = np.array([0.1, -0.3, 0.2, np.nan, -0.05, 0.4])
    >>> arsc.stats.perm_pvalue(null, 0.35)          # (1 + 1) / (5 + 1); the NaN is dropped
    0.3333333333333333
    >>> arsc.stats.perm_pvalue(null, 0.35, alternative="less")   # (1 + 4) / (5 + 1)
    0.8333333333333334
    """
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")
    null = np.asarray(null, float).ravel()
    null = null[np.isfinite(null)]
    obs = float(obs)
    if not np.isfinite(obs) or len(null) == 0:
        return float("nan")
    hits = int(_extreme(null, obs, alternative).sum())
    p = hits / len(null) if exact else (hits + 1) / (len(null) + 1)
    return float(min(p, 1.0))


def _strat_mean_diff(y, G, s, info):
    """Mean over informative strata of mean(y | group 1) - mean(y | group 2), for each row of the 0/1 matrix G."""
    G = np.atleast_2d(G).astype(float)
    tot = np.zeros(G.shape[0])
    for c in info:
        m = s == c
        n1 = G[:, m].sum(1)
        n0 = m.sum() - n1
        s1 = G[:, m] @ y[m]
        tot += s1 / n1 - (y[m].sum() - s1) / n0
    return tot / len(info)


@records_seed("seed")
def strat_perm_test(y, labels, strata, stat="mean_diff", alternative: str = "two-sided", B: int = 20_000,
                    max_exact: int = 20_000, seed: int = 1) -> dict:
    """Permutation test of a two-group difference with labels permuted WITHIN strata (blocks).

    For designs where only within-stratum exchangeability holds: clones within patient (each patient's clones keep
    their patient), patients within batch, sort or cohort, cells within sample. Group sizes are kept in every
    stratum. Exact enumeration of every within-stratum arrangement when there are at most ``max_exact`` of them,
    otherwise ``B`` random ones with the never-zero p ``(1 + #extreme) / (B + 1)``. Strata with only one group
    carry no information and are left out of the statistic (they do not move under the permutation). Units with a
    non-finite ``y`` or a missing label / stratum are dropped first.

    Parameters
    ----------
    y : value per unit (patient, clone, cell).
    labels : group per unit; exactly two distinct values. The statistic is "first minus second" with the label
        values in sorted order, or True minus False for boolean labels.
    strata : stratum per unit (patient for clone-level data, batch for patient-level data).
    stat : ``"mean_diff"`` (default: mean over informative strata of the within-stratum difference of means),
        ``"sum"`` (sum of ``y`` in the first group over informative strata, centered at its permutation mean: the
        stratified linear statistic; it weights strata by size), or a callable
        ``stat(y, is_first_group, strata) -> float``
        (applied to the units that remain after dropping missing values).
    alternative : ``"two-sided"``, ``"greater"`` or ``"less"``.
    B : random within-stratum permutations when not exact.
    max_exact : enumerate when the number of arrangements (product of per-stratum binomial coefficients) is at
        most this.
    seed : seeds a local ``np.random.default_rng`` for random permutations (unused when exact).

    Returns
    -------
    dict with ``stat`` (observed), ``p``, ``exact``, ``n_perm``, ``n_units``, ``n_strata`` (informative strata),
    ``min_p`` (the smallest p the design can give) and ``groups`` (first, second label).

    Examples
    --------
    >>> import numpy as np, arsc
    >>> y = np.array([5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 2.5])
    >>> g = np.array(["R", "R", "NR", "NR", "R", "R", "NR", "NR"])
    >>> s = np.array([1, 1, 1, 1, 2, 2, 2, 2])       # two batches; responders compared within batch
    >>> r = arsc.stats.strat_perm_test(y, g, s)
    >>> r["groups"], r["exact"], r["n_perm"], round(r["p"], 4)
    (('NR', 'R'), True, 36, 0.0556)
    """
    seed = check_seed(seed)
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")
    y = np.asarray(y, float).ravel()
    lab = pd.Series(np.asarray(labels).ravel())
    sx = pd.Series(np.asarray(strata).ravel())
    if not (len(y) == len(lab) == len(sx)):
        raise ValueError("y, labels and strata must have the same length")
    ok = np.isfinite(y) & lab.notna().to_numpy() & sx.notna().to_numpy()
    y, lab, sx = y[ok], lab[ok].to_numpy(), sx[ok].to_numpy()
    vals = list(pd.unique(lab))
    if len(vals) != 2:
        raise ValueError(f"labels must have exactly two distinct values (got {len(vals)})")
    vals = [v.item() if isinstance(v, np.generic) else v for v in vals]  # plain Python labels in the result
    vals = [True, False] if set(vals) == {True, False} else sorted(vals)
    g = lab == vals[0]
    _, s = np.unique(sx, return_inverse=True)
    info = [c for c in range(s.max() + 1) if 0 < g[s == c].sum() < (s == c).sum()]
    out = dict(stat=np.nan, p=np.nan, exact=True, n_perm=0, n_units=int(len(y)), n_strata=len(info),
               min_p=np.nan, groups=(vals[0], vals[1]))
    if not info:
        return out
    inf_mask = np.isin(s, info)
    if callable(stat):
        f = lambda G: np.array([stat(y, row.astype(bool), sx) for row in np.atleast_2d(G)], float)  # noqa: E731
    elif stat == "mean_diff":
        f = lambda G: _strat_mean_diff(y, G, s, info)  # noqa: E731
    elif stat == "sum":
        mu = sum(g[s == c].sum() * y[s == c].mean() for c in info)
        f = lambda G: np.atleast_2d(G)[:, inf_mask].astype(float) @ y[inf_mask] - mu  # noqa: E731
    else:
        raise ValueError("stat must be 'mean_diff', 'sum' or a callable")
    obs = float(f(g)[0])
    out["stat"] = obs
    if not np.isfinite(obs):
        return out
    idx = {c: np.flatnonzero(s == c) for c in info}
    total = 1
    for c in info:
        total *= math.comb(len(idx[c]), int(g[idx[c]].sum()))
    exact = total <= max_exact
    if exact:
        per = [list(itertools.combinations(idx[c], int(g[idx[c]].sum()))) for c in info]
        G = np.zeros((total, len(y)), bool)
        G[:, ~inf_mask] = g[~inf_mask]
        for r, combo in enumerate(itertools.product(*per)):
            for chosen in combo:
                G[r, list(chosen)] = True
    else:
        rng = np.random.default_rng(seed)
        G = np.broadcast_to(g, (B, len(y))).copy()
        for c in info:
            ii = idx[c]
            G[:, ii] = g[ii][np.argsort(rng.random((B, len(ii))), axis=1)]
    null = np.concatenate([f(G[i:i + 4096]) for i in range(0, len(G), 4096)])
    n = len(null)
    fin = null[np.isfinite(null)]
    if exact and len(fin):  # the p of the most extreme arrangement (ties and the mirror image count)
        top = {"two-sided": np.abs(fin).max(), "greater": fin.max(), "less": fin.min()}[alternative]
        min_p = perm_pvalue(fin, top, alternative, exact=True)
    else:
        min_p = 1.0 / (n + 1)
    out.update(p=perm_pvalue(null, obs, alternative, exact=exact), exact=bool(exact), n_perm=int(n),
               min_p=float(min_p))
    return out


@records_seed("seed")
def boot_ci_diff(x, y, f=np.median, paired: bool = False, conf_level: float = 0.95, B: int = 4000,
                 seed: int = 1) -> tuple:
    """Percentile bootstrap CI of ``f(x) - f(y)`` (default: the difference of medians) with patients resampled.

    Unpaired: ``x`` and ``y`` are resampled separately, each within its own group (group sizes kept). Paired
    (``x[i]`` and ``y[i]`` from the same patient): patients are resampled as pairs and the statistic is
    ``f(x*) - f(y*)``; for the median OF the paired differences use :func:`boot_ci` on ``x - y`` (with
    ``f=np.mean`` the two agree exactly). At fewer than ~10 patients per group a percentile bootstrap CI of a
    median is too narrow (it can only take observed values); prefer the exact CI of :func:`hodges_lehmann` for a
    location shift. Non-finite values are dropped (pairwise when paired); fewer than 3 values in a group gives NaN.

    Parameters
    ----------
    x, y : per-patient values of the two groups (or the two conditions when paired).
    f : statistic of one group; ``np.median`` / ``np.mean`` are vectorized, any other callable is applied per
        resample.
    paired : resample pairs (equal lengths required).
    conf_level : central coverage of the percentile interval.
    B : bootstrap resamples.
    seed : seeds a local ``np.random.default_rng``.

    Returns
    -------
    tuple ``(lo, hi)``.

    Examples
    --------
    >>> import numpy as np, arsc
    >>> r = np.array([0.8, 1.1, 0.4, 1.6, 0.9, 1.3])
    >>> nr = np.array([0.2, 0.5, -0.1, 0.3, 0.6])
    >>> lo, hi = arsc.stats.boot_ci_diff(r, nr)
    >>> bool(lo < float(np.median(r) - np.median(nr)) < hi)
    True
    """
    seed = check_seed(seed)
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be in (0, 1)")
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if paired:
        if len(x) != len(y):
            raise ValueError("paired=True needs x and y of equal length")
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
    else:
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if f in (np.median, np.mean):
        stat = lambda M: f(M, axis=1)  # noqa: E731
    else:
        stat = lambda M: np.array([f(r) for r in M], float)  # noqa: E731
    if paired:
        idx = rng.integers(0, len(x), size=(B, len(x)))
        b = stat(x[idx]) - stat(y[idx])
    else:
        ix = rng.integers(0, len(x), size=(B, len(x)))
        iy = rng.integers(0, len(y), size=(B, len(y)))
        b = stat(x[ix]) - stat(y[iy])
    a = (1 - conf_level) / 2
    return tuple(np.quantile(b, [a, 1 - a]))
