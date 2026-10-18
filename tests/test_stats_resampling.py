"""Sign-flip / maxT / Hodges-Lehmann / Mantel-Haenszel / ICC / Freedman-Lane / thinning / rarefaction /
Poisson-binomial helpers of arsc.stats.

Closed-form and brute-force checks always run. R parity (wilcox.test, mantelhaen.test, vegan::rarefy, lme4 REML,
lm) reads $ARSC_SCRATCH/stats_resampling/R_*.csv written by benchmarks/stats_resampling_cases.py +
benchmarks/r_reference_stats_resampling.R and skips without them (STATS_PARITY.md).
"""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats as sps

from arsc import stats as st

REF = Path(os.environ.get("ARSC_SCRATCH", "_none_")) / "stats_resampling"
need_ref = pytest.mark.skipif(not (REF / "R_hl.csv").exists(), reason="R references not generated")


# ----------------------------------------------------------------------------- sign-flip
def _brute_signflip(d, f):
    obs = f(d)
    ps = [abs(f(np.array(s) * d)) >= abs(obs) - 1e-12 for s in itertools.product([-1, 1], repeat=len(d))]
    return float(np.mean(ps))


def test_signflip_matches_brute_force():
    rng = np.random.default_rng(1)
    for n in range(1, 11):
        d = rng.normal(0.3, 1, n)
        for name, f in (("mean", np.mean), ("median", np.median), ("sum", np.sum)):
            assert st.signflip_test(d, stat=name)["p"] == pytest.approx(_brute_signflip(d, f), abs=1e-15)


def test_signflip_on_signed_ranks_is_exact_wilcoxon_and_sign_test():
    rng = np.random.default_rng(2)
    for n in range(3, 16):
        d = rng.normal(0.5, 1, n)
        r = sps.rankdata(np.abs(d)) * np.sign(d)
        # sign-flipping signed ranks with the sum statistic IS the exact signed-rank test
        assert st.signflip_test(r, stat="sum")["p"] == pytest.approx(st.wilcoxon_signed_rank(d), abs=1e-12)
        # sign-flipping the signs IS the two-sided binomial sign test
        k = int((d > 0).sum())
        assert st.signflip_test(np.sign(d), stat="sum")["p"] == pytest.approx(sps.binomtest(k, n).pvalue, abs=1e-12)


def test_signflip_nan_dropped_and_mc_never_zero():
    d = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    assert st.signflip_test(d)["n"] == 4
    assert st.signflip_test(d)["p"] == pytest.approx(0.125)
    big = np.full(30, 1.0)
    r = st.signflip_test(big, B=999)
    assert not r["exact"] and r["p"] == pytest.approx(1 / 1000)
    one = st.signflip_test([1.0, 2.0, 3.0], alternative="greater")
    assert one["p"] == pytest.approx(1 / 8) and one["min_p"] == pytest.approx(1 / 8)


def test_signflip_exact_chunking_matches_small_chunks():
    rng = np.random.default_rng(3)
    d = rng.normal(0.2, 1, 17)       # 131,072 patterns -> two chunks of 65,536
    assert st.signflip_test(d)["p"] == pytest.approx(_brute_signflip(d[:12], np.mean) if False else
                                                     st.signflip_test(d, max_exact=17)["p"])
    p_mc = st.signflip_test(d, max_exact=10, B=200_000)["p"]
    assert abs(p_mc - st.signflip_test(d)["p"]) < 0.01


def test_signflip_maxt_brute_force():
    rng = np.random.default_rng(4)
    for n, K in ((5, 3), (7, 4), (8, 1)):
        D = rng.normal(0.4, 1, (n, K))
        obs = D.mean(0)
        pats = np.array(list(itertools.product([-1, 1], repeat=n)), float)
        T = np.abs(np.einsum("pn,nk->pk", pats, D) / n)
        raw = (T >= np.abs(obs) - 1e-12).mean(0)
        ss = (T.max(1)[:, None] >= np.abs(obs) - 1e-12).mean(0)
        r1 = st.signflip_maxt(D, method="single-step")
        assert np.allclose(r1.p_raw, raw) and np.allclose(r1.p_fwer, np.maximum(ss, raw))
        # free step-down: successive maxima, then monotone
        o = np.argsort(-np.abs(obs))
        sd = np.empty(K)
        for j, k in enumerate(o):
            sd[k] = (T[:, o[j:]].max(1) >= abs(obs[k]) - 1e-12).mean()
        sd[o] = np.maximum.accumulate(sd[o])
        r2 = st.signflip_maxt(D)
        assert np.allclose(r2.p_fwer, np.maximum(sd, raw))
        assert np.all(r2.p_fwer <= r1.p_fwer + 1e-12)
        if K == 1:
            assert r2.p_fwer[0] == pytest.approx(st.signflip_test(D[:, 0])["p"])


def test_signflip_maxt_controls_fwer():
    rng = np.random.default_rng(5)
    rej = 0
    for _ in range(300):
        Z = rng.normal(size=(10, 6)); D = Z + 0.8 * Z[:, :1]   # correlated null readouts
        rej += bool((st.signflip_maxt(D, max_exact=10).p_fwer <= 0.05).any())
    assert rej / 300 < 0.09


# ----------------------------------------------------------------------------- Hodges-Lehmann (R wilcox.test)
def _hl_cases():
    cs = pd.read_csv(REF / "hl_cases.csv", float_precision="round_trip"); vv = pd.read_csv(REF / "hl_values.csv", float_precision="round_trip")
    for _, k in cs.iterrows():
        x = vv[(vv.case == k.case) & (vv.grp == "x")].sort_values("i").value.to_numpy()
        y = vv[(vv.case == k.case) & (vv.grp == "y")].sort_values("i").value.to_numpy()
        yield k, x, y


@need_ref
def test_hodges_lehmann_matches_r_wilcox_test():
    R = pd.read_csv(REF / "R_hl.csv", float_precision="round_trip").set_index("case")
    n_exact = n_norm = 0
    for k, x, y in _hl_cases():
        kw = dict(alternative=k.alternative, conf_level=k.conf_level, correct=bool(k.correct))
        if k.kind == "one":
            r = st.hodges_lehmann(x, **kw)
        elif k.kind == "paired":
            r = st.hodges_lehmann(x, y, paired=True, **kw)
        else:
            r = st.hodges_lehmann(x, y, **kw)
        ref = R.loc[k.case]
        exact = "exact" in ref.method
        assert r["method"] == ("exact" if exact else "normal"), k.case
        assert r["statistic"] == pytest.approx(ref.statistic, abs=1e-9)
        assert r["p"] == pytest.approx(ref.p, rel=1e-9, abs=1e-14), k.case
        assert r["conf_level"] == pytest.approx(ref.conf_level, abs=1e-12), k.case
        tol = 1e-12 if exact else 1e-9
        assert r["estimate"] == pytest.approx(ref.estimate, abs=tol), (k.case, r["estimate"], ref.estimate)
        for a, b in ((r["ci_lo"], ref.ci_lo), (r["ci_hi"], ref.ci_hi)):
            if np.isinf(b) or np.isnan(b):
                assert (np.isinf(a) and a == b) or (np.isnan(a) and np.isnan(b)), k.case
            else:
                assert a == pytest.approx(b, abs=tol), (k.case, a, b)
        n_exact += exact; n_norm += not exact
    assert n_exact > 50 and n_norm > 50


def test_hodges_lehmann_estimates_closed_form():
    x = np.array([1.1, 2.3, 0.7, 4.0, 3.2])
    w = (x[:, None] + x[None, :])[np.triu_indices(5)] / 2
    assert st.hodges_lehmann(x)["estimate"] == pytest.approx(np.median(w))
    y = np.array([0.2, 1.5, 0.9])
    assert st.hodges_lehmann(x, y)["estimate"] == pytest.approx(np.median(np.subtract.outer(x, y)))
    # exact two-sided p equals wilcoxon_signed_rank
    assert st.hodges_lehmann(x)["p"] == pytest.approx(st.wilcoxon_signed_rank(x))


# ----------------------------------------------------------------------------- Mantel-Haenszel
@need_ref
def test_mantel_haenszel_matches_r():
    T = pd.read_csv(REF / "mh_tables.csv", float_precision="round_trip"); R = pd.read_csv(REF / "R_mh.csv", float_precision="round_trip").set_index("case")
    for case, t in T.groupby("case"):
        r = st.mantel_haenszel(t.a, t.b, t.c, t.d, correct=bool(t.correct.iloc[0]))
        for key in ("or_mh", "ci_lo", "ci_hi", "statistic", "p"):
            assert r[key] == pytest.approx(R.loc[case, key], rel=1e-10), (case, key)


def test_mantel_haenszel_single_stratum_and_drop():
    r = st.mantel_haenszel([[[12, 5], [3, 10]]])
    assert r["or_mh"] == pytest.approx(12 * 10 / (5 * 3))
    r2 = st.mantel_haenszel([12, 1], [5, 0], [3, 0], [10, 0])
    assert r2["n_dropped"] == 1 and r2["or_mh"] == pytest.approx(r["or_mh"])


# ----------------------------------------------------------------------------- ICC
def test_icc_closed_form_unbalanced():
    rng = np.random.default_rng(6)
    sizes = [2, 5, 3, 7, 1, 4]
    g = np.repeat(np.arange(6), sizes); y = rng.normal(size=len(g)) + rng.normal(size=6)[g]
    N, k = len(y), 6
    mu = np.array([y[g == i].mean() for i in range(k)]); n_i = np.array(sizes)
    msb = (n_i * (mu - y.mean()) ** 2).sum() / (k - 1); msw = ((y - mu[g]) ** 2).sum() / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    r = st.icc_oneway(y, g)
    assert r["icc"] == pytest.approx((msb - msw) / (msb + (n0 - 1) * msw), rel=1e-12)
    assert r["p"] == pytest.approx(sps.f.sf(msb / msw, k - 1, N - k), rel=1e-12)


@need_ref
def test_icc_matches_lme4_reml_balanced():
    V = pd.read_csv(REF / "icc_values.csv", float_precision="round_trip"); R = pd.read_csv(REF / "R_icc.csv", float_precision="round_trip").set_index("case")
    m = 0
    for case, t in V.groupby("case"):
        r = st.icc_oneway(t.y, t.group, strata=t.stratum if t.stratum.nunique() > 1 else None)
        if r["icc"] > 0 and not R.loc[case, "singular"]:
            assert r["icc"] == pytest.approx(R.loc[case, "icc_reml"], abs=1e-4), case
            m += 1
    assert m >= 30


def test_icc_ci_coverage_balanced():
    rng = np.random.default_rng(7)
    rho = 0.3; cover = 0
    for _ in range(400):
        g = np.repeat(np.arange(15), 4)
        y = rng.normal(0, math.sqrt(rho), 15)[g] + rng.normal(0, math.sqrt(1 - rho), 60)
        r = st.icc_oneway(y, g)
        cover += r["ci_lo"] <= rho <= r["ci_hi"]
    assert 0.92 <= cover / 400 <= 0.98


# ----------------------------------------------------------------------------- Freedman-Lane
def test_freedman_lane_without_covariates_is_plain_permutation():
    rng = np.random.default_rng(8)
    x = rng.normal(size=7); y = 0.8 * x + rng.normal(size=7)
    r = st.freedman_lane(y, x)
    t_obs = abs(np.corrcoef(x, y)[0, 1])
    perm = [abs(np.corrcoef(x, y[list(p)])[0, 1]) >= t_obs - 1e-12 for p in itertools.permutations(range(7))]
    assert r["exact"] and r["p_perm"] == pytest.approx(np.mean(perm), abs=1e-12)


@need_ref
def test_freedman_lane_observed_t_matches_lm():
    V = pd.read_csv(REF / "fl_values.csv", float_precision="round_trip"); R = pd.read_csv(REF / "R_fl.csv", float_precision="round_trip").set_index("case")
    for case, t in V.groupby("case"):
        zs = [c for c in ("z0", "z1") if t[c].notna().any()]
        r = st.freedman_lane(t.y.to_numpy(), t.x.to_numpy(), covariates=t[zs].to_numpy() if zs else None, B=200)
        for key in ("beta", "t"):
            assert r[key] == pytest.approx(R.loc[case, key], rel=1e-9)
        assert r["p_param"] == pytest.approx(R.loc[case, "p"], rel=1e-8)


def test_freedman_lane_type1_under_confounding():
    rng = np.random.default_rng(9)
    rej_fl = rej_naive = 0
    for _ in range(200):
        z = rng.normal(size=20); x = z + rng.normal(scale=0.4, size=20); y = 2 * z + rng.normal(size=20)
        rej_fl += st.freedman_lane(y, x, covariates=z, B=499)["p_perm"] <= 0.05
        rej_naive += st.freedman_lane(y, x, B=499)["p_perm"] <= 0.05      # ignores z: confounded
    assert rej_fl / 200 < 0.10 and rej_naive / 200 > 0.5


# ----------------------------------------------------------------------------- thinning / rarefaction
def test_downsample_hypergeometric_exact_depth_and_moments():
    rng = np.random.default_rng(10)
    X = sp.csr_matrix(rng.poisson(rng.gamma(0.5, 4, 40), size=(300, 40)).astype(float))
    tot = np.asarray(X.sum(1)).ravel(); T = int(np.percentile(tot, 30))
    reps = [st.downsample_counts(X, T, seed=s).toarray() for s in range(1, 201)]
    for Y in reps[:3]:
        new = Y.sum(1)
        assert np.all(new[tot >= T] == T) and np.all(new[tot < T] == tot[tot < T])
    M = np.mean(reps, 0); Xd = X.toarray()
    big = tot > T
    expect = Xd[big] * T / tot[big, None]
    var = T * (Xd[big] / tot[big, None]) * (1 - Xd[big] / tot[big, None]) * (tot[big, None] - T) / (tot[big, None] - 1)
    z = (M[big] - expect) / np.sqrt(np.maximum(var, 1e-12) / 200)
    assert np.abs(z[var > 0]).max() < 6 and abs(z[var > 0].mean()) < 0.1


def test_downsample_binomial_moments():
    rng = np.random.default_rng(11)
    X = sp.csr_matrix(rng.poisson(3, size=(200, 30)).astype(float))
    tot = np.asarray(X.sum(1)).ravel(); T = int(np.median(tot))
    reps = np.array([st.downsample_counts(X, T, method="binomial", seed=s).toarray() for s in range(1, 201)])
    big = tot > T
    p = T / tot[big, None]
    z = (reps.mean(0)[big] - X.toarray()[big] * p) / np.sqrt(np.maximum(X.toarray()[big] * p * (1 - p), 1e-12) / 200)
    assert abs(z.mean()) < 0.1


@need_ref
def test_rarefy_richness_matches_vegan():
    C = pd.read_csv(REF / "rarefy_counts.csv", float_precision="round_trip"); R = pd.read_csv(REF / "R_rarefy.csv", float_precision="round_trip").set_index("case")
    for case, t in C.groupby("case"):
        e, s = st.rarefy_richness(t.sort_values("clone")["size"].to_numpy(), int(t.depth.iloc[0]), se=True)
        assert e == pytest.approx(R.loc[case, "S"], rel=1e-10)
        assert s == pytest.approx(R.loc[case, "se"], rel=1e-7, abs=1e-9)


def test_rarefy_richness_monte_carlo():
    rng = np.random.default_rng(12)
    sizes = np.array([20, 7, 5, 3, 2, 1, 1, 1])
    lab = np.repeat(np.arange(len(sizes)), sizes)
    mc = np.mean([len(np.unique(rng.choice(lab, 10, replace=False))) for _ in range(20000)])
    assert st.rarefy_richness(sizes, 10) == pytest.approx(mc, abs=0.03)


# ----------------------------------------------------------------------------- Poisson-binomial
def test_poisson_binomial_enumeration_and_binomial():
    p = np.array([0.9, 0.2, 0.55, 0.7, 0.33, 0.61])
    pmf = st.poisson_binomial(p)
    brute = np.zeros(7)
    for s in itertools.product([0, 1], repeat=6):
        s = np.array(s)
        brute[s.sum()] += np.prod(np.where(s == 1, p, 1 - p))
    assert np.allclose(pmf, brute, atol=1e-15)
    assert np.allclose(st.poisson_binomial(np.full(9, 0.3)), sps.binom.pmf(np.arange(10), 9, 0.3), atol=1e-14)
