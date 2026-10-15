"""Patient-level statistics + pseudobulk DE ports (arsc.stats / arsc.de).

Fast unit tests run anywhere.  The R-parity tests read references produced by
scratch/arsc/stats_parity/ref.R + ref_small.R (DESeq2 1.42, limma 3.58, edgeR 4.0, fgsea 1.28, lme4 1.1) and the
pseudobulk exports prep.py / prep2.py; they skip when those files are missing.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_stats_de.py   (-m "not slow" skips DESeq2/GLMM parity)
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from arsc import de, stats

REF = Path(os.environ.get("ARSC_SCRATCH", "__unset__")) / "stats_parity"
have_ref = pytest.mark.skipif(not (REF / "R_ct_voom.csv").exists(), reason="R reference outputs not built")
GENESETS = de.GENESET_CACHE / "Homo_sapiens__H__all.csv.gz"


# ------------------------------------------------------------------ unit tests
def test_wilcoxon_exact_matches_scipy_and_hand_value():
    x = np.array([1.2, -0.4, 2.2, 3.1, 0.9, 1.7, 2.5])
    assert stats.wilcoxon_signed_rank(x) == pytest.approx(sps.wilcoxon(x, method="exact").pvalue, abs=1e-12)
    assert stats.wilcoxon_signed_rank(np.arange(1.0, 6.0)) == pytest.approx(0.0625)   # n=5, all positive: 2/2^5
    assert stats.wilcoxon_signed_rank(np.arange(1.0, 8.0)) == pytest.approx(2 / 2 ** 7)


def test_wilcoxon_ties_zeros_use_normal_approx_like_R():
    # R falls back to the normal approximation (tie-corrected variance, continuity correction) with zeros/ties
    x = np.array([0, 1, 1, 2, 3, -1, 2.5])
    nz = x[x != 0]
    r = sps.rankdata(np.abs(nz))
    t = float(r[nz > 0].sum()); n = len(nz)
    _, tc = np.unique(r, return_counts=True)
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(tc ** 3 - tc) / 48)
    z = (t - n * (n + 1) / 4 - 0.5) / sig
    assert stats.wilcoxon_signed_rank(x) == pytest.approx(2 * sps.norm.sf(z))


def test_paired_test_basic_and_floor():
    rows = []
    for i, d in enumerate([0.5, 0.8, 1.1, 0.2, 0.9]):
        rows += [dict(patient_alias=f"p{i}", timepoint="A", value=0.2, celltype="x"),
                 dict(patient_alias=f"p{i}", timepoint="B", value=0.2 + d, celltype="x")]
    r = stats.paired_test(pd.DataFrame(rows), "value", "A", "B", by="celltype")
    assert r.n_patients[0] == 5 and r.k_up[0] == 5 and r.min_possible_p[0] == pytest.approx(0.0625)
    assert r.p_wilcox_exact[0] == pytest.approx(0.0625)


def test_perm_exact_enumeration_and_arm_contrast():
    x, y = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]
    r = stats.perm_test(x, y)
    assert r["p_perm"] == pytest.approx(4 / math.comb(7, 3))  # 4 splits reach |median diff| >= 3.5 (checked by hand)
    a = stats.arm_contrast(x, y)
    assert a["diff_medians"] == pytest.approx(-3.5) and a["loo_sign_reversals"] == 0


def test_composition_zero_fill_and_floor():
    obs = pd.DataFrame({"pt": ["a"] * 60 + ["b"] * 60 + ["c"] * 10, "tp": ["t"] * 130,
                        "ct": ["x"] * 40 + ["y"] * 20 + ["x"] * 60 + ["x"] * 10})
    c = stats.composition(obs, "ct", ["pt", "tp"], min_parent=50)
    assert set(c.pt) == {"a", "b"} and len(c) == 4
    assert c[(c.pt == "b") & (c.celltype == "y")].k.item() == 0
    assert c[(c.pt == "a") & (c.celltype == "x")].prop.item() == pytest.approx(40 / 60)


def test_glmm_recovers_simulated_effects():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(25):
        u = rng.normal(0, 0.8)
        for t in (0, 1):
            e = rng.normal(0, 0.3)
            n = 400
            k = rng.binomial(n, 1 / (1 + np.exp(-(-1.0 + 0.7 * t + u + e))))
            rows.append(dict(patient_alias=f"p{p}", timepoint=f"t{t}", k=k, n=n))
    d = pd.DataFrame(rows)
    f = stats.glmm_prop(d, "timepoint")
    assert abs(f.Estimate[1] - 0.7) < 3 * f["Std. Error"][1]
    assert 0.4 < f.attrs["sigma"]["patient_alias"] < 1.3
    cond = stats.glmm_prop(d, "timepoint", vcov_method="conditional")
    assert (f["Std. Error"] >= cond["Std. Error"] * 0.95).all()


def test_fgsea_es_by_hand():
    st = pd.Series([3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0, 0.1, -0.1], index=list("abcdefghij"))
    r = de.fgsea(st, {"top": ["a", "b", "c"]}, min_size=3, max_size=5, nperm=500)
    # hits at ranks 0,1,2 of N=10: the running sum reaches 1 after the last hit and no misses preceded them
    assert r.ES[0] == pytest.approx(1.0) and r.NES[0] > 1 and r.leadingEdge[0] == "a,b,c"
    r2 = de.fgsea(st, {"bottom": ["h", "g", "f"]}, min_size=3, max_size=5, nperm=500)
    assert r2.ES[0] < 0 and r2.NES[0] < 0


def test_tmm_and_filter_by_expr_shapes():
    rng = np.random.default_rng(0)
    c = rng.poisson(rng.gamma(2, 20, size=(500, 8))).astype(float)
    nf = de.tmm_factors(c)
    assert nf.shape == (8,) and np.exp(np.log(nf).mean()) == pytest.approx(1.0)
    X = np.column_stack([np.ones(8), np.arange(8) % 2])
    assert de.filter_by_expr(c, X).dtype == bool


def test_voom_runs_and_detects_planted_effect():
    rng = np.random.default_rng(1)
    G, n = 600, 12
    pat = np.repeat([f"p{i}" for i in range(6)], 2); tp = np.tile(["a", "b"], 6)
    mu = rng.gamma(2, 60, G)[:, None] * np.ones((G, n))
    mu[:30, tp == "b"] *= 3
    cnt = pd.DataFrame(rng.poisson(mu), index=[f"g{i}" for i in range(G)], columns=[f"s{i}" for i in range(n)])
    cd = pd.DataFrame({"patient": pat, "tp": tp}, index=cnt.columns)
    r = de.voom(cnt, cd, "~ tp", block="patient", coef=("tp", "b"))
    assert len(set(r.gene[:30]) & {f"g{i}" for i in range(30)}) >= 25
    assert -1 < r.attrs["consensus_cor"] < 1


@pytest.mark.skipif(not GENESETS.exists(), reason="msigdb cache not built")
def test_msigdb_cache_roundtrip():
    h = de.msigdb("H")
    assert len(h) == 50 and "HALLMARK_APOPTOSIS" in h


# ------------------------------------------------------------------ R parity
def _load(tag):
    f = {"tp": ("pb_counts.csv.gz", "pb_coldata.csv"), "ct": ("ct_counts.csv.gz", "ct_coldata.csv")}[tag]
    cnt = pd.read_csv(REF / f[0], index_col=0)
    cd = pd.read_csv(REF / f[1], index_col=0)
    cnt.columns = cnt.columns.astype(str)
    cd.index = cd.index.astype(str)
    return cnt, cd


@have_ref
def test_small_vectors_wilcoxon_and_perm_exact_vs_R():
    S = pd.read_csv(REF / "R_small.csv")
    pw = np.array([stats.wilcoxon_signed_rank(np.array(r.x.split(";"), float)) for r in S.itertuples()])
    assert np.nanmax(np.abs(pw - S.p_wilcox)) < 1e-12
    for r in S.itertuples():
        q = stats.perm_test(np.array(r.xp.split(";"), float), np.array(r.yp.split(";"), float))
        assert abs(q["p_perm"] - r.p_perm) < 1e-12 and abs(q["stat"] - r.stat) < 1e-10


@have_ref
@pytest.mark.parametrize("tag,formula,coef,block", [
    ("ct", "~ cell_type", ("cell_type", "gd"), "pt"),
    pytest.param("tp", "~ timepoint", ("timepoint", "Follow Up 1"), "patient_alias", marks=pytest.mark.slow)])
def test_voom_dupcor_vs_limma(tag, formula, coef, block):
    cnt, cd = _load(tag)
    v = de.voom(cnt, cd, formula, block=block, coef=coef)
    R = pd.read_csv(REF / f"R_{tag}_voom.csv")
    meta = pd.read_csv(REF / f"R_{tag}_voom_meta.csv").iloc[0]
    m = v.merge(R, on="gene", suffixes=("", "_R"))
    assert len(m) == len(R) == int(meta.kept)
    assert abs(v.attrs["consensus_cor"] - meta.consensus) < 2e-3
    assert abs(v.attrs["df_prior"] - meta.df_prior) < 0.01
    assert np.corrcoef(m.logFC, m.logFC_R)[0, 1] > 0.99999 and np.corrcoef(m.t, m.t_R)[0, 1] > 0.99999
    assert np.abs(m["adj.P.Val"] - m["adj.P.Val_R"]).max() < 5e-3


@have_ref
@pytest.mark.slow
def test_deseq2_paired_vs_R():
    cnt, cd = _load("ct")
    r = de.deseq2(cnt, cd, "~ pt + cell_type", ("cell_type", "gd", "cd8_tem"))
    R = pd.read_csv(REF / "R_ct_deseq2.csv")
    m = r.reset_index(drop=True).merge(R, on="gene", suffixes=("", "_R"))
    assert len(m) == len(R)
    assert np.corrcoef(m.log2FoldChange, m.log2FoldChange_R)[0, 1] > 0.999
    assert np.corrcoef(m.stat, m.stat_R)[0, 1] > 0.998
    assert set(m.gene[m.padj < .1]) == set(m.gene[m.padj_R < .1])


@have_ref
@pytest.mark.skipif(not GENESETS.exists(), reason="msigdb cache not built")
@pytest.mark.parametrize("tag", ["tp", "ct"])
def test_fgsea_hallmark_vs_R(tag):
    st = pd.read_csv(REF / f"R_{tag}_stat.csv").set_index("gene").stat
    f = de.fgsea(st, de.msigdb("H"), nperm=10000)
    R = pd.read_csv(REF / f"R_{tag}_fgsea_H.csv")
    m = f.merge(R, on="pathway", suffixes=("", "_R"))
    assert len(m) == len(R) == len(f)
    assert np.abs(m.ES - m.ES_R).max() < 1e-9 and (m["size"] == m.size_R).all()
    assert np.corrcoef(m.NES, m.NES_R)[0, 1] > 0.9999
    assert set(f.pathway[f.padj < .05]) == set(R.pathway[R.padj < .05])


@have_ref
@pytest.mark.slow
def test_glmm_lmm_vs_lme4():
    comp = pd.read_csv(REF / "comp.csv")
    G = pd.read_csv(REF / "R_glmer.csv")
    ct = "cd8_tem"
    d = comp[comp.celltype == ct].copy()
    d = d[d.patient_alias.map(d.patient_alias.value_counts()) >= 2]
    d["sample_id"] = np.arange(len(d))
    f = stats.glmm_prop(d, "timepoint")
    R = G[G.celltype == ct].reset_index(drop=True)
    assert np.abs(f.Estimate.values - R.Estimate.values).max() < 1e-3
    assert np.abs(f["Std. Error"].values - R["Std. Error"].values).max() < 5e-3
    assert abs(f.attrs["sigma"]["patient_alias"] - R.sd_patient[0]) < 1e-3
    p = np.clip(d.prop, 1e-3, 1 - 1e-3)
    d["lp"] = np.log(p / (1 - p))
    l = stats.lmm_score(d, "lp ~ timepoint")
    L = pd.read_csv(REF / f"R_lmer_{ct}.csv")
    assert np.abs(l.Estimate.values - L.Estimate.values).max() < 1e-6
