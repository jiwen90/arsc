"""arsc.clones / arsc.genesets: exact parity with the R methods (tools/R/ar_methods.R, ar.R clone helpers).

R references: benchmarks/r_reference_clones.R -> $ARSC_REFS/clones_ref (python benchmarks/regen_refs.py clones_ref).
Published Cyclone assignment = column ``k`` of cache/cd8_nn_labeled_FORCYCLONE_FINAL_cyclone.meta.csv.gz
(identical to the published @misc$clonestats k).  Data-dependent tests skip when inputs are absent.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_clones.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from arsc import clones as C
from arsc import genesets as G

# data-dependent tests read their paths from the environment (tests/conftest.py, tests/local.env)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from refutil import CLONES_REF as REF, missing  # noqa: E402


def _gz(p: Path) -> Path:
    """regen_refs.py stores these references gzipped (<name>.csv.gz)."""
    g = p.with_name(p.name + ".gz")
    return g if g.exists() or not p.exists() else p

META = Path(os.environ.get("ARSC_TEST_CLONES_META", "__unset__"))
V3 = Path(os.environ.get("ARSC_DATA_V3", "__unset__"))
TPS = [0, 3, 6, 9]
have_meta = META.exists()
have_data = have_meta and (_gz(REF / "cyclone_pub_k.csv")).exists()
_SKIP_META = None if have_meta else f"missing the CD8 clone metadata {META} (set ARSC_TEST_CLONES_META; tests/local.env)"
_SKIP_DATA = _SKIP_META or missing(_gz(REF / "cyclone_pub_k.csv"), regen="clones_ref")
needs_meta = pytest.mark.skipif(bool(_SKIP_META), reason=_SKIP_META or "")
needs_data = pytest.mark.skipif(bool(_SKIP_DATA), reason=_SKIP_DATA or "")


# ------------------------------------------------------------------------------------------ synthetic
def test_fisher_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 15, 60)
    b = rng.integers(0, 15, 60)
    N1, N2 = 400, 650
    mine = C.fisher_exact_2x2(a, b, N1, N2)
    ref = np.array([stats.fisher_exact([[x, N1 - x], [y, N2 - y]])[1] for x, y in zip(a, b)])
    assert np.abs(mine - ref).max() < 1e-9


def test_bh_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.5, np.nan])
    q = C.p_adjust_bh(p)
    assert np.allclose(q[:5], [0.05, 0.0666666667, 0.0666666667, 0.25, 0.5]) and np.isnan(q[5])


def _synthetic():
    shapes = {"early": [1, 12, 4, 2], "late": [1, 2, 5, 14], "down": [14, 4, 2, 1]}
    rng = np.random.default_rng(1)
    rows = []
    for pt in ["P1", "P2", "P3", "P4"]:
        for s, sh in shapes.items():
            for j in range(4):
                size = np.array(sh) * rng.integers(2, 4)
                for t, n in zip(TPS, size):
                    rows += [(pt, t, f"{pt}_{s}{j}")] * int(n)
        for t in TPS:
            rows += [(pt, t, f"{pt}_bg{t}_{i}") for i in range(600)]
    d = pd.DataFrame(rows, columns=["patient_alias", "timepoint", "clone_id"])
    d["treatment"] = np.where(d.patient_alias.isin(["P1", "P2"]), "a", "b")
    d["response"] = "PR"
    return d


def test_cyclone_synthetic_recovers_shapes():
    d = _synthetic()
    cy = C.cyclone(d, timepoints=TPS, n_clust=3, verbose=False)
    a = cy.assignment[cy.assignment.k != "Other"]
    assert cy.mat.shape == (4, 48) and len(a) == 48
    truth = a.clone_id.str.replace(r"^P\d_([a-z]+)\d$", r"\1", regex=True)
    assert C.ari(truth, a.k) == 1.0
    assert set(cy.patient_table().columns) >= {"patient_alias", "Traj 1", "Traj 2", "Traj 3"}
    assert (cy.cell_k == "Other").sum() > 0


def test_startrac_toy():
    d = pd.DataFrame({"clone_id": ["a", "a", "b", "b", "b", "c"], "state": ["x", "y", "x", "x", "y", "y"],
                      "patient_alias": "p"})
    cl, pw = C.startrac(d, state_col="state", min_cells=1)
    x = cl.set_index("state")
    # clone a: 1 cell in each state (H=1 bit); b: 2x,1y (H=0.918); c: only y (H=0)
    h_b = -(2 / 3 * np.log2(2 / 3) + 1 / 3 * np.log2(1 / 3))
    assert np.isclose(x.loc["x", "tran"], (1 * 1 + 2 * h_b) / 3)
    assert np.isclose(x.loc["x", "expa"], 1 - h_b / np.log2(3))
    assert pw.n_shared_clones.iloc[0] == 2


def test_genesets_formats(tmp_path):
    (tmp_path / "a.json").write_text('{"s1": {"geneSymbols": ["A", "B", "A"]}, "s2": ["C"]}')
    (tmp_path / "m.csv").write_text("Gene,C,Module\nG1,1,1.0\nG2,1,-1\nG3,1,2.0\nG4,1,1.0\n")
    (tmp_path / "x.gmt").write_text("S\tna\tA\tB\n")
    assert G.read_genesets(tmp_path / "a.json") == {"s1": ["A", "B"], "s2": ["C"]}
    assert G.read_genesets(tmp_path / "m.csv") == {"module1": ["G1", "G4"], "module2": ["G3"]}
    assert G.read_genesets(tmp_path / "x.gmt") == {"S": ["A", "B"]}
    with pytest.raises(FileNotFoundError):
        G.msigdb("H", cache_dir=tmp_path / "none")
    (tmp_path / "h.all.v2024.1.Hs.symbols.gmt").write_text("HALLMARK_X\tna\tA\tB\n")
    assert G.msigdb("H", cache_dir=tmp_path) == {"HALLMARK_X": ["A", "B"]}


def test_clone_tissue_entropy():
    d = pd.DataFrame({"clone_id": list("aabbbc"), "tissue": ["t", "p", "t", "t", "t", "p"]})
    e = C.clone_tissue_entropy(C.clone_tissue_matrix(d))
    assert np.isclose(e.loc["a", "entropy"], 1.0) and e.loc["b", "entropy"] == 0 and e.loc["a", "entropy_norm"] == 1.0


# ------------------------------------------------------------------------------------------ real data
@pytest.fixture(scope="module")
def meta():
    m = pd.read_csv(META, usecols=["cell", "patient_alias", "timepoint", "treatment", "response", "clone_id", "k",
                                   "metaclusters_exh", "TRA_chain_cdr3_nt", "TRB_chain_cdr3_nt"])
    m["week"] = m.timepoint.astype(str)
    return m


RUNS = {
    "pub": dict(n_clust=6, custom_order=[4, 5, 1, 6, 2, 3], outlier_sd=2.5),
    "noorder": dict(n_clust=6),
    "sclone": dict(n_clust=6, scale="clone"),
    "snone": dict(n_clust=4, scale="none", min_c=0.5),
    "pam": dict(n_clust=6, method="pam"),
    "abs": dict(n_clust=6, method="absolute"),
    "reg": dict(n_clust=6, reg=0.5, min_fc=1.5),
}


@pytest.fixture(scope="module")
def cy_pub(meta):
    return C.cyclone(meta, timepoints=TPS, verbose=False, **RUNS["pub"])


@needs_meta
def test_cyclone_reproduces_published_trajectories(meta, cy_pub):
    """Published settings of validate_cyclone.R -> identical clone- and cell-level assignment (ARI 1.0)."""
    pub = meta[meta.clone_id.notna()].groupby("clone_id").k.first()
    mine = cy_pub.clonestats[["clone_id", "k"]].drop_duplicates().set_index("clone_id").k.reindex(pub.index)
    assert mine.notna().all() and (mine == pub).all()
    assert C.ari(pub.values, mine.values) == 1.0
    assert (cy_pub.cell_k[meta.clone_id.notna()] == meta.k[meta.clone_id.notna()]).all()
    assert (pub != "Other").sum() == 500 and len(pub) == 32080


@needs_data
@pytest.mark.parametrize("tag", list(RUNS))
def test_cyclone_variants_match_R(meta, tag):
    cy = C.cyclone(meta, timepoints=TPS, verbose=False, **RUNS[tag])
    rk = pd.read_csv(_gz(REF / f"cyclone_{tag}_k.csv")).set_index("clone_id").k
    mk = cy.clonestats[["clone_id", "k"]].drop_duplicates().set_index("clone_id").k.reindex(rk.index)
    assert (rk == mk).all()
    rc = pd.read_csv(_gz(REF / f"cyclone_{tag}_counts.csv"))
    a = cy.counts.assign(timepoint=lambda x: x.timepoint.astype(int))
    j = rc.merge(a, on=["patient_alias", "treatment", "response", "k", "timepoint"], how="outer", suffixes=("_r", "_p"))
    assert len(j) == len(rc) == len(a) and (j.n_clones_r == j.n_clones_p).all()
    assert np.abs(j.perc_r - j.perc_p).max() < 1e-9
    rp = pd.read_csv(_gz(REF / f"cyclone_{tag}_ptable.csv")).sort_values("patient_alias").reset_index(drop=True)
    pp = cy.patient_table().sort_values("patient_alias").reset_index(drop=True)
    for c in [c for c in rp.columns if c.startswith("Traj")]:
        assert (rp[c].to_numpy() == pp[c].to_numpy()).all()


@needs_data
def test_cyclone_clonestats_and_cor_match_R(cy_pub):
    mc = pd.read_csv(_gz(REF / "cyclone_pub_matcor.csv"), index_col=0)
    assert np.abs(mc.values - cy_pub.mat_cor.loc[mc.index, mc.columns].values).max() < 1e-12
    rs = pd.read_csv(_gz(REF / "cyclone_pub_clonestats.csv"))
    cs = cy_pub.clonestats.assign(timepoint=lambda x: x.timepoint.astype(int))
    j = rs.merge(cs, on=["clone_id", "patient_alias", "timepoint"], suffixes=("_r", "_p"))
    assert len(j) == len(rs) == len(cs)
    assert np.abs(j.perc_r - j.perc_p).max() < 1e-9 and np.abs(j.delta_c_r - j.delta_c_p).max() < 1e-9
    big = 1e12
    assert np.allclose(j.delta_fc_r.replace(np.inf, big), j.delta_fc_p.replace(np.inf, big), rtol=1e-12)
    assert (j.n_max_r == j.n_max_p).all()


@needs_data
def test_cyclone_states_match_R(meta, cy_pub):
    st = C.cyclone_states(cy_pub, meta, "metaclusters_exh").assign(timepoint=lambda x: x.timepoint.astype(int))
    r = pd.read_csv(_gz(REF / "cyclone_states_fc.csv"))
    j = r.merge(st, on=["patient_alias", "treatment", "timepoint", "k", "state"], suffixes=("_r", "_p"))
    assert len(j) == len(r) == len(st) and np.abs(j.value_r - j.value_p).max() < 1e-9


@needs_data
def test_pair_test_exact_vs_R(meta):
    r = pd.read_csv(_gz(REF / "pairtest_12-153_0_3.csv"))
    mp = meta[(meta.patient_alias == "12-153") & meta.clone_id.notna()]
    mine = C.pair_test(mp.clone_id[mp.timepoint == 0].to_numpy(), mp.clone_id[mp.timepoint == 3].to_numpy())
    j = r.merge(mine, on="clone", suffixes=("_r", "_p"))
    assert len(j) == len(r) == len(mine)
    assert np.abs(j.chi_r - j.chi_p).max() < 1e-10 and np.abs(j.p_r - j.p_p).max() < 1e-9
    assert np.abs(C.p_adjust_bh(r.p.to_numpy()) - r.fdr.to_numpy()).max() < 1e-12


@needs_data
def test_hd_null_vs_R(meta):
    files = [V3 / "tcr" / "HD" / f"P6 {t} clonotypes.csv" for t in ("T1", "T2")]
    if not all(f.exists() for f in files):
        pytest.skip(f"missing HD clonotype files {[str(f) for f in files]} (ARSC_DATA_V3)")
    hd = C.read_10x_clonotypes(files)
    assert hd.groupby("timepoint").size().to_dict() == {"T1": 5313, "T2": 3118}
    full = pd.read_csv(_gz(REF / "pairtest_hd_P6_full.csv"))
    mine = C.pair_test(hd.clone_id[hd.timepoint == "T1"].to_numpy(), hd.clone_id[hd.timepoint == "T2"].to_numpy())
    j = full.merge(mine, on="clone", suffixes=("_r", "_p"))
    assert len(j) == len(full) and np.abs(j.chi_r - j.chi_p).max() < 1e-10 and np.abs(j.p_r - j.p_p).max() < 1e-9
    m = meta.assign(clone_pair=meta.TRA_chain_cdr3_nt + " " + meta.TRB_chain_cdr3_nt)
    _, pat = C.clone_expansion_null(m, hd, clone_col="clone_pair", time_col="week", hd_time_col="timepoint",
                                    hd_clone_col="clone_id", reps=5, seed=1)
    r = pd.read_csv(_gz(REF / "expnull_patients.csv"))
    for c in ("t1", "t2"):
        r[c] = r[c].astype(int)
        pat[c] = pat[c].astype(int)
    j = r.merge(pat, on=["patient_alias", "t1", "t2"], suffixes=("_r", "_p"))
    assert len(j) == len(r) == len(pat) == 54
    # depths and clone counts are deterministic; the rest depends on the (unseeded-equivalent) downsampling draws
    assert (j.N_1_r == j.N_1_p).all() and (j.N_2_r == j.N_2_p).all()
    assert np.abs(j.n_clones_r - j.n_clones_p).max() / j.n_clones_r.max() < 0.02
    assert (j.n_fdr_up_r == j.n_fdr_up_p).all()
    assert abs(j.cutoff_r.mean() - j.cutoff_p.mean()) < 0.03
    for c, rho in [("cutoff", 0.9), ("n_sig_up", 0.95), ("n_sig_down", 0.95), ("frac_cells_sig_up_t2", 0.99)]:
        assert j[c + "_r"].corr(j[c + "_p"], method="spearman") > rho - 0.05
    assert abs(j.n_sig_up_r.mean() - j.n_sig_up_p.mean()) / j.n_sig_up_r.mean() < 0.05


@needs_data
@pytest.mark.parametrize("which", ["w3", "all"])
def test_startrac_exact_vs_R(meta, which):
    if which == "w3":
        cl, pw = C.startrac(meta[meta.week == "3"], min_cells=20)
        r = pd.read_csv(_gz(REF / "startrac_w3_cluster.csv"))
        rp = pd.read_csv(_gz(REF / "startrac_w3_pairwise.csv"))
        j = rp.merge(pw, on=["patient_alias", "state_a", "state_b"], suffixes=("_r", "_p"))
        assert len(j) == len(rp) == len(pw) and np.abs(j.tran_r - j.tran_p).max() < 1e-12
        assert (j.n_shared_clones_r == j.n_shared_clones_p).all()
        cols = ["n", "expa", "tran"]
    else:
        cl, _ = C.startrac(meta, loc_col="week", min_cells=20)
        r = pd.read_csv(_gz(REF / "startrac_all_cluster.csv"))
        cols = ["n", "expa", "tran", "migr"]
    j = r.merge(cl, on=["patient_alias", "state"], suffixes=("_r", "_p"))
    assert len(j) == len(r) == len(cl)
    for c in cols:
        assert np.abs(j[c + "_r"] - j[c + "_p"]).max() < 1e-10


@needs_data
def test_clone_table_dynamics_diversity_vs_R(meta):
    ct = C.clone_table(meta, "clone_id")
    r = pd.read_csv(_gz(REF / "clone_table.csv"))
    j = r.merge(ct, on=["patient_alias", "timepoint", "clone"], suffixes=("_r", "_p"))
    assert len(j) == len(r) == len(ct) and (j.n_r == j.n_p).all() and np.abs(j.freq_r - j.freq_p).max() < 1e-12
    dy = C.clone_dynamics(ct, 0, 3)
    rd = pd.read_csv(_gz(REF / "dynamics_0_3.csv"))
    j = rd.merge(dy, on=["patient_alias", "clone"], suffixes=("_r", "_p"))
    assert len(j) == len(rd) == len(dy) and (j.class_r == j.class_p).all()
    assert np.abs(j.p_r - j.p_p).max() < 1e-9 and np.abs(j.fdr_r - j.fdr_p).max() < 1e-9
    assert np.abs(j.lfc_r - j.lfc_p).max() < 1e-10
    dv = C.clone_diversity(ct, reps=20, verbose=False)
    rv = pd.read_csv(_gz(REF / "diversity.csv"))
    j = rv.merge(dv, on=["patient_alias", "timepoint"], suffixes=("_r", "_p"))
    assert len(j) == len(rv) == len(dv) and (j.depth_r == j.depth_p).all() == True  # noqa: E712
    for c in ("shannon", "simpson", "richness", "top10_frac", "clonality"):   # Monte-Carlo (20 draws) agreement
        assert j[c + "_r"].corr(j[c + "_p"]) > 0.995
        assert (np.abs(j[c + "_r"] - j[c + "_p"]) / np.abs(j[c + "_r"])).median() < 0.05


@needs_data
def test_cycling_fraction_vs_R(meta):
    cf = G.cycling_fraction(meta, (meta.metaclusters_exh == "Activated").to_numpy(), by="metaclusters_exh")
    r = pd.read_csv(_gz(REF / "cycling_fraction.csv"))
    j = r.merge(cf, on=["patient_alias", "timepoint", "metaclusters_exh"], suffixes=("_r", "_p"))
    assert len(j) == len(r) == len(cf) and np.abs(j.logit_cc_r - j.logit_cc_p).max() < 1e-12


@pytest.mark.skipif(not (V3 / "jo_pathways.json").exists(), reason=f"missing {V3 / 'jo_pathways.json'} (ARSC_DATA_V3)")
def test_v3_genesets(monkeypatch):
    monkeypatch.setattr(G, "DATA_V3", V3)  # load_v3 reads ARSC_DATA_V3; point it at the test folder
    assert len(G.load_v3("pathways")) == 19 and len(G.load_v3("jo_pathways")) == 4
    mods = G.load_v3("hotspot_modules")
    assert {k: len(v) for k, v in mods.items()} == {"module1": 832, "module2": 882, "module3": 754, "module4": 234}
    assert len(G.load_v3("cpdb_all")) == 4589 and len(G.load_v3("cpdb_selected")) == 174
    assert "MKI67" in G.load_v3("gnf2_mki67")["gnf2_mki67"]


def test_read_10x_contigs(tmp_path):
    f = tmp_path / "filtered_contig_annotations.csv"
    pd.DataFrame({
        "barcode": ["A-1", "A-1", "A-1", "B-1", "B-1", "C-1", "D-1"],
        "is_cell": ["true", "true", "true", "true", "true", "true", "false"],
        "high_confidence": ["true"] * 6 + ["true"],
        "chain": ["TRA", "TRB", "TRB", "TRB", "TRA", "TRB", "TRB"],
        "productive": ["true", "true", "true", "true", "true", "false", "true"],
        "cdr3": ["CA1", "CB1", "CB2", "CB2", "CA1", "CBX", "CBD"],
        "cdr3_nt": ["aa1", "bb1", "bb2", "bb2", "aa1", "bbx", "bbd"],
        "v_gene": ["V1", "V2", "V3", "V3", "V1", "V9", "V8"],
        "j_gene": ["J1", "J2", "J3", "J3", "J1", "J9", "J8"],
        "umis": [4, 2, 9, 5, 3, 1, 6]}).to_csv(f, index=False)
    t = C.read_10x_contigs(f, sample="s1").set_index("barcode")
    assert list(t.index) == ["A-1", "B-1"]                       # C-1 unproductive, D-1 not a cell
    assert t.loc["A-1", "TRB_cdr3_nt"] == "bb2,bb1"             # highest umis first
    assert t.loc["A-1", "clone_id"] == "aa1|bb1,bb2"            # sorted inside the id
    assert t.loc["A-1", "n_TRB"] == 2 and t.loc["B-1", "n_TRA"] == 1
    assert (t["sample"] == "s1").all()
    b = C.read_10x_contigs(f, chains=("TRB",), key="cdr3", productive_only=False).set_index("barcode")
    assert b.loc["C-1", "clone_id"] == "CBX" and b.loc["B-1", "clone_id"] == "CB2"
    with pytest.raises(KeyError, match="contig_annotations"):
        C.read_10x_contigs(f, key="nonexistent")


def test_clone_diversity_single_sample_column():
    """One sample column (list or plain string) used to return an empty table (pandas 1-tuple group keys)."""
    import arsc
    obs = arsc.datasets.toy().obs
    obs = obs.assign(sample_id=obs["patient"].astype(str) + "_" + obs["timepoint"].astype(str))
    for sc in (["sample_id"], "sample_id"):
        ct = arsc.clones.clone_table(obs, "clone_id", sample_cols=sc)
        dv = arsc.clones.clone_diversity(ct, sample_cols=sc, floor=50, verbose=False)
        assert len(dv) == 12 and "clonality" in dv.columns
