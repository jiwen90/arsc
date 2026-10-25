"""arsc.seurat_extra vs Seurat 4.3 (references: benchmarks/r_reference_seurat_extra.R -> $ARSC_REFS/ref_extra;
benchmarks/regen_refs.py ref_extra), on the shared input store $ARSC_STORE_GD..

Run: .venv/Scripts/python.exe -m pytest -q tests/test_seurat_extra.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from refutil import EXP_GD, REF_EXTRA, missing, read_dense, store_gd_missing  # noqa: E402

import arsc  # noqa: E402
from arsc import preprocessing as pp  # noqa: E402
from arsc import seurat_extra as se  # noqa: E402

REF = REF_EXTRA
SKIP = store_gd_missing() or missing(REF / "timings_r.json", regen="ref_extra")
pytestmark = pytest.mark.skipif(bool(SKIP), reason=SKIP or "")


@pytest.fixture(scope="module")
def ds():
    return arsc.read_seurat_export(EXP_GD)


@pytest.fixture(scope="module")
def feats(ds):
    return ds["RNA"].features


@pytest.fixture(scope="module")
def lognorm(ds):
    return pp.log_normalize(ds.X("RNA", "counts"))


@pytest.fixture(scope="module")
def adt(ds):
    return pp.clr(ds.X("TRIMMEDCITE", "counts"), margin=2)


def _ref(name):
    return pd.read_csv(REF / name, index_col=0)


def _cmp_markers(mine, ref, p_rtol_log10=1e-6, fc_tol=1e-9):
    assert list(mine.index) == list(ref.index), "feature set / order differs"
    assert np.abs(mine["avg_log2FC"] - ref["avg_log2FC"]).max() < fc_tol
    assert np.abs(mine["pct.1"] - ref["pct.1"]).max() < 1e-12 and np.abs(mine["pct.2"] - ref["pct.2"]).max() < 1e-12
    ok = ref.p_val > 1e-300
    d = np.abs(np.log10(mine.p_val[ok]) - np.log10(ref.p_val[ok]))
    assert d.max() < p_rtol_log10, d.max()
    assert np.abs(mine.p_val_adj - ref.p_val_adj).max() < 1e-9 * max(1, ref.p_val_adj.max())


@pytest.fixture(scope="module")
def gd(ds, lognorm):
    m = (ds.obs["cell_type"] == "gd").to_numpy()
    return lognorm[m], ds.obs["timepoint"][m].astype(str).reset_index(drop=True), np.flatnonzero(m)


def test_findmarkers_wilcox_matches_limma(gd, feats):
    X, tp, _ = gd
    r = se.find_markers(X, tp, "Follow Up 1", "Baseline", features=feats, min_pct=0.1, logfc_threshold=0.1)
    _cmp_markers(r, _ref("fm_wilcox_gd_fu1_bl.csv"))


def test_findmarkers_default_thresholds(gd, feats):
    X, tp, _ = gd
    r = se.find_markers(X, tp, "Follow Up 2", "Baseline", features=feats)
    _cmp_markers(r, _ref("fm_wilcox_default.csv"))


def test_findmarkers_vs_rest(gd, feats):
    X, tp, _ = gd
    r = se.find_markers(X, tp, "Baseline", None, features=feats, min_pct=0.1, logfc_threshold=0.1)
    _cmp_markers(r, _ref("fm_wilcox_vs_rest.csv"))


def test_findmarkers_only_pos_max_cells_uses_r_rng(gd, feats):
    X, tp, _ = gd
    r = se.find_markers(X, tp, "Follow Up 3", "Follow Up 1", features=feats, only_pos=True, min_pct=0.25,
                        max_cells_per_ident=1500, random_seed=7)
    _cmp_markers(r, _ref("fm_wilcox_maxcells.csv"))


def test_findmarkers_t(gd, feats):
    X, tp, _ = gd
    r = se.find_markers(X, tp, "Follow Up 1", "Baseline", features=feats, test_use="t", min_pct=0.25,
                        logfc_threshold=0.25)
    _cmp_markers(r, _ref("fm_t.csv"), p_rtol_log10=1e-6)


def _lr_inputs(ds, lognorm, gd):
    X, tp, pos = gd
    cells = open(REF / "lr_cells.txt").read().split()
    idx = ds.obs.index.get_indexer(cells)
    assert (idx >= 0).all()
    return lognorm[idx], ds.obs.iloc[idx].reset_index(drop=True)


def test_findmarkers_lr_numeric_latent(ds, lognorm, gd, feats):
    X, ob = _lr_inputs(ds, lognorm, gd)
    r = se.find_markers(X, ob["timepoint"].astype(str), "Follow Up 1", "Baseline", features=feats, test_use="LR",
                        latent_vars=ob[["nCount_RNA", "percent.mt"]], min_pct=0.25, logfc_threshold=0.25)
    _cmp_markers(r, _ref("fm_lr_latent.csv"), p_rtol_log10=1e-3)


def test_findmarkers_lr_factor_latent(ds, lognorm, gd, feats):
    X, ob = _lr_inputs(ds, lognorm, gd)
    r = se.find_markers(X, ob["timepoint"].astype(str), "Follow Up 1", "Baseline", features=feats, test_use="LR",
                        latent_vars=ob[["patient_alias"]], min_pct=0.3, logfc_threshold=0.4)
    _cmp_markers(r, _ref("fm_lr_factor.csv"), p_rtol_log10=1e-3)


def test_findmarkers_adt_clr(ds, adt):
    r = se.find_markers(adt, ds.obs["timepoint"].astype(str), "Follow Up 1", "Baseline",
                        features=ds["TRIMMEDCITE"].features, min_pct=0.1, logfc_threshold=0.25, norm_method="CLR")
    _cmp_markers(r, _ref("fm_adt.csv"))


def test_findallmarkers_onlypos(ds, lognorm, feats):
    big = open(REF / "fam_types.txt").read().split()
    m = ds.obs["cell_type"].isin(big).to_numpy()
    lab = ds.obs["cell_type"][m].reset_index(drop=True)  # keeps the R factor level order (sort(unique(Idents)))
    r = se.find_all_markers(lognorm[m], lab.astype(str), features=feats, only_pos=True, min_pct=0.25,
                            logfc_threshold=0.25, levels=pd.unique(lab.astype(str)))  # Idents<- => first-appearance levels
    ref = pd.read_csv(REF / "fam_onlypos.csv")
    assert len(r) == len(ref)
    r = r.reset_index(drop=True)
    assert (r["cluster"].astype(str).values == ref["cluster"].astype(str).values).all()
    assert (r["gene"].values == ref["gene"].values).all()
    assert np.abs(r["avg_log2FC"] - ref["avg_log2FC"]).max() < 1e-9
    ok = ref.p_val > 1e-300
    assert np.abs(np.log10(r.p_val[ok]) - np.log10(ref.p_val[ok])).max() < 1e-6


def test_average_expression(ds, lognorm, adt):
    ref = read_dense(REF / "avgexp_data")
    mine = se.average_expression(lognorm, ds.obs, ["patient_alias", "timepoint"], features=ds["RNA"].features)
    # Seurat renames underscores in features to dashes in AverageExpression output; groups keep "_" joins
    assert list(mine.columns) == [c.replace("-", "_") if c not in mine.columns else c for c in ref.columns]
    r = ref.copy()
    r.index = [i for i in ref.index]
    mine2 = mine.copy()
    mine2.index = [i.replace("_", "-") for i in mine.index]
    mine2 = mine2.loc[r.index]
    assert np.abs(mine2.values - ref.values).max() < 1e-9
    ref2 = read_dense(REF / "avgexp_counts")
    m2 = se.average_expression(ds.X("RNA", "counts"), ds.obs, "cell_type", features=ds["RNA"].features, layer="counts")
    m2.index = [i.replace("_", "-") for i in m2.index]
    assert list(m2.columns) == list(ref2.columns)
    assert np.abs(m2.loc[ref2.index].values - ref2.values).max() < 1e-9
    ref3 = read_dense(REF / "avgexp_adt")
    m3 = se.average_expression(adt, ds.obs, "cell_type", features=ds["TRIMMEDCITE"].features)
    m3.index = [i.replace("_", "-") for i in m3.index]
    assert np.abs(m3.loc[ref3.index].values - ref3.values).max() < 1e-9


def test_cell_cycle_scoring(ds, lognorm, feats):
    ref = pd.read_csv(REF / "cellcycle.csv.gz", index_col=0)
    r = se.cell_cycle_scoring(lognorm, feats, cells=ds.obs.index)
    assert np.abs(r["S.Score"] - ref["S.Score"]).max() < 1e-12
    assert np.abs(r["G2M.Score"] - ref["G2M.Score"]).max() < 1e-12
    assert (r["Phase"].values == ref["Phase"].values).all()


def _scale_ref(name):
    return read_dense(REF / name)


def _ref_cols(ds, ref):
    """Positions of the reference's stored cells (every 10th cell; r_reference_seurat_extra.R) in the dataset."""
    j = pd.Index(ds.obs.index).get_indexer(ref.columns)
    assert (j >= 0).all() and len(j) > 3000
    return j


def _scale_args(ds, lognorm, feats):
    """Features in assay order: SeuratObject stores scale.data rows in assay order whatever the order of ``features``."""
    sel = set(open(REF / "scale_features.txt").read().split())
    return [f for f in feats if f in sel]


def test_scale_plain_and_options(ds, lognorm, feats):
    sel = _scale_args(ds, lognorm, feats)
    Z, nm = se.scale_data(lognorm, feats, select=sel)
    ref = _scale_ref("scale_plain")
    j = _ref_cols(ds, ref)
    assert list(nm) == list(ref.index)
    assert np.abs(Z.T[:, j] - ref.values).max() < 1e-9
    Z, _ = se.scale_data(lognorm, feats, select=sel, do_center=False, scale_max=3)
    assert np.abs(Z.T[:, j] - _scale_ref("scale_nocenter_max3").values).max() < 1e-9


def test_scale_regress_numeric(ds, lognorm, feats):
    sel = _scale_args(ds, lognorm, feats)
    Z, _ = se.scale_data(lognorm, feats, select=sel, vars_to_regress=["nCount_RNA", "percent.mt"],
                         latent_data=ds.obs)
    ref = _scale_ref("scale_regress_num")
    assert np.abs(Z.T[:, _ref_cols(ds, ref)] - ref.values).max() < 1e-7


def test_scale_regress_factor(ds, lognorm, feats):
    sel = _scale_args(ds, lognorm, feats)
    Z, _ = se.scale_data(lognorm, feats, select=sel, vars_to_regress=["patient_alias", "nFeature_RNA"],
                         latent_data=ds.obs)
    ref = _scale_ref("scale_regress_factor")
    assert np.abs(Z.T[:, _ref_cols(ds, ref)] - ref.values).max() < 1e-7


def test_scale_regress_split_by(ds, lognorm, feats):
    sel = _scale_args(ds, lognorm, feats)
    Z, nm = se.scale_data(lognorm, feats, select=sel, vars_to_regress=["percent.mt"], latent_data=ds.obs,
                          split_by=ds.obs["n10x_lane_name"].astype(str).to_numpy())
    ref = _scale_ref("scale_regress_split").loc[nm]  # reference built via ScaleData.default on a lane-ordered matrix
    j = _ref_cols(ds, ref)
    lane = ds.obs["n10x_lane_name"].astype(str)
    ok = (lane.map(lane.value_counts()) > 3).to_numpy()[j]  # splits with <= rank+1 cells: exactly-zero residuals
    assert ok.mean() > 0.99
    assert np.abs(Z.T[:, j][:, ok] - ref.values[:, ok]).max() < 1e-7


def test_normalize_clr_margin1_and_rc(ds):
    X = ds.X("TRIMMEDCITE", "counts")
    ref = read_dense(REF / "clr_margin1")
    mine = se.normalize_data(X, "CLR", margin=1)[:3000].toarray().T
    assert np.abs(mine - ref.values).max() < 1e-10
    rc = read_dense(REF / "rc_block")
    feats = open(REF / "rc_features.txt").read().split()
    C = ds.X("RNA", "counts")[:2000]
    j = ds["RNA"].features.get_indexer(feats)
    mine = se.normalize_data(C, "RC")[:, j].toarray().T
    assert np.abs(mine - rc.values).max() < 1e-9


# ---------------------------------------------------------------- helpers (no R needed)
def test_helpers():
    obs = pd.DataFrame({"a": [1, 2, 3, 4], "b": list("xyxy")}, index=list("ABCD"))
    assert se.which_cells(obs, "a > 1 and b == 'x'").tolist() == [False, False, True, False]
    assert se.which_cells(obs, idents=["x"], ident_col="b", invert=True).tolist() == [False, True, False, True]
    out = se.add_metadata(obs, pd.Series([10, 20], index=["A", "C"], name="z"))
    assert out["z"].isna().tolist() == [False, True, False, True]
    assert list(se.rename_idents(obs["b"], {"x": "y"}).astype(str)) == ["y"] * 4
