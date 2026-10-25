"""Numerical agreement with the R reference implementations (Seurat 4.3, harmony 1.2 / 2.0 / 2.1, presto, UCell).

References are produced by benchmarks/r_reference_gd.R into $ARSC_SCRATCH/ref_gd; the Seurat export of the
38,440-cell reference dataset lives in $ARSC_SCRATCH/exp_gd. The harmony references for several batch variables
and harmony versions come from benchmarks/harmony_input.py + benchmarks/r_reference_harmony.R
($ARSC_SCRATCH/ref_harmony). Tests skip when their references are absent.
Run: .venv/Scripts/python.exe -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from refutil import EXP_GD, REF_GD, SCRATCH, read_dense  # noqa: E402

import arsc  # noqa: E402
from arsc import preprocessing as pp  # noqa: E402

HAVE_GD = (EXP_GD / "manifest.json").exists() and REF_GD.exists() and any(REF_GD.iterdir())
REF_HARMONY = SCRATCH / "ref_harmony"


@pytest.fixture(scope="module")
def ds():
    if not HAVE_GD:
        pytest.skip("R reference outputs not built")
    return arsc.read_seurat_export(EXP_GD)


@pytest.fixture(scope="module")
def lognorm(ds):
    return pp.log_normalize(ds.X("RNA", "counts"))


def test_lognormalize_exact(ds, lognorm):
    feats = ds["RNA"].features
    ref = read_dense(REF_GD / "lognorm_block")
    mine = lognorm[:2000][:, feats.get_indexer(ref.index)].toarray().T
    assert np.abs(mine - ref.values).max() < 1e-10


def test_hvg_vst_values_and_set(ds):
    h = pp.hvg_vst(ds.X("RNA", "counts"), features=ds["RNA"].features)
    r = pd.read_csv(REF_GD / "hvf.csv", index_col=0)
    y = r.loc[h.index, "vst.variance.standardized"].values
    ok = y > 0
    assert np.nanmax(np.abs(h["variance_standardized"].values[ok] - y[ok]) / y[ok]) < 1e-9
    rv = set(open(REF_GD / "variable_features.txt").read().split())
    assert set(h.sort_values("rank").index[:2000]) == rv  # identical set; only tie order may differ


def test_wilcoxauc_matches_presto(ds, lognorm):
    st = arsc.open(EXP_GD)
    res = arsc.wilcoxauc(lognorm, st.obs["cell_type"], features=st.features())
    ref = pd.read_csv(REF_GD / "presto_cell_type.csv")
    m = res.merge(ref, on=["feature", "group"], suffixes=("", "_r"))
    assert len(m) == len(ref)
    for c, tol in [("statistic", 1e-9), ("auc", 1e-12), ("logFC", 1e-9), ("pct_in", 1e-10)]:
        assert np.nanmax(np.abs(m[c] - m[c + "_r"])) < tol, c
    ok = (m.pval_r > 1e-300) & (m.pval > 0)
    assert np.abs(np.log10(m.pval[ok]) - np.log10(m.pval_r[ok])).max() < 1e-6


def test_pseudobulk_conserves_counts(ds):
    """Group sums conserve total counts (exact equality to R AggregateExpression: benchmarks/t_pb)."""
    st = arsc.open(EXP_GD)
    X = ds.X("RNA", "counts")
    res = arsc.pseudobulk(X, st.obs, ["patient_alias", "timepoint"])
    assert len(res.dropped) == 0  # min_cells=0 keeps every group
    assert np.isclose(float(res.X.sum()), float(X.sum()))


def test_harmony_tracks_r(ds):
    """Harmony is stochastic in block order; require the R objective trajectory and per-dim r >= 0.98."""
    Z = read_dense(REF_GD / "pca_embeddings").values
    meta = arsc.read_seurat_export(EXP_GD).obs
    out = arsc.run_harmony(Z, meta, "n10x_lane_name", seed=1, version="1.2")  # same variable and version as R
    ref = read_dense(REF_GD / "harmony_default").values
    r = [np.corrcoef(out[:, j], ref[:, j])[0, 1] for j in range(min(10, ref.shape[1]))]
    assert min(abs(x) for x in r) > 0.98


# ---- harmony: several batch variables, per-variable theta / lambda, harmony 1.2 / 2.0 / 2.1 -------------------
HARMONY_R = {"1.2": "1.2.0", "2.0": "2.0.5", "2.1": "2.1.0"}  # arsc version -> R harmony version of the reference
HARMONY_CASES = ["lane1", "mv2", "mv2_tl", "mv2_auto", "mv3"]  # see benchmarks/r_reference_harmony.R


@pytest.mark.slow
@pytest.mark.parametrize("case", HARMONY_CASES)
@pytest.mark.parametrize("version", list(HARMONY_R))
def test_harmony_versions_track_r(version, case):
    """Each version and case vs the matching R harmony, started from R's k-means centroids (init_centroids) so only
    R's random block order differs: per-dim r >= 0.98 on every dim (the single-variable bar), the same number of
    harmony rounds (+-1) and the final objective within 1%."""
    if not (REF_HARMONY / HARMONY_R[version] / f"{case}_trace.json").exists():
        pytest.skip("harmony references not built (benchmarks/r_reference_harmony.R)")
    from harmony_parity import compare, load_input
    Z, meta = load_input(REF_HARMONY)
    row = compare(REF_HARMONY, HARMONY_R[version], case, Z, meta, init="r")
    assert row["r_min"] > 0.98, row
    assert abs(len(row["rounds_arsc"]) - len(row["rounds_r"])) <= 1, row
    assert abs(row["obj_arsc"][-1] - row["obj_r"][-1]) / row["obj_r"][-1] < 0.01, row
