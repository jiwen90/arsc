"""Order of tied p values, as Seurat 4.3: FindMarkers keeps the feature order (its tie-break column is p_val
itself); FindAllMarkers re-sorts each cluster by (p_val, -avg_log2FC). Ties are common on large data, where many p
values underflow to 0."""
import numpy as np
import scipy.sparse as sp

import arsc


def _data():
    rng = np.random.default_rng(0)
    n = 400
    g = np.array(["A"] * 200 + ["B"] * 200)
    base = rng.poisson(np.where(g == "A", 3.0, 0.5)).astype(float)
    # gene "lo" and gene "hi" have the same ranks (hi = 3 x lo), so the same p value, but different fold changes;
    # "lo" comes first in feature order
    X = np.column_stack([base, 3 * base, rng.poisson(1.0, n)]).astype(float)
    return sp.csr_matrix(np.log1p(X)), g, np.array(["lo", "hi", "noise"])


def test_find_markers_ties_keep_feature_order():
    L, g, feats = _data()
    fm = arsc.find_markers(L, g, ident_1="A", features=feats)
    tied = fm.loc[["lo", "hi"]]
    assert tied["p_val"].iloc[0] == tied["p_val"].iloc[1]
    assert tied["avg_log2FC"]["hi"] > tied["avg_log2FC"]["lo"]
    assert [x for x in fm.index if x in ("lo", "hi")] == ["lo", "hi"]


def test_find_all_markers_ties_by_fold_change():
    L, g, feats = _data()
    fa = arsc.find_all_markers(L, g, features=feats)
    a = fa[fa["cluster"].astype(str) == "A"]
    assert [x for x in a["gene"] if x in ("lo", "hi")] == ["hi", "lo"]
