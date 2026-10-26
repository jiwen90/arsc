"""rank_genes_groups(groups_subset=...): cells outside the compared groups stay in the "rest" (scanpy ``groups=``).

A cell outside the subset used to carry group code -1, which crashed the sparse Wilcoxon (np.bincount) and, inside
the numba rank kernel, would have wrapped around to the last group.
"""
import numpy as np
import pytest

import arsc
from arsc import scanpy_extra as sx


@pytest.fixture(scope="module")
def toy():
    ds = arsc.datasets.toy()
    X = ds.X("RNA", "counts").tocsr().astype(np.float32)
    g = ds.obs["cell_type"].astype(str).to_numpy().copy()
    g[0] = "lonely"   # a 1-cell group: scanpy and arsc refuse to test it, so it is left out of the subset
    return sx.normalize_total_log1p(X, dtype=np.float32), g, np.asarray(ds.features("RNA"))


def test_subset_rows_equal_full_run_rows(toy):
    L, g, feats = toy
    keep = [x for x in np.unique(g) if x != "lonely"]
    full_g = g.copy()
    full_g[0] = keep[0]   # same cells, no 1-cell group: a full run is possible
    for sub in (keep, keep[:2], keep[-1:]):
        got = sx.rank_genes_groups(L, g, feats, groups_subset=sub)
        assert sorted(got["group"].unique()) == sorted(sub)
        # the lonely cell is part of every group's rest; recompute with it as an ordinary member of the rest
        ref = sx.rank_genes_groups(L, np.where(g == "lonely", "zz_other", g), feats, groups_subset=sub)
        m = got.merge(ref, on=["group", "names"], suffixes=("_a", "_b"))
        assert len(m) == len(got)
        assert np.abs(m.scores_a - m.scores_b).max() == 0
        assert np.abs(m.pvals_a - m.pvals_b).max() == 0


def test_subset_matches_scanpy(toy):
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    L, g, feats = toy
    sub = [x for x in np.unique(g) if x != "lonely"][:3]
    a = ad.AnnData(L.astype(np.float32))
    a.var_names = [str(f) for f in feats]
    a.obs["ct"] = g
    a.obs["ct"] = a.obs["ct"].astype("category")
    sc.tl.rank_genes_groups(a, "ct", method="wilcoxon", use_raw=False, groups=sub)
    ref = sc.get.rank_genes_groups_df(a, None)
    ref["group"] = ref["group"].astype(str)
    got = sx.rank_genes_groups(L, g, feats, groups_subset=sub)
    m = ref.merge(got, on=["group", "names"], suffixes=("_s", "_a"))
    assert len(m) == len(ref) == len(got)
    assert (np.abs(m.scores_s - m.scores_a) / np.maximum(np.abs(m.scores_s), 1)).max() < 1e-3   # scanpy float32
