"""Subset stores as views (arsc.views): a view must be a drop-in replacement for the full store it replaces.

Every test builds full stores with StoreBuilder (plus the extras of a full Seurat export: dense scale.data, a split,
var.csv, a Neighbor object, typed uns, Idents), builds a view of one on its parent(s), and checks with
``arsc.views.compare`` that every accessor returns the same values bit for bit, then the failure modes: a parent that
is missing or changed, a view moved into stores/.superseded/, chains, bad row maps, and that opening a view writes
nothing.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import arsc
from arsc import views


def _counts(rng, n, g, density=0.3):
    return sp.random(n, g, density=density, format="csr", random_state=int(rng.integers(1 << 30)),
                     data_rvs=lambda k: rng.integers(1, 30, k)).astype(np.float64)


def _lognorm(c):
    c = sp.csr_matrix(c, dtype=np.float64)
    tot = np.asarray(c.sum(1)).ravel()
    tot[tot == 0] = 1
    d = sp.diags(1e4 / tot) @ c
    d.data = np.log1p(d.data)
    return d.tocsr()


def _write_full(d, cells, obs, assays, *, extras=False, rng=None, alias=None):
    """assays: {name: (features, {layer: (matrix, dtype)})}; alias: {assay: (layer, alias_of)}."""
    b = arsc.StoreBuilder(d)
    b.set_cells(cells, obs)
    for a, (feats, layers) in assays.items():
        b.add_assay(a, feats, variable_features=list(feats[:3]))
        for l, (m, dt) in layers.items():
            b.add_layer(a, l, m, dtype=dt)
    if extras:
        n = len(cells)
        b.add_reduction("pca", rng.normal(size=(n, 4)), columns=[f"PC_{i}" for i in range(1, 5)],
                        loadings=rng.normal(size=(len(assays["RNA"][0]), 4)), loadings_features=assays["RNA"][0],
                        stdev=np.array([4.0, 3, 2, 1]), assay="RNA", key="PC_")
        b.add_reduction("umap", rng.normal(size=(n, 2)).astype(np.float32), columns=["UMAP_1", "UMAP_2"])
        g = sp.random(n, n, density=0.05, format="csr", random_state=1)
        b.add_graph("RNA_snn", g)
    b.finish(open_store=False)
    m = json.loads((d / "manifest.json").read_text())
    for a, (l, of) in (alias or {}).items():
        m["assays"][a]["layers"][l] = dict(m["assays"][a]["layers"][of], alias_of=of)
    if extras:
        n = len(cells)
        a = m["assays"]["RNA"]
        first = next(iter(a["layers"]))
        a["layers"][first]["split"] = {"parts": [f"{first}.P1", f"{first}.P2"], "file": f"{first}.split.txt"}
        (d / "assays" / "RNA" / f"{first}.split.txt").write_text(
            "\n".join(["P1"] * (n // 2) + ["P2"] * (n - n // 2)) + "\n")
        sc = rng.normal(size=(n, 3)).astype(np.float32)
        np.save(d / "assays" / "RNA" / "scale.data.dense.npy", sc)
        feats = assays["RNA"][0]
        (d / "assays" / "RNA" / "scale.data.dense_features.txt").write_text("\n".join(feats[:3]) + "\n")
        a["dense"] = {"scale.data": {"shape": [n, 3], "dtype": "float32", "file": "scale.data.dense.npy",
                                     "features_file": "scale.data.dense_features.txt"}}
        pd.DataFrame({"feature": feats, "vst.mean": np.arange(len(feats)) / 2.0}).to_csv(
            d / "assays" / "RNA" / "var.csv", index=False)
        a["var"] = {"file": "var.csv", "columns": {"vst.mean": {"type": "double"}}}
        (d / "neighbors").mkdir()
        np.save(d / "neighbors" / "RNA.nn.idx.npy", rng.integers(0, n, size=(n, 3)).astype(np.int32))
        np.save(d / "neighbors" / "RNA.nn.dist.npy", rng.random((n, 3)))
        m["neighbors"] = {"RNA.nn": {"shape": [n, 3], "index_base": 0, "idx": "RNA.nn.idx.npy",
                                     "dist": "RNA.nn.dist.npy"}}
        m["idents_column"] = "celltype"
        m["unconvertible"] = [{"path": "misc/env", "class": "environment", "reason": "an R environment"}]
        (d / "uns").mkdir()
        np.save(d / "uns" / "0001_big.npy", np.arange(6.0))
        (d / "uns.json").write_text(json.dumps({"misc": {"note": "QC", "big": {"__npy__": "uns/0001_big.npy"}}}))
        (d / "assays" / "ADT").mkdir(exist_ok=True)
        (d / "assays" / "ADT" / "cells_measured.txt").write_text("\n".join(cells[:5]))  # an extra file is kept
    (d / "manifest.json").write_text(json.dumps(m))
    return arsc.open(d)  # finalize: obs.parquet, var.parquet and CSC copies, as a registered store has


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(7)
    n, g = 300, 50
    genes = [f"G{j}" for j in range(g)]
    adt = [f"A{j}" for j in range(8)]
    pcells = [f"p{i:04d}_AAAC-1" for i in range(n)]
    pc = _counts(rng, n, g)
    pc[5] = 0  # an empty row
    pc.eliminate_zeros()
    pdat = _lognorm(pc)
    pcite = _counts(rng, n, len(adt), 0.6)
    pobs = pd.DataFrame({"celltype": pd.Categorical(rng.choice(["T", "B", "NK"], n)), "x": rng.normal(size=n)},
                        index=pcells)
    stores = tmp_path / "data" / "own" / "ds" / "stores"
    P = _write_full(stores / "fin", pcells, pobs, {"RNA": (genes, {"counts": (pc, "float32"), "data": (pdat, "float64")}),
                                                   "CITE": (adt, {"counts": (pcite, "float64")})})
    qn = 80
    qcells = [f"q{i:04d}_AAAC-1" for i in range(qn)]
    qc = _counts(rng, qn, g)
    Q = _write_full(stores / "other", qcells, pd.DataFrame({"x": np.zeros(qn)}, index=qcells),
                    {"RNA": (genes, {"counts": (qc, "float32")})})
    return dict(rng=rng, P=P, Q=Q, pc=pc, pdat=pdat, pcite=pcite, qc=qc, genes=genes, adt=adt, stores=stores,
                pcells=pcells, qcells=qcells)


def _ok(res):
    bad = [r for r in res if not r["ok"]]
    assert not bad, bad[:5]
    return len(res)


def _snapshot(d):
    return {str(f.relative_to(d)): (f.stat().st_size, f.stat().st_mtime_ns) for f in d.rglob("*") if f.is_file()}


def test_view_partial_rows_extras(world):
    """Rows in parent order; data equal except 10 rows that differ by one ulp (kept local); ADT = the parent's CITE
    on a reordered feature subset; an own layer (sct) differs everywhere; every full-export extra."""
    w, rng = world, world["rng"]
    pr = np.sort(rng.choice(300, 120, replace=False))
    cells = [w["pcells"][i] for i in pr]
    c = w["pc"][pr].astype(np.float32)
    dat = w["pdat"][pr].copy()
    bump = np.flatnonzero(np.diff(dat.indptr) > 0)[:10]
    for i in bump:
        dat.data[dat.indptr[i]] = np.nextafter(dat.data[dat.indptr[i]], np.inf)
    adt_feats = [w["adt"][j] for j in (3, 0, 5, 1, 7, 2, 6)]  # A4 dropped, order changed
    adt = w["pcite"][pr][:, [w["adt"].index(f) for f in adt_feats]]
    sct = c.copy()
    sct.data = sct.data + 0.5
    obs = pd.DataFrame({"celltype": pd.Categorical(rng.choice(["T", "B"], 120), categories=["T", "B", "unused"]),
                        "s": rng.normal(size=120)}, index=cells)
    F = _write_full(w["stores"] / "child", cells, obs,
                    {"RNA": (w["genes"], {"counts": (c, "float32"), "data": (dat, "float64"), "sct": (sct, "float64")}),
                     "ADT": (adt_feats, {"data": (adt, "float64")})}, extras=True, rng=rng)
    out = w["stores"] / "child_view"
    rep = views.build(F.path, [w["P"].path], out, {0: (np.arange(120), pr)}, log=lambda *_: None)
    L = rep["layers"]
    assert L["RNA/counts"]["matched"] == 120 and L["RNA/counts"]["inherit"]
    assert L["RNA/data"]["matched"] == 110 and L["RNA/data"]["inherit"]
    n_empty = int((np.diff(sct.indptr) == 0).sum())  # an empty row equals an empty parent row
    assert L["RNA/sct"]["matched"] == n_empty and not L["RNA/sct"]["inherit"]
    assert L["ADT/data"]["matched"] == 120 and L["ADT/data"]["sources"][0]["assay"] == "CITE"
    V = arsc.open(out)
    assert V.is_view and not F.is_view and V.view_parents[0].name == "fin"
    assert "a view of fin" in repr(V)
    assert not (out / "assays" / "RNA" / "counts.csr.data.npy").exists()
    assert (out / "assays" / "RNA" / "sct.csr.data.npy").exists()
    assert (out / "assays" / "ADT" / "cells_measured.txt").exists()
    assert _ok(views.compare(V, F, feature_batch=7)) >= 50
    m = V.X("RNA", "counts")
    assert not m.data.flags.writeable and m.dtype == np.float32
    assert np.shares_memory(V.X("RNA", "counts").data, m.data)  # gathered once while the caller holds it


def test_view_shuffled_subset_features_dtype_alias(world):
    """Shuffled rows, a reordered feature subset, float64 counts from a float32 parent, data an alias of counts."""
    w, rng = world, world["rng"]
    pr = rng.choice(300, 150, replace=False)  # not sorted
    keep = rng.permutation(50)[:40]
    feats = [w["genes"][j] for j in keep]
    c = w["pc"][pr][:, keep].astype(np.float64)
    cells = [w["pcells"][i] for i in pr]
    F = _write_full(w["stores"] / "b", cells, pd.DataFrame({"k": np.arange(150)}, index=cells),
                    {"RNA": (feats, {"counts": (c, "float64")})}, alias={"RNA": ("data", "counts")})
    out = w["stores"] / "b_view"
    rep = views.build(F.path, [w["P"].path], out, {0: (np.arange(150), pr)}, log=lambda *_: None)
    assert rep["layers"]["RNA/counts"]["matched"] == 150
    V = arsc.open(out)
    _ok(views.compare(V, F, feature_batch=13))
    assert V.X("RNA", "data").dtype == np.float64


def test_view_two_parents_and_local_rows(world, tmp_path):
    """Rows from two parents plus 20 rows from neither (kept in the view); materialize gives the full store back."""
    w, rng = world, world["rng"]
    pr = rng.choice(300, 60, replace=False)
    qr = rng.choice(80, 40, replace=False)
    lc = _counts(rng, 20, 50)
    c = sp.vstack([w["pc"][pr], w["qc"][qr], lc]).tocsr()
    order = rng.permutation(120)
    c = c[order].astype(np.float32)
    cells = np.array([f"x{i}" for i in range(120)])  # renamed barcodes: the row map, not names, pairs them
    F = _write_full(w["stores"] / "c", list(cells), pd.DataFrame({"k": np.arange(120)}, index=cells),
                    {"RNA": (w["genes"], {"counts": (c, "float32")})})
    pos = np.argsort(order)  # original stacked row -> new row
    rows = {0: (pos[:60], pr), 1: (pos[60:100], qr)}
    out = w["stores"] / "c_view"
    rep = views.build(F.path, [w["P"].path, w["Q"].path], out, rows, log=lambda *_: None)
    assert rep["layers"]["RNA/counts"]["matched"] == 100
    V = arsc.open(out)
    spec = json.loads((out / "view.json").read_text())
    assert spec["layers"]["RNA/counts"]["local"]["n_rows"] == 20
    _ok(views.compare(V, F, feature_batch=11))
    mat = views.materialize(V, tmp_path / "mat")
    for f in F.path.rglob("*"):
        if f.is_file():
            g = mat / f.relative_to(F.path)
            assert g.exists(), f
            if f.suffix == ".npy":
                a, b = np.load(f), np.load(g)
                assert a.dtype == b.dtype and np.array_equal(a, b), f
            elif f.name != ".finalize.lock":
                assert f.read_bytes() == g.read_bytes(), f
    assert not views.is_view(mat)
    _ok(views.compare(V, arsc.open(mat)))


def test_open_writes_nothing_and_anndata(world):
    w = world
    pr = np.arange(0, 300, 3)
    cells = [w["pcells"][i] for i in pr]
    F = _write_full(w["stores"] / "d", cells, pd.DataFrame({"k": np.arange(len(pr))}, index=cells),
                    {"RNA": (w["genes"], {"counts": (w["pc"][pr], "float32")})})
    out = w["stores"] / "d_view"
    views.build(F.path, [w["P"].path], out, {0: (np.arange(len(pr)), pr)}, log=lambda *_: None)
    before = _snapshot(out)
    V = arsc.open(out)  # finalize=True by default: a correct view has nothing to build
    V.X("RNA")
    V.X_csc("RNA")
    V.get_features(["G1", "G7"])
    V.subset(np.arange(5))
    assert _snapshot(out) == before
    ad = pytest.importorskip("anndata")
    assert ad is not None
    a1, a2 = V.to_anndata(cells=cells[:30]), F.to_anndata(cells=cells[:30])
    assert np.array_equal(a1.X.toarray(), a2.X.toarray()) and a1.obs.equals(a2.obs)


def test_parent_missing_changed_superseded(world, tmp_path):
    w = world
    pr = np.arange(10, 60)
    cells = [w["pcells"][i] for i in pr]
    F = _write_full(w["stores"] / "e", cells, pd.DataFrame({"k": np.arange(50)}, index=cells),
                    {"RNA": (w["genes"], {"counts": (w["pc"][pr], "float32")})})
    out = w["stores"] / "e_view"
    views.build(F.path, [w["P"].path], out, {0: (np.arange(50), pr)}, log=lambda *_: None)
    # moved into stores/.superseded/: the parent still resolves
    sup = w["stores"] / ".superseded" / "e_view-1"
    sup.parent.mkdir()
    shutil.copytree(out, sup)
    _ok(views.compare(arsc.open(sup), F))
    # elsewhere: obs works, a matrix names the missing parent
    lone = tmp_path / "elsewhere" / "deep" / "e_view"
    shutil.copytree(out, lone)
    V = arsc.open(lone)
    assert len(V.obs) == 50
    with pytest.raises(views.ViewError, match="parent store 'fin'"):
        V.X("RNA")
    # a changed parent is refused
    mf = w["P"].path / "manifest.json"
    m = json.loads(mf.read_text())
    m["note"] = "re-exported"
    mf.write_text(json.dumps(m))
    with pytest.raises(views.ViewError, match="changed"):
        arsc.open(out).X("RNA")


def test_build_refusals(world):
    w = world
    pr = np.arange(20)
    cells = [w["pcells"][i] for i in pr]
    F = _write_full(w["stores"] / "f", cells, pd.DataFrame({"k": np.arange(20)}, index=cells),
                    {"RNA": (w["genes"], {"counts": (w["pc"][pr], "float32")})})
    with pytest.raises(views.ViewError, match="twice"):
        views.build(F.path, [w["P"].path], w["stores"] / "x", {0: (np.zeros(20, int), pr)}, log=lambda *_: None)
    with pytest.raises(views.ViewError, match="out of range"):
        views.build(F.path, [w["P"].path], w["stores"] / "x", {0: (pr, pr + 1000)}, log=lambda *_: None)
    views.build(F.path, [w["P"].path], w["stores"] / "f_view", {0: (np.arange(20), pr)}, log=lambda *_: None)
    with pytest.raises(views.ViewError, match="is a view"):
        views.build(F.path, [w["stores"] / "f_view"], w["stores"] / "y", {0: (np.arange(20), np.arange(20))},
                    log=lambda *_: None)
    # rows that match nothing: no view, with the reason
    rep = views.build(F.path, [w["Q"].path], w["stores"] / "z", {0: (np.arange(20), np.arange(20))},
                      log=lambda *_: None)
    assert rep["view"] is False and "no layer" in rep["why_not"] and not (w["stores"] / "z").exists()
    # dry run writes nothing
    rep = views.build(F.path, [w["P"].path], w["stores"] / "w", {0: (np.arange(20), pr)}, dry_run=True,
                      log=lambda *_: None)
    assert rep["view"] and rep["bytes_saved"] > 0 and not (w["stores"] / "w").exists()
    assert views.logical_bytes(w["stores"] / "f_view") == sum(
        f.stat().st_size for f in F.path.rglob("*") if f.is_file())
    assert os.path.isdir(w["stores"] / "f_view" / "view")
