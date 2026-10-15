"""IO parity and round trips: h5ad / 10x h5 / 10x mtx <-> arsc store / Dataset, vs anndata and scanpy.

All data are synthetic and generated here (no external files). Run:
    .venv/Scripts/python.exe -m pytest -q tests/test_formats.py
"""
from __future__ import annotations

import gzip
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.io import mmwrite

import arsc
from arsc.formats import ArscFormatError

ad = pytest.importorskip("anndata")
h5py = pytest.importorskip("h5py")
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------------- synthetic data
def make_adata(n: int = 240, g: int = 40, g_raw: int = 55, seed: int = 0, x_format: str = "csr",
               x_dtype=np.float32) -> "ad.AnnData":
    rng = np.random.default_rng(seed)
    counts = sp.random(n, g, density=0.3, format="csr", random_state=seed,
                       data_rvs=lambda k: rng.integers(1, 50, k)).astype(np.int64)
    lib = np.asarray(counts.sum(1)).ravel() + 1
    logn = np.log1p(counts.multiply(1e4 / lib[:, None]).toarray())
    X = counts.astype(x_dtype)
    X = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": lambda m: m.toarray()}[x_format](X)
    obs = pd.DataFrame({
        "sample": pd.Categorical(rng.choice(["s1", "s2", "s3"], n)),
        "stage": pd.Categorical(rng.choice(["early", "late"], n), categories=["early", "late"], ordered=True),
        "n_counts": lib.astype(np.float64),
        "n_genes": np.asarray((counts > 0).sum(1)).ravel().astype(np.int64),
        "age": pd.array(np.where(rng.random(n) < 0.1, None, rng.integers(20, 80, n)), dtype="Int64"),
        "responder": pd.array(np.where(rng.random(n) < 0.1, None, rng.random(n) < 0.5), dtype="boolean"),
        "note": np.array([f"cell note {i}" for i in range(n)], dtype=object),
    }, index=[f"BC{i:05d}-1" for i in range(n)])
    genes = [f"G{j}" for j in range(g)]
    var = pd.DataFrame({"gene_ids": [f"ENSG{j:08d}" for j in range(g)],
                        "highly_variable": rng.random(g) < 0.4,
                        "means": rng.random(g)}, index=genes)
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = counts.copy()
    a.layers["lognorm"] = logn
    a.obsm["X_pca"] = rng.normal(size=(n, 8)).astype(np.float32)
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obsm["protein"] = pd.DataFrame(rng.random((n, 3)), index=a.obs_names, columns=["CD3", "CD4", "CD8"])
    a.varm["PCs"] = rng.normal(size=(g, 8))
    knn = sp.random(n, n, density=0.05, format="csr", random_state=seed + 1)
    a.obsp["connectivities"] = knn
    a.obsp["distances"] = knn.multiply(2.0).tocsr()
    a.uns["sample_colors"] = np.array(["#ff0000", "#00ff00", "#0000ff"], dtype=object)
    a.uns["neighbors"] = {"params": {"n_neighbors": 15, "method": "umap"}, "connectivities_key": "connectivities"}
    a.uns["title"] = "synthetic"
    raw_counts = sp.random(n, g_raw, density=0.3, format="csr", random_state=seed + 2, dtype=np.float32)
    raw_var = pd.DataFrame({"gene_ids": [f"ENSG{j:08d}" for j in range(g_raw)]}, index=[f"G{j}" for j in range(g_raw)])
    a.raw = ad.AnnData(X=raw_counts, obs=pd.DataFrame(index=a.obs_names), var=raw_var)
    return a


def dense(m):
    return m.toarray() if sp.issparse(m) else np.asarray(m)


def assert_matrix_equal(x, y, check_dtype=True):
    assert x.shape == y.shape
    if check_dtype:
        assert np.dtype(x.dtype) == np.dtype(y.dtype), (x.dtype, y.dtype)
    np.testing.assert_array_equal(dense(x), dense(y))


def assert_frame_same(x: pd.DataFrame, y: pd.DataFrame):
    x, y = x.copy(), y.copy()
    x.index, y.index = x.index.astype(str), y.index.astype(str)
    x.index.name = y.index.name = None
    pd.testing.assert_frame_equal(x, y, check_index_type=False)


def assert_adata_equal(a, b, *, raw=True, obsp=True, uns=True):
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    assert_matrix_equal(a.X, b.X)
    assert_frame_same(a.obs, b.obs)
    assert_frame_same(a.var, b.var)
    assert set(a.layers) == set(b.layers)
    for k in a.layers:
        assert_matrix_equal(a.layers[k], b.layers[k])
    assert set(a.obsm) == set(b.obsm)
    for k in a.obsm:
        if isinstance(a.obsm[k], pd.DataFrame):
            assert_frame_same(a.obsm[k], b.obsm[k])
        else:
            assert_matrix_equal(a.obsm[k], b.obsm[k])
    assert set(a.varm) == set(b.varm)
    for k in a.varm:
        assert_matrix_equal(a.varm[k], b.varm[k])
    if obsp:
        assert set(a.obsp) == set(b.obsp)
        for k in a.obsp:
            assert_matrix_equal(a.obsp[k], b.obsp[k], check_dtype=False)
    if uns:
        assert a.uns["title"] == b.uns["title"]
        assert list(a.uns["sample_colors"]) == list(b.uns["sample_colors"])
        assert a.uns["neighbors"]["params"]["n_neighbors"] == b.uns["neighbors"]["params"]["n_neighbors"]
    if raw:
        assert (a.raw is None) == (b.raw is None)
        if a.raw is not None:
            assert list(a.raw.var_names) == list(b.raw.var_names)
            assert_matrix_equal(a.raw.X, b.raw.X)
            assert_frame_same(a.raw.var, b.raw.var)


# ----------------------------------------------------------------------------- h5ad
@pytest.mark.parametrize("x_format", ["csr", "csc", "dense"])
def test_h5ad_store_roundtrip(tmp_path, x_format):
    a = make_adata(x_format=x_format)
    src = tmp_path / "in.h5ad"
    a.write_h5ad(src)
    st = arsc.read(src, store=tmp_path / "store", chunk_nnz=97)  # many small chunks
    assert isinstance(st, arsc.Store)
    assert st.n_cells == a.n_obs and list(st.features("RNA")) == list(a.var_names)
    assert set(st.layers("RNA")) == {"X", "counts", "lognorm"}
    assert_matrix_equal(st.X("RNA", "X"), sp.csr_matrix(dense(a.X)), check_dtype=False)
    # per-feature (CSC) lookups agree
    np.testing.assert_array_equal(st.get_features(["G3", "G7"], layer="lognorm").to_numpy(),
                                  a[:, ["G3", "G7"]].layers["lognorm"])
    # raw is its own assay
    assert list(st.features("raw")) == list(a.raw.var_names)
    assert st.variable_features("RNA") == list(a.var_names[a.var.highly_variable])
    out = tmp_path / "out.h5ad"
    st.write_h5ad(out, obsp=True, chunk_nnz=101)
    b = ad.read_h5ad(out)
    assert_adata_equal(a, b)
    # identical in-memory hand-off
    assert_adata_equal(a, st.to_anndata(obsp=True))


def test_h5ad_in_memory_dataset(tmp_path):
    a = make_adata(x_format="csc", seed=3)
    a.write_h5ad(tmp_path / "in.h5ad")
    ds = arsc.read(tmp_path / "in.h5ad")
    assert isinstance(ds, arsc.Dataset)
    assert_adata_equal(a, ds.to_anndata(obsp=True))
    ds.write_h5ad(tmp_path / "o.h5ad", obsp=True)
    assert_adata_equal(a, ad.read_h5ad(tmp_path / "o.h5ad"))
    # PCA loadings were linked from varm["PCs"]
    assert ds.reductions["pca"].loadings.shape == (a.n_vars, 8)


def test_subset_export_matches_anndata_slicing(tmp_path):
    a = make_adata(seed=5)
    a.write_h5ad(tmp_path / "in.h5ad")
    st = arsc.read(tmp_path / "in.h5ad", store=tmp_path / "st")
    rng = np.random.default_rng(1)
    cells = np.sort(rng.choice(a.n_obs, 90, replace=False))
    genes = ["G10", "G2", "G33", "G5"]  # unsorted on purpose
    ref = a[a.obs_names[cells], genes].copy()
    for got in (st.to_anndata(cells=a.obs_names[cells], genes=genes, obsp=True),
                ad.read_h5ad(st.write_h5ad(tmp_path / "sub.h5ad", cells=cells, genes=genes, obsp=True))):
        assert_adata_equal(ref, got)
    # boolean mask also works, and Store.subset(...).to_anndata agrees
    mask = np.zeros(a.n_obs, bool)
    mask[cells] = True
    assert_adata_equal(a[mask].copy(), st.subset(mask, graphs=True).to_anndata(obsp=True))


def test_scanpy_reads_written_file(tmp_path):
    sc = pytest.importorskip("scanpy")
    a = make_adata(seed=7)
    a.write_h5ad(tmp_path / "in.h5ad")
    st = arsc.read(tmp_path / "in.h5ad", store=tmp_path / "st")
    b = sc.read_h5ad(st.write_h5ad(tmp_path / "o.h5ad"))
    assert_adata_equal(a, b, obsp=False)
    sc.pp.normalize_total(b, target_sum=1e4)  # a writable, ordinary AnnData
    sc.pp.log1p(b)


def test_dtype_policy(tmp_path):
    a = make_adata(seed=2, x_dtype=np.float64)
    a.layers["frac"] = sp.csr_matrix(a.layers["lognorm"] / 7.0)  # non-integer float64
    a.write_h5ad(tmp_path / "in.h5ad")
    st = arsc.read(tmp_path / "in.h5ad", store=tmp_path / "st")
    info = st.manifest["assays"]["RNA"]["layers"]
    assert info["X"]["dtype"] == "float32" and info["X"]["integer_valued"]        # exact downcast
    assert info["counts"]["dtype"] == "float32"                                      # int64 source
    assert info["frac"]["dtype"] == "float64" and not info["frac"]["integer_valued"]
    b = st.to_anndata()
    assert b.X.dtype == np.float64 and b.layers["counts"].dtype == np.int64      # restored
    assert_matrix_equal(a.layers["frac"], b.layers["frac"])
    c = st.to_anndata(restore_dtype=False)
    assert c.X.dtype == np.float32


def test_from_anndata_legacy_api(tmp_path):
    a = make_adata(seed=4, g_raw=40)
    a.raw = ad.AnnData(X=a.layers["counts"].astype(np.float32), obs=pd.DataFrame(index=a.obs_names), var=a.var[[]])
    a.write_h5ad(tmp_path / "in.h5ad")
    out = arsc.from_anndata(tmp_path / "in.h5ad", tmp_path / "st", layers=("counts",), use_raw_for="raw")
    st = arsc.open(out)
    assert set(st.layers("RNA")) == {"counts", "raw"}
    assert_matrix_equal(st.X("RNA", "counts"), a.layers["counts"].astype(np.float32))
    assert "pca" in st.reductions and "umap" in st.reductions
    assert arsc.from_anndata(tmp_path / "in.h5ad", tmp_path / "st") == out  # idempotent


def test_store_exists_and_errors(tmp_path):
    a = make_adata(seed=6)
    a.write_h5ad(tmp_path / "in.h5ad")
    arsc.read(tmp_path / "in.h5ad", store=tmp_path / "st")
    assert isinstance(arsc.read(tmp_path / "in.h5ad", store=tmp_path / "st"), arsc.Store)  # reopened
    a.write_h5ad(tmp_path / "other.h5ad")
    with pytest.raises(FileExistsError):
        arsc.read(tmp_path / "other.h5ad", store=tmp_path / "st")
    with pytest.raises(KeyError, match="layer 'nope'"):
        arsc.open(tmp_path / "st").X("RNA", "nope")
    with pytest.raises(KeyError, match="assay 'ADT'"):
        arsc.open(tmp_path / "st").X("ADT")
    with pytest.raises(FileNotFoundError, match="not an arsc store"):
        arsc.open(tmp_path / "in.h5ad")
    (tmp_path / "x.h5Seurat").write_bytes(b"")
    with pytest.raises(ArscFormatError, match="SeuratDisk"):
        arsc.read(tmp_path / "x.h5Seurat")
    (tmp_path / "x.qs").write_bytes(b"")
    with pytest.raises(ValueError, match="store="):
        arsc.read(tmp_path / "x.qs")
    (tmp_path / "x.txt").write_text("")
    with pytest.raises(ArscFormatError, match="unknown format"):
        arsc.read(tmp_path / "x.txt")
    with pytest.raises(FileExistsError):
        st = arsc.open(tmp_path / "st")
        st.write_h5ad(tmp_path / "in.h5ad")


# ----------------------------------------------------------------------------- Seurat-style store
def test_seurat_style_store_export(tmp_path):
    """A store with counts + data layers and a second assay (as export_seurat.R writes) -> AnnData."""
    rng = np.random.default_rng(0)
    n, g = 120, 30
    counts = sp.random(n, g, density=0.3, format="csr", random_state=0,
                       data_rvs=lambda k: rng.integers(1, 30, k)).astype(np.float32)
    b = arsc.StoreBuilder(tmp_path / "st")
    b.set_cells([f"c{i}" for i in range(n)], pd.DataFrame({"cluster": pd.Categorical(rng.choice(list("abc"), n))},
                                                          index=[f"c{i}" for i in range(n)]))
    b.add_assay("RNA", [f"g{j}" for j in range(g)], variable_features=["g1", "g2"])
    b.add_layer("RNA", "counts", counts)
    b.add_layer("RNA", "data", arsc.pp.log_normalize(counts))
    b.add_assay("ADT", ["CD3", "CD4"])
    b.add_layer("ADT", "counts", rng.integers(0, 100, (n, 2)))
    b.add_reduction("pca", rng.normal(size=(n, 5)))
    st = b.finish()
    a = st.to_anndata()
    assert_matrix_equal(a.X, counts)
    assert list(a.layers) == ["data"]
    assert a.var.highly_variable.sum() == 2
    assert a.obsm["X_pca"].shape == (n, 5)
    adt = st.to_anndata(assay="ADT")
    assert list(adt.var_names) == ["CD3", "CD4"] and adt.X.dtype == np.int64


# ----------------------------------------------------------------------------- 10x
def _write_10x_v3(path, M_cells_x_feat, ids, names, types, barcodes):
    csc = sp.csc_matrix(M_cells_x_feat.T)  # features x cells, CSC = per cell
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["barcodes"] = np.array(barcodes, dtype="S")
        g["data"] = csc.data.astype(np.int32)
        g["indices"] = csc.indices.astype(np.int64)
        g["indptr"] = csc.indptr.astype(np.int64)
        g["shape"] = np.array(csc.shape, dtype=np.int32)
        fe = g.create_group("features")
        fe["id"] = np.array(ids, dtype="S")
        fe["name"] = np.array(names, dtype="S")
        fe["feature_type"] = np.array(types, dtype="S")
        fe["genome"] = np.array(["GRCh38" if t == "Gene Expression" else "" for t in types], dtype="S")
        fe["_all_tag_keys"] = np.array([b"genome"])


def _tenx_data(seed=0, n=150, g=25, n_adt=4):
    rng = np.random.default_rng(seed)
    M = sp.random(n, g + n_adt, density=0.35, format="csr", random_state=seed,
                  data_rvs=lambda k: rng.integers(1, 200, k)).astype(np.int32)
    names = [f"GENE{j % 20}" for j in range(g)] + [f"CD{j}" for j in range(n_adt)]  # duplicates on purpose
    ids = [f"ENSG{j:06d}" for j in range(g)] + [f"ab{j}" for j in range(n_adt)]
    types = ["Gene Expression"] * g + ["Antibody Capture"] * n_adt
    barcodes = [f"AAAC{i:06d}-1" for i in range(n)]
    return M, ids, names, types, barcodes


def test_10x_h5_vs_scanpy(tmp_path):
    sc = pytest.importorskip("scanpy")
    M, ids, names, types, bcs = _tenx_data()
    p = tmp_path / "filtered_feature_bc_matrix.h5"
    _write_10x_v3(p, M, ids, names, types, bcs)
    for store in (None, tmp_path / "st"):
        x = arsc.read(p, store=store, chunk_nnz=50)
        assert set(x.assays) == {"RNA", "ADT"}
        ref = sc.read_10x_h5(p, gex_only=False)
        gex = ref[:, ref.var.feature_types == "Gene Expression"].copy()
        gex.var_names_make_unique()
        rna = x.to_anndata(assay="RNA", restore_dtype=False)
        assert list(rna.obs_names) == list(ref.obs_names)
        assert list(rna.var_names) == list(gex.var_names)
        assert_matrix_equal(rna.X, gex.X)  # float32 both
        assert_frame_same(rna.var, gex.var)
        adt = ref[:, ref.var.feature_types == "Antibody Capture"]
        assert_matrix_equal(x.X("ADT", "counts"), sp.csr_matrix(adt.X))
        assert list(x.features("ADT")) == list(adt.var_names)
    only = arsc.read_10x_h5(p, feature_types=["Gene Expression"])
    assert list(only.assays) == ["RNA"]
    ids_names = arsc.read_10x_h5(p, var_names="gene_ids")
    assert list(ids_names.features("RNA")) == ids[:25]


def test_10x_h5_v2(tmp_path):
    sc = pytest.importorskip("scanpy")
    M, ids, names, _, bcs = _tenx_data(seed=1, n_adt=0)
    csc = sp.csc_matrix(M.T)
    p = tmp_path / "v2.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("GRCh38")
        g["barcodes"] = np.array(bcs, dtype="S")
        g["data"], g["indices"], g["indptr"] = csc.data, csc.indices, csc.indptr
        g["shape"] = np.array(csc.shape, dtype=np.int32)
        g["genes"] = np.array(ids, dtype="S")
        g["gene_names"] = np.array(names, dtype="S")
    ref = sc.read_10x_h5(p)
    ref.var_names_make_unique()
    ds = arsc.read(p)
    assert list(ds.features("RNA")) == list(ref.var_names)
    assert_matrix_equal(ds.X("RNA", "counts"), sp.csr_matrix(ref.X))


@pytest.mark.parametrize("v3", [True, False])
def test_10x_mtx_vs_scanpy(tmp_path, v3):
    """v3: gzipped matrix/features/barcodes with feature types; v2: plain matrix.mtx + genes.tsv."""
    sc = pytest.importorskip("scanpy")
    M, ids, names, types, bcs = _tenx_data(seed=2, n_adt=4 if v3 else 0)
    d = tmp_path / "mtx"
    d.mkdir()
    op = (lambda p, m: gzip.open(str(p) + ".gz", m)) if v3 else (lambda p, m: open(p, m))
    with op(d / "matrix.mtx", "wb") as fh:
        mmwrite(fh, sp.coo_matrix(M.T))
    with op(d / ("features.tsv" if v3 else "genes.tsv"), "wt") as fh:
        rows = zip(ids, names, types) if v3 else zip(ids, names)
        fh.write("".join("\t".join(r) + "\n" for r in rows))
    with op(d / "barcodes.tsv", "wt") as fh:
        fh.write("".join(f"{b}\n" for b in bcs))
    ds = arsc.read(d)
    ref = sc.read_10x_mtx(d, gex_only=False) if v3 else sc.read_10x_mtx(d)
    gex = ref[:, ref.var.feature_types == "Gene Expression"] if v3 else ref
    assert list(ds.features("RNA")) == list(gex.var_names)
    assert_matrix_equal(ds.X("RNA", "counts"), sp.csr_matrix(gex.X))
    assert list(ds.cells) == list(ref.obs_names)
    if v3:
        assert_matrix_equal(ds.X("ADT", "counts"), sp.csr_matrix(ref[:, ref.var.feature_types == "Antibody Capture"].X))
    else:
        assert list(ds.assays) == ["RNA"]


def test_make_unique_matches_anndata():
    from anndata.utils import make_index_unique
    rng = np.random.default_rng(0)
    for _ in range(20):
        names = list(rng.choice(["A", "B", "A-1", "C", "B-1", "B-2"], 30))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = list(make_index_unique(pd.Index(names)))
        assert arsc.make_unique(names) == ref


def test_builder_layer_validation(tmp_path):
    b = arsc.StoreBuilder(None)
    b.set_cells(["a", "b"])
    with pytest.raises(ArscFormatError, match="not unique"):
        b.add_assay("RNA", ["x", "x"])
    b.add_assay("RNA", ["x", "y"])
    with pytest.raises(ArscFormatError, match="2\\*\\*24"):
        b.add_layer("RNA", "counts", np.array([[2 ** 25, 0], [1, 1]], dtype=np.int64))
    with pytest.raises(ArscFormatError, match="no layers"):
        _finish_without_layers()


def _finish_without_layers():
    b = arsc.StoreBuilder(None)
    b.set_cells(["a"])
    b.add_assay("RNA", ["x"])
    b.finish()


def test_dataset_get_features_matches_store(tmp_path):
    import arsc
    ds, st = arsc.datasets.toy(n_cells=400), arsc.datasets.toy(n_cells=400, store=tmp_path / "st")
    assert ds.get_features(["CD8A", "GZMK"]).equals(st.get_features(["CD8A", "GZMK"]))
    with pytest.raises(KeyError, match="NOPE"):
        ds.get_features(["NOPE"])


def test_write_h5ad_uses_one_index_dtype(tmp_path):
    """indices and indptr share one dtype (scipy/scanpy in-place ops reject mixed int32/int64)."""
    ad = pytest.importorskip("anndata")
    ds = arsc.datasets.toy(n_cells=400)
    f = ds.write_h5ad(tmp_path / "toy.h5ad", assay="RNA")
    X = ad.read_h5ad(f).X
    assert X.indices.dtype == X.indptr.dtype == np.int32
    X.data[:] = 1.0
    X += X  # an in-place sparse op that fails on mixed index dtypes in recent scipy
