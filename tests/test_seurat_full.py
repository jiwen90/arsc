"""Full Seurat exports (``export_seurat(..., full=True)``): the store carries everything the object holds.

1. Reader (no R needed): a store laid out as a full export writes it (a data layer that is an alias of counts, a
   dense scale.data, a v5 split, per-feature metadata, a Neighbor object, Idents, typed ``uns`` with file references,
   the unconvertible list) reads back through the Store API.
2. Exporter (skipped without R + SeuratObject + qs): a small Seurat object with split v5 layers, scale.data, typed
   metadata, a reduction with projected loadings, a graph, a Neighbor object and misc (with an environment) exported
   with ``full=True`` holds every component, and the environment is listed as unconvertible.
"""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import arsc


def _npy(path, a):
    np.save(path, np.asarray(a))


def test_full_store_reader(tmp_path):
    rng = np.random.default_rng(3)
    n, g = 40, 12
    cells = [f"c{i}" for i in range(n)]
    counts = sp.random(n, g, density=0.4, format="csr", random_state=3,
                       data_rvs=lambda k: rng.integers(1, 20, k)).astype(np.float32)
    b = arsc.StoreBuilder(tmp_path / "st")
    b.set_cells(cells, pd.DataFrame({"celltype": pd.Categorical(rng.choice(["T", "B"], n))}, index=cells))
    b.add_assay("RNA", [f"g{j}" for j in range(g)])
    b.add_layer("RNA", "counts", counts)
    b.finish(open_store=False)
    d = tmp_path / "st"
    m = json.loads((d / "manifest.json").read_text())
    a = m["assays"]["RNA"]
    a["layers"]["data"] = dict(a["layers"]["counts"], alias_of="counts")
    a["layers"]["counts"]["split"] = {"parts": ["counts.P1", "counts.P2"], "file": "counts.split.txt"}
    (d / "assays" / "RNA" / "counts.split.txt").write_text("\n".join(["P1"] * 25 + ["P2"] * 15) + "\n")
    sc = rng.normal(size=(n, 3)).astype(np.float32)
    _npy(d / "assays" / "RNA" / "scale.data.dense.npy", sc)
    (d / "assays" / "RNA" / "scale.data.dense_features.txt").write_text("g0\ng1\ng2\n")
    a["dense"] = {"scale.data": {"shape": [n, 3], "dtype": "float32", "file": "scale.data.dense.npy",
                                 "features_file": "scale.data.dense_features.txt"}}
    pd.DataFrame({"feature": [f"g{j}" for j in range(g)], "vst.mean": np.arange(g) / 2.0,
                  "hv": ["TRUE"] * 3 + ["FALSE"] * (g - 3)}).to_csv(d / "assays" / "RNA" / "var.csv", index=False)
    a["var"] = {"file": "var.csv", "columns": {"vst.mean": {"type": "double"}, "hv": {"type": "logical"}}}
    (d / "neighbors").mkdir()
    idx = np.tile(np.arange(n)[:, None], (1, 3)).astype(np.int32)
    _npy(d / "neighbors" / "RNA.nn.idx.npy", idx)
    _npy(d / "neighbors" / "RNA.nn.dist.npy", np.ones((n, 3)))
    m["neighbors"] = {"RNA.nn": {"shape": [n, 3], "index_base": 0, "idx": "RNA.nn.idx.npy", "dist": "RNA.nn.dist.npy"}}
    m["idents_column"] = "celltype"
    m["unconvertible"] = [{"path": "misc/env", "class": "environment", "reason": "an R environment"}]
    (d / "uns").mkdir()
    _npy(d / "uns" / "0001_big.npy", np.arange(6.0))
    (d / "uns.json").write_text(json.dumps({
        "misc": {"note": "QC", "mat": {"__ndarray__": [1, 2, 3, 4, 5, 6], "dtype": "<i4", "shape": [2, 3], "order": "F"},
                 "lev": {"__ndarray__": ["a", "b", "a"], "dtype": "str", "levels": ["b", "a"]},
                 "df": {"__dataframe__": {"gene": {"__ndarray__": ["X", "Y"], "dtype": "str"},
                                          "lfc": {"__ndarray__": [1.5, None], "dtype": "<f8"}}, "index": ["1", "2"]},
                 "big": {"__npy__": "uns/0001_big.npy"}, "f": {"__deparsed__": "y ~ x"}}}))
    (d / "manifest.json").write_text(json.dumps(m))
    st = arsc.open(d)
    assert np.array_equal(st.X("RNA", "data").toarray(), counts.toarray())  # alias: the same arrays
    assert st.split("RNA", "counts").value_counts().to_dict() == {"P1": 25, "P2": 15}
    assert st.split("RNA", "data") is None
    assert st.dense_layers("RNA") == ["scale.data"]
    D = st.dense("RNA")
    assert list(D.columns) == ["g0", "g1", "g2"] and np.allclose(D.values, sc)
    with pytest.raises(KeyError, match="dense layers"):
        st.X("RNA", "scale.data")
    v = st.var("RNA")
    assert list(v.columns) == ["vst.mean", "hv"] and v["hv"].sum() == 3 and (d / "assays" / "RNA" / "var.parquet").exists()
    assert st.neighbors == ["RNA.nn"] and st.neighbor("RNA.nn")[0].shape == (n, 3)
    assert st.idents is not None and set(st.idents.cat.categories) == {"T", "B"}
    assert st.unconvertible[0]["path"] == "misc/env"
    u = st.uns["misc"]
    assert u["mat"].shape == (2, 3) and u["mat"][0, 1] == 3  # column-major from R
    assert isinstance(u["lev"], pd.Categorical) and list(u["lev"].categories) == ["b", "a"]
    assert list(u["df"]["gene"]) == ["X", "Y"] and np.isnan(u["df"]["lfc"].iloc[1])
    assert np.array_equal(u["big"], np.arange(6.0)) and u["f"]["__deparsed__"] == "y ~ x"


def test_uns_names_restored(tmp_path):
    """R names and dimnames of uns arrays come back: inline (list or older dict form) and sidecar files (.npy, .txt)."""
    from arsc.formats import _uns_from_json
    (tmp_path / "uns").mkdir()
    _npy(tmp_path / "uns" / "0001_m.npy", np.asfortranarray(np.arange(6.0).reshape(2, 3, order="F")))
    (tmp_path / "uns" / "0001_m.npy.dimnames1.txt").write_text("r1\nr2\n")
    (tmp_path / "uns" / "0001_m.npy.dimnames2.txt").write_text("c1\n\\N\nc3\n")
    _npy(tmp_path / "uns" / "0002_v.npy", np.array([1.0, np.nan, 0.0]))
    (tmp_path / "uns" / "0002_v.npy.names.txt").write_text("a\nb\nc\n")
    (tmp_path / "uns" / "0003_t.txt").write_text("x\n\\N\ny\nz\n")
    (tmp_path / "uns" / "0003_t.txt.dimnames2.txt").write_text("k1\nk2\n")
    j = {"m": {"__npy__": "uns/0001_m.npy", "logical": False, "dimnames_files": ["uns/0001_m.npy.dimnames1.txt",
                                                                             "uns/0001_m.npy.dimnames2.txt"]},
         "v": {"__npy__": "uns/0002_v.npy", "logical": True, "names_file": "uns/0002_v.npy.names.txt"},
         "t": {"__txt__": "uns/0003_t.txt", "shape": [2, 2], "order": "F", "na": "\\N",
               "dimnames_files": [None, "uns/0003_t.txt.dimnames2.txt"]},
         "i": {"__ndarray__": [1, 2, 3, 4], "dtype": "<i4", "shape": [2, 2], "order": "F",
               "dimnames": [["a", "b"], None]},
         "old": {"__ndarray__": [1, 2, 3, 4], "dtype": "<i4", "shape": [2, 2], "order": "F",
                 "dimnames": {"r": ["a", "b"], "c": ["x", "y"]}},
         "plain": {"__npy__": "uns/0001_m.npy"}}
    u = _uns_from_json(j, base=tmp_path)
    assert list(u["m"].index) == ["r1", "r2"] and list(u["m"].columns) == ["c1", None, "c3"]
    assert u["m"].loc["r2", "c3"] == 5.0
    assert list(u["v"].index) == ["a", "b", "c"] and np.isnan(u["v"]["b"])  # a logical with NA stays float
    assert isinstance(u["t"], pd.DataFrame) and list(u["t"].columns) == ["k1", "k2"] and u["t"].iloc[1, 0] is None
    assert u["t"].iloc[0, 1] == "y" and list(u["t"].index) == [0, 1]
    assert list(u["i"].index) == ["a", "b"] and u["i"].iloc[1, 1] == 4
    assert list(u["old"].columns) == ["x", "y"]
    assert isinstance(u["plain"], np.ndarray)


R_FIXTURE = r'''
suppressPackageStartupMessages({ library(SeuratObject); library(Matrix) })
a <- commandArgs(trailingOnly = TRUE); set.seed(1)
nc <- 60; ng <- 30; cells <- paste0("cell", 1:nc); genes <- paste0("G", 1:ng)
m <- rsparsematrix(ng, nc, 0.3, rand.x = function(n) sample(1:9, n, TRUE)); m[1, ] <- m[1, ] + 1
dimnames(m) <- list(genes, cells)
o <- CreateSeuratObject(counts = m, min.cells = 0, min.features = 0)
nd <- m; nd@x <- log1p(nd@x); LayerData(o, assay = "RNA", layer = "data") <- nd
o$grp <- factor(rep(c("a", "b"), 30), levels = c("a", "b", "unused")); o$s <- rnorm(nc); o$s[2] <- NA
emb <- matrix(rnorm(nc * 3), nc, 3, dimnames = list(cells, paste0("PC_", 1:3)))
ld <- matrix(rnorm(ng * 3), ng, 3, dimnames = list(genes, paste0("PC_", 1:3)))
o[["pca"]] <- CreateDimReducObject(embeddings = emb, loadings = ld, stdev = c(3, 2, 1), key = "PC_", assay = "RNA")
o[["pca"]]@feature.loadings.projected <- ld * 2
G <- Matrix::sparseMatrix(i = 1:nc, j = c(2:nc, 1), x = 1, dims = c(nc, nc), dimnames = list(cells, cells))
o[["RNA_nn"]] <- as.Graph(G)
o[["RNA.nn"]] <- new("Neighbor", nn.idx = cbind(1:nc, c(2:nc, 1)), nn.dist = matrix(0.5, nc, 2), cell.names = cells)
# a query -> reference Neighbor (Azimuth): values index a 500-cell reference, not this object's cells
o[["query_ref.nn"]] <- new("Neighbor", nn.idx = cbind(rep(500L, nc), 1:nc + 100L), nn.dist = matrix(1, nc, 2),
                           cell.names = cells)
cl <- paste0("clone", 1:400)  # a big matrix (> 1e5 values: written as .npy) and a big named vector keep their names
big <- matrix(seq_len(400 * 400) / 7, 400, 400, dimnames = list(cl, cl))
bigv <- setNames(seq_len(120000) / 3, paste0("k", seq_len(120000)))
bigc <- matrix(rep(c("u", NA, "w"), 40000), 1000, 120, dimnames = list(NULL, paste0("c", 1:120)))
small <- matrix(1:6, 2, 3, dimnames = list(r = c("a", "b"), c = c("x", "y", "z")))
o@misc <- list(note = "x", env = new.env(), big = big, bigv = bigv, bigc = bigc, small = small,
               tab = table(c("p", "q", "q")), nonames = matrix(as.numeric(1:200000), 400, 500),
               bigdf = data.frame(f = factor(rep(c("b", "a"), 30000), levels = c("b", "a", "z")), s = rep(c("01", "2"), 30000),
                                  n = 1:60000, row.names = paste0("r", 1:60000), stringsAsFactors = FALSE))
Idents(o) <- "grp"
if (!inherits(o@assays$RNA, "Assay5")) o@assays$RNA <- as(o@assays$RNA, "Assay5")
o@assays$RNA <- suppressWarnings(split(o@assays$RNA, f = rep(c("P1", "P2"), each = 30)))
a5 <- o@assays$RNA; LayerData(a5, layer = "scale.data") <- t(scale(t(as.matrix(nd[1:5, ])))); o@assays$RNA <- a5
qs::qsave(o, a[1]); saveRDS(o@misc[setdiff(names(o@misc), "env")], sub("[.]qs$", "_misc.rds", a[1]))
cat("FIXTURE_OK\n")
'''


def _have_r():
    rs = shutil.which("Rscript")
    if not rs:
        return False
    r = subprocess.run([rs, "-e", "cat(all(sapply(c('SeuratObject','qs','jsonlite','data.table'), requireNamespace, "
                        "quietly=TRUE)))"], capture_output=True, text=True)
    return "TRUE" in r.stdout


@pytest.mark.skipif(not _have_r(), reason="needs R with SeuratObject, qs, jsonlite, data.table")
def test_export_seurat_full(tmp_path):
    (tmp_path / "fx.R").write_text(R_FIXTURE)
    r = subprocess.run([shutil.which("Rscript"), str(tmp_path / "fx.R"), str(tmp_path / "o.qs")], capture_output=True,
                       text=True)
    assert "FIXTURE_OK" in r.stdout, r.stderr[-800:]
    arsc.export_seurat(tmp_path / "o.qs", tmp_path / "st", full=True, threads=1)
    st = arsc.open(tmp_path / "st")
    assert st.n_cells == 60 and st.layers("RNA") == ["counts", "data"]
    assert st.split("RNA", "counts").value_counts().to_dict() == {"P1": 30, "P2": 30}
    assert st.dense("RNA", "scale.data").shape == (60, 5)
    assert list(st.obs["grp"].cat.categories) == ["a", "b", "unused"]  # the unused level kept
    assert int(st.obs["s"].isna().sum()) == 1
    assert st.idents is not None and list(st.idents.cat.categories) == ["a", "b"]
    rd = st.reduction("pca")
    assert rd.key == "PC_" and rd.projected is not None and np.allclose(rd.projected.values, rd.loadings.values * 2)
    assert np.allclose(rd.stdev, [3, 2, 1])
    assert st.graphs == ["RNA_nn"] and st.neighbors == ["RNA.nn", "query_ref.nn"]
    idx, dist = st.neighbor("RNA.nn")
    assert idx[0, 1] == 1 and idx[59, 1] == 0 and np.allclose(dist, 0.5)  # 0-based positions in st.cells
    assert st.manifest["neighbors"]["RNA.nn"]["index_space"] == "cells"
    qi, _ = st.neighbor("query_ref.nn")  # reference positions kept as they are (0-based), never mapped or NA
    assert st.manifest["neighbors"]["query_ref.nn"]["index_space"] == "other"
    assert qi.min() >= 0 and qi[0, 0] == 499 and qi[0, 1] == 100 and qi[59, 1] == 159
    u = st.uns["misc"]
    assert u["note"] == "x"
    cl = [f"clone{i}" for i in range(1, 401)]
    assert isinstance(u["big"], pd.DataFrame) and list(u["big"].index) == cl and list(u["big"].columns) == cl
    assert u["big"].loc["clone3", "clone2"] == pytest.approx((1 * 400 + 3) / 7)  # column-major from R
    assert isinstance(u["bigv"], pd.Series) and u["bigv"]["k5"] == pytest.approx(5 / 3)
    assert u["bigc"].shape == (1000, 120) and list(u["bigc"].columns[:2]) == ["c1", "c2"]
    assert u["bigc"].iloc[1, 0] is None and u["bigc"].iloc[2, 0] == "w" and u["bigc"].iloc[0, 1] is None  # column-major, NA kept
    assert list(u["small"].index) == ["a", "b"] and list(u["small"].columns) == ["x", "y", "z"]
    assert (u["small"].index.name, u["small"].columns.name) == ("r", "c") and u["small"].loc["b", "y"] == 4
    assert isinstance(u["tab"], pd.Series) and u["tab"].to_dict() == {"p": 1, "q": 2}
    assert isinstance(u["nonames"], np.ndarray) and u["nonames"].shape == (400, 500)
    bd = u["bigdf"]  # a big data frame (.csv): row names, character text and factor levels as in R
    assert list(bd.index[:2]) == ["r1", "r2"] and list(bd["f"].cat.categories) == ["b", "a", "z"]
    assert bd["s"].iloc[0] == "01" and bd["n"].iloc[-1] == 60000
    assert [u["path"] for u in st.unconvertible] == ["misc/env"]
    assert st.manifest["export"]["mode"] == "full"
    # the R reader gives back the same R values (names, dimnames, dims, types)
    rd_uns = (Path(arsc.__file__).parent / "R" / "read_uns.R").as_posix()
    store = (tmp_path / "st").as_posix()
    chk = (f'source("{rd_uns}"); m <- readRDS("{(tmp_path / "o_misc.rds").as_posix()}"); '
           f'u <- arsc_read_uns("{store}")$misc; '
           'for (k in names(m)) { a <- m[[k]]; if (inherits(a, "table")) a <- c(a); b <- u[[k]]; if (is.list(b) && !is.data.frame(b)) b <- unlist(b); '
           'if (!isTRUE(all.equal(a, b))) cat("DIFF", k, "\\n") }; cat("RCHECK_DONE\\n")')
    r = subprocess.run([shutil.which("Rscript"), "-e", chk], capture_output=True, text=True)
    assert "RCHECK_DONE" in r.stdout and "DIFF" not in r.stdout, r.stdout + r.stderr[-800:]
    one = subprocess.run([shutil.which("Rscript"), "-e", f'source("{rd_uns}"); x <- arsc_read_uns("{store}", "misc/big"); '
                          'cat(dim(x), rownames(x)[2], "\\n")'], capture_output=True, text=True)
    assert "400 400 clone2" in one.stdout, one.stderr[-800:]
