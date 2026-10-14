"""arsc.scanpy_extra vs scanpy 1.11.5 on a 20,000-cell subsample of the 02619 CD8 store.

Run: .venv/Scripts/python.exe -m pytest -q vendor/arsc/tests/test_scanpy_extra.py
Skipped when the store is missing. seurat_v3 needs scikit-misc and the palantir test needs palantir
(neither is in the venv; they were pip-installed with --target into scratch/arsc_scanpy/pkgs for validation).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

warnings.filterwarnings("ignore")
STORE = Path("E:/autoresearch/data/02619_neoadj/stores/cd8_final")
PKGS = Path("E:/autoresearch/scratch/arsc_scanpy/pkgs")
if PKGS.exists():
    sys.path.append(str(PKGS))  # appended: never shadows the venv packages

import arsc  # noqa: E402
from arsc import scanpy_extra as sx  # noqa: E402

pytestmark = pytest.mark.skipif(not STORE.exists(), reason="02619 cd8_final store not available")
sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")


def _need_loess():
    """scanpy seurat_v3 needs the compiled scikit-misc loess extension, not just the package."""
    try:
        from skmisc.loess import loess  # noqa: F401
    except ImportError:
        pytest.skip("scikit-misc loess extension not importable")


@pytest.fixture(scope="module")
def d():
    st = arsc.open(str(STORE))
    idx = np.sort(np.random.default_rng(0).choice(st.n_cells, 20000, replace=False))
    C = st.X()[idx]
    obs = st.obs.iloc[idx].reset_index(drop=True)
    feats = pd.Index(st.features())
    L = sx.normalize_total_log1p(C)
    a = ad.AnnData(sp.csr_matrix(C.astype(np.float64)), obs=obs.copy())
    a.var_names = feats
    a.obs_names = [str(i) for i in range(a.n_obs)]
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.obs["lab"] = a.obs["leiden"].astype(str).astype("category")
    return dict(st=st, idx=idx, C=C, obs=obs, feats=feats, L=L, a=a)


def test_normalize_equals_scanpy(d):
    assert abs(d["L"] - d["a"].X).max() < 1e-12
    # arsc.pp.log_normalize (Seurat arithmetic) agrees to rounding but not bitwise
    assert abs(arsc.pp.log_normalize(d["C"]) - d["a"].X).max() < 1e-12


def test_scale_pca_vs_scanpy(d):
    a = d["a"].copy()
    sc.pp.highly_variable_genes(a, n_top_genes=1000)
    hv = np.flatnonzero(a.var.highly_variable.to_numpy())
    b = a[:, hv].copy()
    sc.pp.scale(b, max_value=10)
    Z, _, _ = arsc.pp.scale(d["L"][:, hv], scale_max=10)
    assert np.abs(b.X - Z).max() < 1e-10
    sc.tl.pca(b, n_comps=30, svd_solver="arpack")
    res = arsc.pp.pca(Z, 30)
    r = [abs(np.corrcoef(b.obsm["X_pca"][:, k], res.embedding[:, k])[0, 1]) for k in range(30)]
    assert min(r) > 0.995
    assert np.abs(b.uns["pca"]["variance_ratio"] - res.variance_ratio).max() < 1e-4


def test_score_genes_exact(d):
    gl = ["TCF7", "GZMK", "PDCD1", "ENTPD1", "LAG3", "TOX", "NOT_A_GENE"]
    ref = sc.tl.score_genes(d["a"], gene_list=gl, score_name="s", use_raw=False, copy=True).obs["s"].to_numpy()
    got = sx.score_genes(d["L"], d["feats"], gl)
    assert np.abs(ref - got).max() < 1e-12
    gl2 = ["IL7R", "CCR7", "SELL", "LEF1", "GZMB", "NKG7"]
    ref = sc.tl.score_genes(d["a"], gene_list=gl2, score_name="s", ctrl_size=20, n_bins=10, random_state=3,
                            use_raw=False, copy=True).obs["s"].to_numpy()
    got = sx.score_genes(d["L"], d["feats"], gl2, ctrl_size=20, n_bins=10, random_state=3)
    assert np.abs(ref - got).max() < 1e-12


@pytest.mark.parametrize("method,kw", [("wilcoxon", {}), ("wilcoxon", {"tie_correct": True}),
                                       ("t-test", {}), ("t-test_overestim_var", {})])
def test_rank_genes_groups(d, method, kw):
    a = d["a"]
    sc.tl.rank_genes_groups(a, "lab", method=method, pts=True, use_raw=False, **kw)
    ref = sc.get.rank_genes_groups_df(a, group=None)
    ref["group"] = ref["group"].astype(str)
    got = sx.rank_genes_groups(d["L"], a.obs["lab"].to_numpy(), d["feats"], method=method, pts=True, **kw)
    m = ref.merge(got, on=["group", "names"], suffixes=("_s", "_a"))
    assert len(m) == len(ref) == len(got)
    # scanpy stores scores / logfoldchanges as float32
    assert (np.abs(m.scores_s - m.scores_a) / np.maximum(np.abs(m.scores_s), 1)).max() < 1e-6
    assert np.abs(m.logfoldchanges_s - m.logfoldchanges_a).max() < 1e-5
    assert np.abs(m.pvals_s - m.pvals_a).max() < 1e-9
    assert np.abs(m.pvals_adj_s - m.pvals_adj_a).max() < 1e-9


def test_rank_genes_groups_reference_group(d):
    a = d["a"]
    sc.tl.rank_genes_groups(a, "lab", method="wilcoxon", reference="0", use_raw=False)
    ref = sc.get.rank_genes_groups_df(a, group=None)
    ref["group"] = ref["group"].astype(str)
    got = sx.rank_genes_groups(d["L"], a.obs["lab"].to_numpy(), d["feats"], reference="0")
    m = ref.merge(got, on=["group", "names"], suffixes=("_s", "_a"))
    assert len(m) == len(ref)
    assert (np.abs(m.scores_s - m.scores_a) / np.maximum(np.abs(m.scores_s), 1)).max() < 1e-6
    assert np.abs(m.pvals_s - m.pvals_a).max() < 1e-9


def test_rank_genes_groups_speed(d):
    a = d["a"]
    t = time.time()
    sc.tl.rank_genes_groups(a, "lab", method="wilcoxon", pts=True, use_raw=False)
    ts = time.time() - t
    t = time.time()
    sx.rank_genes_groups(d["L"], a.obs["lab"].to_numpy(), d["feats"], pts=True)
    ta = time.time() - t
    print(f"rank_genes_groups wilcoxon 20k cells: scanpy {ts:.2f}s, arsc {ta:.2f}s")
    assert ta < ts


@pytest.mark.parametrize("flavor,kw", [("seurat", {"n_top_genes": 1000}), ("seurat", {}),
                                       ("cell_ranger", {"n_top_genes": 1000})])
def test_hvg_log_flavors(d, flavor, kw):
    b = d["a"].copy()
    sc.pp.highly_variable_genes(b, flavor=flavor, **kw)
    r = sx.highly_variable_genes(d["L"], d["feats"], flavor=flavor, **kw)
    assert (b.var.highly_variable.to_numpy() == r.highly_variable.to_numpy()).all()
    for c in ("means", "dispersions"):
        assert np.nanmax(np.abs(b.var[c].to_numpy() - r[c].to_numpy())) < 1e-10
    assert np.nanmax(np.abs(b.var["dispersions_norm"].to_numpy() - r["dispersions_norm"].to_numpy())) < 1e-5  # float32


def test_hvg_seurat_v3(d):
    _need_loess()
    b = d["a"].copy()
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", layer="counts", n_top_genes=1000)
    r = sx.highly_variable_genes(d["C"], d["feats"], flavor="seurat_v3", n_top_genes=1000)
    assert np.nanmax(np.abs(b.var.variances_norm.to_numpy() - r.variances_norm.to_numpy())) < 1e-8
    # sets differ only by exact ties at the cutoff
    ov = len(set(d["feats"][b.var.highly_variable]) & set(d["feats"][r.highly_variable]))
    assert ov >= 995


def test_hvg_seurat_v3_batch(d):
    _need_loess()
    b = d["a"].copy()
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", layer="counts", n_top_genes=1000, batch_key="batch")
    r = sx.highly_variable_genes(d["C"], d["feats"], flavor="seurat_v3", n_top_genes=1000, batch_key=d["obs"].batch.to_numpy())
    assert np.corrcoef(b.var.variances_norm, r.variances_norm)[0, 1] > 0.999999
    ov = len(set(d["feats"][b.var.highly_variable]) & set(d["feats"][r.highly_variable]))
    assert ov >= 930  # tie order among near-identical normalized variances is arbitrary


def test_dendrogram_regress_qc(d):
    a = d["a"].copy()
    sc.pp.highly_variable_genes(a, n_top_genes=1000)
    b = a[:, a.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=30, svd_solver="arpack")
    sc.tl.dendrogram(b, groupby="lab", use_rep="X_pca", n_pcs=30)
    dd = sx.dendrogram(b.obsm["X_pca"], b.obs.lab.to_numpy())
    assert np.abs(b.uns["dendrogram_lab"]["linkage"] - dd["linkage"]).max() < 1e-6
    assert b.uns["dendrogram_lab"]["categories_ordered"] == dd["categories_ordered"]
    # regress_out
    a2 = d["a"].copy()
    sc.pp.regress_out(a2, ["total_counts", "pct_counts_mt"])
    R = sx.regress_out(d["L"], d["obs"][["total_counts", "pct_counts_mt"]].to_numpy())
    assert np.abs(a2.X - R).max() < 1e-9
    # qc metrics
    a3 = ad.AnnData(sp.csr_matrix(d["C"].astype(np.float64)), obs=pd.DataFrame(index=a.obs_names))
    a3.var_names = d["feats"]
    a3.var["mt"] = d["feats"].str.startswith("MT-")
    a3.var["ribo"] = d["feats"].str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(a3, qc_vars=["mt", "ribo"], percent_top=[20], log1p=True, inplace=True)
    q = sx.qc_metrics(d["C"], d["feats"], {"mt": a3.var.mt.to_numpy(), "ribo": a3.var.ribo.to_numpy()}, [20])
    for c in q.columns:
        assert np.nanmax(np.abs(a3.obs[c].to_numpy() - q[c].to_numpy())) < 1e-9, c


@pytest.fixture(scope="module")
def nbr(d):
    n = 3000  # scanpy uses exact kNN below 4,096 cells (approximate pynndescent above)
    E = np.asarray(d["st"].embedding("pca_harmony"))[d["idx"]][:n]
    b = ad.AnnData(np.zeros((n, 1)), obs=pd.DataFrame(index=[str(i) for i in range(n)]))
    b.obsm["X_pca_harmony"] = E
    sc.pp.neighbors(b, n_neighbors=30, use_rep="X_pca_harmony")
    return dict(E=E, b=b, nn=sx.neighbors(E, 30))


def test_neighbors_leiden_umap(nbr):
    b, nn = nbr["b"], nbr["nn"]
    assert abs(b.obsp["connectivities"] - nn["connectivities"]).max() < 1e-10
    assert abs(b.obsp["distances"] - nn["distances"]).max() < 1e-9
    for res in (0.5, 2.0):
        sc.tl.leiden(b, resolution=res, random_state=0)
        lab = sx.leiden(nn["connectivities"], res)
        assert (b.obs.leiden.astype(int).to_numpy() == lab).all()
    b2 = b.copy()
    sc.tl.umap(b2, min_dist=0.4, spread=1.0)
    U = sx.umap_from_connectivities(nn["connectivities"], min_dist=0.4, spread=1.0)
    assert np.abs(b2.obsm["X_umap"] - U).max() < 1e-6


def test_neighbors_cosine(nbr):
    b = ad.AnnData(np.zeros((3000, 1)), obs=pd.DataFrame(index=[str(i) for i in range(3000)]))
    b.obsm["X_e"] = nbr["E"]
    sc.pp.neighbors(b, n_neighbors=16, use_rep="X_e", metric="cosine")
    nc = sx.neighbors(nbr["E"], 16, metric="cosine")
    assert abs(b.obsp["connectivities"] - nc["connectivities"]).max() < 1e-8


def test_diffmap_dpt(nbr):
    b, nn = nbr["b"].copy(), nbr["nn"]
    sc.tl.diffmap(b, n_comps=15)
    dm = sx.diffmap(nn["connectivities"], 15)
    assert np.abs(b.uns["diffmap_evals"] - dm["evals"]).max() < 1e-5
    for k in range(10):
        assert abs(np.corrcoef(b.obsm["X_diffmap"][:, k], dm["X_diffmap"][:, k])[0, 1]) > 0.9999
    b.uns["iroot"] = 0
    sc.tl.dpt(b, n_dcs=10)
    assert np.corrcoef(b.obs.dpt_pseudotime, sx.dpt(dm, 0, 10))[0, 1] > 0.9999


def test_palantir_pieces(d, nbr):
    palantir = pytest.importorskip("palantir")
    if not hasattr(palantir, "utils"):
        pytest.skip("palantir install incomplete (no palantir.utils)")
    E, n = nbr["E"], 3000
    bb = ad.AnnData(d["L"][:n].copy(), obs=pd.DataFrame(index=[str(i) for i in range(n)]))
    bb.obsm["X_pca_harmony"] = E
    ref = palantir.utils.run_diffusion_maps(bb, n_components=5, pca_key="X_pca_harmony")
    got = sx.palantir_diffusion_maps(E, 5)
    assert np.abs(ref["EigenValues"].values - got["EigenValues"]).max() < 1e-8
    assert abs(ref["kernel"] - got["kernel"]).max() < 1e-10
    ms = palantir.utils.determine_multiscale_space(bb)
    mm = sx.palantir_multiscale_space(got)
    assert ms.shape == mm.shape
    for k in range(mm.shape[1]):
        assert abs(np.corrcoef(ms.values[:, k], mm[:, k])[0, 1]) > 0.9999
    palantir.utils.run_magic_imputation(bb)
    im = sx.magic_impute(d["L"][:n], got["T"])
    assert abs(bb.layers["MAGIC_imputed_data"] - im).max() < 1e-4
