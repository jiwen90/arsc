"""One benchmark step for one Python tool (``arsc`` or ``scanpy``), run in a fresh process by run_suite.py.

    python suite_py.py --tool arsc --step pca --data <bench_dir>/n10000 --out <file.json>

Only the analysis call is inside the timed region; loading the data and computing inputs the step does
not own (e.g. log-normalised data for ``markers``) happen before it. The result JSON carries the epoch
timestamps of the timed region so the parent can take peak RSS inside it. Outputs for the agreement
check go to ``<data>/out/<tool>/<step>.*``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

N_HVG, N_PC, K_NN, RES = 2000, 30, 20, 0.8
# Threads: ARSC_BENCH_THREADS (set by paired_bench.py) gives both tools the same count: the parent puts it in
# AR_THREADS / NUMBA_NUM_THREADS / OMP / BLAS variables; here it also sets scanpy's n_jobs and pydeseq2's n_cpus.
BENCH_THREADS = int(os.environ.get("ARSC_BENCH_THREADS", "8"))
SCANPY_NN_SEED = int(os.environ.get("ARSC_BENCH_SCANPY_SEED", "0"))  # pynndescent random_state (run-to-run variation)
N_KEEP = 500  # cells whose normalised values are saved for the agreement check


class Timer:
    def __init__(self):
        self.t0 = self.t1 = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.t1 = time.time()


def lognorm_arsc(X):
    import arsc
    return arsc.pp.log_normalize(X)


# ------------------------------------------------------------------ arsc
def run_arsc(step, data, out, T):
    import arsc
    st = arsc.open(data / "store")
    obs = st.obs
    genes = st.features("RNA")
    common = data / "common"
    truth = json.loads((data / "truth.json").read_text())
    if step == "normalize":
        X = st.X("RNA", "counts")
        X.data[:1].sum(); _touch(X)
        with T:
            L = arsc.pp.log_normalize(X)
        sp.save_npz(out / "normalize.npz", sp.csr_matrix(L[:N_KEEP]))
    elif step == "hvg":
        X = st.X("RNA", "counts"); _touch(X)
        with T:
            h = arsc.pp.hvg_vst(X, nfeatures=N_HVG, features=genes)
        h[["variance_standardized", "highly_variable"]].to_csv(out / "hvg.csv")
    elif step == "pca":
        hv = _hvg(common, genes)
        L = arsc.pp.log_normalize(st.X("RNA", "counts"))[:, hv]
        with T:
            Z, _, _ = arsc.pp.scale(L)
            pc = arsc.pp.pca(Z, n_comps=N_PC)
        np.save(out / "pca.npy", pc.embedding.astype(np.float32))
    elif step == "neighbors":
        E = np.load(common / "pca.npy").astype(np.float64)
        with T:
            nn = arsc.find_neighbors(E, k=K_NN)
        np.save(out / "neighbors.npy", nn["idx"].astype(np.int32))
    elif step == "cluster":
        E = np.load(common / "pca.npy").astype(np.float64)
        with T:
            nn = arsc.find_neighbors(E, k=K_NN)
            lab = arsc.cluster(nn["snn"], method="leiden", resolution=RES)  # arsc's own engine (default since 0.3.0)
        np.save(out / "cluster.npy", np.asarray(lab, np.int32))
    elif step == "markers":
        L = arsc.pp.log_normalize(st.X("RNA", "counts"))
        g = obs["cell_type"].astype(str).to_numpy()
        with T:
            r = arsc.wilcoxauc(L, g, features=genes)
        r[["feature", "group", "auc", "pval", "statistic"]].to_parquet(out / "markers.parquet")
    elif step == "rank_genes":
        # scanpy-parity input: normalize_total + log1p in float32, as scanpy computes on float32 counts
        L = arsc.normalize_total_log1p(st.X("RNA", "counts"), dtype=np.float32)
        g = obs["cell_type"].astype(str).to_numpy()
        with T:
            r = arsc.rank_genes_groups(L, g, genes)
        r.rename(columns={"names": "feature", "pvals": "pval", "scores": "statistic"})[
            ["group", "feature", "statistic", "pval"]].to_parquet(out / "rank_genes.parquet")
    elif step == "pseudobulk":
        X = st.X("RNA", "counts"); _touch(X)
        with T:
            pb = arsc.pseudobulk(X, obs, ["sample", "cell_type"], features=genes)
        _save_pb(pb.X, list(pb.groups.index), out / "pseudobulk.npz")
    elif step == "de_deseq2":
        cnt, cd = _pb_common(common)
        with T:
            r = arsc.de.deseq2(cnt, cd, "~ patient + timepoint", ("timepoint", "post", "pre"), min_count=0,
                               min_samples=0, n_cpus=BENCH_THREADS)
        r[["gene", "log2FoldChange", "pvalue", "padj"]].to_csv(out / "de_deseq2.csv", index=False)
    elif step == "de_voom":
        cnt, cd = _pb_common(common)
        with T:
            r = arsc.de.voom(cnt, cd, "~ timepoint", coef=("timepoint", "post"), block="patient", filter=False)
        r[["gene", "logFC", "t", "P.Value", "adj.P.Val"]].to_csv(out / "de_voom.csv", index=False)
    elif step == "score_module":
        L = arsc.pp.log_normalize(st.X("RNA", "counts"))
        with T:
            s = arsc.add_module_score(L, genes, truth["signatures"], seed=1)
        s.to_parquet(out / "score_module.parquet")
    elif step == "score_genes":
        L = arsc.normalize_total_log1p(st.X("RNA", "counts"), dtype=np.float32)
        with T:
            s = pd.DataFrame({k: arsc.score_genes(L, genes, v, random_state=0)
                              for k, v in truth["signatures"].items()})
        s.to_parquet(out / "score_genes.parquet")
    elif step == "score_ucell":
        X = st.X("RNA", "counts"); _touch(X)
        with T:
            s = arsc.score_ucell(X, genes, truth["signatures"])
        s.to_parquet(out / "score_ucell.parquet")
    elif step == "clones":
        meta = obs[["patient", "timepoint", "cell_type", "clone_id"]].copy()
        with T:
            ct = arsc.clones.clone_table(meta, "clone_id", ["patient", "timepoint"])
            dyn = arsc.clones.clone_dynamics(ct, "pre", "post", patient="patient", time="timepoint")
            cl, pw = arsc.clones.startrac(meta, state_col="cell_type", group_cols="patient", min_cells=20)
        dyn.to_csv(out / "clones.csv", index=False)
    else:
        raise ValueError(step)


# ------------------------------------------------------------------ scanpy (+ pydeseq2)
def run_scanpy(step, data, out, T):
    import anndata as ad
    import scanpy as sc
    sc.settings.n_jobs = BENCH_THREADS
    common = data / "common"
    truth = json.loads((data / "truth.json").read_text())
    if step in ("de_deseq2",):
        cnt, cd = _pb_common(common)
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
        cd = cd.copy()
        for c in ("patient", "timepoint"):
            cd[c] = cd[c].astype("category")
        with T:
            inf = DefaultInference(n_cpus=BENCH_THREADS)
            dds = DeseqDataSet(counts=cnt.T.round().astype(int), metadata=cd, design="~ patient + timepoint",
                               refit_cooks=True, inference=inf, quiet=True)
            dds.deseq2()
            ds = DeseqStats(dds, contrast=["timepoint", "post", "pre"], alpha=0.1, inference=inf, quiet=True)
            ds.summary()
        r = ds.results_df
        pd.DataFrame({"gene": r.index, "log2FoldChange": r.log2FoldChange, "pvalue": r.pvalue,
                      "padj": r.padj}).to_csv(out / "de_deseq2.csv", index=False)
        return
    if step in ("de_voom", "clones", "score_module", "score_ucell"):
        raise NotImplementedError(f"no scanpy equivalent for {step}")
    if step in ("neighbors", "cluster"):
        E = np.load(common / "pca.npy").astype(np.float32)
        a = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(E.shape[0])]))
        a.obsm["X_pca"] = E
        with T:
            sc.pp.neighbors(a, n_neighbors=K_NN, use_rep="X_pca", random_state=SCANPY_NN_SEED)
            if step == "cluster":
                sc.tl.leiden(a, resolution=RES, flavor="igraph", n_iterations=2, directed=False)
        if step == "neighbors":
            D = a.obsp["distances"].tocsr()
            np.save(out / "neighbors.npy", _knn_from_csr(D, K_NN - 1))
        else:
            np.save(out / "cluster.npy", a.obs["leiden"].astype(int).to_numpy(np.int32))
        return
    a = sc.read_h5ad(data / "data.h5ad")
    # arsc.write_h5ad stores int32 indices with an int64 indptr; scipy's in-place kernels (used by
    # rank_genes_groups) reject mixed index dtypes, so give both the same dtype (untimed).
    idt = np.int32 if a.X.nnz < 2 ** 31 else np.int64
    a.X = sp.csr_matrix((a.X.data, a.X.indices.astype(idt, copy=False), a.X.indptr.astype(idt, copy=False)),
                        shape=a.X.shape)
    _touch(a.X)
    if step == "normalize":
        with T:
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
        sp.save_npz(out / "normalize.npz", sp.csr_matrix(a.X[:N_KEEP]))
    elif step == "hvg":
        with T:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=N_HVG)
        a.var[["variances_norm", "highly_variable"]].rename(columns={"variances_norm": "variance_standardized"}).to_csv(out / "hvg.csv")
    elif step == "pca":
        hv = _hvg(common, a.var_names)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a = a[:, hv].copy()
        with T:
            sc.pp.scale(a, max_value=10)
            sc.tl.pca(a, n_comps=N_PC, svd_solver="arpack")
        np.save(out / "pca.npy", a.obsm["X_pca"].astype(np.float32))
    elif step == "rank_genes":
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.obs["cell_type"] = a.obs["cell_type"].astype(str).astype("category")
        with T:
            sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", n_genes=a.n_vars, use_raw=False)
        r = sc.get.rank_genes_groups_df(a, None)
        r.rename(columns={"names": "feature", "pvals": "pval", "scores": "statistic"})[
            ["group", "feature", "statistic", "pval"]].to_parquet(out / "rank_genes.parquet")
    elif step == "pseudobulk":
        with T:
            g = sc.get.aggregate(a, by=["sample", "cell_type"], func="sum")
        lab = [f"{s}_{c}" for s, c in zip(g.obs["sample"], g.obs["cell_type"])]
        _save_pb(np.asarray(g.layers["sum"]), lab, out / "pseudobulk.npz")
    elif step == "score_genes":
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        res = {}
        with T:
            for k, v in truth["signatures"].items():
                sc.tl.score_genes(a, v, score_name=k, random_state=0, use_raw=False)
        pd.DataFrame({k: a.obs[k].to_numpy() for k in truth["signatures"]}).to_parquet(out / "score_genes.parquet")
        del res
    else:
        raise ValueError(step)


# ------------------------------------------------------------------ helpers
def _touch(X):
    """Fault the matrix pages in before timing (equal footing: scanpy's h5ad read is already in RAM)."""
    if sp.issparse(X):
        float(np.asarray(X.data).sum()); int(np.asarray(X.indices)[-1]); int(np.asarray(X.indptr)[-1])


def _hvg(common, names):
    hv = (common / "hvg.txt").read_text().split()
    idx = pd.Index(names).get_indexer(hv)
    return np.sort(idx)


def _pb_common(common):
    cnt = pd.read_parquet(common / "pb_T0_counts.parquet")
    cd = pd.read_csv(common / "pb_T0_coldata.csv", index_col=0)
    cd["timepoint"] = pd.Categorical(cd["timepoint"], categories=["pre", "post"])
    return cnt, cd


def _save_pb(M, labels, path):
    o = np.argsort(np.asarray(labels))
    np.savez_compressed(path, X=np.asarray(M)[o], labels=np.asarray(labels)[o])


def _knn_from_csr(D, k):
    n = D.shape[0]
    out = np.full((n, k + 1), -1, np.int32)
    out[:, 0] = np.arange(n)
    for i in range(n):
        s, e = D.indptr[i], D.indptr[i + 1]
        cols, d = D.indices[s:e], D.data[s:e]
        o = np.argsort(d, kind="stable")[:k]
        out[i, 1:1 + len(o)] = cols[o]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", default="", help="small dataset dir: run the step on it first, untimed, so JIT "
                                               "compilation (numba in arsc, pynndescent/umap in scanpy) is excluded")
    a = ap.parse_args()
    data = Path(a.data)
    od = data / "out" / (a.tool + os.environ.get("ARSC_BENCH_OUT_TAG", ""))
    od.mkdir(parents=True, exist_ok=True)
    T = Timer()
    res = {"tool": a.tool, "step": a.step, "ok": False}
    runner = run_arsc if a.tool == "arsc" else run_scanpy
    try:
        if a.warm:
            wd = Path(a.warm)
            wo = wd / "out" / f"warm_{a.tool}"
            wo.mkdir(parents=True, exist_ok=True)
            tw = time.time()
            runner(a.step, wd, wo, Timer())
            res["warmup_seconds"] = time.time() - tw
        runner(a.step, data, od, T)
        res.update(ok=True)
    except NotImplementedError as e:
        res.update(skipped=str(e))
    except Exception as e:  # noqa: BLE001
        res.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    res.update(t_start=T.t0, t_end=T.t1, seconds=(T.t1 - T.t0) if T.t1 else None)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
