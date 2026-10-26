"""One benchmark step for one Python tool (``arsc`` or ``scanpy``), run in a fresh process by paired_bench.py
(data in memory) or run_suite.py (data on disk).

    python suite_py.py --tool arsc --step pca --source toy --n 20000 --warm toy     # in memory, result on stdout
    python suite_py.py --tool arsc --step pca --data <bench_dir>/n10000 --out <file.json>

Only the analysis call is inside the timed region; loading the data and computing inputs the step does
not own (e.g. log-normalized data for ``markers``) happen before it. The result JSON carries the epoch
timestamps of the timed region so the parent can take peak RSS inside it. In memory mode nothing is written:
the neighbor lists of 1,000 cells and cluster labels of up to 20,000 cells come back in the result
(``payload``) for the agreement check. With ``--data`` outputs go to ``<data>/out/<tool>/<step>.*``.
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
N_KEEP = 500  # cells whose normalized values are saved for the agreement check


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
def run_arsc(step, D, out, T):
    import arsc
    obs, genes, truth = D.obs, D.genes, D.truth
    if step == "normalize":
        X = D.X
        X.data[:1].sum(); _touch(X)
        with T:
            L = arsc.pp.log_normalize(X)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            sp.save_npz(out / "normalize.npz", sp.csr_matrix(L[:N_KEEP]))
    elif step == "hvg":
        X = D.X; _touch(X)
        with T:
            h = arsc.pp.hvg_vst(X, nfeatures=N_HVG, features=genes)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            h[["variance_standardized", "highly_variable"]].to_csv(out / "hvg.csv")
    elif step == "pca":
        hv = D.hvg_index(genes)
        L = arsc.pp.log_normalize(D.X)[:, hv]
        with T:
            Z, _, _ = arsc.pp.scale(L)
            pc = arsc.pp.pca(Z, n_comps=N_PC)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            np.save(out / "pca.npy", pc.embedding.astype(np.float32))
    elif step == "neighbors":
        E = np.asarray(D.pca).astype(np.float64)
        with T:
            nn = arsc.find_neighbors(E, k=K_NN)
        PAYLOAD["nn"] = nn["idx"][eval_cells(E.shape[0])].tolist()
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            np.save(out / "neighbors.npy", nn["idx"].astype(np.int32))
    elif step == "cluster":
        E = np.asarray(D.pca).astype(np.float64)
        with T:
            nn = arsc.find_neighbors(E, k=K_NN)
            lab = arsc.cluster(nn["snn"], method="leiden", resolution=RES)  # arsc's own engine (default since 0.3.0)
        PAYLOAD["labels"] = np.asarray(lab)[label_cells(E.shape[0])].astype(int).tolist()
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            np.save(out / "cluster.npy", np.asarray(lab, np.int32))
    elif step == "markers":
        L = arsc.pp.log_normalize(D.X)
        g = obs["cell_type"].astype(str).to_numpy()
        with T:
            r = arsc.wilcoxauc(L, g, features=genes)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            r[["feature", "group", "auc", "pval", "statistic"]].to_parquet(out / "markers.parquet")
    elif step == "rank_genes":
        # scanpy-parity input: normalize_total + log1p in float32, as scanpy computes on float32 counts
        L = arsc.normalize_total_log1p(D.X, dtype=np.float32)
        g = obs["cell_type"].astype(str).to_numpy()
        with T:
            r = arsc.rank_genes_groups(L, g, genes)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            r.rename(columns={"names": "feature", "pvals": "pval", "scores": "statistic"})[
                ["group", "feature", "statistic", "pval"]].to_parquet(out / "rank_genes.parquet")
    elif step == "pseudobulk":
        X = D.X; _touch(X)
        with T:
            pb = arsc.pseudobulk(X, obs, ["sample", "cell_type"], features=genes)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            _save_pb(pb.X, list(pb.groups.index), out / "pseudobulk.npz")
    elif step == "de_deseq2":
        cnt, cd = _pb(D)
        with T:
            r = arsc.de.deseq2(cnt, cd, "~ patient + timepoint", ("timepoint", "post", "pre"), min_count=0,
                               min_samples=0, n_cpus=BENCH_THREADS)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            r[["gene", "log2FoldChange", "pvalue", "padj"]].to_csv(out / "de_deseq2.csv", index=False)
    elif step == "de_voom":
        cnt, cd = _pb(D)
        with T:
            r = arsc.de.voom(cnt, cd, "~ timepoint", coef=("timepoint", "post"), block="patient", filter=False)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            r[["gene", "logFC", "t", "P.Value", "adj.P.Val"]].to_csv(out / "de_voom.csv", index=False)
    elif step == "score_module":
        L = arsc.pp.log_normalize(D.X)
        with T:
            s = arsc.add_module_score(L, genes, truth["signatures"], seed=1)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            s.to_parquet(out / "score_module.parquet")
    elif step == "score_genes":
        L = arsc.normalize_total_log1p(D.X, dtype=np.float32)
        with T:
            s = pd.DataFrame({k: arsc.score_genes(L, genes, v, random_state=0)
                              for k, v in truth["signatures"].items()})
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            s.to_parquet(out / "score_genes.parquet")
    elif step == "score_ucell":
        X = D.X; _touch(X)
        with T:
            s = arsc.score_ucell(X, genes, truth["signatures"])
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            s.to_parquet(out / "score_ucell.parquet")
    elif step == "clones":
        meta = obs[["patient", "timepoint", "cell_type", "clone_id"]].copy()
        with T:
            ct = arsc.clones.clone_table(meta, "clone_id", ["patient", "timepoint"])
            dyn = arsc.clones.clone_dynamics(ct, "pre", "post", patient="patient", time="timepoint")
            cl, pw = arsc.clones.startrac(meta, state_col="cell_type", group_cols="patient", min_cells=20)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            dyn.to_csv(out / "clones.csv", index=False)
    else:
        raise ValueError(step)


# ------------------------------------------------------------------ scanpy (+ pydeseq2)
def run_scanpy(step, D, out, T):
    import anndata as ad
    import scanpy as sc
    sc.settings.n_jobs = BENCH_THREADS
    truth = D.truth
    if step in ("de_deseq2",):
        cnt, cd = _pb(D)
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
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            pd.DataFrame({"gene": r.index, "log2FoldChange": r.log2FoldChange, "pvalue": r.pvalue,
                          "padj": r.padj}).to_csv(out / "de_deseq2.csv", index=False)
        return
    if step in ("de_voom", "clones", "score_module", "score_ucell"):
        raise NotImplementedError(f"no scanpy equivalent for {step}")
    if step in ("neighbors", "cluster"):
        E = np.asarray(D.pca).astype(np.float32)
        a = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(E.shape[0])]))
        a.obsm["X_pca"] = E
        with T:
            sc.pp.neighbors(a, n_neighbors=K_NN, use_rep="X_pca", random_state=SCANPY_NN_SEED)
            if step == "cluster":
                sc.tl.leiden(a, resolution=RES, flavor="igraph", n_iterations=2, directed=False)
        if step == "neighbors":
            Dm = a.obsp["distances"].tocsr()
            ev = eval_cells(E.shape[0])
            PAYLOAD["nn"] = _knn_from_csr(Dm[ev], K_NN - 1, rows=ev).tolist()
            if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
                np.save(out / "neighbors.npy", _knn_from_csr(Dm, K_NN - 1))
        else:
            PAYLOAD["labels"] = a.obs["leiden"].astype(int).to_numpy()[label_cells(E.shape[0])].tolist()
            if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
                np.save(out / "cluster.npy", a.obs["leiden"].astype(int).to_numpy(np.int32))
        return
    a = D.anndata()   # int32 (or int64) indices and indptr alike: scipy's in-place kernels reject mixed dtypes
    _touch(a.X)
    if step == "normalize":
        with T:
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            sp.save_npz(out / "normalize.npz", sp.csr_matrix(a.X[:N_KEEP]))
    elif step == "hvg":
        with T:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=N_HVG)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            a.var[["variances_norm", "highly_variable"]].rename(columns={"variances_norm": "variance_standardized"}).to_csv(out / "hvg.csv")
    elif step == "pca":
        hv = D.hvg_index(a.var_names)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a = a[:, hv].copy()
        with T:
            sc.pp.scale(a, max_value=10)
            sc.tl.pca(a, n_comps=N_PC, svd_solver="arpack")
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            np.save(out / "pca.npy", a.obsm["X_pca"].astype(np.float32))
    elif step == "rank_genes":
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.obs["cell_type"] = a.obs["cell_type"].astype(str).astype("category")
        with T:
            sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", n_genes=a.n_vars, use_raw=False)
        r = sc.get.rank_genes_groups_df(a, None)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            r.rename(columns={"names": "feature", "pvals": "pval", "scores": "statistic"})[
                ["group", "feature", "statistic", "pval"]].to_parquet(out / "rank_genes.parquet")
    elif step == "pseudobulk":
        with T:
            g = sc.get.aggregate(a, by=["sample", "cell_type"], func="sum")
        lab = [f"{s}_{c}" for s, c in zip(g.obs["sample"], g.obs["cell_type"])]
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            _save_pb(np.asarray(g.layers["sum"]), lab, out / "pseudobulk.npz")
    elif step == "score_genes":
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        res = {}
        with T:
            for k, v in truth["signatures"].items():
                sc.tl.score_genes(a, v, score_name=k, random_state=0, use_raw=False)
        if out is not None:   # file outputs only in the on-disk suite (run_suite.py)
            pd.DataFrame({k: a.obs[k].to_numpy() for k in truth["signatures"]}).to_parquet(out / "score_genes.parquet")
        del res
    else:
        raise ValueError(step)


# ------------------------------------------------------------------ helpers
def _touch(X):
    """Fault the matrix pages in before timing (equal footing: scanpy's h5ad read is already in RAM)."""
    if sp.issparse(X):
        float(np.asarray(X.data).sum()); int(np.asarray(X.indices)[-1]); int(np.asarray(X.indptr)[-1])


PAYLOAD = {}   # small agreement data returned with the result (in-memory mode writes no output files)
N_EVAL, N_LABEL = 1000, 20_000


def eval_cells(n):
    """The cells whose neighbor lists are returned for the recall / overlap check (fixed seed)."""
    return np.sort(np.random.default_rng(0).choice(n, min(n, N_EVAL), replace=False))


def label_cells(n):
    """The cells whose cluster labels are returned for the agreement check (all up to 20,000; else a fixed sample)."""
    return np.arange(n) if n <= N_LABEL else np.sort(np.random.default_rng(1).choice(n, N_LABEL, replace=False))


def _pb(D):
    cnt, cd = D.pb
    cd = cd.copy()
    cd["timepoint"] = pd.Categorical(cd["timepoint"], categories=["pre", "post"])
    return cnt, cd


class DiskBench:
    """The on-disk suite's dataset directory (run_suite.py --bench-dir), with Bench's interface."""

    def __init__(self, data):
        import arsc
        self.dir = Path(data)
        st = arsc.open(self.dir / "store")
        self.obs, self.genes = st.obs, st.features("RNA")
        self.X = st.X("RNA", "counts")
        self.truth = json.loads((self.dir / "truth.json").read_text())
        c = self.dir / "common"
        self.hvg = (c / "hvg.txt").read_text().split() if (c / "hvg.txt").exists() else None
        self._c = c

    def hvg_index(self, names):
        return np.sort(pd.Index(names).get_indexer(self.hvg))

    @property
    def pca(self):
        return np.load(self._c / "pca.npy")

    @property
    def pb(self):
        return (pd.read_parquet(self._c / "pb_T0_counts.parquet"),
                pd.read_csv(self._c / "pb_T0_coldata.csv", index_col=0))

    def anndata(self):
        import scanpy as sc
        a = sc.read_h5ad(self.dir / "data.h5ad")
        idt = np.int32 if a.X.nnz < 2 ** 31 else np.int64
        a.X = sp.csr_matrix((a.X.data, a.X.indices.astype(idt, copy=False), a.X.indptr.astype(idt, copy=False)),
                            shape=a.X.shape)
        return a


def _save_pb(M, labels, path):
    o = np.argsort(np.asarray(labels))
    np.savez_compressed(path, X=np.asarray(M)[o], labels=np.asarray(labels)[o])


def _knn_from_csr(D, k, rows=None):
    n = D.shape[0]
    out = np.full((n, k + 1), -1, np.int32)
    out[:, 0] = np.arange(n) if rows is None else rows
    for i in range(n):
        s, e = D.indptr[i], D.indptr[i + 1]
        cols, d = D.indices[s:e], D.data[s:e]
        o = np.argsort(d, kind="stable")[:k]
        out[i, 1:1 + len(o)] = cols[o]
    return out


def run_both(a, steps):
    """The fast paired mode (speed gate): one process holds the data, warms every step of both tools untimed on
    the small toy data, then times each step ``--repeats`` times per tool, interleaved and alternating which tool
    goes first. One result line per (step, repeat, tool)."""
    from benchdata import Bench, WARM_TOY_N
    D = Bench(a.source, a.n or None, a.seed, pca_shm=a.pca_shm)
    warm = Bench("toy", int(os.environ.get("ARSC_BENCH_WARM_N", WARM_TOY_N)), a.seed + 1)
    runners = {"arsc": run_arsc, "scanpy": run_scanpy}
    tw = time.time()
    for step in steps:
        for tool, f in runners.items():
            try:
                f(step, warm, None, Timer())
            except Exception:  # noqa: BLE001 - the timed run reports the error
                pass
            PAYLOAD.clear()
    warm_s = time.time() - tw
    for step in steps:
        for rep in range(a.repeats):
            order = ("arsc", "scanpy") if rep % 2 == 0 else ("scanpy", "arsc")
            for pos, tool in enumerate(order):
                T = Timer()
                res = {"tool": tool, "step": step, "rep": rep, "order_pos": pos, "ok": False,
                       "warmup_seconds": warm_s}
                try:
                    runners[tool](step, D, None, T)
                    res.update(ok=True)
                except Exception as e:  # noqa: BLE001
                    res.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
                res.update(t_start=T.t0, t_end=T.t1, seconds=(T.t1 - T.t0) if T.t1 else None)
                if PAYLOAD:
                    res["payload"] = dict(PAYLOAD)
                    PAYLOAD.clear()
                print("@@RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--step", required=True, help="a step, or several separated by commas (one process; in-memory "
                                                   "mode only)")
    ap.add_argument("--data", default="", help="on-disk dataset dir (run_suite.py); default: in memory (--source)")
    ap.add_argument("--source", default="auto", help="in-memory data: 'auto', 'toy' or 'store:<dir>' (benchdata.py)")
    ap.add_argument("--n", type=int, default=0, help="cells (in-memory source)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca-shm", default="", help="the parent's PCA in shared memory (<name>:<rows>:<cols>)")
    ap.add_argument("--out", default="", help="result JSON file (on-disk suite); default: one stdout line "
                                             "starting with @@RESULT")
    ap.add_argument("--warm", default="", help="run the step first, untimed, on a 12,000-cell toy dataset (in "
                                               "memory; 'toy') or a dataset dir, so JIT compilation is excluded")
    ap.add_argument("--repeats", type=int, default=1, help="--tool both: timed repeats per step and tool")
    a = ap.parse_args()
    steps = [x for x in a.step.split(",") if x]
    if a.tool == "both":
        return run_both(a, steps)
    runner = run_arsc if a.tool == "arsc" else run_scanpy
    D = od = warm = None
    for step in steps:      # several steps: one process, each step warmed untimed right before its timed run
        T = Timer()
        res = {"tool": a.tool, "step": step, "ok": False}
        try:
            from benchdata import Bench, WARM_TOY_N
            if D is None:
                if a.data:
                    D = DiskBench(a.data)
                    od = Path(a.data) / "out" / (a.tool + os.environ.get("ARSC_BENCH_OUT_TAG", ""))
                    od.mkdir(parents=True, exist_ok=True)
                else:
                    D = Bench(a.source, a.n or None, a.seed, pca_shm=a.pca_shm)
            if a.warm:
                tw = time.time()
                if a.warm == "toy":
                    warm = warm or Bench("toy", int(os.environ.get("ARSC_BENCH_WARM_N", WARM_TOY_N)), a.seed + 1)
                    runner(step, warm, None, Timer())
                else:
                    wo = Path(a.warm) / "out" / f"warm_{a.tool}"
                    wo.mkdir(parents=True, exist_ok=True)
                    runner(step, DiskBench(a.warm), wo, Timer())
                PAYLOAD.clear()
                res["warmup_seconds"] = time.time() - tw
            runner(step, D, od, T)
            res.update(ok=True)
        except NotImplementedError as e:
            res.update(skipped=str(e))
        except Exception as e:  # noqa: BLE001
            res.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        res.update(t_start=T.t0, t_end=T.t1, seconds=(T.t1 - T.t0) if T.t1 else None)
        if PAYLOAD:
            res["payload"] = dict(PAYLOAD)
            PAYLOAD.clear()
        if a.out:
            Path(a.out).write_text(json.dumps(res))
        else:
            print("@@RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
