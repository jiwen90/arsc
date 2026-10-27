"""arsc vs Seurat (R) timing on a real store: the Seurat workflow, step by step, on the same cells.

    ARSC_BENCH_STORE=<store> python benchmarks/seurat_timing.py --n 100000 [--repeats 3] [--out benchmarks/results/seurat]

Steps (arsc default -> Seurat call with the same parameters):

    log_normalize                         NormalizeData (LogNormalize, scale factor 1e4)
    hvg_vst(nfeatures=2000)               FindVariableFeatures(selection.method="vst", nfeatures=2000)
    scale + pca(n_comps=30)               ScaleData(features=HVGs) + RunPCA(npcs=30)
    find_neighbors(k=20)                  FindNeighbors(dims=1:30, k.param=20)            (SNN graph, prune 1/15)
    cluster (Louvain, resolution 0.6)     FindClusters(resolution=0.6, algorithm=1)
    find_all_markers (Wilcoxon)           FindAllMarkers(test.use="wilcox")                (36 cell types)
    pseudobulk(sample x cell type)        AggregateExpression(group.by=c("sample","cell_type"), slot="counts")
    add_module_score(seed=1)              AddModuleScore(seed=1)                          (3 sets of 30 genes)

Each tool runs its own chain on its own outputs (arsc's PCA feeds arsc's graph, Seurat's feeds Seurat's), as a user
would run it. Data: the deterministic subset of ``benchdata.Bench`` (seed 0), read memory-mapped and held in memory.
The R process reads the same rows straight from the store's .npy files (read-only); the row list, labels and gene
sets reach it on stdin and its results come back on stdout. Nothing is written to disk but the summary JSON in --out.

Warm runs only: each tool first runs every step once untimed on the first 12,000 cells of the subset (numba
compilation for arsc; first-call overhead for R); import / package load and the warm-up are reported separately.
arsc runs every step --repeats times (median, min, max); Seurat once (its steps take minutes).

The two sides can run apart: ``--skip-r`` times arsc only (under the machine-wide timing lock, all cores), and
``--r-only`` times Seurat only (one R thread, so it can run as an ordinary job beside other work). ``--r-only`` runs
arsc's chain once untimed for the agreement check, samples the machine's CPU use while R runs, and writes
``seurat_timing_n<n>_seurat.json``; ``render_readme.py`` joins it with the arsc file ``seurat_timing_n<n>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

STEPS = ["normalize", "hvg", "pca", "neighbors", "cluster", "markers", "pseudobulk", "score_module"]
LABELS = {
    "normalize": "Log-normalize (NormalizeData)",
    "hvg": "Highly variable genes, vst (FindVariableFeatures)",
    "pca": "Scale + PCA, 30 PCs (ScaleData + RunPCA)",
    "neighbors": "SNN graph, k = 20 (FindNeighbors)",
    "cluster": "Louvain, resolution 0.6 (FindClusters)",
    "markers": "Markers, Wilcoxon, 36 cell types (FindAllMarkers)",
    "pseudobulk": "Pseudobulk sums, sample x cell type (AggregateExpression)",
    "score_module": "Module score, 3 sets of 30 genes (AddModuleScore)",
}
WARM_N = 12_000     # arsc warm-up cells (numba compiles on first use)
WARM_N_R = 2_000    # Seurat warm-up cells (no compilation; first-call costs only)


def arsc_steps(X, obs, genes, sigs, timed: bool, repeats: int = 1):
    """The arsc chain; returns ({step: [seconds]}, outputs)."""
    import arsc
    t: dict[str, list[float]] = {s: [] for s in STEPS}
    out: dict = {}
    ct = obs["cell_type"].astype(str).to_numpy()

    def tm(step, fn):
        v = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            v = fn()
            t[step].append(time.perf_counter() - t0)
            print(f"[arsc] {'timed' if timed else 'warm-up'} {step}: {t[step][-1]:.2f} s", flush=True)
        return v

    L = tm("normalize", lambda: arsc.pp.log_normalize(X))
    h = tm("hvg", lambda: arsc.pp.hvg_vst(X, nfeatures=2000, features=genes))
    hv_names = list(h.sort_values("rank").index[:2000])
    hv = np.sort(pd.Index(genes).get_indexer(hv_names))

    def pca():
        Z, _, _ = arsc.pp.scale(L[:, hv])
        return arsc.pp.pca(Z, n_comps=30).embedding
    E = tm("pca", pca)
    nn = tm("neighbors", lambda: arsc.find_neighbors(E, k=20))
    lab = tm("cluster", lambda: arsc.cluster(nn["snn"]))
    mk = tm("markers", lambda: arsc.find_all_markers(L, ct, features=genes))
    pb = tm("pseudobulk", lambda: arsc.pseudobulk(X, obs, ["sample", "cell_type"], features=genes))
    sc = tm("score_module", lambda: arsc.add_module_score(L, genes, sigs, seed=1))
    if timed:
        out = {"hvg": hv_names, "cluster_seed0": np.asarray(lab), "cluster_seed1": np.asarray(arsc.cluster(nn["snn"], seed=1)),
               "markers": mk, "pb": pb, "score_module": sc}
    return t, out


def run_seurat(store, rows, obs, sigs, rscript, warm_n=WARM_N_R, markers_budget_s=0.0):
    spec = {"store": str(store), "rows": rows.tolist(), "cell_type": obs["cell_type"].astype(str).tolist(),
            "sample": obs["sample"].astype(str).tolist(), "signatures": sigs, "warm_n": warm_n, "markers_budget_s": markers_budget_s,
            "resolution": 0.6, "k": 20, "npcs": 30, "nhvg": 2000}
    t0 = time.perf_counter()
    p = subprocess.run([rscript, str(HERE / "r_seurat_timing.R")], input=json.dumps(spec), stdout=subprocess.PIPE,
                       text=True, creationflags=0x08000000 if os.name == "nt" else 0)
    wall = time.perf_counter() - t0
    if p.returncode != 0 or "@@RESULT@@" not in p.stdout:
        sys.stderr.write(p.stdout[-4000:])
        raise SystemExit(f"Rscript failed (exit {p.returncode})")
    res = json.loads(p.stdout.split("@@RESULT@@", 1)[1].strip())
    res["process_wall_s"] = wall
    return res


class LoadSampler:
    """The machine's CPU use (percent of all logical CPUs) every 5 s in a background thread: mean, max, samples."""

    def __init__(self, every: float = 5.0):
        import threading
        self.every, self.v, self.stop_ev = every, [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        import psutil
        psutil.cpu_percent(None)
        while not self.stop_ev.wait(self.every):
            self.v.append(psutil.cpu_percent(None))

    def start(self):
        self.th.start()
        return self

    def stop(self) -> dict:
        self.stop_ev.set(); self.th.join()
        v = self.v or [float("nan")]
        return {"cpu_pct_mean": float(np.mean(v)), "cpu_pct_max": float(np.max(v)), "n_samples": len(self.v),
                "every_s": self.every}


def agreement(a: dict, r: dict, cell_type: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score as ari
    ro = r["out"]
    s0, s1 = np.asarray(ro["cluster_seed0"]), np.asarray(ro["cluster_seed1"])
    a0, a1 = a["cluster_seed0"], a["cluster_seed1"]
    ag = {
        "hvg_shared": len(set(a["hvg"]) & set(ro["hvg"])),
        "ari_arsc_seurat": ari(a0, s0), "ari_seurat_seed0_seed1": ari(s0, s1), "ari_arsc_seed0_seed1": ari(a0, a1),
        "ari_arsc_celltype": ari(a0, cell_type), "ari_seurat_celltype": ari(s0, cell_type),
        "n_clusters_arsc": int(len(np.unique(a0))), "n_clusters_seurat": int(len(np.unique(s0))),
        "n_clusters_seurat_seed1": int(len(np.unique(s1))),
    }
    # markers: same (gene, cell type) rows; top-10 positive markers per cell type in each tool's own order
    ma = a["markers"].copy(); ms = pd.DataFrame(ro["markers"])
    for m in (ma, ms):
        m["cluster"] = m["cluster"].astype(str)
    if ro.get("markers_partial"):   # Seurat ran only some cell types: compare those
        ma = ma[ma["cluster"].isin(set(ro["markers_partial"]))]
        ag["markers_cell_types_compared"] = len(ro["markers_partial"])
    ka = set(zip(ma["cluster"], ma["gene"])); ks = set(zip(ms["cluster"], ms["gene"]))
    ag["markers_rows_arsc"], ag["markers_rows_seurat"] = len(ma), len(ms)
    ag["markers_rows_jaccard"] = len(ka & ks) / max(1, len(ka | ks))
    j = ma.merge(ms, on=["cluster", "gene"], suffixes=("_a", "_s"))
    ag["markers_log2fc_maxdiff"] = float(np.max(np.abs(j["avg_log2FC_a"] - j["avg_log2FC_s"]))) if len(j) else None
    pa, ps = j["p_val_a"].to_numpy(float), j["p_val_s"].to_numpy(float)
    ok = (pa > 0) & (ps > 0)
    ag["markers_p_max_rel_diff"] = float(np.max(np.abs(pa[ok] - ps[ok]) / ps[ok])) if ok.any() else None
    top = []
    for c in sorted(set(ms["cluster"]) | set(ma["cluster"])):
        ta = ma[(ma["cluster"] == c) & (ma["avg_log2FC"] > 0)]["gene"].head(10)
        ts = ms[(ms["cluster"] == c) & (ms["avg_log2FC"] > 0)]["gene"].head(10)
        if len(ts):
            top.append(len(set(ta) & set(ts)) / len(ts))
    ag["markers_top10_overlap_mean"] = float(np.mean(top)) if top else None
    ag["markers_top10_overlap_min"] = float(np.min(top)) if top else None
    # pseudobulk sums and module scores
    pb = a["pb"]
    cs_a = pd.Series(np.asarray(pb.X).sum(axis=1), index=[str(i) for i in pb.groups.index])
    ag["pseudobulk_total_arsc"], ag["pseudobulk_total_seurat"] = float(cs_a.sum()), float(ro["pb_total"])
    ag["pseudobulk_groups_arsc"], ag["pseudobulk_groups_seurat"] = int(len(cs_a)), int(ro["pb_dim"][1])
    sm = pd.DataFrame(ro["score_module_first"])
    sa = a["score_module"].iloc[: len(sm)][sm.columns].to_numpy(float)
    ag["score_module_maxdiff_first_cells"] = float(np.max(np.abs(sa - sm.to_numpy(float))))
    ag["score_module_cells_compared"] = int(len(sm))
    return ag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rscript", default=os.environ.get("ARSC_RSCRIPT", "Rscript"))
    ap.add_argument("--out", default=str(HERE / "results" / "seurat"))
    ap.add_argument("--skip-r", action="store_true", help="arsc only (timed; Seurat runs apart with --r-only)")
    ap.add_argument("--r-only", action="store_true", help="Seurat only (timed); arsc runs once untimed for the "
                    "agreement check; writes seurat_timing_n<n>_seurat.json")
    ap.add_argument("--warm-n", type=int, default=WARM_N, help="cells in arsc's untimed warm-up run")
    ap.add_argument("--warm-n-r", type=int, default=WARM_N_R, help="cells in Seurat's untimed warm-up run")
    ap.add_argument("--markers-budget-s", type=float, default=0.0,
                    help="> 0: Seurat markers per cell type (seeded random order) until this many seconds, extrapolated")
    a = ap.parse_args()

    t_imp = time.perf_counter()
    import arsc
    import numba
    t_imp = time.perf_counter() - t_imp
    from benchdata import Bench
    store = os.environ.get("ARSC_BENCH_STORE", "")
    if not store or not Path(store).is_dir():
        raise SystemExit("set ARSC_BENCH_STORE to an arsc store directory")
    B = Bench(f"store:{store}", a.n, a.seed)
    t_read = time.perf_counter()
    X, obs, genes, sigs = B.X, B.obs, np.asarray(B.genes), B.truth["signatures"]
    t_read = time.perf_counter() - t_read
    n_store = len(arsc.open(store).obs)
    rows = (np.sort(np.random.default_rng(a.seed).choice(n_store, a.n, replace=False)) if a.n < n_store
            else np.arange(n_store))   # the rows Bench took (same rule as benchdata.Bench._from_store)

    t_w = time.perf_counter()
    w = min(a.warm_n, X.shape[0])
    warm_t, _ = ({}, None) if a.r_only else arsc_steps(X[:w], obs.iloc[:w].copy(), genes, sigs, timed=False)
    t_w = time.perf_counter() - t_w
    # --r-only: one untimed arsc run, only for its outputs (the agreement check)
    at, ao = arsc_steps(X, obs, genes, sigs, timed=True, repeats=1 if a.r_only else a.repeats)

    res = {"n": int(X.shape[0]), "n_store": int(n_store), "seed": a.seed, "repeats_arsc": a.repeats,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "threads_arsc": int(numba.get_num_threads()), "AR_THREADS": os.environ.get("AR_THREADS"),
           "arsc": {"version": arsc.__version__, "import_s": t_imp, "read_subset_s": t_read, "warmup_s": t_w,
                    "warm_times": {k: v[0] for k, v in warm_t.items()},
                    "times": at},
           "platform": {"python": platform.python_version(), "machine": platform.processor(),
                        "os": platform.platform()}}
    if a.r_only:
        res = {"n": int(X.shape[0]), "n_store": int(n_store), "seed": a.seed, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "AR_THREADS": os.environ.get("AR_THREADS"), "platform": res["platform"],
               "arsc_reference": {"version": arsc.__version__, "threads": int(numba.get_num_threads()),
                                  "note": "one untimed run, for the agreement check only"}}
    if not a.skip_r:
        load = LoadSampler().start() if a.r_only else None
        r = run_seurat(store, rows, obs, sigs, a.rscript, a.warm_n_r, a.markers_budget_s)
        if load is not None:
            res["machine_load_during_r"] = load.stop()
        res["seurat"] = {k: v for k, v in r.items() if k != "out"}
        part = r["out"].get("markers_partial")
        if part:
            secs = [g["seconds"] for g in part.values()]
            tot = int(r["out"]["markers_groups_total"])
            res["seurat"]["markers_partial"] = part
            res["seurat"]["markers_groups_total"] = tot
            res["seurat"]["markers_loop_sorted"] = True   # the R loop re-sorts as FindAllMarkers does
            if len(secs) == tot:   # the budget did not bind: every cell type ran, so the sum is a measurement
                res["seurat"]["times"]["markers"] = float(sum(secs))
                res["seurat"]["markers_summed"] = tot
            else:
                res["seurat"]["markers_estimate_s"] = float(np.mean(secs) * tot)
                res["seurat"]["markers_estimate_basis"] = (f"{len(secs)} of {tot} cell types timed ({sum(secs):.0f} s, "
                                                          f"{min(secs):.0f}-{max(secs):.0f} s each); mean x {tot}")
        res["agreement"] = agreement(ao, r, obs["cell_type"].astype(str).to_numpy())
    if a.r_only:
        od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
        f = od / f"seurat_timing_n{X.shape[0]}_seurat.json"
        f.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res["seurat"]["times"], default=float))
        print(json.dumps(res["agreement"], indent=1, default=float))
        print(f"wrote {f}")
        return
    rows_out = []
    for s in STEPS:
        v = np.asarray(at[s])
        row = {"step": s, "label": LABELS[s], "arsc_median_s": float(np.median(v)), "arsc_min_s": float(v.min()),
               "arsc_max_s": float(v.max())}
        if "seurat" in res:
            v = res["seurat"]["times"].get(s)
            row["seurat_s"] = float(v) if v is not None else float("nan")
            if s == "markers" and v is None and res["seurat"].get("markers_estimate_s"):
                row["seurat_estimate_s"] = res["seurat"]["markers_estimate_s"]
                row["seurat_estimate_basis"] = res["seurat"]["markers_estimate_basis"]
            row["speedup"] = row["seurat_s"] / row["arsc_median_s"]
        rows_out.append(row)
    res["summary"] = rows_out
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    f = od / f"seurat_timing_n{X.shape[0]}.json"
    f.write_text(json.dumps(res, indent=1, default=float))
    print(pd.DataFrame(rows_out).drop(columns="label").to_string(index=False))
    if "agreement" in res:
        print(json.dumps(res["agreement"], indent=1, default=float))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
