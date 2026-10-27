"""R Clonotrace vs arsc.clonotrace: per-step speed and agreement on the same inputs, at three sizes of the public demo.

    python benchmarks/clonotrace_bench.py r     --size S [--workers 1] [--n-boot 48] [--steps a,b]
    python benchmarks/clonotrace_bench.py arsc  --size S --threads 1,16
    python benchmarks/clonotrace_bench.py render            # tables for CLONOTRACE_PARITY.md (between markers)

Sizes: S = the 2,032-cell subset of r_reference_clonotrace.R, M = 200 clones + 4,000 background cells, L = the full
demo (34,782 cells, 802 clones). ``r`` runs benchmarks/r_bench_clonotrace.R, which times every step of the R code
and writes its outputs to $ARSC_REFS/clonotrace_bench/<size>; the R process tree's resident memory is sampled every
20 ms and attributed to steps by their start/end times. ``arsc`` runs each step in a fresh process per thread count
(thread variables set before import), on the R outputs of the previous step as input (the same inputs as R), after an
untimed warm-up of every step on size S (numba compilation excluded); a step under 20 s is timed 3 times (median),
a longer step once. Peak memory = the largest resident-set increase over the step's starting value. Each step's
output is compared with R's. Results: benchmarks/results/clonotrace/<size>_r_w<workers>.json and
<size>_arsc_t<threads>.json. Environment: ARSC_REFS, CLONOTRACE_SRC (pinned Clonotrace checkout),
CLONOTRACE_SRC_MAIN (optional newer checkout), ARSC_RLIB, RSCRIPT.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results" / "clonotrace"
REFS = Path(os.environ.get("ARSC_REFS") or os.environ.get("ARSC_SCRATCH") or ROOT / "benchmarks" / ".refs")
BENCH = REFS / "clonotrace_bench"
THREAD_VARS = ("AR_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
STEPS = ["knn", "transition", "spread", "bootstrap", "sparsify", "snn", "nn", "ot", "revised", "leiden", "cmdscale",
         "isomds", "dpt", "profiles", "enrichment", "gam_f", "gam_lrt", "coembed"]
PIPELINE = ["knn", "transition", "bootstrap", "sparsify", "snn", "nn", "leiden", "isomds", "dpt", "profiles"]
LABEL = {"knn": "kNN kernel graph (k = 30)", "transition": "transition matrix", "spread": "label spreading (reference)",
         "bootstrap": "spreading + bootstrap deviance", "sparsify": "sparsify", "snn": "SNN cost graph (k = 10)",
         "nn": "NN clone distance, all pairs", "nn_main": "NN clone distance, Clonotrace main",
         "ot": "exact OT distance, 9 clones (36 pairs)", "revised": "revised distance (fixed clusterings)",
         "leiden": "clone Leiden", "cmdscale": "classical MDS", "isomds": "non-metric MDS (k = 6)",
         "dpt": "root + DPT", "profiles": "profiles + mass", "enrichment": "enrichment (200 permutations)",
         "gam_f": "spline fit, F test (200 genes)", "gam_lrt": "spline fit, LRT (200 genes)",
         "coembed": "co-embedding graph"}


class MemSampler:
    """Resident memory of a process tree sampled every ``dt`` seconds: [(time, bytes)]."""

    def __init__(self, pid: int, dt: float = 0.02, tree: bool = True):
        import psutil
        self.p = psutil.Process(pid)
        self.dt, self.tree, self.samples, self._stop = dt, tree, [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _rss(self):
        procs = [self.p] + (self.p.children(recursive=True) if self.tree else [])
        tot = 0
        for q in procs:
            try:
                tot += q.memory_info().rss
            except Exception:  # noqa: BLE001  (a worker exited between listing and reading)
                pass
        return tot

    def _run(self):
        import psutil
        self.cpu, last = [], 0.0
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), self._rss()))
            except Exception:  # noqa: BLE001
                break
            if time.time() - last >= 1.0:   # machine-wide CPU use, percent of all logical CPUs
                self.cpu.append(psutil.cpu_percent(None))
                last = time.time()
            time.sleep(self.dt)

    def start(self):
        self.t.start()
        return self

    def stop(self):
        self._stop.set()
        self.t.join()

    def peak_delta(self, t0: float, t1: float) -> float:
        before = [b for t, b in self.samples if t <= t0]
        base = before[-1] if before else (self.samples[0][1] if self.samples else 0)
        inside = [b for t, b in self.samples if t0 <= t <= t1]
        return max(0, max(inside, default=base) - base)


def machine() -> dict:
    import psutil
    vm = psutil.virtual_memory()
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(), "cpu": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False), "ram_gb": round(vm.total / 2 ** 30),
            "cpu_percent_1s_before": psutil.cpu_percent(interval=1.0), "python": sys.version.split()[0]}


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------- R side
def run_r(size: str, workers: int, n_boot: int, steps: str):
    d = BENCH / size
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("CLONOTRACE_SRC", str(REFS / "src" / "Clonotrace-f0c397a"))
    cmd = [os.environ.get("RSCRIPT", "Rscript"), str(HERE / "r_bench_clonotrace.R"), size, str(d), str(n_boot),
           str(workers), steps]
    st = machine()
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, creationflags=0x08000000 if os.name == "nt" else 0)
    ms = MemSampler(p.pid).start()
    rc = p.wait()
    ms.stop()
    if rc != 0:
        raise SystemExit(f"R failed with exit {rc}")
    info = json.loads((d / f"r_timing_w{workers}.json").read_text())
    for name, s in info["steps"].items():
        starts, ends = s["start"], s["end"]
        starts = starts if isinstance(starts, list) else [starts]
        ends = ends if isinstance(ends, list) else [ends]
        s["peak_mem_mb"] = round(max(ms.peak_delta(a, b) for a, b in zip(starts, ends)) / 2 ** 20, 1)
    info["process_peak_mb"] = round(max((b for _, b in ms.samples), default=0) / 2 ** 20, 1)
    info["machine_cpu_percent_mean"] = round(sum(ms.cpu) / max(1, len(ms.cpu)), 1)
    info["machine"] = st
    info["wall_seconds"] = round(time.time() - t0, 1)
    info.pop("session", None)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_steps-{steps.replace(',', '-')}" if steps else ""
    (OUT / f"{size}_r_w{workers}{tag}.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {OUT / f'{size}_r_w{workers}{tag}.json'}")


# ------------------------------------------------------------------------------------------------- arsc side
def _load(size: str):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    d = BENCH / size

    def rd(n, **kw):
        return pd.read_csv(d / n, **kw)

    def binm(n):
        dim = [int(x) for x in (d / f"{n}.dim").read_text().split()]
        return np.fromfile(d / n, dtype=np.float64).reshape(dim[1], dim[0]).T

    def trip(n, N):
        t = rd(n)
        return sp.csr_matrix((t.x.to_numpy(), (t.i.to_numpy(), t.j.to_numpy())), shape=(N, N))

    pca = rd("pca.csv").set_index("cell")
    meta = rd("meta.csv", keep_default_na=False, na_values=["NA"]).set_index("cell")
    X = pca.to_numpy()
    N = X.shape[0]
    levels = rd("clone_levels.csv").level.astype(str).tolist()
    qcells = rd("Q_cells.csv").cell.tolist()
    qclones = rd("Q_clones.csv").clone.astype(str).tolist()
    Q = pd.DataFrame(binm("Q.f64"), index=qcells, columns=qclones)
    pos = pca.index.get_indexer(Q.index)
    return dict(d=d, rd=rd, binm=binm, trip=trip, X=X, N=N, meta=meta, levels=levels, Q=Q, pos=pos, cells=pca.index)


def _steps(I):
    """{step: (fn, check)}: fn() runs arsc on the R inputs; check(result) -> agreement dict vs R."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from arsc import clonotrace as ct
    from arsc.clones import ari
    rd, binm, trip, X, N, meta, Q, pos = (I[k] for k in ("rd", "binm", "trip", "X", "N", "meta", "Q", "pos"))
    d = I["d"]
    K_r = trip("knn.csv", N)
    T_r = trip("transition.csv", N)
    codes, lv = ct.encode_clones(meta.clone, I["levels"])
    C = len(lv)
    F_r = binm("spread_raw.f64")
    emb = X[pos]
    nn_r = rd("dist_nn.csv") if (d / "dist_nn.csv").exists() else rd("dist_nn_main.csv")
    names = list(Q.columns)
    Dn = np.zeros((len(names), len(names)))
    Dn[nn_r.group1 - 1, nn_r.group2 - 1] = nn_r.dis
    Dn = Dn + Dn.T
    Dn = pd.DataFrame(Dn, index=names, columns=names)
    lr = rd("leiden_dis.csv").set_index("clone").cluster
    lr.index = lr.index.astype(str)
    cmd_r = binm("cmdscale.f64")
    clus = meta.cluster.astype(str).to_numpy()[pos]
    mx = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0  # noqa: E731
    spmx = lambda A, B: float(abs(A - B).max())  # noqa: E731
    S = {}
    S["knn"] = (lambda: ct.knn_connectivity(X, 30),
                lambda K: {"max_abs_diff": spmx(K, K_r), "same_edges": bool((K != 0).sum() == (K_r != 0).sum())})
    S["transition"] = (lambda: ct._compute_transition(K_r), lambda T: {"max_abs_diff": spmx(T, T_r)})
    S["spread"] = (lambda: ct.label_spreading(T_r, codes, C, alpha=0.6, block_size=128),
                   lambda F: {"max_abs_diff": mx(F, F_r)})
    if (d / "deviance.csv").exists():
        dev_r = rd("deviance.csv").deviance.to_numpy()

        def chk_boot(cp):
            m = np.isfinite(dev_r) & np.isfinite(cp.deviance)
            return {"corr_with_R": float(np.corrcoef(dev_r[m], cp.deviance[m])[0, 1]),
                    "mean_R": float(np.mean(dev_r[m])), "mean_arsc": float(np.mean(cp.deviance[m])),
                    "pass_0.3_R": int((dev_r < 0.3).sum()), "pass_0.3_arsc": int((cp.deviance < 0.3).sum())}
        S["bootstrap"] = (lambda: ct.propagate_clones(T_r, meta.clone, min_size=1, alpha=0.6, n_boot=48,
                                                      block_size=128, levels=I["levels"], seed=1), chk_boot)
        keep = rd("dev_keep.csv").keep.to_numpy()
        Rn = F_r + 1e-12
        Rn = Rn / Rn.sum(1, keepdims=True)
        Q0 = Rn[keep] / Rn[keep].sum(1, keepdims=True)

        def chk_sp(Sm):
            with np.errstate(invalid="ignore"):
                Sm = np.nan_to_num(Sm / Sm.sum(1, keepdims=True))
            df = pd.DataFrame(Sm, index=I["cells"][keep], columns=I["levels"])
            df = df.loc[df.sum(1) > 0, df.sum(0) > 0]
            df = df.loc[:, (df >= 0.1).sum(0) > 0]
            df = df.loc[df.sum(1) > 0]
            df = df.div(df.sum(1), axis=0)
            return {"max_abs_diff": mx(df.reindex(index=Q.index, columns=Q.columns).to_numpy(), Q.to_numpy()),
                    "same_shape": list(df.shape) == list(Q.shape)}
        S["sparsify"] = (lambda: ct.sparsify(Q0, 0.9, 0.9), chk_sp)
    sn = rd("snn_cost.csv")
    G_r = sp.coo_matrix((sn.w, (sn.i, sn.j)), shape=(len(pos), len(pos))).tocsr()
    G_r = G_r + G_r.T
    S["snn"] = (lambda: ct.snn_cost_graph(emb, 10),
                lambda G: {"max_abs_diff": spmx(G, G_r), "same_edges": bool(G.nnz == G_r.nnz)})
    S["nn"] = (lambda: ct.clone_distance(emb, Q, method="nn", square=False),
               lambda L: {"max_abs_diff": mx(L.dis.to_numpy(), nn_r.dis.to_numpy()),
                          "same_pairs": bool((L.group1.to_numpy() == nn_r.group1.to_numpy()).all())})
    if (d / "dist_ot.csv").exists():
        ot_r = rd("dist_ot.csv")
        ot_r.columns = ["group1", "group2", "dis"]
        S["ot"] = (lambda: ct.clone_distance(emb, Q.iloc[:, :9], method="exact"),
                   lambda D: {"max_abs_diff": mx(D.to_numpy()[ot_r.group1.astype(int) - 1, ot_r.group2.astype(int) - 1],
                                                 ot_r.dis.to_numpy())})
    if (d / "dist_revised.f64").exists():
        cl = pd.DataFrame({"c1": meta.cluster.astype(str).to_numpy()[pos], "c2": meta.celltype.to_numpy()[pos]})
        rv_r = binm("dist_revised.f64")
        w = rd("revised_weight.csv")
        S["revised"] = (lambda: ct.revise_distance(Dn, emb, Q, clusters=cl),
                        lambda r: {"max_abs_diff": mx(r[0].to_numpy(), rv_r),
                                   "same_weight_and_picks": bool(np.isclose(r[1]["mean_weight"], w.mean_weight[0])
                                                                 and np.allclose(r[1]["picks"], [w.pick1[0], w.pick2[0]]))})
    S["leiden"] = (lambda: ct.cluster_clones(Dn, k=8, resolution=0.5, seed=0),
                   lambda c: {"ari_vs_R": float(ari(c.to_numpy(), lr.reindex(c.index).to_numpy()))})
    S["cmdscale"] = (lambda: ct.cmdscale(Dn, 6), lambda c: {"max_abs_diff_up_to_sign": mx(np.abs(c.to_numpy()), np.abs(cmd_r))})
    st_r = float(rd("isomds_stress.csv").stress[0])
    S["isomds"] = (lambda: ct.isomds(Dn, 6), lambda r: {"stress_R": st_r, "stress_arsc": float(r[1])})
    dpt_r = rd("clone_dpt.csv")
    cell_clone = meta.clone.to_numpy()[pos]
    emb_c = pd.DataFrame(cmd_r, index=names)

    def run_dpt():
        root = ct.clone_root(Q.columns, cell_clone, clus, "0")
        return root, ct.clone_dpt(emb_c, root, k=5, dpt_k=6)
    S["dpt"] = (run_dpt, lambda r: {"same_root": bool(r[0] == str(dpt_r.root[0])),
                                    "max_abs_diff": mx(r[1].to_numpy(), dpt_r.t.to_numpy())})
    cpp_r = binm("cell_profile_prob.f64")
    mass_r = binm("profile_mass.f64")

    def run_prof():
        cpp, lab = ct.cell_profiles(Q, lr)
        return cpp, ct.profile_mass(cpp, clus)
    S["profiles"] = (run_prof, lambda r: {"max_abs_diff_cpp": mx(r[0].to_numpy(), cpp_r),
                                          "max_abs_diff_mass": mx(r[1].to_numpy(), mass_r)})
    cpp_df = pd.DataFrame(cpp_r, index=Q.index)
    pv_r = binm("enrich_pval.f64")
    S["enrichment"] = (lambda: ct.profile_enrichment(cpp_df, clus, n_perm=200, seed=0),
                       lambda r: {"frac_p_within_0.1": float(np.mean(np.abs(r[1].to_numpy() - pv_r) < 0.1))})
    G = binm("gam_G.f64").T
    tP = rd("gam_tP.csv")
    gf_r, gl_r = rd("gam_f.csv"), rd("gam_lrt.csv")

    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    S["gam_f"] = (lambda: ct.soft_cluster_fit(G, tP.t, tP[["p1", "p2"]], df=3),
                  lambda s: {"max_rel_diff_stat": rel(s.stat, gf_r.stat), "max_rel_diff_p": rel(s.pval, gf_r.pval)})
    S["gam_lrt"] = (lambda: ct.soft_cluster_fit(G, tP.t, tP[["p1", "p2"]], df=2, test="LRT"),
                    lambda s: {"max_rel_diff_stat": rel(s.stat, gl_r.stat)})
    co_r = trip("coembed.csv", len(pos))
    S["coembed"] = (lambda: ct.coembed_graph(emb, cmd_r, Q, cell_k=15, clone_k=5),
                    lambda M: {"max_abs_diff": spmx(M, co_r), "same_edges": bool(M.nnz == co_r.nnz)})
    return S


def arsc_child(size: str, threads: int, out: Path, steps: list[str] | None):
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    import arsc
    t_imp = time.perf_counter()
    warm = _steps(_load("S"))
    for name, (fn, _) in warm.items():   # untimed warm-up: numba compilation, imports
        if steps and name not in steps:
            continue
        fn()
    warm_s = time.perf_counter() - t_imp
    S = _steps(_load(size))
    ms = MemSampler(os.getpid(), tree=False).start()
    res = {}
    for name, (fn, check) in S.items():
        if steps and name not in steps:
            continue
        runs, peak = [], 0.0
        while True:
            t0w = time.time()
            t0 = time.perf_counter()
            r = fn()
            runs.append(time.perf_counter() - t0)
            peak = max(peak, ms.peak_delta(t0w, time.time()))
            if runs[0] >= 20 or len(runs) >= 3:
                break
        res[name] = {"seconds": runs, "median": float(np.median(runs)), "peak_mem_mb": round(peak / 2 ** 20, 1),
                     "agreement": check(r)}
        print(f"  {name:12s} {np.median(runs):9.3f} s  {res[name]['agreement']}", flush=True)
        del r
    ms.stop()
    out.write_text(json.dumps({"size": size, "threads": threads, "arsc_version": arsc.__version__,
                               "warmup_seconds": warm_s, "steps": res,
                               "process_peak_mb": round(max(b for _, b in ms.samples) / 2 ** 20, 1)}, indent=1))


def run_arsc(size: str, threads: list[int], steps: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for t in threads:
        env = dict(os.environ)
        for v in THREAD_VARS:
            env[v] = str(t)
        tag = f"_steps-{steps.replace(',', '-')}" if steps else ""
        out = OUT / f"{size}_arsc_t{t}{tag}.json"
        st = machine()
        rc = subprocess.run([sys.executable, __file__, "_child", size, str(t), str(out), steps], env=env).returncode
        if rc != 0:
            raise SystemExit(f"arsc child failed ({rc})")
        j = json.loads(out.read_text())
        j["machine"] = st
        j["arsc_commit"] = git_commit(ROOT)
        out.write_text(json.dumps(j, indent=1))
        print(f"wrote {out}")


# ------------------------------------------------------------------------------------------------- render
def _fmt(s):
    if s is None:
        return "-"
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 120:
        return f"{s / 60:.0f} min"
    if s >= 10:
        return f"{s:.0f} s"
    if s >= 0.1:
        return f"{s:.2f} s"
    return f"{s * 1000:.0f} ms"


def _agree(a: dict) -> str:
    parts = []
    for k, v in a.items():
        if isinstance(v, bool):
            parts.append(f"{k.replace('_', ' ')}: {'yes' if v else 'NO'}")
        elif isinstance(v, int):
            parts.append(f"{k.replace('_', ' ')} {v:,}")
        elif isinstance(v, float):
            parts.append(f"{k.replace('_', ' ')} {v:.2g}" if abs(v) < 1e-3 else f"{k.replace('_', ' ')} {v:.3f}")
    return "; ".join(parts)


def _merge(files):
    """Merge per-run JSONs of one tool/size (a step run separately overrides the full run)."""
    out = {}
    for f in sorted(files, key=lambda p: ("_steps-" in p.name, p.name)):
        j = json.loads(f.read_text())
        base = out or {k: v for k, v in j.items() if k != "steps"}
        base.setdefault("steps", {}).update(j["steps"])
        out = base
    return out


def render():
    import numpy as np
    print("<!-- CLONOTRACE-SPEED:START -->")
    for size in ("S", "M", "L"):
        rj = _merge(list(OUT.glob(f"{size}_r_w1*.json")))
        a1 = _merge(list(OUT.glob(f"{size}_arsc_t1.json")) + list(OUT.glob(f"{size}_arsc_t1_steps-*.json")))
        a16 = _merge(list(OUT.glob(f"{size}_arsc_t16.json")) + list(OUT.glob(f"{size}_arsc_t16_steps-*.json")))
        extra = json.loads((OUT / f"{size}_extrapolated.json").read_text()) if (OUT / f"{size}_extrapolated.json").exists() else {}
        if not rj or not a16:
            continue
        print(f"\n#### {size}: {rj['n_cells']:,} cells, {rj['n_clones']:,} clones "
              f"({rj['n_Q_cells']:,} cells x {rj['n_Q_clones']:,} clones after the deviance filter)\n")
        print("| step | R | arsc 1 thread | arsc 16 threads | speed-up, 1 thread | speed-up, 16 threads | R peak mem | "
              "arsc peak mem (16 threads) | agreement with R |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        tot = {"R": 0.0, "a1": 0.0, "a16": 0.0}
        flags = []
        for s in STEPS + ["nn_main"]:
            r = rj["steps"].get(s)
            if r is None and s not in extra:
                continue
            rs = float(np.median(r["seconds"])) if r else None
            mark = ""
            if s in extra:
                rs, mark = extra[s]["seconds"], "~"
                flags.append(f"{LABEL[s]}: {extra[s]['method']}")
            sa = "nn" if s == "nn_main" else s   # Clonotrace main's NN distance vs the same arsc step
            x1 = a1.get("steps", {}).get(sa) if a1 else None
            x16 = a16["steps"].get(sa)
            t1 = x1["median"] if x1 else None
            t16 = x16["median"] if x16 else None
            if s in PIPELINE:
                tot["R"] += rs or 0
                tot["a1"] += t1 or 0
                tot["a16"] += t16 or 0
            sp_ = f"{mark}{rs / t16:.1f}x" if rs and t16 else "-"
            sp1 = f"{mark}{rs / t1:.1f}x" if rs and t1 else "-"
            rm = f"{r['peak_mem_mb']:,.0f} MB" if r and "peak_mem_mb" in r else "-"
            am = f"{x16['peak_mem_mb']:,.0f} MB" if x16 else "-"
            ag = _agree(x16["agreement"]) if x16 and s != "nn_main" else ("as the row above" if x16 else "")
            print(f"| {LABEL[s]} | {mark}{_fmt(rs)} | {_fmt(t1)} | {_fmt(t16)} | {sp1} | {sp_} | {rm} | {am} | {ag} |")
        mk = "~" if any(s in extra for s in PIPELINE) else ""
        print(f"| **pipeline** (sum of the steps {', '.join(PIPELINE)}) | **{mk}{_fmt(tot['R'])}** | **{_fmt(tot['a1'])}** | "
              f"**{_fmt(tot['a16'])}** | **{mk}{tot['R'] / tot['a1']:.1f}x** | **{mk}{tot['R'] / tot['a16']:.1f}x** | "
              f"{rj['process_peak_mb']:,.0f} MB (process) | {a16['process_peak_mb']:,.0f} MB (process) | |")
        if flags:
            print("\n~ not measured in full: " + " ".join(f"{f}." for f in flags))
    print("\n<!-- CLONOTRACE-SPEED:END -->")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["r", "arsc", "render", "_child"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--size", default="S")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=48)
    ap.add_argument("--threads", default="1,16")
    ap.add_argument("--steps", default="")
    a = ap.parse_args()
    if a.cmd == "r":
        run_r(a.size, a.workers, a.n_boot, a.steps)
    elif a.cmd == "arsc":
        run_arsc(a.size, [int(t) for t in a.threads.split(",")], a.steps)
    elif a.cmd == "render":
        render()
    else:
        size, t, out = a.rest[0], int(a.rest[1]), Path(a.rest[2])
        steps = [s for s in (a.rest[3] if len(a.rest) > 3 else "").split(",") if s]
        arsc_child(size, t, out, steps or None)


if __name__ == "__main__":
    main()
