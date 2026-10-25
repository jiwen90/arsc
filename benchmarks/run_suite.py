"""arsc benchmark suite: arsc vs scanpy vs Seurat on the same synthetic data, several sizes.

One command (from the repo root)::

    python benchmarks/run_suite.py --bench-dir <scratch dir>        # 10k + 100k, all tools, all steps (writes data)
    python benchmarks/run_suite.py --bench-dir <scratch dir> --sizes 10000,100000 --tools arsc,scanpy
    python benchmarks/run_suite.py --compare-only          # recompute agreement + summary from saved outputs
    python benchmarks/render_table.py                      # headline table -> README.md

Every (size, tool, step) runs in a fresh process. The parent samples the child's resident set size
(psutil, process + children, every 50 ms) and takes the peak inside the child's timed region, so
times and memory are not polluted by earlier steps. The timed region covers only the analysis call;
data loading and inputs the step does not own are prepared before it, untimed.

To compare like with like, steps that need upstream results (PCA, neighbours, clustering, DE) read
shared inputs computed once with arsc (``common/``): the 2,000 HVGs, the 30-PC embedding and the
pseudobulk matrix of cell type T0. Agreement is then a property of the step alone.

Data on disk is opt-in: this suite needs ``--bench-dir`` (or ``ARSC_BENCH_DIR``), the directory it WRITES the
simulated data and raw outputs to (put it outside the repository). arsc vs scanpy timing without writing any
data is ``paired_bench.py`` (in memory).
Environment: ``ARSC_BENCH_DIR`` (data + raw outputs; no default),
``ARSC_RLIB`` (an extra R library path), ``RSCRIPT`` (default ``Rscript``).
Results: ``benchmarks/results/suite/`` (``runs.jsonl``, ``summary.json``, ``summary.csv``,
``agreement.csv``, ``environment.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

STEPS = ["normalize", "hvg", "pca", "neighbors", "cluster", "markers", "rank_genes", "pseudobulk", "de_deseq2", "de_voom",
         "score_module", "score_genes", "score_ucell", "clones"]
# which tools implement which step (others are recorded as "no equivalent")
IMPL = {
    "arsc": set(STEPS),
    "scanpy": {"normalize", "hvg", "pca", "neighbors", "cluster", "rank_genes", "pseudobulk", "de_deseq2", "score_genes"},
    "seurat": {"normalize", "hvg", "pca", "neighbors", "cluster", "markers", "pseudobulk", "de_deseq2", "de_voom",
               "score_module", "score_ucell"},
}
TIMEOUT = {10_000: 900, 100_000: 2400, 1_000_000: 3600}


# ------------------------------------------------------------------ environment
def environment() -> dict:
    import importlib.metadata as md
    vers = {}
    for p in ["numpy", "scipy", "pandas", "numba", "scikit-learn", "anndata", "scanpy", "pydeseq2", "igraph",
              "leidenalg", "pynndescent", "umap-learn", "h5py", "psutil"]:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    import arsc
    vers["arsc"] = arsc.__version__
    cpu = platform.processor()
    try:
        if platform.system() == "Windows":
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True,
                                 timeout=30).stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                       if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    r = {}
    try:
        code = ('rl<-Sys.getenv("ARSC_RLIB");if(nzchar(rl)).libPaths(c(rl,.libPaths()));'
                'cat(R.version.string,"\\n");for(p in c("Seurat","SeuratObject","presto","DESeq2","limma","edgeR",'
                '"UCell","Matrix"))cat(p,if(requireNamespace(p,quietly=TRUE))as.character(packageVersion(p))else"NA","\\n")')
        out = subprocess.run([os.environ.get("RSCRIPT", "Rscript"), "-e", code], capture_output=True, text=True,
                             timeout=120).stdout.splitlines()
        r["R"] = out[0].strip()
        r.update({l.split()[0]: l.split()[1] for l in out[1:] if len(l.split()) == 2})
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)
    osname = f"{platform.system()} {platform.release()} ({platform.version()})"
    if platform.system() == "Windows" and int(platform.version().split(".")[-1]) >= 22000:
        osname = f"Windows 11 ({platform.version()})"   # platform.release() reports "10" on Windows 11
    return {"os": osname, "cpu": cpu,
            "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "python": platform.python_version(),
            "python_packages": vers, "r_packages": r,
            "threads_note": "default threading for every tool (arsc/numba and scanpy use all cores; most Seurat "
                            "steps are single-threaded R)",
            "warmup_note": "Python steps are first run untimed on a 12,000-cell dataset in the same process, so "
                           "one-off JIT compilation (numba, pynndescent) is not counted",
            "date": time.strftime("%Y-%m-%d")}


# ------------------------------------------------------------------ shared inputs
def prepare_common(data: Path) -> None:
    import arsc
    cd = data / "common"
    if (cd / "done").exists():
        return
    cd.mkdir(exist_ok=True)
    st = arsc.open(data / "store")
    genes = st.features("RNA")
    X = st.X("RNA", "counts")
    h = arsc.pp.hvg_vst(X, nfeatures=2000, features=genes)
    hv = list(h.sort_values("rank").index[:2000])
    (cd / "hvg.txt").write_text("\n".join(hv))
    idx = np.sort(genes.get_indexer(hv))
    L = arsc.pp.log_normalize(X)[:, idx]
    Z, _, _ = arsc.pp.scale(L)
    E = arsc.pp.pca(Z, n_comps=30).embedding.astype(np.float32)
    np.save(cd / "pca.npy", E)
    np.asfortranarray(E).ravel(order="F").tofile(cd / "pca.f32")
    (cd / "pca_shape.json").write_text(json.dumps(list(E.shape)))
    obs = st.obs
    m = (obs["cell_type"].astype(str) == "T0").to_numpy()
    pb = arsc.pseudobulk(X[np.where(m)[0]], obs[m], "sample", features=genes)
    cnt = pd.DataFrame(pb.X.T, index=genes, columns=pb.groups.index)
    keep = (cnt.values >= 10).sum(axis=1) >= 3
    cnt = cnt.loc[keep].round().astype(np.int64)
    cnt.to_parquet(cd / "pb_T0_counts.parquet")
    cnt.to_csv(cd / "pb_T0_counts.csv")
    col = pd.DataFrame({"patient": [s.split("_")[0] for s in cnt.columns],
                        "timepoint": [s.split("_")[1] for s in cnt.columns]}, index=cnt.columns)
    col.to_csv(cd / "pb_T0_coldata.csv")
    (cd / "done").write_text("ok")


# ------------------------------------------------------------------ one child run with RSS sampling
def _tree_rss(p: psutil.Process) -> int:
    tot = 0
    try:
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0
    for q in procs:
        try:
            tot += q.memory_info().rss
        except psutil.Error:
            pass
    return tot


def run_child(cmd: list, res_file: Path | None, timeout: float, env=None) -> dict:
    """Run one child, sampling its memory. Its result is ``res_file`` (on-disk suite) or, with ``res_file=None``,
    the child's stdout line starting with ``@@RESULT`` (in-memory benchmarks: nothing is written)."""
    if res_file is not None and res_file.exists():
        res_file.unlink()
    t_launch = time.time()
    p = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    samples = []
    stop = threading.Event()
    log = []

    def reader():
        for line in p.stdout:
            log.append(line.decode(errors="replace"))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    timed_out = False
    while p.poll() is None:
        samples.append((time.time(), _tree_rss(p)))
        if time.time() - t_launch > timeout:
            for q in p.children(recursive=True):
                q.kill()
            p.kill()
            timed_out = True
            break
        stop.wait(0.05)
    p.wait()
    th.join(timeout=5)
    wall = time.time() - t_launch
    if res_file is not None:
        got = res_file.exists()
        res = json.loads(res_file.read_text()) if got else {"ok": False}
    else:
        hit = [l for l in log if l.startswith("@@RESULT ")]
        got = bool(hit)
        res = json.loads(hit[-1][len("@@RESULT "):]) if got else {"ok": False}
        log = [l for l in log if not l.startswith("@@RESULT ")]
    if timed_out:
        res.update(ok=False, error=f"timeout after {timeout:.0f} s")
    elif not got:
        res["error"] = "child exited without a result: " + "".join(log)[-1500:]
    ts, rs = (np.array([s[0] for s in samples]), np.array([s[1] for s in samples])) if samples else (np.zeros(0), np.zeros(0))
    t0, t1 = res.get("t_start"), res.get("t_end")
    if res.get("ok") and t0 and t1 and len(ts):
        inside = (ts >= t0 - 0.05) & (ts <= t1 + 0.05)
        before = ts < t0
        res["rss_peak_gb"] = float(rs[inside].max() / 2 ** 30) if inside.any() else float(rs.max() / 2 ** 30)
        res["rss_base_gb"] = float(rs[before][-1] / 2 ** 30) if before.any() else None
    res["process_peak_gb"] = float(rs.max() / 2 ** 30) if len(rs) else None
    res["process_wall_s"] = wall
    if not res.get("ok"):
        res["log_tail"] = "".join(log)[-800:]
    root = str(ROOT)
    for k in ("traceback", "log_tail", "error"):   # keep local paths out of the committed results
        if isinstance(res.get(k), str):
            res[k] = res[k].replace(root, "<repo>").replace(root.replace("\\", "/"), "<repo>")
    return res


def run_one(n: int, tool: str, step: str, data: Path, res_dir: Path, py: str, warm: Path | None = None) -> dict:
    rf = res_dir / "tmp" / f"n{n}_{tool}_{step}.json"
    rf.parent.mkdir(parents=True, exist_ok=True)
    if step not in IMPL[tool]:
        return {"tool": tool, "step": step, "ok": False, "skipped": "no equivalent"}
    if tool == "seurat":
        cmd = [os.environ.get("RSCRIPT", "Rscript"), str(HERE / "suite_seurat.R"), step, str(data), str(rf)]
    else:
        cmd = [py, str(HERE / "suite_py.py"), "--tool", tool, "--step", step, "--data", str(data), "--out", str(rf)]
        if warm is not None:
            cmd += ["--warm", str(warm)]
    return run_child(cmd, rf, TIMEOUT.get(n, 3600))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="10000,100000")
    ap.add_argument("--tools", default="arsc,scanpy,seurat")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--bench-dir", default=os.environ.get("ARSC_BENCH_DIR", ""),
                    help="directory the simulated data are WRITTEN to (required: the Seurat side reads files). "
                         "Opt-in; for arsc vs scanpy timing use paired_bench.py, which keeps everything in memory")
    ap.add_argument("--out", default=str(HERE / "results" / "suite"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerun", action="store_true", help="re-run steps that already have a result")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    if not a.bench_dir:
        raise SystemExit("run_suite.py writes its simulated data to disk (Seurat reads files), so it needs an explicit "
                         "--bench-dir (or ARSC_BENCH_DIR). For arsc vs scanpy timing without any data on disk use "
                         "benchmarks/paired_bench.py.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = Path(a.bench_dir)
    sizes = [int(s) for s in a.sizes.split(",") if s]
    tools = [t for t in a.tools.split(",") if t]
    steps = [s for s in a.steps.split(",") if s]
    if not a.compare_only:
        envf = out / "environment.json"
        envf.write_text(json.dumps(environment(), indent=1))
        from suite_sim import simulate
        runs_f = out / "runs.jsonl"
        have = set()
        if runs_f.exists() and not a.rerun:
            for l in runs_f.read_text().splitlines():
                r = json.loads(l)
                if r.get("ok") or r.get("skipped"):
                    have.add((r["n"], r["tool"], r["step"]))
        warm = simulate(12000, bench, seed=a.seed + 1, verbose=False)
        prepare_common(warm)
        for n in sizes:
            data = simulate(n, bench, seed=a.seed)
            prepare_common(data)
            for tool in tools:
                for step in steps:
                    if (n, tool, step) in have:
                        continue
                    t = time.time()
                    r = run_one(n, tool, step, data, out, a.python, warm)
                    r.update(n=n, tool=tool, step=step)
                    msg = (f"{r['seconds']:.2f} s, peak {r.get('rss_peak_gb', float('nan')):.2f} GB" if r.get("ok")
                           else (r.get("skipped") or r.get("error", "?"))[:200])
                    print(f"[n={n:>8,}] {tool:7s} {step:13s} {msg}   ({time.time() - t:.0f} s total)", flush=True)
                    if r.get("skipped") == "no equivalent":
                        continue
                    with open(runs_f, "a") as f:
                        f.write(json.dumps(r) + "\n")
    from suite_compare import compare_all
    compare_all(bench, out, sizes)


if __name__ == "__main__":
    main()
