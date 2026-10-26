"""Paired arsc vs scanpy timing under equal conditions: interleaved runs, alternating order, same threads.

    python benchmarks/paired_bench.py --gate                                  # 20,000 toy cells, in memory
    python benchmarks/paired_bench.py --sizes 20000 --steps neighbors,cluster --repeats 5 --threads 4 --gate
    ARSC_BENCH_STORE=<store dir> python benchmarks/paired_bench.py --sizes 100000,1000000   # a real store

No data is written anywhere. The input is built in memory (``benchdata.py``): ``--source auto`` (default) reads
the real arsc store named by ``ARSC_BENCH_STORE`` memory-mapped and read-only (a seeded subset of ``n`` cells),
and otherwise generates the toy data of ``suite_sim.py`` in memory from a fixed seed (default 20,000 cells;
larger sizes only when asked). The parent computes the shared 30-PC embedding once and hands it to the children
through shared memory (RAM); each child rebuilds the counts itself (untimed) and returns its result, with the
neighbor lists / cluster labels for the agreement check, on stdout.

Every (size, step, repeat) runs arsc and scanpy back to back, each in a fresh process (``run_suite.run_child``,
with the untimed JIT warm-up of ``suite_py``), in alternating order (repeat 0: arsc then scanpy; repeat 1:
scanpy then arsc; ...), with the same thread count for both: ``AR_THREADS`` / ``NUMBA_NUM_THREADS`` /
``OMP_NUM_THREADS`` / BLAS variables, scanpy's ``n_jobs`` and pydeseq2's ``n_cpus`` all equal ``--threads``.
Before each run the machine state is recorded (system CPU use over 1 s, free memory, and, when the framework's
``tools/cpu_budget.py`` is found, its lease summary), so every timing carries its conditions. Only timings from
the same session are compared; nothing is compared with a stored number.

Outputs (``--out``, default ``benchmarks/results/paired``; small tables only): ``runs.jsonl`` (one line per
run), ``summary.csv`` (per size and step: median, min and max seconds per tool, median of the per-pair ratios
arsc / scanpy, the loads seen) and ``quality_<session>.csv`` (``neighbors``: recall against brute force on 1,000
cells for arsc and scanpy, the overlap of arsc with scanpy and of scanpy with itself under another
``random_state``; ``cluster``: adjusted Rand index against the cell types and between the tools, on up to 20,000
cells).

``--gate`` exits non-zero when, for any step, arsc's median time is above scanpy's median times
``1 + --tolerance`` (default 0.05, for timing noise). This is the speed regression guard: both tools are timed in the same
session, interleaved, so the gate never depends on the machine a stored number came from.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from benchdata import DEFAULT_TOY_N, Bench, resolve_source, share_pca  # noqa: E402
from run_suite import IMPL, run_child  # noqa: E402

THREAD_VARS = ("AR_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PAIRED_STEPS = [s for s in ["normalize", "hvg", "pca", "neighbors", "cluster", "rank_genes", "pseudobulk",
                            "de_deseq2", "score_genes"] if s in IMPL["scanpy"]]
TIMEOUT = {10_000: 900, 20_000: 900, 50_000: 1800, 100_000: 2400, 1_000_000: 7200}


def machine_state() -> dict:
    vm = psutil.virtual_memory()
    st = {"cpu_percent_1s": psutil.cpu_percent(interval=1.0), "mem_available_gb": round(vm.available / 2 ** 30, 1),
          "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    cb = ROOT.parents[1] / "tools" / "cpu_budget.py"   # the autoresearch framework, when arsc is vendored there
    if cb.exists():
        try:
            out = subprocess.run([sys.executable, str(cb), "status"], capture_output=True, text=True, timeout=60,
                                 creationflags=0x08000000 if os.name == "nt" else 0).stdout.splitlines()
            st["lease"] = next((l.strip() for l in out if l.strip().startswith("leased")), None)
        except Exception as e:  # noqa: BLE001
            st["lease"] = f"unavailable: {e}"
    return st


def child_env(threads: int, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(threads)
    env["ARSC_BENCH_THREADS"] = str(threads)
    env.update(extra or {})
    return env


def run_tool(n, tool, step, source, seed, pca_shm, py, threads, extra_env=None):
    cmd = [py, str(HERE / "suite_py.py"), "--tool", tool, "--step", step, "--source", source, "--n", str(n),
           "--seed", str(seed), "--pca-shm", pca_shm, "--warm", "toy"]
    return run_child(cmd, None, TIMEOUT.get(n, 3600), env=child_env(threads, extra_env))


def run_inprocess(n, steps, repeats, source, seed, pca_shm, py, threads):
    """The fast mode: one child times every step for both tools (suite_py.py --tool both)."""
    cmd = [py, str(HERE / "suite_py.py"), "--tool", "both", "--step", ",".join(steps), "--source", source, "--n",
           str(n), "--seed", str(seed), "--pca-shm", pca_shm, "--repeats", str(repeats)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=child_env(threads),
                       timeout=TIMEOUT.get(n, 3600) * len(steps),
                       creationflags=0x08000000 if os.name == "nt" else 0)
    out = [json.loads(l[9:]) for l in p.stdout.splitlines() if l.startswith("@@RESULT ")]
    if not out:
        raise SystemExit("in-process benchmark child failed:\n" + (p.stdout + p.stderr)[-3000:])
    return out


def run_batch(n, tool, steps, source, seed, pca_shm, py, threads, extra_env=None):
    """One child process runs several steps, each warmed untimed right before it; a list of per-step results."""
    cmd = [py, str(HERE / "suite_py.py"), "--tool", tool, "--step", ",".join(steps), "--source", source, "--n",
           str(n), "--seed", str(seed), "--pca-shm", pca_shm, "--warm", "toy"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=child_env(threads, extra_env),
                       timeout=TIMEOUT.get(n, 3600) * len(steps),
                       creationflags=0x08000000 if os.name == "nt" else 0)
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT "):
            r = json.loads(line[9:])
            got[r["step"]] = r
    tail = (p.stdout + p.stderr)[-800:]
    return [got.get(s_, {"tool": tool, "step": s_, "ok": False, "error": "no result: " + tail}) for s_ in steps]


def knn_quality(E: np.ndarray, payloads: dict) -> dict:
    """Recall of each tool's k-NN (the 1,000 evaluation cells of suite_py.eval_cells) against brute force;
    overlaps between tools."""
    from suite_py import eval_cells
    nn = {t: np.asarray(p["nn"]) for t, p in payloads.items() if p and "nn" in p}
    out = {}
    if not nn:
        return out
    k = next(iter(nn.values())).shape[1]
    E = np.asarray(E, dtype=np.float64)
    qs = eval_cells(E.shape[0])
    truth = []
    for q in qs:
        d = ((E - E[q]) ** 2).sum(1)
        part = np.argpartition(d, k)[: k + 1]
        truth.append(set(part[np.argsort(d[part], kind="stable")][:k].tolist()))
    for tag, a in nn.items():
        out[f"recall_{tag}"] = float(np.mean([len(set(a[i].tolist()) & truth[i]) / k for i in range(len(qs))]))

    def overlap(a, b):
        return float(np.mean([len(set(a[i].tolist()) & set(b[i].tolist())) / k for i in range(a.shape[0])]))
    if "arsc" in nn and "scanpy" in nn:
        out["overlap_arsc_scanpy"] = overlap(nn["arsc"], nn["scanpy"])
    if "scanpy" in nn and "scanpy_seed1" in nn:
        out["overlap_scanpy_seed0_seed1"] = overlap(nn["scanpy"], nn["scanpy_seed1"])
    return out


def cluster_agreement(cell_type: np.ndarray, payloads: dict) -> dict:
    """Adjusted Rand index of each tool's clusters with the cell types and between tools (suite_py.label_cells)."""
    from sklearn.metrics import adjusted_rand_score as ari
    from suite_py import label_cells
    ct = np.asarray(cell_type)[label_cells(len(cell_type))]
    labs = {t: np.asarray(p["labels"]) for t, p in payloads.items() if p and "labels" in p}
    out = {"n_cells_scored": int(ct.size)}
    for t, l in labs.items():
        out[f"ari_truth_{t}"] = float(ari(ct, l))
        out[f"n_clusters_{t}"] = int(len(np.unique(l)))
    if {"arsc", "scanpy"} <= set(labs):
        out["ari_arsc_scanpy"] = float(ari(labs["arsc"], labs["scanpy"]))
    return out


def summarize(runs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in runs if r.get("ok")])
    rows = []
    for (n, step), g in df.groupby(["n", "step"]):
        row = {"n": n, "step": step, "threads": int(g["threads"].iloc[0])}
        for tool in ("arsc", "scanpy"):
            s = g[g.tool == tool]["seconds"]
            row[f"{tool}_median_s"] = float(s.median()) if len(s) else np.nan
            row[f"{tool}_min_s"] = float(s.min()) if len(s) else np.nan
            row[f"{tool}_max_s"] = float(s.max()) if len(s) else np.nan
            row[f"{tool}_runs"] = int(len(s))
        pr = g.pivot_table(index="rep", columns="tool", values="seconds")
        if {"arsc", "scanpy"} <= set(pr.columns):
            ratios = (pr["arsc"] / pr["scanpy"]).dropna()
            row["pair_ratio_median"] = float(ratios.median())
            row["pair_ratio_max"] = float(ratios.max())
        row["speedup_median"] = row["scanpy_median_s"] / row["arsc_median_s"]
        row["cpu_percent_before_min"] = float(g["cpu_percent_1s"].min())
        row["cpu_percent_before_max"] = float(g["cpu_percent_1s"].max())
        row["mem_available_gb_min"] = float(g["mem_available_gb"].min())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=str(DEFAULT_TOY_N), help="cells per dataset (default 20,000; larger sizes "
                                                                  "only when asked, still in memory)")
    ap.add_argument("--source", default="auto", help="'auto' (ARSC_BENCH_STORE if set, else toy), 'toy' or "
                                                     "'store:<dir>' (read-only, memory-mapped)")
    ap.add_argument("--steps", default=",".join(PAIRED_STEPS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("AR_THREADS", "8")))
    ap.add_argument("--out", default=str(HERE / "results" / "paired"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="exit 1 if arsc's median is slower than scanpy's")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--mode", default="inprocess", choices=["inprocess", "process"],
                    help="inprocess (default, about a minute at the default size): one process holds the data and "
                         "times both tools, interleaved, after warming every step; process: a fresh process per "
                         "(tool, repeat) running all steps (or per step with --per-step-process)")
    ap.add_argument("--per-step-process", action="store_true", help="a fresh process per (tool, step, repeat), "
                    "with peak-memory sampling (slower: every process pays its own start-up and warm-up)")
    ap.add_argument("--quality", action="store_true", help="also run scanpy's neighbors with another seed "
                    "(its run-to-run overlap; one more process)")
    a = ap.parse_args()
    source = resolve_source(a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [s for s in a.steps.split(",") if s]
    bad = [s for s in steps if s not in PAIRED_STEPS]
    if bad:
        raise SystemExit(f"steps without a scanpy equivalent: {bad}")
    runs_f = out / "runs.jsonl"
    session = time.strftime("%Y%m%d-%H%M%S")
    runs, quality = [], []
    print(f"data source: {source if not source.startswith('store:') else 'store (read-only, memory-mapped)'}",
          flush=True)
    for n in [int(x) for x in a.sizes.split(",") if x]:
        D = Bench(source, n, a.seed)
        E = D.pca                                   # shared input, computed once, handed over in RAM
        shm, spec = share_pca(E)
        try:
            # default: one child per (tool, repeat) runs every step, each warmed untimed just before its timed
            # call; --per-step-process: a fresh process per (tool, step, repeat) with peak-memory sampling
            groups = [[s_] for s_ in steps] if a.per_step_process else [steps]
            payloads = {s_: {} for s_ in steps}
            if a.mode == "inprocess":
                groups = []
                st = machine_state()
                for r in run_inprocess(n, steps, a.repeats, source, a.seed, spec, a.python, a.threads):
                    pl = r.pop("payload", None)
                    if r.get("rep") == 0:
                        payloads.setdefault(r["step"], {})[r["tool"]] = pl
                    r.update(n=n, threads=a.threads, session=session, mode="inprocess",
                             source="store" if source.startswith("store:") else source, **st)
                    for k in ("traceback", "log_tail"):
                        r.pop(k, None)
                    if isinstance(r.get("error"), str):
                        r["error"] = re.sub(r"[A-Za-z]:[\\/][^\"\s]*|/(?:home|Users)/[^\"\s]*", "<path>",
                                            r["error"])
                    runs.append(r)
                    with open(runs_f, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    msg = f"{r['seconds']:.3f} s" if r.get("ok") else (r.get("error") or "?")[:200]
                    print(f"[n={n:>8,}] rep {r.get('rep')} {r['tool']:6s} {r['step']:11s} {msg}", flush=True)
            for group in groups:
                for rep in range(a.repeats):
                    order = ("arsc", "scanpy") if rep % 2 == 0 else ("scanpy", "arsc")
                    for pos, tool in enumerate(order):
                        st = machine_state()
                        if a.per_step_process:
                            batch = [run_tool(n, tool, group[0], source, a.seed, spec, a.python, a.threads)]
                        else:
                            batch = run_batch(n, tool, group, source, a.seed, spec, a.python, a.threads)
                        for step, r in zip(group, batch):
                            pl = r.pop("payload", None)
                            if rep == 0:
                                payloads[step][tool] = pl
                            r.update(n=n, tool=tool, step=step, rep=rep, order_pos=pos, threads=a.threads,
                                     session=session, source="store" if source.startswith("store:") else source,
                                     **{k: v for k, v in st.items()})
                            for k in ("traceback", "log_tail"):   # keep local paths out of the committed results
                                r.pop(k, None)
                            if isinstance(r.get("error"), str):
                                r["error"] = re.sub(r"[A-Za-z]:[\\/][^\"\s]*|/(?:home|Users)/[^\"\s]*", "<path>",
                                                    r["error"])
                            runs.append(r)
                            with open(runs_f, "a") as f:
                                f.write(json.dumps(r) + "\n")
                            msg = f"{r['seconds']:.2f} s" if r.get("ok") else (r.get("error") or "?")[:200]
                            print(f"[n={n:>8,}] rep {rep} {tool:6s} {step:11s} {msg}  (cpu {st['cpu_percent_1s']:.0f}%, "
                                  f"free {st['mem_available_gb']} GB; {st.get('lease')})", flush=True)
            if "neighbors" in steps and a.quality:
                # scanpy's own run-to-run variation: another pynndescent random_state (untimed for the table)
                r = run_tool(n, "scanpy", "neighbors", source, a.seed, spec, a.python, a.threads,
                             extra_env={"ARSC_BENCH_SCANPY_SEED": "1"})
                payloads["neighbors"]["scanpy_seed1"] = r.get("payload")
            if "neighbors" in steps:
                q = knn_quality(E, payloads["neighbors"])
                q.update(n=n, session=session, step="neighbors")
                quality.append(q)
                print(f"[n={n:>8,}] knn quality {q}", flush=True)
            if "cluster" in steps:
                q = cluster_agreement(D.obs["cell_type"].astype(str).to_numpy(), payloads["cluster"])
                q.update(n=n, session=session, step="cluster")
                quality.append(q)
                print(f"[n={n:>8,}] cluster agreement {q}", flush=True)
        finally:
            shm.close()
            shm.unlink()
        del D, E
    summ = summarize(runs)
    summ["session"] = session
    summ.to_csv(out / f"summary_{session}.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    if quality:
        pd.DataFrame(quality).to_csv(out / f"quality_{session}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ[["n", "step", "threads", "arsc_median_s", "arsc_min_s", "arsc_max_s", "scanpy_median_s",
                    "scanpy_min_s", "scanpy_max_s", "speedup_median", "pair_ratio_max"]].round(3).to_string(index=False))
    if a.gate:
        slow = summ[summ["arsc_median_s"] > summ["scanpy_median_s"] * (1 + a.tolerance)]
        if len(slow):
            print("SPEED GATE FAILED: arsc slower than scanpy (same session, interleaved):")
            print(slow[["n", "step", "arsc_median_s", "scanpy_median_s"]].to_string(index=False))
            raise SystemExit(1)
        print("speed gate passed: arsc median <= scanpy median x (1 + %.2f) for every step" % a.tolerance)


if __name__ == "__main__":
    main()
