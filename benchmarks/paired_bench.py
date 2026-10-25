"""Paired arsc vs scanpy timing under equal conditions: interleaved runs, alternating order, same threads.

    python benchmarks/paired_bench.py --sizes 100000,1000000 --repeats 3 --threads 8
    python benchmarks/paired_bench.py --sizes 20000 --steps neighbors,cluster --repeats 5 --threads 4 --gate

Every (size, step, repeat) runs arsc and scanpy back to back, each in a fresh process (``run_suite.run_child``,
with the untimed JIT warm-up of ``suite_py``), in alternating order (repeat 0: arsc then scanpy; repeat 1:
scanpy then arsc; ...), with the same thread count for both: ``AR_THREADS`` / ``NUMBA_NUM_THREADS`` /
``OMP_NUM_THREADS`` / BLAS variables, scanpy's ``n_jobs`` and pydeseq2's ``n_cpus`` all equal ``--threads``.
Before each run the machine state is recorded (system CPU use over 1 s, free memory, and, when the framework's
``tools/cpu_budget.py`` is found, its lease summary), so every timing carries its conditions. Only timings from
the same session are compared; nothing is compared with a stored number.

Outputs (``--out``, default ``benchmarks/results/paired``): ``runs.jsonl`` (one line per run),
``summary.csv`` (per size and step: median, min and max seconds per tool, median of the per-pair ratios
arsc / scanpy, the loads seen) and, for ``neighbors``, ``knn_quality.csv`` (recall against brute force on 1,000
cells for arsc and scanpy, the overlap of arsc with scanpy and of scanpy with itself under another
``random_state``).

``--gate`` exits non-zero when, for any step, arsc's median time is above scanpy's median times
``1 + --tolerance`` (default 0.05, for timing noise). This is the speed regression guard: both tools are timed in the same
session, interleaved, so the gate never depends on the machine a stored number came from.
"""
from __future__ import annotations

import argparse
import json
import os
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

from run_suite import IMPL, prepare_common, run_child  # noqa: E402

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


def run_tool(n, tool, step, data, warm, res_dir, py, threads, extra_env=None, tag=""):
    rf = res_dir / "tmp" / f"n{n}_{tool}{tag}_{step}.json"
    rf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(HERE / "suite_py.py"), "--tool", tool, "--step", step, "--data", str(data), "--out", str(rf),
           "--warm", str(warm)]
    return run_child(cmd, rf, TIMEOUT.get(n, 3600), env=child_env(threads, extra_env))


def knn_quality(data: Path, n_eval: int = 1000, seed: int = 0) -> dict:
    """Recall of each tool's k-NN against brute force on n_eval random cells; overlaps between tools."""
    E = np.load(data / "common" / "pca.npy").astype(np.float64)
    out = {}
    nn = {}
    for tag, f in (("arsc", data / "out" / "arsc" / "neighbors.npy"),
                   ("scanpy", data / "out" / "scanpy" / "neighbors.npy"),
                   ("scanpy_seed1", data / "out" / "scanpy_seed1" / "neighbors.npy")):
        if f.exists():
            nn[tag] = np.load(f)
    if not nn:
        return out
    k = next(iter(nn.values())).shape[1]
    rng = np.random.default_rng(seed)
    qs = rng.choice(E.shape[0], min(n_eval, E.shape[0]), replace=False)
    truth = []
    for q in qs:
        d = ((E - E[q]) ** 2).sum(1)
        part = np.argpartition(d, k)[: k + 1]
        truth.append(set(part[np.argsort(d[part], kind="stable")][:k].tolist()))
    for tag, a in nn.items():
        out[f"recall_{tag}"] = float(np.mean([len(set(a[q].tolist()) & truth[i]) / k for i, q in enumerate(qs)]))

    def overlap(a, b):
        return float(np.mean([len(set(a[i].tolist()) & set(b[i].tolist())) / k for i in range(a.shape[0])]))
    if "arsc" in nn and "scanpy" in nn:
        out["overlap_arsc_scanpy"] = overlap(nn["arsc"], nn["scanpy"])
    if "scanpy" in nn and "scanpy_seed1" in nn:
        out["overlap_scanpy_seed0_seed1"] = overlap(nn["scanpy"], nn["scanpy_seed1"])
    return out


def cluster_agreement(data: Path) -> dict:
    from sklearn.metrics import adjusted_rand_score as ari
    truth = json.loads((data / "truth.json").read_text())
    import arsc
    ct = arsc.open(data / "store").obs["cell_type"].astype(str).to_numpy()
    out = {}
    labs = {t: np.load(data / "out" / t / "cluster.npy") for t in ("arsc", "scanpy")
            if (data / "out" / t / "cluster.npy").exists()}
    for t, l in labs.items():
        out[f"ari_truth_{t}"] = float(ari(ct, l))
        out[f"n_clusters_{t}"] = int(len(np.unique(l)))
    if len(labs) == 2:
        out["ari_arsc_scanpy"] = float(ari(labs["arsc"], labs["scanpy"]))
    del truth
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
    ap.add_argument("--sizes", default="100000")
    ap.add_argument("--steps", default=",".join(PAIRED_STEPS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("AR_THREADS", "8")))
    ap.add_argument("--bench-dir", default=os.environ.get("ARSC_BENCH_DIR", str(HERE / ".bench_data")))
    ap.add_argument("--out", default=str(HERE / "results" / "paired"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="exit 1 if arsc's median is slower than scanpy's")
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args()
    from suite_sim import simulate
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = Path(a.bench_dir)
    steps = [s for s in a.steps.split(",") if s]
    bad = [s for s in steps if s not in PAIRED_STEPS]
    if bad:
        raise SystemExit(f"steps without a scanpy equivalent: {bad}")
    warm = simulate(12000, bench, seed=a.seed + 1, verbose=False)
    prepare_common(warm)
    runs_f = out / "runs.jsonl"
    session = time.strftime("%Y%m%d-%H%M%S")
    runs, quality = [], []
    for n in [int(x) for x in a.sizes.split(",") if x]:
        data = simulate(n, bench, seed=a.seed)
        prepare_common(data)
        for step in steps:
            for rep in range(a.repeats):
                order = ("arsc", "scanpy") if rep % 2 == 0 else ("scanpy", "arsc")
                for pos, tool in enumerate(order):
                    st = machine_state()
                    r = run_tool(n, tool, step, data, warm, out, a.python, a.threads)
                    r.update(n=n, tool=tool, step=step, rep=rep, order_pos=pos, threads=a.threads, session=session,
                             **{k: v for k, v in st.items()})
                    for k in ("traceback", "log_tail"):
                        r.pop(k, None) if r.get("ok") else None
                    runs.append(r)
                    with open(runs_f, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    msg = f"{r['seconds']:.2f} s" if r.get("ok") else (r.get("error") or "?")[:200]
                    print(f"[n={n:>8,}] rep {rep} {tool:6s} {step:11s} {msg}  (cpu {st['cpu_percent_1s']:.0f}%, "
                          f"free {st['mem_available_gb']} GB; {st.get('lease')})", flush=True)
            if step == "neighbors":
                # scanpy's own run-to-run variation: another pynndescent random_state (untimed for the table)
                run_tool(n, "scanpy", "neighbors", data, warm, out, a.python, a.threads,
                         extra_env={"ARSC_BENCH_SCANPY_SEED": "1", "ARSC_BENCH_OUT_TAG": "_seed1"}, tag="_seed1")
                q = knn_quality(data)
                q.update(n=n, session=session)
                quality.append(q)
                print(f"[n={n:>8,}] knn quality {q}", flush=True)
            if step == "cluster":
                q = cluster_agreement(data)
                q.update(n=n, session=session, step="cluster")
                quality.append(q)
                print(f"[n={n:>8,}] cluster agreement {q}", flush=True)
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
