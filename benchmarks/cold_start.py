"""First call in a fresh process (JIT compilation included, nothing warmed): arsc vs scanpy k-NN graph, paired.

    python benchmarks/cold_start.py --n 100000 --repeats 3 --threads 3
    python benchmarks/cold_start.py --n 100000 --what knn        # arsc.knn alone (the k-NN backend)

The 30-PC embedding is built in memory (``benchdata.py``: the real store of ``ARSC_BENCH_STORE``, read-only, or
the in-memory toy data) and handed to each child through shared memory; nothing is written but the result table.
Each run is a new process that times one call: ``arsc.find_neighbors`` (or ``arsc.knn``) or ``sc.pp.neighbors``
(both include their libraries' numba compilation unless cached on disk by an earlier process). Alternating
order; ``import`` time is reported separately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchdata import Bench, share_pca  # noqa: E402
from paired_bench import child_env, machine_state  # noqa: E402

CHILD = r'''
import json, os, sys, time
t0 = time.time()
sys.path.insert(0, sys.argv[4])
import numpy as np
from multiprocessing import shared_memory
name, r, c = sys.argv[2].split(":")
shm = shared_memory.SharedMemory(name=name)
E = np.array(np.ndarray((int(r), int(c)), dtype=np.float32, buffer=shm.buf))
k = int(sys.argv[3])
if sys.argv[1] in ("arsc", "arsc_knn"):
    import arsc
    ti = time.time() - t0
    t = time.time()
    if sys.argv[1] == "arsc":
        arsc.find_neighbors(E.astype(np.float64), k=k)
    else:
        arsc.knn(E, k)
    s = time.time() - t
else:
    import anndata as ad, pandas as pd, scanpy as sc
    ti = time.time() - t0
    sc.settings.n_jobs = int(os.environ["AR_THREADS"])
    a = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(E.shape[0])])); a.obsm["X_pca"] = E
    t = time.time(); sc.pp.neighbors(a, n_neighbors=k, use_rep="X_pca"); s = time.time() - t
shm.close()
print("@@RESULT " + json.dumps({"seconds": s, "import_s": ti}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--source", default="auto")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--what", default="find_neighbors", choices=["find_neighbors", "knn"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    D = Bench(a.source, a.n, 0)
    shm, spec = share_pca(D.pca)
    me = "arsc" if a.what == "find_neighbors" else "arsc_knn"
    runs = []
    try:
        for rep in range(a.repeats):
            for tool in ((me, "scanpy") if rep % 2 == 0 else ("scanpy", me)):
                st = machine_state()
                p = subprocess.run([sys.executable, "-c", CHILD, tool, spec, str(a.k), str(HERE.parent / "src")],
                                   env=child_env(a.threads), check=True, capture_output=True, text=True)
                r = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("@@RESULT "))[9:])
                r.update(tool=tool, rep=rep, n=a.n, k=a.k, threads=a.threads, **st)
                runs.append(r)
                print(f"rep {rep} {tool:8s} first call {r['seconds']:.1f} s, import {r['import_s']:.1f} s "
                      f"(cpu {st['cpu_percent_1s']:.0f}%)", flush=True)
    finally:
        shm.close()
        shm.unlink()
    summ = {t: float(np.median([r["seconds"] for r in runs if r["tool"] == t])) for t in (me, "scanpy")}
    print(json.dumps(summ))
    tag = "" if a.what == "find_neighbors" else "_knn"
    Path(a.out, f"cold_start{tag}_n{a.n}.json").write_text(json.dumps({"median_s": summ, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
