"""First call in a fresh process (JIT compilation included, nothing warmed): arsc vs scanpy k-NN graph, paired.

    python benchmarks/cold_start.py --n 100000 --repeats 3 --threads 3

Each run is a new process that loads the shared 30-PC embedding and times one call: ``arsc.find_neighbors`` or
``sc.pp.neighbors`` (both include their libraries' numba compilation unless cached on disk). Alternating order.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paired_bench import child_env, machine_state  # noqa: E402

CHILD = r'''
import json, sys, time
sys.path.insert(0, sys.argv[4])
import numpy as np
E = np.load(sys.argv[2]).astype(np.float32)
if sys.argv[1] == "arsc":
    import arsc
    t = time.time(); arsc.find_neighbors(E.astype(np.float64), k=20); s = time.time() - t
else:
    import anndata as ad, pandas as pd, scanpy as sc
    import os; sc.settings.n_jobs = int(os.environ["AR_THREADS"])
    a = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(E.shape[0])])); a.obsm["X_pca"] = E
    t = time.time(); sc.pp.neighbors(a, n_neighbors=20, use_rep="X_pca"); s = time.time() - t
open(sys.argv[3], "w").write(json.dumps({"seconds": s}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    emb = HERE / ".bench_data" / f"n{a.n}" / "common" / "pca.npy"
    runs = []
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "child.py"
        f.write_text(CHILD)
        for rep in range(a.repeats):
            for tool in (("arsc", "scanpy") if rep % 2 == 0 else ("scanpy", "arsc")):
                st = machine_state()
                out = Path(td) / "o.json"
                subprocess.run([sys.executable, str(f), tool, str(emb), str(out), str(HERE.parent / "src")],
                               env=child_env(a.threads), check=True, capture_output=True)
                r = json.loads(out.read_text())
                r.update(tool=tool, rep=rep, n=a.n, threads=a.threads, **st)
                runs.append(r)
                print(f"rep {rep} {tool:6s} first call {r['seconds']:.1f} s (cpu {st['cpu_percent_1s']:.0f}%)", flush=True)
    summ = {t: float(np.median([r["seconds"] for r in runs if r["tool"] == t])) for t in ("arsc", "scanpy")}
    print(json.dumps(summ))
    (HERE / "results" / f"cold_start_n{a.n}.json").write_text(json.dumps({"median_s": summ, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
