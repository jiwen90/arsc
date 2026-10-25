"""Paired timing of a Symphony reference build + query mapping + kNN label transfer: arsc vs R symphony on the
package's vignette data (750 reference / 450 query cells; run benchmarks/r_reference_symphony.R first).

    python benchmarks/symphony_timing.py --repeats 3 --threads 2

Each run is a fresh process that does one untimed warm-up (numba compilation, R package loading) and one timed
run; R and arsc alternate (repeat 0: arsc first; repeat 1: R first; ...), with the same thread count in the
thread variables. The machine state is recorded before each run. Writes ``results/symphony_timing.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paired_bench import child_env, machine_state  # noqa: E402

D = Path((os.environ.get("ARSC_REFS") or os.environ.get("ARSC_SCRATCH", "."))) / "symphony_ref"

PY_CHILD = r'''
import json, os, sys, time
sys.path.insert(0, sys.argv[3])
import numpy as np, pandas as pd, scipy.io
from arsc import symphony as sy
D = sys.argv[1]
X = scipy.io.mmread(os.path.join(D, "pbmcs_exprs.mtx")).tocsr().T.tocsr()
genes = pd.Index(open(os.path.join(D, "pbmcs_genes.txt")).read().split())
meta = pd.read_csv(os.path.join(D, "pbmcs_meta.csv"), index_col=0)
isref = meta["donor"].isin(["3pv1", "3pv2"]).to_numpy()
def run():
    ref = sy.build_reference(X[isref], genes, meta[isref], vars="donor", normalize=False, topn=2000, d=20, K=100, umap=True)
    q = sy.map_query(X[~isref], genes, meta[~isref], ref, vars="donor", normalize=False, umap=True)
    return sy.transfer_labels(ref, q, "cell_type", k=5)
t0 = time.time(); run(); w = time.time() - t0
t0 = time.time(); run(); s = time.time() - t0
open(sys.argv[2], "w").write(json.dumps({"ok": True, "seconds": s, "warmup_seconds": w}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("AR_THREADS", "2")))
    a = ap.parse_args()
    runs = []
    src = str(HERE.parent / "src")
    with tempfile.TemporaryDirectory() as td:
        pyf = Path(td) / "child.py"
        pyf.write_text(PY_CHILD)
        for rep in range(a.repeats):
            for tool in (("arsc", "R") if rep % 2 == 0 else ("R", "arsc")):
                st = machine_state()
                outf = Path(td) / f"{tool}_{rep}.json"
                if tool == "arsc":
                    cmd = [sys.executable, str(pyf), str(D), str(outf), src]
                else:
                    cmd = [os.environ.get("RSCRIPT", "Rscript"), str(HERE / "r_symphony_timing.R"), str(D), str(outf)]
                subprocess.run(cmd, env=child_env(a.threads), capture_output=True, check=True)
                r = json.loads(outf.read_text())
                r.update(tool=tool, rep=rep, threads=a.threads, **st)
                runs.append(r)
                print(f"rep {rep} {tool:4s} {r['seconds']:.2f} s (warm-up {r['warmup_seconds']:.1f} s; cpu "
                      f"{st['cpu_percent_1s']:.0f}%; {st.get('lease')})", flush=True)
    summ = {t: {"median_s": float(np.median([r["seconds"] for r in runs if r["tool"] == t])),
                "min_s": float(min(r["seconds"] for r in runs if r["tool"] == t)),
                "max_s": float(max(r["seconds"] for r in runs if r["tool"] == t))} for t in ("arsc", "R")}
    print(json.dumps(summ, indent=1))
    out = HERE / "results" / "symphony_timing.json"
    out.write_text(json.dumps({"summary": summ, "runs": runs, "date": time.strftime("%Y-%m-%d")}, indent=1))


if __name__ == "__main__":
    main()
