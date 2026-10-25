"""arsc.run_harmony vs R harmony (1.2.x, 2.0.x, 2.1.x) on the references of benchmarks/r_reference_harmony.R.

Usage: python benchmarks/harmony_parity.py [ref_root] [--versions 1.2.0,2.0.5,2.1.0] [--cases lane1,mv2,...]
(ref_root default: $ARSC_REF_HARMONY, rebuilt by benchmarks/r_reference_harmony_all.R)

For each version x case, runs arsc twice: from its own k-means start (seed 1) and from R's start (``init_centroids``
= R's Y0), and reports the per-dimension Pearson r with R's corrected embedding (min / median over all dims and
the first 10), the same for R run with a second seed (R's own spread), the objective trajectories and the
clustering rounds. Writes benchmarks/results/harmony_parity.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refutil import REF_HARMONY, RESULTS, read_dense, save_json  # noqa: E402

import arsc  # noqa: E402

CASES = ["lane1", "mv2", "mv2_tl", "mv2_auto", "mv3"]


def load_input(root: Path):
    Z = read_dense(root / "input" / "pcs").values
    mf = root / "input" / "meta.csv"
    meta = pd.read_csv(mf if mf.exists() else root / "input" / "meta.csv.gz", dtype=str)
    return Z, meta


def r_kwargs(tr: dict) -> dict:
    """run_harmony arguments matching a reference case (lambda 'default' = the version's own default)."""
    th = tr.get("theta")
    kw = {"theta": th if isinstance(th, (list, int, float)) and th != [] else None}  # R writes NULL as {}
    lam = tr["lambda"]
    if lam == "auto":
        kw["lambda_"] = None
    elif lam != "default":
        kw["lambda_"] = lam
    return kw


def per_dim_r(a, b):
    return np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(b.shape[1])])


def compare(root: Path, ver: str, case: str, Z, meta, init: str):
    pre = root / ver / case
    tr = json.loads(Path(f"{pre}_trace.json").read_text())
    ref = read_dense(f"{pre}_Z").values
    vars_ = tr["vars"] if isinstance(tr["vars"], list) else [tr["vars"]]
    kw = r_kwargs(tr)
    if init == "r":
        kw["init_centroids"] = read_dense(f"{pre}_Y0").values
    t0 = time.perf_counter()
    res = arsc.run_harmony(Z, meta, vars_, version=ver, seed=1, return_object=True, **kw)
    dt = time.perf_counter() - t0
    r = np.abs(per_dim_r(res.Z_corr, ref))
    s2 = Path(f"{pre}_seed2_Z.json")
    rr = np.abs(per_dim_r(read_dense(f"{pre}_seed2_Z").values, ref)) if s2.exists() else np.full(1, np.nan)
    return {"version": ver, "case": case, "init": init, "vars": vars_, "theta": kw["theta"] if "theta" in kw else None,
            "lambda": tr["lambda"], "r_min": float(r.min()), "r_median": float(np.median(r)),
            "r_min_first10": float(r[:10].min()), "r_seed2_min": float(rr.min()), "r_seed2_median": float(np.median(rr)), "obj_r": [round(x, 3) for x in tr["objective_harmony"]],
            "obj_arsc": [round(x, 3) for x in res.objective_harmony], "rounds_r": tr["kmeans_rounds"],
            "rounds_arsc": res.kmeans_rounds, "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(REF_HARMONY))
    ap.add_argument("--versions", default="1.2.0,2.0.5,2.1.0")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--inits", default="own,r")
    a = ap.parse_args()
    root = Path(a.root)
    Z, meta = load_input(root)
    rows = []
    for ver in a.versions.split(","):
        for case in a.cases.split(","):
            for init in a.inits.split(","):
                row = compare(root, ver, case, Z, meta, init)
                rows.append(row)
                print(f"{ver} {case:9s} init={init:3s} r min {row['r_min']:.4f} median {row['r_median']:.4f} "
                      f"(R seed 2 vs 1: min {row['r_seed2_min']:.4f} median {row['r_seed2_median']:.4f}) obj R {row['obj_r'][-1]} arsc {row['obj_arsc'][-1]} "
                      f"rounds R {len(row['rounds_r'])} arsc {len(row['rounds_arsc'])} {row['seconds']} s", flush=True)
    save_json(rows, RESULTS / "harmony_parity.json")


if __name__ == "__main__":
    main()
