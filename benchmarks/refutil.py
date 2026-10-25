"""Helpers for the validation/benchmark scripts: load R reference outputs, time things."""
from __future__ import annotations

import gzip
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# R reference outputs and exported stores live outside the repo; set ARSC_SCRATCH (default: <repo>/../../scratch/arsc)
SCRATCH = Path(os.environ.get("ARSC_SCRATCH", Path(__file__).resolve().parents[3] / "scratch" / "arsc"))
REF_GD = SCRATCH / "ref_gd"
REF_PBMC = SCRATCH / "ref_pbmc"
EXP_GD = SCRATCH / "exp_gd"
EXP_PBMC = SCRATCH / "exp_pbmc"
RESULTS = Path(__file__).with_name("results")
# R harmony references (benchmarks/r_reference_harmony_all.R; about 115 MB, gzip float32). Point ARSC_REF_HARMONY at a
# folder no scratch cleanup prunes (tests/local.env); the default under ARSC_SCRATCH is for a throwaway setup.
REF_HARMONY = Path(os.environ.get("ARSC_REF_HARMONY", SCRATCH / "ref_harmony"))


def read_dense(prefix) -> pd.DataFrame:
    """Read a matrix written by r_common.R::write_dense (<prefix>.f64) or write_dense_gz (<prefix>.f64.gz or
    <prefix>.f32.gz)."""
    prefix = str(prefix)
    meta = json.loads(Path(prefix + ".json").read_text())
    nr, nc = meta["dim"]
    if Path(prefix + ".f64").exists():
        a = np.fromfile(prefix + ".f64", dtype="<f8").reshape(nc, nr).T
    else:
        ext, dt = (".f32.gz", "<f4") if Path(prefix + ".f32.gz").exists() else (".f64.gz", "<f8")
        with gzip.open(prefix + ext, "rb") as f:
            a = np.frombuffer(f.read(), dtype=dt).astype(np.float64).reshape(nc, nr).T
    return pd.DataFrame(a, index=meta.get("rownames") or None, columns=meta.get("colnames") or None)


def read_dgc(prefix, shape) -> sp.csc_matrix:
    i = np.fromfile(f"{prefix}.i.bin", dtype="<i4")
    p = np.fromfile(f"{prefix}.p.bin", dtype="<i4")
    x = np.fromfile(f"{prefix}.x.bin", dtype="<f8")
    return sp.csc_matrix((x, i, p), shape=shape)


def r_timings(ref_dir) -> dict:
    return json.loads((Path(ref_dir) / "timings_r.json").read_text())


TIMINGS: dict[str, float] = {}


@contextmanager
def timer(name: str, store: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    (TIMINGS if store is None else store)[name] = dt
    print(f"[timing] {name:<34s} {dt:8.2f} s", flush=True)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
