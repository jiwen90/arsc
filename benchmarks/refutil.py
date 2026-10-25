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

# Where the test references live (benchmarks/regen_refs.py rebuilds every one; benchmarks/README.md):
#   ARSC_REFS       root of the stored R reference outputs, one folder per set (ref_gd, ref_extra, stats_parity, ...).
#                   Keep it where no scratch cleanup prunes it (autoresearch: <AR_HOME>/.cache/arsc). ARSC_SCRATCH is
#                   the old name and is still read; the default is benchmarks/.refs (git-ignored).
#   ARSC_STORE_GD   the ONE shared input: the 38,440-cell gamma-delta T-cell arsc store (registered, read-only,
#                   memory-mapped, never copied). Python tests read it directly; the R references are computed from
#                   the Seurat object it was converted from (ARSC_GD_RDS), so both sides see the same cells.
#   ARSC_REF_HARMONY / ARSC_TEST_CLONOTRACE_REF override a single set's folder.
REPO = Path(__file__).resolve().parents[1]
REFS = Path(os.environ.get("ARSC_REFS") or os.environ.get("ARSC_SCRATCH") or REPO / "benchmarks" / ".refs")
SCRATCH = REFS  # old name, kept for scripts that still import it
STORE_GD = Path(os.environ.get("ARSC_STORE_GD") or REFS / "exp_gd")
EXP_GD = STORE_GD  # old name
REF_GD = REFS / "ref_gd"
REF_EXTRA = REFS / "ref_extra"
REF_PBMC = REFS / "ref_pbmc"
EXP_PBMC = REFS / "exp_pbmc"
STATS_PARITY = REFS / "stats_parity"
STATS_RESAMPLING = REFS / "stats_resampling"
CLONES_REF = REFS / "clones_ref"
SYMPHONY_REF = REFS / "symphony_ref"
CLONOTRACE_REF = Path(os.environ.get("ARSC_TEST_CLONOTRACE_REF") or REFS / "clonotrace_ref")
RESULTS = Path(__file__).with_name("results")
# R harmony references (benchmarks/r_reference_harmony_all.R; about 105 MB, gzip float32)
REF_HARMONY = Path(os.environ.get("ARSC_REF_HARMONY") or REFS / "ref_harmony")


def have_store_gd() -> bool:
    return (STORE_GD / "manifest.json").exists()


def missing(*files, regen: str) -> str | None:
    """Skip reason naming every missing reference file (and how to rebuild it), or None when all exist.

    >>> missing(REFS, regen="x") is None or "missing" in missing(REFS, regen="x")
    True
    """
    gone = [str(f) for f in files if not Path(f).exists()]
    if not gone:
        return None
    return f"missing {', '.join(gone)} (rebuild: python benchmarks/regen_refs.py {regen})"


def store_gd_missing() -> str | None:
    if have_store_gd():
        return None
    return (f"missing the shared input store {STORE_GD} (set ARSC_STORE_GD to the 38,440-cell gamma-delta T-cell "
            "arsc store; tests/local.env)")


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
