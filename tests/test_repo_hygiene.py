"""Repository hygiene: no dataset is tracked by git, and no test reference lives in a scratch folder.

1. Datasets are never committed at any size (toy, synthetic, R reference outputs included): they are generated in
   memory or kept git-ignored outside the repository, with the script that regenerates them committed instead
   (benchmarks/regen_refs.py). Only the measured benchmark summaries under benchmarks/results/ are tracked.
2. Reference paths must not resolve under a ``scratch`` folder: autoresearch prunes scratch/ after 2 days, which once
   removed most references and left the parity tests skipping without anyone noticing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
import refutil  # noqa: E402

DATA_EXT = (".npy", ".npz", ".h5ad", ".h5", ".hdf5", ".h5seurat", ".rds", ".rda", ".rdata", ".qs", ".parquet",
            ".feather", ".arrow", ".mtx", ".loom", ".f64", ".f32", ".bin", ".pkl", ".pickle", ".joblib", ".gz", ".bz2",
            ".xz", ".zip", ".tar", ".csv", ".tsv", ".xlsx", ".fcs")
# measured benchmark summaries (timings and agreement tables; benchmarks/README.md) are results, not datasets
ALLOWED_PREFIXES = ("benchmarks/results/",)


def _under_scratch(p: str | os.PathLike) -> bool:
    parts = [x.lower() for x in Path(os.path.expandvars(str(p))).resolve().parts]
    return any(x == "scratch" or x.startswith("scratch_") for x in parts)


def _local_env() -> dict[str, str]:
    f = REPO / "tests" / "local.env"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def test_no_reference_path_under_scratch():
    env = _local_env()
    env.update({k: v for k, v in os.environ.items() if k.startswith("ARSC_") and v})
    paths = {k: v for k, v in env.items() if ("/" in v or "\\" in v)}
    resolved = {"refutil.REFS": refutil.REFS, "refutil.REF_HARMONY": refutil.REF_HARMONY,
                "refutil.STORE_GD": refutil.STORE_GD, "refutil.CLONOTRACE_REF": refutil.CLONOTRACE_REF}
    bad = {k: v for k, v in {**paths, **resolved}.items() if _under_scratch(v)}
    assert not bad, f"reference paths under a scratch folder (pruned after 2 days); move them, e.g. to .cache: {bad}"


def test_under_scratch_detects_scratch():
    assert _under_scratch("E:/autoresearch/scratch/arsc/ref_gd")
    assert not _under_scratch("E:/autoresearch/.cache/arsc/ref_gd")


def _tracked() -> list[str]:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (git ls-files failed)")
    return [f for f in r.stdout.decode("utf-8").split("\0") if f]


def is_data_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(DATA_EXT) and not p.startswith(ALLOWED_PREFIXES)


def test_no_data_files_tracked():
    bad = [f for f in _tracked() if is_data_file(f)]
    assert not bad, (f"data files are tracked by git: {bad[:20]}. Datasets are never committed: git rm --cached "
                     "them, keep them outside the repo and commit the script that regenerates them")


def test_is_data_file():
    assert is_data_file("tests/data/x.npz") and is_data_file("benchmarks/ref/R_small.csv")
    assert not is_data_file("benchmarks/results/suite/summary.csv") and not is_data_file("tests/test_x.py")
