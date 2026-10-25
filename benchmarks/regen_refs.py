"""Rebuild the stored R reference outputs that the parity tests compare against, or list what is there.

    python benchmarks/regen_refs.py status                 # every set: folder, files, size, what reads it
    python benchmarks/regen_refs.py ref_gd stats_parity    # rebuild those sets
    python benchmarks/regen_refs.py all                    # rebuild every set (about 1 h; harmony is most of it)

Only R outputs are stored, under $ARSC_REFS/<set> (benchmarks/refutil.py; in autoresearch <AR_HOME>/.cache/arsc, which
no scratch cleanup prunes). Inputs are never stored: Python reads the shared input store ($ARSC_STORE_GD, the
38,440-cell gamma-delta T-cell arsc store, memory-mapped) and seeded synthetic tables built in memory; R gets the
Seurat object the store was converted from ($ARSC_GD_RDS) or a temporary export of the same tables, deleted after the
run. Sets that cannot use the shared input say why below.

Environment (tests/local.env is read too): ARSC_REFS, ARSC_STORE_GD, ARSC_GD_RDS, ARSC_RLIB (extra R library, for
dbscan and transport), RSCRIPT (default Rscript), CLONOTRACE_SRC (a Clonotrace checkout; downloaded at the pinned
commit into $ARSC_REFS/src when unset), ARSC_TEST_CLONES_META and ARSC_DATA_V3 (clones_ref).
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _local_env():
    f = HERE.parent / "tests" / "local.env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_local_env()
sys.path.insert(0, str(HERE))
import refutil as U  # noqa: E402

CLONOTRACE_COMMIT = "f0c397a"  # Clonotrace v1.2.1 (CLONOTRACE_PARITY.md)

# set -> (input, files the tests read)
SETS = {
    "ref_gd": ("shared store's Seurat object ($ARSC_GD_RDS)", "tests/test_vs_r.py (LogNormalize, HVG, presto)"),
    "ref_extra": ("shared store's Seurat object ($ARSC_GD_RDS); needs ref_gd", "tests/test_seurat_extra.py"),
    "ref_harmony": ("shared store's PCA + batch columns (temporary export)", "tests/test_vs_r.py (harmony)"),
    "stats_parity": ("shared store's pseudobulk + composition (temporary export)", "tests/test_stats_de.py"),
    "stats_resampling": ("seeded synthetic tables (temporary export)", "tests/test_stats_resampling.py"),
    "clones_ref": ("CD8 clone metadata of the melanoma ICB dataset (needs TCR clones and the published Cyclone "
                   "assignment, which the gamma-delta store lacks)", "tests/test_clones.py"),
    "clonotrace_ref": ("the public demo bundled with Clonotrace (lineage-traced cells, which the shared store "
                       "lacks)", "tests/test_clonotrace.py"),
    "symphony_ref": ("the R symphony package's vignette data and seeded random inputs (the published "
                     "reference/query case)", "tests/test_symphony.py"),
}


def _dir(name: str) -> Path:
    return {"ref_harmony": U.REF_HARMONY, "clonotrace_ref": U.CLONOTRACE_REF}.get(name, U.REFS / name)


def _rscript(script: str, *args, env: dict | None = None):
    exe = os.environ.get("RSCRIPT", "Rscript")
    e = dict(os.environ, ARSC_REFS=str(U.REFS), **(env or {}))
    print(f"== Rscript {script} {' '.join(map(str, args))}", flush=True)
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    r = subprocess.run([exe, str(HERE / script), *map(str, args)], env=e, **kw)
    if r.returncode:
        raise SystemExit(f"{script} failed ({r.returncode})")


def _fresh(d: Path) -> Path:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _need_store():
    if U.store_gd_missing():
        raise SystemExit(U.store_gd_missing())


def _rds() -> str:
    rds = os.environ.get("ARSC_GD_RDS", "")
    if not Path(rds).is_file():
        raise SystemExit(f"ARSC_GD_RDS={rds!r} is not a file: the Seurat object the shared store was converted from")
    return rds


def ref_gd():
    _rscript("r_reference_gd.R", _rds(), _fresh(_dir("ref_gd")))


def ref_extra():
    if not (_dir("ref_gd") / "variable_features.txt").exists():
        ref_gd()
    _rscript("r_reference_seurat_extra.R", _rds(), _fresh(_dir("ref_extra")))


def ref_harmony():
    _need_store()
    from harmony_input import build, write
    root = _dir("ref_harmony")
    root.mkdir(parents=True, exist_ok=True)
    inp = write(root / "input", *build(U.STORE_GD))
    try:
        _rscript("r_reference_harmony_all.R", env={"ARSC_REF_HARMONY": str(root)})
    finally:
        shutil.rmtree(inp, ignore_errors=True)


def stats_parity():
    _need_store()
    import stats_parity_inputs
    from arsc import de
    out = _fresh(_dir("stats_parity"))
    tmp = stats_parity_inputs.write(U.REFS / ".inputs_stats_parity")
    try:
        _rscript("r_reference_stats_parity.R", tmp, out, de.GENESET_CACHE)
        _rscript("r_reference_stats_small.R", out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stats_resampling():
    import stats_resampling_cases
    out = _fresh(_dir("stats_resampling"))
    tmp = _fresh(U.REFS / ".inputs_stats_resampling")
    try:
        stats_resampling_cases.main(tmp)
        _rscript("r_reference_stats_resampling.R", tmp)
        for f in tmp.glob("R_*.csv"):
            shutil.move(str(f), out / f.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _gzip_csvs(d: Path):
    import gzip
    for f in d.glob("*.csv"):
        with open(f, "rb") as a, gzip.open(f.with_name(f.name + ".gz"), "wb", compresslevel=9) as b:
            shutil.copyfileobj(a, b)
        f.unlink()


def clones_ref():
    out = _fresh(_dir("clones_ref"))
    _rscript("r_reference_clones.R", out)
    _gzip_csvs(out)  # tests/test_clones.py reads <name>.csv.gz


def _clonotrace_src() -> Path:
    src = os.environ.get("CLONOTRACE_SRC")
    if src:
        return Path(src)
    dest = U.REFS / "src" / f"Clonotrace-{CLONOTRACE_COMMIT}"
    if not (dest / "R").exists():
        url = f"https://codeload.github.com/yuntianf/Clonotrace/tar.gz/{CLONOTRACE_COMMIT}"
        print("== downloading", url, flush=True)
        data = urllib.request.urlopen(url, timeout=300).read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            top = t.getmembers()[0].name.split("/")[0]
            keep = [m for m in t.getmembers() if m.name.startswith((f"{top}/R/", f"{top}/inst/extdata/vignette_demo"))]
            t.extractall(dest.parent / ".clonotrace_tmp", members=keep, filter="data")
        shutil.move(str(dest.parent / ".clonotrace_tmp" / top), dest)
        shutil.rmtree(dest.parent / ".clonotrace_tmp", ignore_errors=True)
    return dest


def clonotrace_ref():
    out = _dir("clonotrace_ref")
    _fresh(out)
    # the R script writes to $ARSC_REFS/clonotrace_ref; an ARSC_TEST_CLONOTRACE_REF elsewhere gets a moved copy
    _rscript("r_reference_clonotrace.R", env={"CLONOTRACE_SRC": str(_clonotrace_src())})
    default = U.REFS / "clonotrace_ref"
    if out.resolve() != default.resolve():
        shutil.rmtree(out)
        shutil.move(str(default), out)


def symphony_ref():
    _rscript("r_reference_symphony.R", _fresh(_dir("symphony_ref")))


def _size(d: Path) -> tuple[int, int]:
    fs = [f for f in d.rglob("*") if f.is_file()] if d.exists() else []
    return len(fs), sum(f.stat().st_size for f in fs)


def status():
    print(f"ARSC_REFS = {U.REFS}\nARSC_STORE_GD = {U.STORE_GD} ({'ok' if U.have_store_gd() else 'MISSING'})")
    tot = 0
    for name, (inp, used) in SETS.items():
        d = _dir(name)
        n, b = _size(d)
        tot += b
        print(f"{name:<17} {n:>4} files {b / 2**20:8.1f} MB  {d}\n{'':17} input: {inp}\n{'':17} read by: {used}")
    print(f"total {tot / 2**20:.1f} MB")


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv == ["status"]:
        status()
        return
    names = list(SETS) if argv == ["all"] else argv
    bad = [n for n in names if n not in SETS]
    if bad:
        raise SystemExit(f"unknown set(s) {bad}; choose from {list(SETS)}")
    for n in names:
        globals()[n]()
    status()


if __name__ == "__main__":
    main(sys.argv[1:])
