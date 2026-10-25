"""Speed regression gate: arsc must not be slower than scanpy on any step both implement.

Runs ``benchmarks/paired_bench.py --gate`` (arsc and scanpy interleaved in the same session, alternating order,
the same thread count, median of 3 runs each) on 10,000 toy cells generated in memory (or on a real store read
memory-mapped when ``ARSC_BENCH_STORE`` is set), so the comparison never depends on a timing stored from another
machine or another load, and no benchmark data is written anywhere. One process times both tools, so it takes
a few minutes; it runs only when asked:

    ARSC_SPEED_GATE=1 python -m pytest -q tests/test_speed_gate.py      # optional: ARSC_SPEED_GATE_SIZES=20000,100000

Run it before merging any change to a compute kernel and before a release (CONTRIBUTING.md).

``test_paired_bench_writes_nothing_under_benchmarks`` always runs (a 3,000-cell, one-repeat pass of two steps):
the benchmark tooling must keep its data in memory and write only the small result tables to ``--out``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*") if p.is_file()}


def _env(threads):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("ARSC_BENCH_STORE", None)
    for v in ("AR_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = str(threads)
    return env


def test_paired_bench_writes_nothing_under_benchmarks(tmp_path):
    pytest.importorskip("scanpy")
    bench = ROOT / "benchmarks"
    before = _snapshot(bench)
    cmd = [sys.executable, str(bench / "paired_bench.py"), "--sizes", "3000", "--steps", "normalize,neighbors",
           "--repeats", "1", "--threads", "2", "--source", "toy", "--out", str(tmp_path / "paired")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(2), timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    after = _snapshot(bench)
    new = sorted(set(after) - set(before))
    changed = sorted(p for p in set(after) & set(before) if after[p] != before[p])
    assert not new and not changed, f"benchmark run wrote under benchmarks/: new {new[:10]}, changed {changed[:10]}"
    assert not (bench / ".bench_data").exists()
    out = sorted(p.name for p in (tmp_path / "paired").iterdir())
    assert "runs.jsonl" in out and "summary.csv" in out
    assert all(p.suffix in (".jsonl", ".csv") for p in (tmp_path / "paired").iterdir()), out


@pytest.mark.bench
@pytest.mark.skipif(os.environ.get("ARSC_SPEED_GATE") != "1",
                    reason="speed gate runs only with ARSC_SPEED_GATE=1")
def test_arsc_not_slower_than_scanpy(tmp_path):
    pytest.importorskip("scanpy")
    pytest.importorskip("pydeseq2")
    threads = os.environ.get("AR_THREADS", "4")
    bench = ROOT / "benchmarks"
    before = _snapshot(bench)
    cmd = [sys.executable, str(bench / "paired_bench.py"), "--gate",
           "--sizes", os.environ.get("ARSC_SPEED_GATE_SIZES", "10000"), "--repeats", "3", "--threads", threads,
           "--out", str(tmp_path / "paired")]
    env = _env(threads)
    if os.environ.get("ARSC_BENCH_STORE"):
        env["ARSC_BENCH_STORE"] = os.environ["ARSC_BENCH_STORE"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(r.stdout[-6000:])
    assert r.returncode == 0, "arsc slower than scanpy on some step:\n" + r.stdout[-3000:] + r.stderr[-2000:]
    assert _snapshot(bench) == before, "the speed gate wrote under benchmarks/"
