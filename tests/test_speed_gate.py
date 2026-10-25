"""Speed regression gate: arsc must not be slower than scanpy on any step both implement.

Runs ``benchmarks/paired_bench.py --gate`` (arsc and scanpy interleaved in the same session, alternating order,
the same thread count, median of 3 runs each) on 20,000 simulated cells, so the comparison never depends on a
timing stored from another machine or another load. It takes 20-40 minutes, so it runs only when asked:

    ARSC_SPEED_GATE=1 python -m pytest -q tests/test_speed_gate.py      # optional: ARSC_SPEED_GATE_SIZES=20000,100000

Run it before merging any change to a compute kernel and before a release (CONTRIBUTING.md).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.bench,
              pytest.mark.skipif(os.environ.get("ARSC_SPEED_GATE") != "1",
                                 reason="speed gate runs only with ARSC_SPEED_GATE=1 (20-40 min)")]


def test_arsc_not_slower_than_scanpy(tmp_path):
    pytest.importorskip("scanpy")
    pytest.importorskip("pydeseq2")
    root = Path(__file__).resolve().parents[1]
    threads = os.environ.get("AR_THREADS", "4")
    cmd = [sys.executable, str(root / "benchmarks" / "paired_bench.py"), "--gate",
           "--sizes", os.environ.get("ARSC_SPEED_GATE_SIZES", "20000"), "--repeats", "3", "--threads", threads,
           "--out", str(tmp_path / "paired")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-6000:])
    assert r.returncode == 0, "arsc slower than scanpy on some step:\n" + r.stdout[-3000:] + r.stderr[-2000:]
