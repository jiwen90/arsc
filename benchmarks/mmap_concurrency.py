"""Validate that N processes opening the same arsc store share one physical copy.

Each worker opens the store, touches every byte of the RNA counts matrix
(data + indices + indptr -> all pages resident) plus obs, then holds while the
parent samples per-process memory and system-wide available memory.

Modes:
  mmap  -- arsc.open(): np.load(mmap_mode="r") -> file-backed shared pages
  copy  -- np.load() without mmap (what an ordinary loader does): private copies

Usage: python benchmarks/mmap_concurrency.py [store_dir] [--procs 1,3]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

STORE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else str(Path(os.environ.get("ARSC_SCRATCH", "scratch")) / "exp_pbmc")
GB = 1024 ** 3


def worker(store: str, mode: str) -> None:
    import numpy as np
    import arsc
    if mode == "mmap":
        st = arsc.open(store)
        X = st.X("RNA", "counts")
        data, indices, indptr = X.data, X.indices, X.indptr
        _ = st.obs
    else:
        base = Path(store) / "assays" / "RNA" / "counts.csr"
        data = np.load(f"{base}.data.npy")
        indices = np.load(f"{base}.indices.npy")
        indptr = np.load(f"{base}.indptr.npy")
    # touch every page
    s = float(np.asarray(data, dtype=np.float32).sum(dtype=np.float64)) + float(indices.sum(dtype=np.int64)) + float(indptr[-1])
    mi = psutil.Process().memory_full_info()  # measured in-process (venv python.exe is a launcher)
    print(json.dumps({"ready": True, "checksum": s, "rss_GB": mi.rss / GB, "private_GB": mi.private / GB,
                      "uss_GB": mi.uss / GB}), flush=True)
    sys.stdin.readline()  # hold until parent says stop


def measure(n: int, mode: str) -> dict:
    vm0 = psutil.virtual_memory().available
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", STORE, mode], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) for _ in range(n)]
    t0 = time.perf_counter()
    per = []
    for p in procs:
        rec = json.loads(p.stdout.readline())
        assert rec["ready"]
        per.append({k: rec[k] for k in ("rss_GB", "private_GB", "uss_GB")})
    t_ready = time.perf_counter() - t0
    time.sleep(1.0)
    vm1 = psutil.virtual_memory().available
    for p in procs:
        p.stdin.write("\n"); p.stdin.flush(); p.wait()
    return {"mode": mode, "procs": n, "time_to_ready_s": t_ready,
            "per_process": per,
            "sum_rss_GB": sum(x["rss_GB"] for x in per),
            "sum_private_GB": sum(x["private_GB"] for x in per),
            "sum_uss_GB": sum(x["uss_GB"] for x in per),
            "system_available_drop_GB": (vm0 - vm1) / GB}


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i + 1], sys.argv[i + 2])
        sys.exit(0)
    procs = [1, 3]
    if "--procs" in sys.argv:
        procs = [int(x) for x in sys.argv[sys.argv.index("--procs") + 1].split(",")]
    size = sum(f.stat().st_size for f in (Path(STORE) / "assays" / "RNA").glob("counts.csr.*.npy")) / GB
    print(f"store {STORE}: RNA counts CSR = {size:.2f} GB on disk")
    out = {"store": STORE, "rna_counts_GB": size, "runs": []}
    for mode in ("mmap", "copy"):
        for n in procs:
            r = measure(n, mode)
            out["runs"].append(r)
            print(f"{mode:4s} x{n}: ready {r['time_to_ready_s']:.1f}s | sum RSS {r['sum_rss_GB']:.2f} GB | "
                  f"sum private {r['sum_private_GB']:.2f} GB | sum USS {r['sum_uss_GB']:.2f} GB | system available drop {r['system_available_drop_GB']:.2f} GB")
    res = Path(__file__).with_name("results")
    res.mkdir(exist_ok=True)
    (res / "mmap_concurrency.json").write_text(json.dumps(out, indent=2))
