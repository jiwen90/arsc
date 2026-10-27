"""Seurat's FindAllMarkers on a large subset, split over several single-threaded R processes.

    ARSC_BENCH_STORE=<store> python benchmarks/seurat_markers_shard.py --n 1000000 --shards 4 --shard 0

``FindAllMarkers`` is ``FindMarkers(ident.1 = g)`` for every cell type ``g`` (with at least 3 cells) in turn, each on
one thread. At 1,000,000 cells that takes longer than a day in one process, so the cell types are dealt to
``--shards`` processes (largest first, each to the shard with the fewest cells so far) and each process times its
own ``FindMarkers`` calls (``r_seurat_markers_shard.R``). The sum over all shards is the single-threaded total, a
measurement rather than an extrapolation; running the shards side by side shares memory bandwidth, so each call may be
slower than it would be alone (the machine's CPU use is sampled and recorded). Same cells as ``seurat_timing.py``
(``benchdata.Bench``, seed 0); nothing is written to disk but ``results/seurat/seurat_markers_n<n>_shard<i>of<k>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


def deal(counts: dict, k: int) -> list[list[str]]:
    """Cell types (>= 3 cells) to k shards: largest first, each to the shard with the fewest cells so far."""
    shards, load = [[] for _ in range(k)], [0] * k
    for g, c in sorted(((g, c) for g, c in counts.items() if c >= 3), key=lambda t: (-t[1], t[0])):
        i = int(np.argmin(load))
        shards[i].append(g)
        load[i] += c
    return shards


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--rscript", default=os.environ.get("ARSC_RSCRIPT", "Rscript"))
    ap.add_argument("--out", default=str(HERE / "results" / "seurat"))
    a = ap.parse_args()
    import arsc
    from benchdata import Bench
    from seurat_timing import LoadSampler
    store = os.environ.get("ARSC_BENCH_STORE", "")
    if not store or not Path(store).is_dir():
        raise SystemExit("set ARSC_BENCH_STORE to an arsc store directory")
    B = Bench(f"store:{store}", a.n, a.seed)
    ct = B.obs["cell_type"].astype(str)
    n_store = len(arsc.open(store).obs)
    rows = (np.sort(np.random.default_rng(a.seed).choice(n_store, a.n, replace=False)) if a.n < n_store
            else np.arange(n_store))   # the rows Bench took (as seurat_timing.py)
    groups = deal(ct.value_counts().to_dict(), a.shards)[a.shard]
    print(f"shard {a.shard} of {a.shards}: {len(groups)} cell types, {int(ct.isin(groups).sum()):,} cells", flush=True)
    spec = {"store": str(store), "rows": rows.tolist(), "cell_type": ct.tolist(), "groups": groups}
    del B
    load = LoadSampler().start()
    t0 = time.perf_counter()
    p = subprocess.run([a.rscript, str(HERE / "r_seurat_markers_shard.R")], input=json.dumps(spec),
                       stdout=subprocess.PIPE, text=True, creationflags=0x08000000 if os.name == "nt" else 0)
    wall = time.perf_counter() - t0
    if p.returncode != 0 or "@@RESULT@@" not in p.stdout:
        sys.stderr.write(p.stdout[-4000:])
        raise SystemExit(f"Rscript failed (exit {p.returncode})")
    res = json.loads(p.stdout.split("@@RESULT@@", 1)[1].strip())
    res.update({"shard": a.shard, "shards": a.shards, "groups": groups, "seed": a.seed, "process_wall_s": wall,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "machine_load_during_r": load.stop(),
                "groups_total": int((ct.value_counts() >= 3).sum())})
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    f = od / f"seurat_markers_n{a.n}_shard{a.shard}of{a.shards}.json"
    f.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
