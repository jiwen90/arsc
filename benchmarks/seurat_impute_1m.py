"""Seurat at 1,000,000 cells: the measured steps of an unfinished run, plus imputed values for the rest.

    python benchmarks/seurat_impute_1m.py

Writes results/seurat/seurat_timing_n1000000_seurat.json, which render_readme.py reads like any other
``seurat_timing_n<size>_seurat.json`` (Seurat 4.3 defaults: one BLAS thread, one ``future`` worker).

The one-thread Seurat run at 1M cells (seurat_timing.py --r-only) was stopped before it finished, so it wrote no
results file. The times below are copied from its log and from the logs of seurat_markers_shard.py (4 shards, started
side by side; only shard 0 finished and wrote a results file):

* measured: normalize, hvg, pca, neighbors, cluster, and ``FindMarkers`` for 16 of the 36 cell types (cd4_tfh ran
  twice, 485.8 s alone and 481.2 s in a shard; the first is used);
* imputed, markers: the 16 measured per-type times plus 20 unmeasured types at the mean measured per-type time; the
  range is the 2.5th-97.5th percentile of the measured sum plus 20 per-type times drawn with replacement from the 16
  measured ones (10,000 draws, numpy default_rng seed 0). Per-type time does not track the number of cells in the type;
* imputed, pseudobulk and module score: 10 x the 100,000-cell Seurat time (linear in cells).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent / "results" / "seurat"
N = 1_000_000
SEED = 0
DRAWS = 10_000

MEASURED = {"normalize": 71.24, "hvg": 74.53, "pca": 400.74, "neighbors": 368.06, "cluster": 1558.13}

# FindMarkers per cell type (seconds, cells in the type, which log)
MARKERS = {
    "unidentified": (1441.8, 298467, "seurat_markers_n1000000_shard0of4.json"),
    "cd4_tfh": (485.8, 25689, "main run"),
    "b_memory": (3765.4, 2205, "main run"),
    "b_naive": (4268.5, 6470, "main run"),
    "innate_t": (610.3, 5407, "main run"),
    "cd4_t_naive": (2101.6, 133814, "shard 1"),
    "cd8_t_naive": (1595.3, 26954, "shard 1"),
    "cd4_t_memory": (474.6, 128555, "shard 2"),
    "cd8_tem": (1248.1, 31692, "shard 2"),
    "cd4_t_memory#isg": (769.6, 18651, "shard 2"),
    "cd4_th1": (879.7, 17062, "shard 2"),
    "cd4_th17": (759.2, 54779, "shard 3"),
    "cd8_tcm": (408.2, 32694, "shard 3"),
    "cd4_treg": (674.6, 32481, "shard 3"),
    "cd4_th2": (552.8, 31973, "shard 3"),
    "cd8_temra": (2391.7, 26067, "shard 3"),
}
GROUPS_TOTAL = 36


def main() -> None:
    t = np.array([v[0] for v in MARKERS.values()])
    k = GROUPS_TOTAL - len(t)
    est = float(t.sum() + k * t.mean())
    rng = np.random.default_rng(SEED)
    boot = t.sum() + rng.choice(t, (DRAWS, k)).sum(1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    s100 = json.loads((S / "seurat_timing_n100000_seurat.json").read_text())["seurat"]["times"]
    s10 = json.loads((S / "seurat_timing_n10000_seurat.json").read_text())["seurat"]["times"]
    lin = {st: {"seconds": 10 * s100[st], "status": "imputed",
                "method": "10 x the 100,000-cell Seurat time (linear in cells)",
                "seurat_100k_s": s100[st], "seurat_10k_s": s10[st], "scale_10k_to_100k": s100[st] / s10[st]}
           for st in ("pseudobulk", "score_module")}

    steps = {st: {"seconds": v, "status": "measured"} for st, v in MEASURED.items()}
    steps["markers"] = {
        "seconds": est, "status": "imputed",
        "method": (f"sum of FindMarkers for the {len(t)} measured cell types plus {k} unmeasured types at the mean "
                   f"measured per-type time"),
        "measured_types": len(t), "groups_total": GROUPS_TOTAL, "measured_sum_s": float(t.sum()),
        "mean_per_type_s": float(t.mean()),
        "range_95_s": [lo, hi],
        "range_method": (f"2.5th-97.5th percentile of the measured sum plus {k} per-type times drawn with replacement "
                         f"from the {len(t)} measured ones ({DRAWS:,} draws, numpy default_rng seed {SEED})"),
        "per_type": {g: {"seconds": s, "cells": c, "source": src} for g, (s, c, src) in MARKERS.items()},
        "note": ("per-type time does not track the number of cells: b_memory (2,205 cells) took 3,765 s, cd8_tcm "
                 "(32,694 cells) 408 s; cd4_tfh ran twice, 485.8 s alone and 481.2 s in a shard"),
    }
    steps.update(lin)

    out = {
        "n": N, "time": "2026-10-27 13:20:23", "complete": False,
        "note": ("one-thread Seurat run stopped by the operator before it finished; measured steps copied from its log "
                 "and the seurat_markers_shard.py logs, the rest imputed (see steps[*].status)"),
        "seurat": {
            "tool": "seurat", "n": N,
            "times": {st: d["seconds"] for st, d in steps.items()},
            "steps": steps,
            "imputed": [st for st, d in steps.items() if d["status"] == "imputed"],
            "markers_groups_total": GROUPS_TOTAL,
            "warm_n": 2000, "warmup_s": 209.57, "package_load_s": 2.0, "read_store_s": 135.8,
            "threads": {"blas_threads": 1, "future_workers": 1},
            "r_version": "R version 4.3.1 (2023-06-16 ucrt)", "versions": {"Seurat": "4.3.0"},
        },
        "agreement": None,
    }
    f = S / f"seurat_timing_n{N}_seurat.json"
    f.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{f.name}: markers {est / 3600:.2f} h (95% range {lo / 3600:.2f}-{hi / 3600:.2f} h); pseudobulk "
          f"{lin['pseudobulk']['seconds']:.1f} s; module score {lin['score_module']['seconds']:.1f} s")


if __name__ == "__main__":
    main()
