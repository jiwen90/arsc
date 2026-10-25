"""Write the input for benchmarks/r_reference_harmony.R from an arsc store: its PCA embedding and batch columns.

Usage: python benchmarks/harmony_input.py <store> [out_dir] [--reduction pca] [--dims 30] [--vars a,b,c]

Writes <out_dir>/pcs.{f64,json} (r_common.R::write_dense layout) and <out_dir>/meta.csv. Batch levels are
recoded to zero-padded labels (V1_001, ...) in the store's category order, so R's ``as.factor`` (locale
collation) and pandas (code-point order) sort them identically. The references used by tests/test_vs_r.py
come from the 38,440-cell reference gamma-delta T-cell store with
``--vars patient_alias,n10x_lane_name,patient_timepoint`` (default out_dir: $ARSC_SCRATCH/ref_harmony/input).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refutil import SCRATCH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("out", nargs="?", default=str(SCRATCH / "ref_harmony" / "input"))
    ap.add_argument("--reduction", default="pca")
    ap.add_argument("--dims", type=int, default=30)
    ap.add_argument("--vars", default="patient_alias,n10x_lane_name,patient_timepoint")
    a = ap.parse_args()
    import arsc
    st = arsc.open(a.store)
    Z = np.asarray(st.reduction(a.reduction).embedding, dtype=np.float64)[:, : a.dims]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.asfortranarray(Z).T.astype("<f8").tofile(out / "pcs.f64")  # column-major
    (out / "pcs.json").write_text(json.dumps({"dim": list(Z.shape), "rownames": None, "colnames": None}))
    meta = {}
    for j, v in enumerate(a.vars.split(","), 1):
        cat = pd.Categorical(st.obs[v].astype(str))
        meta[v] = [f"V{j}_{c:04d}" for c in cat.codes]
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    print(f"wrote {Z.shape[0]} cells x {Z.shape[1]} dims and {len(meta)} batch columns to {out}")


if __name__ == "__main__":
    main()
