"""Write the input for benchmarks/r_reference_harmony.R from an arsc store: its PCA embedding and batch columns.

Usage: python benchmarks/harmony_input.py <store> [out_dir] [--reduction pca] [--dims 30] [--vars a,b,c]

Writes <out_dir>/pcs.f64.gz + pcs.json (r_common.R::write_dense_gz layout, gzip float64) and <out_dir>/meta.csv.gz.
Batch levels are recoded to zero-padded labels (V1_001, ...) in the store's category order, so R's ``as.factor``
(locale collation) and pandas (code-point order) sort them identically. The references used by tests/test_vs_r.py
come from the 38,440-cell reference gamma-delta T-cell store with
``--vars patient_alias,n10x_lane_name,patient_timepoint`` (default out_dir: $ARSC_REF_HARMONY/input, see
benchmarks/refutil.py; benchmarks/r_reference_harmony_all.R then rebuilds the references from it).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refutil import REF_HARMONY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("out", nargs="?", default=str(REF_HARMONY / "input"))
    ap.add_argument("--reduction", default="pca")
    ap.add_argument("--dims", type=int, default=30)
    ap.add_argument("--vars", default="patient_alias,n10x_lane_name,patient_timepoint")
    a = ap.parse_args()
    import arsc
    st = arsc.open(a.store)
    Z = np.asarray(st.reduction(a.reduction).embedding, dtype=np.float64)[:, : a.dims]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "pcs.f64.gz", "wb") as f:  # column-major
        f.write(np.ascontiguousarray(Z.T).astype("<f8").tobytes())
    (out / "pcs.json").write_text(json.dumps({"dim": list(Z.shape), "rownames": None, "colnames": None}))
    meta = {}
    for j, v in enumerate(a.vars.split(","), 1):
        cat = pd.Categorical(st.obs[v].astype(str))
        meta[v] = [f"V{j}_{c:04d}" for c in cat.codes]
    pd.DataFrame(meta).to_csv(out / "meta.csv.gz", index=False)
    print(f"wrote {Z.shape[0]} cells x {Z.shape[1]} dims and {len(meta)} batch columns to {out}")


if __name__ == "__main__":
    main()
