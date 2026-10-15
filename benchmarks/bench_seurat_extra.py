"""Wall-clock of arsc.seurat_extra on the reference gamma-delta T-cell export ($ARSC_SCRATCH/exp_gd; same calls as benchmarks/r_reference_seurat_extra.R).

Usage: python benchmarks/bench_seurat_extra.py   -> benchmarks/results/seurat_extra_timings.json (+ speedup table)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refutil import EXP_GD, RESULTS, SCRATCH, r_timings, save_json, timer  # noqa: E402

import arsc  # noqa: E402
from arsc import preprocessing as pp  # noqa: E402
from arsc import seurat_extra as se  # noqa: E402

REF = SCRATCH / "ref_extra"
ds = arsc.read_seurat_export(EXP_GD)
feats = ds["RNA"].features
ln = pp.log_normalize(ds.X("RNA", "counts"))
adt = pp.clr(ds.X("TRIMMEDCITE", "counts"), margin=2)
obs = ds.obs
m = (obs["cell_type"] == "gd").to_numpy()
X, tp = ln[m], obs["timepoint"][m].astype(str).reset_index(drop=True)
cells = open(REF / "lr_cells.txt").read().split()
idx = obs.index.get_indexer(cells)
Xl, ol = ln[idx], obs.iloc[idx].reset_index(drop=True)
big = open(REF / "fam_types.txt").read().split()
mb = obs["cell_type"].isin(big).to_numpy()
T: dict[str, float] = {}
# warm-up (numba compile) on tiny inputs so timings are steady-state
w = np.flatnonzero(tp.isin(["Follow Up 1", "Baseline"]).to_numpy())[:400]
se.find_markers(X[w], tp[w].reset_index(drop=True), "Follow Up 1", "Baseline", features=feats, min_pct=0.3, logfc_threshold=0.5)
se.find_markers(X[w], tp[w].reset_index(drop=True), "Follow Up 1", "Baseline", features=feats, test_use="LR",
                min_pct=0.3, logfc_threshold=0.5, latent_vars=obs.iloc[np.flatnonzero(m)[w]][["nCount_RNA"]].reset_index(drop=True))
se.find_all_markers(X[w], tp[w].reset_index(drop=True), features=feats, min_pct=0.3, logfc_threshold=0.5)
se.scale_data(ln[:300], feats, select=feats[:20], vars_to_regress=["nCount_RNA"], latent_data=obs.iloc[:300])
sel = open(REF / "scale_features.txt").read().split()
sel = [f for f in feats if f in set(sel)]
with timer("FindMarkers_wilcox_t01", T):
    se.find_markers(X, tp, "Follow Up 1", "Baseline", features=feats, min_pct=0.1, logfc_threshold=0.1)
with timer("FindMarkers_wilcox_default", T):
    se.find_markers(X, tp, "Follow Up 2", "Baseline", features=feats)
with timer("FindMarkers_wilcox_onlypos_maxcells", T):
    se.find_markers(X, tp, "Follow Up 3", "Follow Up 1", features=feats, only_pos=True, min_pct=0.25,
                    max_cells_per_ident=1500, random_seed=7)
with timer("FindMarkers_vs_rest", T):
    se.find_markers(X, tp, "Baseline", None, features=feats, min_pct=0.1, logfc_threshold=0.1)
with timer("FindMarkers_t", T):
    se.find_markers(X, tp, "Follow Up 1", "Baseline", features=feats, test_use="t", min_pct=0.25)
with timer("FindMarkers_LR_latent", T):
    se.find_markers(Xl, ol["timepoint"].astype(str), "Follow Up 1", "Baseline", features=feats, test_use="LR",
                    latent_vars=ol[["nCount_RNA", "percent.mt"]], min_pct=0.25, logfc_threshold=0.25)
with timer("FindMarkers_LR_factor", T):
    se.find_markers(Xl, ol["timepoint"].astype(str), "Follow Up 1", "Baseline", features=feats, test_use="LR",
                    latent_vars=ol[["patient_alias"]], min_pct=0.3, logfc_threshold=0.4)
with timer("FindMarkers_ADT", T):
    se.find_markers(adt, obs["timepoint"].astype(str), "Follow Up 1", "Baseline", features=ds["TRIMMEDCITE"].features,
                    min_pct=0.1, logfc_threshold=0.25, norm_method="CLR")
lab = obs["cell_type"][mb].astype(str).reset_index(drop=True)
with timer("FindAllMarkers", T):
    se.find_all_markers(ln[mb], lab, features=feats, only_pos=True, min_pct=0.25, logfc_threshold=0.25,
                        levels=pd.unique(lab))
with timer("AverageExpression_data", T):
    se.average_expression(ln, obs, ["patient_alias", "timepoint"], features=feats)
with timer("AverageExpression_counts", T):
    se.average_expression(ds.X("RNA", "counts"), obs, "cell_type", features=feats, layer="counts")
with timer("CellCycleScoring", T):
    se.cell_cycle_scoring(ln, feats, cells=obs.index)
with timer("ScaleData_regress_200", T):
    se.scale_data(ln, feats, select=sel, vars_to_regress=["nCount_RNA", "percent.mt"], latent_data=obs)
with timer("ScaleData_regress_split_200", T):
    se.scale_data(ln, feats, select=sel, vars_to_regress=["percent.mt"], latent_data=obs,
                  split_by=obs["n10x_lane_name"].astype(str).to_numpy())
with timer("ScaleData_regress_2000", T):
    hvg = pd.Index(feats[:0]).append(pd.Index(open(SCRATCH / "ref_gd/variable_features.txt").read().split()))
    se.scale_data(ln, feats, select=[f for f in feats if f in set(hvg)], vars_to_regress=["nCount_RNA", "percent.mt"],
                  latent_data=obs)

r = r_timings(REF) if (REF / "timings_r.json").exists() else {}
rows = [{"step": k, "arsc_s": round(v, 3), "R_s": r.get(k), "speedup": (round(r[k] / v, 1) if k in r and v > 0 else None)}
        for k, v in T.items()]
print(pd.DataFrame(rows).to_string(index=False))
save_json({"arsc": T, "R": r}, RESULTS / "seurat_extra_timings.json")
