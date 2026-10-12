"""Timing of arsc.clones vs the R timings recorded by scratch/arsc/clones_ref.R (CD8 FORCYCLONE, 83,287 cells)."""
import time
from pathlib import Path
import pandas as pd
from arsc import clones as C

ROOT = Path("E:/autoresearch"); REF = ROOT / "scratch/arsc/clones_ref"; V3 = ROOT / "data/2023 0129 V3"
m = pd.read_csv(ROOT / "projects/cancercell2024-melanoma-icb/cache/cd8_nn_labeled_FORCYCLONE_FINAL_cyclone.meta.csv.gz",
                usecols=["patient_alias", "timepoint", "clone_id", "treatment", "response", "metaclusters_exh",
                         "TRA_chain_cdr3_nt", "TRB_chain_cdr3_nt"])
m["week"] = m.timepoint.astype(str); m["clone_pair"] = m.TRA_chain_cdr3_nt + " " + m.TRB_chain_cdr3_nt
rt = dict(l.split() for l in (REF / "timing.txt").read_text().splitlines())
def t(f):
    s = time.perf_counter(); f(); return time.perf_counter() - s
hd = C.read_10x_clonotypes([V3 / f"tcr/HD/P6 {x} clonotypes.csv" for x in ("T1", "T2")])
ct = C.clone_table(m, "clone_id")
rows = [
 ("cyclone (published settings)", rt["cyclone_pub"], t(lambda: C.cyclone(m, timepoints=[0,3,6,9], n_clust=6, custom_order=[4,5,1,6,2,3], outlier_sd=2.5, verbose=False))),
 ("clone_expansion_null (reps=5)", rt["expnull_reps5"], t(lambda: C.clone_expansion_null(m, hd, clone_col="clone_pair", time_col="week", hd_time_col="timepoint", hd_clone_col="clone_id", reps=5))),
 ("startrac (W3 + all-tp migr)", rt["startrac"], t(lambda: (C.startrac(m[m.week=="3"]), C.startrac(m, loc_col="week")))),
 ("clone_dynamics 0->3", rt["dynamics"], t(lambda: C.clone_dynamics(ct, 0, 3))),
 ("clone_diversity (reps=20)", rt["diversity_reps20"], t(lambda: C.clone_diversity(ct, reps=20, verbose=False))),
]
print("| method | R s | Python s | speedup |\n|---|---|---|---|")
for n, r, p in rows: print(f"| {n} | {float(r):.2f} | {p:.2f} | {float(r)/p:.1f}x |")
