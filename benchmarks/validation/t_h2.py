import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, "E:/autoresearch/vendor/arsc/benchmarks")
from refutil import *
import arsc
from arsc.harmony import run_harmony
ds = arsc.read_seurat_export(EXP_GD, assays=[], graphs=False)
pcs = read_dense(REF_GD/"pca_embeddings").values
Y0 = read_dense(REF_GD/"harmony_bs005_Y0").values
def mr(a,b): return min(np.corrcoef(a[:,k], b[:,k])[0,1] for k in range(a.shape[1]))
Rd = read_dense(REF_GD/"harmony_default").values; Rb = read_dense(REF_GD/"harmony_bs005_Z").values
print("R default vs R bs005 (same seed):", np.abs(Rd-Rb).max())
a = run_harmony(pcs, ds.obs, "n10x_lane_name", init_centroids=Y0.T, seed=0)
b = run_harmony(pcs, ds.obs, "n10x_lane_name", init_centroids=Y0.T, seed=1)
print("py seed0 vs seed1 (same init): min r", mr(a,b))
import cProfile, pstats
cProfile.run('run_harmony(pcs, ds.obs, "n10x_lane_name", seed=3)', 'E:/autoresearch/scratch/arsc/h.prof')
pstats.Stats('E:/autoresearch/scratch/arsc/h.prof').sort_stats('cumtime').print_stats(14)
