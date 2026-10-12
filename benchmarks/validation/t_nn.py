import sys, time, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "E:/autoresearch/vendor/arsc/benchmarks")
from refutil import *
import arsc
pcs = read_dense(REF_GD/"pca_embeddings").values
t=time.perf_counter(); fn = arsc.find_neighbors(pcs, k=20); print("find_neighbors", time.perf_counter()-t)
t=time.perf_counter(); fn = arsc.find_neighbors(pcs, k=20); print("find_neighbors", time.perf_counter()-t)
# brute-force check of knn on a few rows
q = pcs[:50]; d = ((q[:,None,:]-pcs[None])**2).sum(-1); ref = np.argsort(d,1,kind="stable")[:,:20]
print("knn exact match rows:", np.mean([set(a)==set(b) for a,b in zip(ref, fn["idx"][:50])]), "self first", (fn["idx"][:,0]==np.arange(len(pcs))).mean())
# SNN vs scipy reference
idx = fn["idx"]; n,k = idx.shape
A = sp.csr_matrix((np.ones(n*k),(np.repeat(np.arange(n),k), idx.ravel())), shape=(n,n)); S=(A@A.T).tocsr(); S.data = S.data/(2*k-S.data); S.data[S.data<1/15]=0; S.eliminate_zeros()
print("snn == scipy ref:", abs(S - fn["snn"]).max())
# vs Seurat's RNA_snn (Annoy) on same PCA
Rsnn = read_dgc(REF_GD/"snn", (n,n)).tocsr()
ov = []
for i in range(0, n, 37):
    a = set(Rsnn[i].indices); b = set(fn["snn"][i].indices); ov.append(len(a&b)/len(a|b))
print("SNN edge-set Jaccard vs Seurat (Annoy) mean", np.mean(ov))
t=time.perf_counter(); lab = arsc.cluster(fn["snn"], method="louvain", resolution=0.8); print("louvain", time.perf_counter()-t, lab.max()+1)
t=time.perf_counter(); lab2 = arsc.cluster(fn["snn"], method="leiden", resolution=0.8); print("leiden", time.perf_counter()-t, lab2.max()+1)
from sklearn.metrics import adjusted_rand_score as ari
rl = np.array(open(REF_GD/"louvain_clusters.txt").read().split(), dtype=int)
print("ARI louvain vs Seurat louvain", ari(rl, lab), " leiden vs Seurat", ari(rl, lab2), "Seurat n", rl.max()+1)
