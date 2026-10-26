"""UMAP engines: arsc's own (``engine="arsc"``) vs umap-learn, on the same exact kNN.

    python benchmarks/umap_parity.py --out <dir outside the repo> [--real <events x dims .npy>] [--seeds 5]
                                     [--threads 1,2,4,8]

Measures (results in SEURAT_PARITY.md, "UMAP engines"):
* fuzzy graph: same sparsity pattern and max |weight difference| against ``umap.umap_.fuzzy_simplicial_set``;
* spectral initialization: |correlation| of each axis with umap-learn's ``spectral_layout`` on the same pruned graph;
* layout quality per seed: trustworthiness (scikit-learn, 5,000-point subsample, 15 neighbors), kNN preservation
  (share of each point's 15 input-space neighbors among its 15 layout neighbors), and Procrustes disparity
  between layouts (arsc vs arsc, umap-learn vs umap-learn, arsc vs umap-learn);
* seconds: umap-learn's layout (single-threaded, seeded), arsc's at each thread count;
* determinism: SHA-1 of arsc's layout at each thread count.

Writes ``umap_parity.csv`` / ``umap_parity.json`` into ``--out``; only aggregate numbers belong in the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def synthetic(n=20000, d=10, k=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, (k, d))
    lab = rng.integers(0, k, n)
    return centers[lab] + rng.normal(0, 1, (n, d)), lab


def knn_preservation(idx_hi, Y, k=15):
    import arsc
    iy, _ = arsc.knn(np.asarray(Y, np.float64), k + 1)
    a, b = idx_hi[:, 1:k + 1], iy[:, 1:k + 1]
    hit = 0
    for i in range(a.shape[0]):
        hit += len(np.intersect1d(a[i], b[i], assume_unique=True))
    return hit / (a.shape[0] * k)


def trust(X, Y, k=15, m=5000, seed=0):
    from sklearn.manifold import trustworthiness
    s = np.random.default_rng(seed).choice(X.shape[0], min(m, X.shape[0]), replace=False)
    return float(trustworthiness(X[s], Y[s], n_neighbors=k))


def disparity(A, B):
    from scipy.spatial import procrustes
    return float(procrustes(np.asarray(A, np.float64), np.asarray(B, np.float64))[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--min-dist", type=float, default=0.3)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import numba
    import scipy.sparse as sp
    import arsc
    from arsc import _umap
    from sklearn.utils import check_random_state
    from umap.umap_ import find_ab_params as ab_ref, fuzzy_simplicial_set as fss_ref, simplicial_set_embedding
    from umap.spectral import spectral_layout
    os.makedirs(a.out, exist_ok=True)
    data = {"synthetic": synthetic()[0]}
    if a.real:
        data["real"] = np.load(a.real).astype(np.float64)
    ths = [t for t in (int(x) for x in a.threads.split(",")) if t <= numba.config.NUMBA_NUM_THREADS]
    rows, summary = [], {}
    for name, X in data.items():
        n = X.shape[0]
        arsc.set_threads(max(ths))
        idx, d = arsc.knn(X, 15)
        G = _umap.fuzzy_simplicial_set(idx, d, 15)
        R = sp.csr_matrix(fss_ref(sp.coo_matrix((n, 1)), 15, None, None, knn_indices=idx, knn_dists=d)[0])
        G.sort_indices()
        R.sort_indices()
        same_pattern = bool(np.array_equal(G.indptr, R.indptr) and np.array_equal(G.indices, R.indices))
        wdiff = float(abs(G - R).max())
        a_ours = _umap.find_ab_params(1.0, a.min_dist)
        a_ref = ab_ref(1.0, a.min_dist)
        # spectral init on the same pruned graph (200 epochs above 10,000 points, else 500)
        Gp = sp.coo_matrix(G).copy()
        ne = 500 if n <= 10000 else 200
        Gp.data[Gp.data < Gp.data.max() / ne] = 0
        Gp.eliminate_zeros()
        S_ours = _umap.spectral_init(Gp.tocsr(), 2, X=X, seed=0)
        S_ref = spectral_layout(X, Gp, 2, check_random_state(0))
        spec_corr = [float(abs(np.corrcoef(S_ours[:, j], S_ref[:, j])[0, 1])) for j in range(2)]
        a_, b_ = a_ours
        ours, refs, t_ours, t_ref, digests = {}, {}, {}, [], {}
        for s in range(a.seeds):
            t0 = time.perf_counter()
            refs[s] = simplicial_set_embedding(X, R, 2, 1.0, a_ref[0], a_ref[1], 1.0, 5, None, "spectral",
                                               check_random_state(s), "euclidean", {}, False, {}, False)[0]
            t_ref.append(time.perf_counter() - t0)
        for th in ths:
            arsc.set_threads(th)
            t0 = time.perf_counter()
            Y = _umap.embed(G, 2, min_dist=a.min_dist, seed=0, X=X)
            t_ours[th] = time.perf_counter() - t0
            digests[th] = hashlib.sha1(np.ascontiguousarray(Y).tobytes()).hexdigest()[:12]
        arsc.set_threads(max(ths))
        for s in range(a.seeds):
            ours[s] = _umap.embed(G, 2, min_dist=a.min_dist, seed=s, X=X)
        for eng, L in (("arsc", ours), ("umap-learn", refs)):
            for s, Y in L.items():
                rows.append({"data": name, "engine": eng, "seed": s, "trustworthiness": trust(X, Y),
                             "knn_preservation": knn_preservation(idx, Y)})
        pairs = {"arsc-arsc": [disparity(ours[i], ours[j]) for i, j in itertools.combinations(ours, 2)],
                 "ref-ref": [disparity(refs[i], refs[j]) for i, j in itertools.combinations(refs, 2)],
                 "arsc-ref": [disparity(ours[i], refs[j]) for i in ours for j in refs]}
        q = {e: [r for r in rows if r["data"] == name and r["engine"] == e] for e in ("arsc", "umap-learn")}
        summary[name] = {
            "n": n, "dims": X.shape[1], "graph_nnz": int(G.nnz), "graph_same_pattern": same_pattern,
            "graph_max_abs_diff": wdiff, "ab_ours": a_ours, "ab_ref": [float(v) for v in a_ref],
            "spectral_abs_corr": spec_corr,
            "trust": {e: [float(np.median([r["trustworthiness"] for r in v])), float(min(r["trustworthiness"] for r in v)),
                          float(max(r["trustworthiness"] for r in v))] for e, v in q.items()},
            "knn_pres": {e: [float(np.median([r["knn_preservation"] for r in v])), float(min(r["knn_preservation"] for r in v)),
                             float(max(r["knn_preservation"] for r in v))] for e, v in q.items()},
            "disparity": {k: [float(np.median(v)), float(min(v)), float(max(v))] for k, v in pairs.items()},
            "seconds_ref_layout_1thread": [float(np.median(t_ref)), float(min(t_ref)), float(max(t_ref))],
            "seconds_arsc_embed": t_ours, "digests": digests,
            "identical_across_threads": len(set(digests.values())) == 1,
        }
        print(json.dumps({name: summary[name]}, indent=1), flush=True)
    import csv
    with open(os.path.join(a.out, "umap_parity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "umap_parity.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
