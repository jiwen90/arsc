"""Parity of arsc's own Leiden / Louvain (``arsc.cluster(engine="arsc")``) with leidenalg / igraph.

    python benchmarks/leiden_parity.py --out <dir outside the repo> [--emb <n x d .npy>] [--seeds 10]
                                       [--threads 1,8] [--resolution 0.5]

Graphs: (1) a planted partition (8 blocks of 1,000 nodes, weighted), (2) the SNN graph (k = 20) of
``arsc.datasets.toy()`` PCs, (3) optionally the SNN graph (k = 15) of an embedding passed with ``--emb``
(e.g. events x PCs of a cytometry panel). For each graph, objective (modularity at ``--resolution``;
CPM at 0.05, 0.005 on the planted graph whose
block density is 0.02) and method, seeds 0..S-1 of both engines: quality, number of clusters, ARI ours-vs-ours,
ref-vs-ref and ours-vs-ref over seed pairs, seconds (median) at each thread count, and a digest check
across thread counts. Writes ``leiden_parity.csv`` and ``leiden_parity.md`` (a table) into ``--out``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _planted(n_blocks=8, size=1000, p_in=0.02, p_out=0.0015, seed=0):
    import numpy as np
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    truth = np.repeat(np.arange(n_blocks), size)
    rows, cols = [], []
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            p = p_in if i == j else p_out
            m = rng.binomial(size * size, p)
            r = rng.integers(0, size, m) + i * size
            c = rng.integers(0, size, m) + j * size
            rows.append(r)
            cols.append(c)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    keep = r != c
    A = sp.coo_matrix((rng.uniform(0.5, 1.5, keep.sum()), (r[keep], c[keep])), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    return A.tocsr(), truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--resolution", type=float, default=0.5)
    a = ap.parse_args()
    cap = int(os.environ.get("AR_THREADS", "4"))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(cap))
    import numba
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    import arsc
    from arsc import _leiden as L

    ths = [t for t in (int(x) for x in a.threads.split(",")) if t <= numba.config.NUMBA_NUM_THREADS]
    graphs = {}
    A, truth = _planted()
    graphs["planted 8x1000"] = (A, truth)
    ds = arsc.datasets.toy()
    Z, _, _ = arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))
    graphs["toy SNN k20"] = (arsc.find_neighbors(arsc.pp.pca(Z, n_comps=10).embedding, k=20)["snn"], None)
    if a.emb:
        E = np.load(a.emb).astype(np.float64)
        graphs[f"embedding SNN k15 ({E.shape[0]} x {E.shape[1]})"] = (arsc.find_neighbors(E, k=15)["snn"], None)
    os.makedirs(a.out, exist_ok=True)
    rows, table = [], []
    seeds = list(range(a.seeds))
    for gname, (G, tr) in graphs.items():
        arsc.cluster(G[:200, :200], engine="arsc")  # compile
        for method, obj, res in [("leiden", "modularity", a.resolution), ("leiden", "cpm", 0.005 if tr is not None else 0.05),
                                 ("louvain", "modularity", a.resolution)]:
            labs = {"arsc": {}, "ref": {}}
            secs = {"arsc": {}, "ref": {}}
            digests = {}
            for eng, key in (("arsc", "arsc"), ("leidenalg", "ref")):
                for t in ths:
                    numba.set_num_threads(t)
                    sec = []
                    for s in seeds if t == ths[0] else seeds[:3]:
                        t0 = time.perf_counter()
                        lab = arsc.cluster(G, method=method, resolution=res, seed=s, engine=eng, objective=obj,
                                           group_singletons=False)
                        sec.append(time.perf_counter() - t0)
                        if t == ths[0]:
                            labs[key][s] = lab
                        if eng == "arsc":
                            digests.setdefault(s, set()).add(hashlib.sha1(lab.tobytes()).hexdigest())
                    secs[key][t] = float(np.median(sec))
            q = {k: [L.quality(G, labs[k][s], res, obj) for s in seeds] for k in labs}
            ncl = {k: [int(labs[k][s].max()) + 1 for s in seeds] for k in labs}
            pairs = list(itertools.combinations(seeds, 2))
            a_oo = [ari(labs["arsc"][i], labs["arsc"][j]) for i, j in pairs]
            a_rr = [ari(labs["ref"][i], labs["ref"][j]) for i, j in pairs]
            a_or = [ari(labs["arsc"][i], labs["ref"][j]) for i in seeds for j in seeds]
            same = all(len(v) == 1 for v in digests.values())
            r = {"graph": gname, "method": method, "objective": obj, "resolution": res,
                 "q_arsc": q["arsc"], "q_ref": q["ref"], "k_arsc": ncl["arsc"], "k_ref": ncl["ref"],
                 "ari_oo": a_oo, "ari_rr": a_rr, "ari_or": a_or, "sec_arsc": secs["arsc"], "sec_ref": secs["ref"],
                 "same_digest_all_threads": same}
            if tr is not None:
                r["ari_truth_arsc"] = [ari(labs["arsc"][s], tr) for s in seeds]
                r["ari_truth_ref"] = [ari(labs["ref"][s], tr) for s in seeds]
            rows.append(r)

            def mr(v, f="{:.4f}"):
                v = np.asarray(v, float)
                return f"{f.format(np.median(v))} ({f.format(v.min())}-{f.format(v.max())})"
            fq = "{:.5f}" if obj == "modularity" else "{:.1f}"
            table.append({"graph": gname, "method": f"{method} {obj} {res:g}",
                          "quality arsc": mr(q["arsc"], fq), "quality ref": mr(q["ref"], fq),
                          "clusters arsc / ref": f"{np.median(ncl['arsc']):g} / {np.median(ncl['ref']):g}",
                          "ARI arsc-arsc": mr(a_oo, "{:.3f}"), "ARI ref-ref": mr(a_rr, "{:.3f}"),
                          "ARI arsc-ref": mr(a_or, "{:.3f}"),
                          "s arsc / ref (" + ",".join(map(str, ths)) + " thr)":
                              " ; ".join(f"{secs['arsc'][t]:.2f} / {secs['ref'][t]:.2f}" for t in ths),
                          "same labels all threads": "yes" if same else "NO"})
            print(json.dumps(table[-1]), flush=True)
    pd.DataFrame(rows).to_json(os.path.join(a.out, "leiden_parity.json"), orient="records", indent=1)
    md = pd.DataFrame(table).to_markdown(index=False)
    open(os.path.join(a.out, "leiden_parity.md"), "w", encoding="utf-8").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
