"""Which component makes arsc's clustering differ from scanpy's? Swap one at a time (graph, Leiden engine, seed,
resolution) on the same embedding, in memory.

    ARSC_BENCH_STORE=<store> ARSC_BENCH_STORE_COLUMNS='{"patient": "patient_alias"}' \
        python benchmarks/cluster_factors.py --n 100000 --threads 8

The embedding is the paired benchmark's (``benchdata.Bench(source, n, seed).pca``: the seeded subset of ``n``
cells, 2,000 HVGs, 30 PCs). Graphs (k = 20 neighbors counting the cell itself, as the paired benchmark):

* ``scanpy``: ``sc.pp.neighbors(n_neighbors=20, random_state=s)`` (pynndescent + UMAP connectivities);
* ``arsc_umap``: ``arsc.scanpy_extra.neighbors`` (arsc's k-NN + the same UMAP connectivities);
* ``arsc_snn``: ``arsc.find_neighbors(k=20)["snn"]`` (Seurat ``FindNeighbors``: Jaccard SNN, prune 1/15).

Engines (modularity with resolution, ``n_iterations=2``): ``igraph`` (``sc.tl.leiden(flavor="igraph")`` via
``arsc.scanpy_extra.leiden``, as the paired benchmark runs scanpy), ``leidenalg`` (``arsc.cluster(engine=
"leidenalg")``), ``arsc`` (``arsc.cluster``, arsc's own engine, leidenalg's refinement) and
``arsc_igraph_refine`` (``arsc.cluster(refinement="igraph")``). The reference partitions are scanpy's pipeline at
seed 0 and seed 1 (graph and Leiden seeded alike). The ARI is computed on all cells.

``--matrix N`` instead runs each pipeline (graph + engine) with N seeds and reports the mean ARI against scanpy's
pipeline over all seed pairs, next to scanpy's own seed-to-seed ARI (``--only`` picks pipelines).
Output: one small CSV (``--out``); nothing else is written.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpus", default="", help="comma-separated logical CPUs to pin to (e.g. 1,3,5)")
    ap.add_argument("--seeds", type=int, default=3, help="Leiden seeds per (graph, engine)")
    ap.add_argument("--res", type=float, default=0.8)
    ap.add_argument("--sweep", default="0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2")
    ap.add_argument("--out", default=str(HERE / "results" / "cluster_factors.csv"))
    ap.add_argument("--matrix", type=int, default=0, help="instead: N seeds per pipeline, all pairwise ARIs against "
                    "scanpy's pipeline (graph and Leiden seeded alike); writes <out> with one row per pipeline")
    ap.add_argument("--only", default="", help="--matrix: run only the pipelines whose name contains one of these "
                    "';'-separated parts (scanpy's own pipeline always runs)")
    ap.add_argument("--snn-res", type=float, default=0.4, help="--matrix: the SNN resolution matching scanpy's "
                    "cluster count")
    a = ap.parse_args()
    for v in ("AR_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    if a.cpus:
        import psutil
        psutil.Process().cpu_affinity([int(c) for c in a.cpus.split(",")])
    sys.path.insert(0, str(HERE.parent / "src"))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    import arsc
    from arsc import scanpy_extra as sx
    from arsc._leiden import prepare_graph, quality
    from benchdata import Bench, resolve_source

    t0 = time.time()
    E = Bench(resolve_source("auto"), a.n, a.seed).pca.astype(np.float32)
    print(f"embedding {E.shape} in {time.time() - t0:.0f} s", flush=True)

    import anndata as ad
    import scanpy as sc
    sc.settings.n_jobs = a.threads

    def scanpy_graph(s):
        ad_ = ad.AnnData(np.zeros((E.shape[0], 1), np.float32))
        ad_.obsm["X_pca"] = E
        sc.pp.neighbors(ad_, n_neighbors=20, use_rep="X_pca", random_state=s)
        return ad_, ad_.obsp["connectivities"].tocsr()

    if a.matrix:
        return seed_matrix(a, E, scanpy_graph, t0)
    graphs = {}
    t = time.time()
    a0, graphs["scanpy"] = scanpy_graph(0)
    _, graphs["scanpy_seed1"] = scanpy_graph(1)
    graphs["arsc_umap"] = sx.neighbors(E, n_neighbors=20)["connectivities"].tocsr()
    graphs["arsc_snn"] = arsc.find_neighbors(E, k=20)["snn"].tocsr()
    print(f"graphs in {time.time() - t:.0f} s", flush=True)
    for g, G in graphs.items():
        H = G.copy()
        H.setdiag(0)
        H.eliminate_zeros()
        print(f"  {g}: nnz/cell {H.nnz / H.shape[0]:.1f}, symmetric {abs(H - H.T).max() < 1e-12}", flush=True)

    prepared = {g: prepare_graph(G) for g, G in graphs.items()}

    def run(engine, G, res, s):
        if engine == "igraph":
            return sx.leiden(G, res, random_state=s, flavor="igraph", n_iterations=2)
        if engine == "arsc_igraph_refine":
            return arsc.cluster(G, method="leiden", resolution=res, seed=s, n_iterations=2, refinement="igraph")
        return arsc.cluster(G, method="leiden", resolution=res, seed=s, n_iterations=2,
                            engine="arsc" if engine == "arsc" else "leidenalg")

    # the references: scanpy's pipeline, seed 0 and seed 1; check the wrapper against sc.tl.leiden itself
    ref0 = run("igraph", graphs["scanpy"], a.res, 0)
    ref1 = run("igraph", graphs["scanpy_seed1"], a.res, 1)
    sc.tl.leiden(a0, resolution=a.res, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    same = ari(a0.obs["leiden"].astype(int).to_numpy(), ref0)
    print(f"sx.leiden(igraph) vs sc.tl.leiden ARI {same:.6f}", flush=True)

    rows = []

    def record(graph, engine, res, s, lab, secs):
        q = float(quality(prepared[graph], np.asarray(lab), resolution=res))
        r = dict(graph=graph, engine=engine, resolution=res, seed=s, n_clusters=int(len(np.unique(lab))),
                 ari_ref_seed0=float(ari(ref0, lab)), ari_ref_seed1=float(ari(ref1, lab)), modularity=q,
                 seconds=round(secs, 2))
        rows.append(r)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)

    record("scanpy", "igraph", a.res, 0, ref0, 0)
    record("scanpy_seed1", "igraph", a.res, 1, ref1, 0)
    # factor table: every graph x engine at the benchmark's resolution, several seeds
    for g in ("scanpy", "arsc_umap", "arsc_snn"):
        for eng in ("igraph", "leidenalg", "arsc", "arsc_igraph_refine"):
            for s in range(a.seeds):
                if g == "scanpy" and eng == "igraph" and s == 0:
                    continue
                t = time.time()
                lab = run(eng, graphs[g], a.res, s)
                record(g, eng, a.res, s, lab, time.time() - t)
    # resolution sweeps (seed 0) to match cluster counts
    for g in ("scanpy", "arsc_umap", "arsc_snn"):
        for eng in ("igraph", "arsc", "arsc_igraph_refine"):
            for res in [float(x) for x in a.sweep.split(",")]:
                if res == a.res:
                    continue
                t = time.time()
                lab = run(eng, graphs[g], res, 0)
                record(g, eng, res, 0, lab, time.time() - t)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows) in {time.time() - t0:.0f} s", flush=True)


def seed_matrix(a, E, scanpy_graph, t0):
    """Each pipeline run with seeds 0..N-1; mean ARI against scanpy's pipeline over all seed pairs (for scanpy
    against itself: the pairs with different seeds), mean ARI among the pipeline's own seeds, cluster counts."""
    import itertools

    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    import arsc
    from arsc import scanpy_extra as sx

    N, res = a.matrix, a.res
    t = time.time()
    sg = [scanpy_graph(s)[1] for s in range(N)]
    umap = sx.neighbors(E, n_neighbors=20)["connectivities"].tocsr()
    snn = arsc.find_neighbors(E, k=20)["snn"].tocsr()
    print(f"graphs in {time.time() - t:.0f} s", flush=True)
    ig = lambda G, r, s: sx.leiden(G, r, random_state=s, flavor="igraph", n_iterations=2)  # noqa: E731
    ar = lambda G, r, s, it=2, rf="leidenalg": arsc.cluster(G, method="leiden", resolution=r, seed=s,  # noqa: E731
                                                          n_iterations=it, refinement=rf)
    la = lambda G, r, s: arsc.cluster(G, method="leiden", resolution=r, seed=s, n_iterations=2,  # noqa: E731
                                      engine="leidenalg")
    pipes = {
        "scanpy graph + igraph Leiden (scanpy's pipeline)": lambda s: ig(sg[s], res, s),
        "scanpy graph + leidenalg Leiden": lambda s: la(sg[s], res, s),
        "scanpy graph + arsc Leiden": lambda s: ar(sg[s], res, s),
        "scanpy graph + arsc Leiden, n_iterations=-1": lambda s: ar(sg[s], res, s, -1),
        "scanpy graph + arsc Leiden, igraph refinement": lambda s: ar(sg[s], res, s, 2, "igraph"),
        "arsc UMAP graph + igraph Leiden": lambda s: ig(umap, res, s),
        "arsc UMAP graph + arsc Leiden": lambda s: ar(umap, res, s),
        "arsc UMAP graph + arsc Leiden, n_iterations=-1": lambda s: ar(umap, res, s, -1),
        "arsc UMAP graph + arsc Leiden, igraph refinement": lambda s: ar(umap, res, s, 2, "igraph"),
        "arsc SNN graph + arsc Leiden, igraph refinement": lambda s: ar(snn, res, s, 2, "igraph"),
        "arsc SNN graph + igraph Leiden": lambda s: ig(snn, res, s),
        "arsc SNN graph + arsc Leiden (paired benchmark)": lambda s: ar(snn, res, s),
        f"arsc SNN graph + arsc Leiden, resolution {a.snn_res}": lambda s: ar(snn, a.snn_res, s),
    }
    labs, secs = {}, {}
    keep = [x for x in a.only.split(";") if x]
    pipes = {k: f for k, f in pipes.items() if k.startswith("scanpy graph + igraph") or not keep
             or any(x in k for x in keep)}
    for name, f in pipes.items():
        t = time.time()
        labs[name] = [np.asarray(f(s)) for s in range(N)]
        secs[name] = (time.time() - t) / N
        print(f"{name}: {secs[name]:.1f} s per run", flush=True)
    ref = labs["scanpy graph + igraph Leiden (scanpy's pipeline)"]
    rows = []
    for name, L in labs.items():
        if name.startswith("scanpy graph + igraph"):
            vs = [ari(ref[i], ref[j]) for i, j in itertools.combinations(range(N), 2)]
        else:
            vs = [ari(ref[i], L[j]) for i in range(N) for j in range(N)]
        own = [ari(L[i], L[j]) for i, j in itertools.combinations(range(N), 2)]
        k = [len(np.unique(x)) for x in L]
        rows.append(dict(pipeline=name, resolution=a.snn_res if name.endswith(str(a.snn_res)) else res, seeds=N,
                         ari_vs_scanpy_mean=np.mean(vs), ari_vs_scanpy_min=np.min(vs), ari_vs_scanpy_max=np.max(vs),
                         ari_own_seeds_mean=np.mean(own), n_clusters_median=float(np.median(k)),
                         n_clusters_min=min(k), n_clusters_max=max(k), seconds_per_run=round(secs[name], 2)))
        print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
