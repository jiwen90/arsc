"""Graph clustering and UMAP wrappers with deterministic seeds.

``cluster(graph, method="leiden"|"louvain")`` clusters an SNN/WNN graph
(Seurat ``FindClusters``: labels are renumbered by decreasing cluster size,
singletons are merged into their most connected cluster). ``umap()`` wraps
umap-learn with Seurat ``RunUMAP`` defaults, reusing arsc's exact k-NN.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._utils import check_seed

__all__ = ["cluster", "umap"]


def _to_igraph(graph):
    import igraph as ig
    G = sp.csr_matrix(graph)
    G = G.maximum(G.T)  # symmetric
    U = sp.triu(G, k=1).tocoo()
    g = ig.Graph(n=G.shape[0], edges=np.column_stack([U.row, U.col]).tolist(), directed=False)
    g.es["weight"] = U.data.tolist()
    return g, G


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.lexsort((uniq, -counts))]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in labels], dtype=np.int64)


def _group_singletons(labels: np.ndarray, G: sp.csr_matrix) -> np.ndarray:
    labels = labels.copy()
    sizes = np.bincount(labels)
    single = np.flatnonzero(sizes[labels] == 1)
    if single.size == 0 or single.size == labels.size:
        return labels
    multi = sizes > 1
    for i in single:
        row = G.getrow(i)
        conn = np.bincount(labels[row.indices], weights=row.data, minlength=sizes.size)
        conn[~multi] = -1
        labels[i] = int(np.argmax(conn))
    return labels


def cluster(graph, method: str = "leiden", resolution: float = 0.8, seed: int = 0, n_iterations: int = 2,
            n_start: int = 10, group_singletons: bool = True) -> np.ndarray:
    """Cluster a weighted graph (e.g. ``find_neighbors(...)["snn"]`` or ``wnn(...).wsnn``).

    Seurat ``FindClusters``: labels are renumbered by decreasing cluster size (0 = largest) and
    singletons are merged into their most connected cluster.

    Parameters
    ----------
    graph : n x n sparse weighted graph.
    method : ``"leiden"`` (leidenalg, RBConfiguration = modularity with resolution,
        as Seurat ``algorithm = 4``) or ``"louvain"`` (igraph multilevel; best
        modularity over ``n_start`` seeded runs, as Seurat ``algorithm = 1``).
    resolution : modularity resolution (Seurat ``resolution``; higher = more clusters).
    n_iterations : Leiden iterations (leidenalg default 2; -1 = until stable).
    n_start : Louvain random starts (Seurat ``n.start``).
    group_singletons : merge singleton clusters (Seurat ``group.singletons``).
    seed : integer RNG seed (default 0 = Seurat ``FindClusters(random.seed = 0)``). Leiden: leidenalg's
        own seeded RNG. Louvain: start ``s`` uses a private ``random.Random(seed + s)`` installed as
        igraph's RNG for the call (igraph's default RNG is restored afterwards; Python's global
        ``random`` state is never touched). Same graph + seed -> identical labels; vary ``seed`` to
        check a clustering's stability.

    Returns
    -------
    numpy.ndarray of integer labels (n,), 0 = largest cluster.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> Z, _, _ = arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))
    >>> g = arsc.find_neighbors(arsc.pp.pca(Z, n_comps=10).embedding, k=20)
    >>> labels = arsc.cluster(g["snn"], resolution=0.5, seed=0)
    >>> int(labels.min())
    0
    """
    seed = check_seed(seed)
    g, G = _to_igraph(graph)
    if method == "leiden":
        import leidenalg as la
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=resolution, seed=seed, n_iterations=n_iterations)
        labels = np.asarray(part.membership)
    elif method == "louvain":
        # igraph multilevel (Louvain); igraph shuffles node order with its RNG, which we
        # seed per start. (leidenalg's Optimiser with refine_partition=False would be
        # Louvain too, but segfaults in leidenalg 0.12.)
        import random as _random

        import igraph as ig
        rng = _random.Random()
        ig.set_random_number_generator(rng)
        best, best_q = None, -np.inf
        try:
            for s_ in range(n_start):
                rng.seed(seed + s_)
                mem = np.asarray(g.community_multilevel(weights="weight", resolution=resolution).membership)
                q = g.modularity(mem.tolist(), weights="weight", resolution=resolution)
                if q > best_q:
                    best, best_q = mem, q
        finally:
            ig.set_random_number_generator(_random)  # igraph's default
        labels = best
    else:
        raise ValueError(f"unknown method {method!r}; use 'leiden' or 'louvain'")
    labels = _relabel_by_size(labels)
    if group_singletons:
        labels = _relabel_by_size(_group_singletons(labels, G))
    return labels


def umap(emb, n_neighbors: int = 30, min_dist: float = 0.3, metric: str = "cosine", seed: int = 42,
         n_epochs: int | None = None, spread: float = 1.0, init: str = "spectral", knn=None) -> np.ndarray:
    """UMAP with Seurat ``RunUMAP`` defaults (cosine, 30 neighbours, min_dist 0.3).

    The k-NN graph is computed exactly by arsc (``knn`` can pass a precomputed
    ``(idx, dist)``), so only the layout optimisation runs in umap-learn.
    ``seed`` (default 42 = Seurat ``RunUMAP(seed.use = 42)``) seeds umap-learn's
    ``random_state`` and ``transform_seed``; with a seed umap-learn runs its layout
    single-threaded, so the same input + seed gives bit-identical coordinates. Pass other
    integers to see alternative layouts. ``seed=None`` (multi-threaded, non-reproducible)
    is rejected.

    Parameters
    ----------
    emb : (n x d) embedding (PCs, harmony).
    n_neighbors : neighbours (Seurat ``n.neighbors``).
    min_dist : layout ``min_dist``.
    metric : ``"cosine"`` (Seurat default) or ``"euclidean"`` (exact arsc k-NN); others go to umap-learn.
    seed : integer seed.
    n_epochs : optimisation epochs (umap-learn default when None).
    spread : layout ``spread``.
    init : ``"spectral"`` or ``"random"``.
    knn : optional precomputed ``(idx, dist)``.

    Returns
    -------
    numpy.ndarray (n x 2) coordinates.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> Z, _, _ = arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))
    >>> xy = arsc.umap(arsc.pp.pca(Z, n_comps=10).embedding)   # doctest: +SKIP
    >>> xy.shape                                                # doctest: +SKIP
    (3000, 2)
    """
    seed = check_seed(seed)
    import umap as umap_learn

    from .neighbors import knn as _knn, l2_normalize

    X = np.asarray(emb, dtype=np.float64)
    if knn is None:
        if metric == "cosine":
            idx, d = _knn(l2_normalize(X), n_neighbors, seed=seed)
            d = d ** 2 / 2.0  # euclidean on unit vectors -> cosine distance
        elif metric == "euclidean":
            idx, d = _knn(X, n_neighbors, seed=seed)
        else:
            idx = d = None
    else:
        idx, d = knn
    kw = dict(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, init=init,
              n_epochs=n_epochs, random_state=seed, transform_seed=seed)
    if idx is not None:
        kw["precomputed_knn"] = (idx, d, None)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return umap_learn.UMAP(**kw).fit_transform(X)
