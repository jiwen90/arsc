"""Graph clustering and UMAP wrappers with deterministic seeds.

``cluster(graph, method="louvain"|"leiden")`` clusters an SNN/WNN graph
(Seurat ``FindClusters``: labels are renumbered by decreasing cluster size,
singletons are merged into their most connected cluster). ``umap()`` gives a
layout with Seurat ``RunUMAP`` defaults on arsc's k-NN (:func:`arsc.knn`, ``method="auto"``).

Both run arsc's own engines by default (``engine="arsc"``, since 0.3.0; validated in
``SEURAT_PARITY.md`` section 5). The reference engines stay available:
``cluster(engine="leidenalg")`` and ``umap(engine="umap-learn")`` reproduce results computed
with arsc 0.2.x, whose defaults were those engines (and Leiden at resolution 0.8).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._utils import check_seed
from ._seedlog import records_seed

__all__ = ["cluster", "umap"]


def _check_graph(graph):
    if isinstance(graph, dict):
        raise TypeError(f"cluster expects a graph matrix, got a dict with keys {list(graph)}; "
                        "pass find_neighbors(...)['snn'] (or wnn(...).wsnn)")
    if not sp.issparse(graph) and not isinstance(graph, np.ndarray):
        raise TypeError(f"cluster expects a cells x cells sparse graph, got {type(graph).__name__}; "
                        "pass find_neighbors(...)['snn'] (or wnn(...).wsnn)")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"cluster expects a square cells x cells graph, got shape {graph.shape}; "
                         "build one with find_neighbors(embedding)['snn']")
    G = sp.csr_matrix(graph)
    return G.maximum(G.T).tocsr()  # symmetric


def _to_igraph(G):
    import igraph as ig
    U = sp.triu(G, k=1).tocoo()
    g = ig.Graph(n=G.shape[0], edges=np.column_stack([U.row, U.col]), directed=False)
    g.es["weight"] = U.data
    return g


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    uniq, inv, counts = np.unique(np.asarray(labels), return_inverse=True, return_counts=True)
    order = np.lexsort((uniq, -counts))
    remap = np.empty(uniq.size, dtype=np.int64)
    remap[order] = np.arange(uniq.size)
    return remap[inv.ravel()]


def _group_singletons(labels: np.ndarray, G: sp.csr_matrix) -> np.ndarray:
    from ._leiden import group_singletons
    return group_singletons(labels, G)


@records_seed("seed")
def cluster(graph, method: str = "louvain", resolution: float = 0.6, seed: int = 0, n_iterations: int = 2,
            n_start: int = 10, group_singletons: bool = True, engine: str = "arsc",
            objective: str = "modularity") -> np.ndarray:
    """Cluster a weighted graph (e.g. ``find_neighbors(...)["snn"]`` or ``wnn(...).wsnn``).

    Seurat ``FindClusters``: labels are renumbered by decreasing cluster size (0 = largest) and
    singletons are merged into their most connected cluster.

    Defaults (since 0.3.0): Louvain at resolution 0.6 (Seurat ``FindClusters(resolution = 0.6,
    algorithm = 1)``, the setting of the CD8 T-cell workflow arsc was validated on) on arsc's own engine.

    Two engines run the same algorithms. ``engine="arsc"`` (default) runs arsc's own numba ports
    (:mod:`arsc._leiden`: libleidenalg's optimiser and igraph's multilevel, step for step) with no
    igraph / leidenalg import; it is several times faster, its labels do not depend on the thread count,
    and its quality matches the references within their own seed-to-seed range (``SEURAT_PARITY.md``),
    but a seed gives a different (equally good) partition than the same seed in leidenalg.
    ``engine="leidenalg"`` calls leidenalg / igraph; pass it (with ``method`` and ``resolution``) to
    reproduce labels computed with arsc 0.2.x, whose defaults were ``method="leiden"``,
    ``resolution=0.8``, ``engine="leidenalg"``.

    Parameters
    ----------
    graph : n x n sparse weighted graph.
    method : ``"louvain"`` (default; igraph multilevel, best modularity over ``n_start`` seeded runs, as
        Seurat ``algorithm = 1``) or ``"leiden"`` (RBConfiguration = modularity with resolution, as Seurat
        ``algorithm = 4``).
    resolution : modularity resolution (Seurat ``resolution``; higher = more clusters; default 0.6).
    n_iterations : Leiden iterations (leidenalg default 2; -1 = until stable).
    n_start : Louvain random starts (Seurat ``n.start``).
    group_singletons : merge singleton clusters (Seurat ``group.singletons``).
    seed : integer RNG seed (default 0 = Seurat ``FindClusters(random.seed = 0)``). leidenalg engine:
        Leiden uses leidenalg's own seeded RNG; Louvain start ``s`` uses a private
        ``random.Random(seed + s)`` installed as igraph's RNG for the call (igraph's default RNG is
        restored afterwards; Python's global ``random`` state is never touched). arsc engine: numpy
        ``SeedSequence(seed)`` (Louvain start ``s``: its ``s``-th child); the result does not depend on
        the thread count. Same graph + seed -> identical labels; vary ``seed`` to check a clustering's
        stability.
    engine : ``"arsc"`` (default, own implementation) or ``"leidenalg"`` (leidenalg / igraph).
    objective : Leiden quality, ``"modularity"`` (default) or ``"cpm"`` (constant Potts model;
        leidenalg ``CPMVertexPartition``; needs ``method="leiden"``).

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
    if method not in ("leiden", "louvain"):
        raise ValueError(f"unknown method {method!r}; use 'leiden' or 'louvain'")
    if objective not in ("modularity", "cpm"):
        raise ValueError(f"unknown objective {objective!r}; use 'modularity' or 'cpm'")
    if method == "louvain" and objective != "modularity":
        raise ValueError("louvain optimises modularity only; pass method=\"leiden\" for objective=\"cpm\"")
    G = _check_graph(graph)
    if engine == "arsc":
        from . import _leiden
        H = G.copy()
        H.setdiag(0)
        H.eliminate_zeros()
        H.sort_indices()
        gr = (H.indptr.astype(np.int64), H.indices.astype(np.int64), H.data.astype(np.float64))
        if method == "leiden":
            labels = _leiden.leiden(gr, resolution=resolution, seed=seed, n_iterations=n_iterations,
                                    objective=objective)
        else:
            labels = _leiden.louvain(gr, resolution=resolution, seed=seed, n_start=n_start)
    elif engine == "leidenalg":
        g = _to_igraph(G)
        if method == "leiden":
            import leidenalg as la
            kind = la.RBConfigurationVertexPartition if objective == "modularity" else la.CPMVertexPartition
            part = la.find_partition(g, kind, weights="weight", resolution_parameter=resolution, seed=seed,
                                     n_iterations=n_iterations)
            labels = np.asarray(part.membership)
        else:
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
        raise ValueError(f"unknown engine {engine!r}; use 'leidenalg' or 'arsc'")
    labels = _relabel_by_size(labels)
    if group_singletons:
        labels = _relabel_by_size(_group_singletons(labels, G))
    return labels


@records_seed("seed")
def umap(emb, n_neighbors: int = 30, min_dist: float = 0.3, metric: str = "cosine", seed: int = 42,
         n_epochs: int | None = None, spread: float = 1.0, init: str = "spectral", knn=None,
         engine: str = "arsc") -> np.ndarray:
    """UMAP with Seurat ``RunUMAP`` defaults (cosine, 30 neighbours, min_dist 0.3).

    The k-NN graph comes from :func:`arsc.knn` with its default (exact up to 4,096 cells, pynndescent above, as
    umap-learn and uwot search approximately); pass ``knn=arsc.knn(emb, n_neighbors, method="exact")`` (cosine:
    on L2-normalised rows) for exact neighbours at any size.
    ``seed`` (default 42 = Seurat ``RunUMAP(seed.use = 42)``) seeds the layout; the same input + seed
    gives bit-identical coordinates. Pass other integers to see alternative layouts. ``seed=None``
    is rejected.

    ``engine="arsc"`` (default since 0.3.0) runs arsc's own UMAP (``arsc._umap``): the same fuzzy
    graph as umap-learn (identical weights on the same kNN), a spectral initialisation, and a layout
    optimiser that runs on all threads yet gives identical coordinates at any thread count (each epoch
    reads a snapshot; edges are partitioned by head point; negative samples come from per-edge
    counter-based random streams). It is several times faster; layouts differ from umap-learn's by
    about as much as two umap-learn seeds differ from each other (``SEURAT_PARITY.md``, UMAP engines).
    It supports ``metric="cosine"`` or ``"euclidean"`` (or a precomputed ``knn``).
    ``engine="umap-learn"`` runs the layout in umap-learn (seeded, single-threaded; ``random_state``
    and ``transform_seed`` = ``seed``) and accepts its other metrics; pass it to reproduce layouts
    computed with arsc 0.2.x, whose default it was.

    Parameters
    ----------
    emb : (n x d) embedding (PCs, harmony).
    n_neighbors : neighbours (Seurat ``n.neighbors``).
    min_dist : layout ``min_dist``.
    metric : ``"cosine"`` (Seurat default) or ``"euclidean"`` (arsc k-NN); others go to umap-learn.
    seed : integer seed.
    n_epochs : optimisation epochs (umap-learn default when None).
    spread : layout ``spread``.
    init : ``"spectral"`` or ``"random"``.
    knn : optional precomputed ``(idx, dist)``.
    engine : ``"arsc"`` (default) or ``"umap-learn"``.

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
    if engine not in ("umap-learn", "arsc"):
        raise ValueError(f"unknown engine {engine!r}; use 'umap-learn' or 'arsc'")
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
    if engine == "arsc":
        if idx is None:
            raise ValueError(f"engine='arsc' supports metric 'cosine' or 'euclidean' (or a precomputed knn), "
                             f"got {metric!r}; pass engine='umap-learn' for other metrics")
        from . import _umap
        G = _umap.fuzzy_simplicial_set(idx, d, n_neighbors)
        return _umap.embed(G, 2, min_dist=min_dist, spread=spread, n_epochs=n_epochs, init=init, seed=seed, X=X)
    import umap as umap_learn
    kw = dict(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, init=init,
              n_epochs=n_epochs, random_state=seed, transform_seed=seed)
    if idx is not None:
        kw["precomputed_knn"] = (idx, d, None)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return umap_learn.UMAP(**kw).fit_transform(X)
