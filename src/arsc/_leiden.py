"""Leiden and Louvain community detection (numba), arsc's own implementation.

Ports of the reference algorithms, step for step:

* **Leiden** follows libleidenalg's ``Optimiser`` with leidenalg's defaults (the engine behind
  ``leidenalg.find_partition``): fast local moving with a queue (``move_nodes``: neighbor
  communities plus one empty community, a move needs an improvement above ``10 * DBL_EPSILON``),
  refinement by ``merge_nodes_constrained`` with ``RAND_NEIGH_COMM`` (a still-singleton node picks a
  random neighbor inside its community, with probability k / (k + 1), and joins that neighbor's
  refined community when the move does not lower the quality), aggregation on the refined partition
  with the unrefined partition as the starting membership, repeated while the graph shrinks;
  ``n_iterations`` such passes (-1: until a pass no longer improves).
* **Louvain** follows igraph's ``community_multilevel``: per level a random node order, passes in
  that order moving each node to the neighbor community with the largest positive gain until a
  pass no longer raises modularity, then aggregation; stops when a level merges nothing.

Quality functions: modularity with a resolution (leidenalg ``RBConfigurationVertexPartition``,
igraph ``modularity``) and the constant Potts model (``CPMVertexPartition``; node size 1).

The random numbers come from a splitmix64 stream seeded through numpy's PCG64 from the integer
seed, so a seed gives the same partition on every machine; the algorithms are sequential (the
Louvain starts run in parallel, each with its own stream, and the best is chosen by quality, then by
start index), so the result does not depend on the thread count. A seed does not reproduce
leidenalg's or igraph's partition (different random numbers); parity is judged on quality and on
agreement within the reference's own seed-to-seed variability (``SEURAT_PARITY.md``).
"""
from __future__ import annotations

import numba as nb
import numpy as np
import scipy.sparse as sp

__all__ = ["leiden", "louvain", "quality", "prepare_graph"]

_EPS10 = 10 * np.finfo(np.float64).eps
MODULARITY = 0
CPM = 1


# --------------------------------------------------------------------------- RNG (splitmix64)
@nb.njit(cache=True, inline="always")
def _next(state):
    state[0] = (state[0] + np.uint64(0x9E3779B97F4A7C15))
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@nb.njit(cache=True, inline="always")
def _randint(state, n):
    """Uniform integer in [0, n)."""
    return np.int64((_next(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0) * n)


@nb.njit(cache=True)
def _shuffle(a, state):
    for i in range(a.shape[0] - 1, 0, -1):
        j = _randint(state, i + 1)
        t = a[i]
        a[i] = a[j]
        a[j] = t


def _state(seed_seq) -> np.ndarray:
    return np.random.default_rng(seed_seq).integers(0, 2 ** 63, size=1, dtype=np.uint64)


# --------------------------------------------------------------------------- graph helpers
def prepare_graph(graph):
    """A symmetric weighted CSR graph without self-loops, as arsc's clustering uses it.

    ``max(G, G.T)``, the diagonal dropped (the same edges ``arsc.cluster`` gives igraph / leidenalg).

    Parameters
    ----------
    graph : n x n sparse (or dense) non-negative weights, e.g. ``find_neighbors(...)["snn"]``.

    Returns
    -------
    tuple ``(indptr, indices, data)`` (int64, int64, float64), rows sorted by column.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from arsc._leiden import prepare_graph
    >>> p, i, w = prepare_graph(sp.csr_matrix(np.array([[1., 2.], [0., 0.]])))
    >>> p.tolist(), i.tolist(), w.tolist()
    ([0, 1, 2], [1, 0], [2.0, 2.0])
    """
    G = sp.csr_matrix(graph, dtype=np.float64)
    if G.shape[0] != G.shape[1]:
        raise ValueError(f"expected a square cells x cells graph, got shape {G.shape}")
    G = G.maximum(G.T).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    G.sort_indices()
    if G.nnz and G.data.min() < 0:
        raise ValueError("graph weights must be non-negative")
    return G.indptr.astype(np.int64), G.indices.astype(np.int64), G.data.astype(np.float64)


@nb.njit(cache=True)
def _self_weights(indptr, indices, data):
    n = indptr.shape[0] - 1
    s = np.zeros(n)
    for v in range(n):
        for e in range(indptr[v], indptr[v + 1]):
            if indices[e] == v:
                s[v] += data[e]
    return s


@nb.njit(cache=True)
def _strength(indptr, indices, data):
    """igraph strength: a self-loop counts twice."""
    n = indptr.shape[0] - 1
    k = np.zeros(n)
    for v in range(n):
        acc = 0.0
        for e in range(indptr[v], indptr[v + 1]):
            acc += data[e] * (2.0 if indices[e] == v else 1.0)
        k[v] = acc
    return k


@nb.njit(cache=True)
def _renumber(memb):
    """Contiguous ids by first appearance."""
    n = memb.shape[0]
    mx = 0
    for v in range(n):
        if memb[v] > mx:
            mx = memb[v]
    new = np.full(mx + 1, -1, np.int64)
    out = np.empty(n, np.int64)
    c = 0
    for v in range(n):
        m = memb[v]
        if new[m] < 0:
            new[m] = c
            c += 1
        out[v] = new[m]
    return out, c


@nb.njit(cache=True)
def _renumber_by_size(memb, ncomm):
    """libleidenalg ``renumber_communities``: ids by decreasing size, ties by old id."""
    cnt = np.zeros(ncomm, np.int64)
    for v in range(memb.shape[0]):
        cnt[memb[v]] += 1
    order = np.argsort(-cnt, kind="mergesort")
    new = np.empty(ncomm, np.int64)
    c = 0
    for i in range(ncomm):
        if cnt[order[i]] > 0:
            new[order[i]] = c
            c += 1
    out = np.empty_like(memb)
    for v in range(memb.shape[0]):
        out[v] = new[memb[v]]
    return out, c


@nb.njit(cache=True)
def _collapse(indptr, indices, data, size, memb, ncomm):
    """Aggregate graph of a partition (contiguous ids): rows sorted by column, internal weight as a
    self-loop (each internal edge counted once, plus the members' self-loops)."""
    n = indptr.shape[0] - 1
    # nodes grouped by community
    cnt = np.zeros(ncomm + 1, np.int64)
    for v in range(n):
        cnt[memb[v] + 1] += 1
    for c in range(ncomm):
        cnt[c + 1] += cnt[c]
    members = np.empty(n, np.int64)
    pos = cnt[:-1].copy()
    for v in range(n):
        members[pos[memb[v]]] = v
        pos[memb[v]] += 1
    acc = np.zeros(ncomm)
    seen = np.zeros(ncomm, np.bool_)
    touched = np.empty(ncomm, np.int64)
    nsize = np.zeros(ncomm)
    # pass 1: counts
    ptr = np.zeros(ncomm + 1, np.int64)
    for c in range(ncomm):
        nt = 0
        for a in range(cnt[c], cnt[c + 1]):
            v = members[a]
            for e in range(indptr[v], indptr[v + 1]):
                d = memb[indices[e]]
                if not seen[d]:
                    seen[d] = True
                    touched[nt] = d
                    nt += 1
        for t in range(nt):
            seen[touched[t]] = False
        ptr[c + 1] = ptr[c] + nt
    out_i = np.empty(ptr[ncomm], np.int64)
    out_w = np.empty(ptr[ncomm])
    for c in range(ncomm):
        nt = 0
        for a in range(cnt[c], cnt[c + 1]):
            v = members[a]
            nsize[c] += size[v]
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                d = memb[u]
                if not seen[d]:
                    seen[d] = True
                    touched[nt] = d
                    nt += 1
                if d == c:
                    # an internal edge appears twice (both rows) unless it is a self-loop
                    acc[d] += data[e] * (1.0 if u == v else 0.5)
                else:
                    acc[d] += data[e]
        ts = np.sort(touched[:nt])
        b = ptr[c]
        for t in range(nt):
            d = ts[t]
            out_i[b + t] = d
            out_w[b + t] = acc[d]
            acc[d] = 0.0
            seen[d] = False
    return ptr, out_i, out_w, nsize


# --------------------------------------------------------------------------- Leiden
@nb.njit(cache=True, inline="always")
def _diff(kind, gamma, two_m, w_new, w_old_rest, k_v, s_v, K_new, K_old_rest, N_new, N_old_rest):
    """libleidenalg ``diff_move`` (undirected: twice the one-direction change)."""
    if kind == MODULARITY:
        return 2.0 * ((w_new - w_old_rest) - gamma * k_v * (K_new - K_old_rest) / two_m)
    return 2.0 * ((w_new - w_old_rest) - gamma * s_v * (N_new - N_old_rest))


@nb.njit(cache=True)
def _move_nodes(indptr, indices, data, k, size, memb, kind, gamma, two_m, state):
    """libleidenalg ``Optimiser::move_nodes`` (ALL_NEIGH_COMMS, consider_empty_community)."""
    n = indptr.shape[0] - 1
    K = np.zeros(n)
    N = np.zeros(n)
    cnt = np.zeros(n, np.int64)
    for v in range(n):
        c = memb[v]
        K[c] += k[v]
        N[c] += size[v]
        cnt[c] += 1
    empty = np.empty(n, np.int64)
    ne = 0
    for c in range(n - 1, -1, -1):
        if cnt[c] == 0:
            empty[ne] = c
            ne += 1
    order = np.arange(n)
    _shuffle(order, state)
    queue = np.empty(n, np.int64)
    queue[:] = order
    head = 0
    qlen = n
    stable = np.zeros(n, np.bool_)
    neigh_w = np.zeros(n)
    added = np.zeros(n, np.bool_)
    comms = np.empty(n + 1, np.int64)
    total = 0.0
    while qlen > 0:
        v = queue[head]
        head += 1
        if head == n:
            head = 0
        qlen -= 1
        c0 = memb[v]
        nc = 0
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            if u == v:
                continue
            c = memb[u]
            neigh_w[c] += data[e]
            if not added[c]:
                added[c] = True
                comms[nc] = c
                nc += 1
        if cnt[c0] > 1:  # the empty community is a candidate
            ec = empty[ne - 1]
            comms[nc] = ec
            nc += 1
        kv = k[v]
        sv = size[v]
        w_old = neigh_w[c0]
        K_old = K[c0] - kv
        N_old = N[c0] - sv
        best = c0
        best_imp = _EPS10
        for t in range(nc):
            c = comms[t]
            if c == c0:
                continue
            imp = _diff(kind, gamma, two_m, neigh_w[c], w_old, kv, sv, K[c], K_old, N[c], N_old)
            if imp > best_imp:
                best_imp = imp
                best = c
        for t in range(nc):
            c = comms[t]
            neigh_w[c] = 0.0
            added[c] = False
        stable[v] = True
        if best != c0:
            total += best_imp
            K[c0] -= kv
            N[c0] -= sv
            cnt[c0] -= 1
            if cnt[best] == 0:
                ne -= 1  # best is the empty community on top of the stack
            K[best] += kv
            N[best] += sv
            cnt[best] += 1
            memb[v] = best
            if cnt[c0] == 0:
                empty[ne] = c0
                ne += 1
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                if u != v and stable[u] and memb[u] != best:
                    stable[u] = False
                    tail = head + qlen
                    if tail >= n:
                        tail -= n
                    queue[tail] = u
                    qlen += 1
    return total


@nb.njit(cache=True)
def _merge_constrained(indptr, indices, data, k, size, constraint, kind, gamma, two_m, state):
    """libleidenalg ``merge_nodes_constrained`` with ``RAND_NEIGH_COMM`` on a singleton partition."""
    n = indptr.shape[0] - 1
    ref = np.arange(n)
    K = k.copy()
    N = size.copy()
    cnt = np.ones(n, np.int64)
    order = np.arange(n)
    _shuffle(order, state)
    cand = np.empty(indices.shape[0] + n, np.int64)
    for v in order:
        c0 = ref[v]
        if cnt[c0] != 1:
            continue
        nk = 0
        cv = constraint[v]
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            if constraint[u] == cv:
                cand[nk] = ref[u]
                nk += 1
                if u == v:  # igraph lists an undirected self-loop twice
                    cand[nk] = c0
                    nk += 1
        if nk == 0:
            continue
        if _randint(state, nk + 1) == 0:
            continue
        c = cand[_randint(state, nk)]
        if c == c0:
            continue
        w_new = 0.0
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            if u != v and ref[u] == c:
                w_new += data[e]
        kv = k[v]
        sv = size[v]
        imp = _diff(kind, gamma, two_m, w_new, 0.0, kv, sv, K[c], K[c0] - kv, N[c], N[c0] - sv)
        if imp >= 0.0:
            K[c0] -= kv
            N[c0] -= sv
            cnt[c0] -= 1
            K[c] += kv
            N[c] += sv
            cnt[c] += 1
            ref[v] = c
    return ref


@nb.njit(cache=True)
def _leiden_pass(indptr, indices, data, memb0, kind, gamma, state):
    """One ``Optimiser::optimise_partition`` call; ``memb0`` (contiguous ids) is updated in place."""
    n0 = indptr.shape[0] - 1
    ip, ix, w = indptr, indices, data
    size = np.ones(n0)
    k = _strength(ip, ix, w)
    two_m = k.sum()
    if two_m <= 0:
        return 0.0
    memb_l = memb0.copy()
    agg = np.arange(n0)
    improv = 0.0
    while True:
        improv += _move_nodes(ip, ix, w, k, size, memb_l, kind, gamma, two_m, state)
        memb_l, ncomm = _renumber(memb_l)
        for v in range(n0):
            memb0[v] = memb_l[agg[v]]
        nl = ip.shape[0] - 1
        ref = _merge_constrained(ip, ix, w, k, size, memb_l, kind, gamma, two_m, state)
        ref, nref = _renumber_by_size(ref, nl)
        for v in range(n0):
            agg[v] = ref[agg[v]]
        ip2, ix2, w2, size2 = _collapse(ip, ix, w, size, ref, nref)
        memb_new = np.empty(nref, np.int64)
        for v in range(nl):
            memb_new[ref[v]] = memb_l[v]
        further = (nref < nl) and (nl > ncomm)
        ip, ix, w, size = ip2, ix2, w2, size2
        k = _strength(ip, ix, w)
        memb_l, _ = _renumber(memb_new)
        if not further:
            break
    return improv


def _check_kind(objective):
    if objective in ("modularity", "rb", "RBConfiguration"):
        return MODULARITY
    if objective in ("cpm", "CPM"):
        return CPM
    raise ValueError(f"unknown objective {objective!r}; use 'modularity' or 'cpm'")


def leiden(graph, resolution: float = 1.0, seed: int = 0, n_iterations: int = 2,
           objective: str = "modularity", initial=None) -> np.ndarray:
    """Leiden partition of a weighted undirected graph (port of leidenalg's default optimizer).

    Parameters
    ----------
    graph : n x n sparse weights (symmetrized with ``max(G, G.T)``; the diagonal is ignored), or the
        ``(indptr, indices, data)`` of :func:`prepare_graph`.
    resolution : resolution parameter (modularity: ``RBConfigurationVertexPartition``; CPM: density).
    seed : integer seed.
    n_iterations : optimizer passes (leidenalg ``n_iterations``; -1 = until a pass does not improve).
    objective : ``"modularity"`` or ``"cpm"``.
    initial : optional starting membership (default: singletons).

    Returns
    -------
    int64 labels, numbered by decreasing community size (0 = largest).

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from arsc._leiden import leiden
    >>> A = sp.block_diag([np.ones((5, 5)), np.ones((4, 4))]).tocsr()
    >>> leiden(A, seed=0).tolist()
    [0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    kind = _check_kind(objective)
    ip, ix, w = graph if isinstance(graph, tuple) else prepare_graph(graph)
    n = ip.shape[0] - 1
    memb = np.arange(n, dtype=np.int64) if initial is None else _renumber(np.asarray(initial, np.int64))[0]
    state = _state(np.random.SeedSequence(int(seed)))
    it = 0
    while True:
        diff = _leiden_pass(ip, ix, w, memb, kind, float(resolution), state)
        it += 1
        if n_iterations < 0:
            if not diff > 0:
                break
        elif it >= n_iterations:
            break
    out, nc = _renumber(memb)
    return _renumber_by_size(out, nc)[0]


# --------------------------------------------------------------------------- Louvain (igraph multilevel)
@nb.njit(cache=True)
def _modularity(indptr, indices, data, memb, ncomm, gamma, two_m):
    I = np.zeros(ncomm)
    K = np.zeros(ncomm)
    n = indptr.shape[0] - 1
    for v in range(n):
        c = memb[v]
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            wt = data[e]
            if u == v:
                I[c] += 2.0 * wt
                K[c] += 2.0 * wt
            else:
                K[c] += wt
                if memb[u] == c:
                    I[c] += wt
    q = 0.0
    for c in range(ncomm):
        q += I[c] - gamma * K[c] * K[c] / two_m
    return q / two_m


@nb.njit(cache=True)
def _louvain_step(ip, ix, w, k, gamma, two_m, state):
    """igraph ``igraph_i_community_multilevel_step``: returns the level membership and modularity.

    The modularity after each pass is updated from per-community totals (internal weight ``I``,
    strength ``K``) instead of a sweep over all edges."""
    n = ip.shape[0] - 1
    order = np.arange(n)
    _shuffle(order, state)
    memb = np.arange(n)
    K = k.copy()
    selfw = _self_weights(ip, ix, w)
    I = 2.0 * selfw
    q = _modularity(ip, ix, w, memb, n, gamma, two_m)
    saved = np.empty(n, np.int64)
    savedK = np.empty(n)
    savedI = np.empty(n)
    links_w = np.zeros(n)
    added = np.zeros(n, np.bool_)
    comms = np.empty(n, np.int64)
    while True:
        pass_q = q
        changed = False
        saved[:] = memb
        savedK[:] = K
        savedI[:] = I
        for i in range(n):
            v = order[i]
            nc = 0
            for e in range(ip[v], ip[v + 1]):
                u = ix[e]
                if u == v:
                    continue
                c = memb[u]
                links_w[c] += w[e]
                if not added[c]:
                    added[c] = True
                    comms[nc] = c
                    nc += 1
            c0 = memb[v]
            kv = k[v]
            K[c0] -= kv
            I[c0] -= 2.0 * (links_w[c0] + selfw[v])
            best = c0
            best_gain = 0.0
            comms[:nc].sort()  # igraph sorts the links by community (in place: no allocation)
            for t in range(nc):
                c = comms[t]
                gain = links_w[c] - gamma * K[c] * kv / two_m
                if gain > best_gain:
                    best_gain = gain
                    best = c
            K[best] += kv
            I[best] += 2.0 * (links_w[best] + selfw[v])
            for t in range(nc):
                links_w[comms[t]] = 0.0
                added[comms[t]] = False
            memb[v] = best
            if best != c0:
                changed = True
        q = 0.0
        for c in range(n):
            q += I[c] - gamma * K[c] * K[c] / two_m
        q /= two_m
        if not (changed and q > pass_q):
            memb[:] = saved
            K[:] = savedK
            I[:] = savedI
            q = pass_q
            break
    return memb, q


@nb.njit(cache=True)
def _louvain_one(indptr, indices, data, gamma, state):
    n0 = indptr.shape[0] - 1
    ip, ix, w = indptr, indices, data
    size = np.ones(n0)
    k = _strength(ip, ix, w)
    two_m = k.sum()
    level = np.arange(n0)
    if two_m <= 0:
        return level, 0.0
    q = _modularity(ip, ix, w, level, n0, gamma, two_m)
    while True:
        nl = ip.shape[0] - 1
        prev_q = q
        memb, q = _louvain_step(ip, ix, w, k, gamma, two_m, state)
        memb, nc = _renumber(memb)
        if nc == nl or q < prev_q:
            q = prev_q
            break
        ip, ix, w, size = _collapse(ip, ix, w, size, memb, nc)
        k = _strength(ip, ix, w)
        for v in range(n0):
            level[v] = memb[level[v]]
    return level, q


@nb.njit(parallel=True, cache=True)
def _louvain_starts(indptr, indices, data, gamma, states):
    S = states.shape[0]
    n = indptr.shape[0] - 1
    out = np.empty((S, n), np.int64)
    qs = np.empty(S)
    for s in nb.prange(S):
        st = states[s].copy()
        m, q = _louvain_one(indptr, indices, data, gamma, st)
        out[s] = m
        qs[s] = q
    return out, qs


def louvain(graph, resolution: float = 1.0, seed: int = 0, n_start: int = 1) -> np.ndarray:
    """Louvain partition (port of igraph ``community_multilevel``), best modularity of ``n_start`` starts.

    Parameters
    ----------
    graph : n x n sparse weights (symmetrized, diagonal ignored) or :func:`prepare_graph` output.
    resolution : modularity resolution.
    seed : integer seed; start ``s`` uses the ``s``-th child of ``numpy.random.SeedSequence(seed)``.
    n_start : random starts (Seurat ``n.start``); they run in parallel and the highest modularity wins
        (ties: the lowest start), so the result does not depend on the thread count.

    Returns
    -------
    int64 labels, numbered by decreasing community size (0 = largest).

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from arsc._leiden import louvain
    >>> A = sp.block_diag([np.ones((5, 5)), np.ones((4, 4))]).tocsr()
    >>> louvain(A, seed=0, n_start=2).tolist()
    [0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    ip, ix, w = graph if isinstance(graph, tuple) else prepare_graph(graph)
    kids = np.random.SeedSequence(int(seed)).spawn(max(1, int(n_start)))
    states = np.stack([_state(c) for c in kids])
    out, qs = _louvain_starts(ip, ix, w, float(resolution), states)
    best = int(np.flatnonzero(qs == qs.max())[0])
    lab, nc = _renumber(out[best])
    return _renumber_by_size(lab, nc)[0]


def quality(graph, labels, resolution: float = 1.0, objective: str = "modularity") -> float:
    """Quality of a partition: modularity with resolution (igraph ``modularity``) or CPM.

    Modularity is igraph's ``Graph.modularity`` (leidenalg's ``RBConfigurationVertexPartition.quality()``
    divided by the total edge weight 2m); CPM is leidenalg's ``CPMVertexPartition.quality()``: twice the
    internal edge weight minus ``resolution * n_c (n_c - 1)``, summed over communities.

    Parameters
    ----------
    graph : n x n sparse weights (symmetrized, diagonal ignored) or :func:`prepare_graph` output.
    labels : community per node.
    resolution : resolution parameter.
    objective : ``"modularity"`` or ``"cpm"``.

    Returns
    -------
    float.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from arsc._leiden import quality
    >>> A = sp.block_diag([np.ones((5, 5)), np.ones((4, 4))]).tocsr()
    >>> round(quality(A, [0] * 5 + [1] * 4), 4)
    0.4688
    """
    kind = _check_kind(objective)
    ip, ix, w = graph if isinstance(graph, tuple) else prepare_graph(graph)
    lab, nc = _renumber(np.asarray(labels, np.int64))
    if kind == MODULARITY:
        k = _strength(ip, ix, w)
        return float(_modularity(ip, ix, w, lab, nc, float(resolution), k.sum()))
    n = ip.shape[0] - 1
    rows = np.repeat(np.arange(n), np.diff(ip))
    inside = float(w[lab[rows] == lab[ix]].sum()) / 2.0
    sz = np.bincount(lab).astype(float)
    return 2.0 * (inside - float(resolution) * float((sz * (sz - 1) / 2).sum()))


# --------------------------------------------------------------------------- post-processing
@nb.njit(cache=True)
def _group_singletons_nb(labels, indptr, indices, data, nlab):
    """Seurat ``GroupSingletons``: each singleton joins the cluster it is most connected to
    (sequentially, ties to the lowest label; clusters of size 1 at the start are never targets)."""
    sizes = np.zeros(nlab, np.int64)
    for v in range(labels.shape[0]):
        sizes[labels[v]] += 1
    nsingle = 0
    for v in range(labels.shape[0]):
        if sizes[labels[v]] == 1:
            nsingle += 1
    if nsingle == 0 or nsingle == labels.shape[0]:
        return labels
    conn = np.zeros(nlab)
    for i in range(labels.shape[0]):
        if sizes[labels[i]] != 1:
            continue
        for c in range(nlab):
            conn[c] = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            conn[labels[indices[e]]] += data[e]
        best = -1
        bv = -np.inf
        for c in range(nlab):
            v = conn[c] if sizes[c] > 1 else -1.0
            if v > bv:
                bv = v
                best = c
        labels[i] = best
    return labels


def group_singletons(labels, G) -> np.ndarray:
    """Merge singleton clusters into their most connected cluster (Seurat ``GroupSingletons``).

    Parameters
    ----------
    labels : 0..K-1 integer labels.
    G : the symmetric graph (CSR) the labels came from (diagonal kept, as arsc.cluster passes it).

    Returns
    -------
    int64 labels (a copy).

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from arsc._leiden import group_singletons
    >>> G = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0.]]))
    >>> group_singletons(np.array([0, 0, 1]), G).tolist()
    [0, 0, 0]
    """
    G = sp.csr_matrix(G)
    lab = np.asarray(labels, np.int64).copy()
    return _group_singletons_nb(lab, G.indptr.astype(np.int64), G.indices.astype(np.int64),
                                G.data.astype(np.float64), int(lab.max()) + 1)
