"""Subset stores as views: a store whose cells are rows of other (parent) stores keeps only what is new.

A *view* is an ordinary store directory (``manifest.json``, ``cells.txt``, ``obs``, reductions, graphs, neighbors,
``uns``, dense layers, features, every layer whose values differ) **without** the sparse layer arrays whose rows equal
rows of a parent store. ``view.json`` records, for each such layer, where every row comes from: a row of a parent
layer (with a feature map) or a row kept in the view itself (``view/<assay>__<layer>.local.*``).

``arsc.open(view)`` returns a normal :class:`arsc.Store` with the same API and the same values, bit for bit, as the
full store it replaces (``compare`` checks every accessor; ``tests/test_views.py``). The parents' arrays are
memory-mapped (zero-copy, shared through the page cache) and only the view's rows are read. A row subset of a CSR
matrix cannot itself be a zero-copy ``scipy.sparse`` matrix, so ``X()`` on an inherited layer gathers the view's rows
into RAM once per process (read-only arrays, cached while the caller holds the matrix); ``get_features`` reads only
the requested parent columns; ``subset`` gathers only the requested rows.

Layout added to a store directory::

    view.json                      parents (relative path + identity), per inherited layer its row sources
    view/src-<h>.npy, pos-<h>.npy  per view row: source index (int16; -1 = local) and the row in that source (int64)
    view/cols-<h>.npy              per source: the parent column of every view feature (int32; -1 = not in parent)
    view/<A>__<L>.local.{csr,csc}.{data,indices,indptr}.npy   rows kept in the view (in view row order)

``manifest.json`` and ``cells.txt`` stay byte-identical to the full store's, so anything keyed on them (artifact
hashes) keeps resolving. Parents must be full stores (no chains of views).

Build with :func:`build` (from the full store and its parents, with the row candidates), check with :func:`compare`,
turn back into a full store with :func:`materialize`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import weakref
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

VIEW_FILE = "view.json"
VIEW_DIR = "view"
FORMAT = "arsc-view"
VERSION = 1
LOCAL = -1

__all__ = ["VIEW_FILE", "is_view", "build", "compare", "materialize", "ViewError", "parent_paths", "logical_bytes",
           "array_sha256"]


class ViewError(RuntimeError):
    """A view's parent is missing or changed, or a view cannot be built.

    Attributes
    ----------
    args : the message (as for any exception).

    Examples
    --------
    >>> from arsc.views import ViewError
    >>> issubclass(ViewError, RuntimeError)
    True
    """


def is_view(path) -> bool:
    """True when ``path`` is a store directory holding a view (``view.json``).

    Parameters
    ----------
    path : a store directory.

    Returns
    -------
    bool.

    Examples
    --------
    >>> import arsc, tempfile
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> arsc.views.is_view(t["view"]), arsc.views.is_view(t["full"])
    (True, False)
    """
    return (Path(path) / VIEW_FILE).is_file()


def _read_spec(path) -> dict:
    spec = json.loads((Path(path) / VIEW_FILE).read_text(encoding="utf-8"))
    if spec.get("format") != FORMAT or int(spec.get("version", 0)) > VERSION:
        raise ViewError(f"{path}: unsupported view format {spec.get('format')!r} v{spec.get('version')} "
                        f"(this arsc reads {FORMAT} v{VERSION}; upgrade arsc)")
    return spec


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _candidates(view_dir: Path, rel: str) -> list:
    """Where a parent may be: relative to the view, and relative to the stores/ folder when the view itself was moved
    into stores/.superseded/ (one level deeper)."""
    out = [view_dir / rel]
    if view_dir.parent.name == ".superseded":
        out.append(view_dir.parent / rel)
    return out


def parent_paths(path) -> list:
    """Resolved parent store directories of a view (``[]`` for a full store). Raises ViewError when one is missing.

    Parameters
    ----------
    path : a store directory.

    Returns
    -------
    list of Path.

    Examples
    --------
    >>> import arsc, tempfile
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> [p.name for p in arsc.views.parent_paths(t["view"])]
    ['parent']
    """
    path = Path(path)
    if not is_view(path):
        return []
    spec = _read_spec(path)
    return [_resolve_parent(path, p) for p in spec["parents"]]


def _resolve_parent(view_dir: Path, p: dict) -> Path:
    tried = []
    for c in _candidates(view_dir, p["relpath"]):
        c = Path(os.path.normpath(c))
        tried.append(str(c))
        if (c / "manifest.json").is_file():
            return c
    raise ViewError(f"view {view_dir.name}: its parent store {p['name']!r} is not at {' or '.join(tried)}. A view needs "
                    "its parent store (mount or restore it; arsc.views.materialize needs it too)")


def logical_bytes(path) -> int:
    """Bytes the full store would take on disk (recorded at build; the store's own size for a full store).

    Parameters
    ----------
    path : a store directory.

    Returns
    -------
    int.

    Examples
    --------
    >>> import arsc, tempfile
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> arsc.views.logical_bytes(t["view"]) > 0
    True
    """
    path = Path(path)
    if is_view(path):
        b = (_read_spec(path).get("bytes") or {}).get("full")
        if b:
            return int(b)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- kernels
@nb.njit(parallel=True, cache=True)
def _row_lengths(indptr, indices, rows, fwd, out):
    for t in nb.prange(rows.size):
        r = rows[t]
        c = 0
        for p in range(indptr[r], indptr[r + 1]):
            if fwd[indices[p]] >= 0:
                c += 1
        out[t] = c


@nb.njit(parallel=True, cache=True)
def _fill(indptr, indices, data, rows, fwd, dest, out_indptr, out_indices, out_data, need_sort):
    for t in nb.prange(rows.size):
        r = rows[t]
        s = out_indptr[dest[t]]
        o = s
        for p in range(indptr[r], indptr[r + 1]):
            j = fwd[indices[p]]
            if j >= 0:
                out_indices[o] = j
                out_data[o] = data[p]
                o += 1
        if need_sort and o - s > 1:
            seg_i = out_indices[s:o].copy()
            seg_x = out_data[s:o].copy()
            order = np.argsort(seg_i)
            for q in range(o - s):
                out_indices[s + q] = seg_i[order[q]]
                out_data[s + q] = seg_x[order[q]]


@nb.njit(parallel=True, cache=True)
def _rows_equal(a_ptr, a_idx, a_bits, b_ptr, b_idx, b_bits, b_rows, out):
    """out[t] = row t of A (gathered, 0-based rows) equals row b_rows[t] of B, bit for bit."""
    for t in nb.prange(out.size):
        s, e = a_ptr[t], a_ptr[t + 1]
        r = b_rows[t]
        s2, e2 = b_ptr[r], b_ptr[r + 1]
        ok = (e - s) == (e2 - s2)
        if ok:
            for q in range(e - s):
                if a_idx[s + q] != b_idx[s2 + q] or a_bits[s + q] != b_bits[s2 + q]:
                    ok = False
                    break
        out[t] = ok


@nb.njit(parallel=True, cache=True)
def _col_lengths(cptr, cidx, cols, inv, out):
    for t in nb.prange(cols.size):
        j = cols[t]
        c = 0
        if j >= 0:
            for p in range(cptr[j], cptr[j + 1]):
                if inv[cidx[p]] >= 0:
                    c += 1
        out[t] += c


@nb.njit(parallel=True, cache=True)
def _col_fill(cptr, cidx, cdata, cols, inv, fillpos, out_rows, out_data):
    for t in nb.prange(cols.size):
        j = cols[t]
        if j < 0:
            continue
        o = fillpos[t]
        for p in range(cptr[j], cptr[j + 1]):
            v = inv[cidx[p]]
            if v >= 0:
                out_rows[o] = v
                out_data[o] = cdata[p]
                o += 1
        fillpos[t] = o


@nb.njit(parallel=True, cache=True)
def _sort_segments(ptr, keys, vals):
    for t in nb.prange(ptr.size - 1):
        s, e = ptr[t], ptr[t + 1]
        if e - s > 1:
            sorted_ok = True
            for q in range(s + 1, e):
                if keys[q - 1] > keys[q]:
                    sorted_ok = False
                    break
            if sorted_ok:
                continue
            k = keys[s:e].copy()
            v = vals[s:e].copy()
            order = np.argsort(k)
            for q in range(e - s):
                keys[s + q] = k[order[q]]
                vals[s + q] = v[order[q]]


def _uint_view(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _fwd_map(cols: np.ndarray, n_parent_features: int) -> np.ndarray:
    """parent column -> view column (-1 when the parent column is not a view feature)."""
    fwd = np.full(n_parent_features, -1, dtype=np.int64)
    ok = cols >= 0
    fwd[cols[ok]] = np.flatnonzero(ok)
    return fwd


def _monotonic(cols: np.ndarray) -> bool:
    c = cols[cols >= 0]
    return bool(c.size < 2 or np.all(np.diff(c) > 0))


# --------------------------------------------------------------------------- runtime
class _Hold:
    """Keeps a gathered matrix's arrays together; the cache holds it weakly."""
    __slots__ = ("arrays", "__weakref__")

    def __init__(self, arrays):
        self.arrays = arrays


class _Layer:
    def __init__(self, view: "View", key: str, e: dict):
        self.view = weakref.proxy(view)  # no View <-> _Layer cycle (it would keep the memory maps open until gc)
        self.key, self.e = key, e
        self.shape = tuple(e["shape"])
        self.dtype = {k: np.dtype(v) for k, v in e["dtype"].items()}
        self.csc_dtype = {k: np.dtype(v) for k, v in e["csc_dtype"].items()}
        d = view.store.path
        self.src = np.load(d / e["src"], mmap_mode="r")
        self.pos = np.load(d / e["pos"], mmap_mode="r")
        self.sources = e["sources"]
        self._cols = {}
        self._inv = {}
        self._local_rows = None

    # -- sources
    def parent_matrix(self, k: int, fmt: str = "csr"):
        s = self.sources[k]
        P = self.view.parent(s["parent"])
        return P.X(s["assay"], s["layer"]) if fmt == "csr" else P.X_csc(s["assay"], s["layer"])

    def cols(self, k: int) -> np.ndarray:
        if k not in self._cols:
            s = self.sources[k]
            if s.get("cols"):
                self._cols[k] = np.load(self.view.store.path / s["cols"], mmap_mode="r")
            else:
                self._cols[k] = np.arange(self.shape[1], dtype=np.int32)
        return self._cols[k]

    def local(self, fmt: str = "csr"):
        lo = self.e.get("local")
        if not lo:
            return None
        from .io import _mmap_sparse
        base = str(self.view.store.path / lo["prefix"])
        return _mmap_sparse(base + f".{fmt}", (lo["n_rows"], self.shape[1]), fmt)

    def local_rows(self) -> np.ndarray:
        if self._local_rows is None:
            self._local_rows = np.flatnonzero(np.asarray(self.src) == LOCAL).astype(np.int64)
        return self._local_rows

    def inv(self, k: int) -> np.ndarray:
        """parent row -> view row for the rows of this layer that come from source k (-1 elsewhere)."""
        if k not in self._inv:
            P = self.view.parent(self.sources[k]["parent"])
            inv = np.full(P.n_cells, -1, dtype=np.int64)
            rows = np.flatnonzero(np.asarray(self.src) == k)
            inv[np.asarray(self.pos)[rows]] = rows
            self._inv[k] = inv
        return self._inv[k]

    # -- rows
    def gather(self, vrows: np.ndarray):
        """(indptr int64, indices, data) of view rows ``vrows`` in this layer's dtypes."""
        vrows = np.asarray(vrows, dtype=np.int64)
        src = np.asarray(self.src)[vrows]
        pos = np.asarray(self.pos)[vrows]
        lens = np.zeros(vrows.size, dtype=np.int64)
        groups = []
        for k in list(range(len(self.sources))) + [LOCAL]:
            dest = np.flatnonzero(src == k)
            if not dest.size:
                continue
            if k == LOCAL:
                M = self.local("csr")
                fwd = np.arange(self.shape[1], dtype=np.int64)
                sort = False
            else:
                M = self.parent_matrix(k)
                cols = np.asarray(self.cols(k))
                fwd = _fwd_map(cols, M.shape[1])
                sort = not _monotonic(cols)
            rows = pos[dest].astype(np.int64)
            ln = np.empty(dest.size, dtype=np.int64)
            _row_lengths(M.indptr, M.indices, rows, fwd, ln)
            lens[dest] = ln
            groups.append((M, rows, fwd, dest, sort))
        indptr = np.zeros(vrows.size + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, dtype=self.dtype["indices"])
        data = np.empty(nnz, dtype=self.dtype["data"])
        for M, rows, fwd, dest, sort in groups:
            _fill(M.indptr, M.indices, M.data, rows, fwd, dest, indptr, indices, data, sort)
        return indptr, indices, data

    def X(self) -> sp.csr_matrix:
        h = self.view.cache_get(("csr", self.key))
        if h is None:
            indptr, indices, data = self.gather(np.arange(self.shape[0], dtype=np.int64))
            indptr = indptr.astype(self.dtype["indptr"], copy=False)
            for a in (indptr, indices, data):
                a.flags.writeable = False
            h = self.view.cache_put(("csr", self.key), (data, indices, indptr))
        data, indices, indptr = h.arrays
        m = sp.csr_matrix((data, indices, indptr), shape=self.shape, copy=False)
        m.has_sorted_indices = True
        m._arsc_view_hold = h  # the cache entry lives as long as the matrix
        return m

    def take(self, vrows: np.ndarray) -> sp.csr_matrix:
        """Same arrays and dtypes as ``io._take_rows(full.X(), vrows)``."""
        indptr, indices, data = self.gather(vrows)
        nnz = int(indptr[-1])
        if nnz < 2**31:
            indptr = indptr.astype(np.int32)
            indices = indices.astype(np.int32, copy=False)
        res = sp.csr_matrix((data, indices, indptr), shape=(len(vrows), self.shape[1]), copy=False)
        res.has_sorted_indices = True
        return res

    def X_csc(self) -> sp.csc_matrix:
        h = self.view.cache_get(("csc", self.key))
        if h is None:
            m = self.X()
            arrays = _csr_to_csc_arrays(m, self.csc_dtype)
            for a in arrays:
                a.flags.writeable = False
            h = self.view.cache_put(("csc", self.key), arrays)
        data, indices, indptr = h.arrays
        m = sp.csc_matrix((data, indices, indptr), shape=self.shape, copy=False)
        m.has_sorted_indices = True
        m._arsc_view_hold = h
        return m

    # -- columns (get_features)
    def col_parts(self, fidx: np.ndarray):
        """(ptr int64, rows, data) of view columns ``fidx``: rows ascending per column, in the full store's CSC dtypes."""
        fidx = np.asarray(fidx, dtype=np.int64)
        lens = np.zeros(fidx.size, dtype=np.int64)
        plan = []
        for k in range(len(self.sources)):
            if not np.any(np.asarray(self.src) == k):
                continue
            C = self.parent_matrix(k, "csc")
            pc = np.asarray(self.cols(k))[fidx].astype(np.int64)
            inv = self.inv(k)
            _col_lengths(C.indptr, C.indices, pc, inv, lens)
            plan.append((C, pc, inv))
        lo = self.local("csc")
        if lo is not None:
            inv = self.local_rows()  # local row -> view row
            _col_lengths(lo.indptr, lo.indices, fidx, inv, lens)
            plan.append((lo, fidx, inv))
        ptr = np.zeros(fidx.size + 1, dtype=np.int64)
        np.cumsum(lens, out=ptr[1:])
        rows = np.empty(int(ptr[-1]), dtype=np.int64)
        data = np.empty(int(ptr[-1]), dtype=self.dtype["data"])
        fillpos = ptr[:-1].copy()
        for C, pc, inv in plan:
            _col_fill(C.indptr, C.indices, C.data, pc, inv, fillpos, rows, data)
        _sort_segments(ptr, rows, data)
        return ptr, rows.astype(self.csc_dtype["indices"], copy=False), data


class View:
    """Runtime side of a view, attached to a :class:`arsc.Store` as ``st._view``."""

    def __init__(self, store, spec: dict):
        self._store = weakref.ref(store)  # no Store <-> View cycle: the store's memory maps close when it is dropped
        self.spec = spec
        self._parents = {}
        self._layers = {}
        self._cache = {}

    @property
    def store(self):
        st = self._store()
        if st is None:
            raise ViewError("the store of this view was closed")
        return st

    @property
    def inherited(self) -> list:
        return list(self.spec.get("layers") or {})

    def has(self, assay: str, layer: str) -> bool:
        return f"{assay}/{layer}" in (self.spec.get("layers") or {})

    def layer(self, assay: str, layer: str) -> _Layer:
        key = f"{assay}/{layer}"
        if key not in self._layers:
            self._layers[key] = _Layer(self, key, self.spec["layers"][key])
        return self._layers[key]

    def parent(self, i: int):
        if i not in self._parents:
            from .io import Store
            p = self.spec["parents"][i]
            path = _resolve_parent(self.store.path, p)
            if is_view(path):
                raise ViewError(f"view {self.store.path.name}: parent {path} is itself a view (chains are not allowed)")
            P = Store(path, finalize=False)
            if P.n_cells != int(p["n_cells"]):
                raise ViewError(f"view {self.store.path.name}: parent {path} has {P.n_cells} cells, the view was built "
                                f"on {p['n_cells']}")
            want = p.get("manifest_sha256")
            if want and _sha256_file(path / "manifest.json") != want:
                raise ViewError(f"view {self.store.path.name}: parent {path} changed since the view was built "
                                "(manifest.json differs); rebuild or materialize the view against the old parent")
            for key, sizes in (p.get("layer_bytes") or {}).items():
                a, l = key.split("/", 1)
                base = path / "assays" / a
                got = [(base / f"{l}.csr.{c}.npy").stat().st_size if (base / f"{l}.csr.{c}.npy").exists() else -1
                       for c in ("data", "indices", "indptr")]
                if got != list(sizes):
                    raise ViewError(f"view {self.store.path.name}: parent {path} layer {key} changed size since the "
                                    "view was built")
            self._parents[i] = P
        return self._parents[i]

    def cache_get(self, key):
        r = self._cache.get(key)
        return r() if r is not None else None

    def cache_put(self, key, arrays) -> _Hold:
        h = _Hold(arrays)
        self._cache[key] = weakref.ref(h)
        return h


def attach(store) -> View | None:
    """Called by ``Store.__init__``: a View for a view directory, else None."""
    if not is_view(store.path):
        return None
    return View(store, _read_spec(store.path))


@nb.njit(parallel=True, cache=True)
def _csc_counts_mem(indptr, indices, n_cols, n_chunks):
    n_rows = indptr.size - 1
    step = (n_rows + n_chunks - 1) // n_chunks
    counts = np.zeros((n_chunks, n_cols), dtype=np.int64)
    for c in nb.prange(n_chunks):
        lo = c * step
        hi = min(n_rows, lo + step)
        for i in range(lo, hi):
            for p in range(indptr[i], indptr[i + 1]):
                counts[c, indices[p]] += 1
    return counts


def _csr_to_csc_arrays(m: sp.csr_matrix, dt: dict):
    """In-memory equivalent of ``io._build_csc`` (same algorithm and dtypes)."""
    from .io import _csr_to_csc_scatter
    n_rows, n_cols = m.shape
    indptr, indices = np.asarray(m.indptr), np.asarray(m.indices)
    n_chunks = max(1, nb.get_num_threads())
    counts = _csc_counts_mem(indptr, indices, n_cols, n_chunks)
    colptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(counts.sum(0), out=colptr[1:])
    offsets = (colptr[:-1][None, :] + np.cumsum(counts, axis=0) - counts).astype(np.int64)
    nnz = int(colptr[-1])
    idx_dtype = np.int32 if max(nnz, n_rows) < 2**31 else np.int64
    out_i = np.empty(nnz, dtype=idx_dtype)
    out_x = np.empty(nnz, dtype=np.asarray(m.data).dtype)
    _csr_to_csc_scatter(indptr, indices, np.asarray(m.data), offsets, out_i, out_x, n_chunks)
    colptr = colptr.astype(idx_dtype)
    # the full store's files carry these dtypes (recorded at build); they equal _build_csc's rule
    return out_x.astype(dt["data"], copy=False), out_i.astype(dt["indices"], copy=False), \
        colptr.astype(dt["indptr"], copy=False)


# --------------------------------------------------------------------------- build
def _layer_files(d: Path, assay: str, layer: str) -> list:
    base = d / "assays" / assay
    return [base / f"{layer}.{f}.{c}.npy" for f in ("csr", "csc") for c in ("data", "indices", "indptr")]


def _save(path: Path, arr) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        np.lib.format.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)
    return path.name


def _save_once(vd: Path, prefix: str, arr: np.ndarray) -> str:
    """Save under a content-addressed name (layers with the same row map share one file) -> 'view/<name>'."""
    arr = np.ascontiguousarray(arr)
    name = f"{prefix}-{hashlib.sha256(arr.reshape(-1).view(np.uint8)).hexdigest()[:16]}.npy"
    if not (vd / name).exists():
        _save(vd / name, arr)
    return f"{VIEW_DIR}/{name}"


def _parent_layer_candidates(F, assay: str, layer: str, P) -> list:
    """(assay, layer) of the parent to try for a full-store layer: same assay and layer first, then the same layer
    name in another assay, then the other layers; only assays sharing at least half of the features."""
    feats = F.features(assay)
    out = []
    for pa in P.assays:
        pf = P.features(pa)
        if not pf.is_unique:
            continue
        shared = int(pf.get_indexer(feats).__ge__(0).sum())
        if shared < max(1, len(feats) // 2):
            continue
        for pl in P.layers(pa):
            info = P.manifest["assays"][pa]["layers"][pl]
            real = info.get("alias_of") or pl
            rank = (pa != assay, pl != layer, pl)
            out.append((rank, pa, real))
    seen, res = set(), []
    for _, pa, pl in sorted(out):
        if (pa, pl) not in seen:
            seen.add((pa, pl))
            res.append((pa, pl))
    return res


def _match_layer(F, assay, layer, parents, rows_by_parent, chunk=1 << 20, sample=2000, log=print):
    """For each view row: (source index, row) of an equal parent row, or LOCAL. -> (src, pos, sources)."""
    info = F.manifest["assays"][assay]["layers"][layer]
    X = F.X(assay, layer)
    n = X.shape[0]
    src = np.full(n, LOCAL, dtype=np.int16)
    pos = np.zeros(n, dtype=np.int64)
    sources = []
    dt_data = np.asarray(X.data).dtype
    dt_idx = np.asarray(X.indices).dtype
    fb = _uint_view(X.data)
    feats = F.features(assay)
    for pi, P in enumerate(parents):
        vr_all, pr_all = rows_by_parent.get(pi, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        if not len(vr_all):
            continue
        for pa, pl in _parent_layer_candidates(F, assay, layer, P):
            todo = src[vr_all] == LOCAL
            vr, pr = vr_all[todo], pr_all[todo]
            if not vr.size:
                break
            M = P.X(pa, pl)
            cols = P.features(pa).get_indexer(feats).astype(np.int32)
            fwd = _fwd_map(cols, M.shape[1])
            sort = not _monotonic(cols)

            def eq(vrows, prows):
                ln = np.empty(prows.size, dtype=np.int64)
                _row_lengths(M.indptr, M.indices, prows, fwd, ln)
                ptr = np.zeros(prows.size + 1, dtype=np.int64)
                np.cumsum(ln, out=ptr[1:])
                gi = np.empty(int(ptr[-1]), dtype=dt_idx)
                gx = np.empty(int(ptr[-1]), dtype=dt_data)
                _fill(M.indptr, M.indices, M.data, prows, fwd, np.arange(prows.size, dtype=np.int64), ptr, gi, gx, sort)
                out = np.empty(prows.size, dtype=np.bool_)
                _rows_equal(ptr, gi, _uint_view(gx), X.indptr, X.indices, fb, vrows, out)
                return out

            probe = np.linspace(0, vr.size - 1, min(sample, vr.size)).astype(np.int64)
            if not eq(vr[probe], pr[probe]).any():
                log(f"  {assay}/{layer}: parent {pi} {pa}/{pl}: no equal row in a sample of {probe.size}")
                continue
            ok = np.zeros(vr.size, dtype=bool)
            for s in range(0, vr.size, chunk):
                ok[s:s + chunk] = eq(vr[s:s + chunk], pr[s:s + chunk])
            if not ok.any():
                continue
            k = len(sources)
            identity = len(cols) == M.shape[1] and np.array_equal(cols, np.arange(len(cols)))
            sources.append({"parent": pi, "assay": pa, "layer": pl, "cols_arr": None if identity else cols,
                            "rows": int(ok.sum())})
            src[vr[ok]] = k
            pos[vr[ok]] = pr[ok]
            log(f"  {assay}/{layer}: {int(ok.sum()):,} of {vr.size:,} candidate rows equal parent {pi} {pa}/{pl}")
    return src, pos, sources, info


def build(full, parents, out, rows, *, min_fraction: float = 0.1, min_bytes: int = 0, dry_run: bool = False,
          log=print) -> dict:
    """Build a view of the full store ``full`` on ``parents``.

    Parameters
    ----------
    full : path of the full store (or an open Store) whose content the view must reproduce exactly.
    parents : list of parent store paths (full stores, not views).
    out : the new view directory (must not exist; ignored with ``dry_run``).
    rows : ``{parent index: (view_rows, parent_rows)}`` candidate pairs (int arrays). Each pair is only a candidate:
        a layer inherits a row only when the parent row equals it bit for bit (indices and values, after the
        feature map and the full store's dtype).
    min_fraction : a layer inherits from parents only when at least this fraction of its rows matched (else it stays
        whole in the view).
    min_bytes : no view when it would save fewer bytes than this (the report says so).
    dry_run : only match and report.

    log : a function taking one message (default print).

    Returns
    -------
    dict report: per layer the rows matched per source, whether it is inherited and the bytes saved; ``view``
    (built or not, with ``why_not``), ``out`` and the full and view sizes.

    Examples
    --------
    >>> import arsc, tempfile, numpy as np
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> rep = arsc.views.build(t["full"], [t["parent"]], None, {0: (np.arange(300), t["rows"])}, dry_run=True,
    ...                        log=lambda *a: None)
    >>> rep["layers"]["RNA/counts"]["matched"], rep["layers"]["RNA/counts"]["inherit"]
    (300, True)
    """
    from .io import Store
    F = full if isinstance(full, Store) else Store(full, finalize=False)
    if is_view(F.path):
        raise ViewError(f"{F.path} is already a view")
    P = []
    for p in parents:
        p = Path(p)
        if is_view(p):
            raise ViewError(f"parent {p} is a view; build views on full stores only")
        P.append(Store(p, finalize=False))
    rows_by_parent = {}
    claimed = np.zeros(F.n_cells, dtype=bool)
    for pi, (vr, pr) in rows.items():
        vr, pr = np.asarray(vr, dtype=np.int64), np.asarray(pr, dtype=np.int64)
        if vr.size != pr.size:
            raise ViewError("rows: view_rows and parent_rows differ in length")
        if vr.size and (vr.min() < 0 or vr.max() >= F.n_cells or pr.min() < 0 or pr.max() >= P[pi].n_cells):
            raise ViewError(f"rows for parent {pi}: index out of range")
        if np.unique(vr).size != vr.size or np.unique(pr).size != pr.size:
            raise ViewError(f"rows for parent {pi}: a view row or parent row appears twice")
        rows_by_parent[int(pi)] = (vr, pr)
        claimed[vr] = True
    report = {"full": str(F.path), "parents": [str(p.path) for p in P], "n_cells": F.n_cells,
              "rows_with_candidate": int(claimed.sum()), "layers": {}, "bytes_saved": 0}
    plan = {}
    for a in F.assays:
        for l, info in (F.manifest["assays"][a].get("layers") or {}).items():
            if info.get("alias_of"):
                continue
            src, pos, sources, _ = _match_layer(F, a, l, P, rows_by_parent, log=log)
            matched = int((src != LOCAL).sum())
            frac = matched / max(1, F.n_cells)
            files = _layer_files(F.path, a, l)
            lbytes = sum(f.stat().st_size for f in files if f.exists())
            X = F.X(a, l)
            inherit = matched > 0 and frac >= min_fraction
            if inherit:
                lr = np.flatnonzero(src == LOCAL)
                local_nnz = int(np.sum(np.asarray(X.indptr)[lr + 1] - np.asarray(X.indptr)[lr])) if lr.size else 0
                saved = int(lbytes * (1 - local_nnz / max(1, int(X.nnz)))) if X.nnz else lbytes
            else:
                saved = 0
            report["layers"][f"{a}/{l}"] = {
                "rows": F.n_cells, "matched": matched, "fraction": round(frac, 6), "inherit": bool(inherit),
                "sources": [{k: v for k, v in s.items() if k != "cols_arr"} for s in sources],
                "layer_bytes": lbytes, "bytes_saved": saved}
            report["bytes_saved"] += saved
            if inherit:
                plan[(a, l)] = (src, pos, sources)
    report["view"] = bool(plan) and report["bytes_saved"] >= min_bytes
    if dry_run or not report["view"]:
        if not report["view"]:
            report["why_not"] = ("no layer has rows equal to a parent" if not plan else
                                 f"saves {report['bytes_saved']:,} bytes < min_bytes {min_bytes:,}")
        return report
    out = Path(out)
    if out.exists():
        raise ViewError(f"{out} exists")
    tmp = out.with_name(out.name + f".building{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    skip = {f.resolve() for (a, l) in plan for f in _layer_files(F.path, a, l)}
    t0 = time.time()
    for f in sorted(F.path.rglob("*")):
        if f.is_dir() or f.name.endswith(".lock") or f.resolve() in skip:
            continue
        if f.name == VIEW_FILE or f.relative_to(F.path).parts[0] == VIEW_DIR:
            raise ViewError(f"{F.path} holds view files already")
        dst = tmp / f.relative_to(F.path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
    vd = tmp / VIEW_DIR
    vd.mkdir(parents=True, exist_ok=True)
    spec = {"format": FORMAT, "version": VERSION, "parents": [], "layers": {}}
    for pi, p in enumerate(P):
        rel = os.path.relpath(p.path.resolve(), out.resolve())
        lb = {}
        for (a, l), (_, _, sources) in plan.items():
            for s in sources:
                if s["parent"] == pi:
                    base = p.path / "assays" / s["assay"]
                    lb[f"{s['assay']}/{s['layer']}"] = [(base / f"{s['layer']}.csr.{c}.npy").stat().st_size
                                                        for c in ("data", "indices", "indptr")]
        spec["parents"].append({"name": p.path.name, "relpath": Path(rel).as_posix(), "n_cells": p.n_cells,
                                "manifest_sha256": _sha256_file(p.path / "manifest.json"), "layer_bytes": lb})
    from .io import _build_csc
    for (a, l), (src, pos, sources) in plan.items():
        X = F.X(a, l)
        C = F.X_csc(a, l)
        e = {"shape": list(X.shape), "nnz": int(X.nnz),
             "dtype": {"data": str(np.asarray(X.data).dtype), "indices": str(np.asarray(X.indices).dtype),
                       "indptr": str(np.asarray(X.indptr).dtype)},
             "csc_dtype": {"data": str(np.asarray(C.data).dtype), "indices": str(np.asarray(C.indices).dtype),
                           "indptr": str(np.asarray(C.indptr).dtype)},
             # the arrays' dtypes in the full store's files (scipy may narrow an index array when it opens them)
             "file_dtype": {f"{fm}.{c}": str(np.load(fp, mmap_mode="r").dtype)
                            for fm in ("csr", "csc") for c in ("data", "indices", "indptr")
                            for fp in [F.path / "assays" / a / f"{l}.{fm}.{c}.npy"] if fp.exists()}}
        lr = np.flatnonzero(src == LOCAL)
        pos = pos.copy()
        pos[lr] = np.arange(lr.size, dtype=np.int64)  # a local row's position in the view's local CSR
        e["src"] = _save_once(vd, "src", src)
        e["pos"] = _save_once(vd, "pos", pos)
        srcs = []
        for s in sources:
            d = {"parent": s["parent"], "assay": s["assay"], "layer": s["layer"], "rows": s["rows"], "cols": None}
            if s["cols_arr"] is not None:
                d["cols"] = _save_once(vd, "cols", s["cols_arr"])
            srcs.append(d)
        e["sources"] = srcs
        if lr.size:
            from .io import _take_rows
            Lm = _take_rows(X, lr)
            prefix = f"{VIEW_DIR}/{_safe(a)}__{_safe(l)}.local"
            base = tmp / prefix
            ip = np.asarray(Lm.indptr).astype(np.int64)
            _save(Path(str(base) + ".csr.data.npy"), np.asarray(Lm.data))
            _save(Path(str(base) + ".csr.indices.npy"), np.asarray(Lm.indices))
            _save(Path(str(base) + ".csr.indptr.npy"), ip)
            _build_csc(str(base) + ".csr", str(base) + ".csc", (lr.size, X.shape[1]))
            e["local"] = {"prefix": prefix, "n_rows": int(lr.size), "nnz": int(Lm.nnz)}
        spec["layers"][f"{a}/{l}"] = e
    full_bytes = sum(f.stat().st_size for f in F.path.rglob("*") if f.is_file())
    view_bytes = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
    spec["bytes"] = {"full": full_bytes, "view": view_bytes}
    try:
        from . import __version__ as ver
    except ImportError:  # pragma: no cover
        ver = "?"
    spec["built"] = {"arsc": ver, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": F.path.name,
                     "seconds": round(time.time() - t0, 1)}
    (tmp / VIEW_FILE).write_text(json.dumps(spec, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    report["out"] = str(out)
    report["bytes_full"], report["bytes_view"] = full_bytes, view_bytes
    return report


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


# --------------------------------------------------------------------------- parity
def _arr_eq(a, b, chunk=1 << 26) -> tuple:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype == object:
        return (bool(np.array_equal(a, b)), "object values differ")
    fa, fb = a.reshape(-1), b.reshape(-1)
    ua, ub = _uint_view(np.ascontiguousarray(fa)), _uint_view(np.ascontiguousarray(fb))
    for s in range(0, ua.size, chunk):
        if not np.array_equal(ua[s:s + chunk], ub[s:s + chunk]):
            i = s + int(np.flatnonzero(ua[s:s + chunk] != ub[s:s + chunk])[0])
            return False, f"first difference at {i}: {fa[i]!r} vs {fb[i]!r}"
    return True, ""


def _sparse_eq(a, b) -> tuple:
    if type(a) is not type(b):
        return False, f"type {type(a).__name__} vs {type(b).__name__}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    for c in ("indptr", "indices", "data"):
        ok, why = _arr_eq(getattr(a, c), getattr(b, c))
        if not ok:
            return False, f"{c}: {why}"
    if bool(a.has_sorted_indices) != bool(b.has_sorted_indices):
        return False, "has_sorted_indices differs"
    return True, ""


def _frame_eq(a, b) -> tuple:
    if a is None or b is None:
        return (a is None and b is None), "one is None"
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError as e:
        return False, str(e)[:300]
    return True, ""


def _deep_eq(a, b, path="") -> tuple:
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return False, f"{path}: keys {list(a)[:5]} vs {list(b)[:5]}"
        for k in a:
            ok, why = _deep_eq(a[k], b[k], f"{path}/{k}")
            if not ok:
                return ok, why
        return True, ""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False, f"{path}: length/type"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, why = _deep_eq(x, y, f"{path}[{i}]")
            if not ok:
                return ok, why
        return True, ""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False, f"{path}: array vs {type(b).__name__}"
        ok, why = _arr_eq(a, b)
        return ok, f"{path}: {why}"
    if isinstance(a, pd.DataFrame):
        ok, why = _frame_eq(a, b)
        return ok, f"{path}: {why}"
    if sp.issparse(a):
        ok, why = _sparse_eq(a, b)
        return ok, f"{path}: {why}"
    if type(a) is not type(b):
        return False, f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, float) and np.isnan(a) and np.isnan(b):
        return True, ""
    return (a == b), f"{path}: {a!r} vs {b!r}"


def compare(view, full, *, feature_batch: int = 4096, subset_n: int = 2000, seed: int = 0, log=None) -> list:
    """Compare a view with the full store it replaces on every accessor, bit for bit.

    Returns a list of ``{"check", "ok", "detail"}``; all ``ok`` means the view is a drop-in replacement. Checked:
    manifest, cells, n_cells, obs, assays, layers, features, var, varm, variable_features, dense layers, splits,
    neighbors, idents, unconvertible, uns, reductions (embedding, loadings, projected, stdev, key, assay, columns),
    graphs, ``X`` and ``X_csc`` of every layer (aliases included), ``get_features`` of every feature (sparse) and of
    one batch (dense), ``subset`` of random cells (every layer, obs, reductions) and the parent store files' sizes.

    Parameters
    ----------
    view : the view (path or Store).
    full : the full store it replaces (path or Store).
    feature_batch : features per ``get_features`` call.
    subset_n : random cells for the ``subset`` check.
    seed : integer seed of that draw.
    log : optional function called with each failure.

    Returns
    -------
    list of dict (``check``, ``ok``, ``detail``).

    Examples
    --------
    >>> import arsc, tempfile
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> res = arsc.views.compare(t["view"], t["full"])
    >>> all(r["ok"] for r in res), len(res) > 30
    (True, True)
    """
    from .io import Store
    V = view if isinstance(view, Store) else Store(view, finalize=False)
    Fs = full if isinstance(full, Store) else Store(full, finalize=False)
    res = []

    def add(what, ok, detail=""):
        res.append({"check": what, "ok": bool(ok), "detail": "" if ok else str(detail)[:400]})
        if log and not ok:
            log(f"  FAIL {what}: {detail}")

    def run(what, fn):
        try:
            ok, why = fn()
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"{e.__class__.__name__}: {e}"
        add(what, ok, why)

    add("the view is a view", getattr(V, "_view", None) is not None, "no view.json")
    run("manifest", lambda: _deep_eq(V.manifest, Fs.manifest))
    run("manifest.json bytes", lambda: ((V.path / "manifest.json").read_bytes() == (Fs.path / "manifest.json").read_bytes(), "differ"))
    run("cells.txt bytes", lambda: ((V.path / "cells.txt").read_bytes() == (Fs.path / "cells.txt").read_bytes(), "differ"))
    run("n_cells", lambda: (V.n_cells == Fs.n_cells, f"{V.n_cells} vs {Fs.n_cells}"))
    run("cells", lambda: (V.cells.equals(Fs.cells) and V.cells.dtype == Fs.cells.dtype, "differ"))
    run("obs", lambda: _frame_eq(V.obs, Fs.obs))
    run("assays", lambda: (V.assays == Fs.assays, f"{V.assays} vs {Fs.assays}"))
    run("uns", lambda: _deep_eq(V.uns, Fs.uns))
    run("unconvertible", lambda: (V.unconvertible == Fs.unconvertible, "differ"))
    run("idents", lambda: ((V.idents is None and Fs.idents is None) or V.idents.equals(Fs.idents), "differ"))
    run("reductions", lambda: (V.reductions == Fs.reductions, "differ"))
    for r in Fs.reductions:
        def red(r=r):
            a, b = V.reduction(r), Fs.reduction(r)
            for c in ("embedding", "stdev"):
                x, y = getattr(a, c), getattr(b, c)
                if (x is None) != (y is None):
                    return False, f"{c} None"
                if x is not None:
                    ok, why = _arr_eq(x, y)
                    if not ok:
                        return False, f"{c}: {why}"
            for c in ("loadings", "projected"):
                ok, why = _frame_eq(getattr(a, c), getattr(b, c))
                if not ok:
                    return False, f"{c}: {why}"
            return (a.columns == b.columns and a.assay == b.assay and a.key == b.key), "columns/assay/key"
        run(f"reduction {r}", red)
        run(f"embedding {r}", lambda r=r: _arr_eq(V.embedding(r), Fs.embedding(r)))
    run("graphs", lambda: (V.graphs == Fs.graphs, "differ"))
    for g in Fs.graphs:
        run(f"graph {g}", lambda g=g: _sparse_eq(V.graph(g), Fs.graph(g)))
    run("neighbors", lambda: (V.neighbors == Fs.neighbors, "differ"))
    for nn in Fs.neighbors:
        def nb_(nn=nn):
            (i1, d1), (i2, d2) = V.neighbor(nn), Fs.neighbor(nn)
            ok, why = _arr_eq(i1, i2)
            if not ok or (d1 is None) != (d2 is None):
                return False, why or "dist None"
            return _arr_eq(d1, d2) if d1 is not None else (True, "")
        run(f"neighbor {nn}", nb_)
    rng = np.random.default_rng(seed)
    sub = rng.choice(Fs.n_cells, min(subset_n, Fs.n_cells), replace=False) if Fs.n_cells else np.zeros(0, int)
    for a in Fs.assays:
        run(f"{a}: layers", lambda a=a: (V.layers(a) == Fs.layers(a), "differ"))
        run(f"{a}: features", lambda a=a: (V.features(a).equals(Fs.features(a)), "differ"))
        run(f"{a}: var", lambda a=a: _frame_eq(V.var(a), Fs.var(a)))
        run(f"{a}: variable_features", lambda a=a: (V.variable_features(a) == Fs.variable_features(a), "differ"))
        for k in (Fs.manifest["assays"][a].get("varm") or {}):
            run(f"{a}: varm {k}", lambda a=a, k=k: _arr_eq(V.varm(a, k), Fs.varm(a, k)))
        run(f"{a}: dense_layers", lambda a=a: (V.dense_layers(a) == Fs.dense_layers(a), "differ"))
        for dl in Fs.dense_layers(a):
            def dn(a=a, dl=dl):
                x, y = V.dense(a, dl), Fs.dense(a, dl)
                if not (x.index.equals(y.index) and x.columns.equals(y.columns)):
                    return False, "index/columns"
                return _arr_eq(x.to_numpy(), y.to_numpy())
            run(f"{a}: dense {dl}", dn)
            run(f"{a}: split {dl}", lambda a=a, dl=dl: _split_eq(V.split(a, dl), Fs.split(a, dl)))
        feats = Fs.features(a)
        for l in Fs.layers(a):
            run(f"{a}/{l}: split", lambda a=a, l=l: _split_eq(V.split(a, l), Fs.split(a, l)))
            run(f"{a}/{l}: X", lambda a=a, l=l: _sparse_eq(V.X(a, l), Fs.X(a, l)))
            run(f"{a}/{l}: X_csc", lambda a=a, l=l: _sparse_eq(V.X_csc(a, l), Fs.X_csc(a, l)))

            def gf(a=a, l=l):
                for s in range(0, len(feats), feature_batch):
                    names = list(feats[s:s + feature_batch])
                    ok, why = _sparse_eq(V.get_features(names, a, l, dense=False), Fs.get_features(names, a, l, dense=False))
                    if not ok:
                        return False, f"features {s}..: {why}"
                return True, ""
            run(f"{a}/{l}: get_features (all {len(feats)} features, sparse)", gf)
            names = list(feats[: min(50, len(feats))])
            run(f"{a}/{l}: get_features (dense)", lambda a=a, l=l, names=names: _frame_eq(
                V.get_features(names, a, l), Fs.get_features(names, a, l)))
    def sb():
        x, y = V.subset(sub, graphs=False), Fs.subset(sub, graphs=False)
        ok, why = _frame_eq(x.obs, y.obs)
        if not ok:
            return False, f"obs: {why}"
        for a in y.assays:
            for l in y.assays[a].layers:
                ok, why = _sparse_eq(x.X(a, l), y.X(a, l))
                if not ok:
                    return False, f"{a}/{l}: {why}"
        for r in y.reductions:
            ok, why = _arr_eq(x.reductions[r].embedding, y.reductions[r].embedding)
            if not ok:
                return False, f"reduction {r}: {why}"
        return True, ""
    run(f"subset of {len(sub)} random cells", sb)
    if getattr(V, "_view", None) is not None:
        def par():
            for i in range(len(V._view.spec["parents"])):
                V._view.parent(i)
            return True, ""
        run("parents present and unchanged", par)
    return res


def _split_eq(a, b) -> tuple:
    if a is None or b is None:
        return (a is None and b is None), "one is None"
    return (a.equals(b) and list(a.cat.categories) == list(b.cat.categories)), "differ"


# --------------------------------------------------------------------------- materialize
def materialize(view, out) -> Path:
    """Write the full store a view stands for (every file of the view except view files, plus each inherited layer's
    CSR and CSC arrays in the dtypes of the original files). Its arrays and other files equal the store the view was
    built from.

    Parameters
    ----------
    view : the view (path or Store); its parents must be reachable.
    out : the new full store directory (must not exist).

    Returns
    -------
    Path of the full store.

    Examples
    --------
    >>> import arsc, tempfile, os
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> m = arsc.views.materialize(t["view"], os.path.join(tempfile.mkdtemp(), "full_again"))
    >>> arsc.open(m).is_view
    False
    """
    from .io import Store, _build_csc
    V = view if isinstance(view, Store) else Store(view, finalize=False)
    if getattr(V, "_view", None) is None:
        raise ViewError(f"{V.path} is not a view")
    out = Path(out)
    if out.exists():
        raise ViewError(f"{out} exists")
    tmp = out.with_name(out.name + f".materializing{os.getpid()}")
    for f in sorted(V.path.rglob("*")):
        rel = f.relative_to(V.path)
        if f.is_dir() or rel.parts[0] == VIEW_DIR or f.name == VIEW_FILE or f.name.endswith(".lock"):
            continue
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
    for key in V._view.inherited:
        a, l = key.split("/", 1)
        L = V._view.layer(a, l)
        m = L.X()
        base = tmp / "assays" / a / l
        fdt = L.e.get("file_dtype") or {}
        for c in ("data", "indices", "indptr"):
            arr = np.asarray(getattr(m, c))
            _save(Path(str(base) + f".csr.{c}.npy"), arr.astype(fdt.get(f"csr.{c}", arr.dtype), copy=False))
        del m
        _build_csc(str(base) + ".csr", str(base) + ".csc", L.shape)
        for c in ("data", "indices", "indptr"):
            f = Path(str(base) + f".csc.{c}.npy")
            want = fdt.get(f"csc.{c}")
            if want and np.load(f, mmap_mode="r").dtype != np.dtype(want):
                _save(f, np.load(f).astype(want))
    os.replace(tmp, out)
    return out


def array_sha256(a) -> str:
    """sha256 of an array's data bytes (C order); equal for a file's array and the view's reconstruction.

    Parameters
    ----------
    a : array-like.

    Returns
    -------
    str (hex digest).

    Examples
    --------
    >>> import numpy as np, arsc
    >>> arsc.views.array_sha256(np.arange(3)) == arsc.views.array_sha256(np.array([0, 1, 2]))
    True
    """
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    flat = a.reshape(-1).view(np.uint8)
    step = 1 << 26
    for s in range(0, flat.size, step):
        h.update(flat[s:s + step])
    return h.hexdigest()
