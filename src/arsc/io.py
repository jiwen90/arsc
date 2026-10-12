"""I/O bridge between Seurat (R) and arsc: a memory-mappable on-disk store.

Workflow::

    Rscript $(python -c "import arsc; print(arsc.exporter_script())") obj.qs store/   # once
    st = arsc.open("store/")            # lazy, zero-copy: nothing is read yet
    st.obs                              # metadata (parquet)
    st.get_features(["CD8A", "GZMB"])   # touches only those genes' bytes (CSC)
    sub = st.subset(st.obs.sample == "ALQ13912")   # materializes only those cells
    X = st.X("RNA", "counts")           # full CSR backed by the page cache (no copy)

Every array is an NPY file opened with ``np.load(mmap_mode="r")`` and wrapped
in a ``scipy.sparse`` matrix without copying, so N worker processes opening the
same store share one physical copy through the OS page cache (validated in
``benchmarks/mmap_concurrency.py``).

Layout (written by ``arsc/R/export_seurat.R``; Python adds the files marked *)::

    manifest.json  cells.txt  obs.csv  obs.parquet*
    assays/<A>/features.txt  [variable_features.txt]
    assays/<A>/<layer>.csr.{data,indices,indptr}.npy      cells x features
    assays/<A>/<layer>.csc.{data,indices,indptr}.npy*     per-feature slices
    reductions/<R>.npy  [<R>.loadings.npy, <R>.loadings_features.txt]
    graphs/<G>.csr.{data,indices,indptr}.npy

The ``*`` files are produced once by the first :func:`open` (atomically, under
a lock, so concurrent first opens are safe).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "Assay", "Reduction", "Dataset", "Store",
    "open", "read_seurat_export", "export_seurat", "exporter_script", "write_table",
]


def exporter_script() -> Path:
    """Path of the bundled R exporter (``Rscript <path> <obj.rds|.qs> <out_dir>``)."""
    return Path(__file__).with_name("R") / "export_seurat.R"


def export_seurat(obj_path, out_dir, *, assays=None, layers=("counts",), graphs: bool = True,
                  reductions: bool = True, rscript: str = "Rscript", threads: int = 16) -> Path:
    """Run the R exporter on a ``.rds``/``.qs`` Seurat object (v4 or v5); returns ``out_dir``."""
    cmd = [rscript, str(exporter_script()), str(obj_path), str(out_dir),
           "--layers", ",".join(layers), "--threads", str(threads)]
    if assays:
        cmd += ["--assays", ",".join(assays)]
    if not graphs:
        cmd.append("--no-graphs")
    if not reductions:
        cmd.append("--no-reductions")
    subprocess.run(cmd, check=True)
    return Path(out_dir)


# --------------------------------------------------------------------------- containers
@dataclass
class Assay:
    """One modality. Every layer is a cells x features ``scipy.sparse.csr_matrix``."""

    name: str
    features: pd.Index
    layers: dict = field(default_factory=dict)
    variable_features: list = field(default_factory=list)

    def __getitem__(self, layer: str) -> sp.csr_matrix:
        return self.layers[layer]

    def __repr__(self) -> str:
        ls = ", ".join(f"{k} (nnz={v.nnz:,})" for k, v in self.layers.items())
        return f"Assay({self.name!r}, {len(self.features):,} features, layers: {ls or 'none'})"


@dataclass
class Reduction:
    """Cell embedding (cells x k), optional feature loadings and stdev."""

    embedding: np.ndarray
    columns: list
    loadings: pd.DataFrame | None = None
    stdev: np.ndarray | None = None
    assay: str | None = None

    def to_frame(self, index=None) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(self.embedding), index=index, columns=self.columns)

    def __repr__(self) -> str:
        return f"Reduction({self.embedding.shape[0]:,} x {self.embedding.shape[1]})"


@dataclass
class Dataset:
    """In-memory (or mmap-backed) container: ``obs``, ``assays``, ``reductions``, ``graphs``."""

    obs: pd.DataFrame
    assays: dict = field(default_factory=dict)
    reductions: dict = field(default_factory=dict)
    graphs: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    def __getitem__(self, assay: str) -> Assay:
        return self.assays[assay]

    @property
    def n_cells(self) -> int:
        return len(self.obs)

    @property
    def cells(self) -> pd.Index:
        return self.obs.index

    def X(self, assay: str = "RNA", layer: str | None = None) -> sp.csr_matrix:
        """Cells x features matrix for ``assay``/``layer`` (default: counts, else first layer)."""
        a = self.assays[assay]
        return a.layers[layer or _default_layer(a.layers)]

    def embedding(self, name: str, dims=None) -> np.ndarray:
        e = self.reductions[name].embedding
        return np.asarray(e if dims is None else e[:, list(dims)])

    def subset(self, cells) -> "Dataset":
        """Row subset (boolean mask, positions or barcodes); copies only the selected rows."""
        idx = _as_positions(cells, self.obs.index)
        assays = {k: Assay(a.name, a.features, {l: _take_rows(m, idx) for l, m in a.layers.items()},
                           list(a.variable_features)) for k, a in self.assays.items()}
        reds = {k: Reduction(np.asarray(r.embedding)[idx], r.columns, r.loadings, r.stdev, r.assay)
                for k, r in self.reductions.items()}
        graphs = {k: sp.csr_matrix(g)[idx][:, idx].tocsr() for k, g in self.graphs.items()}
        return Dataset(self.obs.iloc[idx].copy(), assays, reds, graphs, dict(self.manifest))

    def __repr__(self) -> str:
        lines = [f"arsc.Dataset: {self.n_cells:,} cells"]
        lines += [f"  {a!r}" for a in self.assays.values()]
        if self.reductions:
            lines.append("  reductions: " + ", ".join(f"{k} ({v.embedding.shape[1]})" for k, v in self.reductions.items()))
        if self.graphs:
            lines.append("  graphs: " + ", ".join(self.graphs))
        lines.append(f"  obs: {self.obs.shape[1]} columns")
        return "\n".join(lines)


def _default_layer(layers) -> str:
    if "counts" in layers:
        return "counts"
    if not layers:
        raise KeyError("assay has no layers")
    return next(iter(layers))


def _as_positions(cells, index: pd.Index) -> np.ndarray:
    if isinstance(cells, pd.Series):
        cells = cells.to_numpy()
    cells = np.asarray(cells)
    if cells.dtype == bool:
        if cells.size != len(index):
            raise ValueError("boolean mask has wrong length")
        return np.flatnonzero(cells)
    if np.issubdtype(cells.dtype, np.integer):
        return cells.astype(np.int64)
    pos = index.get_indexer(cells)
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} barcodes not found")
    return pos.astype(np.int64)


# --------------------------------------------------------------------------- sparse kernels
@nb.njit(parallel=True, cache=True)
def _gather_rows(indptr, indices, data, rows, out_indptr, out_indices, out_data):
    n = rows.size
    for t in nb.prange(n):
        r = rows[t]
        s, e = indptr[r], indptr[r + 1]
        o = out_indptr[t]
        for p in range(e - s):
            out_indices[o + p] = indices[s + p]
            out_data[o + p] = data[s + p]


def _take_rows(m, rows: np.ndarray) -> sp.csr_matrix:
    """Copy selected rows of a (possibly mmap-backed) CSR; reads only those rows' bytes."""
    m = m if isinstance(m, sp.csr_matrix) else sp.csr_matrix(m)
    indptr = np.asarray(m.indptr)
    lens = (indptr[rows + 1] - indptr[rows]).astype(np.int64)
    out_indptr = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(lens, out=out_indptr[1:])
    nnz = int(out_indptr[-1])
    out_indices = np.empty(nnz, dtype=m.indices.dtype)
    out_data = np.empty(nnz, dtype=m.data.dtype)
    _gather_rows(indptr, m.indices, m.data, rows, out_indptr, out_indices, out_data)
    if nnz < 2**31:
        out_indptr = out_indptr.astype(np.int32)
        out_indices = out_indices.astype(np.int32, copy=False)
    res = sp.csr_matrix((out_data, out_indices, out_indptr), shape=(rows.size, m.shape[1]), copy=False)
    res.has_sorted_indices = True
    return res


@nb.njit(parallel=True, cache=True)
def _csr_col_counts(indptr, indices, n_cols, n_chunks):
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


@nb.njit(parallel=True, cache=True)
def _csr_to_csc_scatter(indptr, indices, data, offsets, out_indices, out_data, n_chunks):
    """Each chunk of rows scatters into its precomputed per-column slots (keeps row order)."""
    n_rows = indptr.size - 1
    step = (n_rows + n_chunks - 1) // n_chunks
    for c in nb.prange(n_chunks):
        lo = c * step
        hi = min(n_rows, lo + step)
        pos = offsets[c].copy()
        for i in range(lo, hi):
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                q = pos[j]
                out_indices[q] = i
                out_data[q] = data[p]
                pos[j] = q + 1


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        np.lib.format.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def _build_csc(prefix_csr: str, prefix_csc: str, shape) -> None:
    """Write the CSC (per-feature) copy of an on-disk CSR, streaming through memmaps."""
    n_rows, n_cols = shape
    indptr = np.load(prefix_csr + ".indptr.npy", mmap_mode="r")
    indices = np.load(prefix_csr + ".indices.npy", mmap_mode="r")
    data = np.load(prefix_csr + ".data.npy", mmap_mode="r")
    n_chunks = max(1, nb.get_num_threads())
    counts = _csr_col_counts(indptr, indices, n_cols, n_chunks)
    colptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(counts.sum(0), out=colptr[1:])
    offsets = (colptr[:-1][None, :] + np.cumsum(counts, axis=0) - counts).astype(np.int64)
    nnz = int(colptr[-1])
    idx_dtype = np.int32 if max(nnz, n_rows) < 2**31 else np.int64
    tmp_i = Path(prefix_csc + f".indices.npy.tmp{os.getpid()}")
    tmp_x = Path(prefix_csc + f".data.npy.tmp{os.getpid()}")
    out_i = np.lib.format.open_memmap(tmp_i, mode="w+", dtype=idx_dtype, shape=(nnz,))
    out_x = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=data.dtype, shape=(nnz,))
    _csr_to_csc_scatter(indptr, indices, data, offsets, out_i, out_x, n_chunks)
    out_i.flush(); out_x.flush()
    del out_i, out_x
    _atomic_save(Path(prefix_csc + ".indptr.npy"), colptr.astype(idx_dtype))
    os.replace(tmp_i, prefix_csc + ".indices.npy")
    os.replace(tmp_x, prefix_csc + ".data.npy")


def _mmap_sparse(prefix: str, shape, fmt: str = "csr"):
    data = np.load(prefix + ".data.npy", mmap_mode="r")
    indices = np.load(prefix + ".indices.npy", mmap_mode="r")
    indptr = np.load(prefix + ".indptr.npy", mmap_mode="r")
    cls = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
    m = cls((data, indices, indptr), shape=tuple(shape), copy=False)
    m.has_sorted_indices = True
    return m


# --------------------------------------------------------------------------- obs
def _read_obs_csv(d: Path, manifest: dict, cells: list) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    spec = manifest.get("obs", {}) or {}
    types = {"cell": pa.string()}
    for col, s in spec.items():
        types[col] = {"double": pa.float64(), "integer": pa.int64(), "logical": pa.bool_()}.get(s["type"], pa.string())
    tbl = pacsv.read_csv(
        d / "obs.csv",
        convert_options=pacsv.ConvertOptions(column_types=types, null_values=["NA"], strings_can_be_null=True,
                                             quoted_strings_can_be_null=False, true_values=["TRUE"],
                                             false_values=["FALSE"]),
        read_options=pacsv.ReadOptions(block_size=1 << 26),
    )
    obs = tbl.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}.get)
    obs = obs.set_index("cell")
    obs.index.name = None
    for col, s in spec.items():
        if s["type"] == "factor":
            obs[col] = pd.Categorical(obs[col], categories=list(s.get("levels") or []))
        elif s["type"] == "integer" and not obs[col].isna().any():
            obs[col] = obs[col].astype(np.int64)
        elif s["type"] == "logical" and not obs[col].isna().any():
            obs[col] = obs[col].astype(bool)
        elif s["type"] == "character":
            # compact: low-cardinality -> category, otherwise Arrow-backed strings
            # (Python str objects cost ~60 B each; 260k cells x 24 columns ~ 0.5 GB per process)
            nun = obs[col].nunique(dropna=True)
            obs[col] = obs[col].astype("category" if nun <= max(1000, len(obs) // 2) else "string[pyarrow]")
    if not obs.index.equals(pd.Index(cells)):
        raise ValueError("obs.csv row order does not match cells.txt")
    return obs


# --------------------------------------------------------------------------- store
class _Lock:
    """Cross-process lock file (O_EXCL). Stale locks older than ``stale`` seconds are broken."""

    def __init__(self, path: Path, timeout: float = 3600, stale: float = 1800):
        self.path, self.timeout, self.stale = path, timeout, stale

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"lock {self.path} held too long")
                time.sleep(0.2)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


class Store:
    """Lazy, memory-mapped handle on an arsc store directory (see :func:`open`)."""

    def __init__(self, path, finalize: bool = True, build_csc: bool = True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != "arsc-store":
            raise ValueError(f"{path} is not an arsc store (re-export with arsc/R/export_seurat.R)")
        self._obs = None
        self._features: dict[str, pd.Index] = {}
        if finalize:
            self.finalize(build_csc=build_csc)

    # -- one-time finalization -------------------------------------------------
    def _missing(self, build_csc: bool) -> list:
        todo = []
        if not (self.path / "obs.parquet").exists():
            todo.append(("obs", None, None))
        if build_csc:
            for a, entry in (self.manifest.get("assays") or {}).items():
                for l in (entry.get("layers") or {}):
                    if not (self.path / "assays" / a / f"{l}.csc.indptr.npy").exists():
                        todo.append(("csc", a, l))
        return todo

    def finalize(self, build_csc: bool = True) -> None:
        """Create ``obs.parquet`` and per-feature CSC copies if missing (idempotent, process-safe)."""
        if not self._missing(build_csc):
            return
        with _Lock(self.path / ".finalize.lock"):
            for kind, a, l in self._missing(build_csc):
                if kind == "obs":
                    obs = _read_obs_csv(self.path, self.manifest, self._cells_txt())
                    tmp = self.path / f"obs.parquet.tmp{os.getpid()}"
                    obs.to_parquet(tmp)
                    os.replace(tmp, self.path / "obs.parquet")
                else:
                    info = self.manifest["assays"][a]["layers"][l]
                    base = str(self.path / "assays" / a / l)
                    _build_csc(base + ".csr", base + ".csc", info["shape"])

    def _cells_txt(self) -> list:
        return (self.path / "cells.txt").read_text(encoding="utf-8").splitlines()

    # -- metadata ---------------------------------------------------------------
    @property
    def obs(self) -> pd.DataFrame:
        if self._obs is None:
            if (self.path / "obs.parquet").exists():
                self._obs = pd.read_parquet(self.path / "obs.parquet")
                self._obs.index.name = None
                try:  # hand pyarrow's transient read buffers back to the OS
                    import pyarrow as pa
                    pa.default_memory_pool().release_unused()
                except Exception:
                    pass
            else:
                self._obs = _read_obs_csv(self.path, self.manifest, self._cells_txt())
        return self._obs

    @property
    def cells(self) -> pd.Index:
        return self.obs.index

    @property
    def n_cells(self) -> int:
        return int(self.manifest["n_cells"])

    @property
    def assays(self) -> list:
        return list(self.manifest.get("assays") or {})

    def layers(self, assay: str) -> list:
        return list(self.manifest["assays"][assay].get("layers") or {})

    def features(self, assay: str = "RNA") -> pd.Index:
        if assay not in self._features:
            txt = (self.path / "assays" / assay / "features.txt").read_text(encoding="utf-8")
            self._features[assay] = pd.Index(txt.splitlines())
        return self._features[assay]

    def variable_features(self, assay: str = "RNA") -> list:
        p = self.path / "assays" / assay / "variable_features.txt"
        return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    def _layer(self, assay, layer):
        layers = self.manifest["assays"][assay].get("layers") or {}
        layer = layer or _default_layer(layers)
        return layer, layers[layer]

    # -- matrices (zero-copy) ---------------------------------------------------------
    def X(self, assay: str = "RNA", layer: str | None = None) -> sp.csr_matrix:
        """Cells x features CSR backed by memory maps (no copy)."""
        layer, info = self._layer(assay, layer)
        return _mmap_sparse(str(self.path / "assays" / assay / f"{layer}.csr"), info["shape"], "csr")

    def X_csc(self, assay: str = "RNA", layer: str | None = None) -> sp.csc_matrix:
        """Cells x features CSC (column slices = features) backed by memory maps."""
        layer, info = self._layer(assay, layer)
        base = self.path / "assays" / assay / f"{layer}.csc"
        if not Path(str(base) + ".indptr.npy").exists():
            self.finalize(build_csc=True)
        return _mmap_sparse(str(base), info["shape"], "csc")

    def get_features(self, features, assay: str = "RNA", layer: str | None = None, dense: bool = True):
        """Values of selected features for all cells; reads only those features' bytes.

        Returns a cells x features DataFrame (``dense=True``) or a CSC matrix.
        Unknown names raise ``KeyError``.
        """
        names = [features] if isinstance(features, str) else list(features)
        fidx = self.features(assay).get_indexer(names)
        if (fidx < 0).any():
            raise KeyError(f"not in {assay}: {[n for n, i in zip(names, fidx) if i < 0]}")
        C = self.X_csc(assay, layer)
        indptr = np.asarray(C.indptr)
        parts_i, parts_x, ptr = [], [], [0]
        for j in fidx:
            s, e = int(indptr[j]), int(indptr[j + 1])
            parts_i.append(np.asarray(C.indices[s:e]))
            parts_x.append(np.asarray(C.data[s:e]))
            ptr.append(ptr[-1] + e - s)
        sub = sp.csc_matrix((np.concatenate(parts_x) if parts_x else np.zeros(0, C.dtype),
                             np.concatenate(parts_i) if parts_i else np.zeros(0, np.int32),
                             np.asarray(ptr, dtype=np.int64)), shape=(C.shape[0], len(names)))
        if not dense:
            return sub
        return pd.DataFrame(sub.toarray(), index=self.cells, columns=names)

    def embedding(self, name: str, dims=None) -> np.ndarray:
        """Cells x k embedding (memory-mapped, read-only)."""
        e = np.load(self.path / "reductions" / f"{name}.npy", mmap_mode="r")
        return e if dims is None else np.asarray(e[:, list(dims)])

    def reduction(self, name: str) -> Reduction:
        e = self.manifest["reductions"][name]
        rd = self.path / "reductions"
        loadings = None
        if "loadings_shape" in e:
            L = np.load(rd / f"{name}.loadings.npy", mmap_mode="r")
            lf = (rd / f"{name}.loadings_features.txt").read_text(encoding="utf-8").splitlines()
            loadings = pd.DataFrame(np.asarray(L), index=lf, columns=e["columns"])
        stdev = np.asarray(e["stdev"]) if e.get("stdev") is not None else None
        return Reduction(self.embedding(name), list(e["columns"]), loadings, stdev, e.get("assay"))

    @property
    def reductions(self) -> list:
        return list(self.manifest.get("reductions") or {})

    @property
    def graphs(self) -> list:
        return list(self.manifest.get("graphs") or {})

    def graph(self, name: str):
        """Cells x cells graph (e.g. Seurat ``wsnn``), memory-mapped (returned as CSC view of t(G)^T)."""
        info = self.manifest["graphs"][name]
        m = _mmap_sparse(str(self.path / "graphs" / f"{name}.csr"), info["shape"], "csr")
        return m.T if info.get("transposed", True) else m

    # -- materialization -----------------------------------------------------------
    def to_dataset(self, assays=None, layers=None, reductions: bool = True, graphs: bool = True) -> Dataset:
        """All (or selected) data as a :class:`Dataset`; arrays stay memory-mapped (zero-copy)."""
        out_assays = {}
        for a in self.assays:
            if assays is not None and a not in assays:
                continue
            lay = {l: self.X(a, l) for l in self.layers(a) if layers is None or l in layers}
            out_assays[a] = Assay(a, self.features(a), lay, self.variable_features(a))
        reds = {r: self.reduction(r) for r in self.reductions} if reductions else {}
        gr = {g: self.graph(g) for g in self.graphs} if graphs else {}
        return Dataset(self.obs, out_assays, reds, gr, self.manifest)

    def subset(self, cells, assays=None, layers=None, reductions: bool = True, graphs: bool = False) -> Dataset:
        """Materialize only the selected cells (mask / positions / barcodes) as an in-memory Dataset."""
        idx = _as_positions(cells, self.cells)
        out_assays = {}
        for a in self.assays:
            if assays is not None and a not in assays:
                continue
            lay = {l: _take_rows(self.X(a, l), idx) for l in self.layers(a) if layers is None or l in layers}
            out_assays[a] = Assay(a, self.features(a), lay, self.variable_features(a))
        reds = {}
        if reductions:
            for r in self.reductions:
                rr = self.reduction(r)
                reds[r] = Reduction(np.asarray(rr.embedding[idx]), rr.columns, rr.loadings, rr.stdev, rr.assay)
        gr = {}
        if graphs:
            for g in self.graphs:
                G = sp.csr_matrix(self.graph(g))
                gr[g] = _take_rows(G, idx)[:, idx].tocsr()
        return Dataset(self.obs.iloc[idx].copy(), out_assays, reds, gr, self.manifest)

    def __repr__(self) -> str:
        lines = [f"arsc.Store({str(self.path)!r}): {self.n_cells:,} cells (memory-mapped)"]
        for a in self.assays:
            ls = ", ".join(f"{l} (nnz={i['nnz']:,}, {i['dtype']})"
                           for l, i in (self.manifest['assays'][a].get('layers') or {}).items())
            lines.append(f"  {a}: {len(self.features(a)):,} features; {ls or 'no layers'}")
        if self.reductions:
            lines.append("  reductions: " + ", ".join(self.reductions))
        if self.graphs:
            lines.append("  graphs: " + ", ".join(self.graphs))
        return "\n".join(lines)


def open(path, finalize: bool = True, build_csc: bool = True) -> Store:  # noqa: A001 - deliberate API name
    """Open an arsc store lazily (memory-mapped, zero-copy). See :class:`Store`.

    The first open of a fresh export writes ``obs.parquet`` and the per-feature
    CSC copies (one-time; concurrent first opens wait on a lock instead of
    duplicating work). Pass ``finalize=False`` for strictly read-only use.
    """
    return Store(path, finalize=finalize, build_csc=build_csc)


def read_seurat_export(path, *, assays=None, layers=None, graphs: bool = True, reductions: bool = True) -> Dataset:
    """Load an arsc store as a :class:`Dataset` (memory-mapped arrays; nothing is copied)."""
    return open(path).to_dataset(assays=assays, layers=layers, reductions=reductions, graphs=graphs)


# --------------------------------------------------------------------------- tidy outputs
def write_table(df: pd.DataFrame, path, index: bool | None = None) -> Path:
    """Write a tidy table for the R plotting side; format from the extension (.parquet/.csv/.tsv).

    Read in R with ``arrow::read_parquet`` or ``data.table::fread``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if index is None:
        index = not isinstance(df.index, pd.RangeIndex)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=index)
    elif p.suffix in (".csv", ".tsv"):
        df.to_csv(p, sep="\t" if p.suffix == ".tsv" else ",", index=index)
    else:
        raise ValueError(f"unsupported extension {p.suffix!r}")
    return p
