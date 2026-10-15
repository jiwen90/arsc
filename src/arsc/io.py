"""I/O bridge between Seurat (R) and arsc: a memory-mappable on-disk store.

Workflow::

    Rscript $(python -c "import arsc; print(arsc.exporter_script())") obj.qs store/   # once
    st = arsc.open("store/")            # lazy, zero-copy: nothing is read yet
    st.obs                              # metadata (parquet)
    st.get_features(["CD8A", "GZMB"])   # touches only those genes' bytes (CSC)
    sub = st.subset(st.obs["sample"] == "P1_pre")   # materializes only those cells
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
    "open", "read_seurat_export", "export_seurat", "exporter_script", "write_table", "from_anndata",
]


def exporter_script() -> Path:
    """Path of the bundled R exporter (``Rscript <path> <obj.rds|.qs> <out_dir>``).

    Returns
    -------
    pathlib.Path of ``export_seurat.R`` inside the installed package.

    Examples
    --------
    >>> import arsc
    >>> arsc.exporter_script().name
    'export_seurat.R'
    """
    return Path(__file__).with_name("R") / "export_seurat.R"


def export_seurat(obj_path, out_dir, *, assays=None, layers=("counts",), graphs: bool = True,
                  reductions: bool = True, rscript: str = "Rscript", threads: int = 16) -> Path:
    """Run the R exporter on a ``.rds``/``.qs`` Seurat object (v4 or v5); prefer :func:`arsc.read_seurat`.

    Parameters
    ----------
    obj_path : ``.rds`` or ``.qs`` file.
    out_dir : store directory to write.
    assays : assays to export (default all).
    layers : layers to export (``"counts"``, ``"data"``, ``"scale.data"``).
    graphs, reductions : export graphs / reductions.
    rscript : path of ``Rscript``.
    threads : R threads.

    Returns
    -------
    pathlib.Path ``out_dir`` (open it with :func:`open`). Needs R with SeuratObject (and qs).

    Examples
    --------
    >>> import arsc
    >>> arsc.export_seurat("obj.qs", "stores/obj", layers=("counts", "data"))   # doctest: +SKIP
    """
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
    """One modality (RNA, ADT, ...). Every layer is a cells x features ``scipy.sparse.csr_matrix``.

    Attributes
    ----------
    name : assay name.
    features : feature names.
    layers : ``{layer: csr_matrix}`` (e.g. ``counts``, ``data``).
    variable_features : variable features recorded at import.
    var : optional per-feature annotations.
    varm : per-feature matrices.

    Examples
    --------
    >>> import arsc
    >>> a = arsc.datasets.toy()["ADT"]
    >>> a.name, len(a.features), list(a.layers)
    ('ADT', 6, ['counts'])
    """

    name: str
    features: pd.Index
    layers: dict = field(default_factory=dict)
    variable_features: list = field(default_factory=list)
    var: pd.DataFrame | None = None
    varm: dict = field(default_factory=dict)

    def __getitem__(self, layer: str) -> sp.csr_matrix:
        return self.layers[layer]

    def var_frame(self) -> pd.DataFrame:
        """Per-feature annotations (empty frame indexed by feature names if none were imported).

        Examples
        --------
        >>> import arsc
        >>> arsc.datasets.toy()["RNA"].var_frame().shape
        (300, 0)
        """
        if self.var is None:
            return pd.DataFrame(index=pd.Index(self.features, dtype=object))
        v = self.var.copy()
        v.index = pd.Index(self.features, dtype=object)
        return v

    def __repr__(self) -> str:
        ls = ", ".join(f"{k} (nnz={v.nnz:,})" for k, v in self.layers.items())
        return f"Assay({self.name!r}, {len(self.features):,} features, layers: {ls or 'none'})"


@dataclass
class Reduction:
    """Cell embedding (cells x k), optional feature loadings and stdev.

    Attributes
    ----------
    embedding : cells x k array.
    columns : column names (e.g. ``PC_1``...).
    loadings : optional features x k DataFrame.
    stdev : optional per-component s.d.
    assay : assay the reduction was computed from.

    Examples
    --------
    >>> import arsc, numpy as np
    >>> r = arsc.Reduction(np.zeros((5, 2)), ["PC_1", "PC_2"])
    >>> r.to_frame().shape
    (5, 2)
    """

    embedding: np.ndarray
    columns: list
    loadings: pd.DataFrame | None = None
    stdev: np.ndarray | None = None
    assay: str | None = None

    def to_frame(self, index=None) -> pd.DataFrame:
        """Embedding as a DataFrame (``index`` = cell barcodes, e.g. ``ds.cells``)."""
        return pd.DataFrame(np.asarray(self.embedding), index=index, columns=self.columns)

    def __repr__(self) -> str:
        return f"Reduction({self.embedding.shape[0]:,} x {self.embedding.shape[1]})"


@dataclass
class Dataset:
    """In-memory (or mmap-backed) container: ``obs``, ``assays``, ``reductions``, ``graphs``.

    Returned by :func:`arsc.read` (no ``store=``), :meth:`Store.subset` and :func:`arsc.datasets.toy`.
    Every matrix is cells x features (CSR).

    Attributes
    ----------
    obs : per-cell metadata (index = barcodes).
    assays : ``{name: Assay}``; ``ds.X(assay, layer)`` returns a matrix.
    reductions : ``{name: Reduction}``; ``ds.embedding(name)``.
    graphs : ``{name: csr_matrix}``.
    manifest : store manifest (provenance).
    uns : unstructured metadata.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> ds.n_cells, list(ds.assays), ds.layers("RNA")
    (3000, ['RNA', 'ADT'], ['counts'])
    >>> cd8 = ds.subset((ds.obs["cell_type"] == "CD8 T").to_numpy())
    >>> cd8.X("RNA", "counts").shape[1]
    300
    """

    obs: pd.DataFrame
    assays: dict = field(default_factory=dict)
    reductions: dict = field(default_factory=dict)
    graphs: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    uns: dict = field(default_factory=dict)

    def __getitem__(self, assay: str) -> Assay:
        return self.assays[assay]

    def features(self, assay: str = "RNA") -> pd.Index:
        """Feature (gene) names of an assay."""
        return self._assay_or_raise(assay).features

    def var(self, assay: str = "RNA") -> pd.DataFrame:
        """Per-feature annotations of an assay."""
        return self._assay_or_raise(assay).var_frame()

    def layers(self, assay: str = "RNA") -> list:
        """Layer names of an assay."""
        return list(self._assay_or_raise(assay).layers)

    def to_anndata(self, cells=None, genes=None, **kw):
        """In-memory ``anndata.AnnData`` (see :func:`arsc.to_anndata`)."""
        from .formats import to_anndata
        return to_anndata(self, cells, genes, **kw)

    def write_h5ad(self, path, cells=None, genes=None, **kw) -> Path:
        """Write ``.h5ad`` (see :func:`arsc.write_h5ad`)."""
        from .formats import write_h5ad
        return write_h5ad(self, path, cells, genes, **kw)

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.obs)

    @property
    def cells(self) -> pd.Index:
        """Cell barcodes (``obs.index``)."""
        return self.obs.index

    def X(self, assay: str = "RNA", layer: str | None = None) -> sp.csr_matrix:
        """Cells x features matrix for ``assay``/``layer`` (default: counts, else first layer).

        Examples
        --------
        >>> import arsc
        >>> arsc.datasets.toy().X("ADT", "counts").shape
        (3000, 6)
        """
        a = self._assay_or_raise(assay)
        layer = layer or _default_layer(a.layers)
        if layer not in a.layers:
            raise KeyError(f"layer {layer!r} not in assay {assay!r} (has {list(a.layers)})")
        return a.layers[layer]

    def get_features(self, features, assay: str = "RNA", layer: str | None = None, dense: bool = True):
        """Values of selected features for all cells (mirrors :meth:`Store.get_features`).

        Returns a cells x features DataFrame (``dense=True``) or a CSC matrix.
        Unknown names raise ``KeyError``.

        Examples
        --------
        >>> import arsc
        >>> ds = arsc.datasets.toy()
        >>> df = ds.get_features(["CD8A", "MS4A1"])
        >>> list(df.columns), len(df) == ds.n_cells
        (['CD8A', 'MS4A1'], True)
        """
        names = [features] if isinstance(features, str) else list(features)
        fidx = self.features(assay).get_indexer(names)
        if (fidx < 0).any():
            raise KeyError(f"not in {assay}: {[n for n, i in zip(names, fidx) if i < 0]} (gene symbols are "
                           "case-sensitive; check ds.features(assay))")
        sub = sp.csc_matrix(self.X(assay, layer)[:, fidx])
        if not dense:
            return sub
        return pd.DataFrame(sub.toarray(), index=self.cells, columns=names)

    def _assay_or_raise(self, assay: str) -> "Assay":
        if assay not in self.assays:
            raise KeyError(f"assay {assay!r} not in dataset (has {list(self.assays)})")
        return self.assays[assay]

    def embedding(self, name: str, dims=None) -> np.ndarray:
        """Cells x k embedding of reduction ``name`` (optionally only columns ``dims``)."""
        if name not in self.reductions:
            raise KeyError(f"reduction {name!r} not found (has {list(self.reductions)})")
        e = self.reductions[name].embedding
        return np.asarray(e if dims is None else e[:, list(dims)])

    def subset(self, cells) -> "Dataset":
        """Row subset (boolean mask, positions or barcodes); copies only the selected rows.

        Examples
        --------
        >>> import arsc
        >>> ds = arsc.datasets.toy()
        >>> ds.subset((ds.obs["timepoint"] == "pre").to_numpy()).n_cells < ds.n_cells
        True
        """
        idx = _as_positions(cells, self.obs.index)
        assays = {k: Assay(a.name, a.features, {l: _take_rows(m, idx) for l, m in a.layers.items()},
                           list(a.variable_features), a.var, dict(a.varm)) for k, a in self.assays.items()}
        reds = {k: Reduction(np.asarray(r.embedding)[idx], r.columns, r.loadings, r.stdev, r.assay)
                for k, r in self.reductions.items()}
        graphs = {k: sp.csr_matrix(g)[idx][:, idx].tocsr() for k, g in self.graphs.items()}
        return Dataset(self.obs.iloc[idx].copy(), assays, reds, graphs, dict(self.manifest), dict(self.uns))

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
    if cells.dtype == object and cells.size and all(isinstance(v, (bool, np.bool_)) or v is pd.NA for v in cells[:100]):
        raise TypeError("cells looks like a nullable boolean mask (object/'boolean' dtype); convert it with "
                        "mask.fillna(False).to_numpy(dtype=bool)")
    if cells.dtype == bool:
        if cells.size != len(index):
            raise ValueError(f"boolean mask has length {cells.size} but there are {len(index)} cells; build the mask "
                             "from this object's obs (e.g. (st.obs['cell_type'] == 'B').to_numpy())")
        return np.flatnonzero(cells)
    if np.issubdtype(cells.dtype, np.integer):
        return cells.astype(np.int64)
    pos = index.get_indexer(cells)
    if (pos < 0).any():
        bad = [c for c, p in zip(cells[:1000], pos[:1000]) if p < 0][:3]
        raise KeyError(f"{int((pos < 0).sum())} barcodes not found (e.g. {bad}); pass a boolean mask, integer "
                       "positions, or barcodes present in obs.index")
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
            lev = s.get("levels") or []
            lev = [lev] if isinstance(lev, str) else list(lev)  # jsonlite auto_unbox writes a 1-level factor as a string
            obs[col] = pd.Categorical(obs[col], categories=lev)
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


def _warn_if_source_changed(path: Path) -> None:
    """A store registered with a ``PROVENANCE.json`` (AutoResearch ``tools/store_register.py``) records its source
    files' size and mtime; warn when a source now differs, so analyses never silently use a stale conversion.
    A source is ``<dataset>/raw/<relative_path>`` for a store at ``<data root>/<owner>/<dataset>/stores/<store>``
    (or ``stores/.superseded/<x>``; AutoResearch layout B), falling back to the legacy ``<data root>/<dataset>/`` raw
    copy, and to ``<data root>/<data>`` for the older ``<data root>/_derived/<name>/<store>``; a source that is not
    reachable (e.g. not mounted) is skipped. Costs a few ``stat`` calls."""
    prov = path / "PROVENANCE.json"
    if not prov.exists():
        return
    try:
        pv = json.loads(prov.read_text(encoding="utf-8"))
        rp = path.resolve()
        stores = rp.parents[1] if rp.parent.name == ".superseded" else rp.parent
        if stores.name == "stores":
            root = stores.parent / "raw"
            if not root.is_dir():
                root = stores.parents[2] / str(pv.get("data", stores.parent.name))
        else:
            root = rp.parents[2] / str(pv.get("data", ""))
    except (OSError, ValueError, IndexError):
        return
    for s in pv.get("sources") or []:
        f = root / str(s.get("relative_path", ""))
        try:
            fs = [f] if f.is_file() else [x for x in f.rglob("*") if x.is_file()] if f.is_dir() else []
            if not fs:
                continue
            size, mt = sum(x.stat().st_size for x in fs), max(x.stat().st_mtime_ns for x in fs)
        except OSError:
            continue
        if size != s.get("bytes") or mt != s.get("mtime_ns"):
            import warnings
            warnings.warn(f"arsc store {path.name}: its source {s.get('path')} changed since the store was registered "
                          f"({pv.get('registered_at', '?')}): size {s.get('bytes')} -> {size}. The store holds the OLD "
                          "data; re-convert and register it again (skill data-stores).", stacklevel=3)


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
    """Lazy, memory-mapped handle on an arsc store directory (see :func:`open`).

    Nothing is read until asked for: ``st.obs`` reads the metadata, ``st.X(assay, layer)`` returns a
    zero-copy CSR backed by the OS page cache (shared by all processes), ``st.get_features`` reads
    only the requested genes, ``st.subset(mask)`` materializes only those cells.

    Parameters
    ----------
    path : store directory (contains ``manifest.json``).
    finalize : build the one-time ``obs.parquet`` / CSC copies if missing.
    build_csc : build per-feature CSC copies (needed by ``get_features``).

    Attributes
    ----------
    path : the store directory.
    manifest : parsed ``manifest.json``.

    Examples
    --------
    >>> import arsc
    >>> st = arsc.datasets.toy(store="toy_store_cls")   # writes a memory-mapped store
    >>> st.n_cells, st.assays
    (3000, ['RNA', 'ADT'])
    >>> X = st.X("RNA", "counts")                     # memory-mapped, no copy
    >>> st.get_features(["CD8A", "GZMK"]).shape       # reads only these genes
    (3000, 2)
    """

    def __init__(self, path, finalize: bool = True, build_csc: bool = True):
        self.path = Path(path)
        mf = self.path / "manifest.json"
        if not mf.exists():
            hint = (" (use arsc.read(path, store=dir) to convert a file)" if self.path.is_file() else
                    "" if self.path.exists() else " (the directory does not exist; check the path)")
            raise FileNotFoundError(f"{path} is not an arsc store: no manifest.json{hint}")
        self.manifest = json.loads(mf.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "arsc-store":
            raise ValueError(f"{path} is not an arsc store (re-export with arsc/R/export_seurat.R)")
        self._obs = None
        self._uns = None
        self._features: dict[str, pd.Index] = {}
        _warn_if_source_changed(self.path)
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
        """Per-cell metadata (read once from ``obs.parquet`` and cached).

        Examples
        --------
        >>> import arsc
        >>> st = arsc.datasets.toy(store="toy_store_obs")
        >>> list(st.obs.columns)
        ['patient', 'timepoint', 'response', 'sample', 'cell_type', 'clone_id']
        """
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
        """Cell barcodes (``obs.index``)."""
        return self.obs.index

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return int(self.manifest["n_cells"])

    @property
    def assays(self) -> list:
        """Assay names."""
        return list(self.manifest.get("assays") or {})

    def layers(self, assay: str) -> list:
        """Layer names of an assay."""
        self._check_assay(assay)
        return list(self.manifest["assays"][assay].get("layers") or {})

    def _check_assay(self, assay: str) -> None:
        if assay not in (self.manifest.get("assays") or {}):
            raise KeyError(f"assay {assay!r} not in store (has {self.assays})")

    def features(self, assay: str = "RNA") -> pd.Index:
        """Feature (gene) names of an assay."""
        self._check_assay(assay)
        if assay not in self._features:
            txt = (self.path / "assays" / assay / "features.txt").read_text(encoding="utf-8")
            self._features[assay] = pd.Index(txt.splitlines())
        return self._features[assay]

    def var(self, assay: str = "RNA") -> pd.DataFrame:
        """Per-feature annotations (``var.parquet`` from h5ad/10x imports; else an empty frame)."""
        p = self.path / "assays" / assay / "var.parquet"
        feats = self.features(assay)
        if p.exists():
            v = pd.read_parquet(p)
            v.index = pd.Index(feats, dtype=object)
            return v
        return pd.DataFrame(index=pd.Index(feats, dtype=object))

    def varm(self, assay: str, key: str) -> np.ndarray:
        """Per-feature matrix imported from AnnData ``varm`` (memory-mapped)."""
        info = (self.manifest["assays"][assay].get("varm") or {})[key]
        return np.load(self.path / "assays" / assay / "varm" / info["file"], mmap_mode="r")

    @property
    def uns(self) -> dict:
        """Unstructured metadata imported from AnnData ``uns`` (JSON-able part)."""
        if self._uns is None:
            p = self.path / "uns.json"
            if p.exists():
                from .formats import _uns_from_json
                self._uns = _uns_from_json(json.loads(p.read_text(encoding="utf-8")))
            else:
                self._uns = {}
        return self._uns

    def variable_features(self, assay: str = "RNA") -> list:
        """Variable features recorded at import (Seurat ``VariableFeatures`` / ``var.highly_variable``)."""
        p = self.path / "assays" / assay / "variable_features.txt"
        return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    def _layer(self, assay, layer):
        if assay not in (self.manifest.get("assays") or {}):
            raise KeyError(f"assay {assay!r} not in store (has {self.assays})")
        layers = self.manifest["assays"][assay].get("layers") or {}
        layer = layer or _default_layer(layers)
        if layer not in layers:
            raise KeyError(f"layer {layer!r} not in assay {assay!r} (has {list(layers)})")
        return layer, layers[layer]

    # -- matrices (zero-copy) ---------------------------------------------------------
    def X(self, assay: str = "RNA", layer: str | None = None) -> sp.csr_matrix:
        """Cells x features CSR backed by memory maps (no copy).

        Examples
        --------
        >>> import arsc
        >>> st = arsc.datasets.toy(store="toy_store_x")
        >>> st.X("RNA", "counts").shape
        (3000, 300)
        """
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

        Examples
        --------
        >>> import arsc
        >>> st = arsc.datasets.toy(store="toy_store_gf")
        >>> df = st.get_features(["CD8A", "MS4A1"], layer="counts")
        >>> list(df.columns), len(df) == st.n_cells
        (['CD8A', 'MS4A1'], True)
        """
        names = [features] if isinstance(features, str) else list(features)
        fidx = self.features(assay).get_indexer(names)
        if (fidx < 0).any():
            raise KeyError(f"not in {assay}: {[n for n, i in zip(names, fidx) if i < 0]} (gene symbols are "
                           "case-sensitive; check st.features(assay))")
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
        if name not in self.reductions:
            raise KeyError(f"reduction {name!r} not in store (has {self.reductions})")
        e = np.load(self.path / "reductions" / f"{name}.npy", mmap_mode="r")
        return e if dims is None else np.asarray(e[:, list(dims)])

    def reduction(self, name: str) -> Reduction:
        """Reduction ``name`` with embedding, loadings and stdev."""
        if name not in self.reductions:
            raise KeyError(f"reduction {name!r} not in store (has {self.reductions})")
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
        """Names of the stored reductions."""
        return list(self.manifest.get("reductions") or {})

    @property
    def graphs(self) -> list:
        """Names of the stored graphs."""
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
            out_assays[a] = self._assay(a, lay)
        reds = {r: self.reduction(r) for r in self.reductions} if reductions else {}
        gr = {g: self.graph(g) for g in self.graphs} if graphs else {}
        return Dataset(self.obs, out_assays, reds, gr, self.manifest, dict(self.uns))

    def _assay(self, a: str, layers: dict) -> Assay:
        var = self.var(a) if (self.path / "assays" / a / "var.parquet").exists() else None
        varm = {k: self.varm(a, k) for k in (self.manifest["assays"][a].get("varm") or {})}
        return Assay(a, self.features(a), layers, self.variable_features(a), var, varm)

    def subset(self, cells, assays=None, layers=None, reductions: bool = True, graphs: bool = False) -> Dataset:
        """Materialize only the selected cells (mask / positions / barcodes) as an in-memory Dataset.

        Examples
        --------
        >>> import arsc
        >>> st = arsc.datasets.toy(store="toy_store_sub")
        >>> cd8 = st.subset((st.obs["cell_type"] == "CD8 T").to_numpy(), assays=["RNA"])
        >>> list(cd8.assays), cd8.n_cells < st.n_cells
        (['RNA'], True)
        """
        idx = _as_positions(cells, self.cells)
        out_assays = {}
        for a in self.assays:
            if assays is not None and a not in assays:
                continue
            lay = {l: _take_rows(self.X(a, l), idx) for l in self.layers(a) if layers is None or l in layers}
            out_assays[a] = self._assay(a, lay)
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
        return Dataset(self.obs.iloc[idx].copy(), out_assays, reds, gr, self.manifest, dict(self.uns))

    # -- hand-off -------------------------------------------------------------------
    def to_anndata(self, cells=None, genes=None, **kw):
        """In-memory ``anndata.AnnData`` of all or selected cells/genes (see :func:`arsc.to_anndata`)."""
        from .formats import to_anndata
        return to_anndata(self, cells, genes, **kw)

    def write_h5ad(self, path, cells=None, genes=None, **kw) -> Path:
        """Stream all or selected cells/genes to ``.h5ad`` in bounded memory (see :func:`arsc.write_h5ad`)."""
        from .formats import write_h5ad
        return write_h5ad(self, path, cells, genes, **kw)

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

    Parameters
    ----------
    path : store directory.
    finalize : build missing one-time files.
    build_csc : build per-feature CSC copies.

    Returns
    -------
    arsc.Store.

    Examples
    --------
    >>> import arsc
    >>> _ = arsc.datasets.toy(store="toy_store_open")
    >>> st = arsc.open("toy_store_open")
    >>> st.n_cells
    3000
    """
    return Store(path, finalize=finalize, build_csc=build_csc)


def read_seurat_export(path, *, assays=None, layers=None, graphs: bool = True, reductions: bool = True) -> Dataset:
    """Load an arsc store as a :class:`Dataset` (memory-mapped arrays; nothing is copied).

    Parameters
    ----------
    path : store directory.
    assays, layers : restrict to these assays / layers.
    graphs, reductions : include graphs / reductions.

    Returns
    -------
    arsc.Dataset.

    Examples
    --------
    >>> import arsc
    >>> _ = arsc.datasets.toy(store="toy_store_rse")
    >>> arsc.read_seurat_export("toy_store_rse", assays=["RNA"]).n_cells
    3000
    """
    return open(path).to_dataset(assays=assays, layers=layers, reductions=reductions, graphs=graphs)


# --------------------------------------------------------------------------- tidy outputs
def write_table(df: pd.DataFrame, path, index: bool | None = None) -> Path:
    """Write a tidy table (e.g. for plotting in R); format from the extension (.parquet/.csv/.tsv).

    Read in R with ``arrow::read_parquet`` or ``data.table::fread``. Parent directories are created.

    Parameters
    ----------
    df : the table.
    path : output path ending in ``.parquet``, ``.csv`` or ``.tsv``.
    index : write the index (default: only if it is not a RangeIndex).

    Returns
    -------
    pathlib.Path of the written file.

    Examples
    --------
    >>> import arsc, pandas as pd
    >>> arsc.write_table(pd.DataFrame({"a": [1, 2]}), "out/table.csv").name
    'table.csv'
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
        raise ValueError(f"unsupported extension {p.suffix!r}; use .parquet, .csv or .tsv")
    return p


def from_anndata(h5ad, out_dir, *, layers=("counts",), assay: str = "RNA", use_raw_for=None,
                 obsm: bool = True, chunk: int | None = None, obsp: bool = False) -> Path:
    """Convert an .h5ad into an arsc store so that many processes share one memory-mapped copy.

    Compatibility wrapper around :func:`arsc.read_h5ad`; prefer ``arsc.read(h5ad, store=out_dir)``,
    which imports every layer by default.

    ``layers``: AnnData layer names to export (``"X"`` exports ``.X`` as layer ``X``).
    ``use_raw_for``: name under which to export ``.raw.X``. This is a layer of ``assay`` when raw has
    the same genes, and otherwise a separate assay of that name. The matrices are streamed from
    disk in chunks (memory-safe; ``chunk`` is accepted for backward compatibility and ignored).
    ``obsm`` embeddings become reductions. Idempotent: if ``out_dir`` already holds a store, it is
    returned untouched.

    Parameters
    ----------
    h5ad : input ``.h5ad``.
    out_dir : store directory.
    layers : AnnData layers to export.
    assay : assay name.
    use_raw_for : name for ``.raw.X``.
    obsm : export embeddings.
    chunk : ignored (backward compatibility).
    obsp : export graphs.

    Returns
    -------
    pathlib.Path ``out_dir``.

    Examples
    --------
    >>> import arsc
    >>> f = arsc.datasets.toy_h5ad("toy_fa.h5ad", n_cells=300)
    >>> arsc.open(arsc.from_anndata(f, "toy_fa_store", layers=("X",))).n_cells
    300
    """
    from .formats import ArscFormatError, read_h5ad

    out = Path(out_dir)
    if (out / "manifest.json").exists():
        return out
    kw = dict(assay=assay, layers=list(layers), obsm=obsm, obsp=obsp)
    if use_raw_for:
        try:
            read_h5ad(h5ad, out, raw=use_raw_for, raw_as_layer=True, **kw)
        except ArscFormatError:
            read_h5ad(h5ad, out, raw=use_raw_for, raw_as_layer=False, **kw)
    else:
        read_h5ad(h5ad, out, raw=None, **kw)
    return out
