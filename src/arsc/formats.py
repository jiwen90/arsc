"""Format readers and writers: h5ad, 10x (HDF5 / Matrix Market), Seurat (through R) <-> arsc.

Every reader streams its matrices through :class:`StoreBuilder`. The builder writes row chunks (or
column chunks for CSC sources) straight into ``.npy`` files of unknown final length, so converting
a file never holds a full matrix in RAM. Given ``out=None``, the same builder collects the chunks in
memory and returns an in-memory :class:`~arsc.io.Dataset`.

Entry points (all re-exported as ``arsc.*``)::

    arsc.read(path)                        # in-memory Dataset, format detected from the path
    arsc.read(path, store="dir/")          # converted once into a memory-mapped Store
    arsc.read_h5ad / read_10x_h5 / read_10x_mtx / read_seurat
    arsc.write_h5ad(store_or_dataset, "out.h5ad", cells=..., genes=...)
    store.to_anndata(cells=..., genes=...)  # scanpy hand-off

Dtype policy (``dtype="auto"``): integer-valued matrices are stored as float32. This is exact
below 2**24, and it is what the Seurat exporter writes. Non-integer float64 data stays float64.
Float32 stays float32. The source dtype is recorded in the manifest, and h5ad export restores it.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import struct
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .io import Assay, Dataset, Reduction, Store, _as_positions, _build_csc, _take_rows
from .io import open as _open_store

PathLike = Union[str, "os.PathLike[str]"]
DtypeArg = Literal["auto", "float32", "float64"]

__all__ = [
    "ArscFormatError", "StoreBuilder", "read", "read_h5ad", "read_10x_h5", "read_10x_mtx", "read_seurat",
    "write_h5ad", "to_anndata", "make_unique", "detect_format",
]

F32_EXACT = 2 ** 24          # largest integer range float32 represents exactly
CHUNK_NNZ = 1 << 25          # ~33M non-zeros per streamed chunk (~270 MB at float64 + int32)
TENX_ASSAYS = {"Gene Expression": "RNA", "Antibody Capture": "ADT", "CRISPR Guide Capture": "CRISPR",
               "Multiplexing Capture": "HTO", "Peaks": "ATAC", "Custom": "Custom"}


class ArscFormatError(ValueError):
    """Input file is not in a supported format, or has an inconsistent structure (a ``ValueError``).

    Attributes
    ----------
    args : the message (says which file and what was expected).

    Examples
    --------
    >>> import arsc
    >>> try:
    ...     arsc.detect_format("data.xyz")
    ... except arsc.ArscFormatError as e:
    ...     print("unknown format" in str(e))
    True
    """


# =========================================================================== small helpers
def make_unique(names: Sequence[str], join: str = "-") -> list[str]:
    """Deduplicate names exactly like ``anndata.utils.make_index_unique`` (``a, a -> a, a-1``).

    Parameters
    ----------
    names : sequence of names.
    join : separator before the counter.

    Returns
    -------
    list of str, unique, first occurrences unchanged.

    Examples
    --------
    >>> import arsc
    >>> arsc.make_unique(["TBCE", "TBCE", "CD8A"])
    ['TBCE', 'TBCE-1', 'CD8A']
    """
    values = [str(v) for v in names]
    seen_first: set[str] = set()
    dup_pos = []
    for i, v in enumerate(values):
        if v in seen_first:
            dup_pos.append(i)
        else:
            seen_first.add(v)
    if not dup_pos:
        return values
    taken = set(values)
    counter: Counter = Counter()
    for i in dup_pos:
        v = values[i]
        while True:
            counter[v] += 1
            cand = f"{v}{join}{counter[v]}"
            if cand not in taken:
                taken.add(cand)
                values[i] = cand
                break
    return values


def _npy_header(dtype: np.dtype, n: int, size: int = 128) -> bytes:
    d = "{'descr': %r, 'fortran_order': False, 'shape': (%d,), }" % (np.lib.format.dtype_to_descr(dtype), n)
    body = d.encode("latin1")
    pad = size - 10 - len(body) - 1
    if pad < 0:
        raise ArscFormatError("npy header overflow")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", size - 10) + body + b" " * pad + b"\n"


class _NpyAppender:
    """Append-only 1-D ``.npy`` writer of unknown final length (header patched on close)."""

    HEADER = 128

    def __init__(self, path: Path, dtype):
        self.path, self.dtype, self.n = Path(path), np.dtype(dtype), 0
        self._f = self.path.open("wb")
        self._f.write(b"\0" * self.HEADER)

    def append(self, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=self.dtype)
        if a.size:
            self._f.write(memoryview(a).cast("B"))
            self.n += a.size

    def close(self) -> None:
        self._f.seek(0)
        self._f.write(_npy_header(self.dtype, self.n, self.HEADER))
        self._f.close()


def _is_integer_valued(x: np.ndarray) -> bool:
    if x.dtype.kind in "iub":
        return True
    if x.size == 0:
        return True
    with np.errstate(invalid="ignore"):
        return bool(np.all(np.floor(x) == x))


def _absmax(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(max(abs(float(np.nanmax(x))), abs(float(np.nanmin(x)))))


def _target_dtype(src: np.dtype, dtype: DtypeArg) -> tuple[np.dtype, bool]:
    """(dtype written while streaming, whether to try float32 down-conversion at the end)."""
    if dtype in ("float32", "float64"):
        return np.dtype(dtype), False
    if dtype != "auto":
        raise ValueError(f"dtype must be 'auto', 'float32' or 'float64', not {dtype!r}")
    if src == np.float32 or src.kind in "ub" or (src.kind == "i" and src.itemsize <= 2):
        return np.dtype(np.float32), False
    if src.kind == "i":  # int32/int64 counts: float32 directly, overflow checked per chunk
        return np.dtype(np.float32), False
    return np.dtype(np.float64), True


def _uns_to_json(x: Any, path: str, skipped: list) -> Any:
    """JSON-able copy of an ``uns`` element; unsupported leaves are listed in ``skipped``."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            r = _uns_to_json(v, f"{path}/{k}", skipped)
            if r is not _SKIP:
                out[str(k)] = r
        return out
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, (list, tuple)):
        r = [_uns_to_json(v, f"{path}[]", skipped) for v in x]
        return _SKIP if any(v is _SKIP for v in r) else r
    if isinstance(x, np.ndarray) and x.size <= 1_000_000 and x.dtype.kind in "biufUSO":
        if x.dtype.kind == "O" and not all(isinstance(v, str) for v in x.ravel()):
            skipped.append(path)
            return _SKIP
        vals = x.astype(str).tolist() if x.dtype.kind in "SO" else x.tolist()
        return {"__ndarray__": vals, "dtype": "str" if x.dtype.kind in "USO" else x.dtype.str}
    skipped.append(path)
    return _SKIP


_SKIP = object()


def _uns_from_json(x: Any, base: Path | None = None) -> Any:
    """Inverse of :func:`_uns_to_json`, plus the shapes the R exporter writes for a full Seurat export: arrays with
    ``shape``/``order``/``levels``, ``__dataframe__``, and files under ``uns/`` (``__npy__``, ``__csv__``, ``__mtx__``,
    ``__txt__``; resolved against the store folder ``base``, memory-mapped where possible). ``__deparsed__`` (an R
    formula or function) stays a dict with its R source text."""
    if isinstance(x, dict):
        if "__ndarray__" in x:
            vals = x["__ndarray__"]
            vals = vals if isinstance(vals, list) else [vals]
            if x.get("dtype") == "str":
                a = np.array(vals, dtype=object)
            else:
                dt = np.dtype(x.get("dtype", "<f8"))
                a = np.array([np.nan if v is None and dt.kind == "f" else v for v in vals], dtype=dt)
            shape = x.get("shape")
            if shape is not None:
                shape = [int(v) for v in (shape if isinstance(shape, list) else [shape])]
                if len(shape) > 1 and int(np.prod(shape)) == a.size:
                    a = a.reshape(shape, order=x.get("order", "C"))
            if x.get("levels") is not None:
                lev = x["levels"] if isinstance(x["levels"], list) else [x["levels"]]
                return pd.Categorical(a.ravel(), categories=lev)
            if x.get("names") is not None and a.ndim == 1:
                return pd.Series(a, index=x["names"] if isinstance(x["names"], list) else [x["names"]])
            return a
        if "__dataframe__" in x:
            cols = {k: _uns_from_json(v, base) for k, v in (x["__dataframe__"] or {}).items()}
            idx = x.get("index")
            idx = idx if isinstance(idx, list) or idx is None else [idx]
            cols = {k: (np.asarray(v).ravel() if isinstance(v, np.ndarray) else v) for k, v in cols.items()}
            return pd.DataFrame(cols, index=idx)
        if base is not None:
            if "__npy__" in x:
                a = np.load(Path(base) / x["__npy__"], mmap_mode="r")
                return a.astype(bool) if x.get("logical") else a
            if "__csv__" in x:
                df = pd.read_csv(Path(base) / x["__csv__"], keep_default_na=True, na_values=["NA"])
                return df.set_index("_index") if "_index" in df.columns else df
            if "__mtx__" in x:
                import scipy.io
                m = sp.csr_matrix(scipy.io.mmread(str(Path(base) / x["__mtx__"])))
                return {"matrix": m, "rownames": x.get("rownames"), "colnames": x.get("colnames")}
            if "__txt__" in x:
                vals = (Path(base) / x["__txt__"]).read_text(encoding="utf-8").split("\n")
                vals = vals[:-1] if vals and vals[-1] == "" else vals
                return np.array([None if v == x.get("na") else v for v in vals], dtype=object)
        return {k: _uns_from_json(v, base) for k, v in x.items()}
    if isinstance(x, list):
        return [_uns_from_json(v, base) for v in x]
    return x


def _write_parquet_safe(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write ``df`` to parquet. Columns Arrow cannot type (mixed objects) are cast to str, with a warning."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df.index = df.index.astype(str)
    try:
        df.to_parquet(path)
        return df
    except Exception:  # noqa: BLE001 - arrow raises several types; fall back column by column
        pass
    bad = []
    for c in df.columns:
        try:
            df[[c]].to_parquet(path.with_suffix(".probe"))
        except Exception:  # noqa: BLE001
            bad.append(c)
            df[c] = df[c].astype(str)
    path.with_suffix(".probe").unlink(missing_ok=True)
    if bad:
        warnings.warn(f"columns stored as text (mixed types Arrow cannot type): {bad}", stacklevel=3)
    df.to_parquet(path)
    return df


# =========================================================================== builder
class _LayerSink:
    """Receives one layer as row chunks (CSR) or column chunks (CSC), on disk or in memory."""

    def __init__(self, builder: "StoreBuilder", assay: str, name: str, shape: tuple[int, int],
                 src_dtype, dtype: DtypeArg, orient: str):
        self.b, self.assay, self.name, self.shape, self.orient = builder, assay, name, tuple(shape), orient
        self.src_dtype = np.dtype(src_dtype)
        self.dtype, self.try_f32 = _target_dtype(self.src_dtype, dtype)
        self.forced = dtype != "auto"
        self.int_valued, self.maxabs, self.done, self.nnz = True, 0.0, 0, 0
        self.n_major = shape[0] if orient == "rows" else shape[1]
        n_minor = shape[1] if orient == "rows" else shape[0]
        self.idx_dtype = np.int32 if n_minor < 2 ** 31 else np.int64
        tag = "csr" if orient == "rows" else "csc"
        if builder.tmp is not None:
            base = builder.tmp / "assays" / assay
            self.prefix = str(base / f"{name}.{tag}")
            self.f_data = _NpyAppender(Path(self.prefix + ".data.npy"), self.dtype)
            self.f_ind = _NpyAppender(Path(self.prefix + ".indices.npy"), self.idx_dtype)
            self.ptr = [np.zeros(1, np.int64)]
        else:
            self.chunks = []

    def add(self, m) -> None:
        """Append the next chunk: CSR rows (``orient='rows'``) or CSC columns (``orient='cols'``)."""
        if self.orient == "rows":
            m = m.tocsr() if sp.issparse(m) else sp.csr_matrix(m)
            if m.shape[1] != self.shape[1]:
                raise ArscFormatError(f"{self.assay}/{self.name}: chunk has {m.shape[1]} columns, expected {self.shape[1]}")
            k = m.shape[0]
        else:
            m = m.tocsc() if sp.issparse(m) else sp.csc_matrix(m)
            if m.shape[0] != self.shape[0]:
                raise ArscFormatError(f"{self.assay}/{self.name}: chunk has {m.shape[0]} rows, expected {self.shape[0]}")
            k = m.shape[1]
        if self.done + k > self.n_major:
            raise ArscFormatError(f"{self.assay}/{self.name}: more {self.orient} than declared ({self.n_major})")
        m.sum_duplicates()
        m.sort_indices()
        d = m.data
        if self.int_valued and not _is_integer_valued(d):
            self.int_valued = False
        mx = _absmax(d) if d.dtype.kind in "iuf" else 0.0
        self.maxabs = max(self.maxabs, mx)
        if self.dtype == np.float32 and self.src_dtype.kind in "iu" and not self.forced and mx >= F32_EXACT:
            raise ArscFormatError(f"{self.assay}/{self.name}: integer values >= 2**24 are not exact in float32; "
                                  "pass dtype='float64'")
        if self.b.tmp is not None:
            self.f_data.append(d)
            self.f_ind.append(m.indices)
            self.ptr.append(np.asarray(m.indptr[1:], np.int64) + self.nnz)
        else:
            self.chunks.append(m.astype(self.dtype, copy=False))
        self.nnz += int(m.nnz)
        self.done += k

    def close(self) -> dict:
        if self.done != self.n_major:
            raise ArscFormatError(f"{self.assay}/{self.name}: received {self.done} {self.orient}, expected {self.n_major}")
        final = self.dtype
        if self.try_f32 and self.int_valued and self.maxabs < F32_EXACT:
            final = np.dtype(np.float32)
        entry = {"shape": [int(self.shape[0]), int(self.shape[1])], "nnz": int(self.nnz),
                 "dtype": "float32" if final == np.float32 else "float64", "integer_valued": bool(self.int_valued),
                 "source_dtype": self.src_dtype.str, "prefix": f"{self.name}.csr", "format": "csr"}
        if self.b.tmp is None:
            stack = sp.vstack if self.orient == "rows" else sp.hstack
            m = stack(self.chunks, format="csr") if self.chunks else sp.csr_matrix(self.shape, dtype=final)
            m = sp.csr_matrix(m.astype(final, copy=False))
            m.sort_indices()
            self.b._mem_layers.setdefault(self.assay, {})[self.name] = m
            return entry
        self.f_data.close()
        self.f_ind.close()
        ptr = np.concatenate(self.ptr)
        np.save(self.prefix + ".indptr.npy", ptr)
        if final != self.dtype:
            _recast_npy(Path(self.prefix + ".data.npy"), final)
        if self.orient == "cols":  # build the row-major copy from the column-major one
            base = self.prefix[: -len(".csc")]
            _build_csc(self.prefix, base + ".csr", (self.shape[1], self.shape[0]))
        return entry


def _recast_npy(path: Path, dtype) -> None:
    src = np.load(path, mmap_mode="r")
    tmp = path.with_name(path.name + ".recast")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=src.shape)
    step = 1 << 26
    for s in range(0, src.shape[0], step):
        out[s:s + step] = src[s:s + step]
    out.flush()
    del out, src
    os.replace(tmp, path)


class StoreBuilder:
    """Assemble an arsc store (``out`` = directory) or an in-memory Dataset (``out=None``) piece by piece.

    The order is cells, then assays and layers, then the optional reductions, graphs, varm and uns,
    and finally :meth:`finish`. The directory version writes into ``<out>.tmp<pid>`` and renames it
    when finished, so an interrupted conversion never leaves a half-written store behind.

    Parameters
    ----------
    out : store directory, or ``None`` for an in-memory :class:`arsc.Dataset`.
    source : free-text provenance recorded in the manifest.
    overwrite : replace an existing store at ``out``.

    Examples
    --------
    >>> import arsc, numpy as np, pandas as pd, scipy.sparse as sp
    >>> counts = sp.random(100, 20, density=0.2, format="csr", random_state=0) * 10
    >>> b = arsc.StoreBuilder("my_store")
    >>> b.set_cells([f"c{i}" for i in range(100)], pd.DataFrame({"patient": ["P1", "P2"] * 50}))
    >>> b.add_assay("RNA", [f"g{j}" for j in range(20)])
    >>> b.add_layer("RNA", "counts", counts.floor())
    >>> st = b.finish()
    >>> st.n_cells, st.layers("RNA")
    (100, ['counts'])
    """

    def __init__(self, out: PathLike | None, *, source: str = "", overwrite: bool = False):
        self.out = Path(out) if out is not None else None
        self.tmp: Path | None = None
        if self.out is not None:
            if (self.out / "manifest.json").exists() and not overwrite:
                raise FileExistsError(f"store already exists: {self.out} (pass overwrite=True to replace it)")
            if self.out.exists() and any(self.out.iterdir()) and not (self.out / "manifest.json").exists():
                raise FileExistsError(f"{self.out} exists and is not an arsc store; refusing to overwrite it")
            self.tmp = self.out.with_name(self.out.name + f".tmp{os.getpid()}")
            if self.tmp.exists():
                shutil.rmtree(self.tmp)
            self.tmp.mkdir(parents=True)
        self.manifest: dict[str, Any] = {"format": "arsc-store", "format_version": 1, "source": source,
                                         "n_cells": None, "default_assay": None, "assays": {},
                                         "reductions": {}, "graphs": {}, "obs": {}, "writer": "arsc.formats"}
        self._cells: pd.Index | None = None
        self._obs: pd.DataFrame | None = None
        self._mem_layers: dict[str, dict] = {}
        self._mem_assays: dict[str, dict] = {}
        self._mem_reds: dict[str, Reduction] = {}
        self._mem_graphs: dict[str, sp.csr_matrix] = {}
        self._uns: dict = {}

    # ---------------------------------------------------------------- cells / obs
    def set_cells(self, cells: Sequence[str], obs: pd.DataFrame | None = None) -> None:
        """Set the cell barcodes (and optional per-cell metadata, row-aligned with ``cells``)."""
        cells = pd.Index([str(c) for c in cells])
        self._cells = cells
        self.manifest["n_cells"] = int(len(cells))
        if obs is None:
            obs = pd.DataFrame(index=cells)
        if len(obs) != len(cells):
            raise ArscFormatError(f"obs has {len(obs)} rows for {len(cells)} cells")
        obs = obs.copy()
        obs.index = cells
        if self.tmp is not None:
            (self.tmp / "cells.txt").write_text("\n".join(cells), encoding="utf-8")
            obs = _write_parquet_safe(obs, self.tmp / "obs.parquet")
        obs.index.name = None
        self._obs = obs

    @property
    def n_cells(self) -> int:
        if self._cells is None:
            raise RuntimeError("call set_cells() first")
        return len(self._cells)

    # ---------------------------------------------------------------- assays / layers
    def add_assay(self, name: str, features: Sequence[str], *, var: pd.DataFrame | None = None,
                  variable_features: Sequence[str] | None = None) -> None:
        """Declare an assay with its feature names (optional per-feature ``var`` and variable features)."""
        feats = [str(f) for f in features]
        if len(set(feats)) != len(feats):
            raise ArscFormatError(f"assay {name!r}: feature names are not unique (use arsc.make_unique)")
        if var is not None and len(var) != len(feats):
            raise ArscFormatError(f"assay {name!r}: var has {len(var)} rows for {len(feats)} features")
        if variable_features is None and var is not None and "highly_variable" in var:
            hv = var["highly_variable"].to_numpy()
            if hv.dtype == bool:
                variable_features = [f for f, h in zip(feats, hv) if h]
        entry = {"class": "Assay", "layers": {}, "n_features": len(feats)}
        self.manifest["assays"][name] = entry
        if self.manifest["default_assay"] is None:
            self.manifest["default_assay"] = name
        var_df = None
        if var is not None and var.shape[1] > 0:
            var_df = var.copy()
            var_df.index = pd.Index(feats)
        if self.tmp is not None:
            d = self.tmp / "assays" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "features.txt").write_text("\n".join(feats), encoding="utf-8")
            if variable_features:
                (d / "variable_features.txt").write_text("\n".join(map(str, variable_features)), encoding="utf-8")
            if var_df is not None:
                var_df = _write_parquet_safe(var_df, d / "var.parquet")
                entry["var"] = True
        self._mem_assays[name] = {"features": pd.Index(feats), "var": var_df,
                                  "variable_features": list(map(str, variable_features or [])), "varm": {}}

    def layer(self, assay: str, name: str, *, n_features: int | None = None, src_dtype="float32",
              dtype: DtypeArg = "auto", orient: Literal["rows", "cols"] = "rows") -> _LayerSink:
        """Open a sink for layer ``name``. Feed chunks with ``.add(m)``, then call ``.close()``
        (or use the sink as a context manager)."""
        if assay not in self.manifest["assays"]:
            raise KeyError(f"add_assay({assay!r}) first")
        nf = n_features if n_features is not None else self.manifest["assays"][assay]["n_features"]
        return _LayerCtx(self, _LayerSink(self, assay, name, (self.n_cells, nf), src_dtype, dtype, orient))

    def add_layer(self, assay: str, name: str, m, *, dtype: DtypeArg = "auto", chunk_rows: int = 50_000) -> None:
        """Add an in-memory matrix (cells x features; dense or sparse) as one layer."""
        src = np.asarray(m.data if sp.issparse(m) else m).dtype if (sp.issparse(m) or isinstance(m, np.ndarray)) else np.float64
        m = m.tocsr() if sp.issparse(m) else sp.csr_matrix(np.asarray(m))
        with self.layer(assay, name, n_features=m.shape[1], src_dtype=src, dtype=dtype) as sink:
            for s in range(0, m.shape[0], chunk_rows):
                sink.add(m[s:s + chunk_rows])

    # ---------------------------------------------------------------- reductions / graphs / varm / uns
    def add_reduction(self, name: str, embedding, *, columns: Sequence[str] | None = None,
                      loadings: np.ndarray | None = None, loadings_features: Sequence[str] | None = None,
                      stdev: np.ndarray | None = None, assay: str | None = None, key: str | None = None,
                      frame: bool = False) -> None:
        """Add a cells x k embedding (PCA, harmony, UMAP...), optionally with loadings and stdev."""
        emb = np.asarray(embedding)
        if emb.ndim != 2 or emb.shape[0] != self.n_cells:
            raise ArscFormatError(f"reduction {name!r}: expected a {self.n_cells} x k array, got {emb.shape}")
        cols = [str(c) for c in columns] if columns is not None else [f"{name}_{i + 1}" for i in range(emb.shape[1])]
        e = {"shape": list(emb.shape), "columns": cols, "assay": assay or self.manifest["default_assay"],
             "source_dtype": emb.dtype.str, "obsm_key": key or f"X_{name}"}
        if frame:
            e["frame"] = True
        if stdev is not None:
            e["stdev"] = [float(v) for v in np.asarray(stdev)]
        e64 = emb.astype(np.float64)
        L = None
        if loadings is not None:
            L = pd.DataFrame(np.asarray(loadings, dtype=np.float64), index=list(map(str, loadings_features)), columns=cols)
            e["loadings_shape"] = list(L.shape)
        if self.tmp is not None:
            rd = self.tmp / "reductions"
            rd.mkdir(exist_ok=True)
            np.save(rd / f"{name}.npy", e64)
            if L is not None:
                np.save(rd / f"{name}.loadings.npy", L.to_numpy())
                (rd / f"{name}.loadings_features.txt").write_text("\n".join(L.index), encoding="utf-8")
        self.manifest["reductions"][name] = e
        self._mem_reds[name] = Reduction(e64, cols, L, np.asarray(stdev) if stdev is not None else None, e["assay"])

    def add_graph(self, name: str, m, *, key: str | None = None) -> None:
        """Add a cells x cells graph (e.g. an SNN)."""
        g = sp.csr_matrix(m)
        if g.shape != (self.n_cells, self.n_cells):
            raise ArscFormatError(f"graph {name!r}: expected {self.n_cells} x {self.n_cells}, got {g.shape}")
        g.sort_indices()
        entry = {"shape": list(g.shape), "nnz": int(g.nnz), "dtype": g.dtype.name, "transposed": False,
                 "prefix": f"{name}.csr", "obsp_key": key or name}
        if self.tmp is not None:
            gd = self.tmp / "graphs"
            gd.mkdir(exist_ok=True)
            for comp in ("data", "indices", "indptr"):
                np.save(gd / f"{name}.csr.{comp}.npy", getattr(g, comp))
        self.manifest["graphs"][name] = entry
        self._mem_graphs[name] = g

    def add_varm(self, assay: str, key: str, arr) -> None:
        """Add a per-feature matrix (features x k) to an assay."""
        a = np.asarray(arr)
        nf = self.manifest["assays"][assay]["n_features"]
        if a.shape[0] != nf:
            raise ArscFormatError(f"varm {key!r}: {a.shape[0]} rows for {nf} features")
        if self.tmp is not None:
            d = self.tmp / "assays" / assay / "varm"
            d.mkdir(exist_ok=True)
            np.save(d / f"{_safe_name(key)}.npy", a)
        self.manifest["assays"][assay].setdefault("varm", {})[key] = {"file": f"{_safe_name(key)}.npy",
                                                                     "shape": list(a.shape), "dtype": a.dtype.str}
        self._mem_assays[assay]["varm"][key] = a

    def set_uns(self, uns: dict) -> list[str]:
        """Store the JSON-able part of ``uns``; returns the paths that were skipped."""
        skipped: list[str] = []
        js = _uns_to_json(uns or {}, "uns", skipped)
        self._uns = _uns_from_json(js)
        if self.tmp is not None and js:
            (self.tmp / "uns.json").write_text(json.dumps(js), encoding="utf-8")
        if skipped:
            self.manifest["uns_skipped"] = skipped
        return skipped

    # ---------------------------------------------------------------- finish
    def finish(self, *, open_store: bool = True, build_csc: bool = True) -> Store | Dataset | Path:
        """Write the manifest and return the opened Store (or the Dataset when building in memory)."""
        if self._cells is None:
            raise RuntimeError("no cells set")
        for a, e in self.manifest["assays"].items():
            if not e["layers"]:
                raise ArscFormatError(f"assay {a!r} has no layers")
        if self.tmp is None:
            return self._finish_memory()
        (self.tmp / "manifest.json").write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        if self.out.exists():
            shutil.rmtree(self.out)
        os.replace(self.tmp, self.out)
        if not open_store:
            return self.out
        return _open_store(self.out, build_csc=build_csc)

    def _finish_memory(self) -> Dataset:
        assays = {}
        for a, info in self._mem_assays.items():
            layers = self._mem_layers.get(a, {})
            assays[a] = Assay(a, info["features"], dict(layers), info["variable_features"], info["var"], info["varm"])
        obs = self._obs if self._obs is not None else pd.DataFrame(index=self._cells)
        return Dataset(obs, assays, dict(self._mem_reds), dict(self._mem_graphs), self.manifest, self._uns)


class _LayerCtx:
    def __init__(self, b: StoreBuilder, sink: _LayerSink):
        self.b, self.sink = b, sink

    def __enter__(self) -> _LayerSink:
        return self.sink

    def __exit__(self, et, ev, tb):
        if et is None:
            self.b.manifest["assays"][self.sink.assay]["layers"][self.sink.name] = self.sink.close()
        return False

    # allow non-context use
    def add(self, m) -> None:
        self.sink.add(m)

    def close(self) -> None:
        self.__exit__(None, None, None)


def _safe_name(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(key))


# =========================================================================== HDF5 matrix streaming
def _h5():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading HDF5 files needs h5py: pip install 'arsc[h5ad]'") from e
    return h5py


def _matrix_kind(node) -> tuple[str, tuple[int, int], np.dtype]:
    h5py = _h5()
    if isinstance(node, h5py.Dataset):
        if node.ndim != 2:
            raise ArscFormatError(f"{node.name}: expected a 2-D matrix, got shape {node.shape}")
        return "dense", tuple(int(s) for s in node.shape), node.dtype
    enc = node.attrs.get("encoding-type")
    if isinstance(enc, bytes):
        enc = enc.decode()
    if enc is None and "h5sparse_format" in node.attrs:  # anndata < 0.7
        fmt = node.attrs["h5sparse_format"]
        enc = (fmt.decode() if isinstance(fmt, bytes) else fmt) + "_matrix"
    shape = node.attrs.get("shape", node.attrs.get("h5sparse_shape"))
    if enc not in ("csr_matrix", "csc_matrix") or shape is None:
        raise ArscFormatError(f"{node.name}: unsupported matrix encoding {enc!r}")
    return enc[:3], (int(shape[0]), int(shape[1])), node["data"].dtype


def _ptr_chunks(indptr: np.ndarray, chunk_nnz: int, max_major: int = 1 << 20) -> Iterator[tuple[int, int]]:
    n = indptr.size - 1
    s = 0
    while s < n:
        target = indptr[s] + chunk_nnz
        e = int(np.searchsorted(indptr, target, side="right")) - 1
        e = min(max(e, s + 1), n, s + max_major)
        yield s, e
        s = e


def _stream_compressed(node, kind: str, shape, chunk_nnz: int) -> Iterator[sp.spmatrix]:
    """Yield CSR row chunks (kind 'csr') or CSC column chunks (kind 'csc') of an on-disk sparse matrix."""
    indptr = np.asarray(node["indptr"][:], dtype=np.int64)
    data, indices = node["data"], node["indices"]
    for s, e in _ptr_chunks(indptr, chunk_nnz):
        a, b = int(indptr[s]), int(indptr[e])
        ptr = indptr[s:e + 1] - a
        d, i = data[a:b], indices[a:b]
        if kind == "csr":
            yield sp.csr_matrix((d, i, ptr), shape=(e - s, shape[1]))
        else:
            yield sp.csc_matrix((d, i, ptr), shape=(shape[0], e - s))


def _stream_dense(node, shape, chunk_elems: int = 1 << 25) -> Iterator[sp.csr_matrix]:
    rows = max(1, chunk_elems // max(1, shape[1]))
    for s in range(0, shape[0], rows):
        yield sp.csr_matrix(np.asarray(node[s:s + rows]))


def _copy_matrix(node, builder: StoreBuilder, assay: str, layer: str, *, dtype: DtypeArg, chunk_nnz: int) -> None:
    kind, shape, src = _matrix_kind(node)
    if shape[0] != builder.n_cells:
        raise ArscFormatError(f"{node.name}: {shape[0]} rows for {builder.n_cells} cells")
    orient = "cols" if kind == "csc" else "rows"
    with builder.layer(assay, layer, n_features=shape[1], src_dtype=src, dtype=dtype, orient=orient) as sink:
        it = _stream_dense(node, shape) if kind == "dense" else _stream_compressed(node, kind, shape, chunk_nnz)
        for m in it:
            sink.add(m)


# =========================================================================== h5ad
def _anndata_elem_io():
    try:
        from anndata.io import read_elem, write_elem  # anndata >= 0.11
    except ImportError:  # pragma: no cover
        try:
            from anndata.experimental import read_elem, write_elem  # 0.8 - 0.10
        except ImportError as e:
            raise ImportError("h5ad support needs anndata >= 0.8: pip install 'arsc[h5ad]'") from e
    return read_elem, write_elem


def read_h5ad(path: PathLike, store: PathLike | None = None, *, assay: str = "RNA", x_layer: str = "X",
              layers: Sequence[str] | None = None, raw: str | None = "raw", raw_as_layer: bool = False,
              obsm: bool = True, varm: bool = True, obsp: bool = True, uns: bool = True,
              dtype: DtypeArg = "auto", chunk_nnz: int = CHUNK_NNZ, overwrite: bool = False) -> Store | Dataset:
    """Read an ``.h5ad`` file into a memory-mapped store (``store=dir``) or an in-memory Dataset.

    The expression matrices (``X``, ``layers``, ``raw/X``) are streamed from disk in chunks of
    about ``chunk_nnz`` non-zeros, so a file of any size converts in bounded memory. Dense, CSR and
    CSC encodings are all supported; for CSC the row-major copy is built out of core. Mapping:

    ===================  =====================================================================
    ``X``                layer ``x_layer`` (default ``"X"``) of assay ``assay``
    ``layers[k]``        layer ``k`` of ``assay`` (``layers=None``: all; or a list; ``"X"`` = X)
    ``raw``              assay ``raw`` (own feature set, layer ``"X"``); ``raw=None`` skips it.
                         ``raw_as_layer=True`` stores it as layer ``raw`` of ``assay`` instead
                         (only if raw has the same genes)
    ``obs`` / ``var``    ``obs.parquet`` / ``assays/<assay>/var.parquet`` (categoricals, nullable
                         ints/bools, strings preserved); ``var.highly_variable`` -> variable features
    ``obsm[k]``          reduction ``k`` without the ``X_`` prefix (DataFrames kept with columns)
    ``varm[k]``          ``Store.varm(assay, k)``; ``varm["PCs"]`` also becomes the PCA loadings
    ``obsp[k]``          graph ``k``
    ``uns``              JSON-able part (scalars, strings, lists, arrays, nested dicts) -> ``Store.uns``
    ===================  =====================================================================

    With ``store`` set, the store is written atomically and then opened. If the same source was
    already converted there, the existing store is opened and nothing is rewritten.

    Parameters
    ----------
    path : the ``.h5ad`` file.
    store : output store directory; ``None`` returns an in-memory Dataset.
    assay : assay name for ``X`` and ``layers``.
    x_layer : layer name for ``X``.
    layers : AnnData layers to import (``None`` = all).
    raw : assay name for ``.raw`` (``None`` skips it).
    raw_as_layer : store ``.raw`` as a layer of ``assay`` (same genes only).
    obsm, varm, obsp, uns : import these AnnData slots.
    dtype : value dtype on disk (``"auto"``: float32 when exact, else the source dtype).
    chunk_nnz : non-zeros per streamed chunk.
    overwrite : replace an existing store. Without it, an existing store built from a file of the same name
        is reopened as is (with a warning if the file is newer than the store).

    Returns
    -------
    arsc.Store (with ``store=``) or arsc.Dataset. Needs ``anndata`` and ``h5py`` (``pip install "arsc[h5ad]"``).

    Examples
    --------
    >>> import arsc
    >>> f = arsc.datasets.toy_h5ad("toy_a.h5ad", n_cells=500)
    >>> ds = arsc.read_h5ad(f)                      # in memory
    >>> st = arsc.read_h5ad(f, store="toy_h5ad_store")   # memory-mapped store, converted once
    >>> st.n_cells, st.layers("RNA")
    (500, ['X'])
    """
    h5py = _h5()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file (check the path; relative paths resolve from {Path.cwd()})")
    if store is not None and (Path(store) / "manifest.json").exists() and not overwrite:
        st = _open_store(store)
        if Path(st.manifest.get("source", "")).name == path.name:
            if path.stat().st_mtime > (Path(store) / "manifest.json").stat().st_mtime:
                warnings.warn(f"{store} was built before {path.name} was last modified; returning the existing "
                              "store. Pass overwrite=True to rebuild it from the current file", stacklevel=2)
            return st
        raise FileExistsError(f"{store} holds a different source ({st.manifest.get('source')}); pass overwrite=True")
    read_elem, _ = _anndata_elem_io()
    b = StoreBuilder(store, source=str(path), overwrite=overwrite)
    try:
        with h5py.File(path, "r") as f:
            if "X" not in f and "layers" not in f:
                raise ArscFormatError(f"{path}: no X or layers group; not an AnnData h5ad file")
            obs, var = _read_frames(f, path, read_elem)
            b.set_cells(obs.index, obs)
            b.add_assay(assay, var.index, var=var)
            names = ["X"] + list(f["layers"].keys() if "layers" in f else [])
            want = names if layers is None else list(layers)
            for ln in want:
                if ln == "X":
                    if "X" not in f:
                        continue
                    node = f["X"]
                else:
                    if "layers" not in f or ln not in f["layers"]:
                        raise KeyError(f"layer {ln!r} not in {path.name} (has: {names})")
                    node = f["layers"][ln]
                _copy_matrix(node, b, assay, x_layer if ln == "X" else ln, dtype=dtype, chunk_nnz=chunk_nnz)
            info: dict[str, Any] = {"assay": assay, "x_layer": x_layer if "X" in want and "X" in f else None}
            if raw and "raw" in f and "X" in f["raw"]:
                rvar = _read_df(f["raw"]["var"], read_elem) if "var" in f["raw"] else None
                rfeat = list(map(str, rvar.index)) if rvar is not None else list(map(str, var.index))
                if raw_as_layer:
                    if rfeat != list(map(str, var.index)):
                        raise ArscFormatError("raw has a different gene set from var; use raw_as_layer=False")
                    _copy_matrix(f["raw"]["X"], b, assay, raw, dtype=dtype, chunk_nnz=chunk_nnz)
                    info["raw"] = {"assay": assay, "layer": raw}
                else:
                    b.add_assay(raw, rfeat, var=rvar)
                    b.manifest["assays"][raw]["anndata_raw"] = True
                    _copy_matrix(f["raw"]["X"], b, raw, "X", dtype=dtype, chunk_nnz=chunk_nnz)
                    info["raw"] = {"assay": raw, "layer": "X"}
                    if varm and "varm" in f["raw"]:
                        for k in f["raw"]["varm"]:
                            b.add_varm(raw, k, np.asarray(read_elem(f["raw"]["varm"][k])))
            b.manifest["anndata"] = info
            skipped = []
            if obsm and "obsm" in f:
                used = set()
                for k in f["obsm"]:
                    e = read_elem(f["obsm"][k])
                    frame = isinstance(e, pd.DataFrame)
                    cols = list(map(str, e.columns)) if frame else None
                    arr = e.to_numpy() if frame else (e.toarray() if sp.issparse(e) else np.asarray(e))
                    if arr.ndim != 2 or arr.dtype.kind not in "biuf":
                        skipped.append(f"obsm/{k}")
                        continue
                    nm = k[2:] if k.startswith("X_") and len(k) > 2 else k
                    if nm in used:
                        nm = _safe_name(k)
                    used.add(nm)
                    b.add_reduction(_safe_name(nm), arr, columns=cols, key=k, frame=frame, assay=assay)
            if varm and "varm" in f:
                for k in f["varm"]:
                    e = read_elem(f["varm"][k])
                    arr = e.to_numpy() if isinstance(e, pd.DataFrame) else np.asarray(e)
                    if arr.ndim < 1 or arr.dtype.kind not in "biuf":
                        skipped.append(f"varm/{k}")
                        continue
                    b.add_varm(assay, k, arr)
                    pca = next((n for n, r in b.manifest["reductions"].items() if r["obsm_key"] == "X_pca"), None)
                    if k == "PCs" and pca and arr.ndim == 2 and arr.shape[1] == b.manifest["reductions"][pca]["shape"][1]:
                        _attach_loadings(b, pca, arr, list(map(str, var.index)))
            if obsp and "obsp" in f:
                for k in f["obsp"]:
                    g = read_elem(f["obsp"][k])
                    b.add_graph(_safe_name(k), g, key=k)
            if uns and "uns" in f:
                try:
                    skipped += b.set_uns(read_elem(f["uns"]))
                except Exception as ex:  # noqa: BLE001 - uns is best effort
                    skipped.append(f"uns ({type(ex).__name__}: {ex})")
            if skipped:
                b.manifest["skipped"] = skipped
        return b.finish()
    finally:
        if b.tmp is not None and b.tmp.exists():
            shutil.rmtree(b.tmp, ignore_errors=True)


def _attach_loadings(b: StoreBuilder, red: str, L: np.ndarray, features: list) -> None:
    e = b.manifest["reductions"][red]
    df = pd.DataFrame(np.asarray(L, np.float64), index=features, columns=e["columns"])
    e["loadings_shape"] = list(df.shape)
    if b.tmp is not None:
        rd = b.tmp / "reductions"
        np.save(rd / f"{red}.loadings.npy", df.to_numpy())
        (rd / f"{red}.loadings_features.txt").write_text("\n".join(features), encoding="utf-8")
    r = b._mem_reds[red]
    b._mem_reds[red] = Reduction(r.embedding, r.columns, df, r.stdev, r.assay)


def _read_df(node, read_elem) -> pd.DataFrame:
    df = read_elem(node)
    if not isinstance(df, pd.DataFrame):
        raise ArscFormatError(f"{node.name}: expected a dataframe")
    df.index = df.index.astype(str)
    return df


def _read_frames(f, path: Path, read_elem) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return _read_df(f["obs"], read_elem), _read_df(f["var"], read_elem)
    except Exception:  # noqa: BLE001 - legacy (< 0.7) compound-dataset layout: let anndata parse it
        import anndata as ad
        a = ad.read_h5ad(path, backed="r")
        try:
            obs, var = a.obs.copy(), a.var.copy()
        finally:
            a.file.close()
        obs.index, var.index = obs.index.astype(str), var.index.astype(str)
        return obs, var


# =========================================================================== 10x
def _decode(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind == "S" or a.dtype.kind == "O":
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)
    return a


def _assay_name(feature_type: str) -> str:
    return TENX_ASSAYS.get(feature_type, _safe_name(feature_type).replace(".", "_"))


def read_10x_h5(path: PathLike, store: PathLike | None = None, *, var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
                feature_types: Sequence[str] | None = None, join: str = "-", dtype: DtypeArg = "auto",
                chunk_nnz: int = CHUNK_NNZ, overwrite: bool = False) -> Store | Dataset:
    """Read a CellRanger ``filtered_feature_bc_matrix.h5`` (v3+; v2 single-genome also supported).

    Features are split by ``feature_type`` into separate assays: Gene Expression -> ``RNA``,
    Antibody Capture -> ``ADT``, CRISPR Guide Capture -> ``CRISPR``, Multiplexing Capture -> ``HTO``,
    Peaks -> ``ATAC``. Each assay gets one layer, ``counts``. Names are deduplicated within each assay
    like ``var_names_make_unique`` (``join``). ``var`` keeps ``gene_ids``, ``feature_types`` and the
    other per-feature fields, as ``scanpy.read_10x_h5`` does.

    Parameters
    ----------
    path : the ``.h5`` file.
    store : output store directory; ``None`` returns an in-memory Dataset.
    var_names : ``"gene_symbols"`` or ``"gene_ids"`` as feature names.
    feature_types : restrict the import, e.g. ``["Gene Expression"]``.
    join : separator for deduplicated names.
    dtype : value dtype on disk.
    chunk_nnz : non-zeros per streamed chunk.
    overwrite : replace an existing store.

    Returns
    -------
    arsc.Store (with ``store=``) or arsc.Dataset with assays ``RNA`` (and ``ADT``, ...), layer ``counts``.

    Examples
    --------
    >>> import arsc
    >>> st = arsc.read_10x_h5("filtered_feature_bc_matrix.h5", store="run1")   # doctest: +SKIP
    >>> st.assays                                                             # doctest: +SKIP
    ['RNA', 'ADT']
    """
    h5py = _h5()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    b = StoreBuilder(store, source=str(path), overwrite=overwrite)
    try:
        with h5py.File(path, "r") as f:
            if "matrix" in f:
                g = f["matrix"]
                feat = g["features"]
                fields = {k: _decode(feat[k][:]) for k in feat if isinstance(feat[k], h5py.Dataset) and k != "_all_tag_keys"}
                ftype = fields.get("feature_type", np.array(["Gene Expression"] * int(g["shape"][0]), dtype=object))
                var = pd.DataFrame(index=np.arange(len(ftype)))
                if "gene_id" in fields:
                    var["gene_ids"], var["probe_ids"] = fields["gene_id"], fields["id"]
                else:
                    var["gene_ids"] = fields["id"]
                var["feature_types"] = ftype
                for k, v in fields.items():
                    if k not in ("name", "feature_type", "id", "gene_id"):
                        var[k] = v.astype(bool) if v.dtype.kind == "b" else v
                symbols = fields["name"]
            else:  # v2: one group per genome
                genomes = [k for k in f if isinstance(f[k], h5py.Group) and "barcodes" in f[k]]
                if len(genomes) != 1:
                    raise ArscFormatError(f"{path.name}: not a 10x h5 (found groups {list(f)})")
                g = f[genomes[0]]
                symbols = _decode(g["gene_names"][:])
                var = pd.DataFrame({"gene_ids": _decode(g["genes"][:])})
                var["feature_types"] = "Gene Expression"
                var["genome"] = genomes[0]
                ftype = var["feature_types"].to_numpy()
            n_feat, n_cells = (int(x) for x in g["shape"][:])
            barcodes = _decode(g["barcodes"][:])
            if len(barcodes) != n_cells:
                raise ArscFormatError(f"{path.name}: {len(barcodes)} barcodes for shape {n_feat} x {n_cells}")
            names = np.asarray(symbols if var_names == "gene_symbols" else var["gene_ids"].to_numpy(), dtype=object)
            b.set_cells(barcodes)
            types = list(dict.fromkeys(ftype.tolist()))
            if feature_types is not None:
                missing = set(feature_types) - set(types)
                if missing:
                    raise KeyError(f"feature types {sorted(missing)} not in file (has {types})")
                types = [t for t in types if t in feature_types]
            cols = {t: np.flatnonzero(ftype == t) for t in types}
            for t in types:
                v = var.iloc[cols[t]].reset_index(drop=True)
                b.add_assay(_assay_name(t), make_unique(names[cols[t]], join), var=v)
            sinks = {t: b.layer(_assay_name(t), "counts", src_dtype=g["data"].dtype, dtype=dtype) for t in types}
            single = len(types) == 1 and cols[types[0]].size == n_feat
            # the file stores features x cells CSC == cells x features CSR
            for m in _stream_compressed(g, "csr", (n_cells, n_feat), chunk_nnz):
                for t in types:
                    sinks[t].add(m if single else m[:, cols[t]])
            for s in sinks.values():
                s.close()
        return b.finish()
    finally:
        if b.tmp is not None and b.tmp.exists():
            shutil.rmtree(b.tmp, ignore_errors=True)


def _find(d: Path, names: Sequence[str]) -> Path | None:
    for n in names:
        for cand in (d / n, d / (n + ".gz")):
            if cand.exists():
                return cand
    return None


def _open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")


def read_10x_mtx(path: PathLike, store: PathLike | None = None, *, var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
                 feature_types: Sequence[str] | None = None, join: str = "-", dtype: DtypeArg = "auto",
                 overwrite: bool = False) -> Store | Dataset:
    """Read a 10x Matrix Market directory (``matrix.mtx[.gz]``, ``features.tsv[.gz]`` or ``genes.tsv``,
    ``barcodes.tsv[.gz]``). Feature types are split into assays as in :func:`read_10x_h5`.

    The triplets are parsed in one pass (scipy's fast Matrix Market reader), so peak memory is about
    16 bytes per non-zero. For very large matrices, the 10x ``.h5`` is the memory-bounded route.

    Parameters
    ----------
    path : the directory (or a file inside it).
    store : output store directory; ``None`` returns an in-memory Dataset.
    var_names : ``"gene_symbols"`` or ``"gene_ids"``.
    feature_types : restrict the import.
    join : separator for deduplicated names.
    dtype : value dtype on disk.
    overwrite : replace an existing store.

    Returns
    -------
    arsc.Store (with ``store=``) or arsc.Dataset.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.read_10x_mtx("filtered_feature_bc_matrix/")    # doctest: +SKIP
    """
    from scipy.io import mmread

    d = Path(path)
    if d.is_file():
        d = d.parent
    mtx = _find(d, ["matrix.mtx"])
    bc = _find(d, ["barcodes.tsv"])
    ft = _find(d, ["features.tsv", "genes.tsv"])
    if mtx is None or bc is None or ft is None:
        raise ArscFormatError(f"{d}: expected matrix.mtx, barcodes.tsv and features.tsv/genes.tsv (optionally .gz)")
    with _open_text(bc) as fh:
        barcodes = [ln.rstrip("\n").split("\t")[0] for ln in fh if ln.strip()]
    feats = pd.read_csv(ft, sep="\t", header=None, dtype=str, compression="infer")
    var = pd.DataFrame({"gene_ids": feats[0].to_numpy()})
    symbols = feats[1].to_numpy() if feats.shape[1] > 1 else feats[0].to_numpy()
    var["feature_types"] = feats[2].to_numpy() if feats.shape[1] > 2 else "Gene Expression"
    opener = gzip.open if mtx.suffix == ".gz" else open
    with opener(mtx, "rb") as fh:
        M = mmread(fh)
    M = sp.csr_matrix(M.T)  # features x cells -> cells x features
    if M.shape != (len(barcodes), len(var)):
        raise ArscFormatError(f"{d}: matrix {M.shape[::-1]} vs {len(var)} features x {len(barcodes)} barcodes")
    b = StoreBuilder(store, source=str(d), overwrite=overwrite)
    try:
        b.set_cells(barcodes)
        ftype = var["feature_types"].to_numpy()
        types = list(dict.fromkeys(ftype.tolist()))
        if feature_types is not None:
            types = [t for t in types if t in feature_types]
        names = np.asarray(symbols if var_names == "gene_symbols" else var["gene_ids"].to_numpy(), dtype=object)
        for t in types:
            cols = np.flatnonzero(ftype == t)
            b.add_assay(_assay_name(t), make_unique(names[cols], join), var=var.iloc[cols].reset_index(drop=True))
            b.add_layer(_assay_name(t), "counts", M[:, cols] if cols.size != M.shape[1] else M, dtype=dtype)
        return b.finish()
    finally:
        if b.tmp is not None and b.tmp.exists():
            shutil.rmtree(b.tmp, ignore_errors=True)


# =========================================================================== Seurat
def read_seurat(path: PathLike, store: PathLike, **export_kw) -> Store:
    """Convert a Seurat v4/v5 object (``.qs`` preferred, ``.rds`` accepted) with the bundled R exporter.

    Needs R with SeuratObject (and ``qs`` for ``.qs``). Writes ``store`` once; later calls open it.
    ``export_kw`` are passed to :func:`arsc.export_seurat` (``assays``, ``layers``, ``graphs``, ...).
    For ``.h5Seurat`` files, see :func:`read` (convert to h5ad with SeuratDisk, or save as qs in R).

    Parameters
    ----------
    path : ``.qs`` or ``.rds`` file with a Seurat object.
    store : output store directory (required).

    Returns
    -------
    arsc.Store.

    Examples
    --------
    >>> import arsc
    >>> st = arsc.read_seurat("obj.qs", store="stores/obj")      # doctest: +SKIP
    """
    from .io import export_seurat

    store = Path(store)
    if (store / "manifest.json").exists():
        return _open_store(store)
    tmp = store.with_name(store.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        export_seurat(path, tmp, **export_kw)
    except FileNotFoundError as e:
        raise RuntimeError("Rscript not found: reading Seurat objects needs R (SeuratObject, qs); "
                           "pass rscript='<path to Rscript>'") from e
    os.replace(tmp, store)
    return _open_store(store)


# =========================================================================== dispatch
def detect_format(path: PathLike) -> str:
    """Detect the format of a path from its name (and HDF5 layout for ``.h5``).

    Parameters
    ----------
    path : file or directory.

    Returns
    -------
    str, one of ``store, h5ad, 10x_h5, 10x_mtx, seurat, h5seurat, loom``; raises
    :class:`ArscFormatError` otherwise.

    Examples
    --------
    >>> import arsc
    >>> arsc.detect_format("obj.qs"), arsc.detect_format("pbmc.h5ad")
    ('seurat', 'h5ad')
    """
    p = Path(path)
    if p.is_dir():
        if (p / "manifest.json").exists():
            return "store"
        if _find(p, ["matrix.mtx"]) is not None:
            return "10x_mtx"
        raise ArscFormatError(f"{p}: directory is neither an arsc store (manifest.json) nor a 10x mtx directory")
    name = p.name.lower()
    if name.endswith(".h5ad"):
        return "h5ad"
    if name.endswith((".mtx", ".mtx.gz")):
        return "10x_mtx"
    if name.endswith((".qs", ".rds")):
        return "seurat"
    if name.endswith(".h5seurat"):
        return "h5seurat"
    if name.endswith(".loom"):
        return "loom"
    if name.endswith((".h5", ".hdf5")):
        h5py = _h5()
        with h5py.File(p, "r") as f:
            if "matrix" in f and "barcodes" in f["matrix"]:
                return "10x_h5"
            if any(isinstance(f[k], h5py.Group) and "barcodes" in f[k] for k in f):
                return "10x_h5"
            if "X" in f and "obs" in f:
                return "h5ad"
        raise ArscFormatError(f"{p.name}: HDF5 file is neither 10x nor AnnData")
    raise ArscFormatError(f"{p.name}: unknown format (supported: arsc store dir, .h5ad, 10x .h5, 10x mtx dir, "
                          ".qs/.rds Seurat)")


def read(path: PathLike, store: PathLike | None = None, **kw) -> Store | Dataset:
    """Read any supported single-cell file.

    Returns an in-memory :class:`~arsc.io.Dataset`, or, with ``store=<dir>``, converts once into a
    memory-mapped :class:`~arsc.io.Store` and opens it. Arsc store directories always open as a Store.

    =========================  ===============================  =========================
    input                      reader                           needs
    =========================  ===============================  =========================
    arsc store directory       :func:`arsc.open`                --
    ``.h5ad``                  :func:`read_h5ad`                anndata, h5py
    10x ``.h5``                :func:`read_10x_h5`              h5py
    10x mtx directory          :func:`read_10x_mtx`             --
    ``.qs`` / ``.rds``         :func:`read_seurat`              R + SeuratObject (store=)
    =========================  ===============================  =========================

    ``.h5Seurat`` is not parsed natively. In R, ``SeuratDisk::Convert("x.h5Seurat", dest = "h5ad")``
    gives an h5ad, or ``qs::qsave(LoadH5Seurat("x.h5Seurat"), "x.qs")`` gives a qs for the exporter.

    Parameters
    ----------
    path : file or directory (any format above).
    store : output store directory for a conversion (required for Seurat objects).

    Returns
    -------
    arsc.Dataset (in memory) or arsc.Store (memory-mapped).

    Examples
    --------
    >>> import arsc
    >>> f = arsc.datasets.toy_h5ad("toy_b.h5ad", n_cells=500)
    >>> ds = arsc.read(f)
    >>> ds.n_cells
    500
    >>> st = arsc.read(f, store="toy_store2")
    >>> arsc.read("toy_store2").n_cells      # a store directory reopens as a Store
    500
    """
    fmt = detect_format(path)
    if fmt == "store":
        if store is not None:
            raise ValueError("path is already an arsc store; open it with arsc.open(path) (store= is for conversions)")
        return _open_store(path, **kw)
    if fmt == "h5ad":
        return read_h5ad(path, store, **kw)
    if fmt == "10x_h5":
        return read_10x_h5(path, store, **kw)
    if fmt == "10x_mtx":
        return read_10x_mtx(path, store, **kw)
    if fmt == "seurat":
        if store is None:
            raise ValueError("Seurat objects are converted through R into a store: pass store='<dir>'")
        return read_seurat(path, store, **kw)
    if fmt == "h5seurat":
        raise ArscFormatError(
            f"{Path(path).name}: h5Seurat is not read natively. In R either "
            "SeuratDisk::Convert('<file>.h5Seurat', dest = 'h5ad') then arsc.read('<file>.h5ad'), or "
            "qs::qsave(SeuratDisk::LoadH5Seurat('<file>.h5Seurat'), '<file>.qs') then arsc.read('<file>.qs', store=...)")
    raise ArscFormatError(f"{Path(path).name}: loom is not supported yet; convert with "
                          "anndata.read_loom(...).write_h5ad(...) and read the h5ad")


# =========================================================================== export: shared plan
class _Source:
    """Uniform read access to a Store or a Dataset for export."""

    def __init__(self, obj: Store | Dataset):
        if not isinstance(obj, (Store, Dataset)):
            raise TypeError(f"expected an arsc Store or Dataset, got {type(obj).__name__}")
        self.obj, self.is_store = obj, isinstance(obj, Store)
        self.manifest = obj.manifest or {}

    @property
    def obs(self) -> pd.DataFrame:
        return self.obj.obs

    def assays(self) -> list:
        return list(self.obj.assays) if self.is_store else list(self.obj.assays.keys())

    def layers(self, a: str) -> list:
        return self.obj.layers(a) if self.is_store else list(self.obj.assays[a].layers)

    def X(self, a: str, l: str) -> sp.csr_matrix:
        return self.obj.X(a, l)

    def features(self, a: str) -> pd.Index:
        return self.obj.features(a) if self.is_store else self.obj.assays[a].features

    def var(self, a: str) -> pd.DataFrame:
        return self.obj.var(a) if self.is_store else self.obj.assays[a].var_frame()

    def variable_features(self, a: str) -> list:
        return self.obj.variable_features(a) if self.is_store else list(self.obj.assays[a].variable_features)

    def varm(self, a: str) -> dict:
        if self.is_store:
            return {k: self.obj.varm(a, k) for k in (self.manifest["assays"][a].get("varm") or {})}
        return dict(self.obj.assays[a].varm or {})

    def layer_info(self, a: str, l: str) -> dict:
        return ((self.manifest.get("assays") or {}).get(a, {}).get("layers") or {}).get(l, {})

    def reductions(self) -> dict:
        if self.is_store:
            return {r: self.obj.embedding(r) for r in self.obj.reductions}
        return {k: r.embedding for k, r in self.obj.reductions.items()}

    def reduction_info(self, r: str) -> dict:
        return (self.manifest.get("reductions") or {}).get(r, {})

    def reduction_columns(self, r: str) -> list:
        return self.obj.reduction(r).columns if self.is_store else self.obj.reductions[r].columns

    def graphs(self) -> list:
        return list(self.obj.graphs) if self.is_store else list(self.obj.graphs.keys())

    def graph(self, g: str):
        return self.obj.graph(g) if self.is_store else self.obj.graphs[g]

    def uns(self) -> dict:
        return self.obj.uns if self.obj.uns is not None else {}

    def default_assay(self) -> str:
        a = (self.manifest.get("anndata") or {}).get("assay") or self.manifest.get("default_assay")
        assays = self.assays()
        return a if a in assays else ("RNA" if "RNA" in assays else assays[0])


def _restore_dtype(m: sp.csr_matrix, info: dict, restore: bool) -> sp.csr_matrix:
    src = info.get("source_dtype")
    if restore and src and np.dtype(src) != m.dtype:
        m = m.astype(np.dtype(src))
    return m


def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame writable by anndata: string extension columns -> object/categorical."""
    df = df.copy()
    df.index = df.index.astype(str)
    df.index.name = None
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.StringDtype) or (s.dtype == object and s.isna().any()):
            df[c] = s.astype("category") if s.isna().any() else s.astype(object)
    return df


def _plan(src: _Source, cells, genes, assay, x_layer, layers, raw):
    assay = assay or src.default_assay()
    if assay not in src.assays():
        raise KeyError(f"assay {assay!r} not found (has {src.assays()})")
    info = src.manifest.get("anndata") or {}
    lays = src.layers(assay)
    if x_layer is None:
        x_layer = info.get("x_layer") if info.get("assay") == assay and info.get("x_layer") in lays else None
        x_layer = x_layer or ("counts" if "counts" in lays else lays[0])
    if x_layer not in lays:
        raise KeyError(f"layer {x_layer!r} not in assay {assay!r} (has {lays})")
    raw_spec = info.get("raw") if raw else None
    if raw_spec and raw_spec.get("assay") == assay:
        others_excl = {raw_spec["layer"]}
    else:
        others_excl = set()
    if layers is None:
        layers = [l for l in lays if l != x_layer and l not in others_excl]
    else:
        bad = [l for l in layers if l not in lays]
        if bad:
            raise KeyError(f"layers {bad} not in assay {assay!r} (has {lays})")
        layers = [l for l in layers if l != x_layer]
    n = len(src.obs)
    cidx = None if cells is None else _as_positions(cells, src.obs.index)
    feats = src.features(assay)
    gidx = None if genes is None else _as_positions(genes, feats)
    if raw_spec and raw_spec.get("assay") not in src.assays():
        raw_spec = None
    return assay, x_layer, list(layers), raw_spec, cidx, gidx, n


def _var_frame(src: _Source, assay: str, gidx) -> pd.DataFrame:
    var = src.var(assay)
    vf = src.variable_features(assay)
    if vf and "highly_variable" not in var.columns:
        var = var.copy()
        var["highly_variable"] = var.index.isin(vf)
    if gidx is not None:
        var = var.iloc[gidx]
    return _sanitize_frame(var)


def _obsm(src: _Source, cidx, assay: str, obs_names: pd.Index) -> dict:
    out = {}
    for name, emb in src.reductions().items():
        e = src.reduction_info(name)  # reductions are cell-level: exported whatever their assay
        arr = np.asarray(emb if cidx is None else emb[cidx])
        if e.get("source_dtype"):
            arr = arr.astype(np.dtype(e["source_dtype"]))
        key = e.get("obsm_key") or f"X_{name}"
        if e.get("frame"):
            out[key] = pd.DataFrame(arr, index=obs_names, columns=src.reduction_columns(name))
        else:
            out[key] = np.ascontiguousarray(arr)
    return out


def _subset_graph(g, cidx):
    g = sp.csr_matrix(g)
    if cidx is None:
        return g
    return _take_rows(g, cidx)[:, cidx].tocsr()


def _rows(M: sp.csr_matrix, cidx, gidx, s: int, e: int) -> sp.csr_matrix:
    rows = np.arange(s, e, dtype=np.int64) if cidx is None else cidx[s:e]
    m = _take_rows(M, rows)
    if gidx is not None:
        m = m[:, gidx]
    m.sort_indices()
    return m


# =========================================================================== export: AnnData in memory
def to_anndata(obj: Store | Dataset, cells=None, genes=None, *, assay: str | None = None, x_layer: str | None = None,
               layers: Sequence[str] | None = None, raw: bool = True, obsm: bool = True, varm: bool = True,
               obsp: bool = False, uns: bool = True, restore_dtype: bool = True):
    """Build an in-memory :class:`anndata.AnnData` from a store or dataset (scanpy hand-off).

    ``cells`` / ``genes``: boolean mask, positions or names (``None`` = all). Only the selected rows
    are read from the memory map. ``X`` is ``x_layer``: the original ``.X`` for h5ad-derived stores,
    otherwise ``counts`` when present. The other layers of ``assay`` go to ``.layers``. ``raw``
    restores ``.raw`` when the store came from an h5ad that had one. ``obsp=True`` also exports the
    graphs. Values are cast back to the source dtype recorded at import (``restore_dtype``).

    Parameters
    ----------
    obj : arsc.Store or arsc.Dataset.
    cells : cells to export (mask, positions or barcodes).
    genes : genes to export (mask, positions or names).
    assay : assay for ``X`` (default: the store's default assay).
    x_layer : layer used as ``X``.
    layers : other layers to export (default all).
    raw : restore ``.raw``.
    obsm, varm, obsp, uns : export these slots.
    restore_dtype : cast to the source dtype.

    Returns
    -------
    anndata.AnnData.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> ad = arsc.to_anndata(ds, cells=(ds.obs.cell_type == "CD8 T").to_numpy(), genes=["CD8A", "GZMK"], assay="RNA")
    >>> ad.n_vars, list(ad.var_names)
    (2, ['CD8A', 'GZMK'])
    """
    import anndata as ad

    src = _Source(obj)
    assay, x_layer, layers, raw_spec, cidx, gidx, n = _plan(src, cells, genes, assay, x_layer, layers, raw)
    n_sel = n if cidx is None else cidx.size

    def full(a, l, g):
        M = src.X(a, l)
        m = _rows(M, cidx, g, 0, n_sel) if (cidx is not None or g is not None) else sp.csr_matrix(M, copy=True)
        return _restore_dtype(m, src.layer_info(a, l), restore_dtype)

    obs = src.obs if cidx is None else src.obs.iloc[cidx]
    obs = _sanitize_frame(obs)
    var = _var_frame(src, assay, gidx)
    a = ad.AnnData(X=full(assay, x_layer, gidx), obs=obs, var=var)
    for l in layers:
        a.layers[l] = full(assay, l, gidx)
    if obsm:
        for k, v in _obsm(src, cidx, assay, a.obs_names).items():
            a.obsm[k] = v
    if varm:
        for k, v in src.varm(assay).items():
            a.varm[k] = np.asarray(v if gidx is None else np.asarray(v)[gidx])
    if obsp:
        for g in src.graphs():
            key = (src.manifest.get("graphs") or {}).get(g, {}).get("obsp_key", g)
            a.obsp[key] = _subset_graph(src.graph(g), cidx)
    if uns:
        a.uns.update(src.uns())
    if raw_spec:
        ra, rl = raw_spec["assay"], raw_spec["layer"]
        rvar = _var_frame(src, ra, None) if ra != assay else var
        rX = full(ra, rl, None if ra != assay else gidx)
        r = ad.AnnData(X=rX, obs=pd.DataFrame(index=a.obs_names), var=rvar)
        if ra != assay:
            for k, v in src.varm(ra).items():
                r.varm[k] = np.asarray(v)
        a.raw = r
    return a


# =========================================================================== export: streamed h5ad
def _write_csr_stream(group_parent, name: str, M: sp.csr_matrix, cidx, gidx, n_sel: int, n_cols: int,
                      info: dict, restore: bool, chunk_nnz: int, compression) -> None:
    g = group_parent.create_group(name)
    g.attrs["encoding-type"] = "csr_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = np.array([n_sel, n_cols], dtype=np.int64)
    src_dt = np.dtype(info["source_dtype"]) if (restore and info.get("source_dtype")) else M.dtype
    kw = {"compression": compression} if compression else {}
    indptr = np.asarray(M.indptr)
    rows_all = np.arange(M.shape[0]) if cidx is None else cidx
    lens = indptr[rows_all + 1] - indptr[rows_all]
    cum = np.concatenate([[0], np.cumsum(lens)])
    # chunk no larger than the matrix: a fixed 256k-element chunk wrote 1-2 MB per tiny test matrix
    ck = (int(max(1, min(1 << 18, int(cum[-1])))),)
    dd = g.create_dataset("data", shape=(0,), maxshape=(None,), dtype=src_dt, chunks=ck, **kw)
    # one index dtype for indices and indptr (scipy refuses mixed int32/int64 in in-place operations, which
    # breaks e.g. scanpy on the written file); int32 unless the column count or the nnz needs int64
    idx_dt = np.int32 if (n_cols < 2 ** 31 and int(cum[-1]) < 2 ** 31) else np.int64
    di = g.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=idx_dt, chunks=ck, **kw)
    ptr = np.zeros(n_sel + 1, dtype=np.int64)
    s, nnz = 0, 0
    while s < n_sel:
        e = int(np.searchsorted(cum, cum[s] + chunk_nnz, side="right")) - 1
        e = min(max(e, s + 1), n_sel)
        m = _rows(M, cidx, gidx, s, e)
        k = m.nnz
        dd.resize((nnz + k,))
        di.resize((nnz + k,))
        dd[nnz:nnz + k] = m.data.astype(src_dt, copy=False)
        di[nnz:nnz + k] = m.indices
        ptr[s + 1:e + 1] = np.asarray(m.indptr[1:], dtype=np.int64) + nnz
        nnz += k
        s = e
    g.create_dataset("indptr", data=ptr.astype(idx_dt))


def write_h5ad(obj: Store | Dataset, path: PathLike, cells=None, genes=None, *, assay: str | None = None,
               x_layer: str | None = None, layers: Sequence[str] | None = None, raw: bool = True,
               obsm: bool = True, varm: bool = True, obsp: bool = False, uns: bool = True,
               restore_dtype: bool = True, compression: str | None = None, chunk_nnz: int = CHUNK_NNZ,
               overwrite: bool = False) -> Path:
    """Write a store, dataset or subset to ``.h5ad`` without materializing the full matrix.

    The expression matrices are streamed in row chunks (CSR on disk), so a store larger than RAM
    exports in bounded memory. Arguments are as in :func:`to_anndata`. ``compression`` is passed to
    h5py (``"gzip"``, ``"lzf"``, or ``None``, which is uncompressed and fastest). The file is
    written to ``<path>.tmp`` and renamed. The output reads back with ``anndata.read_h5ad`` and
    ``scanpy.read_h5ad``.

    Parameters
    ----------
    obj : arsc.Store or arsc.Dataset.
    path : output ``.h5ad``.
    cells, genes, assay, x_layer, layers, raw, obsm, varm, obsp, uns, restore_dtype : as in :func:`to_anndata`.
    compression : h5py compression.
    chunk_nnz : non-zeros per streamed chunk.
    overwrite : replace an existing file.

    Returns
    -------
    pathlib.Path of the written file.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> p = arsc.write_h5ad(ds, "cd8.h5ad", cells=(ds.obs.cell_type == "CD8 T").to_numpy(), assay="RNA")
    >>> p.name
    'cd8.h5ad'
    """
    h5py = _h5()
    _, write_elem = _anndata_elem_io()
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True)")
    src = _Source(obj)
    assay, x_layer, layers, raw_spec, cidx, gidx, n = _plan(src, cells, genes, assay, x_layer, layers, raw)
    n_sel = n if cidx is None else cidx.size
    feats = src.features(assay)
    n_cols = len(feats) if gidx is None else gidx.size
    obs = _sanitize_frame(src.obs if cidx is None else src.obs.iloc[cidx])
    var = _var_frame(src, assay, gidx)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
            _write_csr_stream(f, "X", src.X(assay, x_layer), cidx, gidx, n_sel, n_cols,
                              src.layer_info(assay, x_layer), restore_dtype, chunk_nnz, compression)
            write_elem(f, "layers", {})
            for l in layers:
                _write_csr_stream(f["layers"], l, src.X(assay, l), cidx, gidx, n_sel, n_cols,
                                  src.layer_info(assay, l), restore_dtype, chunk_nnz, compression)
            write_elem(f, "obsm", _obsm(src, cidx, assay, obs.index) if obsm else {})
            vm = {}
            if varm:
                vm = {k: np.asarray(v if gidx is None else np.asarray(v)[gidx]) for k, v in src.varm(assay).items()}
            write_elem(f, "varm", vm)
            op = {}
            if obsp:
                for g in src.graphs():
                    key = (src.manifest.get("graphs") or {}).get(g, {}).get("obsp_key", g)
                    op[key] = _subset_graph(src.graph(g), cidx)
            write_elem(f, "obsp", op)
            write_elem(f, "varp", {})
            write_elem(f, "uns", src.uns() if uns else {})
            if raw_spec:
                ra, rl = raw_spec["assay"], raw_spec["layer"]
                rg = f.create_group("raw")
                rg.attrs["encoding-type"] = "raw"
                rg.attrs["encoding-version"] = "0.1.0"
                same = ra == assay
                rgidx = gidx if same else None
                rn = (len(src.features(ra)) if rgidx is None else rgidx.size)
                _write_csr_stream(rg, "X", src.X(ra, rl), cidx, rgidx, n_sel, rn, src.layer_info(ra, rl),
                                  restore_dtype, chunk_nnz, compression)
                write_elem(rg, "var", var if same else _var_frame(src, ra, None))
                write_elem(rg, "varm", {} if same else {k: np.asarray(v) for k, v in src.varm(ra).items()})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
