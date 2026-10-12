"""I/O bridge between Seurat (R) and arsc.

The R side (``arsc/R/export_seurat.R``) writes a directory whose binary files
are the verbatim ``@i/@p/@x`` slots of each dgCMatrix. A genes x cells CSC
matrix *is* a cells x genes CSR matrix, so loading is a few ``np.fromfile``
calls with no parsing or transposition. 260k cells x 28k genes (367M nnz)
loads in a few seconds.

Layout (see the R script header for details)::

    manifest.json  cells.txt  obs.csv  [obs.parquet, cached on first read]
    assays/<A>/features.txt  assays/<A>/<layer>.{i,p,x}.bin  [variable_features.txt]
    reductions/<R>.f64  [<R>.loadings.f64, <R>.loadings_features.txt]
    graphs/<G>.{i,p,x}.bin
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "Assay",
    "Reduction",
    "Dataset",
    "read_seurat_export",
    "export_seurat",
    "exporter_script",
    "write_table",
]

_XDTYPES = {"int32": np.int32, "float64": np.float64}


def exporter_script() -> Path:
    """Path of the bundled R exporter (``Rscript <path> <obj.rds|.qs> <out_dir>``)."""
    return Path(__file__).with_name("R") / "export_seurat.R"


def export_seurat(obj_path: str | os.PathLike, out_dir: str | os.PathLike, *, assays=None,
                  layers=("counts",), graphs: bool = True, reductions: bool = True,
                  rscript: str = "Rscript", threads: int = 16) -> Path:
    """Run the R exporter on a ``.rds`` / ``.qs`` Seurat object (v4 or v5).

    Thin subprocess wrapper; equivalent to calling the R script by hand.
    Returns ``out_dir``.
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
    """One modality. Every layer is a cells x features ``scipy.sparse.csr_matrix``."""

    name: str
    features: pd.Index
    layers: dict[str, sp.csr_matrix] = field(default_factory=dict)
    variable_features: list[str] = field(default_factory=list)

    def __getitem__(self, layer: str) -> sp.csr_matrix:
        return self.layers[layer]

    @property
    def shape(self) -> tuple[int, int]:
        return (next(iter(self.layers.values())).shape[0] if self.layers else 0, len(self.features))

    def __repr__(self) -> str:
        ls = ", ".join(f"{k} (nnz={v.nnz:,})" for k, v in self.layers.items())
        return f"Assay({self.name!r}, {len(self.features):,} features, layers: {ls or 'none'})"


@dataclass
class Reduction:
    """Cell embedding (cells x k), optional feature loadings and stdev."""

    embedding: np.ndarray
    columns: list[str]
    loadings: pd.DataFrame | None = None
    stdev: np.ndarray | None = None
    assay: str | None = None

    def to_frame(self, index=None) -> pd.DataFrame:
        return pd.DataFrame(self.embedding, index=index, columns=self.columns)

    def __repr__(self) -> str:
        return f"Reduction({self.embedding.shape[0]:,} x {self.embedding.shape[1]})"


@dataclass
class Dataset:
    """Light container returned by :func:`read_seurat_export`.

    Attributes
    ----------
    obs : DataFrame indexed by cell barcode (R factors -> ordered-level ``category``).
    assays : ``{name: Assay}``; ``ds["RNA"]`` is shorthand for ``ds.assays["RNA"]``.
    reductions : ``{name: Reduction}``.
    graphs : ``{name: csr_matrix}`` (cells x cells, e.g. Seurat's ``wsnn``).
    """

    obs: pd.DataFrame
    assays: dict[str, Assay] = field(default_factory=dict)
    reductions: dict[str, Reduction] = field(default_factory=dict)
    graphs: dict[str, sp.csr_matrix] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    # -- access helpers -------------------------------------------------------
    def __getitem__(self, assay: str) -> Assay:
        return self.assays[assay]

    @property
    def n_cells(self) -> int:
        return len(self.obs)

    @property
    def cells(self) -> pd.Index:
        return self.obs.index

    def X(self, assay: str = "RNA", layer: str = "counts") -> sp.csr_matrix:
        """Cells x features matrix for ``assay``/``layer``."""
        return self.assays[assay].layers[layer]

    def embedding(self, name: str, dims=None) -> np.ndarray:
        """Embedding array; ``dims`` is a 0-based slice/index list (``range(30)``)."""
        e = self.reductions[name].embedding
        return e if dims is None else e[:, list(dims)]

    def subset(self, cells) -> "Dataset":
        """Row-subset by boolean mask, integer positions or barcodes. Graphs are subset too."""
        idx = _as_positions(cells, self.obs.index)
        assays = {k: Assay(a.name, a.features, {l: m[idx] for l, m in a.layers.items()},
                           list(a.variable_features)) for k, a in self.assays.items()}
        reds = {k: Reduction(r.embedding[idx], r.columns, r.loadings, r.stdev, r.assay)
                for k, r in self.reductions.items()}
        graphs = {k: g[idx][:, idx].tocsr() for k, g in self.graphs.items()}
        return Dataset(self.obs.iloc[idx].copy(), assays, reds, graphs, dict(self.manifest))

    def __repr__(self) -> str:
        lines = [f"arsc.Dataset: {self.n_cells:,} cells"]
        for a in self.assays.values():
            lines.append(f"  {a!r}")
        if self.reductions:
            lines.append("  reductions: " + ", ".join(
                f"{k} ({v.embedding.shape[1]})" for k, v in self.reductions.items()))
        if self.graphs:
            lines.append("  graphs: " + ", ".join(self.graphs))
        lines.append(f"  obs: {self.obs.shape[1]} columns")
        return "\n".join(lines)


def _as_positions(cells, index: pd.Index) -> np.ndarray:
    cells = np.asarray(cells)
    if cells.dtype == bool:
        return np.flatnonzero(cells)
    if np.issubdtype(cells.dtype, np.integer):
        return cells
    pos = index.get_indexer(cells)
    if (pos < 0).any():
        raise KeyError(f"{(pos < 0).sum()} barcodes not found")
    return pos


# --------------------------------------------------------------------------- readers
def _read_sparse(prefix: Path, info: dict, n_rows: int, dtype) -> sp.csr_matrix:
    """Load R dgCMatrix slots (n_features x n_cells CSC) as an n_cells x n_features CSR."""
    n_feat, n_cols = info["shape"]
    indices = np.fromfile(f"{prefix}.i.bin", dtype="<i4")
    indptr = np.fromfile(f"{prefix}.p.bin", dtype="<i4")
    data = np.fromfile(f"{prefix}.x.bin", dtype=_XDTYPES[info["xdtype"]])
    if dtype is not None and data.dtype != dtype:
        data = data.astype(dtype)
    m = sp.csr_matrix((data, indices, indptr), shape=(n_cols, n_feat), copy=False)
    m.has_sorted_indices = True  # dgCMatrix row indices are sorted by construction
    return m


def _read_obs(d: Path, manifest: dict, cells: list[str], cache: bool) -> pd.DataFrame:
    pq = d / "obs.parquet"
    if pq.exists() and pq.stat().st_mtime >= (d / "obs.csv").stat().st_mtime:
        obs = pd.read_parquet(pq)
        obs.index.name = None
        return obs
    import pyarrow as pa
    import pyarrow.csv as pacsv

    spec = manifest.get("obs", {}) or {}
    types = {"cell": pa.string()}
    for col, s in spec.items():
        t = s["type"]
        types[col] = {"double": pa.float64(), "integer": pa.int64(), "logical": pa.bool_()}.get(t, pa.string())
    tbl = pacsv.read_csv(
        d / "obs.csv",
        convert_options=pacsv.ConvertOptions(
            column_types=types, null_values=["NA"], strings_can_be_null=True,
            quoted_strings_can_be_null=False, true_values=["TRUE"], false_values=["FALSE"]),
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
    if not obs.index.equals(pd.Index(cells)):
        raise ValueError("obs.csv row order does not match cells.txt")
    if cache:
        try:
            obs.to_parquet(pq)
        except OSError:
            pass
    return obs


def read_seurat_export(path: str | os.PathLike, *, assays=None, layers=None, graphs: bool = True,
                       reductions: bool = True, count_dtype=np.float32, cache_obs: bool = True) -> Dataset:
    """Load a directory written by ``export_seurat.R``.

    Parameters
    ----------
    path : export directory.
    assays, layers : optional lists restricting what is loaded (default: everything).
    count_dtype : dtype for integer-valued layers (raw counts). ``float32`` is exact
        for counts < 2**24 and halves memory; pass ``np.float64`` or ``None``
        (keep int32) if needed. Non-integer layers are always float64.
    cache_obs : write ``obs.parquet`` next to ``obs.csv`` on first read.

    Returns
    -------
    Dataset
    """
    d = Path(path)
    manifest = json.loads((d / "manifest.json").read_text())
    cells = (d / "cells.txt").read_text(encoding="utf-8").splitlines()
    n = len(cells)
    obs = _read_obs(d, manifest, cells, cache_obs)

    out_assays: dict[str, Assay] = {}
    for a, entry in (manifest.get("assays") or {}).items():
        if assays is not None and a not in assays:
            continue
        ad = d / "assays" / a
        feats = pd.Index((ad / "features.txt").read_text(encoding="utf-8").splitlines())
        vf_path = ad / "variable_features.txt"
        vf = vf_path.read_text(encoding="utf-8").splitlines() if vf_path.exists() else []
        lay = {}
        for lname, info in (entry.get("layers") or {}).items():
            if layers is not None and lname not in layers:
                continue
            dt = count_dtype if info["xdtype"] == "int32" else None
            lay[lname] = _read_sparse(ad / info["prefix"], info, n, dt)
        out_assays[a] = Assay(a, feats, lay, vf)

    out_reds: dict[str, Reduction] = {}
    if reductions:
        for r, e in (manifest.get("reductions") or {}).items():
            rd = d / "reductions"
            nr, k = e["shape"]
            emb = np.fromfile(rd / f"{r}.f64", dtype="<f8").reshape(k, nr).T.copy()
            loadings = None
            if "loadings_shape" in e:
                fl, kl = e["loadings_shape"]
                L = np.fromfile(rd / f"{r}.loadings.f64", dtype="<f8").reshape(kl, fl).T
                lf = (rd / f"{r}.loadings_features.txt").read_text(encoding="utf-8").splitlines()
                loadings = pd.DataFrame(L, index=lf, columns=e["columns"])
            stdev = np.asarray(e["stdev"]) if e.get("stdev") is not None else None
            out_reds[r] = Reduction(emb, list(e["columns"]), loadings, stdev, e.get("assay"))

    out_graphs: dict[str, sp.csr_matrix] = {}
    if graphs:
        for g, info in (manifest.get("graphs") or {}).items():
            # Graphs are cells x cells; the CSC->CSR reinterpretation transposes,
            # which is harmless for symmetric graphs and transposed back otherwise.
            m = _read_sparse(d / "graphs" / info["prefix"], info, n, np.float64)
            out_graphs[g] = m.T.tocsr()
    return Dataset(obs, out_assays, out_reds, out_graphs, manifest)


# --------------------------------------------------------------------------- tidy outputs
def write_table(df: pd.DataFrame, path: str | os.PathLike, index: bool | None = None) -> Path:
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
