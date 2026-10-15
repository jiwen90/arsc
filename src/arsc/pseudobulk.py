"""Pseudobulk: per-group sparse sums (Seurat ``AggregateExpression`` / muscat-style).

Typical use, one profile per sample x cell type with a 10-cell floor::

    pb = arsc.pseudobulk(ds.X("RNA", "counts"), ds.obs, by=["sample", "cell_type"],
                         features=ds["RNA"].features, min_cells=10)
    pb.write("pb/")          # counts.parquet (genes x groups) + groups.parquet

The counts matrix is the exact integer sum, ready for DESeq2/edgeR in R.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from ._utils import as_csr, factorize_groups

__all__ = ["Pseudobulk", "pseudobulk"]


@nb.njit(parallel=True, cache=True)
def _group_sums(indptr, indices, data, order, starts, n_features, expm1):
    G = starts.size - 1
    out = np.zeros((G, n_features), dtype=np.float64)
    nnz = np.zeros((G, n_features), dtype=np.int64)
    for g in nb.prange(G):
        for t in range(starts[g], starts[g + 1]):
            c = order[t]
            for p in range(indptr[c], indptr[c + 1]):
                v = data[p]
                if expm1:
                    v = np.expm1(v)
                out[g, indices[p]] += v
                if v != 0:
                    nnz[g, indices[p]] += 1
    return out, nnz


@dataclass
class Pseudobulk:
    """Result of :func:`pseudobulk`: one aggregated profile per group (e.g. per patient x timepoint).

    Attributes
    ----------
    X : (n_groups, n_features) float64 array of summed (or averaged) values.
    groups : DataFrame, one row per kept group: grouping columns, ``n_cells``.
        Index = combined label (``"sampleA_CD8 TEM"``, Seurat's naming).
    features : feature names.
    n_detected : (n_groups, n_features) number of cells with a nonzero value.
    dropped : groups removed by the ``min_cells`` floor (same columns as ``groups``).

    Note: the grouping columns of ``groups`` are plain strings (object dtype). For DE, set the
    reference level explicitly, e.g. ``cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> pb = arsc.pseudobulk(ds.X("RNA", "counts"), ds.obs, ["patient", "timepoint"],
    ...                      features=ds.features("RNA"), min_cells=10)
    >>> pb.X.shape
    (12, 300)
    >>> list(pb.groups.columns)
    ['patient', 'timepoint', 'n_cells']
    >>> pb.to_frame().shape                  # genes x samples, the DESeq2 orientation
    (300, 12)
    """

    X: np.ndarray
    groups: pd.DataFrame
    features: pd.Index
    n_detected: np.ndarray
    dropped: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        """Wide features x groups table (the orientation DESeq2 expects).

        Examples
        --------
        >>> import arsc
        >>> ds = arsc.datasets.toy()
        >>> pb = arsc.pseudobulk(ds.X("RNA", "counts"), ds.obs, "sample", features=ds.features("RNA"))
        >>> pb.to_frame().loc["CD8A"].shape
        (12,)
        """
        return pd.DataFrame(self.X.T, index=self.features, columns=self.groups.index)

    def to_long(self, min_value: float = 0.0) -> pd.DataFrame:
        """Tidy (group, feature, value, n_detected) rows with value > ``min_value``.

        Examples
        --------
        >>> import arsc
        >>> ds = arsc.datasets.toy()
        >>> pb = arsc.pseudobulk(ds.X("RNA", "counts"), ds.obs, "sample", features=ds.features("RNA"))
        >>> list(pb.to_long().columns)
        ['group', 'feature', 'value', 'n_detected']
        """
        gi, fi = np.nonzero(self.X > min_value)
        return pd.DataFrame({"group": self.groups.index.to_numpy()[gi], "feature": self.features.to_numpy()[fi],
                             "value": self.X[gi, fi], "n_detected": self.n_detected[gi, fi]})

    def write(self, out_dir) -> Path:
        """Write ``counts.parquet`` (features x groups, first column ``feature``) and ``groups.parquet``.

        Examples
        --------
        >>> import arsc
        >>> ds = arsc.datasets.toy()
        >>> pb = arsc.pseudobulk(ds.X("RNA", "counts"), ds.obs, "sample", features=ds.features("RNA"))
        >>> sorted(p.name for p in pb.write("pb_out").iterdir())
        ['counts.parquet', 'groups.parquet']
        """
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        wide = self.to_frame()
        wide.index.name = "feature"
        wide.reset_index().to_parquet(d / "counts.parquet", index=False)
        g = self.groups.copy()
        g.index.name = "group"
        g.reset_index().to_parquet(d / "groups.parquet", index=False)
        return d


def pseudobulk(X, obs: pd.DataFrame | None, by, *, features=None, min_cells: int = 0,
               method: str = "sum", expm1: bool = False) -> Pseudobulk:
    """Aggregate cells into groups (pseudobulk; Seurat ``AggregateExpression``), exact integer sums.

    The patient (sample) is the unit of inference for condition comparisons: aggregate counts per
    patient x condition (x cell type), then test with :func:`arsc.de.deseq2` or :func:`arsc.de.voom`.

    Parameters
    ----------
    X : cells x features sparse matrix (CSR preferred; e.g. raw counts).
    obs : cell metadata (row-aligned with ``X``); may be ``None`` if ``by`` is an array.
    by : column name, list of column names (combined, Seurat order ``a_b``), or labels.
        Cells with NA in any grouping column are excluded (as in Seurat).
    features : feature names (defaults to ``0..n-1``).
    min_cells : drop groups with fewer cells (cell floor); reported in ``.dropped``.
    method : ``"sum"`` (AggregateExpression) or ``"mean"`` (AverageExpression).
    expm1 : apply ``expm1`` before aggregating (Seurat does this for the ``data`` slot).

    Returns
    -------
    Pseudobulk with ``X`` (groups x features), ``groups`` (one row per group with ``n_cells``),
    ``features``, ``n_detected`` and ``dropped``; ``.to_frame()`` gives features x groups.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy()
    >>> cd8 = (ds.obs["cell_type"] == "CD8 T").to_numpy()
    >>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[cd8], ds.obs[cd8], ["patient", "timepoint"],
    ...                      features=ds.features("RNA"), min_cells=10)
    >>> pb.groups.index[:2].tolist()
    ['P1_pre', 'P1_post']
    """
    X = as_csr(X)
    n, p = X.shape
    codes, groups = factorize_groups(obs, by, n)
    keep = codes >= 0
    order = np.argsort(codes, kind="stable")[np.count_nonzero(~keep):]
    counts = np.bincount(codes[keep], minlength=len(groups))
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    S, nnz = _group_sums(X.indptr, X.indices, X.data, order.astype(np.int64), starts, p, expm1)
    groups = groups.copy()
    groups["n_cells"] = counts
    ok = counts >= min_cells
    if method == "mean":
        S = S / counts[:, None]
    elif method != "sum":
        raise ValueError(f"method must be 'sum' (AggregateExpression) or 'mean' (AverageExpression); got {method!r}")
    feats = pd.Index(features) if features is not None else pd.RangeIndex(p)
    if len(feats) != p:
        raise ValueError(f"features has {len(feats)} names but X has {p} columns")
    return Pseudobulk(S[ok], groups[ok], feats, nnz[ok], groups[~ok])
