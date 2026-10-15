"""Small shared helpers (grouping, multiple testing, sparse plumbing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def as_csr(X, dtype=None) -> sp.csr_matrix:
    """Return ``X`` as CSR with sorted indices and no explicit zeros (copy only if needed)."""
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    elif not isinstance(X, sp.csr_matrix):
        X = X.tocsr()
    if dtype is not None and X.dtype != dtype:
        X = X.astype(dtype)
    if not X.has_sorted_indices:
        X = X.copy()
        X.sort_indices()
    return X


def factorize_groups(obs: pd.DataFrame | None, by, n: int | None = None, sep: str = "_"):
    """Encode one or more grouping columns as integer codes.

    ``by`` may be a column name, a list of column names (combined like Seurat's
    ``group.by = c(a, b)`` -> ``"a_b"``), or an array-like of labels.
    Cells with a missing value in any grouping column get code -1.

    Returns
    -------
    codes : int64 array (n_cells,)
    groups : DataFrame with one row per observed group (in sorted level order),
        columns = grouping variables, index = combined label.
    """
    if isinstance(by, str) and (obs is None or by not in obs.columns):
        where = "obs is None" if obs is None else f"obs has columns {list(obs.columns)}"
        raise KeyError(f"grouping column {by!r} not found ({where}); pass obs= and an existing column name, "
                       "or pass the labels themselves (an array/Series with one label per cell)")
    if (isinstance(by, (list, tuple)) and by and all(isinstance(b, str) for b in by) and obs is not None
            and any(b in obs.columns for b in by) and not all(b in obs.columns for b in by)):
        raise KeyError(f"grouping columns {[b for b in by if b not in obs.columns]} not in obs; "
                       f"available: {list(obs.columns)}")
    if isinstance(by, str) or (isinstance(by, (list, tuple)) and all(isinstance(b, str) for b in by)
                               and obs is not None and all(b in obs.columns for b in by)):
        cols = [by] if isinstance(by, str) else list(by)
        frame = obs[cols]
    else:
        arr = by if isinstance(by, pd.Series) else pd.Series(np.asarray(by))
        cols = [arr.name if arr.name is not None else "group"]
        frame = pd.DataFrame({cols[0]: arr.to_numpy()})
    if n is not None and len(frame) != n:
        raise ValueError(f"grouping has {len(frame)} entries but the matrix has {n} rows (cells); obs/labels "
                         "must be row-aligned with X (subset both with the same mask)")
    cats, mats = [], []
    valid = np.ones(len(frame), dtype=bool)
    for c in cols:
        s = frame[c]
        cat = s if isinstance(s.dtype, pd.CategoricalDtype) else pd.Categorical(s)
        cat = pd.Categorical(cat)
        codes = np.asarray(cat.codes, dtype=np.int64)
        valid &= codes >= 0
        cats.append(cat.categories)
        mats.append(codes)
    # Mixed-radix combination in level order of each column (first column slowest).
    combo = np.zeros(len(frame), dtype=np.int64)
    for codes, levels in zip(mats, cats):
        combo = combo * len(levels) + np.where(codes >= 0, codes, 0)
    combo[~valid] = -1
    uniq, inv = np.unique(combo[valid], return_inverse=True)
    codes_out = np.full(len(frame), -1, dtype=np.int64)
    codes_out[valid] = inv
    # Decode group levels
    rows = {}
    rem = uniq.copy()
    for c, levels in reversed(list(zip(cols, cats))):
        rows[c] = np.asarray(levels)[rem % len(levels)]
        rem //= len(levels)
    groups = pd.DataFrame({c: rows[c] for c in cols})
    groups.index = groups.astype(str).agg(sep.join, axis=1).to_numpy() if len(cols) > 1 else groups[cols[0]].astype(str).to_numpy()
    return codes_out, groups


def p_adjust_bh(p: np.ndarray, axis: int = 0) -> np.ndarray:
    """Benjamini-Hochberg adjustment along ``axis`` (matches R ``p.adjust(.., "BH")``; NaNs kept)."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        return _bh_1d(p)
    return np.apply_along_axis(_bh_1d, axis, p)


def _bh_1d(p):
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    o = np.argsort(-q, kind="stable")
    ranks = np.arange(n, 0, -1, dtype=np.float64)
    adj = np.minimum.accumulate((n / ranks) * q[o])  # same op order as R: n/i * p[o]
    res = np.empty(n)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def check_seed(seed, name: str = "seed", r_seed: bool = False) -> int:
    """Validate an explicit RNG seed and return it as a Python int (see SEEDS.md).

    arsc results must be exactly reproducible, so every stochastic function takes a fixed integer
    seed. ``None`` (fresh OS entropy -> a different answer on every call) is rejected rather than
    silently accepted; to explore stochastic variation, pass different integers (1, 2, 3, ...).
    ``r_seed=True``: an R ``set.seed`` value (any 32-bit signed integer, negatives allowed).
    """
    if seed is None:
        raise ValueError(f"{name}=None is not allowed: arsc results must be reproducible. Pass an integer "
                         f"(e.g. {name}=1, 2, 3 to check stability across seeds); see SEEDS.md.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    if r_seed:
        if not -2 ** 31 <= seed < 2 ** 31:
            raise ValueError(f"{name} must be a 32-bit integer (R set.seed), got {seed}")
        return seed
    if seed < 0 or seed >= 2 ** 32:
        raise ValueError(f"{name} must be in [0, 2**32), got {seed}")
    return seed
