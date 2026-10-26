"""1-D port of R's ``stats::loess`` default fit (netlib dloess).

R's defaults are ``degree = 2, family = "gaussian", surface = "interpolate",
cell = 0.2``. "interpolate" does not evaluate the local regression at every
point: it builds a k-d tree over x (cells hold <= floor(n*span*cell) points),
fits local quadratics only at the cell vertices (value + slope), and evaluates
everything else by cubic Hermite interpolation between neighboring vertices.
Seurat's vst HVG selection uses this fit, so we reproduce it (kd split rule
incl. the 2006 tie fix in ehg129/ehg124, 0.5% bounding-box margin, tricube
weights on the floor(n*span + 1e-5) nearest points).
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _local_fit(xs, ys, v, nf, degree):
    """Local polynomial fit at ``v``: returns (value, slope)."""
    n = xs.size
    # nf nearest neighbors of v in sorted xs: contiguous window
    pos = np.searchsorted(xs, v)
    lo, hi = pos, pos  # window [lo, hi)
    while hi - lo < nf:
        if lo == 0:
            hi += 1
        elif hi == n:
            lo -= 1
        elif v - xs[lo - 1] <= xs[hi] - v:
            lo -= 1
        else:
            hi += 1
    xw = xs[lo:hi] - v
    yw = ys[lo:hi]
    d = np.abs(xw)
    rho = d.max()
    if rho <= 0:
        return float(np.mean(yw)), 0.0
    u = d / rho
    w = np.where(u < 1.0, (1.0 - u ** 3) ** 3, 0.0)
    cols = [np.ones_like(xw), xw] + ([xw ** 2] if degree == 2 else [])
    A = np.stack(cols, axis=1)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], yw * sw, rcond=None)
    return float(coef[0]), float(coef[1])


def _kd_vertices(xs, fc):
    """Vertices of the 1-D loess k-d tree for sorted ``xs`` (ehg126/ehg124/ehg129)."""
    n = xs.size
    alpha, beta = xs[0], xs[-1]
    mu = 0.005 * max(beta - alpha, 1e-10 * max(abs(alpha), abs(beta)) + 1e-30)
    vlo0, vhi0 = alpha - mu, beta + mu
    verts = [vlo0, vhi0]
    x1 = np.concatenate([[np.nan], xs])  # 1-based view
    cells = deque([(1, n, vlo0, vhi0)])
    nvmax = max(200, n)
    while cells:
        L, U, vlo, vhi = cells.popleft()
        if (U - L + 1) <= fc or len(verts) + 1 > nvmax:
            continue
        m = (L + U) // 2
        offset = 0
        while True:
            if (m + offset) >= U or (m + offset) < L:
                break
            if x1[m + offset] == x1[m + offset + 1]:
                offset = -offset
                if offset >= 0:
                    offset += 1
            else:
                m = m + offset
                break
        xi = x1[m]
        if vlo == xi or vhi == xi:
            continue
        verts.append(xi)
        cells.append((L, m, vlo, xi))
        cells.append((m + 1, U, xi, vhi))
    return np.unique(np.asarray(verts))


def loess_fit(x, y, span: float = 0.75, degree: int = 2, surface: str = "interpolate",
              cell: float = 0.2) -> np.ndarray:
    """Fitted values of ``loess(y ~ x, span, degree)`` at the data points.

    ``surface="direct"`` evaluates the local fit at every point instead.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    nf = int(min(n, np.floor(n * span + 1e-5)))
    if nf <= degree:
        raise ValueError("span too small")
    if surface == "direct":
        fit_sorted = np.array([_local_fit(xs, ys, v, nf, degree)[0] for v in xs])
    else:
        fc = int(np.floor(n * span * cell))
        verts = _kd_vertices(xs, fc)
        vals = np.array([_local_fit(xs, ys, v, nf, degree) for v in verts])
        g, s = vals[:, 0], vals[:, 1]
        j = np.clip(np.searchsorted(verts, xs, side="right") - 1, 0, verts.size - 2)
        v0, v1 = verts[j], verts[j + 1]
        h = v1 - v0
        u = (xs - v0) / h
        phi0 = (1 - u) ** 2 * (1 + 2 * u)
        phi1 = u ** 2 * (3 - 2 * u)
        psi0 = u * (1 - u) ** 2
        psi1 = -(u ** 2) * (1 - u)
        fit_sorted = phi0 * g[j] + phi1 * g[j + 1] + (psi0 * s[j] + psi1 * s[j + 1]) * h
    out = np.empty(n)
    out[o] = fit_sorted
    return out
