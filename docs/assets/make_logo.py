"""Generate the arsc logo, its dark-mode variant and the square icon (deterministic).

    python docs/assets/make_logo.py

Writes, next to this script:
    logo.png, logo.svg            mark + wordmark, for light backgrounds (transparent)
    logo-dark.png, logo-dark.svg  the same for dark backgrounds
    icon.png (256 px), icon.svg   the mark alone
    icon-512.png                  the mark alone, 512 px

The mark is a single cell drawn as a point cloud: four clusters of dots (like a UMAP of cell types)
inside a round membrane. It needs only numpy and matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = Path(__file__).resolve().parent

# cluster colors: teal, indigo, coral, amber (distinct in both themes)
PALETTE = ["#14B8A6", "#6366F1", "#F43F5E", "#F59E0B"]
INK_LIGHT = "#0F172A"      # wordmark on light backgrounds
INK_DARK = "#F1F5F9"       # wordmark on dark backgrounds
RING_LIGHT = "#334155"
RING_DARK = "#CBD5E1"
SUB_LIGHT = "#64748B"
SUB_DARK = "#94A3B8"


def _font() -> str:
    """First available clean sans-serif font (falls back to DejaVu Sans, shipped with matplotlib)."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Montserrat", "Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "DejaVu Sans"


def _cloud(seed: int = 7):
    """Point cloud: 4 anisotropic Gaussian clusters inside the unit disc, plus a few bridging cells."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-0.30, 0.30], [0.33, 0.28], [0.26, -0.34], [-0.34, -0.26]])
    sizes = [70, 60, 64, 56]
    pts, cols, rad = [], [], []
    for k, (c, n) in enumerate(zip(centers, sizes)):
        ang = rng.uniform(0, np.pi)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        z = rng.normal(size=(n, 2)) * np.array([0.13, 0.075])
        p = c + z @ R.T
        pts.append(p)
        cols += [PALETTE[k]] * n
        rad.append(rng.uniform(0.020, 0.030, n))
    P = np.vstack(pts)
    r = np.concatenate(rad)
    keep = np.hypot(P[:, 0], P[:, 1]) + r < 0.80          # stay inside the membrane
    return P[keep], np.array(cols)[keep], r[keep]


def _draw_mark(ax, ring: str, cx: float = 0.0, cy: float = 0.0, scale: float = 1.0):
    P, cols, r = _cloud()
    ax.add_patch(Circle((cx, cy), 0.92 * scale, fill=False, lw=0, ec="none"))
    ax.add_patch(Circle((cx, cy), 0.90 * scale, fill=False, ec=ring, lw=4.0 * scale, alpha=0.95))
    for (x, y), c, rr in zip(P, cols, r):
        ax.add_patch(Circle((cx + x * scale, cy + y * scale), rr * scale, color=c, lw=0, alpha=0.92))
    # nucleus-like highlight: one larger dot per cluster center (subtle)
    return ax


def make_logo(dark: bool, out_stem: str):
    ink, ring, sub = (INK_DARK, RING_DARK, SUB_DARK) if dark else (INK_LIGHT, RING_LIGHT, SUB_LIGHT)
    fig = plt.figure(figsize=(8.9, 3.333), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.34)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_mark(ax, ring, cx=1.05, cy=1.0, scale=0.86)
    font = _font()
    ax.text(2.12, 1.12, "arsc", fontsize=104, fontweight="bold", color=ink, family=font,
            va="center", ha="left")
    ax.text(2.18, 0.50, "fast, validated single-cell analysis", fontsize=19, color=sub, family=font,
            va="center", ha="left")
    for ext, kw in (("png", dict(dpi=120)), ("svg", {})):
        fig.savefig(HERE / f"{out_stem}.{ext}", transparent=True, **kw)
    plt.close(fig)


def make_icon(size_px: int, out_name: str, svg: bool = False):
    fig = plt.figure(figsize=(2.56, 2.56), dpi=size_px / 2.56)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_mark(ax, RING_LIGHT, scale=1.05)
    fig.savefig(HERE / out_name, transparent=True)
    if svg:
        fig.savefig(HERE / out_name.replace(".png", ".svg"), transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    make_logo(dark=False, out_stem="logo")
    make_logo(dark=True, out_stem="logo-dark")
    make_icon(256, "icon.png", svg=True)
    make_icon(512, "icon-512.png")
    print("wrote", ", ".join(sorted(p.name for p in HERE.iterdir() if p.suffix in (".png", ".svg"))))
