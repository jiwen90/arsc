"""Small synthetic datasets for tutorials, docstring examples and tests (no real data).

:func:`toy` simulates a PBMC-like study with a known ground truth, so every recipe in the docs has a
checkable answer:

* 6 patients (``P1``..``P6``) x 2 timepoints (``pre``, ``post``); ``P1``-``P3`` are responders
  (``response == "R"``), ``P4``-``P6`` non-responders (``"NR"``). One sample = one patient x timepoint.
* 5 cell types (``CD8 T``, ``CD4 T``, ``B``, ``NK``, ``Mono``) with real marker genes
  (``CD8A``, ``CD4``, ``MS4A1``, ``NKG7``, ``LYZ``...), plus Seurat cell-cycle genes, ``MT-`` genes and
  filler genes ``GENE0001``...
* Planted effects: in responders, CD8 T cells at ``post`` up-regulate ``GZMK``, ``GZMB``, ``IFNG``,
  ``PRF1`` and ``CXCL13`` (about 3x) and become more frequent; a few TCR clones per responder expand
  from ``pre`` to ``post``. Non-responders have no effect.
* T cells carry a ``clone_id`` (``NaN`` for B, NK and Mono cells); an ``ADT`` assay holds 6 proteins.

Example
-------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ds.n_cells
3000
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import check_seed

__all__ = ["toy", "toy_h5ad", "toy_view", "TOY_EFFECT_GENES", "TOY_MARKERS"]

#: marker genes per cell type in :func:`toy` (up-regulated about 8x in that type)
TOY_MARKERS = {
    "CD8 T": ["CD3E", "CD3D", "CD8A", "CD8B"],
    "CD4 T": ["CD3E", "CD3D", "CD4", "IL7R"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "NK": ["NKG7", "GNLY", "NCAM1", "KLRD1"],
    "Mono": ["LYZ", "CD14", "S100A8", "FCGR3A"],
}
#: genes up-regulated in responder CD8 T cells at ``post`` in :func:`toy`
TOY_EFFECT_GENES = ["GZMK", "GZMB", "IFNG", "PRF1", "CXCL13"]

_CC_S = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6"]
_CC_G2M = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "MKI67"]
_OTHER = ["PDCD1", "HAVCR2", "LAG3", "TOX", "TCF7", "SELL", "CCR7", "FOXP3", "ACTB", "GAPDH", "B2M"]
_MT = ["MT-CO1", "MT-ND1", "MT-ATP6"]
_ADT = ["CD3", "CD4", "CD8", "CD19", "CD14", "CD56"]
_ADT_HIGH = {"CD8 T": ["CD3", "CD8"], "CD4 T": ["CD3", "CD4"], "B": ["CD19"], "NK": ["CD56"], "Mono": ["CD14"]}


def _gene_list(n_genes: int) -> list[str]:
    named = list(dict.fromkeys([g for v in TOY_MARKERS.values() for g in v] + TOY_EFFECT_GENES + _CC_S
                               + _CC_G2M + _OTHER + _MT))
    if n_genes < len(named) + 10:
        raise ValueError(f"n_genes must be at least {len(named) + 10} (the toy needs its {len(named)} named "
                         f"genes plus fillers); got {n_genes}")
    return named + [f"GENE{i:04d}" for i in range(1, n_genes - len(named) + 1)]


def toy(n_cells: int = 3000, n_genes: int = 300, *, seed: int = 0, store=None, overwrite: bool = False):
    """Simulate a small PBMC-like study (6 patients x pre/post, known planted effects).

    Parameters
    ----------
    n_cells : number of cells (spread over the 12 samples).
    n_genes : number of genes (the named marker/effect/cell-cycle genes first, then fillers).
    seed : integer seed; the same seed gives an identical dataset.
    store : optional directory; if given, the data is written as a memory-mapped arsc store and an
        :class:`arsc.Store` is returned instead of an in-memory :class:`arsc.Dataset`.
    overwrite : replace an existing store at ``store``.

    Returns
    -------
    arsc.Dataset (or arsc.Store with ``store=``) with assays ``RNA`` (layer ``counts``) and ``ADT``
    (layer ``counts``), and ``obs`` columns ``patient``, ``timepoint``, ``response``, ``sample``,
    ``cell_type``, ``clone_id``.

    Examples
    --------
    >>> import arsc
    >>> ds = arsc.datasets.toy(n_cells=1200)
    >>> sorted(ds.obs["cell_type"].unique())
    ['B', 'CD4 T', 'CD8 T', 'Mono', 'NK']
    >>> ds.X("RNA", "counts").shape
    (1200, 300)
    """
    seed = check_seed(seed)
    rng = np.random.default_rng(seed)
    genes = _gene_list(n_genes)
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    patients = [f"P{i}" for i in range(1, 7)]
    resp = {p: ("R" if i < 3 else "NR") for i, p in enumerate(patients)}
    types = list(TOY_MARKERS)

    # ---- cells: sample assignment and cell-type composition (CD8 T expands in responders at post)
    samples = [(p, t) for p in patients for t in ("pre", "post")]
    sidx = rng.integers(0, len(samples), n_cells)
    base_p = np.array([0.25, 0.25, 0.15, 0.15, 0.20])
    ct = np.empty(n_cells, dtype=object)
    for s, (p, t) in enumerate(samples):
        m = sidx == s
        pr = base_p.copy()
        if resp[p] == "R" and t == "post":
            pr[0] *= 1.8
        pr = pr / pr.sum()
        ct[m] = rng.choice(types, size=int(m.sum()), p=pr)

    # ---- expression: gene base means, cell-type markers, planted effect, patient noise, NB counts
    base = rng.lognormal(mean=-1.0, sigma=1.0, size=G)
    for g in ("ACTB", "GAPDH", "B2M"):
        base[gi[g]] = 20.0
    for g in _MT:
        base[gi[g]] = 3.0
    for g in TOY_EFFECT_GENES:
        base[gi[g]] = 0.3
    lfc = np.zeros((len(types), G))
    for k, t in enumerate(types):
        for g in TOY_MARKERS[t]:
            lfc[k, gi[g]] += np.log(8.0)
    lfc[types.index("CD8 T"), [gi[g] for g in TOY_EFFECT_GENES]] += np.log(2.0)   # CD8 T baseline cytotoxic
    lfc[types.index("NK"), [gi[g] for g in ("GZMB", "PRF1")]] += np.log(6.0)
    pat_eff = rng.normal(0, 0.15, size=(len(patients), G))
    tix = np.array([types.index(t) for t in ct])
    pix = sidx // 2
    is_post = (sidx % 2) == 1
    eff_cells = (tix == 0) & is_post & np.array([resp[patients[i]] == "R" for i in pix])
    size = rng.lognormal(mean=0.0, sigma=0.35, size=n_cells)
    cycling = rng.random(n_cells) < 0.05
    mu = base[None, :] * np.exp(lfc[tix] + pat_eff[pix]) * size[:, None]
    mu[np.ix_(eff_cells, [gi[g] for g in TOY_EFFECT_GENES])] *= 3.0
    mu[np.ix_(cycling, [gi[g] for g in _CC_S + _CC_G2M])] *= 6.0
    disp = 0.2
    lam = rng.gamma(shape=1 / disp, scale=mu * disp)
    counts = sp.csr_matrix(rng.poisson(lam).astype(np.float32))

    # ---- ADT
    amu = np.full((n_cells, len(_ADT)), 5.0)
    for k, t in enumerate(types):
        for a in _ADT_HIGH[t]:
            amu[tix == k, _ADT.index(a)] = 150.0
    adt = sp.csr_matrix(rng.poisson(amu * size[:, None]).astype(np.float32))

    # ---- TCR clones for T cells: a pool per patient; responders expand 3 clones at post
    clone = np.full(n_cells, None, dtype=object)
    for i, p in enumerate(patients):
        expanded = [f"{p}_c{j}" for j in range(1, 4)]
        for t_post in (False, True):
            m = np.where((pix == i) & (is_post == t_post) & (tix <= 1))[0]
            if not len(m):
                continue
            # 25% of T cells in a shared moderately sized pool (clones seen at both timepoints), rest unique
            pool = [f"{p}_c{j}" for j in range(4, 40)]
            u = rng.random(len(m))
            lab = np.array([f"{p}_u{t_post:d}_{k}" for k in range(len(m))], dtype=object)
            sh = u < 0.25
            lab[sh] = rng.choice(pool, size=int(sh.sum()))
            p_exp = 0.35 if (resp[p] == "R" and t_post) else 0.03
            ex = u > 1 - p_exp
            lab[ex] = rng.choice(expanded, size=int(ex.sum()))
            clone[m] = lab

    cells = [f"cell{i:05d}" for i in range(n_cells)]
    obs = pd.DataFrame({
        "patient": pd.Categorical([patients[i] for i in pix], categories=patients),
        "timepoint": pd.Categorical(np.where(is_post, "post", "pre"), categories=["pre", "post"]),
        "response": pd.Categorical([resp[patients[i]] for i in pix], categories=["R", "NR"]),
        "sample": [f"{patients[i]}_{'post' if q else 'pre'}" for i, q in zip(pix, is_post)],
        "cell_type": pd.Categorical(ct, categories=types),
        "clone_id": pd.array(clone, dtype="object"),
    }, index=pd.Index(cells))
    obs["clone_id"] = obs["clone_id"].where(obs["clone_id"].notna(), np.nan)

    from .formats import StoreBuilder
    b = StoreBuilder(store, source="arsc.datasets.toy", overwrite=overwrite)
    b.set_cells(cells, obs)
    b.add_assay("RNA", genes)
    b.add_layer("RNA", "counts", counts)
    b.add_assay("ADT", _ADT)
    b.add_layer("ADT", "counts", adt)
    return b.finish()


def toy_h5ad(path, n_cells: int = 3000, n_genes: int = 300, *, seed: int = 0) -> Path:
    """Write :func:`toy` to an ``.h5ad`` file (RNA counts as ``X``), e.g. to try :func:`arsc.read`.

    Parameters
    ----------
    path : output ``.h5ad`` path.
    n_cells, n_genes, seed : passed to :func:`toy`.

    Returns
    -------
    pathlib.Path of the written file. Needs ``anndata`` (``pip install "arsc[h5ad]"``).

    Examples
    --------
    >>> import arsc, tempfile, os
    >>> f = arsc.datasets.toy_h5ad(os.path.join(tempfile.mkdtemp(), "toy.h5ad"), n_cells=500)
    >>> arsc.read(f).n_cells
    500
    """
    ds = toy(n_cells, n_genes, seed=seed)
    return Path(ds.write_h5ad(path, assay="RNA"))


def toy_view(path, n_cells: int = 600, *, seed: int = 0) -> dict:
    """A toy parent store, a full subset store of every other cell, and a view of that subset on the parent.

    Parameters
    ----------
    path : a new or empty directory; ``parent/``, ``full/`` (the subset as a full copy) and ``view/`` are written
        inside it (an existing set is reused).
    n_cells : cells of the parent (the subset keeps every other one).
    seed : integer seed of the toy data.

    Returns
    -------
    dict with the three store paths (``parent``, ``full``, ``view``) and the subset's parent rows (``rows``).

    Examples
    --------
    >>> import arsc, tempfile
    >>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
    >>> v = arsc.open(t["view"])
    >>> v.is_view, v.n_cells
    (True, 300)
    """
    from . import views
    from .formats import StoreBuilder
    from .io import Store
    path = Path(path)
    out = {"parent": path / "parent", "full": path / "full", "view": path / "view"}
    rows = np.arange(0, n_cells, 2)
    out["rows"] = rows
    if (out["view"] / views.VIEW_FILE).is_file():
        return out
    P = toy(n_cells, seed=seed, store=out["parent"], overwrite=True)
    sub = P.subset(rows)
    b = StoreBuilder(out["full"])
    b.set_cells(list(sub.obs.index), sub.obs)
    for a in sub.assays:
        b.add_assay(a, list(sub.features(a)))
        for l in sub.layers(a):
            b.add_layer(a, l, sub.X(a, l))
    b.finish(open_store=False)
    F = Store(out["full"])
    views.build(F, [out["parent"]], out["view"], {0: (np.arange(rows.size), rows)}, log=lambda *a: None)
    return out
