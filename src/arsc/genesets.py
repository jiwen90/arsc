"""Gene-set readers (Python ports of the R helpers ``ar_read_genesets`` / ``ar_msigdb`` / ``ar_cycling_fraction``).

Gene sets are plain ``dict[str, list[str]]``. Readers cover JSON (MSigDB-style ``{name: {"geneSymbols": [...]}}``
or ``{name: [genes]}``), GMT, hotspot-style module CSVs, CellPhoneDB pathway tables and one-gene-per-line
``.txt`` lists. MSigDB is read from local GMT files only (no network): set ``ARSC_MSIGDB_DIR`` or drop GMTs into
``<cache>/msigdb`` (cache = ``$ARSC_CACHE``, else ``$AR_ROOT/.cache`` when ``AR_ROOT`` is set, else
``~/.cache/arsc``); :func:`export_msigdb_gmt` can fill that cache through R ``msigdbr`` when its data package
(msigdbdf) is installed.

Environment variables: ``ARSC_CACHE`` (cache root), ``ARSC_MSIGDB_DIR`` (extra GMT directory),
``ARSC_DATA_V3`` (folder of the named collections loaded by :func:`load_v3`), ``ARSC_RLIB`` (extra R library
for the R helpers), ``AR_ROOT`` (host project root; adds its ``.cache``, ``.Rlib``, ``scratch/arsc/msigdb`` and
``data/msigdb`` to the search paths).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["read_genesets", "read_json_sets", "read_gmt", "read_module_csv", "read_cpdb", "load_v3",
           "V3_SETS", "msigdb", "export_msigdb_gmt", "filter_genesets", "cycling_fraction"]

_v3 = os.environ.get("ARSC_DATA_V3")
DATA_V3 = Path(_v3) if _v3 else None
_ar = os.environ.get("AR_ROOT")
AR_ROOT = Path(_ar) if _ar else None


def _cache_root() -> Path:
    """``$ARSC_CACHE``, else ``$AR_ROOT/.cache``, else ``~/.cache/arsc``."""
    env = os.environ.get("ARSC_CACHE")
    if env:
        return Path(env)
    if AR_ROOT is not None:
        return AR_ROOT / ".cache"
    return Path.home() / ".cache" / "arsc"


def _r_libpaths_code() -> str:
    """R code that prepends ``$ARSC_RLIB`` / ``$AR_ROOT/.Rlib`` to ``.libPaths()`` when set (else nothing)."""
    libs = []
    if os.environ.get("ARSC_RLIB"):
        libs.append(Path(os.environ["ARSC_RLIB"]).as_posix())
    if AR_ROOT is not None and (AR_ROOT / ".Rlib").exists():
        libs.append((AR_ROOT / ".Rlib").as_posix())
    if not libs:
        return ""
    return ".libPaths(unique(c(" + ", ".join(f'"{x}"' for x in libs) + ", .libPaths()))); "


def _v3_root() -> Path:
    if DATA_V3 is None:
        raise RuntimeError("load_v3() needs the folder of the named gene-set collections: set the environment "
                           "variable ARSC_DATA_V3 to it (before importing arsc), or read the files directly with "
                           "arsc.genesets.read_genesets(path).")
    return DATA_V3


def _clean(gs: dict, min_size=1) -> dict:
    out = {}
    for k, g in gs.items():
        v = list(dict.fromkeys(str(x) for x in g if pd.notna(x) and str(x) != ""))
        if len(v) >= min_size:
            out[str(k)] = v
    return out


def read_json_sets(path) -> dict:
    """Read gene sets from JSON: MSigDB-style ``{name: {"geneSymbols": [...]}}`` or plain ``{name: [genes]}``.

    Parameters
    ----------
    path : path of the ``.json`` file.

    Returns
    -------
    dict ``{set name: [gene symbols]}`` (not de-duplicated; :func:`read_genesets` cleans).

    Examples
    --------
    >>> import json, os, tempfile
    >>> from arsc import genesets
    >>> f = os.path.join(tempfile.mkdtemp(), "sets.json")
    >>> json.dump({"TCELL": {"geneSymbols": ["CD3E", "CD8A"]}, "BCELL": ["MS4A1"]}, open(f, "w"))
    >>> genesets.read_json_sets(f)
    {'TCELL': ['CD3E', 'CD8A'], 'BCELL': ['MS4A1']}
    """
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return {k: (v["geneSymbols"] if isinstance(v, dict) and "geneSymbols" in v else v) for k, v in j.items()}


def read_gmt(path) -> dict:
    r"""Read a GMT file (``name <TAB> description <TAB> gene1 <TAB> gene2 ...``, one set per line).

    Parameters
    ----------
    path : path of the ``.gmt`` file.

    Returns
    -------
    dict ``{set name: [genes]}``; lines with no genes are skipped.

    Examples
    --------
    >>> import os, tempfile
    >>> from arsc import genesets
    >>> f = os.path.join(tempfile.mkdtemp(), "sets.gmt")
    >>> _ = open(f, "w").write("TCELL\tna\tCD3E\tCD8A\nBCELL\tna\tMS4A1\tCD79A\n")
    >>> genesets.read_gmt(f)
    {'TCELL': ['CD3E', 'CD8A'], 'BCELL': ['MS4A1', 'CD79A']}
    """
    gs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 2:
                gs[p[0]] = p[2:]
    return gs


def read_module_csv(path) -> dict:
    """Read gene modules from a CSV: long (gene + module/set/pathway column) or wide (one column per set).

    Long format (e.g. hotspot ``module_genes``): module ``-1`` (unassigned) is dropped and numeric modules are
    named ``module<k>``. Wide format: every column is a set (index-like columns are ignored).

    Parameters
    ----------
    path : path of the ``.csv`` file.

    Returns
    -------
    dict ``{set name: [genes]}``.

    Examples
    --------
    >>> import os, tempfile
    >>> import pandas as pd
    >>> from arsc import genesets
    >>> f = os.path.join(tempfile.mkdtemp(), "modules.csv")
    >>> pd.DataFrame({"gene": ["CD3E", "CD8A", "LYZ"], "module": [1, 1, -1]}).to_csv(f, index=False)
    >>> genesets.read_module_csv(f)
    {'module1': ['CD3E', 'CD8A']}
    """
    d = pd.read_csv(path)
    lc = {c.lower(): c for c in d.columns}
    gcol = next((lc[x] for x in ("gene", "genes", "symbol") if x in lc), None)
    scol = next((lc[x] for x in ("module", "set", "pathway", "geneset") if x in lc), None)
    if gcol and scol:
        d = d[d[scol].notna() & (d[scol] != -1)]
        names = d[scol].map(lambda v: f"module{int(v)}" if isinstance(v, (int, float, np.number)) and float(v).is_integer()
                            else f"module{v}")
        return {k: g.astype(str).tolist() for k, g in d.groupby(names, sort=True)[gcol]}
    drop = {"", "X", "...1"} | {c for c in d.columns if str(c).startswith("Unnamed")}
    return {c: d[c].dropna().astype(str).tolist() for c in d.columns if c not in drop}


def read_cpdb(cpdb_csv, names_from=None) -> dict:
    """Read a CellPhoneDB pathway table (columns ``pathway``, ``hgnc_symbol_ids`` comma-separated).

    The first row of a duplicated pathway name wins, as in the original R helper.

    Parameters
    ----------
    cpdb_csv : path of the pathway CSV.
    names_from : optional ``.txt`` file, or directory of ``.txt`` files, listing the pathway names to keep
        (in that order).

    Returns
    -------
    dict ``{pathway: [genes]}``.

    Examples
    --------
    >>> import os, tempfile
    >>> import pandas as pd
    >>> from arsc import genesets
    >>> f = os.path.join(tempfile.mkdtemp(), "cpdb.csv")
    >>> pd.DataFrame({"pathway": ["IFN", "TNF"], "hgnc_symbol_ids": ["IFNG,IFNGR1", "TNF,TNFRSF1A"]}).to_csv(f, index=False)
    >>> genesets.read_cpdb(f)
    {'IFN': ['IFNG', 'IFNGR1'], 'TNF': ['TNF', 'TNFRSF1A']}
    """
    d = pd.read_csv(cpdb_csv).drop_duplicates("pathway")
    gs = {p: str(s).split(",") for p, s in zip(d.pathway, d.hgnc_symbol_ids)}
    if names_from is not None:
        p = Path(names_from)
        files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        want = [ln.strip() for f in files for ln in f.read_text(encoding="utf-8").splitlines()]
        gs = {n: gs[n] for n in dict.fromkeys(want) if n in gs}
    return gs


def read_genesets(path=None, cpdb_csv=None, min_size=1) -> dict:
    r"""Read gene sets from any supported file into ``{name: [genes]}`` (Python port of ``ar_read_genesets``).

    Dispatches on the path: ``.json`` (:func:`read_json_sets`), ``.gmt`` (:func:`read_gmt`), ``.csv``
    (:func:`read_module_csv`), a directory of ``.txt`` files (one set per file, named by the file stem), or a
    single ``.txt`` file (one gene per line). Genes are de-duplicated in order and empty entries dropped.

    Parameters
    ----------
    path : file or directory (for ``cpdb_csv``: optional list of pathway names to keep, see :func:`read_cpdb`).
    cpdb_csv : path of a CellPhoneDB pathway table (then ``path`` restricts the pathways).
    min_size : drop sets with fewer genes (default 1).

    Returns
    -------
    dict ``{set name: [genes]}``.

    Examples
    --------
    >>> import os, tempfile
    >>> from arsc import genesets
    >>> f = os.path.join(tempfile.mkdtemp(), "sets.gmt")
    >>> _ = open(f, "w").write("TCELL\tna\tCD3E\tCD8A\nBCELL\tna\tMS4A1\tCD79A\n")
    >>> gs = genesets.read_genesets(f, min_size=2)
    >>> sorted(gs)
    ['BCELL', 'TCELL']
    """
    if cpdb_csv is not None:
        gs = read_cpdb(cpdb_csv, path)
    else:
        p = Path(path)
        if p.is_dir():
            gs = {f.stem: f.read_text(encoding="utf-8").split() for f in sorted(p.glob("*.txt"))}
        elif p.suffix == ".json":
            gs = read_json_sets(p)
        elif p.suffix == ".gmt":
            gs = read_gmt(p)
        elif p.suffix == ".csv":
            gs = read_module_csv(p)
        else:
            gs = {p.stem: [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]}
    return _clean(gs, min_size)


# Named collections of the ARSC_DATA_V3 folder -> loader (evaluated lazily)
V3_SETS = {
    "jo_pathways": lambda: read_json_sets(_v3_root() / "jo_pathways.json"),
    "jo_pathways_subsets": lambda: read_json_sets(_v3_root() / "jo_pathways_subsets.json"),
    "pathways": lambda: read_json_sets(_v3_root() / "pathways.json"),
    "hotspot_modules": lambda: read_module_csv(_v3_root() / "hotspot" / "2023 0203" / "module_genes_clean_3000_200.csv"),
    "cpdb_all": lambda: read_cpdb(_v3_root() / "hotspot" / "CPDB_pathways_genes.csv"),
    "cpdb_selected": lambda: read_cpdb(_v3_root() / "hotspot" / "CPDB_pathways_genes.csv", _v3_root() / "cpdb gene sets"),
    "cpdb_categories": lambda: {f.stem: [x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                                for f in sorted((_v3_root() / "cpdb gene sets").glob("*.txt"))},
    "cellcycle_go": lambda: {"cellcycle_go": _read_listfile(_v3_root() / "cellcycle_go.txt")},
    "cellcycle_kegg": lambda: {"cellcycle_kegg": _read_listfile(_v3_root() / "cellcycle_kegg.txt")},
    "cellcycle_reactome": lambda: {"cellcycle_reactome": _read_listfile(_v3_root() / "cellcycle_reactome.txt")},
    "gnf2_mki67": lambda: {"gnf2_mki67": _read_listfile(_v3_root() / "gnf2_mki67.txt")},
    "regev_cell_cycle": lambda: {"regev_cell_cycle": _read_listfile(_v3_root() / "regev_lab_cell_cycle_genes.txt")},
}


def _read_listfile(path) -> list:
    """One gene per line, or a single comma-separated (optionally quoted) line."""
    txt = Path(path).read_text(encoding="utf-8")
    out = []
    for ln in txt.splitlines():
        out.extend(x.strip().strip('"') for x in ln.split(","))
    return [x for x in out if x]


def load_v3(name: str, min_size=1) -> dict:
    """Load a named gene-set collection from the folder given by the ``ARSC_DATA_V3`` environment variable.

    The names and file layout are listed in ``V3_SETS`` (JSON pathway files, hotspot modules, CellPhoneDB
    tables, cell-cycle lists). Raises a RuntimeError that explains the fix when ``ARSC_DATA_V3`` is not set.

    Parameters
    ----------
    name : one of ``sorted(arsc.genesets.V3_SETS)``.
    min_size : drop sets with fewer genes (default 1).

    Returns
    -------
    dict ``{set name: [genes]}``.

    Examples
    --------
    >>> from arsc import genesets
    >>> "pathways" in genesets.V3_SETS
    True
    >>> gs = genesets.load_v3("pathways")  # doctest: +SKIP
    """
    if name not in V3_SETS:
        raise KeyError(f"unknown set {name!r}; available: {sorted(V3_SETS)}")
    return _clean(V3_SETS[name](), min_size)


# ---------------------------------------------------------------------------------------------------- MSigDB
def _msigdb_dirs():
    env = os.environ.get("ARSC_MSIGDB_DIR")
    dirs = ([Path(env)] if env else []) + [_cache_root() / "msigdb"]
    if AR_ROOT is not None:
        dirs += [AR_ROOT / "scratch" / "arsc" / "msigdb", AR_ROOT / "data" / "msigdb"]
    return dirs


def msigdb(collection="H", subcollection=None, species="Homo sapiens", cache_dir=None) -> dict:
    r"""MSigDB gene sets from a local GMT cache (no network).

    Looks for ``<collection>[.<subcollection>].gmt`` (case-insensitive, ``:`` replaced by ``_``) or any GMT
    whose name follows the MSigDB convention (``h.all*``, ``c2.cp.reactome*``) in ``cache_dir``, else in
    ``$ARSC_MSIGDB_DIR``, ``<cache>/msigdb`` (``$ARSC_CACHE``, ``$AR_ROOT/.cache`` or ``~/.cache/arsc``) and, when
    ``AR_ROOT`` is set, ``$AR_ROOT/scratch/arsc/msigdb`` and ``$AR_ROOT/data/msigdb``. Raises FileNotFoundError
    listing the searched folders when absent (fill them with :func:`export_msigdb_gmt` or GMTs downloaded from
    gsea-msigdb.org). For the R-msigdbr route with a csv cache see :func:`arsc.de.msigdb`.

    Parameters
    ----------
    collection : MSigDB collection, e.g. ``"H"`` (hallmark), ``"C2"``, ``"C7"``.
    subcollection : e.g. ``"CP:REACTOME"``; ``None`` for the whole collection.
    species : kept for API symmetry with :func:`export_msigdb_gmt` (the GMT file decides the species).
    cache_dir : search only this directory.

    Returns
    -------
    dict ``{set name: [genes]}``.

    Examples
    --------
    >>> import os, tempfile
    >>> from arsc import genesets
    >>> d = tempfile.mkdtemp()
    >>> _ = open(os.path.join(d, "h.all.v2024.1.Hs.symbols.gmt"), "w").write("HALLMARK_X\tna\tA\tB\n")
    >>> genesets.msigdb("H", cache_dir=d)
    {'HALLMARK_X': ['A', 'B']}
    """
    dirs = [Path(cache_dir)] if cache_dir else _msigdb_dirs()
    sub = (subcollection or "").lower().replace(":", "_").replace("-", "_")
    c = collection.lower()
    cands = [f"{c}.{sub}.gmt"] if sub else [f"{c}.gmt"]
    for d in dirs:
        if not d.exists():
            continue
        files = {f.name.lower(): f for f in d.glob("*.gmt")}
        for cn in cands:
            if cn in files:
                return read_gmt(files[cn])
        pref = f"{c}.{sub.replace('cp_', 'cp.')}" if sub else f"{c}.all"
        for n, f in sorted(files.items()):
            if n.startswith(pref):
                return read_gmt(f)
    raise FileNotFoundError(
        f"no local MSigDB GMT for {collection}/{subcollection}; searched {[str(d) for d in dirs]}. "
        "Put GMT files there or run arsc.genesets.export_msigdb_gmt() (needs R msigdbr + msigdbdf).")


def export_msigdb_gmt(collection="H", subcollection=None, species="Homo sapiens", out_dir=None) -> Path:
    """Write one MSigDB collection to a GMT file through R ``msigdbr`` (one-off; needs msigdbr + msigdbdf in R).

    Extra R libraries come from ``$ARSC_RLIB`` and ``$AR_ROOT/.Rlib`` when set.

    Parameters
    ----------
    collection : MSigDB collection, e.g. ``"H"``.
    subcollection : e.g. ``"CP:REACTOME"``.
    species : msigdbr species name (default ``"Homo sapiens"``).
    out_dir : output directory (default: the first folder :func:`msigdb` searches).

    Returns
    -------
    pathlib.Path of the written ``.gmt``.

    Examples
    --------
    >>> from arsc import genesets
    >>> f = genesets.export_msigdb_gmt("H")  # doctest: +SKIP
    >>> hallmark = genesets.msigdb("H")  # doctest: +SKIP
    """
    out_dir = Path(out_dir or _msigdb_dirs()[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = collection.lower() + (("." + subcollection.lower().replace(":", "_").replace("-", "_")) if subcollection else "")
    out = out_dir / f"{name}.gmt"
    sub = f', subcollection = "{subcollection}"' if subcollection else ""
    code = (_r_libpaths_code() + 'library(msigdbr); '
            f'd <- msigdbr(species = "{species}", collection = "{collection}"{sub}); '
            'l <- split(d$gene_symbol, d$gs_name); '
            f'writeLines(vapply(names(l), function(n) paste(c(n, "NA", unique(l[[n]])), collapse = "\\t"), ""), "{out.as_posix()}")')
    subprocess.run(["Rscript", "-e", code], check=True)
    return out


# ---------------------------------------------------------------------------------------------------- misc
def filter_genesets(gs: dict, universe, min_size=1) -> dict:
    """Restrict gene sets to genes in ``universe`` (e.g. the dataset's features) and drop small sets.

    Parameters
    ----------
    gs : dict ``{set name: [genes]}``.
    universe : iterable of genes to keep (e.g. ``ds.features("RNA")``).
    min_size : drop sets with fewer than this many genes left (default 1).

    Returns
    -------
    dict ``{set name: [genes in universe]}``.

    Examples
    --------
    >>> import arsc
    >>> from arsc import genesets
    >>> ds = arsc.datasets.toy()
    >>> genesets.filter_genesets({"T": ["CD3E", "CD8A", "NOT_A_GENE"]}, ds.features("RNA"))
    {'T': ['CD3E', 'CD8A']}
    """
    u = set(universe)
    return {k: [g for g in v if g in u] for k, v in gs.items() if sum(g in u for g in v) >= min_size}


def cycling_fraction(meta, positive, sample_cols=("patient_alias", "timepoint"), by=None, min_cells=20) -> pd.DataFrame:
    """Per-sample counts of positive cells by group, with a count-corrected logit for paired deltas.

    ``logit_cc = log((k + 0.5) / (n - k + 0.5))`` stays finite when ``k`` is 0 or ``n`` (sparse positives such as
    MKI67+ cells). The sample (patient x timepoint) is the unit; compare samples, not cells.

    Parameters
    ----------
    meta : cell-level metadata.
    positive : boolean per cell (e.g. ``MKI67 counts > 0``), row-aligned with ``meta``.
    sample_cols : columns defining a sample (default ``("patient_alias", "timepoint")``; pass yours).
    by : optional extra grouping column(s) (e.g. cell type).
    min_cells : drop sample x group rows with fewer cells (default 20).

    Returns
    -------
    DataFrame: ``sample_cols`` (+ ``by``), ``n``, ``k``, ``prop``, ``logit_cc``.

    Examples
    --------
    >>> import arsc
    >>> from arsc import genesets
    >>> ds = arsc.datasets.toy()
    >>> mki67 = ds.X("RNA", "counts")[:, list(ds.features("RNA")).index("MKI67")].toarray().ravel() > 0
    >>> cf = genesets.cycling_fraction(ds.obs, mki67, sample_cols=("patient", "timepoint"), by="cell_type")
    >>> list(cf.columns)
    ['patient', 'timepoint', 'cell_type', 'n', 'k', 'prop', 'logit_cc']
    """
    by = [by] if isinstance(by, str) else list(by or [])
    keys = list(sample_cols) + by
    d = pd.DataFrame(meta)[keys].copy()
    d["pos"] = np.asarray(positive, dtype=bool)
    g = d.groupby(keys, observed=True, sort=True).agg(n=("pos", "size"), k=("pos", "sum")).reset_index()
    g = g[g.n >= min_cells].reset_index(drop=True)
    g["prop"] = g.k / g.n
    g["logit_cc"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
    return g
