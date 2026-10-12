"""Gene-set readers (port of ``ar_read_genesets`` / ``ar_msigdb`` / ``ar_cycling_fraction``).

Gene sets are plain ``dict[str, list[str]]``.  Readers cover the formats used by the Cancer Cell 2024 analysis
under ``data/2023 0129 V3`` (jo_pathways.json, pathways.json, hotspot module CSVs, CPDB lists, cell-cycle .txt
lists) plus GMT.  MSigDB is read from local GMT files only (no network): set ``ARSC_MSIGDB_DIR`` or drop GMTs into
``scratch/arsc/msigdb``; :func:`export_msigdb_gmt` can fill that cache through R ``msigdbr`` when its data
package (msigdbdf) is installed.
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

DATA_V3 = Path(os.environ.get("ARSC_DATA_V3", "E:/autoresearch/data/2023 0129 V3"))
AR_ROOT = Path(os.environ.get("AR_ROOT", "E:/autoresearch"))


def _clean(gs: dict, min_size=1) -> dict:
    out = {}
    for k, g in gs.items():
        v = list(dict.fromkeys(str(x) for x in g if pd.notna(x) and str(x) != ""))
        if len(v) >= min_size:
            out[str(k)] = v
    return out


def read_json_sets(path) -> dict:
    """MSigDB-style JSON ``{name: {"geneSymbols": [...]}}`` (jo_pathways.json, pathways.json,
    jo_pathways_subsets.json) or ``{name: [genes]}``."""
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return {k: (v["geneSymbols"] if isinstance(v, dict) and "geneSymbols" in v else v) for k, v in j.items()}


def read_gmt(path) -> dict:
    gs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 2:
                gs[p[0]] = p[2:]
    return gs


def read_module_csv(path) -> dict:
    """Long CSV (gene, module|set|pathway) such as hotspot ``module_genes_clean_*.csv`` (module -1 dropped,
    sets named ``module<k>``) or wide CSV with one column per set (hotspot ``gene_list_*.csv``)."""
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
    """CPDB pathway table (pathway, hgnc_symbol_ids comma-separated); first duplicate name wins, as the original.
    ``names_from``: a .txt file or directory of .txt files listing pathway names to keep (e.g. ``cpdb gene sets/``
    -> the 174 curated pathways)."""
    d = pd.read_csv(cpdb_csv).drop_duplicates("pathway")
    gs = {p: str(s).split(",") for p, s in zip(d.pathway, d.hgnc_symbol_ids)}
    if names_from is not None:
        p = Path(names_from)
        files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        want = [ln.strip() for f in files for ln in f.read_text(encoding="utf-8").splitlines()]
        gs = {n: gs[n] for n in dict.fromkeys(want) if n in gs}
    return gs


def read_genesets(path=None, cpdb_csv=None, min_size=1) -> dict:
    """Read gene sets into ``{name: [genes]}`` (see ``ar_read_genesets``): JSON, GMT, CSV (long/wide), a .txt
    (one gene per line) or a directory of .txt files; ``cpdb_csv`` for CPDB (optionally restricted by ``path``)."""
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


# Named sets of the V3 data folder -> loader
V3_SETS = {
    "jo_pathways": lambda: read_json_sets(DATA_V3 / "jo_pathways.json"),
    "jo_pathways_subsets": lambda: read_json_sets(DATA_V3 / "jo_pathways_subsets.json"),
    "pathways": lambda: read_json_sets(DATA_V3 / "pathways.json"),
    "hotspot_modules": lambda: read_module_csv(DATA_V3 / "hotspot" / "2023 0203" / "module_genes_clean_3000_200.csv"),
    "cpdb_all": lambda: read_cpdb(DATA_V3 / "hotspot" / "CPDB_pathways_genes.csv"),
    "cpdb_selected": lambda: read_cpdb(DATA_V3 / "hotspot" / "CPDB_pathways_genes.csv", DATA_V3 / "cpdb gene sets"),
    "cpdb_categories": lambda: {f.stem: [x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                                for f in sorted((DATA_V3 / "cpdb gene sets").glob("*.txt"))},
    "cellcycle_go": lambda: {"cellcycle_go": _read_listfile(DATA_V3 / "cellcycle_go.txt")},
    "cellcycle_kegg": lambda: {"cellcycle_kegg": _read_listfile(DATA_V3 / "cellcycle_kegg.txt")},
    "cellcycle_reactome": lambda: {"cellcycle_reactome": _read_listfile(DATA_V3 / "cellcycle_reactome.txt")},
    "gnf2_mki67": lambda: {"gnf2_mki67": _read_listfile(DATA_V3 / "gnf2_mki67.txt")},
    "regev_cell_cycle": lambda: {"regev_cell_cycle": _read_listfile(DATA_V3 / "regev_lab_cell_cycle_genes.txt")},
}


def _read_listfile(path) -> list:
    """One gene per line, or a single comma-separated (optionally quoted) line."""
    txt = Path(path).read_text(encoding="utf-8")
    out = []
    for ln in txt.splitlines():
        out.extend(x.strip().strip('"') for x in ln.split(","))
    return [x for x in out if x]


def load_v3(name: str, min_size=1) -> dict:
    """Load a named V3 gene-set collection (see ``V3_SETS``)."""
    if name not in V3_SETS:
        raise KeyError(f"unknown set {name!r}; available: {sorted(V3_SETS)}")
    return _clean(V3_SETS[name](), min_size)


# ---------------------------------------------------------------------------------------------------- MSigDB
def _msigdb_dirs():
    env = os.environ.get("ARSC_MSIGDB_DIR")
    return ([Path(env)] if env else []) + [AR_ROOT / "scratch" / "arsc" / "msigdb", AR_ROOT / "data" / "msigdb"]


def msigdb(collection="H", subcollection=None, species="Homo sapiens", cache_dir=None) -> dict:
    """MSigDB sets from a local GMT cache (no network).

    Looks for ``<collection>[.<subcollection>].gmt`` (case-insensitive, ``:`` replaced by ``_``) or any GMT whose
    name starts with the MSigDB convention (``h.all*``, ``c2.cp.reactome*``) in ``cache_dir`` /
    ``$ARSC_MSIGDB_DIR`` / ``scratch/arsc/msigdb`` / ``data/msigdb``.  Raises FileNotFoundError with instructions
    when absent (use :func:`export_msigdb_gmt` or R ``ar_msigdb`` + ``write``).
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
    """Write one MSigDB collection to GMT through R ``msigdbr`` (one-off; needs the msigdbdf data package)."""
    out_dir = Path(out_dir or AR_ROOT / "scratch" / "arsc" / "msigdb")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = collection.lower() + (("." + subcollection.lower().replace(":", "_").replace("-", "_")) if subcollection else "")
    out = out_dir / f"{name}.gmt"
    sub = f', subcollection = "{subcollection}"' if subcollection else ""
    code = ('.libPaths(c("E:/autoresearch/.Rlib", .libPaths())); library(msigdbr); '
            f'd <- msigdbr(species = "{species}", collection = "{collection}"{sub}); '
            'l <- split(d$gene_symbol, d$gs_name); '
            f'writeLines(vapply(names(l), function(n) paste(c(n, "NA", unique(l[[n]])), collapse = "\\t"), ""), "{out.as_posix()}")')
    subprocess.run(["Rscript", "-e", code], check=True)
    return out


# ---------------------------------------------------------------------------------------------------- misc
def filter_genesets(gs: dict, universe, min_size=1) -> dict:
    """Restrict sets to genes in ``universe`` (e.g. object features) and drop sets smaller than ``min_size``."""
    u = set(universe)
    return {k: [g for g in v if g in u] for k, v in gs.items() if sum(g in u for g in v) >= min_size}


def cycling_fraction(meta, positive, sample_cols=("patient_alias", "timepoint"), by=None, min_cells=20) -> pd.DataFrame:
    """Per-sample positive counts (e.g. MKI67 counts > 0 or score > threshold) by group with the count-corrected
    logit log((k+0.5)/(n-k+0.5)) for paired deltas (sparse positives)."""
    by = [by] if isinstance(by, str) else list(by or [])
    keys = list(sample_cols) + by
    d = pd.DataFrame(meta)[keys].copy()
    d["pos"] = np.asarray(positive, dtype=bool)
    g = d.groupby(keys, observed=True, sort=True).agg(n=("pos", "size"), k=("pos", "sum")).reset_index()
    g = g[g.n >= min_cells].reset_index(drop=True)
    g["prop"] = g.k / g.n
    g["logit_cc"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
    return g
