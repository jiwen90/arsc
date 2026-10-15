"""Render the benchmark headline tables (markdown) from benchmarks/results/suite/ and put them into README.md.

    python benchmarks/render_table.py                 # replace the block between the BENCH markers in README.md
    python benchmarks/render_table.py --print         # only print
    python benchmarks/render_table.py --results DIR --readme FILE

The text between ``<!-- BENCH:START -->`` and ``<!-- BENCH:END -->`` is replaced. If the markers are
absent the table is printed and README.md is left alone. Every number comes from summary.csv /
agreement.csv; "exact" is written only when the measured max |difference| is below 1e-10 (or the
outputs are bit-identical); otherwise the measured value is shown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from run_suite import IMPL  # noqa: E402
START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"

LABEL = {
    "normalize": ("Log-normalise", "`pp.log_normalize`", "`normalize_total`+`log1p`", "`NormalizeData`"),
    "hvg": ("Highly variable genes (2,000)", "`pp.hvg_vst`", "`highly_variable_genes(seurat_v3)`", "`FindVariableFeatures(vst)`"),
    "pca": ("Scale + PCA (30 PCs)", "`pp.scale`+`pp.pca`", "`scale`+`pca(arpack)`", "`ScaleData`+`RunPCA`"),
    "neighbors": ("k-NN graph (k = 20)", "`find_neighbors` (exact)", "`neighbors` (pynndescent)", "`FindNeighbors` (Annoy)"),
    "cluster": ("k-NN + clustering", "`find_neighbors`+`cluster` (Leiden)", "`neighbors`+`leiden`", "`FindNeighbors`+`FindClusters` (Louvain)"),
    "markers": ("Markers, presto-style Wilcoxon + AUC (20k genes)", "`wilcoxauc`", "-", "`presto::wilcoxauc`"),
    "rank_genes": ("Markers, scanpy-style Wilcoxon (20k genes)", "`rank_genes_groups`", "`tl.rank_genes_groups(wilcoxon)`", "-"),
    "pseudobulk": ("Pseudobulk sums (sample x cell type)", "`pseudobulk`", "`get.aggregate`", "`AggregateExpression`"),
    "de_deseq2": ("Pseudobulk DESeq2 (paired, 16 samples)", "`de.deseq2`", "pydeseq2", "DESeq2"),
    "de_voom": ("Pseudobulk limma-voom + duplicateCorrelation", "`de.voom`", "-", "limma/edgeR"),
    "score_module": ("Module score (5 sets)", "`add_module_score`", "-", "`AddModuleScore`"),
    "score_genes": ("scanpy gene-set score (5 sets)", "`score_genes`", "`tl.score_genes`", "-"),
    "score_ucell": ("UCell score (5 sets)", "`score_ucell`", "-", "UCell"),
    "clones": ("Clone table + dynamics + STARTRAC", "`clones.*`", "-", "-"),
}


def _fmt_s(s):
    if s is None or not np.isfinite(s):
        return "-"
    return f"{s:.2f} s" if s < 10 else (f"{s:.0f} s" if s < 600 else f"{s / 60:.0f} min")


def _fmt_x(a, b):
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return "-"
    r = b / a
    return f"**{r:.0f}x**" if r >= 10 else (f"**{r:.1f}x**" if r >= 1.05 else f"{r:.2f}x")


def _diff(v):
    if v == 0:
        return "identical (max diff 0)"
    if v < 1e-10:
        return f"exact (max diff {v:.0e})"
    return f"max diff {v:.1e}"


def _agree(ag: pd.DataFrame, n: int, step: str, vs: str) -> str:
    a = ag[(ag.n == n) & (ag.step == step)]
    if vs == "scanpy":
        a = a[a.vs == "scanpy"]
    else:
        a = a[a.vs.astype(str).str.startswith("seurat")]
    m = dict(zip(a.metric, a.value))
    if not m:
        return "-"
    if step in ("normalize", "pseudobulk", "score_module", "score_genes", "score_ucell"):
        return _diff(m["max_abs_diff"]) if "max_abs_diff" in m else "-"
    if step == "hvg":
        return f"{int(m['hvg_shared_of_2000']):,}/2,000 genes shared; r = {m['var_std_r']:.6f}"
    if step == "pca":
        return f"per-PC abs r >= {m['pc_abs_r_min_top10']:.4f} (PC1-10), >= {m['pc_abs_r_min_all30']:.3f} (all 30)"
    if step == "neighbors":
        return f"{100 * m['knn_overlap_mean']:.1f}% of neighbours shared"
    if step == "cluster":
        return f"ARI {m['ari']:.3f}"
    if step == "rank_genes":
        return (f"z-score {_diff(m['score_max_abs_diff'])}; p max rel. diff {m['pval_max_rel_diff']:.0e}; "
                f"top-50 overlap {100 * m['top50_overlap_mean']:.0f}%") if "score_max_abs_diff" in m else "-"
    if step == "markers":
        return (f"AUC {_diff(m['auc_max_abs_diff'])}; p max rel. diff {m['pval_max_rel_diff']:.0e}"
                if "auc_max_abs_diff" in m else "-")
    if step == "de_deseq2":
        return (f"log2FC r = {m['log2fc_r']:.4f}; -log10 p r = {m['mlog10p_r']:.4f}; padj<0.1 Jaccard {m['padj0.1_jaccard']:.2f}"
                if "log2fc_r" in m else "-")
    if step == "de_voom":
        return f"t r = {m['t_r']:.6f}; max diff t {m['t_max_abs_diff']:.1e}, logFC {m['logfc_max_abs_diff']:.1e}"
    return "-"


def _dates(ts, env):
    if len(ts):
        import datetime as dt
        a, b = (dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d") for x in (ts.min(), ts.max()))
        return a if a == b else f"{a} to {b}"
    return env.get("date", "?")


def render(res: Path) -> str:
    s = pd.read_csv(res / "summary.csv")
    ag = pd.read_csv(res / "agreement.csv") if (res / "agreement.csv").exists() else pd.DataFrame(
        columns=["n", "step", "tool", "vs", "metric", "value"])
    env = json.loads((res / "environment.json").read_text()) if (res / "environment.json").exists() else {}
    allr = s.drop_duplicates(["n", "tool", "step"], keep="last")
    tt = {(r.n, r.tool, r.step): r for r in allr.itertuples()}
    sizes = sorted(allr.n.unique())

    def g(n, tool, step, f="seconds"):
        r = tt.get((n, tool, step))
        return None if r is None or r.ok != True else getattr(r, f)  # noqa: E712

    def status(n, tool, step, short=False):
        r = tt.get((n, tool, step))
        if r is None:
            if step not in IMPL[tool]:
                return "-"
            return "n/r" if short else "not run"
        if r.ok == True:  # noqa: E712
            return _fmt_s(r.seconds).replace(" s", "") if short else _fmt_s(r.seconds)
        err = str(getattr(r, "error", "") or "")
        if "timeout" in err:
            return "t/o" if short else f"timed out (> {err.split()[-2]} s)"
        if "did not finish" in err:
            return "dnf" if short else "did not finish"
        return "err" if short else "failed"

    def lower(n, tool, step, a):  # a timed-out run gives a lower bound on the speed-up
        r = tt.get((n, tool, step))
        err = str(getattr(r, "error", "") or "") if r is not None else ""
        if "timeout after" in err:
            lim = float(err.split("timeout after")[1].split()[0])
            return f"**> {lim / a:.0f}x**"
        return "-"

    def complete(n):  # every tool ran every step it implements, successfully
        rr = allr[allr.n == n]
        return len(rr) and rr.ok.all() and set(rr.tool) >= {"arsc", "scanpy", "seurat"}

    full = [n for n in sizes if complete(n)]
    mains = ([max(full)] if full else []) + [n for n in sizes[-1:] if not full or n > max(full)]
    L = []
    L.append(f"Synthetic data generated in code (`benchmarks/suite_sim.py`: 20,000 genes, negative-binomial counts, "
             f"~1,600 genes detected per cell, 10 cell types, 8 patients x 2 timepoints). Time = wall time of the "
             f"analysis call only; memory = peak resident memory of the process during that call (GB, includes the "
             f"loaded data). Shared inputs (HVGs, PCs, pseudobulk) are computed once so every step is compared on "
             f"identical input. Speed-up = the other tool's time / arsc's time.\n")
    for nh in mains:
        L.append(f"**{nh:,} cells**:\n")
        L.append("| Step | arsc | scanpy | Seurat / R | speed-up vs scanpy | speed-up vs Seurat | memory arsc / scanpy / Seurat | agreement with scanpy | agreement with Seurat / R |")
        L.append("|---|---:|---:|---:|---:|---:|---|---|---|")
        for step, lab in LABEL.items():
            a = g(nh, "arsc", step)
            if a is None:
                continue
            sc, se = g(nh, "scanpy", step), g(nh, "seurat", step)
            cells = [status(nh, tl, step) if (nh, tl, step) in tt else ("not run" if step in IMPL[tl] else "-")
                     for tl in ("arsc", "scanpy", "seurat")]
            mem = " / ".join("-" if g(nh, tl, step, "rss_peak_gb") is None else f"{g(nh, tl, step, 'rss_peak_gb'):.1f}"
                             for tl in ("arsc", "scanpy", "seurat"))
            L.append(f"| {lab[0]} | {cells[0]} | {cells[1]} | {cells[2]} | {_fmt_x(a, sc) if sc is not None else lower(nh, 'scanpy', step, a)} | {_fmt_x(a, se) if se is not None else lower(nh, 'seurat', step, a)} | {mem} | "
                     f"{_agree(ag, nh, step, 'scanpy')} | {_agree(ag, nh, step, 'seurat')} |")
        L.append("")
    if len(sizes) > 1:
        L.append("Scaling (seconds; `-` = no equivalent, `n/r` = not run (time budget), `t/o` = timed out, `dnf` = did not finish, `err` = failed):\n")
        L.append("| Step | " + " | ".join(f"{n // 1000:,}k arsc / scanpy / Seurat" for n in sizes) + " |")
        L.append("|---|" + "---|" * len(sizes))
        for step, lab in LABEL.items():
            L.append(f"| {lab[0]} | " + " | ".join(" / ".join(status(n, tl, step, True) for tl in ("arsc", "scanpy", "seurat"))
                                                     for n in sizes) + " |")
    rec = ag[(ag.metric == "knn_recall")].set_index("n").value.to_dict() if len(ag) else {}
    if rec:
        L.append("\nNotes: arsc's k-NN is exact (recall vs brute force on 1,000 random cells: "
                 + ", ".join(f"{v:.3f} at {int(n):,}" for n, v in sorted(rec.items()))
                 + "); scanpy (pynndescent) and Seurat (Annoy) are approximate, so their overlap with arsc is "
                   "their recall. Exact search is the slowest arsc step at 1M cells; "
                   "`arsc.knn(method=\"pynndescent\")` exists but is not benchmarked here. "
                   "Python steps are timed after an untimed warm-up run in the same process (JIT compilation "
                   "excluded; scanpy's first `neighbors` call in a fresh process spends 25-40 s compiling).")
    ts = pd.to_numeric(allr.get("t_start"), errors="coerce").dropna() if "t_start" in allr else pd.Series([], dtype=float)
    if env:
        pk = env.get("python_packages", {})
        rp = env.get("r_packages", {})
        L.append(f"\nMachine: {env.get('cpu', '?')}, {env.get('logical_cpus')} threads, {env.get('ram_gb')} GB RAM, "
                 f"{env.get('os', '?')}. arsc {pk.get('arsc')}, scanpy {pk.get('scanpy')}, anndata {pk.get('anndata')}, "
                 f"pydeseq2 {pk.get('pydeseq2')}, numpy {pk.get('numpy')}; {rp.get('R', 'R ?')}, Seurat {rp.get('Seurat')}, "
                 f"presto {rp.get('presto')}, DESeq2 {rp.get('DESeq2')}, limma {rp.get('limma')}, UCell {rp.get('UCell')}. "
                 f"Default threading for every tool (arsc and scanpy use all cores; most Seurat steps are single-threaded). "
                 f"Measured {_dates(ts, env)}; reproduce with `python benchmarks/run_suite.py` "
                 f"(details in `benchmarks/README.md`).")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results" / "suite"))
    ap.add_argument("--readme", default=str(HERE.parent / "README.md"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    md = render(Path(a.results))
    (Path(a.results) / "headline.md").write_text(md, encoding="utf-8")
    rd = Path(a.readme)
    txt = rd.read_text(encoding="utf-8") if rd.exists() else ""
    if a.print or START not in txt or END not in txt:
        print(md)
        if not a.print:
            print(f"[render_table] markers {START} / {END} not found in {rd}; README not changed")
        return
    i, j = txt.index(START) + len(START), txt.index(END)
    rd.write_text(txt[:i] + "\n" + md + txt[j:], encoding="utf-8")
    print(f"[render_table] updated {rd}")


if __name__ == "__main__":
    main()
