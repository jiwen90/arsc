"""The README's compact timing tables: arsc vs scanpy (paired_bench.py) and arsc vs Seurat (seurat_timing.py).

    python benchmarks/render_readme.py [--dry-run]

arsc vs Seurat runs are included only when they ran under the timing lock with other processes below
MAX_OTHER_LOAD; then the detailed tables also go between ``<!-- SEURAT_TIMING:START -->`` and
``<!-- SEURAT_TIMING:END -->`` in SEURAT_PARITY.md (when those markers are present).

One table per size, from the most recent real-store session of each (size, step) in results/paired/runs.jsonl and
from results/seurat/seurat_timing_n<size>.json (arsc, timed under the lock), joined with
results/seurat/seurat_timing_n<size>_seurat.json (Seurat timed apart on its one thread: seurat_timing.py --r-only)
when that exists (at 1M cells that file is written by seurat_impute_1m.py: measured steps plus imputed ones, each
marked ~value\* in the tables with a footnote); written between ``<!-- SPEED:START -->`` and ``<!-- SPEED:END -->``
in README.md. The full tables (ranges, machine state, agreement) are in SCANPY_PARITY.md and SEURAT_PARITY.md.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = HERE / "results" / "paired"
S = HERE / "results" / "seurat"

# row label, scanpy-path step (paired_bench), Seurat-path step (seurat_timing)
ROWS = [("Log-normalize", "normalize", "normalize"),
        ("Variable genes (2,000)", "hvg", "hvg"),
        ("Scale + PCA (30 PCs)", "pca", "pca"),
        ("Neighbor graph (k = 20)", "neighbors", "neighbors"),
        ("Clustering", "cluster", "cluster"),
        ("Markers, Wilcoxon, each cell type vs rest", "rank_genes", "markers"),
        ("Pseudobulk sums", "pseudobulk", "pseudobulk"),
        ("Pseudobulk DESeq2", "de_deseq2", None),
        ("Gene-set score (3 sets)", "score_genes", "score_module")]


def fmt(s):
    if s is None or not np.isfinite(s):
        return "-"
    if s >= 36000:
        return f"{s / 3600:.0f} h"
    return f"{s:.2f} s" if s < 10 else (f"{s:.0f} s" if s < 600 else f"{s / 60:.0f} min")


def imputed(se: dict, step: str):
    """The imputation record of a Seurat step that was not measured (seurat_impute_1m.py), else None."""
    d = (se.get("steps") or {}).get(step) or {}
    return d if d.get("status") == "imputed" else None


def imputed_note(se: dict) -> str:
    """One footnote for the imputed Seurat values of a run: what was measured, the method and the range."""
    parts = []
    mk = imputed(se, "markers")
    if mk:
        lo, hi = mk["range_95_s"]
        per = mk["per_type"]
        few, many = max(per.values(), key=lambda d: d["seconds"]), min(per.values(), key=lambda d: d["seconds"])
        parts.append(f"markers: `FindMarkers` (one thread) was timed for {mk['measured_types']} of the "
                     f"{mk['groups_total']} cell types, in that run and in separate one-thread R processes "
                     f"({mk['measured_sum_s'] / 3600:.1f} h in all); the other "
                     f"{mk['groups_total'] - mk['measured_types']} are counted at the mean measured time per type "
                     f"({fmt(mk['mean_per_type_s'])}), giving {mk['seconds'] / 3600:.1f} h (95% range "
                     f"{lo / 3600:.1f}-{hi / 3600:.1f} h, from resampling the measured per-type times). Time per type "
                     f"does not follow the number of cells in it: the slowest type ({few['cells']:,} cells) took "
                     f"{fmt(few['seconds'])}, the fastest ({many['cells']:,} cells) {fmt(many['seconds'])}")
    lin = [k for k in ("pseudobulk", "score_module") if imputed(se, k)]
    if lin:
        sc = [se["steps"][k]["scale_10k_to_100k"] for k in lin]
        names = " and ".join({"pseudobulk": "pseudobulk sums", "score_module": "gene-set score"}[k] for k in lin)
        parts.append(f"{names}: 10 times the 100,000-cell Seurat time; from 10,000 to 100,000 cells these grew "
                     f"{min(sc):.1f}-{max(sc):.1f}x, so the linear guess is likely an overestimate")
    if not parts:
        return ""
    return ("\\*Imputed, not measured: Seurat's one-thread run at this size was stopped during the markers step, so "
            "the pseudobulk and gene-set score steps never ran; its earlier steps are measured. " +". ".join(p[0].upper() + p[1:] for p in parts))


def ratio(a, b):
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return "-"
    r = b / a
    return f"{r:.1f}x" if r < 100 else f"{r:.0f}x"


def paired() -> dict:
    runs = pd.DataFrame([json.loads(l) for l in (P / "runs.jsonl").read_text().splitlines() if l.strip()])
    runs = runs[(runs["ok"] == True) & (runs.get("source") == "store")]  # noqa: E712
    last = runs.groupby(["n", "step"])["session"].max().rename("last").reset_index()
    runs = runs.merge(last, on=["n", "step"])
    runs = runs[runs["session"] == runs["last"]]
    med = runs.groupby(["n", "step", "tool"])["seconds"].agg(["median", "size"])
    out = {(int(n), st, tool): float(v) for (n, st, tool), v in med["median"].items()}
    out["_runs"] = {(int(n), st, tool): int(v) for (n, st, tool), v in med["size"].items()}
    return out


def complete_markers(r: dict) -> dict:
    """A budgeted marker run that finished every cell type is a measurement, not an estimate: its time is the sum."""
    se = r.get("seurat") or {}
    part = se.get("markers_partial")
    tot = se.get("markers_groups_total")
    if tot is None and se.get("markers_estimate_basis"):   # runs before the total was stored: "k of N cell types"
        m = re.match(r"\d+ of (\d+) cell types", se["markers_estimate_basis"])
        tot = int(m.group(1)) if m else None
    if part and tot is not None and len(part) == int(tot):
        se["times"]["markers"] = float(sum(d["seconds"] for d in part.values()))
        se.pop("markers_estimate_s", None)
        se["markers_summed"] = len(part)
    return r


SIZES = (10_000, 100_000, 1_000_000)   # cell counts shown in the README; other sizes stay in results/

MAX_OTHER_LOAD = 10.0   # % of the machine's CPU used by other processes (mean) above which a run is not published (the
# paired arsc vs scanpy tables publish runs up to this load too; the load is printed with every table)


def tag(n: int) -> str:
    return f"{n // 1_000_000}m" if n >= 1_000_000 else f"{n // 1000}k"


def other_load(f: Path):
    """Mean CPU use of other processes during a locked run (tools/timing_lock.py JSON), or None."""
    return json.loads(f.read_text()).get("other_cpu_pct_mean") if f.exists() else None


def seurat(n: int):
    """The arsc vs Seurat run at this size, only if it ran under the timing lock with low load from other processes."""
    f = S / f"seurat_timing_n{n}.json"
    load = other_load(S / f"timing_arsc-seurat-{tag(n)}.json")
    if not f.exists() or load is None or load > MAX_OTHER_LOAD:
        return None
    r = json.loads(f.read_text())
    fs = S / f"seurat_timing_n{n}_seurat.json"   # Seurat timed apart (--r-only), on its one thread
    if fs.exists():
        rs = json.loads(fs.read_text())
        r.update({"seurat": rs["seurat"], "agreement": rs["agreement"], "seurat_apart": True,
                  "seurat_time": rs["time"], "machine_load_during_r": rs.get("machine_load_during_r")})
    r = complete_markers(r)
    sh = sharded_markers(n)
    if sh and "seurat" in r:   # FindAllMarkers split over single-threaded processes: the sum is the one-thread total
        r["seurat"]["times"]["markers"] = sh["seconds"]
        r["seurat"]["markers_sharded"] = sh
    return r


def sharded_markers(n: int):
    """seurat_markers_shard.py results at this size, if one complete set of shards covers every cell type."""
    for k in sorted({int(f.stem.rsplit("of", 1)[1]) for f in S.glob(f"seurat_markers_n{n}_shard*of*.json")}):
        fs = sorted(S.glob(f"seurat_markers_n{n}_shard*of{k}.json"))
        rs = [json.loads(f.read_text()) for f in fs]
        groups = {g for x in rs for g in x["markers"]}
        if len(rs) != k or len(groups) != rs[0]["groups_total"]:
            continue
        return {"shards": k, "seconds": float(sum(d["seconds"] for x in rs for d in x["markers"].values())),
                "groups": len(groups), "wall_s": max(x["process_wall_s"] for x in rs),
                "load_mean": float(np.mean([x["machine_load_during_r"]["cpu_pct_mean"] for x in rs])),
                "rows": int(sum(d["rows"] for x in rs for d in x["markers"].values())),
                "time": min(x["time"] for x in rs)}
    return None


def dnf(se: dict):
    """A budgeted marker run stopped before every cell type: (seconds used, cell types done, total); else None."""
    part = se.get("markers_partial")
    if not part or se["times"].get("markers") is not None:
        return None
    used = se["times"].get("markers_partial_s") or sum(d["seconds"] for d in part.values())
    return float(used), len(part), int(se.get("markers_groups_total") or 0)


def paired_pinned(n: int) -> bool:
    """True if part of the arsc vs scanpy run at this size ran without the timing lock (pinned to 16 cores)."""
    return any(json.loads(f.read_text()).get("lock") is False for f in P.glob(f"timing_arsc-scanpy-{tag(n)}*.json"))


def table(n: int, pr: dict) -> list[str]:
    sr = seurat(n)
    has_s = sr is not None and "seurat" in sr
    head = "| Step | arsc | scanpy | speed-up |" + (" arsc | Seurat | speed-up |" if has_s else "")
    sep = "|---|---:|---:|---:|" + ("---:|---:|---:|" if has_s else "")
    L = [head, sep]
    for label, sp, ss in ROWS:
        a, b = pr.get((n, sp, "arsc")), pr.get((n, sp, "scanpy"))
        row = f"| {label} | {fmt(a)} | {fmt(b)} | {ratio(a, b)} |"
        if has_s:
            if ss is None:
                row += " - | - | - |"
            else:
                sa = float(np.median(sr["arsc"]["times"][ss]))
                v = sr["seurat"]["times"].get(ss)
                d = dnf(sr["seurat"]) if ss == "markers" else None
                if d:
                    row += f" {fmt(sa)} | did not finish (> {fmt(d[0])})\\* | > {ratio(sa, d[0])} |"
                elif imputed(sr["seurat"], ss):
                    row += f" {fmt(sa)} | ~{fmt(v)}\\* | ~{ratio(sa, v)} |"
                else:
                    row += f" {fmt(sa)} | {fmt(v)} | {ratio(sa, v)} |"
        L.append(row)
    return L


def render() -> str:
    pr = paired()
    out = []
    for n in sorted({k[0] for k in pr if k != "_runs"} & set(SIZES)):
        title = {10_000: "10,000 cells", 100_000: "100,000 cells", 1_000_000: "1,000,000 cells"}.get(n, f"{n:,} cells")
        sr = seurat(n)
        out += [f"#### {title}", ""] + table(n, pr) + [""]
        cond = (f"16 cores; arsc and scanpy at 16 threads in one process, interleaved, median of 3 warm runs "
                f"(numba compilation excluded)")
        few = [f"{lab.split(',')[0].lower()} ran {k} time{'s' if k > 1 else ''} per tool"
               for lab, sp, _ in ROWS
               if (k := min(pr["_runs"].get((n, sp, t), 3) for t in ("arsc", "scanpy"))) < 3]
        if few:
            cond += f" ({'; '.join(few)})"
        missing = [lab.split(",")[0].lower() for lab, sp, _ in ROWS
                   if (n, sp, "arsc") not in pr or (n, sp, "scanpy") not in pr]
        if missing:
            cond += f"; not yet timed against scanpy at this size: {', '.join(missing)}"
        if paired_pinned(n):
            cond += ("; some steps ran without the timing lock (refused three times), pinned to 16 cores while "
                     "other work ran (SCANPY_PARITY.md gives the load)")
        if sr is not None and "seurat" in sr:
            th = sr["seurat"]["threads"]
            cond += (f". arsc vs Seurat: a separate session, arsc median of {sr['repeats_arsc']} warm runs at "
                     f"{sr['threads_arsc']} threads; Seurat 4.3 one run after an untimed warm-up, on "
                     f"{th.get('blas_threads')} thread (R's reference BLAS, `future` with "
                     f"{th.get('future_workers')} worker: Seurat's default)")
            ml = sr.get("machine_load_during_r")
            if sr.get("seurat_apart") and ml:
                cond += (f", timed apart from arsc as an ordinary job beside other work (machine CPU use "
                         f"{ml['cpu_pct_mean']:.0f}% on average while it ran)")
            if sr["seurat"].get("markers_summed"):
                cond += (f"; Seurat's markers ran as `FindMarkers` for each of the {sr['seurat']['markers_summed']} "
                         f"cell types, what `FindAllMarkers` runs, summed")
            sh = sr["seurat"].get("markers_sharded")
            if sh:
                cond += (f"; Seurat's markers are the sum of `FindMarkers` for each of the {sh['groups']} cell types "
                         f"(what `FindAllMarkers` runs), each on one thread, split over {sh['shards']} R processes "
                         f"run side by side ({fmt(sh['wall_s'])} of wall time)")
            d = dnf(sr["seurat"])
            if d:
                cond += (f". \\*Seurat's markers did not finish: `FindAllMarkers` runs `FindMarkers` for each cell "
                         f"type, and in {fmt(d[0])} it completed {d[1]} of {d[2]} cell types (seeded random order); "
                         f"no estimate of the total is given")
            if imputed_note(sr["seurat"]):
                cond += ". " + imputed_note(sr["seurat"])

        out += [cond + ".", ""]
    return "\n".join(out).rstrip() + "\n"


SEURAT_CALL = {"normalize": "`NormalizeData`", "hvg": "`FindVariableFeatures` (vst, 2,000)",
               "pca": "`ScaleData` + `RunPCA` (30)", "neighbors": "`FindNeighbors` (k = 20)",
               "cluster": "`FindClusters` (Louvain, resolution 0.6)", "markers": "`FindAllMarkers` (Wilcoxon)",
               "pseudobulk": "`AggregateExpression` (sample x cell type)",
               "score_module": "`AddModuleScore` (3 sets of 30 genes)"}
ARSC_CALL = {"normalize": "`pp.log_normalize`", "hvg": "`pp.hvg_vst`", "pca": "`pp.scale` + `pp.pca`",
             "neighbors": "`find_neighbors`", "cluster": "`cluster`", "markers": "`find_all_markers`",
             "pseudobulk": "`pseudobulk`", "score_module": "`add_module_score`"}


def seurat_detail() -> str:
    out = []
    arsc_files = [f for f in S.glob("seurat_timing_n*.json") if f.stem.split("_n")[-1].isdigit()]
    for f in sorted(arsc_files, key=lambda f: int(f.stem.split("_n")[-1])):
        r = seurat(int(f.stem.split("_n")[-1]))
        if r is None or "seurat" not in r:
            continue
        se, ag, th = r["seurat"], r["agreement"], r["seurat"]["threads"]
        out += [f"**{r['n']:,} cells** (run {r['time']}):", "",
                "| Seurat call | arsc call | arsc, median (min-max) | Seurat | speed-up |", "|---|---|---:|---:|---:|"]
        for step in SEURAT_CALL:
            v = np.asarray(r["arsc"]["times"][step], float)
            sv = se["times"].get(step)
            d = dnf(se) if step == "markers" else None
            if d:
                sc_txt, sp_txt = f"did not finish (> {fmt(d[0])})\\*", f"> {ratio(float(np.median(v)), d[0])}"
            elif imputed(se, step):
                sc_txt, sp_txt = f"~{fmt(sv)}\\*", f"~{ratio(float(np.median(v)), sv)}"
            else:
                sc_txt, sp_txt = fmt(sv), ratio(float(np.median(v)), sv)
            out.append(f"| {SEURAT_CALL[step]} | {ARSC_CALL[step]} | {fmt(float(np.median(v)))} "
                       f"({fmt(v.min())}-{fmt(v.max())}) | {sc_txt} | {sp_txt} |")
        out.append("")
        tl = json.loads((S / f"timing_arsc-seurat-{tag(r['n'])}.json").read_text())
        out += [f"arsc ran {tl['start']} under the machine-wide timing lock ({tl['threads']} threads, "
                f"{tl['wall_s'] / 60:.0f} min including data loading and warm-up); other processes used "
                f"{tl['other_cpu_pct_mean']:.1f}% of the machine's CPU on average (at most "
                f"{tl['other_cpu_pct_max']:.0f}%).", ""]
        if se.get("markers_summed"):
            out += [f"Seurat's markers ran as `FindMarkers` for each of the {se['markers_summed']} cell types in turn "
                    f"(what `FindAllMarkers` runs, here with a time budget that turned out not to bind); the time is "
                    f"the sum.", ""]
        sh = se.get("markers_sharded")
        if sh:
            out += [f"Seurat's markers: `FindAllMarkers` is `FindMarkers` for each cell type in turn, which at this "
                    f"size takes more than a day in one process, so `benchmarks/seurat_markers_shard.py` dealt the "
                    f"{sh['groups']} cell types to {sh['shards']} single-threaded R processes run side by side "
                    f"(started {sh['time']}; {fmt(sh['wall_s'])} of wall time; machine CPU use {sh['load_mean']:.0f}% "
                    f"on average). The time in the table is the sum of the {sh['groups']} timed `FindMarkers` calls "
                    f"({sh['rows']:,} rows at p < 0.01): the one-thread total, measured, not extrapolated. The "
                    f"agreement below uses the budgeted single-process run.", ""]
        if dnf(se) and not sh:
            part = se["markers_partial"]
            per = ", ".join(f"{g} ({d['cells']:,} cells) {fmt(d['seconds'])}" for g, d in part.items())
            out += [f"\\*`FindAllMarkers` did not finish within its time budget. It is `FindMarkers` for each cell "
                    f"type in turn, so the cell types were run one by one in a seeded random order until "
                    f"{fmt(se['times'].get('markers_partial_s'))} had passed: {per} ({len(part)} of "
                    f"{se.get('markers_groups_total')} cell types). No estimate of the total is given.", ""]
        if imputed_note(se):
            out += [imputed_note(se) + ". The measured and imputed values, the per-type marker times and the method "
                    f"are in `benchmarks/results/seurat/seurat_timing_n{r['n']}_seurat.json` "
                    f"(written by `benchmarks/seurat_impute_1m.py`).", ""]
        if r.get("seurat_apart") and r.get("machine_load_during_r"):
            ml = r.get("machine_load_during_r") or {}
            out += [f"Seurat ran apart from arsc (started {r['seurat_time']}), as an ordinary job on its one thread "
                    f"beside other work: the machine's CPU use averaged {ml.get('cpu_pct_mean', float('nan')):.0f}% "
                    f"(at most {ml.get('cpu_pct_max', float('nan')):.0f}%) while it ran. arsc's times come from a "
                    f"separate run under the timing lock.", ""]
        elif r.get("seurat_apart"):
            out += [f"Seurat ran apart from arsc (started {r['seurat_time']}), as an ordinary job on its one thread "
                    f"beside other work. arsc's times come from a separate run under the timing lock.", ""]
        ct = f", `CreateSeuratObject` {fmt(se['create_object_s'])}" if "create_object_s" in se else ""
        wu = f" {fmt(se['warmup_s'])}" if "warmup_s" in se else ""
        out += [f"Not timed: R package load {fmt(se['package_load_s'])}, reading the cells from the store "
                f"{fmt(se['read_store_s'])}{ct}, warm-up on {se['warm_n']:,} cells{wu}; arsc import {fmt(r['arsc']['import_s'])}, warm-up "
                f"(numba compilation) on {min(12000, r['n']):,} cells {fmt(r['arsc']['warmup_s'])}. Seurat threads: "
                f"BLAS {th.get('blas_threads')} (R's reference BLAS), `future` workers {th.get('future_workers')}; "
                f"arsc {r['threads_arsc']} threads.", ""]
        if not ag:
            out += ["Agreement: not computed at this size (Seurat's run did not finish).", ""]
            continue
        mk = ag.get("markers_cell_types_compared")
        out += ["Agreement (each tool on its own chain):", "",
                f"* Variable genes: {ag['hvg_shared']:,} of 2,000 shared.",
                f"* Clusters: ARI arsc-Seurat {ag['ari_arsc_seurat']:.3f}; Seurat seed 0-seed 1 "
                f"{ag['ari_seurat_seed0_seed1']:.3f}; arsc seed 0-seed 1 {ag['ari_arsc_seed0_seed1']:.3f} "
                f"({ag['n_clusters_arsc']} / {ag['n_clusters_seurat']} / {ag['n_clusters_seurat_seed1']} clusters: "
                f"arsc / Seurat / Seurat seed 1); against the cell-type labels arsc {ag['ari_arsc_celltype']:.3f}, "
                f"Seurat {ag['ari_seurat_celltype']:.3f}.",
                f"* Markers{f' ({mk} cell types Seurat finished)' if mk else ''}: {ag['markers_rows_seurat']:,} "
                f"Seurat rows, {ag['markers_rows_arsc']:,} arsc rows, Jaccard of (cell type, gene) "
                f"{ag['markers_rows_jaccard']:.3f}; avg_log2FC max diff {ag['markers_log2fc_maxdiff']:.1e}, p max "
                f"rel. diff {ag['markers_p_max_rel_diff']:.1e}; " + (
                    f"top-10 positive markers per cell type shared {100 * ag['markers_top10_overlap_mean']:.0f}% on "
                    f"average (min {100 * ag['markers_top10_overlap_min']:.0f}%)."
                    if not (se.get("markers_partial") and not se.get("markers_loop_sorted")) else
                    "top-10 not compared: this run's per-cell-type loop kept `FindMarkers`' tie order instead of "
                    "`FindAllMarkers`' re-sort (see the note below the tables)."),
                f"* Pseudobulk: {ag['pseudobulk_groups_seurat']:,} groups each, total counts "
                f"{'identical' if ag['pseudobulk_total_arsc'] == ag['pseudobulk_total_seurat'] else 'differ'}.",
                f"* Module scores: max diff {ag['score_module_maxdiff_first_cells']:.1e} on the first "
                f"{ag['score_module_cells_compared']:,} cells.", ""]
    return "\n".join(out).rstrip() + "\n"


def inject(path: Path, text: str, a="<!-- SPEED:START -->", b="<!-- SPEED:END -->") -> None:
    s = path.read_text(encoding="utf-8")
    if a not in s:
        raise SystemExit(f"{path.name}: no {a} marker")
    s = s[: s.index(a) + len(a)] + "\n" + text + s[s.index(b):]
    path.write_text(s, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t = render()
    print(t)
    d = seurat_detail()
    print(d)
    if not a.dry_run:
        inject(ROOT / "README.md", t)
        if d.strip() and "<!-- SEURAT_TIMING:START -->" in (ROOT / "SEURAT_PARITY.md").read_text(encoding="utf-8"):
            inject(ROOT / "SEURAT_PARITY.md", d, "<!-- SEURAT_TIMING:START -->", "<!-- SEURAT_TIMING:END -->")
