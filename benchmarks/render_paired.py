"""Paired arsc vs scanpy table (benchmarks/paired_bench.py results) -> SCANPY_PARITY.md (README.md: render_readme.py).

    python benchmarks/render_paired.py [--sessions S1,S2,...] [--note TEXT] [--label step=TEXT ...]
                                       [--timing N=results/paired/timing_<name>.json ...]

Uses, for every (size, step), the most recent session in results/paired/runs.jsonl (or the sessions given), and
writes the table between ``<!-- PAIRED:START -->`` and ``<!-- PAIRED:END -->``. Each row gives the median and
range of the interleaved runs, the number of runs, the thread count, and the machine state seen before the runs.
``--note`` replaces the sentence describing the data (default: the synthetic data), ``--label`` a step's row label
(e.g. the gene count of a real store), and ``--timing`` adds, per size, the other processes' CPU use recorded by the
framework's ``tools/timing_lock.py`` during the run (repeat it for a size whose steps ran in several parts; a part
that ran without the lock, pinned to 16 cores, says so).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = HERE / "results" / "paired"
LABEL = {"normalize": "Log-normalize", "hvg": "Highly variable genes (2,000)", "pca": "Scale + PCA (30 PCs)",
         "neighbors": "k-NN graph (k = 20)", "cluster": "k-NN + Leiden clustering", "rank_genes":
         "Markers, scanpy-style Wilcoxon (20k genes)", "pseudobulk": "Pseudobulk sums", "de_deseq2":
         "Pseudobulk DESeq2 (paired, 16 samples)", "score_genes": "scanpy gene-set score (5 sets)"}
ORDER = list(LABEL)
SIZES = (10_000, 100_000, 1_000_000)   # cell counts shown in SCANPY_PARITY.md; other sizes stay in results/


def fmt(s):
    return f"{s:.2f} s" if s < 10 else (f"{s:.0f} s" if s < 600 else f"{s / 60:.1f} min")


SYNTH_NOTE = "Synthetic data of `benchmarks/suite_sim.py`."


def table(sessions=None, note=SYNTH_NOTE, labels=None, timing=None) -> str:
    label = {**LABEL, **(labels or {})}
    runs = pd.DataFrame([json.loads(l) for l in (P / "runs.jsonl").read_text().splitlines() if l.strip()])
    runs = runs[runs["ok"] == True]  # noqa: E712
    runs = runs[runs["n"].isin(SIZES)]
    if sessions:
        runs = runs[runs["session"].isin(sessions)]
    last = runs.groupby(["n", "step"])["session"].max().rename("last").reset_index()
    runs = runs.merge(last, on=["n", "step"])
    runs = runs[runs["session"] == runs["last"]]
    qual = []
    for f in sorted(P.glob("quality_*.csv")):
        qual.append(pd.read_csv(f))
    qual = pd.concat(qual) if qual else pd.DataFrame()
    L = ["Same session, interleaved (arsc and scanpy alternate first), the same thread count for both; median "
         "(min-max) of the runs; machine state = system CPU use in the second before each run and the framework's "
         "CPU leases (other jobs on the shared machine). " + note, ""]
    for n in sorted(runs["n"].unique()):
        g = runs[runs["n"] == n]
        L += [f"**{n:,} cells**:", "",
              "| Step | arsc | scanpy | speed-up (median; slowest pair) | runs each | threads | CPU use before runs | agreement |",
              "|---|---:|---:|---|---:|---:|---|---|"]
        for step in [s for s in ORDER if s in set(g["step"])]:
            h = g[g["step"] == step]
            a, s = h[h.tool == "arsc"]["seconds"], h[h.tool == "scanpy"]["seconds"]
            if not len(a) or not len(s):
                continue
            pr = h.pivot_table(index="rep", columns="tool", values="seconds")
            worst = float((pr["scanpy"] / pr["arsc"]).min())
            agree = ""
            if len(qual) and step == "neighbors":
                q = qual[(qual["n"] == n) & qual.get("recall_arsc", pd.Series(dtype=float)).notna()]
                if len(q):
                    q = q.iloc[-1]
                    agree = (f"recall vs exact: arsc {q['recall_arsc']:.3f}, scanpy {q['recall_scanpy']:.3f}; overlap "
                             f"arsc-scanpy {q['overlap_arsc_scanpy']:.3f} vs scanpy seed 0-seed 1 "
                             f"{q['overlap_scanpy_seed0_seed1']:.3f}")
            if len(qual) and step == "cluster" and "ari_arsc_scanpy" in qual:
                q = qual[(qual["n"] == n) & qual["ari_arsc_scanpy"].notna()]
                if len(q):
                    q = q.iloc[-1]
                    agree = (f"ARI arsc-scanpy {q['ari_arsc_scanpy']:.3f}; vs truth arsc {q['ari_truth_arsc']:.3f}, "
                             f"scanpy {q['ari_truth_scanpy']:.3f}")
                    if pd.notna(q.get("ari_scanpy_seed0_seed1", np.nan)):
                        agree = (f"ARI arsc-scanpy {q['ari_arsc_scanpy']:.3f} vs scanpy seed 0-seed 1 "
                                 f"{q['ari_scanpy_seed0_seed1']:.3f} ({int(q['n_clusters_arsc'])} / "
                                 f"{int(q['n_clusters_scanpy'])} / {int(q['n_clusters_scanpy_seed1'])} clusters); "
                                 f"vs the cell-type labels arsc {q['ari_truth_arsc']:.3f}, scanpy "
                                 f"{q['ari_truth_scanpy']:.3f}")
            L.append(f"| {label[step]} | {fmt(a.median())} ({fmt(a.min())}-{fmt(a.max())}) | {fmt(s.median())} "
                     f"({fmt(s.min())}-{fmt(s.max())}) | **{s.median() / a.median():.1f}x**; {worst:.1f}x | "
                     f"{len(a)} | {int(h['threads'].iloc[0])} | {h['cpu_percent_1s'].min():.0f}-"
                     f"{h['cpu_percent_1s'].max():.0f}% | {agree} |")
        lease = g["lease"].dropna().unique()
        line = (f"Measured {g['time'].min()} to {g['time'].max()}; leases seen: "
                + "; ".join(sorted(set(str(x) for x in lease))[:3]) + ".")
        tfs = (timing or {}).get(int(n)) or []
        lines = [("* " if len(tfs) > 1 else "") + timing_line(json.loads(Path(tf).read_text(encoding="utf-8")), len(tfs) > 1)
                 for tf in tfs]
        L += [""] + (lines or [line]) + [""]
    return "\n".join(L)


def timing_line(t: dict, parts: bool = False) -> str:
    """One sentence on a run's conditions from its tools/timing_lock.py JSON (or a pinned run's JSON, lock false)."""
    st = steps_of(t.get("cmd", ""))
    what = f"Steps {st}: run" if parts and st else "Run"
    how = ("under the machine-wide timing lock" if t.get("lock", True) else
           "without the timing lock (it was refused three times), pinned to 16 cores while other work ran")
    return (f"{what} {t['start']} {how} ({t['threads']} threads, {t['physical_cores']} physical cores, "
            f"{t['wall_s'] / 60:.0f} min including data loading and warm-up): other processes used "
            f"{t['other_cpu_pct_mean']:.1f}% of the machine's CPU on average (at most {t['other_cpu_pct_max']:.1f}%; "
            f"{t['other_cores_mean']:.2f} cores on average), sampled every second ({t['n_samples']:,} samples).")


def steps_of(cmd: str) -> str:
    """The --steps of a paired_bench.py command line, readable (\"normalize, hvg, pca\"), or ''."""
    parts = cmd.split()
    return ", ".join(parts[parts.index("--steps") + 1].split(",")) if "--steps" in parts else ""


def inject(path: Path, text: str) -> None:
    s = path.read_text(encoding="utf-8")
    a, b = "<!-- PAIRED:START -->", "<!-- PAIRED:END -->"
    if a not in s:
        raise SystemExit(f"{path.name}: no {a} marker")
    s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
    path.write_text(s, encoding="utf-8", newline="\n")


def timings(specs: list[str]) -> dict:
    """['N=path', ...] -> {N: [path, ...]}, each JSON tagged with its steps (from its command line)."""
    out: dict = {}
    for x in specs:
        k, v = x.split("=", 1)
        out.setdefault(int(k), []).append(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="")
    ap.add_argument("--note", default=SYNTH_NOTE, help="the sentence describing the data")
    ap.add_argument("--label", action="append", default=[], help="step=row label (repeatable)")
    ap.add_argument("--timing", action="append", default=[], help="N=timing JSON of tools/timing_lock.py (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the table only")
    a = ap.parse_args()
    t = table([s for s in a.sessions.split(",") if s] or None, a.note,
              dict(x.split("=", 1) for x in a.label), timings(a.timing))
    print(t)
    for f in [] if a.dry_run else ("SCANPY_PARITY.md",):   # README.md: render_readme.py (compact tables)
        inject(ROOT / f, t)
