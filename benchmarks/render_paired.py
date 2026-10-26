"""Paired arsc vs scanpy table (benchmarks/paired_bench.py results) -> README.md and SCANPY_PARITY.md.

    python benchmarks/render_paired.py [--sessions S1,S2,...] [--note TEXT] [--label step=TEXT ...]
                                       [--timing N=results/paired/timing_<name>.json ...]

Uses, for every (size, step), the most recent session in results/paired/runs.jsonl (or the sessions given), and
writes the table between ``<!-- PAIRED:START -->`` and ``<!-- PAIRED:END -->``. Each row gives the median and
range of the interleaved runs, the number of runs, the thread count, and the machine state seen before the runs.
``--note`` replaces the sentence describing the data (default: the synthetic data), ``--label`` a step's row label
(e.g. the gene count of a real store), and ``--timing`` adds, per size, the other processes' CPU use recorded by the
framework's ``tools/timing_lock.py`` during the run.
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


def fmt(s):
    return f"{s:.2f} s" if s < 10 else (f"{s:.0f} s" if s < 600 else f"{s / 60:.1f} min")


SYNTH_NOTE = "Synthetic data of `benchmarks/suite_sim.py`."


def table(sessions=None, note=SYNTH_NOTE, labels=None, timing=None) -> str:
    label = {**LABEL, **(labels or {})}
    runs = pd.DataFrame([json.loads(l) for l in (P / "runs.jsonl").read_text().splitlines() if l.strip()])
    runs = runs[runs["ok"] == True]  # noqa: E712
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
        tf = (timing or {}).get(int(n))
        if tf:
            t = json.loads(Path(tf).read_text(encoding="utf-8"))
            line = (f"Run {t['start']} under the machine-wide timing lock ({t['threads']} threads, "
                    f"{t['physical_cores']} physical cores, {t['wall_s'] / 60:.0f} min including data loading and "
                    f"warm-up): other processes used {t['other_cpu_pct_mean']:.1f}% of the machine's CPU on average "
                    f"(at most {t['other_cpu_pct_max']:.1f}%; {t['other_cores_mean']:.2f} cores on average), "
                    f"sampled every second ({t['n_samples']:,} samples).")
        L += ["", line, ""]
    return "\n".join(L)


def inject(path: Path, text: str) -> None:
    s = path.read_text(encoding="utf-8")
    a, b = "<!-- PAIRED:START -->", "<!-- PAIRED:END -->"
    if a not in s:
        raise SystemExit(f"{path.name}: no {a} marker")
    s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
    path.write_text(s, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="")
    ap.add_argument("--note", default=SYNTH_NOTE, help="the sentence describing the data")
    ap.add_argument("--label", action="append", default=[], help="step=row label (repeatable)")
    ap.add_argument("--timing", action="append", default=[], help="N=timing JSON of tools/timing_lock.py (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the table only")
    a = ap.parse_args()
    t = table([s for s in a.sessions.split(",") if s] or None, a.note,
              dict(x.split("=", 1) for x in a.label), {int(k): v for k, v in (x.split("=", 1) for x in a.timing)})
    print(t)
    for f in [] if a.dry_run else ("README.md", "SCANPY_PARITY.md"):
        inject(ROOT / f, t)
