"""Paired arsc vs scanpy table (benchmarks/paired_bench.py results) -> README.md and SCANPY_PARITY.md.

    python benchmarks/render_paired.py [--sessions S1,S2,...]

Uses, for every (size, step), the most recent session in results/paired/runs.jsonl (or the sessions given), and
writes the table between ``<!-- PAIRED:START -->`` and ``<!-- PAIRED:END -->``. Each row gives the median and
range of the interleaved runs, the number of runs, the thread count, and the machine state seen before the runs.
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
LABEL = {"normalize": "Log-normalise", "hvg": "Highly variable genes (2,000)", "pca": "Scale + PCA (30 PCs)",
         "neighbors": "k-NN graph (k = 20)", "cluster": "k-NN + Leiden clustering", "rank_genes":
         "Markers, scanpy-style Wilcoxon (20k genes)", "pseudobulk": "Pseudobulk sums", "de_deseq2":
         "Pseudobulk DESeq2 (paired, 16 samples)", "score_genes": "scanpy gene-set score (5 sets)"}
ORDER = list(LABEL)


def fmt(s):
    return f"{s:.2f} s" if s < 10 else (f"{s:.0f} s" if s < 600 else f"{s / 60:.1f} min")


def table(sessions=None) -> str:
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
         "CPU leases (other jobs on the shared machine). Synthetic data of `benchmarks/suite_sim.py`.", ""]
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
            L.append(f"| {LABEL[step]} | {fmt(a.median())} ({fmt(a.min())}-{fmt(a.max())}) | {fmt(s.median())} "
                     f"({fmt(s.min())}-{fmt(s.max())}) | **{s.median() / a.median():.1f}x**; {worst:.1f}x | "
                     f"{len(a)} | {int(h['threads'].iloc[0])} | {h['cpu_percent_1s'].min():.0f}-"
                     f"{h['cpu_percent_1s'].max():.0f}% | {agree} |")
        lease = g["lease"].dropna().unique()
        L += ["", f"Measured {g['time'].min()} to {g['time'].max()}; leases seen: "
              + "; ".join(sorted(set(str(x) for x in lease))[:3]) + ".", ""]
    return "\n".join(L)


def inject(path: Path, text: str) -> None:
    s = path.read_text(encoding="utf-8")
    a, b = "<!-- PAIRED:START -->", "<!-- PAIRED:END -->"
    if a not in s:
        raise SystemExit(f"{path.name}: no {a} marker")
    s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
    path.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="")
    a = ap.parse_args()
    t = table([s for s in a.sessions.split(",") if s] or None)
    print(t)
    for f in ("README.md", "SCANPY_PARITY.md"):
        inject(ROOT / f, t)
