"""arsc.bam side of the pysam parity (BAM_PARITY.md); compares with bam_pysam_ref.py outputs.

    python bam_parity.py count    <bam> <out.tsv>                       every record per reference, timed
    python bam_parity.py region   <bam> <region> <out.tsv> [ref.tsv]    fetch, field by field vs pysam
    python bam_parity.py pileup   <bam> <sites.tsv> <barcodes> <out.tsv> [ref.tsv]   per-cell counts vs pysam
    python bam_parity.py coverage <bam> <region> <out.tsv> [ref.tsv]    pooled base counts vs pysam count_coverage
    python bam_parity.py input    <sites.tsv> <barcodes> <out>          the stdin file of bam_pysam_ref.py pileup

sites.tsv: columns chrom, pos (1-based). barcodes: a text file, one barcode per line (a Cell Ranger
``*_barcodes.csv`` "genome,barcode" line is accepted). Threads: AR_THREADS.
"""
import sys
import time

import numpy as np
import pandas as pd

import arsc


def _barcodes(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(line.split(",")[-1])
    return out


def count(bam, out):
    t = time.time()
    with arsc.bam.BamFile(bam) as f:
        s = f.count_per_reference()
        th, infl = f.threads, f.inflater_name
    dt = time.time() - t
    s.rename_axis("ref").rename("n").to_csv(out, sep="\t")
    print(f"arsc count {dt:.1f} s ({th} threads, {infl}): {int(s.sum())} records", file=sys.stderr)


def region(bam, reg, out, ref=None):
    tags = ("CB", "UB", "xf", "RE")
    t = time.time()
    with arsc.bam.BamFile(bam) as f:
        df = f.fetch(reg, tags=tags).to_frame()
    print(f"arsc fetch {time.time() - t:.2f} s, {len(df)} records", file=sys.stderr)
    df["qual"] = [",".join(map(str, q)) if q and q[0] != 255 else "" for q in df["qual"]]
    for tg in tags:
        df[tg] = ["" if v is None else v for v in df[tg]]
    df.to_csv(out, sep="\t", index=False)
    if ref:
        a = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
        b = pd.read_csv(ref, sep="\t", dtype=str, keep_default_na=False)
        print(f"records: arsc {len(a)}, pysam {len(b)}")
        if len(a) == len(b):
            for c in b.columns:
                n = int((a[c] != b[c]).sum())
                print(f"  {c:<12} {'identical' if n == 0 else f'{n} differ'}")


def pileup(bam, sites, barcodes, out, ref=None):
    s = pd.read_csv(sites, sep="\t")
    bcs = _barcodes(barcodes)
    t = time.time()
    ac = arsc.bam.pileup_cells(bam, s[["chrom", "pos"]], bcs)
    dt = time.time() - t
    print(f"arsc pileup {dt:.1f} s, {len(s)} sites, {len(bcs)} barcodes; stats {ac.stats}", file=sys.stderr)
    lg = ac.long()[["site", "barcode", "A", "C", "G", "T", "N"]]
    lg.to_csv(out, sep="\t", index=False)
    if ref:
        b = pd.read_csv(ref, sep="\t")
        m = lg.merge(b, on=["site", "barcode"], how="outer", suffixes=("_arsc", "_pysam"), indicator=True)
        print(f"(site, cell) pairs: arsc {len(lg)}, pysam {len(b)}, only arsc {(m._merge == 'left_only').sum()}, "
              f"only pysam {(m._merge == 'right_only').sum()}")
        both = m[m._merge == "both"]
        for x in "ACGTN":
            d = int((both[f"{x}_arsc"] != both[f"{x}_pysam"]).sum())
            print(f"  {x}: {d} pairs differ; total UMIs arsc {int(lg[x].sum())}, pysam {int(b[x].sum())}")


def coverage(bam, reg, out, ref=None):
    chrom, rest = reg.split(":")
    s, e = (int(x.replace(",", "")) for x in rest.split("-"))
    t = time.time()
    d = arsc.bam.discover_snps(bam, regions=[(chrom, s - 1, e)], min_depth=1, min_alt=0, min_maf=0.0, min_len=0)
    print(f"arsc coverage {time.time() - t:.1f} s", file=sys.stderr)
    d = d[d.depth > 0][["pos", "A", "C", "G", "T"]]
    d.to_csv(out, sep="\t", index=False)
    if ref:
        b = pd.read_csv(ref, sep="\t")
        m = d.merge(b, on="pos", how="outer", suffixes=("_arsc", "_pysam")).fillna(0)
        diff = sum(int((m[f"{x}_arsc"] != m[f"{x}_pysam"]).sum()) for x in "ACGT")
        print(f"covered positions: arsc {len(d)}, pysam {len(b)}; differing base counts {diff}; "
              f"bases arsc {int(d[list('ACGT')].to_numpy().sum())}, pysam {int(b[list('ACGT')].to_numpy().sum())}")


def make_input(sites, barcodes, out):
    s = pd.read_csv(sites, sep="\t")
    with open(out, "w", newline="\n") as fh:
        fh.write("BARCODES\n" + "\n".join(_barcodes(barcodes)) + "\nSITES\n")
        fh.write("\n".join(f"{c}\t{p}" for c, p in zip(s.chrom, s.pos)) + "\n")


if __name__ == "__main__":
    {"count": count, "region": region, "pileup": pileup, "coverage": coverage, "input": make_input}[sys.argv[1]](
        *sys.argv[2:])
