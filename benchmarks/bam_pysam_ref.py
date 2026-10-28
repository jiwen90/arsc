"""pysam reference for arsc.bam parity (BAM_PARITY.md). Runs where pysam is installed (Linux / a container).

    python bam_pysam_ref.py count   <bam> <threads>                      -> TSV ref, n (every record, file order)
    python bam_pysam_ref.py region  <bam> <region> [tags]                -> TSV, one row per record (fetch)
    python bam_pysam_ref.py pileup  <bam> <threads> < input              -> TSV site, barcode, A, C, G, T, N
    python bam_pysam_ref.py coverage <bam> <region>                      -> TSV pos, A, C, G, T (count_coverage)

pileup reads stdin: a line "BARCODES", one barcode per line, a line "SITES", then "chrom<TAB>pos" (1-based) lines.
Its rules are the ones arsc.bam.pileup_cells documents (defaults): flags 0x304 excluded, MAPQ >= 20, aligned length
(M/=/X) >= 30, base quality >= 20, the site in an aligned block (pysam pileup: not is_del / is_refskip), CB in the
list, UB present; one base per (site, cell, UMI) by majority, ties dropped. The pileup engine is htslib's
(``AlignmentFile.pileup`` with stepper "nofilter", no depth cap, overlaps and orphans kept, so pysam applies no
filter of its own). Timing goes to stderr.
"""
import sys
import time
from collections import Counter, defaultdict

import pysam


def count(bam, threads):
    t = time.time()
    n = Counter()
    with pysam.AlignmentFile(bam, threads=int(threads)) as f:
        refs = list(f.references)
        for r in f.fetch(until_eof=True):
            n[r.reference_id] += 1
    print("ref\tn")
    for i, name in enumerate(refs):
        print(f"{name}\t{n.get(i, 0)}")
    print(f"*\t{n.get(-1, 0)}")
    print(f"pysam count {time.time() - t:.1f} s", file=sys.stderr)


def region(bam, reg, tags="CB,UB,xf,RE"):
    tags = tags.split(",")
    with pysam.AlignmentFile(bam) as f:
        print("\t".join(["name", "flag", "ref_id", "pos", "end", "mapq", "cigar", "next_ref_id", "next_pos", "tlen",
                         "seq", "qual"] + tags))
        for r in f.fetch(region=reg):
            q = r.query_qualities
            vals = [r.query_name, r.flag, r.reference_id, r.reference_start,
                    r.reference_end if r.reference_end is not None else r.reference_start + 1, r.mapping_quality,
                    r.cigarstring or "*", r.next_reference_id, r.next_reference_start, r.template_length,
                    r.query_sequence or "*", ",".join(map(str, q)) if q is not None else ""]
            vals += [r.get_tag(t) if r.has_tag(t) else "" for t in tags]
            print("\t".join(map(str, vals)))


def pileup(bam, threads):
    lines = sys.stdin.read().split("\n")
    i = lines.index("BARCODES")
    j = lines.index("SITES")
    bcs = set(x for x in lines[i + 1:j] if x)
    sites = [(x.split("\t")[0], int(x.split("\t")[1])) for x in lines[j + 1:] if x]
    t = time.time()
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    print("site\tbarcode\tA\tC\tG\tT\tN")
    ties = 0
    with pysam.AlignmentFile(bam, threads=int(threads)) as f:
        for si, (chrom, pos) in enumerate(sites):
            if chrom not in f.references:
                continue
            obs = defaultdict(lambda: [0, 0, 0, 0, 0])
            for col in f.pileup(chrom, pos - 1, pos, truncate=True, stepper="nofilter", max_depth=10 ** 9,
                                min_base_quality=0, ignore_overlaps=False, ignore_orphans=False):
                for pr in col.pileups:
                    if pr.is_del or pr.is_refskip:
                        continue
                    a = pr.alignment
                    if a.flag & 0x304 or a.mapping_quality < 20:
                        continue
                    if sum(n for op, n in a.cigartuples if op in (0, 7, 8)) < 30:
                        continue
                    qp = pr.query_position
                    if a.query_qualities[qp] < 20:
                        continue
                    if not a.has_tag("CB") or not a.has_tag("UB"):
                        continue
                    cb = a.get_tag("CB")
                    if cb not in bcs:
                        continue
                    obs[(cb, a.get_tag("UB"))][code.get(a.query_sequence[qp], 4)] += 1
            per_cell = defaultdict(lambda: [0, 0, 0, 0, 0])
            for (cb, ub), c in obs.items():
                m = max(c)
                if c.count(m) > 1:
                    ties += 1
                    continue
                per_cell[cb][c.index(m)] += 1
            for cb in sorted(per_cell):
                print(si, cb, *per_cell[cb], sep="\t")
    print(f"pysam pileup {time.time() - t:.1f} s, {len(sites)} sites, umi ties {ties}", file=sys.stderr)


def coverage(bam, reg):
    chrom, rest = reg.split(":")
    s, e = (int(x.replace(",", "")) for x in rest.split("-"))
    with pysam.AlignmentFile(bam) as f:
        def keep(r):
            return not (r.flag & 0x704) and r.mapping_quality >= 20
        cov = f.count_coverage(chrom, s - 1, e, quality_threshold=20, read_callback=keep)
    print("pos\tA\tC\tG\tT")
    for k in range(e - s + 1):
        a, c, g, t = (cov[b][k] for b in range(4))
        if a + c + g + t:
            print(s + k, a, c, g, t, sep="\t")


if __name__ == "__main__":
    {"count": count, "region": region, "pileup": pileup, "coverage": coverage}[sys.argv[1]](*sys.argv[2:])
