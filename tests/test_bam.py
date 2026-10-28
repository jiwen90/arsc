"""arsc.bam: BAM reader, per-cell pileup, de-novo SNPs, genotype comparison (synthetic BAMs written by
arsc.bam.write_bam; the pysam parity on a real 10x BAM is in BAM_PARITY.md).

The pileup is checked against a slow, independent pure-Python pileup that walks each read's SAM CIGAR string.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

import arsc
from arsc import bam as B


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return arsc.datasets.toy_bam(tmp_path_factory.mktemp("bam") / "toy.bam")


@pytest.fixture(scope="module")
def frame(toy):
    with B.BamFile(toy["path"]) as f:
        return f.fetch(tags=("CB", "UB", "xf", "RE")).to_frame()


# ------------------------------------------------------------------------------------------ slow reference
def _aligned_len(cigar):
    return sum(int(n) for n, o in re.findall(r"(\d+)([MIDNSHP=X])", cigar) if o in "M=X")


def _base_at(row, pos0):
    """(base, qual) at reference position pos0, or None when not in an aligned block."""
    rp, qp = row.pos, 0
    for n, o in re.findall(r"(\d+)([MIDNSHP=X])", row.cigar):
        n = int(n)
        if o in "M=X":
            if rp <= pos0 < rp + n:
                q = qp + pos0 - rp
                return row.seq[q], row.qual[q]
            rp += n
            qp += n
        elif o in "DN":
            if rp <= pos0 < rp + n:
                return None
            rp += n
        elif o in "IS":
            qp += n
    return None


def _ref_pileup(frame, sites, barcodes, *, excl=0x304, min_mapq=20, min_baseq=20, min_len=30, rule="majority",
                umi=True, cell=True):
    bcs = {b: i for i, b in enumerate(barcodes)} if barcodes is not None else None
    refs = {"chr1": 0, "chr2": 1}
    out = defaultdict(lambda: np.zeros(5, int))
    ties = 0
    for si, (c, p) in enumerate(zip(sites.chrom, sites.pos)):
        p0 = p - 1
        rows = frame[(frame.ref_id == refs[c]) & (frame.pos <= p0) & (frame.end > p0)]
        groups = defaultdict(list)
        for k, row in enumerate(rows.itertuples()):
            if row.flag & excl or row.mapq < min_mapq or _aligned_len(row.cigar) < min_len:
                continue
            bq = _base_at(row, p0)
            if bq is None or bq[1] < min_baseq:
                continue
            if cell and row.CB is None:
                continue
            if bcs is not None and row.CB not in bcs:
                continue
            if umi and row.UB is None:
                continue
            ci = bcs[row.CB] if bcs is not None else 0
            key = (row.CB if bcs is None and cell else None, row.UB if umi else k)
            groups[(ci, key)].append("ACGT".find(bq[0]) if bq[0] in "ACGT" else 4)
        for (ci, _), bases in groups.items():
            if rule == "first":
                out[(si, ci)][bases[0]] += 1
                continue
            cnt = np.bincount(bases, minlength=5)
            if (cnt == cnt.max()).sum() > 1:
                ties += 1
                continue
            out[(si, ci)][cnt.argmax()] += 1
    return out, ties


def _as_dict(ac):
    tot = {}
    for bi, b in enumerate("ACGTN"):
        m = ac.counts[b].tocoo()
        for r, c, v in zip(m.row, m.col, m.data):
            tot.setdefault((int(r), int(c)), np.zeros(5, int))[bi] += v
    return tot


def _same(ac, ref):
    got = _as_dict(ac)
    keys = set(got) | {k for k, v in ref.items() if v.sum()}
    bad = [k for k in keys if not np.array_equal(got.get(k, np.zeros(5)), ref.get(k, np.zeros(5)))]
    return bad


# ------------------------------------------------------------------------------------------ reader
def test_header_and_counts(toy):
    with B.BamFile(toy["path"]) as f:
        assert f.references == ["chr1", "chr2"] and f.lengths == [60000, 30000]
        assert "@SQ\tSN:chr1" in f.header_text
        assert f.count() == toy["n_records"]
        s = f.count_per_reference()
        assert s["*"] == 5 and s.sum() == toy["n_records"]
        assert f.count(exclude_flags=0x400) < f.count()


def test_roundtrip_fields(tmp_path):
    recs = [dict(name="a", flag=16, chrom="c1", pos=10, mapq=60, cigar="3S10M2I5M1D4M100N6M", seq="ACGT" * 7 + "A",
                 qual=list(range(29)), tags={"CB": "AAAC-1", "UB": "GGTT", "xf": 25, "RE": "E", "fl": 1.5}),
            dict(name="b", flag=0, chrom="c1", pos=500, mapq=255, cigar="8M", seq="NNACGTRY", tags={"xf": -3}),
            dict(name="u", flag=4, chrom=None, pos=-1, mapq=0, cigar="*", seq="ACG")]
    p = B.write_bam(tmp_path / "x.bam", [("c1", 2000)], recs)
    with B.BamFile(p, inflater="zlib") as f:
        df = f.fetch(tags=("CB", "UB", "xf", "RE", "fl")).to_frame()
    assert df.name.tolist() == ["a", "b", "u"]
    assert df.cigar.tolist() == ["3S10M2I5M1D4M100N6M", "8M", "*"]
    assert df.seq.tolist() == [recs[0]["seq"], "NNACGTRY", "ACG"]
    assert df.qual[0] == list(range(29)) and df.qual[1] == [255] * 8
    assert df.end.tolist() == [10 + 10 + 5 + 1 + 4 + 100 + 6, 508, 0]
    assert df.CB.tolist() == ["AAAC-1", None, None] and df.RE.tolist() == ["E", None, None]
    assert df.xf.tolist() == [25, -3, None] and df.fl[0] == 1.5
    assert df.ref_id.tolist() == [0, 0, -1] and df.flag.tolist() == [16, 0, 4]


@pytest.mark.parametrize("region", [("chr1", 0, 60000), ("chr1", 5000, 5001), ("chr1", 12345, 30000),
                                   ("chr2", 0, 100), ("chr2", 20000, 30000), ("chr1", 59990, 60000)])
def test_fetch_matches_brute_force(toy, frame, region):
    c, s, e = region
    with B.BamFile(toy["path"]) as f:
        got = f.fetch(c, s, e).to_frame()
    r = {"chr1": 0, "chr2": 1}[c]
    exp = frame[(frame.ref_id == r) & (frame.pos < e) & (frame.end > s)]
    assert got.name.tolist() == exp.name.tolist()


def test_region_string_and_small_chunks(toy, frame):
    with B.BamFile(toy["path"], chunk_mb=0.01, threads=3) as f:  # many batches: records split across batches
        assert f.count() == toy["n_records"]
        a = f.fetch("chr1:10,001-20,000").name.tolist()
        assert f.fetch().name.tolist() == frame.name.tolist()
    exp = frame[(frame.ref_id == 0) & (frame.pos < 20000) & (frame.end > 10000)].name.tolist()
    assert a == exp


@pytest.mark.parametrize("name", list(B._INFLATERS))
def test_inflaters_agree(toy, frame, name):
    with B.BamFile(toy["path"], inflater=name, threads=2) as f:
        assert f.fetch(tags=("UB",)).to_frame().UB.tolist() == frame.UB.tolist()


def test_index_reading(toy):
    idx = B.read_bai(toy["path"] + ".bai")
    assert len(idx) == 2 and all(len(i.bins) for i in idx)
    assert B._reg2bin(0, 1) == 4681 and B._reg2bin(0, 1 << 14) == 4681 and B._reg2bin(0, (1 << 14) + 1) == 585


# ------------------------------------------------------------------------------------------ pileup
@pytest.mark.parametrize("kw", [dict(), dict(umi_rule="first"), dict(min_mapq=0, min_len=0, min_baseq=0),
                                dict(exclude_flags=0x704), dict(exclude_flags=0)])
def test_pileup_matches_reference(toy, frame, kw):
    ac = B.pileup_cells(toy["path"], toy["sites"], toy["barcodes"], **kw)
    ref, ties = _ref_pileup(frame, toy["sites"], toy["barcodes"], excl=kw.get("exclude_flags", 0x304),
                            min_mapq=kw.get("min_mapq", 20), min_baseq=kw.get("min_baseq", 20),
                            min_len=kw.get("min_len", 30), rule=kw.get("umi_rule", "majority"))
    assert _same(ac, ref) == []
    if kw.get("umi_rule", "majority") == "majority":
        assert ac.stats["umi_ties"] == ties


def test_pileup_pooled_and_no_umi(toy, frame):
    ac = B.pileup_cells(toy["path"], toy["sites"], None)
    ref, _ = _ref_pileup(frame, toy["sites"], None)
    assert ac.barcodes == ["pooled"] and _same(ac, ref) == []
    ac = B.pileup_cells(toy["path"], toy["sites"], None, cell_tag=None, umi_tag=None)
    ref, _ = _ref_pileup(frame, toy["sites"], None, excl=0x704, umi=False, cell=False)
    assert _same(ac, ref) == []


def test_ad_dp_layout(toy):
    ac = B.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
    s = toy["sites"]
    for i in range(len(s)):
        r, a = s.ref[i], s.alt[i]
        exp_ad = ac.counts[a][i].toarray().ravel()
        exp_dp = exp_ad + ac.counts[r][i].toarray().ravel()
        assert np.array_equal(ac.AD[i].toarray().ravel(), exp_ad)
        assert np.array_equal(ac.DP[i].toarray().ravel(), exp_dp)
    tot = sum(ac.counts.values())
    assert np.array_equal((ac.DP + ac.OTH).toarray(), tot.toarray())
    lg = ac.long()
    assert (lg.DP == lg.AD + lg[["A", "C", "G", "T", "N"]].to_numpy()[np.arange(len(lg)),
            pd.Index(list("ACGTN")).get_indexer(s.ref.to_numpy()[lg.site])]).all()


def _cigar_bam(tmp_path, reads):
    """reads: (cigar, start, seq, flag, mapq, CB, UB, qual) on a 'c1' of 'A' * 1000 with a site at pos 101."""
    recs = []
    for i, (cig, start, seq, flag, mapq, cb, ub, q) in enumerate(sorted(reads, key=lambda r: r[1])):
        tags = {"CB": cb}
        if ub:
            tags["UB"] = ub
        recs.append(dict(name=f"r{i}", flag=flag, chrom="c1", pos=start, mapq=mapq, cigar=cig, seq=seq,
                         qual=q if q is not None else [30] * len(seq), tags=tags))
    return B.write_bam(tmp_path / "c.bam", [("c1", 1000)], recs)


def test_cigar_cases(tmp_path):
    # site: 1-based 101 = 0-based 100
    G = "G"
    reads = [
        ("40M", 80, "A" * 20 + G + "A" * 19, 0, 255, "C1", "U1", None),            # plain: G
        ("5S40M", 80, "TTTTT" + "A" * 20 + G + "A" * 19, 0, 255, "C1", "U2", None),  # soft clip shifts: G
        ("10M3I30M", 80, "A" * 10 + "CCC" + "A" * 10 + G + "A" * 19, 0, 255, "C1", "U3", None),  # insertion: G
        ("10M5D30M", 85, "A" * 10 + G + "A" * 29, 0, 255, "C1", "U4", None),       # site after deletion: G
        ("10M20D30M", 85, "A" * 40, 0, 255, "C1", "U5", None),                     # site inside deletion: none
        ("10M500N30M", 60, "A" * 40, 0, 255, "C1", "U6", None),                    # site inside intron: none
        ("10M40N30M", 45, "A" * 15 + G + "A" * 24, 0, 255, "C1", "U7", None),      # second exon covers: G
        ("40M", 80, "A" * 20 + "T" + "A" * 19, 0, 255, "C1", "U8", None),          # T
    ]
    ac = B.pileup_cells(_cigar_bam(tmp_path, reads), pd.DataFrame({"chrom": ["c1"], "pos": [101]}), ["C1"],
                        min_len=0)
    assert ac.counts["G"][0, 0] == 5 and ac.counts["T"][0, 0] == 1 and ac.counts["A"][0, 0] == 0


def test_umi_collapse_and_filters(tmp_path):
    s = "A" * 20 + "G" + "A" * 19
    t = "A" * 20 + "T" + "A" * 19
    low = [30] * 20 + [5] + [30] * 19
    reads = [
        ("40M", 80, s, 0, 255, "C1", "U1", None), ("40M", 81 - 1, s, 0x400, 255, "C1", "U1", None),
        ("40M", 80, t, 0x400, 255, "C1", "U1", None),                      # U1: G, G, T -> G
        ("40M", 80, s, 0, 255, "C1", "U2", None), ("40M", 80, t, 0x400, 255, "C1", "U2", None),  # tie -> dropped
        ("40M", 80, t, 0x100, 255, "C1", "U3", None),                      # secondary: skipped
        ("40M", 80, t, 0x200, 255, "C1", "U4", None),                      # QC fail: skipped
        ("40M", 80, t, 0, 3, "C1", "U5", None),                            # MAPQ 3: skipped
        ("40M", 80, t, 0, 255, "C1", "U6", low),                           # base quality 5: skipped
        ("40M", 80, t, 0, 255, "C9", "U7", None),                          # barcode not listed
        ("40M", 80, t, 0, 255, "C1", None, None),                          # no UMI
        ("40M", 80, t, 0, 255, "C2", "U1", None),                          # same UMI, other cell: T
    ]
    p = _cigar_bam(tmp_path, reads)
    site = pd.DataFrame({"chrom": ["c1"], "pos": [101], "ref": ["G"], "alt": ["T"]})
    ac = B.pileup_cells(p, site, ["C1", "C2"], min_len=0)
    assert ac.AD.toarray().tolist() == [[0, 1]] and ac.DP.toarray().tolist() == [[1, 1]]
    assert ac.stats["umi_ties"] == 1 and ac.stats["cell_not_listed"] == 1 and ac.stats["no_umi_tag"] == 1
    assert ac.stats["low_baseq"] == 1
    first = B.pileup_cells(p, site, ["C1", "C2"], min_len=0, umi_rule="first")
    assert first.DP.toarray().tolist() == [[2, 1]]  # U1 -> G, U2 -> G (first read)
    reads_only = B.pileup_cells(p, site, ["C1", "C2"], min_len=0, umi_tag=None)  # duplicates excluded by flag
    assert reads_only.counts["G"].toarray().tolist() == [[2, 0]]
    assert reads_only.counts["T"].toarray().tolist() == [[1, 1]]  # no-UMI read counts; MAPQ/secondary/QC still out
    agg = ac.aggregate({"C1": "S1", "C2": "S1"})
    assert agg[["AD", "DP", "n_cells"]].values.tolist() == [[1, 2, 2]]


def test_unknown_reference_sites(toy):
    s = pd.concat([toy["sites"], pd.DataFrame({"chrom": ["chrZ"], "pos": [5], "ref": ["A"], "alt": ["C"]})],
                  ignore_index=True)
    ac = B.pileup_cells(toy["path"], s, toy["barcodes"])
    assert ac.DP.shape[0] == len(s) and ac.DP[len(s) - 1].nnz == 0


# ------------------------------------------------------------------------------------------ de novo
def test_discover_matches_brute_force(toy, frame):
    d = B.discover_snps(toy["path"], min_depth=1, min_alt=0, min_maf=0.0, min_len=0)
    exp = defaultdict(lambda: np.zeros(5, int))
    keep = frame[((frame.flag & 0x704) == 0) & (frame.mapq >= 20)]
    for row in keep.itertuples():
        rp, qp = row.pos, 0
        for n, o in re.findall(r"(\d+)([MIDNSHP=X])", row.cigar):
            n = int(n)
            if o in "M=X":
                for t in range(n):
                    if row.qual[qp + t] >= 20:
                        b = row.seq[qp + t]
                        exp[(row.ref_id, rp + t)]["ACGT".find(b) if b in "ACGT" else 4] += 1
                rp += n
                qp += n
            elif o in "DN":
                rp += n
            elif o in "IS":
                qp += n
    exp = {k: v for k, v in exp.items() if v[:4].sum() >= 1}
    got = {( {"chr1": 0, "chr2": 1}[c], p - 1): np.array(v) for c, p, *v in
           d[["chrom", "pos", "A", "C", "G", "T", "N"]].itertuples(index=False)}
    assert got.keys() == exp.keys()
    assert all(np.array_equal(got[k], exp[k]) for k in exp)
    assert d.attrs["dropped_bases"] == 0


def test_discover_finds_planted_snps(toy):
    d = B.discover_snps(toy["path"], min_depth=10, min_alt=3, min_maf=0.1)
    found = set(zip(d.chrom, d.pos))
    het_any = toy["genotypes"].apply(lambda g: (g > 0).any() and (g < 2).any(), axis=1).to_numpy()
    planted = set(zip(toy["sites"].chrom[het_any], toy["sites"].pos[het_any]))
    assert len(planted & found) >= 0.8 * len(planted)
    assert len(found - set(zip(toy["sites"].chrom, toy["sites"].pos))) == 0  # no false sites (errors are rare)
    with pytest.raises(KeyError):
        B.inflater("nope")


# ------------------------------------------------------------------------------------------ genotype comparison
def test_genotype_concordance_separates_donors(toy):
    ac = B.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
    agg = ac.aggregate(toy["donor"].str.cat(toy["half"], sep="_"))
    pairs = B.genotype_concordance(agg, min_depth=5)
    same = pairs.sample_a.str[:2] == pairs.sample_b.str[:2]
    assert len(pairs) == 15 and same.sum() == 3
    assert (pairs[same].concordance_informative >= 0.9).all() and (pairs[same].n_opposite_hom == 0).all()
    assert pairs[same].af_pearson.min() > pairs[~same].af_pearson.max()
    assert pairs[~same].concordance_informative.max() < 0.8
    af = B.sample_allele_fractions(agg, min_depth=5)
    assert af.shape[1] == 6 and np.nanmax(af.to_numpy()) <= 1
