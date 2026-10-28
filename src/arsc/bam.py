"""BAM reading and per-cell allele counting, in native Python (no pysam / htslib).

What genotype checks need from pysam / htslib / cellsnp-lite, written from the SAM/BAM specification
(https://samtools.github.io/hts-specs/SAMv1.pdf, sections 4.1-4.2 and 5.2):

* :class:`BamFile` reads the header, the ``.bai`` index (region queries) and the records of a coordinate-sorted
  BAM. BGZF blocks are inflated in parallel threads (libdeflate through the ``deflate`` wheel when installed,
  else ``isal``, else the standard ``zlib``; all release the GIL) into a reused buffer, and records are parsed by
  numba kernels into columnar arrays (:class:`Records`). Files are read sequentially in large chunks by a
  prefetch thread, so a whole BAM streams with bounded memory.
* :func:`pileup_cells` counts REF / ALT UMIs per cell barcode at given SNPs (cellsnp-lite mode 1 style
  AD / DP / OTH sparse matrices), with the read filters of a pileup (flag, MAPQ, base quality, aligned length)
  and CIGAR M / I / D / N / S / = / X handled as htslib's pileup does (a site in a deletion or an intron is not
  covered).
* :func:`discover_snps` piles up every covered position (pooled over cells) and keeps candidate SNPs by depth and
  minor-allele count / fraction (cellsnp-lite mode 2 style).
* :func:`genotype_concordance` / :func:`sample_allele_fractions` compare samples (donors, hashtag groups,
  libraries) by their allele fractions at shared informative sites.
* :func:`write_bam` writes a small BAM with its ``.bai`` (tests, docs, toy data: :func:`arsc.datasets.toy_bam`).

Parity with pysam 0.22.1 (htslib 1.18) on a 10x Genomics BAM: BAM_PARITY.md.

Example
-------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> with arsc.bam.BamFile(toy["path"]) as bam:
...     bam.references
['chr1', 'chr2']
"""
from __future__ import annotations

import os
import queue
import re
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["BamFile", "Records", "AlleleCounts", "read_bai", "write_bam", "pileup_cells", "discover_snps",
           "sample_allele_fractions", "genotype_concordance", "inflater"]

FUNMAP, FSECONDARY, FQCFAIL, FDUP, FSUPPLEMENTARY = 0x4, 0x100, 0x200, 0x400, 0x800
_SEQ_ALPHABET = np.frombuffer(b"=ACMGRSVTWYHKDBN", np.uint8)
_CIGAR_CHARS = "MIDNSHP=X"
_BASES = np.array(list("ACGTN"))
_MAX_BLOCK = 65536


# ------------------------------------------------------------------------------------------ inflaters
def _inflaters():
    out = {}
    try:
        import deflate  # libdeflate
        f = deflate.deflate_decompress
        out["libdeflate"] = lambda data, size: f(data, size)
    except ImportError:  # pragma: no cover - optional
        pass
    try:
        from isal import isal_zlib
        out["isal"] = lambda data, size: isal_zlib.decompress(data, wbits=-15, bufsize=max(size, 1))
    except ImportError:  # pragma: no cover - optional
        pass
    out["zlib"] = lambda data, size: zlib.decompress(data, -15, max(size, 1))
    return out


_INFLATERS = _inflaters()


def inflater(name: str | None = None) -> str:
    """Name of the raw-DEFLATE engine BGZF blocks are inflated with.

    Parameters
    ----------
    name : ``None`` (the fastest installed: ``"libdeflate"`` from the ``deflate`` wheel, else ``"isal"``, else the
        standard library ``"zlib"``) or one of those names (``KeyError`` if it is not installed).

    Returns
    -------
    str, the engine name.

    Examples
    --------
    >>> import arsc
    >>> arsc.bam.inflater("zlib")
    'zlib'
    """
    if name is None:
        return next(iter(_INFLATERS))
    if name not in _INFLATERS:
        raise KeyError(f"inflater {name!r} is not installed (available: {', '.join(_INFLATERS)})")
    return name


# ------------------------------------------------------------------------------------------ numba kernels
@nb.njit(cache=True, inline="always")
def _u16(b, i):
    return np.int64(b[i]) | (np.int64(b[i + 1]) << 8)


@nb.njit(cache=True, inline="always")
def _u32(b, i):
    return np.int64(b[i]) | (np.int64(b[i + 1]) << 8) | (np.int64(b[i + 2]) << 16) | (np.int64(b[i + 3]) << 24)


@nb.njit(cache=True, inline="always")
def _i32(b, i):
    v = _u32(b, i)
    return v - 4294967296 if v >= 2147483648 else v


@nb.njit(cache=True, nogil=True)
def _scan_blocks(buf, n):
    """Complete BGZF blocks in buf[:n]: data offset, data length, inflated size, block offset; bytes consumed;
    status 0 ok / -1 not a BGZF block at the returned position."""
    cap = n // 28 + 1
    doff = np.empty(cap, np.int64)
    dlen = np.empty(cap, np.int64)
    isz = np.empty(cap, np.int64)
    boff = np.empty(cap, np.int64)
    i = 0
    k = 0
    while i + 18 <= n:
        if buf[i] != 31 or buf[i + 1] != 139 or buf[i + 2] != 8 or (buf[i + 3] & 4) == 0:
            return doff[:k], dlen[:k], isz[:k], boff[:k], i, -1
        xlen = _u16(buf, i + 10)
        if i + 12 + xlen > n:
            break
        bsize = -1
        j = i + 12
        while j + 4 <= i + 12 + xlen:
            slen = _u16(buf, j + 2)
            if buf[j] == 66 and buf[j + 1] == 67 and slen == 2:
                bsize = _u16(buf, j + 4) + 1
            j += 4 + slen
        if bsize < 0:
            return doff[:k], dlen[:k], isz[:k], boff[:k], i, -1
        if i + bsize > n:
            break
        e = i + bsize
        doff[k] = i + 12 + xlen
        dlen[k] = bsize - xlen - 20
        isz[k] = _u32(buf, e - 4)
        boff[k] = i
        k += 1
        i = e
    return doff[:k], dlen[:k], isz[:k], boff[:k], i, 0


@nb.njit(cache=True, nogil=True)
def _find_records(buf, start, n, limit):
    """Offsets of the complete records that start in [start, limit) within buf[:n]; and the offset reached."""
    m = min(n, limit)
    cap = max((m - start) // 36 + 1, 1)
    offs = np.empty(cap, np.int64)
    o = start
    k = 0
    while o < limit and o + 4 <= n:
        bs = _i32(buf, o)
        if o + 4 + bs > n:
            break
        offs[k] = o
        k += 1
        o += 4 + bs
    return offs[:k], o


@nb.njit(cache=True, inline="always")
def _ref_len(buf, co, ncig):
    r = 0
    for c in range(ncig):
        v = _u32(buf, co + 4 * c)
        op = v & 15
        if op == 0 or op == 2 or op == 3 or op == 7 or op == 8:
            r += v >> 4
    return r


@nb.njit(cache=True, inline="always")
def _aln_len(buf, co, ncig):
    r = 0
    for c in range(ncig):
        v = _u32(buf, co + 4 * c)
        op = v & 15
        if op == 0 or op == 7 or op == 8:
            r += v >> 4
    return r


@nb.njit(cache=True, inline="always")
def _find_tag(buf, p, e, t1, t2):
    """(type byte, value offset) of tag t1t2 in the tag area buf[p:e], or (0, -1)."""
    while p + 3 <= e:
        a = buf[p]
        b = buf[p + 1]
        t = buf[p + 2]
        v = p + 3
        if a == t1 and b == t2:
            return t, v
        if t == 65 or t == 99 or t == 67:  # A c C
            p = v + 1
        elif t == 115 or t == 83:  # s S
            p = v + 2
        elif t == 105 or t == 73 or t == 102:  # i I f
            p = v + 4
        elif t == 90 or t == 72:  # Z H
            while v < e and buf[v] != 0:
                v += 1
            p = v + 1
        elif t == 66:  # B
            st = buf[v]
            cnt = _u32(buf, v + 1)
            sz = 1
            if st == 115 or st == 83:
                sz = 2
            elif st == 105 or st == 73 or st == 102:
                sz = 4
            p = v + 5 + sz * cnt
        else:
            return 0, -1
    return 0, -1


@nb.njit(cache=True, inline="always")
def _zlen(buf, v, e):
    w = v
    while w < e and buf[w] != 0:
        w += 1
    return w - v


@nb.njit(cache=True, inline="always")
def _fnv(buf, v, n, h):
    for i in range(n):
        h = (h ^ np.uint64(buf[v + i])) * np.uint64(1099511628211)
    return h


@nb.njit(cache=True, inline="always")
def _umi_code(buf, v, n):
    """Exact 2-bit code (with length) for an ACGT string of <= 28 bases, else a 64-bit FNV-1a hash with the top
    bit set (cannot equal an exact code)."""
    if n <= 28:
        code = np.uint64(n) << np.uint64(56)
        ok = True
        for i in range(n):
            c = buf[v + i]
            if c == 65:
                x = 0
            elif c == 67:
                x = 1
            elif c == 71:
                x = 2
            elif c == 84:
                x = 3
            else:
                ok = False
                break
            code |= np.uint64(x) << np.uint64(2 * i)
        if ok:
            return code
    return _fnv(buf, v, n, np.uint64(14695981039346656037)) | np.uint64(9223372036854775808)


@nb.njit(cache=True, inline="always")
def _cell_lookup(buf, v, n, bc_hash, bc_idx, bc_flat, bc_off):
    h = _fnv(buf, v, n, np.uint64(14695981039346656037))
    j = np.searchsorted(bc_hash, h)
    while j < bc_hash.shape[0] and bc_hash[j] == h:
        c = bc_idx[j]
        s = bc_off[c]
        if bc_off[c + 1] - s == n:
            same = True
            for i in range(n):
                if bc_flat[s + i] != buf[v + i]:
                    same = False
                    break
            if same:
                return c
        j += 1
    return -1


@nb.njit(cache=True, nogil=True)
def _hash_strings(flat, off):
    n = off.shape[0] - 1
    h = np.empty(n, np.uint64)
    for i in range(n):
        h[i] = _fnv(flat, off[i], off[i + 1] - off[i], np.uint64(14695981039346656037))
    return h


@nb.njit(cache=True, nogil=True)
def _parse_fixed(buf, offs):
    n = offs.shape[0]
    ref = np.empty(n, np.int32)
    pos = np.empty(n, np.int32)
    end = np.empty(n, np.int32)
    mapq = np.empty(n, np.uint8)
    flag = np.empty(n, np.uint16)
    nref = np.empty(n, np.int32)
    npos = np.empty(n, np.int32)
    tlen = np.empty(n, np.int32)
    lname = np.empty(n, np.int64)
    ncig = np.empty(n, np.int64)
    lseq = np.empty(n, np.int64)
    for k in range(n):
        o = offs[k]
        ref[k] = _i32(buf, o + 4)
        pos[k] = _i32(buf, o + 8)
        ln = buf[o + 12]
        mapq[k] = buf[o + 13]
        nc = _u16(buf, o + 16)
        fl = _u16(buf, o + 18)
        flag[k] = fl
        ls = _i32(buf, o + 20)
        nref[k] = _i32(buf, o + 24)
        npos[k] = _i32(buf, o + 28)
        tlen[k] = _i32(buf, o + 32)
        lname[k] = ln
        ncig[k] = nc
        lseq[k] = ls
        rl = _ref_len(buf, o + 36 + ln, nc)
        end[k] = pos[k] + rl if (rl > 0 and (fl & 4) == 0) else pos[k] + 1
    return ref, pos, end, mapq, flag, nref, npos, tlen, lname, ncig, lseq


@nb.njit(cache=True, nogil=True)
def _parse_var(buf, offs, keep, lname, ncig, lseq, name_off, cig_off, seq_off, alpha):
    """Names (ASCII without NUL), CIGAR words, sequence (ASCII) and qualities of the kept records."""
    name = np.empty(name_off[-1], np.uint8)
    cig = np.empty(cig_off[-1], np.uint32)
    seq = np.empty(seq_off[-1], np.uint8)
    qual = np.empty(seq_off[-1], np.uint8)
    for j in range(keep.shape[0]):
        k = keep[j]
        o = offs[k]
        ln = lname[k]
        p = o + 36
        for i in range(ln - 1):
            name[name_off[j] + i] = buf[p + i]
        p += ln
        for c in range(ncig[k]):
            cig[cig_off[j] + c] = _u32(buf, p + 4 * c)
        p += 4 * ncig[k]
        ls = lseq[k]
        s0 = seq_off[j]
        for i in range(ls):
            byte = np.int64(buf[p + i // 2])
            code = (byte >> 4) if i % 2 == 0 else (byte & 15)
            seq[s0 + i] = alpha[code]
        p += (ls + 1) // 2
        for i in range(ls):
            qual[s0 + i] = buf[p + i]
    return name, cig, seq, qual


@nb.njit(cache=True, nogil=True)
def _parse_tags(buf, offs, keep, lname, ncig, lseq, tags1, tags2):
    """For each kept record and requested tag: kind (0 missing, 1 string Z/H, 2 char A, 3 int, 4 float, 5 B array),
    integer value, float value, string offset and length in buf."""
    n = keep.shape[0]
    nt = tags1.shape[0]
    kind = np.zeros((n, nt), np.int8)
    ival = np.zeros((n, nt), np.int64)
    fval = np.zeros((n, nt), np.float64)
    soff = np.zeros((n, nt), np.int64)
    slen = np.zeros((n, nt), np.int64)
    for j in range(n):
        k = keep[j]
        o = offs[k]
        e = o + 4 + _i32(buf, o)
        p = o + 36 + lname[k] + 4 * ncig[k] + (lseq[k] + 1) // 2 + lseq[k]
        for t in range(nt):
            ty, v = _find_tag(buf, p, e, tags1[t], tags2[t])
            if v < 0:
                continue
            if ty == 90 or ty == 72:
                kind[j, t] = 1
                soff[j, t] = v
                slen[j, t] = _zlen(buf, v, e)
            elif ty == 65:
                kind[j, t] = 2
                soff[j, t] = v
                slen[j, t] = 1
            elif ty == 99:
                x = np.int64(buf[v])
                ival[j, t] = x - 256 if x >= 128 else x
                kind[j, t] = 3
            elif ty == 67:
                ival[j, t] = buf[v]
                kind[j, t] = 3
            elif ty == 115:
                x = _u16(buf, v)
                ival[j, t] = x - 65536 if x >= 32768 else x
                kind[j, t] = 3
            elif ty == 83:
                ival[j, t] = _u16(buf, v)
                kind[j, t] = 3
            elif ty == 105:
                ival[j, t] = _i32(buf, v)
                kind[j, t] = 3
            elif ty == 73:
                ival[j, t] = _u32(buf, v)
                kind[j, t] = 3
            elif ty == 102:
                kind[j, t] = 4
                soff[j, t] = v
            elif ty == 66:
                kind[j, t] = 5
                soff[j, t] = v
    return kind, ival, fval, soff, slen


@nb.njit(cache=True, nogil=True)
def _gather(buf, off, ln):
    out = np.empty(ln.sum(), np.uint8)
    p = 0
    for i in range(off.shape[0]):
        for j in range(ln[i]):
            out[p + j] = buf[off[i] + j]
        p += ln[i]
    return out


@nb.njit(cache=True, nogil=True)
def _count_kernel(buf, offs, excl, min_mapq, ref_counts):
    n = 0
    for k in range(offs.shape[0]):
        o = offs[k]
        if (_u16(buf, o + 18) & excl) != 0 or buf[o + 13] < min_mapq:
            continue
        r = _i32(buf, o + 4)
        if r >= 0:
            ref_counts[r] += 1
        else:
            ref_counts[ref_counts.shape[0] - 1] += 1
        n += 1
    return n


@nb.njit(cache=True, nogil=True)
def _pileup_kernel(buf, offs, k0, site_lo, site_hi, site_pos, excl, min_mapq, min_baseq, min_len,
                   ct1, ct2, ut1, ut2, use_cell, use_umi, pooled, bc_hash, bc_idx, bc_flat, bc_off, rec_base,
                   o_site, o_cell, o_key, o_ckey, o_base, nout, stats):
    """Emit one observation (site, cell, UMI key, pooled cell key, base) per read covering a site.
    Returns (next record index, observations written); stops early when the output arrays are full."""
    cap = o_site.shape[0]
    nref = site_lo.shape[0]
    for k in range(k0, offs.shape[0]):
        o = offs[k]
        fl = _u16(buf, o + 18)
        r = _i32(buf, o + 4)
        if r < 0 or r >= nref or site_lo[r] == site_hi[r]:
            continue
        ln = buf[o + 12]
        nc = _u16(buf, o + 16)
        co = o + 36 + ln
        pos = _i32(buf, o + 8)
        rl = _ref_len(buf, co, nc)
        if rl == 0:
            continue
        end = pos + rl
        # sites in [pos, end)
        lo = site_lo[r] + np.searchsorted(site_pos[site_lo[r]:site_hi[r]], pos)
        if lo >= site_hi[r] or site_pos[lo] >= end:
            continue
        stats[0] += 1  # reads overlapping a site (before filters)
        if (fl & excl) != 0 or buf[o + 13] < min_mapq:
            continue
        if min_len > 0 and _aln_len(buf, co, nc) < min_len:
            continue
        ls = _i32(buf, o + 20)
        so = co + 4 * nc
        qo = so + (ls + 1) // 2
        e = o + 4 + _i32(buf, o)
        cell = 0
        ckey = np.uint64(0)
        if use_cell:
            ty, v = _find_tag(buf, qo + ls, e, ct1, ct2)
            if v < 0 or ty != 90:
                stats[1] += 1
                continue
            n = _zlen(buf, v, e)
            if pooled:
                ckey = _fnv(buf, v, n, np.uint64(14695981039346656037))
            else:
                cell = _cell_lookup(buf, v, n, bc_hash, bc_idx, bc_flat, bc_off)
                if cell < 0:
                    stats[2] += 1
                    continue
        if use_umi:
            ty, v = _find_tag(buf, qo + ls, e, ut1, ut2)
            if v < 0 or ty != 90:
                stats[3] += 1
                continue
            key = _umi_code(buf, v, _zlen(buf, v, e))
        else:
            key = np.uint64(rec_base + k)
        # upper bound on emissions for this record
        hi = lo
        while hi < site_hi[r] and site_pos[hi] < end:
            hi += 1
        if nout + (hi - lo) > cap:
            return k, nout
        s = lo
        rpos = pos
        qpos = 0
        for c in range(nc):
            if s >= hi:
                break
            v32 = _u32(buf, co + 4 * c)
            op = v32 & 15
            L = v32 >> 4
            if op == 0 or op == 7 or op == 8:
                while s < hi and site_pos[s] < rpos + L:
                    q = qpos + (site_pos[s] - rpos)
                    if buf[qo + q] >= min_baseq:
                        byte = np.int64(buf[so + q // 2])
                        code = (byte >> 4) if q % 2 == 0 else (byte & 15)
                        b = 4
                        if code == 1:
                            b = 0
                        elif code == 2:
                            b = 1
                        elif code == 4:
                            b = 2
                        elif code == 8:
                            b = 3
                        o_site[nout] = s
                        o_cell[nout] = cell
                        o_key[nout] = key
                        o_ckey[nout] = ckey
                        o_base[nout] = b
                        nout += 1
                    else:
                        stats[4] += 1
                    s += 1
                rpos += L
                qpos += L
            elif op == 2 or op == 3:
                while s < hi and site_pos[s] < rpos + L:
                    s += 1  # site inside a deletion / intron: not covered
                rpos += L
            elif op == 1 or op == 4:
                qpos += L
    return offs.shape[0], nout


@nb.njit(cache=True, nogil=True)
def _collapse_umis(site, cell, ckey, key, base, order, rule):
    """One base per (site, cell, pooled cell key, UMI) group of the sorted observations. rule 0 = majority (ties
    dropped), 1 = first read in file order. Returns (site, cell, base) of the kept UMIs and the number of ties."""
    n = order.shape[0]
    rs = np.empty(n, np.int32)
    rc = np.empty(n, np.int32)
    rb = np.empty(n, np.int8)
    m = 0
    ties = 0
    i = 0
    cnt = np.zeros(5, np.int64)
    while i < n:
        a = order[i]
        j = i
        cnt[:] = 0
        while j < n:
            b = order[j]
            if site[b] != site[a] or cell[b] != cell[a] or ckey[b] != ckey[a] or key[b] != key[a]:
                break
            cnt[base[b]] += 1
            j += 1
        if rule == 1:
            rs[m] = site[a]
            rc[m] = cell[a]
            rb[m] = base[a]
            m += 1
        else:
            best = 0
            for x in range(1, 5):
                if cnt[x] > cnt[best]:
                    best = x
            tie = False
            for x in range(5):
                if x != best and cnt[x] == cnt[best]:
                    tie = True
            if tie:
                ties += 1
            else:
                rs[m] = site[a]
                rc[m] = cell[a]
                rb[m] = best
                m += 1
        i = j
    return rs[:m], rc[:m], rb[:m], ties


@nb.njit(cache=True, nogil=True)
def _grow(a, n):
    b = np.empty(max(2 * a.shape[0], n), a.dtype)
    b[:a.shape[0]] = a
    return b


@nb.njit(cache=True, nogil=True)
def _flush(ring, W, ref, p0, p1, min_depth, min_alt, min_maf, min_top2, o_ref, o_pos, o_cnt, k):
    for p in range(p0, p1):
        i = (p % W) * 5
        d = ring[i] + ring[i + 1] + ring[i + 2] + ring[i + 3]
        if d >= min_depth and d > 0:
            b1 = 0
            for x in range(1, 4):
                if ring[i + x] > ring[i + b1]:
                    b1 = x
            b2 = 1 if b1 == 0 else 0
            for x in range(4):
                if x != b1 and ring[i + x] > ring[i + b2]:
                    b2 = x
            c2 = ring[i + b2]
            if c2 >= min_alt and c2 >= min_maf * d and ring[i + b1] + c2 >= min_top2 * d:
                if k >= o_pos.shape[0]:
                    o_ref = _grow(o_ref, k + 1)
                    o_pos = _grow(o_pos, k + 1)
                    o_cnt2 = np.empty((max(2 * o_cnt.shape[0], k + 1), 5), o_cnt.dtype)
                    o_cnt2[:o_cnt.shape[0]] = o_cnt
                    o_cnt = o_cnt2
                o_ref[k] = ref
                o_pos[k] = p
                for x in range(5):
                    o_cnt[k, x] = ring[i + x]
                k += 1
        for x in range(5):
            ring[i + x] = 0
    return o_ref, o_pos, o_cnt, k


@nb.njit(cache=True, nogil=True)
def _discover_kernel(buf, offs, ring, W, state, excl, min_mapq, min_baseq, min_len, ct1, ct2, use_cell,
                     bc_hash, bc_idx, bc_flat, bc_off, ref_ok, min_depth, min_alt, min_maf, min_top2,
                     o_ref, o_pos, o_cnt, k):
    """Pooled base counts at every covered position; positions behind the current read start are final and are
    tested as candidate SNPs. state = [current ref, flushed-up-to, highest position + 1, dropped bases,
    unsorted flag]."""
    for kk in range(offs.shape[0]):
        o = offs[kk]
        fl = _u16(buf, o + 18)
        r = _i32(buf, o + 4)
        if r < 0 or (fl & excl) != 0 or buf[o + 13] < min_mapq or not ref_ok[r]:
            continue
        ln = buf[o + 12]
        nc = _u16(buf, o + 16)
        co = o + 36 + ln
        if min_len > 0 and _aln_len(buf, co, nc) < min_len:
            continue
        ls = _i32(buf, o + 20)
        so = co + 4 * nc
        qo = so + (ls + 1) // 2
        if use_cell:
            e = o + 4 + _i32(buf, o)
            ty, v = _find_tag(buf, qo + ls, e, ct1, ct2)
            if v < 0 or ty != 90:
                continue
            if _cell_lookup(buf, v, _zlen(buf, v, e), bc_hash, bc_idx, bc_flat, bc_off) < 0:
                continue
        pos = _i32(buf, o + 8)
        if r != state[0]:
            if state[0] >= 0:
                o_ref, o_pos, o_cnt, k = _flush(ring, W, state[0], state[1], state[2], min_depth, min_alt,
                                                min_maf, min_top2, o_ref, o_pos, o_cnt, k)
            state[0] = r
            state[1] = pos
            state[2] = pos
        elif pos < state[1]:
            state[4] = 1
            return o_ref, o_pos, o_cnt, k
        elif pos > state[1]:
            o_ref, o_pos, o_cnt, k = _flush(ring, W, r, state[1], min(pos, state[2]), min_depth, min_alt,
                                            min_maf, min_top2, o_ref, o_pos, o_cnt, k)
            state[1] = pos
            if state[2] < pos:
                state[2] = pos
        rpos = pos
        qpos = 0
        lim = state[1] + W
        for c in range(nc):
            v32 = _u32(buf, co + 4 * c)
            op = v32 & 15
            L = v32 >> 4
            if op == 0 or op == 7 or op == 8:
                for t in range(L):
                    q = qpos + t
                    p = rpos + t
                    if p >= lim:
                        state[3] += 1
                        continue
                    if buf[qo + q] < min_baseq:
                        continue
                    byte = np.int64(buf[so + q // 2])
                    code = (byte >> 4) if q % 2 == 0 else (byte & 15)
                    b = 4
                    if code == 1:
                        b = 0
                    elif code == 2:
                        b = 1
                    elif code == 4:
                        b = 2
                    elif code == 8:
                        b = 3
                    ring[(p % W) * 5 + b] += 1
                    if p + 1 > state[2]:
                        state[2] = p + 1
                rpos += L
                qpos += L
            elif op == 2 or op == 3:
                rpos += L
            elif op == 1 or op == 4:
                qpos += L
    return o_ref, o_pos, o_cnt, k


# ------------------------------------------------------------------------------------------ index
@dataclass
class _RefIndex:
    bins: dict
    ioffsets: np.ndarray


def read_bai(path) -> list:
    """Read a BAM index (``.bai``).

    Parameters
    ----------
    path : the ``.bai`` file.

    Returns
    -------
    list with one entry per reference: an object with ``bins`` (dict bin number -> ``(n, 2)`` uint64 array of
    chunk virtual offsets ``[begin, end)``, the pseudo-bin 37450 removed) and ``ioffsets`` (linear index, uint64
    virtual offset of the first record overlapping each 16 kb window).

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> idx = arsc.bam.read_bai(toy["path"] + ".bai")
    >>> len(idx)
    2
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"BAI\x01":
        raise ValueError(f"{path}: not a BAI index")
    p = 4
    (n_ref,) = struct.unpack_from("<i", data, p)
    p += 4
    out = []
    for _ in range(n_ref):
        (n_bin,) = struct.unpack_from("<i", data, p)
        p += 4
        bins = {}
        for _ in range(n_bin):
            b, n_chunk = struct.unpack_from("<Ii", data, p)
            p += 8
            ch = np.frombuffer(data, np.uint64, 2 * n_chunk, p).reshape(n_chunk, 2).copy()
            p += 16 * n_chunk
            if b != 37450:
                bins[b] = ch
        (n_intv,) = struct.unpack_from("<i", data, p)
        p += 4
        io = np.frombuffer(data, np.uint64, n_intv, p).copy()
        p += 8 * n_intv
        out.append(_RefIndex(bins, io))
    return out


def _reg2bins(beg: int, end: int) -> list:
    end -= 1
    bins = [0]
    for shift, off in ((26, 1), (23, 9), (20, 73), (17, 585), (14, 4681)):
        bins.extend(range(off + (beg >> shift), off + (end >> shift) + 1))
    return bins


def _reg2bin(beg: int, end: int) -> int:
    end -= 1
    for shift, off in ((14, 4681), (17, 585), (20, 73), (23, 9), (26, 1)):
        if beg >> shift == end >> shift:
            return off + (beg >> shift)
    return 0


def _merge_chunks(chunks: np.ndarray, gap_bytes: int) -> np.ndarray:
    """Sort and merge [beg, end) virtual-offset chunks; chunks whose compressed gap is < gap_bytes are joined."""
    if len(chunks) == 0:
        return np.zeros((0, 2), np.uint64)
    chunks = chunks[np.argsort(chunks[:, 0], kind="stable")]
    out = [list(chunks[0])]
    for b, e in chunks[1:]:
        cur = out[-1]
        if int(b) <= int(cur[1]) or (int(b) >> 16) - (int(cur[1]) >> 16) < gap_bytes:
            if e > cur[1]:
                cur[1] = e
        else:
            out.append([b, e])
    return np.array(out, np.uint64)


# ------------------------------------------------------------------------------------------ records
@dataclass
class Records:
    """Columnar BAM records (what pysam's ``AlignedSegment`` fields hold, for many reads at once).

    Attributes
    ----------
    ref_id, pos, end : int32 arrays: reference index (-1 unplaced), 0-based leftmost position, 0-based exclusive
        end on the reference (``pos + 1`` for unmapped reads or reads without reference-consuming CIGAR, as htslib's
        ``bam_endpos``).
    mapq, flag : uint8 / uint16 arrays.
    next_ref_id, next_pos, tlen : int32 arrays (mate reference, mate position, template length).
    name : object array of read names.
    cigar_ops, cigar_lens, cigar_off : flat CIGAR (op codes 0-8 = ``MIDNSHP=X``, lengths) and per-record offsets
        (record i owns ``[cigar_off[i], cigar_off[i+1])``).
    seq, qual, seq_off : flat ASCII sequence and base qualities (uint8, Phred; 255 = absent) with per-record offsets.
    tags : dict tag -> array: int64 when every record has the tag as an integer, else an object array (str for
        Z / H / A tags, int, float, list for B arrays; ``None`` where the tag is absent).

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> recs = arsc.bam.BamFile(toy["path"]).fetch("chr2", tags=("CB",))
    >>> len(recs) == len(recs.tags["CB"])
    True
    """
    ref_id: np.ndarray
    pos: np.ndarray
    end: np.ndarray
    mapq: np.ndarray
    flag: np.ndarray
    next_ref_id: np.ndarray
    next_pos: np.ndarray
    tlen: np.ndarray
    name: np.ndarray
    cigar_ops: np.ndarray
    cigar_lens: np.ndarray
    cigar_off: np.ndarray
    seq: np.ndarray
    qual: np.ndarray
    seq_off: np.ndarray
    tags: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.pos)

    def cigar_string(self, i: int) -> str:
        """CIGAR of record ``i`` as a SAM string.

        Parameters
        ----------
        i : record index.

        Returns
        -------
        str (``"*"`` when the record has no CIGAR).

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     recs = bam.fetch("chr1", 0, 2000)
        >>> isinstance(recs.cigar_string(0), str)
        True
        """
        a, b = self.cigar_off[i], self.cigar_off[i + 1]
        if a == b:
            return "*"
        return "".join(f"{n}{_CIGAR_CHARS[o]}" for o, n in zip(self.cigar_ops[a:b], self.cigar_lens[a:b]))

    def to_frame(self) -> pd.DataFrame:
        """One row per record with SAM-like columns (sequence and CIGAR as strings, qualities as Phred lists).

        Returns
        -------
        pandas.DataFrame with columns name, flag, ref_id, pos, end, mapq, cigar, next_ref_id, next_pos, tlen, seq,
        qual and one column per extracted tag.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     df = bam.fetch("chr1", 0, 2000).to_frame()
        >>> list(df.columns[:4])
        ['name', 'flag', 'ref_id', 'pos']
        """
        n = len(self)
        so = self.seq_off
        seq_bytes = self.seq.tobytes()
        df = pd.DataFrame({
            "name": self.name, "flag": self.flag, "ref_id": self.ref_id, "pos": self.pos, "end": self.end,
            "mapq": self.mapq, "cigar": [self.cigar_string(i) for i in range(n)],
            "next_ref_id": self.next_ref_id, "next_pos": self.next_pos, "tlen": self.tlen,
            "seq": [seq_bytes[so[i]:so[i + 1]].decode("ascii") if so[i + 1] > so[i] else "*" for i in range(n)],
            "qual": [self.qual[so[i]:so[i + 1]].tolist() for i in range(n)],
        })
        for t, v in self.tags.items():
            df[t] = pd.Series(v, dtype=v.dtype if v.dtype != object else object)  # None stays None (pandas 3)
        return df


def _empty_records(tags) -> Records:
    z32 = np.zeros(0, np.int32)
    return Records(z32, z32, z32, np.zeros(0, np.uint8), np.zeros(0, np.uint16), z32, z32, z32,
                   np.zeros(0, object), np.zeros(0, np.uint8), np.zeros(0, np.int64), np.zeros(1, np.int64),
                   np.zeros(0, np.uint8), np.zeros(0, np.uint8), np.zeros(1, np.int64),
                   {t: np.zeros(0, object) for t in tags})


def _concat_records(parts: list, tags) -> Records:
    if not parts:
        return _empty_records(tags)
    if len(parts) == 1:
        return parts[0]

    def offs(name):
        out, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            o = getattr(p, name)
            out.append(o[1:] + base)
            base += o[-1]
        return np.concatenate(out)

    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
    tg = {}
    for t in tags:
        vals = [p.tags[t] for p in parts]
        tg[t] = np.concatenate(vals) if all(v.dtype == vals[0].dtype for v in vals) else \
            np.concatenate([v.astype(object) for v in vals])
    return Records(cat("ref_id"), cat("pos"), cat("end"), cat("mapq"), cat("flag"), cat("next_ref_id"),
                   cat("next_pos"), cat("tlen"), cat("name"), cat("cigar_ops"), cat("cigar_lens"),
                   offs("cigar_off"), cat("seq"), cat("qual"), offs("seq_off"), tg)


def _tag_values(buf, kind, ival, soff, slen, t):
    k = kind[:, t]
    n = len(k)
    if n and np.all(k == 3):
        return ival[:, t].copy()
    out = np.empty(n, object)
    isstr = (k == 1) | (k == 2)
    if isstr.any():  # strings: one gather + one decode, then slices
        idx = np.flatnonzero(isstr)
        ln = slen[idx, t]
        flat = _gather(buf, soff[idx, t], ln)
        text = flat.tobytes().decode("ascii", "replace")
        o = np.concatenate([[0], np.cumsum(ln)]).tolist()
        out[idx] = [text[o[j]:o[j + 1]] for j in range(len(idx))]
    for i in np.flatnonzero(~isstr):
        ki = k[i]
        if ki == 0:
            out[i] = None
        elif ki == 3:
            out[i] = int(ival[i, t])
        elif ki == 4:
            out[i] = struct.unpack_from("<f", buf, soff[i, t])[0]
        else:
            o = soff[i, t]
            st = chr(buf[o])
            (cnt,) = struct.unpack_from("<i", buf, o + 1)
            fmt = {"c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I", "f": "f"}[st]
            out[i] = list(struct.unpack_from(f"<{cnt}{fmt}", buf, o + 5))
    return out


def _records_from(buf, offs, keep_mask, tags) -> Records:
    ref, pos, end, mapq, flag, nref, npos, tlen, lname, ncig, lseq = _parse_fixed(buf, offs)
    keep = np.flatnonzero(keep_mask) if keep_mask is not None else np.arange(len(offs))
    name_off = np.concatenate([[0], np.cumsum(lname[keep] - 1)]).astype(np.int64)
    cig_off = np.concatenate([[0], np.cumsum(ncig[keep])]).astype(np.int64)
    seq_off = np.concatenate([[0], np.cumsum(np.maximum(lseq[keep], 0))]).astype(np.int64)
    name_b, cig, seq, qual = _parse_var(buf, offs, keep, lname, ncig, lseq, name_off, cig_off, seq_off,
                                        _SEQ_ALPHABET)
    text = name_b.tobytes().decode("ascii", "replace")
    no = name_off.tolist()
    names = np.empty(len(keep), object)
    names[:] = [text[no[i]:no[i + 1]] for i in range(len(keep))]
    tg = {}
    if tags:
        t1 = np.array([ord(t[0]) for t in tags], np.uint8)
        t2 = np.array([ord(t[1]) for t in tags], np.uint8)
        kind, ival, fval, soff, slen = _parse_tags(buf, offs, keep, lname, ncig, lseq, t1, t2)
        for j, t in enumerate(tags):
            tg[t] = _tag_values(buf, kind, ival, soff, slen, j)
    return Records(ref[keep], pos[keep], end[keep], mapq[keep], flag[keep], nref[keep], npos[keep], tlen[keep],
                   names, (cig & 15).astype(np.uint8), (cig >> 4).astype(np.int64), cig_off, seq, qual, seq_off,
                   tg)


# ------------------------------------------------------------------------------------------ reader
class _Prefetcher:
    """Background thread: sequential reads of whole BGZF blocks in large chunks into a pool of buffers."""

    def __init__(self, path, start, stop_coff, chunk, nbuf=3):
        self.q = queue.Queue(maxsize=nbuf)
        self.free = queue.Queue()
        for _ in range(nbuf):
            self.free.put(bytearray(chunk + _MAX_BLOCK))
        self.chunk = chunk
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, args=(path, start, stop_coff), daemon=True)
        self.t.start()

    def _run(self, path, start, stop_coff):
        try:
            with open(path, "rb", buffering=0) as f:
                f.seek(start)
                file_off = start
                carry = b""
                while not self._stop.is_set():
                    buf = self.free.get()
                    nc = len(carry)
                    buf[:nc] = carry
                    mv = memoryview(buf)
                    n = nc
                    while n < nc + self.chunk:
                        got = f.readinto(mv[n:nc + self.chunk])
                        if not got:
                            break
                        n += got
                    arr = np.frombuffer(buf, np.uint8)
                    doff, dlen, isz, boff, used, status = _scan_blocks(arr, n)
                    if status < 0:
                        raise ValueError(f"not a BGZF block at file offset {file_off + used}")
                    eof = n < nc + self.chunk
                    if stop_coff is not None and len(boff) and file_off + boff[-1] >= stop_coff:
                        keep = int(np.searchsorted(file_off + boff, stop_coff, side="right"))
                        doff, dlen, isz, boff = doff[:keep], dlen[:keep], isz[:keep], boff[:keep]
                        eof = True
                    self.q.put((buf, doff, dlen, isz, boff, file_off))
                    if eof:
                        if used < n and stop_coff is None and not len(doff):
                            raise ValueError("truncated BGZF file")
                        break
                    carry = bytes(mv[used:n])
                    file_off += used
            self.q.put(None)
        except BaseException as e:  # noqa: BLE001 - handed to the consumer
            self.q.put(e)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def release(self, buf):
        self.free.put(buf)

    def close(self):
        self._stop.set()
        while self.t.is_alive():
            try:
                item = self.q.get(timeout=0.05)
                if isinstance(item, tuple):
                    self.free.put(item[0])
            except queue.Empty:
                pass


class BamFile:
    """A coordinate-sorted BAM file, read natively (the parts of ``pysam.AlignmentFile`` genotype checks use).

    Parameters
    ----------
    path : the ``.bam`` file.
    index : the ``.bai`` path (default: ``path + ".bai"``, else ``path`` with ``.bai`` replacing ``.bam``); needed
        only for region queries.
    threads : inflating threads (default: arsc's thread setting, ``AR_THREADS``).
    inflater : ``None`` (fastest installed) or ``"libdeflate"``, ``"isal"``, ``"zlib"``.
    chunk_mb : compressed bytes read per sequential read (default 32 MB; large reads suit hard disks).

    Attributes
    ----------
    header_text : the SAM header text.
    references, lengths : reference names and lengths.

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> with arsc.bam.BamFile(toy["path"]) as bam:
    ...     n = bam.count()
    >>> n > 0
    True
    """

    def __init__(self, path, index=None, threads: int | None = None, inflater: str | None = None,
                 chunk_mb: float = 32):
        from ._threads import get_threads
        self.path = os.fspath(path)
        self.threads = max(1, int(threads if threads is not None else get_threads()))
        self.inflater_name = globals()["inflater"](inflater)
        self._inflate = _INFLATERS[self.inflater_name]
        self.chunk = int(chunk_mb * 2 ** 20)
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        self._out = np.empty(0, np.uint8)
        self._index_path = index
        self._index = None
        self.file_size = os.path.getsize(self.path)
        self._read_header()

    # -- context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Release the thread pool.

        Returns
        -------
        None

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> bam = arsc.bam.BamFile(toy["path"]); bam.close()
        """
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # -- low level
    def _inflate_into(self, cbuf, doff, dlen, isz, out, start):
        pos = start + np.concatenate([[0], np.cumsum(isz)[:-1]]).astype(np.int64)
        mv = memoryview(cbuf)
        infl = self._inflate

        def work(lo, hi):
            for b in range(lo, hi):
                n = int(isz[b])
                if n == 0:
                    continue
                d = int(doff[b])
                r = infl(mv[d:d + int(dlen[b])], n)
                if len(r) != n:
                    raise ValueError("BGZF block inflated to an unexpected size")
                p = int(pos[b])
                out[p:p + n] = np.frombuffer(r, np.uint8)

        nb_ = len(isz)
        if self._pool is None or nb_ < 8:
            work(0, nb_)
            return
        parts = min(nb_, self.threads * 4)
        cuts = np.linspace(0, nb_, parts + 1).astype(int)
        list(self._pool.map(lambda i: work(cuts[i], cuts[i + 1]), range(parts)))

    def _buffer(self, n):
        if self._out.shape[0] < n:
            self._out = np.empty(int(n * 1.25) + 1024, np.uint8)
        return self._out

    def _batches(self, beg_voff: int, end_voff: int | None):
        """Yield (buf, n, start, limit): decompressed data buf[:n] holding records from offset ``start``; records
        starting at or after ``limit`` lie beyond ``end_voff``. A partial record at the end is carried over."""
        coff, uoff = beg_voff >> 16, beg_voff & 0xFFFF
        stop_coff = None if end_voff is None else end_voff >> 16
        end_uoff = 0 if end_voff is None else end_voff & 0xFFFF
        if stop_coff is not None and end_uoff == 0:
            stop_coff -= 1  # the end block itself is not needed
            if stop_coff < coff:
                return
        span = None if stop_coff is None else stop_coff - coff + _MAX_BLOCK
        chunk = self.chunk if span is None else min(self.chunk, max(span, 1 << 16))
        pf = _Prefetcher(self.path, coff, stop_coff, chunk)
        carry = np.zeros(0, np.uint8)
        start = uoff
        try:
            for cbuf, doff, dlen, isz, boff, file_off in pf:
                total = int(isz.sum())
                nc = len(carry)
                out = self._buffer(nc + total)
                out[:nc] = carry
                self._inflate_into(cbuf, doff, dlen, isz, out, nc)
                limit = 1 << 62  # open: the records continue in the next batch
                if stop_coff is not None and len(boff) and file_off + int(boff[-1]) == (end_voff >> 16) \
                        and end_uoff:
                    limit = nc + total - int(isz[-1]) + end_uoff
                pf.release(cbuf)
                n = nc + total
                yield out, n, start, limit
                carry = out[self._consumed:n].copy() if self._consumed < n else np.zeros(0, np.uint8)
                start = 0
        finally:
            pf.close()

    def _record_batches(self, ranges):
        """Yield (buf, record offsets) for each batch of records in the [beg, end) virtual-offset ranges."""
        for beg, end in ranges:
            done = False
            for buf, n, start, limit in self._batches(int(beg), None if end is None else int(end)):
                if done:
                    self._consumed = n
                    continue
                offs, reached = _find_records(buf, start, n, limit)
                self._consumed = reached
                if reached >= limit:
                    done = True
                if len(offs):
                    yield buf, offs

    def _read_header(self):
        with open(self.path, "rb") as f:
            head = f.read(min(self.file_size, 4 * 2 ** 20))
        arr = np.frombuffer(head, np.uint8)
        data = b""
        coffs, ucum = [], [0]
        pos = 0
        while True:
            doff, dlen, isz, boff, used, status = _scan_blocks(arr[pos:], len(arr) - pos)
            if status < 0 or not len(doff):
                raise ValueError(f"{self.path}: not a BGZF/BAM file")
            for d, l_, s, b in zip(doff, dlen, isz, boff):
                data += self._inflate(head[pos + d:pos + d + l_], int(s)) if s else b""
                coffs.append(pos + int(b))
                ucum.append(len(data))
            if self._header_complete(data) or pos + used >= len(arr):
                break
            pos += used
        if data[:4] != b"BAM\x01":
            raise ValueError(f"{self.path}: not a BAM file (magic {data[:4]!r})")
        (l_text,) = struct.unpack_from("<i", data, 4)
        self.header_text = data[8:8 + l_text].split(b"\0")[0].decode("utf-8", "replace")
        p = 8 + l_text
        (n_ref,) = struct.unpack_from("<i", data, p)
        p += 4
        names, lens = [], []
        for _ in range(n_ref):
            (ln,) = struct.unpack_from("<i", data, p)
            names.append(data[p + 4:p + 4 + ln - 1].decode())
            (lr,) = struct.unpack_from("<i", data, p + 4 + ln)
            lens.append(lr)
            p += 8 + ln
        self.references, self.lengths = names, lens
        self._ref_id = {n: i for i, n in enumerate(names)}
        # virtual offset of the first record
        b = int(np.searchsorted(ucum, p, side="right")) - 1
        self._first_voff = (coffs[b] << 16) | (p - ucum[b]) if b < len(coffs) else (self.file_size << 16)

    @staticmethod
    def _header_complete(data):
        if len(data) < 12:
            return False
        (l_text,) = struct.unpack_from("<i", data, 4)
        p = 8 + l_text
        if len(data) < p + 4:
            return False
        (n_ref,) = struct.unpack_from("<i", data, p)
        p += 4
        for _ in range(n_ref):
            if len(data) < p + 4:
                return False
            (ln,) = struct.unpack_from("<i", data, p)
            p += 8 + ln
            if len(data) < p:
                return False
        return True

    @property
    def index(self):
        """The ``.bai`` index (:func:`read_bai`), loaded on first use."""
        if self._index is None:
            cands = [self._index_path] if self._index_path else [self.path + ".bai",
                                                                 re.sub(r"\.bam$", ".bai", self.path)]
            for c in cands:
                if c and os.path.exists(c):
                    self._index = read_bai(c)
                    break
            else:
                raise FileNotFoundError(f"no .bai index for {self.path} (tried {cands})")
        return self._index

    def _ref(self, contig):
        if isinstance(contig, (int, np.integer)):
            return int(contig)
        if contig not in self._ref_id:
            raise KeyError(f"reference {contig!r} not in the BAM header")
        return self._ref_id[contig]

    def region_chunks(self, regions, gap_kb: float = 0) -> np.ndarray:
        """Merged ``.bai`` chunks (virtual offsets) holding every record that overlaps the regions.

        Parameters
        ----------
        regions : iterable of ``(contig, start, stop)`` (0-based, half-open).
        gap_kb : also join chunks less than this many compressed kilobytes apart (a short read beats a seek).

        Returns
        -------
        ``(n, 2)`` uint64 array of ``[begin, end)`` virtual offsets, sorted and non-overlapping.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     ch = bam.region_chunks([("chr1", 1000, 2000)])
        >>> ch.shape[1]
        2
        """
        idx = self.index
        out = []
        for contig, start, stop in regions:
            r = self._ref(contig)
            if r >= len(idx):
                continue
            ri = idx[r]
            start, stop = max(int(start), 0), int(stop)
            if stop <= start:
                continue
            io = ri.ioffsets
            min_off = int(io[min(start >> 14, len(io) - 1)]) if len(io) else 0
            for b in _reg2bins(start, stop):
                ch = ri.bins.get(b)
                if ch is None:
                    continue
                ch = ch[ch[:, 1] > np.uint64(min_off)]
                if len(ch):
                    out.append(ch)
        if not out:
            return np.zeros((0, 2), np.uint64)
        return _merge_chunks(np.concatenate(out), int(gap_kb * 1024))

    def _parse_region(self, region, start, stop):
        if region is None:
            return None
        if isinstance(region, str) and start is None and stop is None and ":" in region:
            m = re.fullmatch(r"(.+):([\d,]+)(?:-([\d,]+))?", region)
            if m:
                c = m.group(1)
                s = int(m.group(2).replace(",", "")) - 1
                e = int(m.group(3).replace(",", "")) if m.group(3) else self.lengths[self._ref(c)]
                return c, s, e
        r = self._ref(region)
        return region, 0 if start is None else int(start), self.lengths[r] if stop is None else int(stop)

    def fetch(self, contig=None, start: int | None = None, stop: int | None = None,
              tags=("CB", "UB")) -> Records:
        """Records overlapping a region (all records, unmapped ones included, when no region is given).

        Parameters
        ----------
        contig : reference name (or index), or a samtools region string ``"chr1:1,000-2,000"`` (1-based,
            inclusive); ``None`` reads the whole file in file order.
        start, stop : 0-based half-open interval on ``contig`` (default: the whole reference). Overlap is
            htslib's: ``pos < stop and end > start``.
        tags : two-letter tags to extract (e.g. ``("CB", "UB", "xf", "RE")``).

        Returns
        -------
        :class:`Records` (columnar arrays). For a whole large BAM use :meth:`iter_records` instead.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     recs = bam.fetch("chr1:1,001-2,000", tags=("CB", "UB"))
        >>> bool((recs.pos < 2000).all() and (recs.end > 1000).all())
        True
        """
        return _concat_records(list(self.iter_records(contig, start, stop, tags=tags)), tags)

    def iter_records(self, contig=None, start: int | None = None, stop: int | None = None, tags=("CB", "UB")):
        """Stream records batch by batch (bounded memory), as :class:`Records`.

        Parameters
        ----------
        contig, start, stop : as :meth:`fetch` (``None``: the whole file).
        tags : tags to extract.

        Yields
        ------
        :class:`Records`, one per decompressed batch.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     n = sum(len(r) for r in bam.iter_records(tags=()))
        >>> n == toy["n_records"]
        True
        """
        tags = tuple(tags)
        reg = self._parse_region(contig, start, stop)
        if reg is None:
            ranges = [(self._first_voff, None)]
        else:
            c, s, e = reg
            ranges = self.region_chunks([(c, s, e)])
            r = self._ref(c)
        for buf, offs in self._record_batches(ranges):
            if reg is None:
                yield _records_from(buf, offs, None, tags)
            else:
                ref, pos, end = _parse_fixed(buf, offs)[:3]
                keep = (ref == r) & (pos < e) & (end > s)
                if keep.any():
                    yield _records_from(buf, offs, keep, tags)

    def count(self, contig=None, start: int | None = None, stop: int | None = None, exclude_flags: int = 0,
              min_mapq: int = 0) -> int:
        """Number of records (like ``samtools view -c``), optionally filtered.

        Parameters
        ----------
        contig, start, stop : region as :meth:`fetch` (``None``: the whole file, unmapped records included).
        exclude_flags : skip records with any of these flag bits (e.g. ``0x904``).
        min_mapq : skip records with a lower MAPQ.

        Returns
        -------
        int

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     bam.count() == toy["n_records"]
        True
        """
        return int(self.count_per_reference(contig, start, stop, exclude_flags, min_mapq).sum())

    def count_per_reference(self, contig=None, start=None, stop=None, exclude_flags: int = 0,
                            min_mapq: int = 0) -> pd.Series:
        """Records per reference (last entry ``"*"``: unplaced reads), like ``samtools idxstats`` by reading.

        Parameters
        ----------
        contig, start, stop : region as :meth:`fetch` (``None``: the whole file).
        exclude_flags : skip records with any of these flag bits.
        min_mapq : skip records with a lower MAPQ.

        Returns
        -------
        pandas.Series indexed by reference name.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> with arsc.bam.BamFile(toy["path"]) as bam:
        ...     s = bam.count_per_reference()
        >>> list(s.index)
        ['chr1', 'chr2', '*']
        """
        counts = np.zeros(len(self.references) + 1, np.int64)
        reg = self._parse_region(contig, start, stop)
        if reg is None:
            for buf, offs in self._record_batches([(self._first_voff, None)]):
                _count_kernel(buf, offs, exclude_flags, min_mapq, counts)
        else:
            c, s, e = reg
            r = self._ref(c)
            for buf, offs in self._record_batches(self.region_chunks([(c, s, e)])):
                ref, pos, end, mapq, flag = _parse_fixed(buf, offs)[:5]
                keep = (ref == r) & (pos < e) & (end > s) & ((flag & exclude_flags) == 0) & (mapq >= min_mapq)
                counts[r] += int(keep.sum())
        return pd.Series(counts, index=self.references + ["*"])


# ------------------------------------------------------------------------------------------ pileup
def _barcode_table(barcodes):
    bcs = [str(b) for b in barcodes]
    enc = [b.encode() for b in bcs]
    off = np.concatenate([[0], np.cumsum([len(e) for e in enc])]).astype(np.int64)
    flat = np.frombuffer(b"".join(enc), np.uint8).copy() if enc else np.zeros(0, np.uint8)
    h = _hash_strings(flat, off)
    order = np.argsort(h, kind="stable")
    return bcs, h[order], order.astype(np.int64), flat, off


def _site_table(bam: BamFile, sites: pd.DataFrame):
    s = sites.reset_index(drop=True)
    for c in ("chrom", "pos"):
        if c not in s.columns:
            raise ValueError("sites need columns 'chrom' and 'pos' (1-based); optional 'ref', 'alt'")
    known = s["chrom"].astype(str).isin(bam.references)
    ref_id = np.where(known, s["chrom"].astype(str).map(bam._ref_id).fillna(-1), -1).astype(np.int64)
    pos0 = s["pos"].to_numpy(np.int64) - 1
    order = np.lexsort((pos0, ref_id))
    order = order[ref_id[order] >= 0]
    nref = len(bam.references)
    sref = ref_id[order]
    lo = np.searchsorted(sref, np.arange(nref), side="left").astype(np.int64)
    hi = np.searchsorted(sref, np.arange(nref), side="right").astype(np.int64)
    return s, order, pos0[order].astype(np.int64), lo, hi, sref


@dataclass
class AlleleCounts:
    """Per-cell allele counts at SNPs (the output of :func:`pileup_cells`).

    Attributes
    ----------
    sites : the input sites (rows in input order), with columns chrom, pos (1-based), ref, alt when given.
    barcodes : list of cell barcodes (columns of the matrices; ``["pooled"]`` in pooled mode).
    counts : dict base -> sparse CSR ``sites x cells`` matrix of UMI counts for ``"A"``, ``"C"``, ``"G"``, ``"T"``,
        ``"N"`` (N = any non-ACGT base).
    AD, DP, OTH : CSR ``sites x cells`` (cellsnp-lite layout): ALT UMIs, REF + ALT UMIs, other-allele UMIs; ``None``
        when the sites have no ref / alt columns.
    stats : dict of read counters (reads overlapping a site before filters, missing cell tag, barcode not in the
        list, missing UMI tag, bases below the quality cut, UMIs dropped for an allele tie).

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
    >>> int(ac.DP.sum()) == ac.stats["umis"] - int(ac.OTH.sum())
    True
    """
    sites: pd.DataFrame
    barcodes: list
    counts: dict
    AD: sp.csr_matrix | None
    DP: sp.csr_matrix | None
    OTH: sp.csr_matrix | None
    stats: dict

    def long(self) -> pd.DataFrame:
        """Non-zero (site, cell) pairs as a tidy table.

        Returns
        -------
        pandas.DataFrame with columns site (row of ``sites``), chrom, pos, barcode, A, C, G, T, N and, with ref /
        alt, AD, DP, OTH.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
        >>> {"barcode", "AD", "DP"} <= set(ac.long().columns)
        True
        """
        tot = sum(self.counts.values()).tocoo()
        r, c = tot.row, tot.col
        df = pd.DataFrame({"site": r, "chrom": self.sites["chrom"].to_numpy()[r],
                           "pos": self.sites["pos"].to_numpy()[r], "barcode": np.asarray(self.barcodes)[c]})
        for b, m in self.counts.items():
            df[b] = np.asarray(m.tocsr()[r, c]).ravel()
        if self.AD is not None:
            for name in ("AD", "DP", "OTH"):
                df[name] = np.asarray(getattr(self, name)[r, c]).ravel()
        return df.sort_values(["site", "barcode"], kind="stable").reset_index(drop=True)

    def aggregate(self, groups) -> pd.DataFrame:
        """Sum the counts of cells per group (sample, donor, hashtag): one pseudo-bulk allele table per group.

        Parameters
        ----------
        groups : mapping barcode -> group label (dict or pandas Series); cells not in it are left out.

        Returns
        -------
        pandas.DataFrame, one row per group x site with depth > 0: group, chrom, pos, (ref, alt), A, C, G, T, N,
        n_cells and, with ref / alt, AD, DP, OTH. Input for :func:`genotype_concordance`.

        Examples
        --------
        >>> import os, tempfile, arsc
        >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
        >>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
        >>> agg = ac.aggregate(toy["donor"])
        >>> sorted(agg["group"].unique())
        ['D1', 'D2', 'D3']
        """
        g = pd.Series(groups)
        lab = g.reindex(pd.Index(self.barcodes)).to_numpy()
        ok = pd.notna(lab)
        cats = pd.Index(pd.unique(lab[ok]))
        col = cats.get_indexer(lab[ok])
        G = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), col)), shape=(len(self.barcodes), len(cats)))
        out = {}
        for b, m in self.counts.items():
            out[b] = np.asarray((m @ G).todense())
        anyc = sum((m > 0).astype(np.int64) for m in self.counts.values())
        anyc = np.asarray(((anyc > 0).astype(np.int64) @ G).todense())
        tot = sum(out.values())
        r, c = np.nonzero(tot)
        base_cols = [x for x in ("chrom", "pos", "ref", "alt") if x in self.sites.columns]
        df = self.sites.iloc[r][base_cols].reset_index(drop=True)
        df.insert(0, "group", cats.to_numpy()[c])
        for b in out:
            df[b] = out[b][r, c].astype(np.int64)
        df["n_cells"] = anyc[r, c].astype(np.int64)
        if self.AD is not None:
            ad = np.asarray((self.AD @ G).todense())
            dp = np.asarray((self.DP @ G).todense())
            oth = np.asarray((self.OTH @ G).todense())
            df["AD"], df["DP"], df["OTH"] = ad[r, c].astype(np.int64), dp[r, c].astype(np.int64), \
                oth[r, c].astype(np.int64)
        return df.sort_values(["group", "chrom", "pos"], kind="stable").reset_index(drop=True)


def pileup_cells(bam, sites: pd.DataFrame, barcodes=None, *, cell_tag: str | None = "CB",
                 umi_tag: str | None = "UB", min_mapq: int = 20, min_baseq: int = 20, min_len: int = 30,
                 exclude_flags: int | None = None, umi_rule: str = "majority", threads: int | None = None,
                 gap_kb: float = 256, inflater: str | None = None) -> AlleleCounts:
    """Count alleles per cell barcode at SNP positions, one count per UMI (cellsnp-lite mode 1 style).

    For every read covering a site: the read must pass the flag / MAPQ / aligned-length filters, the base at the
    site must have quality >= ``min_baseq``, the site must fall in an M / = / X block of the CIGAR (a site in a
    deletion ``D`` or intron ``N`` is not covered; insertions and soft clips shift the read position), the read
    must carry ``cell_tag`` with a barcode in ``barcodes`` and ``umi_tag``. Reads are then collapsed per
    (site, cell, UMI): ``umi_rule="majority"`` keeps the most frequent base of the UMI's reads (a UMI whose reads
    tie between bases is dropped and counted in ``stats["umi_ties"]``), ``"first"`` keeps the first read's base.

    Parameters
    ----------
    bam : path of a coordinate-sorted, indexed BAM, or an open :class:`BamFile`.
    sites : DataFrame with ``chrom`` and 1-based ``pos`` (VCF convention); optional ``ref`` and ``alt`` give
        AD / DP / OTH. Sites on references absent from the BAM get zero counts.
    barcodes : cell barcodes to count (e.g. the filtered barcodes of Cell Ranger). ``None`` pools all reads that
        carry ``cell_tag`` into one column (``"pooled"``), UMIs then keyed by cell barcode + UMI.
    cell_tag : cell barcode tag (``None``: ignore barcodes, pool all reads).
    umi_tag : UMI tag (``None``: count reads, no UMI collapsing).
    min_mapq : minimum MAPQ (default 20, as cellsnp-lite; STAR / Cell Ranger unique reads have 255).
    min_baseq : minimum base quality at the site (default 20).
    min_len : minimum aligned length (bases in M / = / X; default 30, as cellsnp-lite).
    exclude_flags : reads with any of these flag bits are skipped. Default: unmapped, secondary, QC-fail
        (``0x304``) with a UMI tag (duplicates are collapsed by UMI, as cellsnp-lite) and also duplicates
        (``0x704``) without one.
    umi_rule : ``"majority"`` (default) or ``"first"``.
    threads : inflating threads (default ``AR_THREADS``).
    gap_kb : index chunks closer than this many compressed kilobytes are read in one go (default 256: on a hard
        disk reading a few hundred kB is cheaper than a seek). Dense site lists end up reading the BAM front to back.
    inflater : DEFLATE engine (see :func:`inflater`).

    Returns
    -------
    :class:`AlleleCounts` (sparse ``sites x cells`` matrices AD / DP / OTH and per-base counts; ``.long()``,
    ``.aggregate(groups)``).

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
    >>> ac.AD.shape == (len(toy["sites"]), len(toy["barcodes"]))
    True
    """
    if umi_rule not in ("majority", "first"):
        raise ValueError("umi_rule must be 'majority' or 'first'")
    own = not isinstance(bam, BamFile)
    f = BamFile(bam, threads=threads, inflater=inflater) if own else bam
    try:
        s, order, spos, lo, hi, sref = _site_table(f, sites)
        if exclude_flags is None:
            exclude_flags = (FUNMAP | FSECONDARY | FQCFAIL) | (0 if umi_tag else FDUP)
        use_cell = cell_tag is not None
        pooled = barcodes is None
        if pooled:
            bcs = ["pooled"]
            bc_hash, bc_idx, bc_flat, bc_off = (np.zeros(0, np.uint64), np.zeros(0, np.int64),
                                                np.zeros(0, np.uint8), np.zeros(1, np.int64))
        else:
            if not use_cell:
                raise ValueError("barcodes given but cell_tag is None")
            bcs, bc_hash, bc_idx, bc_flat, bc_off = _barcode_table(barcodes)
        ct = (cell_tag or "CB").encode()
        ut = (umi_tag or "UB").encode()
        regions = [(f.references[r], int(p), int(p) + 1) for r, p in zip(sref, spos)]
        chunks = f.region_chunks(regions, gap_kb=gap_kb) if regions else np.zeros((0, 2), np.uint64)
        cap = 1 << 20
        bufs = dict(site=np.empty(cap, np.int64), cell=np.empty(cap, np.int32), key=np.empty(cap, np.uint64),
                    ckey=np.empty(cap, np.uint64), base=np.empty(cap, np.int8))
        parts = {k: [] for k in bufs}
        stats = np.zeros(5, np.int64)
        rec_base = 0
        for buf, offs in f._record_batches(chunks):
            k = 0
            while True:
                k, nout = _pileup_kernel(buf, offs, k, lo, hi, spos, exclude_flags, min_mapq, min_baseq, min_len,
                                         ct[0], ct[1], ut[0], ut[1], use_cell, umi_tag is not None,
                                         pooled and use_cell, bc_hash, bc_idx, bc_flat, bc_off, rec_base,
                                         bufs["site"], bufs["cell"], bufs["key"], bufs["ckey"], bufs["base"],
                                         0, stats)
                for name in bufs:
                    parts[name].append(bufs[name][:nout].copy())
                if k >= len(offs):
                    break
            rec_base += len(offs)
        obs = {k: (np.concatenate(v) if v else np.zeros(0, bufs[k].dtype)) for k, v in parts.items()}
        # sort by (site, cell, cell key, umi) keeping file order within groups; then collapse UMIs
        srt = np.lexsort((obs["key"], obs["ckey"], obs["cell"], obs["site"]))
        rs, rc, rb, ties = _collapse_umis(obs["site"], obs["cell"], obs["ckey"], obs["key"], obs["base"], srt,
                                          0 if umi_rule == "majority" else 1)
        ns, nc = len(s), len(bcs)
        rows = order[rs] if len(rs) else np.zeros(0, np.int64)  # back to input site order
        counts = {}
        for bi, b in enumerate("ACGTN"):
            m = rb == bi
            counts[b] = sp.csr_matrix((np.ones(int(m.sum()), np.int64), (rows[m], rc[m])), shape=(ns, nc))
            counts[b].sum_duplicates()
        AD = DP = OTH = None
        if "ref" in s.columns and "alt" in s.columns:
            ref = s["ref"].astype(str).str.upper().to_numpy()
            alt = s["alt"].astype(str).str.upper().to_numpy()
            refc = sp.csr_matrix((ns, nc), dtype=np.int64)
            altc = sp.csr_matrix((ns, nc), dtype=np.int64)
            tot = sum(counts.values())
            for b in "ACGTN":
                dr = sp.diags((ref == b).astype(np.int64), dtype=np.int64)
                da = sp.diags((alt == b).astype(np.int64), dtype=np.int64)
                refc = refc + dr @ counts[b]
                altc = altc + da @ counts[b]
            AD = sp.csr_matrix(altc)
            DP = sp.csr_matrix(refc + altc)
            OTH = sp.csr_matrix(tot - refc - altc)
            for m in (AD, DP, OTH):
                m.eliminate_zeros()
        st = dict(zip(["reads_at_sites", "no_cell_tag", "cell_not_listed", "no_umi_tag", "low_baseq"],
                      map(int, stats)))
        st["observations"] = int(len(srt))
        st["umi_ties"] = int(ties)
        st["umis"] = int(len(rs))
        return AlleleCounts(s, bcs, counts, AD, DP, OTH, st)
    finally:
        if own:
            f.close()


def discover_snps(bam, *, regions=None, barcodes=None, cell_tag: str = "CB", min_depth: int = 20,
                  min_alt: int = 4, min_maf: float = 0.1, min_top2: float = 0.0, min_mapq: int = 20,
                  min_baseq: int = 20, min_len: int = 30, exclude_flags: int = FUNMAP | FSECONDARY | FQCFAIL | FDUP,
                  threads: int | None = None, window: int = 1 << 23, inflater: str | None = None) -> pd.DataFrame:
    """Find candidate SNPs de novo: pooled pileup of every covered position (cellsnp-lite mode 2 style).

    Counts reads (not UMIs: duplicates are excluded by flag, which in Cell Ranger BAMs leaves one read per UMI)
    with A / C / G / T / N at every position, over all reads or only those of listed barcodes, and keeps
    positions with depth (A + C + G + T) >= ``min_depth`` whose second most frequent base has >= ``min_alt`` reads
    and >= ``min_maf`` of the depth. Without a reference genome the alleles are the pooled major (``ref``) and
    minor (``alt``) bases, not the genome's reference allele. Feed the result to :func:`pileup_cells`.

    Parameters
    ----------
    bam : path of a coordinate-sorted BAM (or an open :class:`BamFile`).
    regions : ``None`` (the whole file) or a list of reference names / ``(contig, start, stop)`` regions.
    barcodes : count only reads whose ``cell_tag`` is in this list (``None``: all reads).
    cell_tag : the cell barcode tag used with ``barcodes``.
    min_depth, min_alt, min_maf : site thresholds (defaults 20, 4, 0.1).
    min_top2 : the two top bases must hold at least this fraction of the depth (e.g. 0.95; default 0, off).
    min_mapq, min_baseq, min_len, exclude_flags : read and base filters as in :func:`pileup_cells`.
    threads : inflating threads.
    window : positions held in the pileup ring buffer (default 8.4 Mb); bases of a read more than this beyond the
        read start (very long introns) are dropped and counted in ``df.attrs["dropped_bases"]``.
    inflater : DEFLATE engine.

    Returns
    -------
    pandas.DataFrame: chrom, pos (1-based), ref (major), alt (minor), A, C, G, T, N, depth, maf.

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> snps = arsc.bam.discover_snps(toy["path"], min_depth=10)
    >>> len(snps) > 0
    True
    """
    own = not isinstance(bam, BamFile)
    f = BamFile(bam, threads=threads, inflater=inflater) if own else bam
    try:
        nref = len(f.references)
        ref_ok = np.ones(nref, np.bool_)
        if regions is None:
            ranges = [(f._first_voff, None)]
            bounds = None
        else:
            regs = [(r, 0, f.lengths[f._ref(r)]) if isinstance(r, (str, int, np.integer)) else tuple(r)
                    for r in regions]
            ref_ok[:] = False
            for r in regs:
                ref_ok[f._ref(r[0])] = True
            ranges = f.region_chunks(regs)
            bounds = regs
        if barcodes is not None:
            _, bc_hash, bc_idx, bc_flat, bc_off = _barcode_table(barcodes)
        else:
            bc_hash, bc_idx, bc_flat, bc_off = (np.zeros(0, np.uint64), np.zeros(0, np.int64),
                                                np.zeros(0, np.uint8), np.zeros(1, np.int64))
        ct = cell_tag.encode()
        W = int(window)
        ring = np.zeros(W * 5, np.uint32)
        state = np.array([-1, 0, 0, 0, 0], np.int64)
        o_ref = np.empty(1 << 16, np.int64)
        o_pos = np.empty(1 << 16, np.int64)
        o_cnt = np.empty((1 << 16, 5), np.int64)
        k = 0
        for buf, offs in f._record_batches(ranges):
            o_ref, o_pos, o_cnt, k = _discover_kernel(buf, offs, ring, W, state, exclude_flags, min_mapq,
                                                      min_baseq, min_len, ct[0], ct[1], barcodes is not None,
                                                      bc_hash, bc_idx, bc_flat, bc_off, ref_ok, min_depth, min_alt,
                                                      float(min_maf), float(min_top2), o_ref, o_pos, o_cnt, k)
            if state[4]:
                raise ValueError(f"{f.path} is not coordinate-sorted")
        if state[0] >= 0:
            o_ref, o_pos, o_cnt, k = _flush(ring, W, state[0], state[1], state[2], min_depth, min_alt,
                                            float(min_maf), float(min_top2), o_ref, o_pos, o_cnt, k)
        cnt = o_cnt[:k]
        df = pd.DataFrame({"chrom": np.asarray(f.references, dtype=object)[o_ref[:k]], "pos": o_pos[:k] + 1})
        acgt = cnt[:, :4]
        top = np.argsort(-acgt, axis=1, kind="stable")
        df["ref"] = _BASES[top[:, 0]] if k else np.zeros(0, object)
        df["alt"] = _BASES[top[:, 1]] if k else np.zeros(0, object)
        for i, b in enumerate("ACGTN"):
            df[b] = cnt[:, i]
        df["depth"] = acgt.sum(1)
        df["maf"] = np.take_along_axis(acgt, top[:, 1:2], 1).ravel() / np.maximum(df["depth"].to_numpy(), 1) \
            if k else np.zeros(0)
        if bounds is not None and k:
            keep = np.zeros(k, bool)
            for c, s0, e0 in bounds:
                keep |= (df["chrom"].to_numpy() == c) & (df["pos"].to_numpy() > s0) & (df["pos"].to_numpy() <= e0)
            df = df[keep]
        df = df.reset_index(drop=True)
        df.attrs["dropped_bases"] = int(state[3])
        return df
    finally:
        if own:
            f.close()


# ------------------------------------------------------------------------------------------ genotype comparison
def _calls(af, het):
    return np.where(af < het[0], 0, np.where(af > het[1], 2, 1))


def sample_allele_fractions(table: pd.DataFrame, *, group: str = "group", min_depth: int = 10) -> pd.DataFrame:
    """Wide table of ALT allele fractions (sites x samples), NaN where a sample's depth is below ``min_depth``.

    Parameters
    ----------
    table : long table with columns ``group`` (sample), chrom, pos, AD, DP (e.g. :meth:`AlleleCounts.aggregate`,
        or several samples' pooled :func:`pileup_cells` results concatenated with a group column).
    group : the sample column.
    min_depth : minimum REF + ALT UMIs for a sample's fraction to count.

    Returns
    -------
    pandas.DataFrame indexed by ``chrom:pos``, one column per sample.

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
    >>> af = arsc.bam.sample_allele_fractions(ac.aggregate(toy["donor"]), min_depth=5)
    >>> list(af.columns)
    ['D1', 'D2', 'D3']
    """
    t = table[table["DP"] >= min_depth]
    key = t["chrom"].astype(str) + ":" + t["pos"].astype(str)
    af = (t["AD"] / t["DP"]).to_numpy()
    w = pd.DataFrame({"site": key.to_numpy(), "group": t[group].to_numpy(), "af": af})
    w = w.pivot_table(index="site", columns="group", values="af", aggfunc="first")
    w.columns.name = None
    w.index.name = None
    return w


def genotype_concordance(table: pd.DataFrame, *, group: str = "group", min_depth: int = 10,
                         het: tuple = (0.1, 0.9)) -> pd.DataFrame:
    """Pairwise genotype agreement between samples at shared sites (same donor or not).

    Each sample's ALT fraction at a site with depth >= ``min_depth`` is called 0 (< ``het[0]``), 1 (het) or 2
    (> ``het[1]``). For each pair: the sites both samples cover, the informative ones among them (at least one
    sample not homozygous REF: 0 / 0 sites agree for any two people), the fraction of identical calls on both, the
    opposite-homozygous sites (0 vs 2: impossible for one donor short of errors) and the Pearson correlation of
    the fractions. Same donor: concordance near 1, no opposite homozygotes, correlation near 1 (lower at shallow
    depth: compare against pairs known to be the same donor and pairs known to differ in the same data). Different
    donors: lower concordance, opposite homozygotes, low or (at sites chosen as polymorphic in the pool) negative
    correlation.

    Parameters
    ----------
    table : long table with ``group``, chrom, pos, AD, DP (see :func:`sample_allele_fractions`).
    group : the sample column.
    min_depth : minimum REF + ALT UMIs per sample and site.
    het : allele-fraction bounds of a heterozygous call.

    Returns
    -------
    pandas.DataFrame, one row per sample pair: sample_a, sample_b, n_shared, n_informative, concordance (all shared
    sites), concordance_informative, n_opposite_hom, af_pearson.

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
    >>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
    >>> agg = ac.aggregate(toy["donor"].str.cat(toy["half"], sep="_"))
    >>> pairs = arsc.bam.genotype_concordance(agg, min_depth=5)
    >>> same = pairs[pairs.sample_a.str[:2] == pairs.sample_b.str[:2]]
    >>> bool((same.concordance_informative > 0.9).all())
    True
    """
    af = sample_allele_fractions(table, group=group, min_depth=min_depth)
    names = list(af.columns)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = af[names[i]].to_numpy(), af[names[j]].to_numpy()
            m = ~np.isnan(a) & ~np.isnan(b)
            ca, cb = _calls(a[m], het), _calls(b[m], het)
            inf = (ca > 0) | (cb > 0)
            n, ni = int(m.sum()), int(inf.sum())
            r = np.corrcoef(a[m], b[m])[0, 1] if n >= 3 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan
            rows.append(dict(sample_a=names[i], sample_b=names[j], n_shared=n, n_informative=ni,
                             concordance=float(np.mean(ca == cb)) if n else np.nan,
                             concordance_informative=float(np.mean(ca[inf] == cb[inf])) if ni else np.nan,
                             n_opposite_hom=int(np.sum(np.abs(ca - cb) == 2)), af_pearson=float(r)))
    return pd.DataFrame(rows, columns=["sample_a", "sample_b", "n_shared", "n_informative", "concordance",
                                       "concordance_informative", "n_opposite_hom", "af_pearson"])


# ------------------------------------------------------------------------------------------ writer
def _bgzf_block(data: bytes, level: int) -> bytes:
    c = zlib.compressobj(level, zlib.DEFLATED, -15)
    cdata = c.compress(data) + c.flush()
    bsize = len(cdata) + 25
    return (b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00" + struct.pack("<H", bsize)
            + cdata + struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data)))


_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")
_SEQ_CODE = {c: i for i, c in enumerate("=ACMGRSVTWYHKDBN")}


def _encode_record(rec: dict, ref_id: dict) -> bytes:
    name = rec.get("name", "r").encode() + b"\0"
    cig = re.findall(r"(\d+)([MIDNSHP=X])", rec.get("cigar", "*") if rec.get("cigar", "*") != "*" else "")
    cig_words = [int(n) << 4 | _CIGAR_CHARS.index(o) for n, o in cig]
    seq = rec.get("seq", "*")
    seq = "" if seq == "*" else seq
    ls = len(seq)
    codes = [_SEQ_CODE.get(c, 15) for c in seq.upper()] + [0]
    sbytes = bytes((codes[i] << 4) | codes[i + 1] for i in range(0, ls, 2)) if ls else b""
    qual = rec.get("qual")
    qbytes = bytes([255] * ls) if qual is None else bytes(qual)
    tags = b""
    for t, v in rec.get("tags", {}).items():
        if isinstance(v, str) and len(v) == 1 and t in ("RE",):
            tags += t.encode() + b"A" + v.encode()
        elif isinstance(v, str):
            tags += t.encode() + b"Z" + v.encode() + b"\0"
        elif isinstance(v, float):
            tags += t.encode() + b"f" + struct.pack("<f", v)
        else:
            tags += t.encode() + b"i" + struct.pack("<i", int(v))
    r = ref_id.get(rec.get("chrom"), -1) if rec.get("chrom") not in (None, "*") else -1
    pos = int(rec.get("pos", -1))
    flag = int(rec.get("flag", 0))
    rl = sum(int(n) for n, o in cig if o in "MDN=X")
    end = pos + rl if (rl and not flag & FUNMAP) else pos + 1
    b = _reg2bin(pos, end) if pos >= 0 else 4680
    nr = ref_id.get(rec.get("next_chrom"), -1) if rec.get("next_chrom") not in (None, "*") else -1
    body = struct.pack("<iiBBHHHiiii", r, pos, len(name), int(rec.get("mapq", 255)), b, len(cig_words), flag, ls,
                       nr, int(rec.get("next_pos", -1)), int(rec.get("tlen", 0)))
    body += name + struct.pack(f"<{len(cig_words)}I", *cig_words) + sbytes + qbytes + tags
    return struct.pack("<i", len(body)) + body


def write_bam(path, references, records, *, header_text: str | None = None, level: int = 6,
              block_bytes: int = 4096, index: bool = True) -> str:
    """Write a small coordinate-sorted BAM (and its ``.bai``) from Python dictionaries.

    Meant for tests, documentation and synthetic data, not for large outputs.

    Parameters
    ----------
    path : output ``.bam`` path.
    references : list of ``(name, length)``.
    records : iterable of dicts sorted by (reference, pos) with keys name, flag, chrom, pos (0-based), mapq,
        cigar (SAM string), seq, qual (list of Phred values, optional), next_chrom, next_pos, tlen and tags (dict
        tag -> str / int / float; a one-letter ``RE`` value is written as type ``A``). Unplaced records
        (``chrom=None``) go last.
    header_text : SAM header text (default: ``@HD`` + one ``@SQ`` line per reference).
    level : zlib compression level.
    block_bytes : uncompressed bytes per BGZF block (small blocks exercise multi-block reading).
    index : also write ``path + ".bai"``.

    Returns
    -------
    str, the BAM path.

    Examples
    --------
    >>> import os, tempfile, arsc
    >>> p = os.path.join(tempfile.mkdtemp(), "one.bam")
    >>> rec = dict(name="r1", chrom="chr1", pos=99, mapq=255, cigar="10M", seq="ACGTACGTAC", tags={"CB": "AAAC-1"})
    >>> _ = arsc.bam.write_bam(p, [("chr1", 1000)], [rec])
    >>> arsc.bam.BamFile(p).fetch("chr1", 100, 101).tags["CB"].tolist()
    ['AAAC-1']
    """
    path = os.fspath(path)
    ref_id = {n: i for i, (n, _) in enumerate(references)}
    if header_text is None:
        header_text = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(f"@SQ\tSN:{n}\tLN:{ln}\n" for n, ln in references)
    ht = header_text.encode()
    head = b"BAM\x01" + struct.pack("<i", len(ht)) + ht + struct.pack("<i", len(references))
    for n, ln in references:
        nb_ = n.encode() + b"\0"
        head += struct.pack("<i", len(nb_)) + nb_ + struct.pack("<i", int(ln))
    out = bytearray()
    pending = bytearray(head)
    voffs = []  # (ref, pos, end, bin, voff_begin, voff_end) per record

    def flush():
        nonlocal pending
        if pending:
            out.extend(_bgzf_block(bytes(pending), level))
            pending = bytearray()

    flush()  # header in its own block(s) so record offsets are simple
    for rec in records:
        enc = _encode_record(rec, ref_id)
        if len(pending) + len(enc) > block_bytes and pending:
            flush()
        beg = (len(out) << 16) | len(pending)
        pending.extend(enc)
        r, pos, flag = struct.unpack_from("<ii", enc, 4) + struct.unpack_from("<H", enc, 18)
        (bn,) = struct.unpack_from("<H", enc, 14)
        nc = struct.unpack_from("<H", enc, 16)[0]
        ln = enc[12]
        words = struct.unpack_from(f"<{nc}I", enc, 36 + ln)
        rl = sum(w >> 4 for w in words if (w & 15) in (0, 2, 3, 7, 8))
        end = pos + rl if (rl and not flag & FUNMAP) else pos + 1
        voffs.append([r, pos, end, bn, beg, len(enc)])
    flush()
    out.extend(_BGZF_EOF)
    with open(path, "wb") as fh:
        fh.write(out)
    if index:
        _write_bai(path + ".bai", out, voffs, len(references))
    return path


def _write_bai(path, data, recs, n_ref):
    # recover each record's end virtual offset from the stored (begin, length) and the block layout
    blocks = []
    p = 0
    arr = np.frombuffer(bytes(data), np.uint8)
    doff, dlen, isz, boff, _, _ = _scan_blocks(arr, len(arr))
    coff_list = boff.tolist()
    isz_list = isz.tolist()

    def end_voff(beg, length):
        c, u = beg >> 16, beg & 0xFFFF
        i = coff_list.index(c)
        u += length
        while i < len(isz_list) - 1 and u >= isz_list[i]:
            if u == isz_list[i] and i + 1 < len(coff_list):
                u -= isz_list[i]
                i += 1
                break
            u -= isz_list[i]
            i += 1
        return (coff_list[i] << 16) | u

    del blocks, p
    per_ref = [dict(bins={}, lin={}) for _ in range(n_ref)]
    for r, pos, end, bn, beg, length in recs:
        if r < 0:
            continue
        ev = end_voff(beg, length)
        ix = per_ref[r]
        ch = ix["bins"].setdefault(bn, [])
        if ch and ch[-1][1] == beg:
            ch[-1][1] = ev
        else:
            ch.append([beg, ev])
        for w in range(pos >> 14, ((end - 1) >> 14) + 1):
            if w not in ix["lin"] or beg < ix["lin"][w]:
                ix["lin"][w] = beg
    out = bytearray(b"BAI\x01" + struct.pack("<i", n_ref))
    for ix in per_ref:
        out += struct.pack("<i", len(ix["bins"]))
        for bn in sorted(ix["bins"]):
            ch = ix["bins"][bn]
            out += struct.pack("<Ii", bn, len(ch))
            for b, e in ch:
                out += struct.pack("<QQ", b, e)
        lin = ix["lin"]
        n_intv = (max(lin) + 1) if lin else 0
        vals, last = [], 0
        for w in range(n_intv):
            if w in lin:
                last = lin[w]
            vals.append(last)
        # windows without reads take the next window's offset (htslib fills forward; either is a valid bound)
        for w in range(n_intv - 1, -1, -1):
            if w not in lin and w + 1 < n_intv:
                vals[w] = vals[w + 1]
        out += struct.pack("<i", n_intv) + struct.pack(f"<{n_intv}Q", *vals)
    with open(path, "wb") as fh:
        fh.write(out)
