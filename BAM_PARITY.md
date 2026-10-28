# arsc.bam: parity with pysam / htslib

`arsc.bam` reads BAM files and counts alleles per cell natively (numba kernels, BGZF inflated in threads with
libdeflate), so genotype checks (sample mix-ups, donor identity of hashtag groups) run on Windows without pysam,
htslib, cellsnp-lite or a Linux container. It is written from the SAM/BAM specification; no htslib or pysam code is
used.

Reference: pysam 0.22.1 (htslib 1.18) in a Linux container (`benchmarks/bam_pysam_ref.py`); arsc side and comparison:
`benchmarks/bam_parity.py`. Data: one Cell Ranger 10.1 `multi` per-sample BAM (`sample_alignments.bam`; gene expression + antibody capture)
(9.15 GB, 147,131,128 records, 194 references, 229 filtered cell barcodes), on a hard disk. SNPs: 28,454 exonic
sites found de novo in another library of the same study (depth >= 20, minor allele >= 4 reads and >= 10%); the
1,000-site set is a random subset of them (seed 0). Synthetic checks: `tests/test_bam.py` (26 tests, ~15 s).

## Agreement

| arsc | pysam reference | Data | Agreement |
|---|---|---|---|
| `BamFile.count_per_reference()` (whole file) | `fetch(until_eof=True)`, records per `reference_id` | 147,131,128 records | identical for all 194 references and the unplaced reads |
| `BamFile.fetch(region, tags=("CB","UB","xf","RE"))` | `fetch(region=...)` | chr12:6,530,000-6,540,000 (GAPDH; 86,537 records, spliced) | identical record set and order; every field identical: name, flag, ref_id, pos, end (`reference_end`), mapq, CIGAR, mate ref / pos, tlen, sequence, base qualities, CB, UB, xf, RE |
| `pileup_cells(bam, sites, barcodes)` (defaults) | `pileup(..., stepper="nofilter")` per site with the same filters and UMI collapsing in Python | 1,000 SNPs x 229 cells | identical: 15,242 (site, cell) pairs, A / C / G / T / N UMI counts equal in every pair (45,612 UMIs); 67 tied UMIs dropped in both |
| `pileup_cells` (defaults) | same | 28,454 SNPs x 229 cells (whole file) | identical: 419,418 (site, cell) pairs at 24,453 sites, every count equal (1,105,869 UMIs); 2,793 tied UMIs dropped in both |
| `discover_snps(regions=..., min_depth=1, min_alt=0, min_maf=0, min_len=0)` (pooled base counts) | `count_coverage(quality_threshold=20, read_callback=flag 0x704 and MAPQ >= 20)` | chr12:6,530,000-6,540,000 | identical: 4,347 covered positions, 1,243,525 bases, every A / C / G / T count equal |

What the pileup does, in both implementations: reads with flag 0x304 (unmapped, secondary, QC fail) are skipped;
MAPQ >= 20; aligned length (CIGAR M / = / X) >= 30; the site must lie in an aligned block (a site inside a
deletion or an intron `N` is not covered; insertions and soft clips shift the read position); base quality >= 20;
`CB` in the barcode list and `UB` present; then one base per (site, cell, UMI) by majority vote of its reads, a UMI
whose reads tie between bases dropped. Duplicate-flagged reads are kept (they are the other reads of a UMI and vote).
pysam's pileup applies none of its own filters here (stepper "nofilter", no depth cap, overlapping mates and
orphans kept), so the comparison tests the CIGAR walk, the tag lookup and the collapsing, not two filter sets.

Not compared: CRAM, CSI indexes (chromosomes > 512 Mb), BGZF CRC checks (arsc does not verify the CRC32 of each block), reads with more than 65,535 CIGAR operations (`CG` tag), and the `min_top2` option of `discover_snps`
(unit-tested only).

## Speed

Same machine (16 cores, hard disk), same thread count for both tools (2: the share the machine's CPU budget gave the
run; arsc ran with `AR_THREADS=2`, pysam with `threads=2` and `docker --cpus 2`). The BAM is on a local hard disk;
pysam reads it through Docker Desktop's bind mount (read-only).

| Task | arsc | pysam (Docker) | Speedup |
|---|---|---|---|
| Count every record (whole 9.15 GB BAM), first read (disk) | 48.0 s | - | - |
| Count every record, file in the OS cache | 39.8 s | 110.7 s | 2.8x |
| Per-cell UMI counts at 1,000 SNPs (index queries) | 13.5 s | 82.0 s | 6.1x |
| Per-cell UMI counts at 28,454 SNPs | 44.5 s | 2,438 s (41 min) | 55x |
| De-novo SNP candidates, whole file (`discover_snps` defaults: 25,225 sites) | 72.5 s | - (no pysam equivalent; `count_coverage` per region) | - |

Each tool ran on the file already in the OS cache except the first arsc count. In the pysam pileup most of the time is the
Python loop over the reads at each site (filters, tags, UMI collapsing), which any pysam-based genotype script has;
the share of the Docker bind mount was not measured separately.
At 1,000 SNPs arsc reads the index chunks of the sites (1.0-1.7 GB of the 9.15 GB compressed, merged chunks), at
28,454 SNPs effectively the whole file. At 2 threads arsc's whole-file read is bound by inflating (~95 GB of uncompressed records; libdeflate ~1.2 GB/s per
thread); with more threads it becomes bound by the disk (about 190 MB/s here, 48 s for this file).

## Method notes

* BGZF: block headers are scanned in numba; blocks are inflated by a thread pool into one reused buffer (fresh
  buffers per block cost page faults that stop the threads from scaling: 4 GB/s vs 12 GB/s at 8 threads). A
  background thread reads the file sequentially in 32 MB chunks. Inflaters: libdeflate (`deflate` wheel, the `bam`
  extra), else isal, else the standard zlib; all three give identical records (tested).
* Region queries use the `.bai` bins and linear index as htslib does (`reg2bins`, chunks ending before the linear
  offset dropped, chunks merged). For many sites the chunks of all sites are merged, and chunks less than 256 kB apart
  (compressed) are read in one go, so each record is read once and dense site lists read the file front to back.
* Cell barcodes are matched exactly (64-bit hash, then a byte comparison). UMIs of up to 28 ACGT bases are packed
  exactly into 64 bits; longer or non-ACGT UMIs use a 64-bit hash (a collision would have to happen inside one cell
  at one site). In pooled mode (`barcodes=None`) reads are keyed by a 64-bit hash of CB plus the UMI.
* `discover_snps` counts reads (duplicates excluded by flag, which in Cell Ranger BAMs leaves one read per UMI) in a
  ring buffer of 8.4 Mb positions; positions behind the current read start are final. Bases of a read further than
  that beyond its start are dropped and counted (`df.attrs["dropped_bases"]`; 0 on the synthetic data).
