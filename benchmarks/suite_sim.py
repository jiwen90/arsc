"""Synthetic single-cell benchmark data, generated in code (no real data).

One dataset per size ``n``, deterministic for a given seed:

* 20,000 genes (``G00001``...), negative-binomial counts (gamma-Poisson, dispersion 0.3) with a
  log-normal gene abundance profile; library size ~ log-normal around 2,500 UMI, so about 1,500-1,700
  genes are detected per cell (similar to 10x 3' PBMC data).
* 10 cell types (``cell_type``: ``T0``..``T9``), each up-regulating 60 marker genes about 6x.
* 8 patients x 2 timepoints (``pre``/``post``) = 16 samples (``sample``). Patient effects are a small
  per-gene log-normal shift; in cell type ``T0`` 150 genes go up 2x at ``post`` (the DE truth).
* ``clone_id`` for cell types ``T0``-``T3`` (T-cell-like): a per-patient clone pool, with 5 clones per
  patient expanding at ``post`` in ``T0``.

``simulate_memory(n, seed)`` returns the counts, cell table, genes and truth in memory (what paired_bench.py and
the speed gate use; nothing is written). ``simulate(n, root)`` writes the same data to disk, only for the full
three-tool suite (``run_suite.py --bench-dir``), whose Seurat side needs files; it never runs by default.

``simulate`` writes, under ``<root>/n<size>/``:

* ``store/``          arsc store (RNA/counts, float32 CSR)
* ``data.h5ad``       the same counts for scanpy (CSR X)
* ``r/``              raw little-endian binaries of the CSR (= dgCMatrix genes x cells) for R:
                      ``x.f32``, ``i.i32``, ``p.f64``, ``genes.txt``, ``cells.txt``, ``obs.csv``
* ``truth.json``      planted marker / DE genes and the gene signatures used for scoring
* ``common/``         shared inputs so each step is compared on identical input (see run_suite.py)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

N_GENES = 20_000
N_TYPES = 10
N_PATIENTS = 8
N_MARK = 60
N_DE = 150
DISP = 0.3
CHUNK = 50_000


@nb.njit(parallel=True)   # no cache=True: numba's cache would write next to this file
def _nnz_pass(mu_type, tix, pix, size, pat, post_up, is_post, seed0, disp, out_nnz):
    n = tix.shape[0]
    G = mu_type.shape[1]
    shape = 1.0 / disp
    for c in nb.prange(n):
        np.random.seed(seed0 + c)
        k = 0
        t = tix[c]
        for g in range(G):
            mu = mu_type[t, g] * pat[pix[c], g] * size[c]
            if is_post[c] and t == 0:
                mu *= post_up[g]
            lam = np.random.gamma(shape, mu * disp)
            if np.random.poisson(lam) > 0:
                k += 1
        out_nnz[c] = k


@nb.njit(parallel=True)   # no cache=True: numba's cache would write next to this file
def _fill_pass(mu_type, tix, pix, size, pat, post_up, is_post, seed0, disp, indptr, indices, data):
    n = tix.shape[0]
    G = mu_type.shape[1]
    shape = 1.0 / disp
    for c in nb.prange(n):
        np.random.seed(seed0 + c)
        p = indptr[c]
        t = tix[c]
        for g in range(G):
            mu = mu_type[t, g] * pat[pix[c], g] * size[c]
            if is_post[c] and t == 0:
                mu *= post_up[g]
            lam = np.random.gamma(shape, mu * disp)
            v = np.random.poisson(lam)
            if v > 0:
                indices[p] = g
                data[p] = v
                p += 1


def _design(seed: int):
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:05d}" for i in range(1, N_GENES + 1)])
    prof = rng.lognormal(0.0, 1.9, N_GENES)
    prof /= prof.sum()
    order = rng.permutation(N_GENES)
    # markers: draw from moderately expressed genes so they are detectable
    mid = np.argsort(prof)[N_GENES // 4: N_GENES - 200]
    pick = rng.choice(mid, N_TYPES * N_MARK + N_DE, replace=False)
    markers = {f"T{k}": pick[k * N_MARK:(k + 1) * N_MARK] for k in range(N_TYPES)}
    de_genes = pick[N_TYPES * N_MARK:]
    lfc = np.zeros((N_TYPES, N_GENES))
    for k in range(N_TYPES):
        lfc[k, markers[f"T{k}"]] = np.log(6.0)
    mu_type = prof[None, :] * np.exp(lfc)
    mu_type /= mu_type.sum(axis=1, keepdims=True)
    pat = np.exp(rng.normal(0.0, 0.12, (N_PATIENTS, N_GENES)))
    post_up = np.ones(N_GENES)
    post_up[de_genes] = 2.0
    sigs = {f"sig_T{k}": list(genes[markers[f"T{k}"][:30]]) for k in range(5)}
    truth = {"markers": {k: list(genes[v]) for k, v in markers.items()}, "de_up_T0_post": list(genes[de_genes]),
             "signatures": sigs, "n_genes": N_GENES, "dispersion": DISP, "seed": seed}
    del order
    return genes, mu_type, pat, post_up, truth


def _cells(n: int, seed: int):
    rng = np.random.default_rng(seed + 1)
    tp = rng.dirichlet(np.full(N_TYPES, 3.0))
    tix = rng.choice(N_TYPES, n, p=tp).astype(np.int64)
    pix = rng.integers(0, N_PATIENTS, n).astype(np.int64)
    is_post = rng.random(n) < 0.5
    size = rng.lognormal(np.log(2500.0), 0.4, n)
    patients = np.array([f"P{i + 1}" for i in range(N_PATIENTS)])
    tps = np.where(is_post, "post", "pre")
    obs = pd.DataFrame({"cell_type": pd.Categorical([f"T{k}" for k in tix], categories=[f"T{k}" for k in range(N_TYPES)]),
                        "patient": patients[pix], "timepoint": tps})
    obs["sample"] = obs["patient"] + "_" + obs["timepoint"]
    # clones: T0-T3; per patient pool of 2,000 recurring clones (Zipf-like) + unique; 5 expanded in T0 at post
    clone = np.empty(n, dtype=object)
    clone[:] = None
    tcell = tix <= 3
    u = rng.random(n)
    zipf = np.minimum(rng.zipf(1.6, n), 2000)
    for c in np.where(tcell)[0]:
        p = patients[pix[c]]
        if tix[c] == 0 and is_post[c] and u[c] < 0.25:
            clone[c] = f"{p}_E{int(u[c] * 20)}"          # 5 expanded clones
        elif u[c] < 0.6:
            clone[c] = f"{p}_C{zipf[c]}"
        else:
            clone[c] = f"{p}_U{c}"
    obs["clone_id"] = clone
    obs.index = [f"c{i:07d}" for i in range(n)]
    return obs, tix, pix, is_post, size


def _chunks(n: int, seed: int, mu_type, tix, pix, size, pat, post_up, is_post):
    """CSR chunks of CHUNK cells; cell c always draws from seed * 1_000_003 + c, so the chunking never changes it."""
    import scipy.sparse as sp
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        sl = slice(s, e)
        nz = np.zeros(e - s, np.int64)
        args = (mu_type, tix[sl], pix[sl], size[sl], pat, post_up, is_post[sl], seed * 1_000_003 + s, DISP)
        _nnz_pass(*args, nz)
        indptr = np.zeros(e - s + 1, np.int64)
        np.cumsum(nz, out=indptr[1:])
        ind = np.empty(indptr[-1], np.int32)
        dat = np.empty(indptr[-1], np.float32)
        _fill_pass(*args, indptr, ind, dat)
        yield s, e, indptr, ind, dat


def simulate_memory(n: int, seed: int = 0):
    """The benchmark data of size n in memory: (X, obs, genes, truth). X is a cells x genes CSR (float32 counts),
    identical value for value to what ``simulate`` writes for the same (n, seed). Writes nothing."""
    import scipy.sparse as sp
    genes, mu_type, pat, post_up, truth = _design(seed)
    obs, tix, pix, is_post, size = _cells(n, seed)
    ptr, inds, dats, nnz = [np.zeros(1, np.int64)], [], [], 0
    for s, e, indptr, ind, dat in _chunks(n, seed, mu_type, tix, pix, size, pat, post_up, is_post):
        ptr.append(indptr[1:] + nnz)
        nnz += int(indptr[-1])
        inds.append(ind)
        dats.append(dat)
    X = sp.csr_matrix((np.concatenate(dats) if dats else np.zeros(0, np.float32),
                       np.concatenate(inds) if inds else np.zeros(0, np.int32), np.concatenate(ptr)),
                      shape=(n, N_GENES))
    return X, obs, pd.Index(genes), truth


def simulate(n: int, root: Path, seed: int = 0, formats=("store", "h5ad", "r"), verbose: bool = True) -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import arsc
    from arsc.formats import StoreBuilder

    out = Path(root) / f"n{n}"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "sim_done.json"
    if done.exists():
        return out
    t0 = time.time()
    genes, mu_type, pat, post_up, truth = _design(seed)
    obs, tix, pix, is_post, size = _cells(n, seed)
    (out / "truth.json").write_text(json.dumps(truth))
    rdir = out / "r"
    rdir.mkdir(exist_ok=True)
    fx, fi = open(rdir / "x.f32", "wb"), open(rdir / "i.i32", "wb")
    ptr = [np.zeros(1, np.int64)]
    nnz_tot = 0
    b = StoreBuilder(out / "store", source="benchmarks/suite_sim.py", overwrite=True)
    b.set_cells(list(obs.index), obs)
    b.add_assay("RNA", list(genes))
    with b.layer("RNA", "counts", src_dtype="float32") as sink:
        for s, e, indptr, ind, dat in _chunks(n, seed, mu_type, tix, pix, size, pat, post_up, is_post):
            m = sp.csr_matrix((dat, ind, indptr), shape=(e - s, N_GENES))
            sink.add(m)
            dat.tofile(fx)
            ind.tofile(fi)
            ptr.append(indptr[1:] + nnz_tot)
            nnz_tot += int(indptr[-1])
            if verbose:
                print(f"[sim n={n}] {e:,} cells, nnz {nnz_tot:,} ({time.time() - t0:.0f} s)", flush=True)
    fx.close()
    fi.close()
    np.concatenate(ptr).astype(np.float64).tofile(rdir / "p.f64")
    (rdir / "genes.txt").write_text("\n".join(genes))
    (rdir / "cells.txt").write_text("\n".join(obs.index))
    obs.to_csv(rdir / "obs.csv")
    st = b.finish()
    if "h5ad" in formats:
        st.write_h5ad(out / "data.h5ad", overwrite=True)
    info = {"n_cells": n, "n_genes": N_GENES, "nnz": nnz_tot, "genes_per_cell_mean": nnz_tot / n,
            "seconds": time.time() - t0, "seed": seed}
    done.write_text(json.dumps(info, indent=1))
    if verbose:
        print(f"[sim n={n}] done: {info}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--root", required=True, help="directory to WRITE the data to (opt-in; paired_bench.py and the "
                                                   "speed gate never need it: they generate the data in memory)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    simulate(a.n, Path(a.root), a.seed)
