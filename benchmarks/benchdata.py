"""Benchmark input data, held in memory only: nothing here writes a file.

Two sources:

* ``toy``: the synthetic counts of ``suite_sim.py`` (negative-binomial, 10 cell types, 8 patients x pre/post,
  planted markers and DE), generated in memory from a fixed seed. Identical cell by cell for a given (n, seed).
* ``store:<dir>``: a real arsc store, opened memory-mapped and read-only (``arsc.open``). With a size ``n``
  smaller than the store, a deterministic subset (seeded, sorted) is taken; nothing is copied to disk.

``resolve_source("auto")`` picks ``store:$ARSC_BENCH_STORE`` when that variable names an existing store, else
``toy``. Columns of a real store are mapped with ``ARSC_BENCH_STORE_COLUMNS`` (JSON), e.g.
``{"patient": "patient_alias", "timepoint": "timepoint", "cell_type": "cell_type", "pre": "Baseline",
"post": "Follow Up 1"}``; unmapped names default to themselves and pre/post to the two most frequent timepoints.

The inputs a step does not own (2,000 HVGs, the 30-PC embedding, the pseudobulk counts of one cell type) are
computed here with arsc, in memory, before the timed region (``Bench.hvg`` / ``Bench.pca`` / ``Bench.pb``). A
parent process can compute the PCA once and hand it to its children through shared memory (RAM, not a file):
``share_pca`` and ``Bench(..., pca_shm=)``.
"""
from __future__ import annotations

import json
import os
import sys
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

N_HVG, N_PC = 2000, 30
DEFAULT_TOY_N = 20_000
WARM_TOY_N = 12_000


def resolve_source(source: str = "auto") -> str:
    if source != "auto":
        return source
    st = os.environ.get("ARSC_BENCH_STORE", "")
    if st and Path(st).is_dir():
        return f"store:{st}"
    return "toy"


class Bench:
    """One benchmark dataset in memory: counts ``X`` (cells x genes CSR, float32), ``obs`` (patient, timepoint,
    sample, cell_type, clone_id), ``genes``, ``truth`` (gene signatures; planted truth for toy data)."""

    def __init__(self, source: str, n: int | None = None, seed: int = 0, pca_shm: str = ""):
        self.source = resolve_source(source)
        self.seed = seed
        self.pca_shm = pca_shm   # "<name>:<rows>:<cols>": the parent's embedding in shared memory
        self._n = n
        if self.source != "toy" and not self.source.startswith("store:"):
            raise ValueError(f"unknown benchmark source {source!r}; use 'auto', 'toy' or 'store:<dir>'")

    def __getattr__(self, name):
        """X, obs, genes, truth and n are made on first use (a step that needs only the shared PCA never
        generates or reads the counts)."""
        if name not in ("X", "obs", "genes", "truth", "n"):
            raise AttributeError(name)
        if self.source == "toy":
            from suite_sim import simulate_memory
            self.X, self.obs, self.genes, self.truth = simulate_memory(self._n or DEFAULT_TOY_N, self.seed)
        else:
            self._from_store(self.source[len("store:"):], self._n, self.seed)
        self.n = self.X.shape[0]
        return getattr(self, name)

    def _from_store(self, path, n, seed):
        import arsc
        st = arsc.open(path)
        cols = json.loads(os.environ.get("ARSC_BENCH_STORE_COLUMNS", "{}") or "{}")
        obs = st.obs
        rows = None
        if n and n < len(obs):
            rows = np.sort(np.random.default_rng(seed).choice(len(obs), n, replace=False))
        X = st.X("RNA", "counts")
        if rows is not None:
            X = X[rows]
            obs = obs.iloc[rows]
        X = X.tocsr().astype(np.float32)
        o = pd.DataFrame(index=obs.index.astype(str))
        for key in ("patient", "timepoint", "cell_type"):
            c = cols.get(key, key)
            if c not in obs.columns:
                raise KeyError(f"store column for {key!r} not found ({c!r}); set ARSC_BENCH_STORE_COLUMNS")
            o[key] = obs[c].astype(str).to_numpy()
        tps = o["timepoint"].value_counts()
        pre = cols.get("pre", "pre" if "pre" in tps.index else tps.index[0])
        post = cols.get("post", "post" if "post" in tps.index else [t for t in tps.index if t != pre][0])
        o["timepoint"] = o["timepoint"].map({pre: "pre", post: "post"}).fillna("other")
        o["sample"] = o["patient"] + "_" + o["timepoint"]
        cc = cols.get("clone_id", "clone_id")
        o["clone_id"] = obs[cc].to_numpy() if cc in obs.columns else None
        o["cell_type"] = o["cell_type"].astype("category")
        self.X, self.obs, self.genes = X, o, st.features("RNA")
        # signatures for score_genes: three sets of 30 genes, fixed seed, among genes detected in >= 1% of cells
        det = np.flatnonzero(np.bincount(X.indices, minlength=X.shape[1]) >= 0.01 * X.shape[0])
        rng = np.random.default_rng(seed + 7)
        self.truth = {"signatures": {f"sig_{i}": list(self.genes[rng.choice(det, 30, replace=False)])
                                     for i in range(3)}}

    # ---------------------------------------------------------------- shared inputs (in memory)
    @cached_property
    def hvg(self) -> list:
        import arsc
        h = arsc.pp.hvg_vst(self.X, nfeatures=N_HVG, features=self.genes)
        return list(h.sort_values("rank").index[:N_HVG])

    def hvg_index(self, names) -> np.ndarray:
        return np.sort(pd.Index(names).get_indexer(self.hvg))

    @cached_property
    def pca(self) -> np.ndarray:
        shm = self.pca_shm
        if shm:
            from multiprocessing import shared_memory
            name, r, c = shm.split(":")
            self._shm = attach_shm(name)
            return np.array(np.ndarray((int(r), int(c)), dtype=np.float32, buffer=self._shm.buf))
        import arsc
        L = arsc.pp.log_normalize(self.X)[:, self.hvg_index(self.genes)]
        Z, _, _ = arsc.pp.scale(L)
        return arsc.pp.pca(Z, n_comps=N_PC).embedding.astype(np.float32)

    @cached_property
    def pb(self):
        """Pseudobulk counts (genes x samples) of the most frequent cell type at pre/post, with its coldata."""
        import arsc
        obs = self.obs
        ct = "T0" if "T0" in set(obs["cell_type"].astype(str)) else obs["cell_type"].astype(str).value_counts().index[0]
        m = ((obs["cell_type"].astype(str) == ct) & obs["timepoint"].isin(["pre", "post"])).to_numpy()
        pb = arsc.pseudobulk(self.X[np.flatnonzero(m)], obs[m], "sample", features=self.genes)
        cnt = pd.DataFrame(pb.X.T, index=self.genes, columns=pb.groups.index)
        keep = (cnt.values >= 10).sum(axis=1) >= 3
        cnt = cnt.loc[keep].round().astype(np.int64)
        col = pd.DataFrame({"patient": [s.rsplit("_", 1)[0] for s in cnt.columns],
                            "timepoint": [s.rsplit("_", 1)[1] for s in cnt.columns]}, index=cnt.columns)
        both = col.groupby("patient")["timepoint"].nunique()
        keepc = col["patient"].isin(both[both == 2].index).to_numpy()
        return cnt.loc[:, keepc], col[keepc]

    def anndata(self):
        """An in-memory AnnData of the counts (scanpy input); int32 index arrays when they fit."""
        import anndata as ad
        import scipy.sparse as sp
        X = self.X
        idt = np.int32 if X.nnz < 2 ** 31 else np.int64
        X = sp.csr_matrix((X.data.copy(), X.indices.astype(idt), X.indptr.astype(idt)), shape=X.shape)
        a = ad.AnnData(X=X, obs=self.obs.copy(), var=pd.DataFrame(index=pd.Index(self.genes, dtype=str)))
        return a


def attach_shm(name):
    """Attach the parent's shared-memory block without letting this process's resource tracker unlink it at exit
    (POSIX, Python < 3.13: the first child to exit would otherwise remove the block for every later child)."""
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python >= 3.13
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        if os.name != "nt":
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        return shm


def share_pca(E: np.ndarray):
    """Put the embedding in a shared-memory block (RAM) for child processes; returns (block, env value)."""
    from multiprocessing import shared_memory
    E = np.ascontiguousarray(E, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=max(1, E.nbytes))
    np.ndarray(E.shape, dtype=np.float32, buffer=shm.buf)[:] = E
    return shm, f"{shm.name}:{E.shape[0]}:{E.shape[1]}"
