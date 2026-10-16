"""Seed recording (arsc._seedlog, SEEDS.md "Recording the seed") and 1-thread vs N-thread determinism.

In-memory synthetic inputs only; nothing is written to disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import arsc
from arsc import clonotrace as ct
from arsc import stats as st
from arsc._seedlog import records_seed

from test_seeds import TOY_CLONES, TOY_X, X, FEATS, same  # synthetic fixtures (module-level, in memory)

# (module, function, seed parameter) of every function in the SEEDS.md table
TABLE = [
    ("pp", "pca", "seed"), ("arsc", "run_harmony", "seed"), ("arsc", "knn", "seed"),
    ("arsc", "find_neighbors", "seed"), ("arsc", "wnn", "seed"), ("arsc", "cluster", "seed"),
    ("arsc", "umap", "seed"), ("scanpy_extra", "score_genes", "random_state"),
    ("scanpy_extra", "neighbors", "random_state"), ("scanpy_extra", "leiden", "random_state"),
    ("scanpy_extra", "umap_from_connectivities", "random_state"), ("scanpy_extra", "diffmap", "random_state"),
    ("scanpy_extra", "palantir_diffusion_maps", "seed"), ("arsc", "add_module_score", "seed"),
    ("arsc", "cell_cycle_scoring", "seed"), ("arsc", "find_markers", "random_seed"),
    ("arsc", "find_all_markers", "random_seed"), ("stats", "boot_ci", "seed"), ("stats", "paired_test", "seed"),
    ("stats", "perm_test", "seed"), ("stats", "arm_contrast", "seed"), ("de", "fgsea", "seed"),
    ("clones", "clone_expansion_null", "seed"), ("clones", "startrac", "seed"), ("clones", "clone_diversity", "seed"),
    ("clonotrace", "propagate_clones", "seed"), ("clonotrace", "revise_distance", "seed"),
    ("clonotrace", "cluster_clones", "seed"), ("clonotrace", "profile_enrichment", "seed"),
    ("clonotrace", "fate_genes", "seed"), ("clonotrace", "run", "seed"),
    ("symphony", "build_reference", "seed"), ("symphony", "transfer_labels", "seed"),
]


def _obj(mod, fn):
    m = arsc if mod == "arsc" else getattr(arsc, mod)
    return getattr(m, fn)


@pytest.mark.parametrize("mod,fn,param", TABLE)
def test_every_table_function_records_its_seed(mod, fn, param):
    assert getattr(_obj(mod, fn), "__arsc_seed_param__", None) == param


def test_seeds_md_table_matches():
    import pathlib
    text = (pathlib.Path(__file__).resolve().parents[1] / "SEEDS.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("| `")]
    assert len(rows) == len(TABLE)


def test_log_outermost_only_and_counts():
    arsc.reset_seeds_used()
    df = pd.DataFrame({"patient_alias": np.repeat(["P1", "P2", "P3", "P4"], 2),
                       "timepoint": ["pre", "post"] * 4, "value": [1, 2, 1.5, 2.5, 0.8, 1.9, 1.1, 2.2]})
    out = st.paired_test(df, t0="pre", t1="post", seed=5)  # calls boot_ci inside: one entry only
    st.boot_ci([1.0, 2.0, 3.0], seed=3)
    st.boot_ci([1.0, 2.0, 3.0], seed=3)
    st.perm_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])  # default seed is recorded too
    used = arsc.seeds_used(reset=True)
    assert used == [{"function": "stats.paired_test", "param": "seed", "seed": 5, "calls": 1},
                    {"function": "stats.boot_ci", "param": "seed", "seed": 3, "calls": 2},
                    {"function": "stats.perm_test", "param": "seed", "seed": 1, "calls": 1}]
    assert arsc.seeds_used() == []
    assert out.attrs["seed"] == 5 and out.attrs["seed_function"] == "stats.paired_test"


def test_attrs_on_pandas_and_positional_seed():
    arsc.reset_seeds_used()
    sigs = {"m1": FEATS[:10]}
    s = arsc.add_module_score(X, FEATS, sigs, ctrl=5, seed=9)
    assert s.attrs["seed"] == 9
    st.boot_ci([1.0, 2.0, 3.0], np.mean, 100, 11)  # seed passed positionally
    assert {"function": "stats.boot_ci", "param": "seed", "seed": 11, "calls": 1} in arsc.seeds_used(reset=True)


def test_failed_call_not_logged_and_none_still_rejected():
    arsc.reset_seeds_used()
    with pytest.raises(ValueError):
        st.boot_ci([1.0, 2.0], seed=None)
    assert arsc.seeds_used() == []


def test_decorator_rejects_unknown_param():
    with pytest.raises(TypeError):
        records_seed("seed")(lambda x: x)


def test_wrapped_signature_and_doc_kept():
    import inspect
    assert "seed" in inspect.signature(st.boot_ci).parameters
    assert st.boot_ci.__doc__ and st.boot_ci.__name__ == "boot_ci"


# ------------------------------------------------------------------------- 1 thread vs N threads
@pytest.fixture
def threads():
    n0 = arsc.get_threads()
    import numba
    nmax = min(4, numba.config.NUMBA_NUM_THREADS)
    if nmax < 2:
        pytest.skip("needs >= 2 numba threads")

    def run(f):
        arsc.set_threads(1)
        a = f()
        arsc.set_threads(nmax)
        b = f()
        return a, b
    yield run
    arsc.set_threads(n0)


def _blobs(n=600, d=8, seed=0):
    r = np.random.default_rng(seed)
    return np.vstack([c + r.normal(size=(n // 4, d)) for c in r.normal(scale=3, size=(4, d))]).astype(np.float32)


def test_knn_pynndescent_repeatable_at_fixed_threads(threads):
    pytest.importorskip("pynndescent")
    E = _blobs()
    a, b = threads(lambda: (arsc.knn(E, 10, method="pynndescent", seed=0),
                            arsc.knn(E, 10, method="pynndescent", seed=0)))
    assert same(a[0], a[1]) and same(b[0], b[1])  # same seed + same thread count -> identical


@pytest.mark.xfail(reason="pynndescent 0.6 is not thread-invariant: its parallel NN-descent updates depend on the "
                          "numba thread count (measured: 5 of 6000 neighbour entries differ, 1 vs 4 threads). "
                          "Documented in SEEDS.md; use method='exact' when the graph must not depend on threads.",
                   strict=False)
def test_knn_pynndescent_thread_invariant(threads):
    pytest.importorskip("pynndescent")
    E = _blobs()
    a, b = threads(lambda: arsc.knn(E, 10, method="pynndescent", seed=0))
    assert same(a, b)


def test_fgsea_thread_invariant(threads):
    r = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(400)]
    stats = pd.Series(r.normal(size=400), index=genes)
    sets = {f"s{j}": list(r.choice(genes, size=int(r.integers(15, 60)), replace=False)) for j in range(6)}
    a, b = threads(lambda: arsc.fgsea(stats, sets, nperm=500, seed=1))
    assert same(a.reset_index(drop=True), b.reset_index(drop=True))


def test_propagate_clones_thread_invariant(threads):
    T = ct.cell_transition(TOY_X, 10)
    a, b = threads(lambda: ct.propagate_clones(T, TOY_CLONES, n_boot=3, seed=0))
    assert same(a, b)


def test_add_module_score_thread_invariant(threads):
    sigs = {"m1": FEATS[:15], "m2": FEATS[40:60]}
    a, b = threads(lambda: arsc.add_module_score(X, FEATS, sigs, ctrl=5, seed=1))
    assert same(a, b)
