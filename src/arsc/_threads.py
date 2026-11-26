"""Thread control. arsc parallelizes with threads (numba ``prange`` + BLAS), never
with process copies. ``AR_THREADS`` caps both. Without it arsc uses a small default
(``DEFAULT_THREADS`` = 4), never every core: several analyses often run side by side on one
machine, and each one sizing its pools by the core count oversubscribes the CPU many times over.
An explicit ``NUMBA_NUM_THREADS`` / ``OMP_NUM_THREADS`` set by the caller is respected."""
from __future__ import annotations

import os

_ENV_VARS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
DEFAULT_THREADS = 4  # when AR_THREADS is unset: small, so concurrent processes cannot swamp the machine


def env_threads() -> int | None:
    v = os.environ.get("AR_THREADS")
    try:
        return max(1, int(v)) if v else None
    except ValueError:
        return None


def default_threads() -> int:
    """Threads arsc uses: ``AR_THREADS`` if set, else an explicit ``NUMBA_NUM_THREADS``, else ``DEFAULT_THREADS``."""
    n = env_threads()
    if n is not None:
        return n
    try:
        return max(1, int(os.environ.get("NUMBA_NUM_THREADS", "")))
    except ValueError:
        return DEFAULT_THREADS


def preconfigure_env() -> None:
    """Before numba/numpy load: map AR_THREADS (or the small default) onto the libraries' own env vars.
    Variables the caller already set are kept.

    When numba was imported before arsc (``import scanpy`` first, or a pytest plugin), its pool size is already
    fixed and ``NUMBA_NUM_THREADS`` is left alone: setting it now would make numba's next compile fail with
    "Cannot set NUMBA_NUM_THREADS ... once the threads have been launched". ``set_threads`` then caps the threads
    numba uses with ``numba.set_num_threads``."""
    import sys
    n = default_threads()
    for var in _ENV_VARS:
        if var == "NUMBA_NUM_THREADS" and "numba" in sys.modules:
            continue
        os.environ.setdefault(var, str(n))


_limiter = None


def set_threads(n: int) -> int:
    """Set the number of compute threads (numba and BLAS) for this process.

    The environment variable ``AR_THREADS`` sets the same cap at import time.

    Parameters
    ----------
    n : requested threads (clipped to 1 .. the numba maximum).

    Returns
    -------
    int, the number of threads actually used.

    Examples
    --------
    >>> import arsc
    >>> n0 = arsc.get_threads()
    >>> arsc.set_threads(1)
    1
    >>> _ = arsc.set_threads(n0)
    """
    global _limiter
    import numba
    n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits
        _limiter = threadpool_limits(limits=n)
    except Exception:  # threadpoolctl optional
        pass
    return n


def get_threads() -> int:
    """Number of compute threads currently used by numba kernels.

    Returns
    -------
    int.

    Examples
    --------
    >>> import arsc
    >>> arsc.get_threads() >= 1
    True
    """
    import numba
    return numba.get_num_threads()


# --------------------------------------------------------------------------- heavy steps: a share only while they run
# A host that budgets CPU across processes (e.g. a lab framework) names a provider in ARSC_RESOURCE_PROVIDER as
# "<module or path/to/file.py>:<function>". arsc calls it around each heavy function below with
# (op, threads, mem_bytes, size) and it returns a context manager yielding the threads granted; arsc runs the step on
# them (set_threads) and restores its own count afterwards. Without a provider nothing changes. Inputs under
# LIGHT_CELLS cells never ask (a few seconds at most). Cost per op: (threads wanted, bytes of memory per unit of work,
# work unit); work is the input's stored values ("nnz": nonzeros of a sparse matrix, else cells x columns) or its
# full shape ("dense"). Measured 2026-11-26 on 300,000 cells of a real store (430 M nonzeros; 2,000 variable genes;
# warm runs at 4 / 8 / 16 threads): PCA 11.1 / 6.2 / 3.9 s, neighbors 4.7 / 2.9 / 2.6 s, log-normalize
# 1.7 / 1.0 / 0.6 s and Wilcoxon markers 3.3 / 2.1 / 1.9 s scale; variable genes (~4.5 s), scaling (~2 s) and module
# scores (~4.5 s) do not, so they ask for 4. Peak RSS growth per unit: log-normalize 11.3 B per nonzero, variable
# genes 7.5, module score 5.4, markers 50 (per nonzero of the tested genes), scaling 9.3 B per dense value, PCA 0.2,
# neighbors 45-80 B per embedding value. Not measured (estimates): UMAP, Harmony, FindAllMarkers, pseudobulk.
COST = {
    "log_normalize": (8, 12, "nnz"), "normalize_data": (8, 12, "nnz"), "normalize_total_log1p": (8, 12, "nnz"),
    "hvg_vst": (4, 8, "nnz"), "highly_variable_genes": (4, 8, "nnz"),
    "scale": (4, 10, "dense"), "scale_data": (4, 10, "dense"),
    "pca": (16, 1, "dense"),
    "neighbors": (8, 80, "dense"), "umap": (8, 80, "dense"), "umap_from_connectivities": (8, 24, "nnz"),
    "run_harmony": (8, 32, "dense"),
    "score_genes": (4, 6, "nnz"), "add_module_score": (4, 6, "nnz"),
    "rank_genes_groups": (8, 52, "nnz"), "find_all_markers": (8, 52, "nnz"),
    "pseudobulk": (4, 4, "nnz"),
}
LIGHT_CELLS = 100_000  # fewer cells: every step takes seconds at most, no lease
_provider_cache = {}


def _provider():
    spec = os.environ.get("ARSC_RESOURCE_PROVIDER", "")
    if not spec or ":" not in spec:
        return None
    if spec in _provider_cache:
        return _provider_cache[spec]
    fn = None
    try:
        mod, _, name = spec.rpartition(":")
        if mod.endswith(".py"):
            import importlib.util
            s = importlib.util.spec_from_file_location("_arsc_resource_provider", mod)
            m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
        else:
            import importlib
            m = importlib.import_module(mod)
        fn = getattr(m, name)
    except Exception:  # a broken hook must never break an analysis
        fn = None
    _provider_cache[spec] = fn
    return fn


def work_of(x, unit="nnz"):
    """(work units, n cells, n columns) of an input matrix-like ``x`` for the cost table; (0, 0, 0) if unknown."""
    shape = getattr(x, "shape", None)
    if not shape or len(shape) != 2:
        return 0, 0, 0
    n, g = int(shape[0]), int(shape[1])
    if unit == "nnz" and hasattr(x, "nnz"):
        return int(x.nnz), n, g
    return n * g, n, g


def need_of(op, x):
    """(threads, memory bytes, size label) a heavy ``op`` asks for on input ``x``, or None when it is light."""
    cost = COST.get(op)
    if cost is None:
        return None
    threads, per, unit = cost
    w, n, g = work_of(x, unit)
    if n < LIGHT_CELLS:
        return None
    return threads, int(per * w), f"{n}x{g}"


def heavy_op(op):
    """Decorator: run the function under the host's provider (a share held only while it runs). Re-entrant: a heavy
    function called inside another runs on the outer one's share."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            prov = _provider()
            if prov is None or getattr(_tls, "depth", 0) or not args:
                return fn(*args, **kwargs)
            need = need_of(op, args[0])
            if need is None:
                return fn(*args, **kwargs)
            threads, mem, size = need
            _tls.depth = 1
            try:
                try:
                    cm = prov(op, threads, mem, size)
                    granted = cm.__enter__()
                except Exception:
                    cm, granted = None, None
                before = None
                try:
                    if granted:
                        before = get_threads()
                        set_threads(int(granted))
                    return fn(*args, **kwargs)
                finally:
                    if before is not None:
                        set_threads(before)
                    if cm is not None:
                        try:
                            cm.__exit__(None, None, None)
                        except Exception:
                            pass
            finally:
                _tls.depth = 0
        return wrapper
    return deco


import threading as _threading  # noqa: E402

_tls = _threading.local()
