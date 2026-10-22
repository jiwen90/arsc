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
