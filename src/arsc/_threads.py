"""Thread control. arsc parallelizes with threads (numba ``prange`` + BLAS), never
with process copies. ``AR_THREADS`` (framework convention) caps both."""
from __future__ import annotations

import os

_ENV_VARS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def env_threads() -> int | None:
    v = os.environ.get("AR_THREADS")
    try:
        return max(1, int(v)) if v else None
    except ValueError:
        return None


def preconfigure_env() -> None:
    """Before numba/numpy load: map AR_THREADS onto the libraries' own env vars."""
    n = env_threads()
    if n is not None:
        for var in _ENV_VARS:
            os.environ.setdefault(var, str(n))


_limiter = None


def set_threads(n: int) -> int:
    """Set the number of compute threads (numba and BLAS) for this process. Returns the value used."""
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
    import numba
    return numba.get_num_threads()
