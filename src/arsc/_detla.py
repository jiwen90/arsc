"""Thread-count-invariant dense linear algebra (SEEDS.md, "Parallelism and floating-point order").

A multithreaded BLAS splits a long inner (reduction) dimension between its threads, so ``Z.T @ Z`` over many cells sums
in an order that depends on the thread count (measured: PCA differs by up to 3e-11 between 2 and 8 threads, harmony by
5e-14). Here every product is cut into blocks of a FIXED size (never derived from the thread count). Each block runs
on single-threaded BLAS, blocks run in parallel on a pool of ``arsc.get_threads()`` Python threads (numpy releases the
GIL), and the partial products of one output block are summed in block order. The result is bit-identical for any
thread count; the speed is close to multithreaded BLAS for the large products this is used on.

``single_thread()`` pins BLAS/LAPACK (and OpenMP, for sklearn) to one thread for steps that cannot be blocked
(``eigh``, QR, ARPACK's own vector updates, k-means).
"""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BLOCK = 4096  # rows / inner length of one block; fixed, so results do not depend on threads or machine


@contextlib.contextmanager
def single_thread(user_api=None):
    """BLAS (``user_api="blas"``) or every native pool (None: BLAS and OpenMP) limited to one thread."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a dependency of scikit-learn
        yield
        return
    with threadpool_limits(limits=1, user_api=user_api):
        yield


def _threads() -> int:
    from ._threads import get_threads
    return max(1, int(get_threads()))


def matmul(A, B, block: int = BLOCK) -> np.ndarray:
    """``A @ B`` (2-D), bit-identical for any thread count (module docstring).

    Parameters
    ----------
    A, B : 2-D arrays with ``A.shape[1] == B.shape[0]`` (B may also be 1-D).
    block : block size (keep the default; it is part of the result's summation order).

    Returns
    -------
    ndarray, ``A @ B``.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    vec = B.ndim == 1
    if vec:
        B = B[:, None]
    n, k = A.shape
    if B.shape[0] != k:
        raise ValueError(f"matmul: shapes {A.shape} and {B.shape} do not align")
    m = B.shape[1]
    out = np.empty((n, m), dtype=np.result_type(A.dtype, B.dtype))
    rows = [(i, min(i + block, n)) for i in range(0, max(n, 1), block)]
    inner = [(j, min(j + block, k)) for j in range(0, max(k, 1), block)]
    tasks = [(r, c) for r in rows for c in inner]

    def part(t):
        (r0, r1), (c0, c1) = t
        return A[r0:r1, c0:c1] @ B[c0:c1]

    with single_thread("blas"):
        nt = min(_threads(), len(tasks))
        if nt <= 1:
            for t in tasks:
                _add(t, part(t), out)
        else:
            with ThreadPoolExecutor(nt) as ex:
                # futures are consumed strictly in task order, so each output block is summed in inner-block order;
                # a window of 4 x threads bounds the partial products held in memory
                window, pending = 4 * nt, []
                for t in tasks:
                    pending.append((t, ex.submit(part, t)))
                    if len(pending) >= window:
                        t0, f0 = pending.pop(0)
                        _add(t0, f0.result(), out)
                for t0, f0 in pending:
                    _add(t0, f0.result(), out)
    return out[:, 0] if vec else out


def _add(t, p, out):
    (r0, r1), (c0, _c1) = t
    if c0 == 0:
        out[r0:r1] = p
    else:
        out[r0:r1] += p
