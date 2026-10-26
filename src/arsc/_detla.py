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


PANEL = 256  # columns of one output panel of ``gram``; fixed, like BLOCK


def gram(Z, block: int = BLOCK, panel: int = PANEL) -> np.ndarray:
    """``Z.T @ Z`` (symmetric), bit-identical for any thread count and equal to ``matmul(Z.T, Z)``.

    ``matmul`` cuts only the rows of ``Z.T`` and the inner (cell) dimension, so a Gram matrix of a few thousand
    features over ~10,000 cells is 1 x 3 blocks and runs on 3 threads. Here the output is also cut into fixed
    ``panel``-wide column panels: each upper-triangle panel pair sums its partial products over the same
    ``block``-row inner blocks, in the same order, as ``matmul`` (diagonal panels use the symmetric BLAS product,
    off-diagonal ones the general product, then mirrored), so every entry is summed exactly as before
    (measured equal on 10,000 and 100,000 real cells x 2,000 genes) and the panel pairs run in parallel.

    Parameters
    ----------
    Z : 2-D array (cells x features).
    block : inner block size (keep the default; it is part of the summation order).
    panel : output panel width (does not change any entry's summation order).

    Returns
    -------
    ndarray, ``Z.T @ Z`` (features x features).
    """
    Z = np.asarray(Z)
    n, p = Z.shape
    inner = [(i, min(i + block, n)) for i in range(0, max(n, 1), block)]
    pans = [(j, min(j + panel, p)) for j in range(0, p, panel)]
    pairs = [(a, b) for ai, a in enumerate(pans) for b in pans[ai:]]
    out = np.empty((p, p), dtype=Z.dtype)

    def part(t):
        (a0, a1), (b0, b1) = t
        res = None
        for r0, r1 in inner:  # summed in inner-block order, as matmul's _add
            A = Z[r0:r1, a0:a1]
            P = A.T @ A if a0 == b0 else A.T @ Z[r0:r1, b0:b1]
            if res is None:
                res = P
            else:
                res += P
        return t, res

    def put(t, P):
        (a0, a1), (b0, b1) = t
        out[a0:a1, b0:b1] = P
        if a0 != b0:
            out[b0:b1, a0:a1] = P.T

    with single_thread("blas"):
        nt = min(_threads(), len(pairs))
        if nt <= 1:
            for t in pairs:
                put(*part(t))
        else:
            with ThreadPoolExecutor(nt) as ex:
                for t, P in ex.map(part, pairs):
                    put(t, P)
    return out
