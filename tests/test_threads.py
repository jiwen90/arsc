"""Thread defaults: arsc never sizes its pools by the core count (several analyses share one machine)."""
import os
import subprocess
import sys

_PROBE = ("import os, arsc, numba; print(arsc.get_threads(), os.environ['OMP_NUM_THREADS'], "
          "os.environ['OPENBLAS_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])")


def _probe(**env):
    e = {k: v for k, v in os.environ.items() if not k.endswith("_THREADS") and k != "AR_THREADS"}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=e, check=True)
    return out.stdout.split()


def test_default_is_small_not_cpu_count():
    assert _probe() == ["4", "4", "4", "4"]


def test_ar_threads_sets_every_pool():
    assert _probe(AR_THREADS="3") == ["3", "3", "3", "3"]


def test_explicit_library_var_is_kept():
    got = _probe(AR_THREADS="2", OMP_NUM_THREADS="1")
    assert got[0] == "2" and got[1] == "1"


def test_knn_not_all_cores():
    import inspect
    from arsc import neighbors
    assert "n_jobs=-1" not in inspect.getsource(neighbors)


def test_numba_imported_before_arsc_still_compiles():
    """numba imported first (scanpy first, or a pytest plugin) fixed its pool size; arsc must not then set
    NUMBA_NUM_THREADS, or numba's next compile raises "Cannot set NUMBA_NUM_THREADS ... threads launched"."""
    code = ("import numba, arsc\n@numba.njit\ndef f(x):\n    return x + 1\n"
            "print(f(1), arsc.get_threads())")
    e = {k: v for k, v in os.environ.items() if not k.endswith("_THREADS") and k != "AR_THREADS"}
    e["AR_THREADS"] = "2"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["2", "2"]


_HEAVY = r'''
import contextlib, os, sys, tempfile
prov = os.path.join(tempfile.mkdtemp(), "prov.py")
open(prov, "w").write(
    "import contextlib\ncalls = []\n"
    "@contextlib.contextmanager\ndef _cm(op, t, m, s):\n    calls.append((op, t, m, s))\n    yield 3\n"
    "def provider(op, t, m=None, s=None):\n    return _cm(op, t, m, s)\n")
os.environ["ARSC_RESOURCE_PROVIDER"] = prov + ":provider"
import numpy as np, arsc
from arsc import _threads as T
seen = []
real = T.set_threads
T.set_threads = lambda n: (seen.append(n), real(n))[1]
Z = np.random.default_rng(0).standard_normal((2000, 50))
T.LIGHT_CELLS = 100          # make this small input count as heavy
arsc.pp.pca(Z, n_comps=5)
calls = T._provider().__globals__["calls"]
T.LIGHT_CELLS = 10**9        # and now as light: no call
arsc.pp.pca(Z, n_comps=5)
print(len(calls), calls[0][0], calls[0][1], calls[0][2], seen[0], seen[-1], arsc.get_threads())
'''


def test_heavy_op_asks_the_provider_and_restores_threads():
    """A heavy function (pca) asks the host's provider (ARSC_RESOURCE_PROVIDER) for its share while it runs, runs on
    the threads granted (3), restores its own count (2) after; a light input never asks."""
    e = {k: v for k, v in os.environ.items() if not k.endswith("_THREADS") and k != "AR_THREADS"}
    e.update(AR_THREADS="2", NUMBA_NUM_THREADS="4")
    out = subprocess.run([sys.executable, "-c", _HEAVY], capture_output=True, text=True, env=e)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["1", "pca", "16", str(1 * 2000 * 50), "3", "2", "2"], out.stdout


def test_heavy_op_without_provider_changes_nothing():
    code = "import numpy as np, arsc; arsc.pp.pca(np.ones((50, 10)) + np.arange(10), n_comps=2); print(arsc.get_threads())"
    e = {k: v for k, v in os.environ.items() if not k.endswith("_THREADS") and k not in ("AR_THREADS",
                                                                                         "ARSC_RESOURCE_PROVIDER")}
    e["AR_THREADS"] = "2"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["2"]
