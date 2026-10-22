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


def test_pynndescent_not_all_cores():
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
