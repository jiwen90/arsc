"""Test configuration.

Tests that compare against R reference outputs or real datasets read their locations from environment
variables and skip when they are unset or missing:

    ARSC_REFS                stored R reference outputs, one folder per set (ref_gd, ref_extra, stats_parity,
                             stats_resampling, clones_ref, clonotrace_ref, symphony_ref, ref_harmony); rebuilt by
                             benchmarks/regen_refs.py. Old name: ARSC_SCRATCH. Never under a scratch folder
                             (tests/test_repo_hygiene.py): autoresearch prunes scratch/ after 2 days.
    ARSC_STORE_GD            the ONE shared input: the 38,440-cell gamma-delta T-cell arsc store, read memory-mapped
    ARSC_GD_RDS              the Seurat object that store was converted from (R references only)
    ARSC_TEST_CLONES_META    cell metadata (csv.gz) with a published Cyclone assignment (tests/test_clones.py)
    ARSC_DATA_V3             in-house gene-set folder (tests/test_clones.py::test_v3_genesets)
    ARSC_TEST_SCANPY_STORE   arsc store used for the scanpy-parity tests (tests/test_scanpy_extra.py)
    ARSC_TEST_CLONOTRACE_REF CloneTrace R reference outputs (default: $ARSC_REFS/clonotrace_ref)
    ARSC_TEST_EXTRA_PKGS     optional extra site-packages dir (scikit-misc, palantir) for those tests
    ARSC_REF_HARMONY         R harmony references (default: $ARSC_REFS/ref_harmony)

A test whose reference is missing skips with a message that names the missing file and the regen_refs.py command.
A local, untracked ``tests/local.env`` (KEY=VALUE lines) is read first, so a workstation can keep its paths
out of the repository. Existing environment variables win.

The tests import THIS checkout's ``src``: it goes first on ``sys.path`` (and on ``PYTHONPATH`` for child processes)
before anything imports arsc, and the run stops if ``arsc`` would still load from elsewhere. An editable install
points at one checkout (the main one); without this, tests run in a git worktree silently exercised main's code.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def pin_source(src=SRC, pkg="arsc"):
    """Put `src` first on sys.path and PYTHONPATH; raise if `pkg` resolves (or was already imported) elsewhere."""
    s = str(src)
    sys.path[:] = [s] + [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != os.path.normcase(s)]
    os.environ["PYTHONPATH"] = os.pathsep.join([s] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                                                       if p and os.path.normcase(p) != os.path.normcase(s)])
    mod = sys.modules.get(pkg)
    origin = getattr(mod, "__file__", None) if mod is not None else getattr(importlib.util.find_spec(pkg), "origin", None)
    if origin is None or not Path(origin).resolve().is_relative_to(Path(src).resolve()):
        raise RuntimeError(f"{pkg} {'was imported' if mod is not None else 'resolves'} from {origin}, not from this "
                           f"checkout's {src}: the tests would exercise another checkout's code. Run pytest from "
                           f"this checkout (PYTHONPATH={src}) and import {pkg} only after tests/conftest.py.")
    return origin


pin_source()

_env = Path(__file__).with_name("local.env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def pytest_configure(config):
    """Refuse a whole-directory run (`pytest tests`, bare `pytest`) unless ARSC_FULL=1: on a workstation, run the
    test files a change can break (`python tests/testsel.py`); the full suite runs on GitHub CI (ARSC_FULL=1 there)."""
    if hasattr(config, "workerinput") or os.environ.get("ARSC_FULL") == "1":
        return
    import pytest
    here = Path(str(config.invocation_params.dir))
    dirs = [a for a in config.args if (here / str(a).split("::")[0]).is_dir()]
    if dirs:
        raise pytest.UsageError(
            f"refusing to run a whole directory ({', '.join(map(str, dirs))}). Name test files: "
            "`python tests/testsel.py` selects and runs the tests your change can break (`--list` to only print), "
            "or `python -m pytest -q tests/test_x.py[::test_name]`. The full suite runs on GitHub CI on push; "
            "locally only when the operator asks: ARSC_FULL=1 python -m pytest -q tests")
