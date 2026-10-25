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

Tests import THIS checkout's ``src`` (a git worktree's own code, not the editable install's main checkout).

Run only the test files your change can break, named explicitly (``python -m pytest -q tests/test_x.py[::name]``),
or a group of them by area: ``python tests/run.py knn`` (the table is in tests/run.py). Never a whole directory:
a directory run, and ``tests/run.py everything``, need ``ARSC_FULL=1`` (GitHub CI sets it; locally only on request).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # the editable install points at the main checkout: a worktree's tests ran main's code
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")]))  # child processes
import arsc  # noqa: E402
if not Path(arsc.__file__).resolve().is_relative_to(SRC.resolve()):
    raise RuntimeError(f"arsc imports from {arsc.__file__}, not from this checkout's {SRC}")

_env = Path(__file__).with_name("local.env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def pytest_configure(config):
    """Refuse a whole-directory run (`pytest tests`, bare `pytest`, `tests/run.py everything`) unless ARSC_FULL=1: the
    full suite runs on GitHub CI on push (ARSC_FULL=1 there), locally only when the operator asks."""
    if hasattr(config, "workerinput") or os.environ.get("ARSC_FULL") == "1":
        return
    import pytest
    here = Path(str(config.invocation_params.dir))
    dirs = [a for a in config.args if (here / str(a).split("::")[0]).is_dir()]
    if dirs:
        raise pytest.UsageError(
            f"refusing to run a whole directory ({', '.join(map(str, dirs))}). Name the test files your change can "
            "break (`python -m pytest -q tests/test_x.py[::test_name]`) or a group by area (`python tests/run.py knn`; "
            "`python tests/run.py` lists the groups). The full suite runs on GitHub CI on push; locally only when the "
            "operator asks: ARSC_FULL=1 python -m pytest -q tests")
