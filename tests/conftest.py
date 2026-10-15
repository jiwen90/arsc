"""Test configuration.

Tests that compare against R reference outputs or real datasets read their locations from environment
variables and skip when they are unset or missing:

    ARSC_SCRATCH             R reference outputs and exported stores (ref_gd, exp_gd, stats_parity, clones_ref)
    ARSC_TEST_CLONES_META    cell metadata (csv.gz) with a published Cyclone assignment (tests/test_clones.py)
    ARSC_DATA_V3             in-house gene-set folder (tests/test_clones.py::test_v3_genesets)
    ARSC_TEST_SCANPY_STORE   arsc store used for the scanpy-parity tests (tests/test_scanpy_extra.py)
    ARSC_TEST_CLONOTRACE_REF CloneTrace R reference outputs (default: $ARSC_SCRATCH/clonotrace_ref)
    ARSC_TEST_EXTRA_PKGS     optional extra site-packages dir (scikit-misc, palantir) for those tests

A local, untracked ``tests/local.env`` (KEY=VALUE lines) is read first, so a workstation can keep its paths
out of the repository. Existing environment variables win.
"""
import os
from pathlib import Path

_env = Path(__file__).with_name("local.env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
