"""tests/testsel.py: a change selects the test files that use it, by exact path, and not unrelated ones."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testsel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def repo():
    return testsel.Repo()


def files(targets):
    return {t.split("::")[0] for t in targets}


def test_module_selects_its_tests_only(repo):
    targets, why, unc = testsel.select(["src/arsc/clones.py"], repo)
    assert "tests/test_clones.py" in targets
    assert "tests/test_docstrings.py::test_doctests[clones]" in targets
    for unrelated in ("tests/test_harmony_versions.py", "tests/test_symphony.py", "tests/test_formats.py",
                      "tests/test_docs_snippets.py", "tests/test_speed_gate.py"):
        assert unrelated not in files(targets), why
    assert not unc


def test_reexported_names_resolve_to_their_module(repo):
    assert repo.resolve("arsc.pp.pca") == "arsc.preprocessing"
    assert repo.resolve("arsc.open") == "arsc.io"
    assert repo.resolve("arsc.stats.mde") == "arsc.stats"
    targets, _, _ = testsel.select(["src/arsc/views.py"], repo)
    assert "tests/test_views.py" in targets and "tests/test_clones.py" not in targets


def test_shared_core_module_runs_its_direct_tests_not_the_suite(repo):
    targets, _, _ = testsel.select(["src/arsc/io.py"], repo)
    assert "tests/test_views.py" in targets and "tests/test_formats.py" in targets
    assert len(files(targets)) < len(repo.tests) / 2
    # a module no test uses directly: a sample of at most FALLBACK files through its importers
    targets, why, _ = testsel.select(["src/arsc/_utils.py"], repo)
    assert 0 < len(files(targets)) <= testsel.FALLBACK
    assert all("via" in why[t] for t in targets)


def test_docs_only_change_runs_that_docs_node_only(repo):
    targets, _, _ = testsel.select(["docs/getting-started.md"], repo)
    assert targets == ["tests/test_docs_snippets.py::test_doc_snippets_run[docs/getting-started.md]"]
    targets, _, _ = testsel.select(["CHANGELOG.md", "HARMONY_PARITY.md"], repo)
    assert targets == []


def test_test_file_and_helper(repo):
    assert testsel.select(["tests/test_seeds.py"], repo)[0] == ["tests/test_seeds.py"]
    assert "tests/test_testsel.py" in testsel.select(["tests/testsel.py"], repo)[0]


def test_commands_name_files_never_the_directory(repo):
    targets, _, _ = testsel.select(["src/arsc/clones.py", "src/arsc/_recipes.py"], repo)
    for cwd, ts in testsel.commands(targets):
        assert ts and all(t.startswith("tests/test_") and t.split("::")[0].endswith(".py") for t in ts)


def test_cli_list_runs_nothing():
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "testsel.py"), "--list", "src/arsc/views.py"],
                       capture_output=True, text=True, cwd=str(ROOT), env=dict(os.environ))
    assert r.returncode == 0 and "python -m pytest -q tests/test_views.py" in r.stdout
