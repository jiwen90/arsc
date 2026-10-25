"""tests/conftest.py pins this checkout's src: a test run from another checkout (a git worktree) imports that
checkout's arsc, never the editable install's (which points at the main checkout).

The sentinel test copies the test setup and src/arsc into a temporary checkout, adds a module attribute there, and
runs a test through tests/testsel.py from that copy: the test must see the attribute. A plain `import arsc` without
the pin (the installed arsc) must not."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import arsc
from conftest import ROOT, SRC, pin_source

SENTINEL = '\n_WT_SENTINEL = "copy"\n'


def test_package_imports_from_this_checkout():
    assert Path(arsc.__file__).resolve().is_relative_to(SRC.resolve()), arsc.__file__


def test_pin_fails_loudly_when_the_package_comes_from_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "")
    with pytest.raises(RuntimeError, match="another checkout"):
        pin_source(tmp_path, "arsc")  # arsc is already imported from this checkout, not from tmp_path


def _copy_checkout(dst):
    (dst / "tests").mkdir(parents=True)
    for f in ("conftest.py", "testsel.py"):
        shutil.copy2(ROOT / "tests" / f, dst / "tests" / f)
    shutil.copy2(ROOT / "pyproject.toml", dst / "pyproject.toml")
    shutil.copytree(SRC / "arsc", dst / "src" / "arsc", ignore=shutil.ignore_patterns("__pycache__"))
    with open(dst / "src" / "arsc" / "__init__.py", "a", encoding="utf-8") as fh:
        fh.write(SENTINEL)
    (dst / "tests" / "test_sentinel.py").write_text(
        "import arsc\n\n\ndef test_sees_sentinel():\n"
        "    assert getattr(arsc, '_WT_SENTINEL', None) == 'copy', arsc.__file__\n", encoding="utf-8")


def test_selected_test_in_another_checkout_sees_its_sentinel(tmp_path):
    wt = tmp_path / "wt"
    _copy_checkout(wt)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ARSC_FULL")}
    r = subprocess.run([sys.executable, str(wt / "tests" / "testsel.py"), "tests/test_sentinel.py", "--",
                        "-p", "no:cacheprovider"], cwd=str(wt), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    # the same import without the pin (the installed arsc, from the copy's directory) does not see it
    r = subprocess.run([sys.executable, "-c", "import arsc; print(getattr(arsc, '_WT_SENTINEL', None))"],
                       cwd=str(wt), env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "None", r.stdout + r.stderr
    assert getattr(arsc, "_WT_SENTINEL", None) is None  # this checkout's arsc has no sentinel
