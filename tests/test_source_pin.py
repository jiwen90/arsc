"""tests/conftest.py makes the tests import THIS checkout's src: a test run in another checkout (a git worktree) sees
that checkout's arsc, never the editable install's (the main checkout). The sentinel test copies the test setup and
src/arsc into a temporary checkout, adds a module attribute there and runs a test in the copy: it must see the
attribute, and a plain `import arsc` (the installed one) must not."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import arsc
from conftest import ROOT, SRC


def test_package_imports_from_this_checkout():
    assert Path(arsc.__file__).resolve().is_relative_to(SRC.resolve()), arsc.__file__


def test_test_in_another_checkout_sees_its_sentinel(tmp_path):
    wt = tmp_path / "wt"
    (wt / "tests").mkdir(parents=True)
    shutil.copy2(ROOT / "tests" / "conftest.py", wt / "tests" / "conftest.py")
    shutil.copy2(ROOT / "pyproject.toml", wt / "pyproject.toml")
    shutil.copytree(SRC / "arsc", wt / "src" / "arsc", ignore=shutil.ignore_patterns("__pycache__"))
    with open(wt / "src" / "arsc" / "__init__.py", "a", encoding="utf-8") as fh:
        fh.write('\n_WT_SENTINEL = "copy"\n')
    (wt / "tests" / "test_sentinel.py").write_text(
        "import arsc\n\n\ndef test_sees_sentinel():\n"
        "    assert getattr(arsc, '_WT_SENTINEL', None) == 'copy', arsc.__file__\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ARSC_FULL")}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_sentinel.py"],
                       cwd=str(wt), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    r = subprocess.run([sys.executable, "-c", "import arsc; print(getattr(arsc, '_WT_SENTINEL', None))"],
                       cwd=str(wt), env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "None", r.stdout + r.stderr  # the installed arsc: no sentinel
    assert getattr(arsc, "_WT_SENTINEL", None) is None  # this checkout's arsc: no sentinel
