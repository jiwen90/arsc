# arsc

Read `AGENTS.md` first: it is the complete guide to using arsc (mental model, task map, runnable recipes,
statistics rules, gotchas). `llms.txt` indexes the rest of the docs.

When changing arsc itself, also read `CONTRIBUTING.md`. In short:
- Tests run on GitHub CI on push (any branch; `pytest -m "not slow"`, Linux + Windows, py3.10-3.12): push, then
  check the run (`python tools/ci_status.py arsc --wait` in the autoresearch framework) and fix failures before
  relying on the change. Locally run nothing by default; to iterate, run only named test files:
  `python tests/testsel.py` (the files your change can break; `--list` to print) or
  `python -m pytest -q tests/test_x.py[::test_name]`. Never `pytest tests` or a directory (the conftest refuses it
  without `ARSC_FULL=1`; the full suite runs locally only when the maintainer asks).
- In a worktree (`vendor/arsc-wt/<name>`), run scripts with `PYTHONPATH=<worktree>/src` (the editable install
  loads main's code otherwise; `python tools/wtenv.py <worktree>` in the framework prints the line); check with
  `python -c 'import arsc; print(arsc.__file__)'`. The tests and `tests/testsel.py` pin the worktree's `src`
  themselves (CONTRIBUTING.md, "Working in a git worktree").
- After editing `src/arsc/_recipes.py` or any docstring, run `python docs/build_docs.py`;
  `tests/test_docs_snippets.py` runs every Python block in the docs and fails if generated docs are stale.
- Every public function needs a numpy-style docstring with Parameters, Returns and a runnable Example
  (`tests/test_docstrings.py`). Seeds are integers, never `None`. A new port of an R/Python method needs a
  parity table (`*_PARITY.md`) with measured agreement.
