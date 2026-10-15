# arsc

Read `AGENTS.md` first: it is the complete guide to using arsc (mental model, task map, runnable recipes,
statistics rules, gotchas). `llms.txt` indexes the rest of the docs.

When changing arsc itself, also read `CONTRIBUTING.md`. In short:
- Tests: `python -m pytest -q tests -m "not slow"` (the `slow` tests need R reference outputs).
- After editing `src/arsc/_recipes.py` or any docstring, run `python docs/build_docs.py`;
  `tests/test_docs_snippets.py` runs every Python block in the docs and fails if generated docs are stale.
- Every public function needs a numpy-style docstring with Parameters, Returns and a runnable Example
  (`tests/test_docstrings.py`). Seeds are integers, never `None`. A new port of an R/Python method needs a
  parity table (`*_PARITY.md`) with measured agreement.
