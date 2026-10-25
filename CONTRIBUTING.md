# Contributing to arsc

Thank you for helping. arsc's promise is that every method is fast **and** numerically checked
against its reference implementation, so contributions are judged on correctness evidence first.

## Development install

```bash
git clone https://github.com/jiwen90/arsc.git
cd arsc
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,clonotrace]"
```

## Running the tests

The tests run on **GitHub CI on every push** (any branch) and on pull requests: Linux and Windows, Python
3.10-3.12, the docstring and docs-snippet tests and then `pytest -m "not slow"` (`.github/workflows/ci.yml`).
After a change, push and check the run (in the autoresearch framework: `python tools/ci_status.py arsc --wait`);
fix a failure before relying on the change.

Locally, run nothing by default. To iterate on a change, run only the test files it can break, by name:

```bash
python tests/testsel.py --list            # what your change (vs main, plus uncommitted files) can break
python tests/testsel.py                   # run exactly those files / tests
python tests/testsel.py src/arsc/views.py # the tests of named paths
python -m pytest -q tests/test_views.py   # or one file, or one test: tests/test_views.py::test_name
```

`tests/testsel.py` maps each source module to the test files that import or use it (direct links only, plus a small
override table in the file); a docs-only change runs that document's snippet test only; a module no test uses
directly runs a sample of at most three test files through the modules that import it. A local command always
names test files: `tests/conftest.py` refuses a whole directory (`pytest tests`, bare `pytest`) unless
`ARSC_FULL=1`. The full suite runs locally only when the maintainer asks for it:
`ARSC_FULL=1 python -m pytest -q tests` (add `-m "not slow"` for CI's selection).

The R-parity tests compare against reference outputs produced by the R scripts in `benchmarks/`
(`r_reference_*.R`, R 4.3 with Seurat 4.3, harmony 1.2 (and 2.0.5 / 2.1.0 in separate libraries), presto, UCell, DESeq2, limma, edgeR,
fgsea, lme4). Without those outputs the tests **skip** with a message naming the missing file and the command that
rebuilds it (`python benchmarks/regen_refs.py <set>`); they never fail for a missing reference. Set `ARSC_REFS` to the
directory that holds the references and `ARSC_STORE_GD` to the shared input store (the other data variables are
listed in `tests/conftest.py`; `benchmarks/README.md` has the table of sets). Keep local paths in an untracked
`tests/local.env` (KEY=VALUE lines), never in the code, and never under a `scratch` folder. Never commit a dataset
(toy, synthetic or R output): commit the script that regenerates it (`tests/test_repo_hygiene.py`).

## Rules for code

1. **Validated ports only.** A new method that reproduces another package needs a parity table:
   reference package and version, data (synthetic or public, with size), the metric
   (max abs diff, correlation, identical-call rate), and a test that asserts the threshold. Add it
   to the matching `*_PARITY.md` (or a new one). Say "exact" only when a test shows identity
   (`np.array_equal` or a tolerance at floating-point rounding); give the measured number otherwise.
2. **Licences.** arsc is GPL-3.0-or-later. A port may follow code whose licence is compatible with
   GPL-3.0-or-later: GPL-3, GPL-2-or-later, LGPL, MIT, BSD, Apache-2.0. **Not** GPL-2.0-only code
   (incompatible with GPL-3). Credit the original in `NOTICE` (package, version, licence) and the method's
   paper; for an MIT/BSD/Apache original, add its licence text to `LICENSES/`.
3. **Seeds.** Every stochastic function takes an integer seed with a fixed default, rejects
   `None` (use `arsc._utils.check_seed`) and never touches the global NumPy/Python RNG. Add the
   function to `SEEDS.md` and `tests/test_seeds.py`.
4. **Docstrings.** Every public function and class has a one-line summary, `Parameters`,
   `Returns` and an `Examples` section that runs on `arsc.datasets.toy()`.
   `tests/test_docstrings.py` fails if one is missing.
5. **Docs never drift.** Every Python code block in `README.md`, `AGENTS.md` and `docs/` is run by
   `tests/test_docs_snippets.py`. If you change an API, update the snippets in the same change.
6. **Arrays and tables, not plots.** Functions take cells x features matrices (CSR) and
   DataFrames and return arrays or tidy DataFrames. Nothing is modified in place.
7. **Statistics.** For patient-level questions the sample/patient is the unit; functions that
   test conditions should work on per-sample tables (pseudobulk, composition), not cells.
8. **Never slower than scanpy.** An arsc step that has a scanpy equivalent must be at least as fast,
   measured in the same session: `ARSC_SPEED_GATE=1 python -m pytest -q tests/test_speed_gate.py`
   (`benchmarks/paired_bench.py --gate`: both tools interleaved, alternating order, the same threads, median of 3
   runs). It is opt-in (it skips unless `ARSC_SPEED_GATE=1`; CI does not run it): run it when a change to a compute
   kernel claims or may cost speed, and before a release. Prefer a proven, optimized
   library (for example pynndescent, scikit-learn, BLAS) over re-implementing one.
9. **No private data.** Tests and docs use synthetic data generated in code (`arsc.datasets`) or
   public data downloaded by a script. Never commit patient data, internal project names or local
   paths.

## Pull requests

- One topic per pull request; include the parity numbers in the description for numerical changes.
- Add an entry under `[Unreleased]` in `CHANGELOG.md`.
- CI (Linux and Windows, Python 3.10-3.12) must pass; it is the test run of record.

## Reporting problems

Open an issue with a minimal example, ideally on `arsc.datasets.toy()`, plus `arsc.__version__`,
your OS and Python version.
