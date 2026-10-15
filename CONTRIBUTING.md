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

```bash
python -m pytest -q -m "not slow" tests          # fast suite (what CI runs)
python -m pytest -q tests                        # everything, including slow parity tests
python -m pytest -q tests/test_docstrings.py tests/test_docs_snippets.py   # docs only
```

The R-parity tests compare against reference outputs produced by the R scripts in `benchmarks/`
(`r_reference_*.R`, R 4.3 with Seurat 4.3, harmony 1.2, presto, UCell, DESeq2, limma, edgeR,
fgsea, lme4). Without those outputs the tests **skip**; they never fail for a missing reference.
Set `ARSC_SCRATCH` to the directory that holds the references (the other data variables are listed in
`tests/conftest.py`). Keep local paths in an untracked `tests/local.env` (KEY=VALUE lines), never in the code.

## Rules for code

1. **Validated ports only.** A new method that reproduces another package needs a parity table:
   reference package and version, data (synthetic or public, with size), the metric
   (max abs diff, correlation, identical-call rate), and a test that asserts the threshold. Add it
   to the matching `*_PARITY.md` (or a new one). Say "exact" only when a test shows identity
   (`np.array_equal` or a tolerance at floating-point rounding); give the measured number otherwise.
2. **Licences.** Implement methods from the paper and the mathematics; do not copy or translate
   source code from GPL/LGPL packages. Credit the method's paper in `NOTICE`. Optional
   dependencies with copyleft licences must stay optional.
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
8. **No private data.** Tests and docs use synthetic data generated in code (`arsc.datasets`) or
   public data downloaded by a script. Never commit patient data, internal project names or local
   paths.

## Pull requests

- One topic per pull request; include the parity numbers in the description for numerical changes.
- Add an entry under `[Unreleased]` in `CHANGELOG.md`.
- CI (Linux and Windows, Python 3.10-3.12) must pass.

## Reporting problems

Open an issue with a minimal example, ideally on `arsc.datasets.toy()`, plus `arsc.__version__`,
your OS and Python version.
