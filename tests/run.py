#!/usr/bin/env python
"""Run the test files of one or more areas.

    python tests/run.py knn                 # the files of one area
    python tests/run.py de stats -- -x      # several areas; anything after -- goes to pytest unchanged
    python tests/run.py                     # list the areas and their files

Naming single files stays the everyday alternative: python -m pytest -q tests/test_x.py[::test_name].
`everything` is the full suite: tests/conftest.py refuses it unless ARSC_FULL=1 (GitHub CI runs it on every push;
locally only when the operator asks).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# area -> test files (globs relative to the repository root)
GROUPS = {
    "knn": ["tests/test_knn_engine.py", "tests/test_leiden.py", "tests/test_umap_knn.py"],
    "pp": ["tests/test_scanpy_extra.py", "tests/test_seurat_extra.py", "tests/test_seurat_full.py",
           "tests/test_vs_r.py"],
    "de": ["tests/test_stats_de.py"],
    "stats": ["tests/test_stats_de.py", "tests/test_stats_resampling.py"],
    "clones": ["tests/test_clones.py", "tests/test_clonotrace.py"],
    "harmony": ["tests/test_harmony_versions.py", "tests/test_symphony.py"],
    "symphony": ["tests/test_harmony_versions.py", "tests/test_symphony.py"],
    "io": ["tests/test_formats.py", "tests/test_views.py", "tests/test_bam.py"],
    "bam": ["tests/test_bam.py"],
    "docs": ["tests/test_docs_snippets.py", "tests/test_docstrings.py", "tests/test_repo_hygiene.py"],
    "setup": ["tests/test_source_pin.py", "tests/test_seeds.py", "tests/test_seedlog.py", "tests/test_threads.py"],
    "everything": ["tests"],
}


def files(groups):
    out = [f.relative_to(ROOT).as_posix() for g in groups for pat in GROUPS[g] for f in sorted(ROOT.glob(pat))]
    return list(dict.fromkeys(out))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    bad = [g for g in argv if g not in GROUPS]
    if not argv or bad:
        if bad:
            print(f"unknown area(s): {', '.join(bad)}")
        for g in GROUPS:
            print(f"  {g:<11} {' '.join(files([g]))}")
        return 2 if bad else 0
    cmd = [sys.executable, "-m", "pytest", "-q", *files(argv), *extra]
    print("python -m pytest -q " + " ".join(cmd[4:]), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
