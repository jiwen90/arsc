"""Every ```python block in the user/agent docs runs, so the docs cannot drift from the code.

Blocks of one file run in order in one namespace (like a notebook), in a temporary working directory.
Blocks fenced as ```python no-run (or any other language) are not executed. The generated parts of the docs
(AGENTS.md task map and recipes, docs/recipes.md, docs/api.md) must be up to date with the package.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["AGENTS.md", "README.md", "docs/getting-started.md", "docs/from-scanpy.md", "docs/from-seurat.md",
        ".claude/skills/arsc/SKILL.md"]
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.S | re.M)


def blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    out = []
    for m in FENCE.finditer(text):
        info = m.group(1).strip().split()
        if info[:1] == ["python"] and "no-run" not in info:
            out.append((text[: m.start()].count("\n") + 1, m.group(2)))
    return out


@pytest.mark.parametrize("doc", [d for d in DOCS if (ROOT / d).exists()])
def test_doc_snippets_run(doc, tmp_path, monkeypatch):
    bs = blocks(ROOT / doc)
    assert bs, f"{doc} has no runnable python blocks"
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__main__"}
    for line, code in bs:
        try:
            exec(compile(code, f"{doc}:{line}", "exec"), ns)
        except Exception as e:  # pragma: no cover - the message is the point
            raise AssertionError(f"{doc}, code block at line {line} failed: {type(e).__name__}: {e}") from e


def test_generated_docs_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "docs" / "build_docs.py"), "--check"], capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stdout + r.stderr


def test_recipe_questions_find_their_recipe():
    import arsc
    from arsc._recipes import RECIPES
    wrong = []
    for r in RECIPES:
        for q in r["asks"]:
            got = arsc.recipes(q, show=False).splitlines()[0]
            if f"(recipe '{r['id']}')" not in got:
                wrong.append(f"{q!r} -> {got}")
    assert not wrong, "\n".join(wrong)


def test_help_covers_the_workflow():
    import arsc
    txt = arsc.help(show=False)
    for call in ["arsc.open", "arsc.read", "arsc.pp.log_normalize", "arsc.pp.hvg_vst", "arsc.pp.pca", "arsc.run_harmony",
                 "arsc.find_neighbors", "arsc.cluster", "arsc.umap", "arsc.wilcoxauc", "arsc.score_ucell",
                 "arsc.pseudobulk", "arsc.de.deseq2", "arsc.de.voom", "arsc.de.fgsea", "arsc.clones.cyclone",
                 "arsc.clones.startrac", "arsc.stats.glmm_prop", "arsc.stats.paired_test", "arsc.stats.perm_test",
                 "arsc.stats.mde"]:
        assert call in txt, f"arsc.help() does not mention {call}"
