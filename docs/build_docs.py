"""Regenerate the generated parts of the docs from the package itself.

    python docs/build_docs.py          # write AGENTS.md (task map + recipes), docs/recipes.md, docs/api.md
    python docs/build_docs.py --check  # exit 1 if any of them is out of date (used by the tests)

Sources: ``arsc._recipes.TASKS`` / ``RECIPES`` and the docstrings of every public function.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arsc  # noqa: E402
from arsc._recipes import RECIPES, TASKS  # noqa: E402

API_MODULES = [
    ("arsc", "Top level (I/O, preprocessing shortcuts, markers, scores, graphs)"),
    ("arsc.datasets", "Toy datasets"),
    ("arsc.pp", "Preprocessing (Seurat conventions)"),
    ("arsc.formats", "Readers and writers"),
    ("arsc.io", "Store, Dataset and the Seurat bridge"),
    ("arsc.views", "Subset stores as views of their parent stores"),
    ("arsc.markers", "Markers (presto)"),
    ("arsc.pseudobulk", "Pseudobulk"),
    ("arsc.scores", "Signature scores"),
    ("arsc.neighbors", "Neighbours and WNN"),
    ("arsc.cluster", "Clustering and UMAP"),
    ("arsc.harmony", "Harmony"),
    ("arsc.symphony", "Symphony reference mapping"),
    ("arsc.de", "Differential expression and GSEA"),
    ("arsc.stats", "Patient-level statistics"),
    ("arsc.clones", "TCR clones"),
    ("arsc.clonotrace", "CloneTrace (clonal fate)"),
    ("arsc.genesets", "Gene sets"),
    ("arsc.seurat_extra", "Seurat parity"),
    ("arsc.scanpy_extra", "scanpy parity"),
]


def tasks_md() -> str:
    lines = ["| Area | I want to ... | Call |", "|---|---|---|"]
    for sec, task, call in TASKS:
        call = call.replace("|", "\\|")
        lines.append(f"| {sec} | {task} | `{call}` |")
    return "\n".join(lines)


def recipes_md(level: str = "###") -> str:
    out = []
    for r in RECIPES:
        asks = "; ".join(f'"{a}"' for a in r["asks"])
        out.append(f"{level} `{r['id']}`: {r['title']}\n\nAsks like: {asks}\n\n```python\n{r['code'].rstrip()}\n```")
    return "\n\n".join(out)


def _replace(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker <!-- {tag}:START --> missing")
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def agents_md() -> str:
    t = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    t = _replace(t, "TASKS", tasks_md())
    return _replace(t, "RECIPES", recipes_md())


def recipes_page() -> str:
    head = ("# Recipes\n\nGenerated from `src/arsc/_recipes.py` by `docs/build_docs.py`; do not edit by hand.\n"
            "Each recipe runs as written on `arsc.datasets.toy()` (tested in `tests/test_docs_snippets.py`).\n"
            "In Python: `arsc.recipes()` lists them, `arsc.recipes(\"<id or question>\")` prints one.\n\n")
    return head + recipes_md("##") + "\n"


def _public(modname: str):
    mod = arsc if modname == "arsc" else eval(modname, {"arsc": arsc})  # noqa: S307 - fixed list above
    names = getattr(mod, "__all__", None)
    if modname == "arsc":
        names = [n for n, o in vars(arsc).items() if not n.startswith("_") and callable(o)
                 and not inspect.ismodule(o) and getattr(o, "__module__", "").startswith("arsc")]
        names = [n for n in names if getattr(getattr(arsc, n), "__module__", "") in ("arsc._help", "arsc.io", "arsc.formats", "arsc._seedlog")]
    if names is None:
        names = [n for n, o in vars(mod).items() if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o))
                 and getattr(o, "__module__", "") == mod.__name__]
    return mod, sorted(dict.fromkeys(names), key=str.lower)


def api_md() -> str:
    out = ["# API reference", "",
           "Generated from the docstrings by `docs/build_docs.py`; do not edit by hand. Start with `arsc.help()`.", ""]
    toc = []
    body = []
    seen = set()
    for modname, title in API_MODULES:
        mod, names = _public(modname)
        items = []
        for n in names:
            obj = getattr(mod, n, None)
            if obj is None or not (callable(obj) or inspect.isclass(obj)):
                continue
            key = (getattr(obj, "__module__", ""), getattr(obj, "__qualname__", n))
            if key in seen and modname == "arsc":
                continue
            seen.add(key)
            items.append((n, obj))
        if not items:
            continue
        anchor = modname.replace(".", "")
        toc.append(f"* [{modname}](#{anchor}): {title}")
        body.append(f"## {modname}\n\n{title}.\n")
        for n, obj in items:
            try:
                sig = re.sub(r"<function (\w+) at 0x[0-9A-Fa-f]+>", lambda m: m.group(1), str(inspect.signature(obj)))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = inspect.getdoc(obj) or "(no docstring)"
            body.append(f"### `{modname}.{n}{sig}`\n\n```text\n{doc}\n```\n")
    return "\n".join(out + toc + [""] + body)


def targets() -> dict:
    return {ROOT / "AGENTS.md": agents_md(), ROOT / "docs" / "recipes.md": recipes_page(),
            ROOT / "docs" / "api.md": api_md()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    stale = []
    for p, txt in targets().items():
        cur = p.read_text(encoding="utf-8") if p.exists() else ""
        if cur != txt:
            stale.append(p)
            if not a.check:
                p.write_text(txt, encoding="utf-8", newline="\n")
    if a.check and stale:
        print("out of date (run python docs/build_docs.py):", *[str(s.relative_to(ROOT)) for s in stale])
        return 1
    print("written:" if not a.check else "up to date", *[str(s.relative_to(ROOT)) for s in stale] or [""])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
