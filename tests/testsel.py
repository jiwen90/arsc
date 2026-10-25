#!/usr/bin/env python
"""testsel: run only the tests a change can break, by exact file (and test) name.

    python tests/testsel.py                     # changed files vs the merge base with main + uncommitted + untracked
    python tests/testsel.py src/pkg/mod.py ...  # these paths instead of git's list
    python tests/testsel.py --base HEAD~1       # changes since another ref (e.g. the last commit, on main)
    python tests/testsel.py --list              # print the selection and the pytest command(s); run nothing
    python tests/testsel.py --why               # also print, per selected test, which changed path chose it
    python tests/testsel.py --deep              # also tests that reach a changed module through other modules
    python tests/testsel.py ... -- -x -n 0      # anything after -- goes to pytest unchanged

How a changed path maps to tests (DIRECT links only; the full suite is not run here):
  * a test file              -> that file
  * a helper the tests import by bare name (helpers.py, fixtures.py, this file) -> the test files importing it
  * a package module         -> the test files that import it or use it as an attribute (pkg.mod.f, from pkg import f
                                resolved through the package __init__ re-exports, "pkg.mod" named in a string), plus
                                that module's own doctest / docstring nodes in the docs test file. A module no test
                                uses directly falls back to the nearest modules that import it (printed as "via").
  * a docs file (.md)        -> only the docs-test nodes of that file (e.g. test_doc_snippets_run[README.md])
  * any other file           -> test files that name it (path or file name) + the OVERRIDES table below
  * anything else            -> printed as "no test reads it"; nothing runs for it.
The output is always exact paths (tests/test_x.py, tests/test_x.py::test_y[param]), never the tests directory.

The full suite runs on GitHub CI on push; locally only when the maintainer asks (ARSC_FULL=1 python -m pytest -q tests): the conftest
refuses a directory argument without that variable. Pure standard library; reads source with ast, imports nothing.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- per-repository configuration
CONFIG = {
    "full_env": "ARSC_FULL",
    # suites: each runs as `python -m pytest -q <files>` from its own directory (paths relative to ROOT)
    "suites": [{"dir": ".", "tests": "tests"}],
    # importable packages -> their source directory
    "packages": {"arsc": "src/arsc"},
    # docs test nodes: file -> (test function, how its parameter ids are built: "doc" = the doc's relative path,
    # "mod" = the module name relative to the package); a changed doc / module selects only its own nodes
    "doc_tests": {
        "tests/test_docs_snippets.py": [("test_doc_snippets_run", "doc")],
        "tests/test_docstrings.py": [("test_doctests", "mod")],
    },
    # paths (fnmatch globs) -> extra tests (files or node ids) that the import map cannot see
    "overrides": {
        "src/arsc/_recipes.py": ["tests/test_docs_snippets.py::test_generated_docs_up_to_date",
                                 "tests/test_docs_snippets.py::test_recipe_questions_find_their_recipe"],
        "src/arsc/_help.py": ["tests/test_docs_snippets.py::test_generated_docs_up_to_date",
                              "tests/test_docs_snippets.py::test_help_covers_the_workflow"],
        "docs/build_docs.py": ["tests/test_docs_snippets.py::test_generated_docs_up_to_date"],
        "AGENTS.md": ["tests/test_docs_snippets.py::test_generated_docs_up_to_date"],
        "docs/api.md": ["tests/test_docs_snippets.py::test_generated_docs_up_to_date"],
        "docs/recipes.md": ["tests/test_docs_snippets.py::test_generated_docs_up_to_date"],
        "tests/conftest.py": ["tests/test_repo_hygiene.py"],
        ".gitignore": ["tests/test_repo_hygiene.py"],
        "SEEDS.md": ["tests/test_seeds.py"],
    },
    # never select anything for these (records, licences, CI config: GitHub runs it)
    "ignore": ["CHANGELOG.md", "LICENSE", "LICENSES/*", "NOTICE", "CITATION.cff", "MANIFEST.in", ".github/*",
               "*_PARITY.md", "ROADMAP.md", "llms.txt", "benchmarks/results/*"],
    # tests that only run on request (skip themselves otherwise); never selected by a source change
    "opt_in": ["tests/test_speed_gate.py"],
}
# ---------------------------------------------------------------- end of configuration

DOC_EXT = {".md", ".rst", ".txt"}
FALLBACK = 3  # test files sampled for a module no test uses directly


def git(*args):
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []


def changed(base=None):
    """Committed changes since `base` (default: merge base with main), plus staged, unstaged and untracked files."""
    if base is None:
        mb = git("merge-base", "HEAD", "main") or git("merge-base", "HEAD", "origin/main")
        base = mb[0] if mb else "HEAD"
    out = git("diff", "--name-only", base) + git("ls-files", "--others", "--exclude-standard")
    return sorted(set(p.replace("\\", "/") for p in out))


def _parse(path):
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None


class Repo:
    """Module index, package re-exports, test files, and the direct links between them."""

    def __init__(self, cfg=CONFIG, root=ROOT):
        self.cfg, self.root = cfg, Path(root)
        self.mods = {}  # dotted module -> repo-relative path
        for pkg, src in cfg["packages"].items():
            base = self.root / src
            for f in sorted(base.rglob("*.py")):
                rel = f.relative_to(base).with_suffix("")
                parts = [pkg, *rel.parts]
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                self.mods[".".join(parts)] = f.relative_to(self.root).as_posix()
        self.path2mod = {p: m for m, p in self.mods.items()}
        self.exports = {m: self._exports(m) for m in self.mods if self.mods[m].endswith("__init__.py")}
        self.tests, self.helpers = [], {}  # test files; bare-name helper module -> path
        for s in cfg["suites"]:
            td = self.root / s["tests"]
            for f in sorted(td.glob("*.py")):
                rel = f.relative_to(self.root).as_posix()
                if f.name.startswith("test_"):
                    self.tests.append(rel)
                elif f.name != "conftest.py":
                    self.helpers[f.stem] = rel
        self.uses = {t: self._uses(t) for t in self.tests}  # test -> {module or helper path}
        self.texts = {t: (self.root / t).read_text(encoding="utf-8", errors="replace") for t in self.tests}
        self._imports, self._collected = None, {}

    # -- package structure
    def _exports(self, pkg):
        """name -> defining module, from `from .x import a` / `from . import x` in a package __init__."""
        out = {}
        tree = _parse(self.root / self.mods[pkg])
        if tree is None:
            return out
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level:
                base = self._rel(pkg, n.level, n.module, is_pkg=True)
                for a in n.names:
                    sub = f"{base}.{a.name}"
                    if sub in self.mods:
                        out[a.asname or a.name] = sub
                    elif base in self.mods:
                        out[a.asname or a.name] = base
        return out

    def _rel(self, mod, level, name, is_pkg=False):
        parts = mod.split(".")
        if not is_pkg:
            parts = parts[:-1]
        parts = parts[: len(parts) - (level - 1)] if level > 1 else parts
        return ".".join(parts + ([name] if name else []))

    def resolve(self, dotted):
        """'arsc.stats.paired_test' -> 'arsc.stats'; 'arsc.pp' -> 'arsc.preprocessing'; unknown -> None."""
        parts = dotted.split(".")
        if parts[0] not in self.cfg["packages"]:
            return None
        cur = parts[0]
        for p in parts[1:]:
            nxt = f"{cur}.{p}"
            if nxt in self.mods:
                cur = nxt
            elif cur in self.exports and p in self.exports[cur]:
                tgt = self.exports[cur][p]
                if tgt == cur:
                    return cur
                cur = tgt
                if not self.mods[cur].endswith("__init__.py"):
                    return cur  # a name defined in (or a module re-exported as) a module: stop here
            else:
                return cur
        return cur

    # -- what a test file uses
    def _uses(self, test):
        tree = _parse(self.root / test)
        out = set()
        if tree is None:
            return out
        alias = {}  # local name -> dotted module path it stands for
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    top = a.name.split(".")[0]
                    if top in self.cfg["packages"]:
                        m = self.resolve(a.name)
                        if m:
                            out.add(m)
                        alias[a.asname or top] = a.name if a.asname else top
                    elif a.name in self.helpers:
                        out.add(self.helpers[a.name])
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                top = n.module.split(".")[0]
                if top in self.cfg["packages"]:
                    for a in n.names:
                        m = self.resolve(f"{n.module}.{a.name}") if a.name != "*" else self.resolve(n.module)
                        if m:
                            out.add(m)
                            if f"{n.module}.{a.name}" in self.mods:
                                alias[a.asname or a.name] = f"{n.module}.{a.name}"
                elif n.module in self.helpers:
                    out.add(self.helpers[n.module])
        for n in ast.walk(tree):  # attribute chains rooted at an alias: arsc.stats.mde, pp.pca
            if isinstance(n, ast.Attribute):
                chain, cur = [], n
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in alias:
                    m = self.resolve(".".join([alias[cur.id], *reversed(chain)]))
                    if m:
                        out.add(m)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):  # "arsc.views", import_module(...)
                for s in re.findall(r"\b((?:%s)(?:\.\w+)+)" % "|".join(self.cfg["packages"]), n.value):
                    m = self.resolve(s)
                    if m and m in self.mods and "." in m:
                        out.add(m)
        top = set(self.cfg["packages"])
        return {u for u in out if u not in top}  # `import arsc` alone links to nothing in particular

    # -- module import graph (for --deep and the fallback)
    def imports(self):
        if self._imports is None:
            g = {}
            for m, p in self.mods.items():
                tree, deps = _parse(self.root / p), set()
                is_pkg = p.endswith("__init__.py")
                for n in ast.walk(tree) if tree else ():
                    if isinstance(n, ast.ImportFrom):
                        base = self._rel(m, n.level, n.module, is_pkg) if n.level else (n.module or "")
                        for a in n.names:
                            d = self.resolve(f"{base}.{a.name}") or self.resolve(base)
                            if d:
                                deps.add(d)
                    elif isinstance(n, ast.Import):
                        for a in n.names:
                            d = self.resolve(a.name)
                            if d:
                                deps.add(d)
                g[m] = deps - {m} - set(self.cfg["packages"])
            self._imports = g
        return self._imports

    def importers(self, m):
        return sorted(k for k, v in self.imports().items() if m in v and not self.mods[k].endswith("__init__.py"))

    def closure(self, m):
        seen, todo = {m}, [m]
        while todo:
            for k in self.importers(todo.pop()):
                if k not in seen:
                    seen.add(k)
                    todo.append(k)
        return seen

    # -- docs test nodes
    def doc_nodes(self, kind, key):
        out = []
        for f, entries in self.cfg["doc_tests"].items():
            for fn, k in entries:
                if k != kind:
                    continue
                for pid, node in self.param_nodes(f, fn):
                    norm = pid.replace("\\", "/")
                    if norm == key or norm.startswith(key + "#") or norm.startswith(key + "."):
                        out.append(node)
        return out

    def param_nodes(self, test_file, fn):
        """[(parameter id, node id)] of fn: from its parametrize list when that is a literal (or a module-level
        literal list such as MODULES / DOCS); otherwise by asking pytest (--collect-only, once per file)."""
        tree = _parse(self.root / test_file)
        if tree is None:
            return []
        consts = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                try:
                    consts[n.targets[0].id] = ast.literal_eval(n.value)
                except ValueError:
                    pass
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name == fn:
                for d in n.decorator_list:
                    if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize" and len(d.args) > 1:
                        src, names = d.args[1], None
                        if isinstance(src, ast.Name):
                            names = consts.get(src.id)
                        elif isinstance(src, ast.ListComp) and isinstance(src.generators[0].iter, ast.Name):
                            names = consts.get(src.generators[0].iter.id)
                        if isinstance(names, (list, tuple)) and not d.keywords:
                            # pytest writes an id the way unicode_escape does (a backslash doubled)
                            return [(str(x), f"{test_file}::{fn}[{str(x).encode('unicode_escape').decode('ascii')}]")
                                    for x in names]
                        pre = f"{test_file}::{fn}["
                        return [(nd[len(pre):-1], nd) for nd in self.collect(test_file) if nd.startswith(pre)]
        return []

    def collect(self, test_file):
        if test_file not in self._collected:
            r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                                *self.cfg.get("collect_args", []), test_file], cwd=str(self.root),
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self._collected[test_file] = [l.strip() for l in r.stdout.splitlines() if "::" in l]
        return self._collected[test_file]


def select(paths, repo=None, deep=False):
    """-> (ordered list of pytest targets, {target: reason}, [paths no test reads])."""
    repo = repo or Repo()
    cfg = repo.cfg
    chosen, why, uncovered = [], {}, []

    def add(t, reason):
        if t not in why:
            chosen.append(t)
            why[t] = reason

    by_helper = {}  # module or helper -> test files using it (docs test files only through their own nodes)
    for t, u in repo.uses.items():
        if t in cfg["doc_tests"]:
            continue
        for x in u:
            by_helper.setdefault(x, []).append(t)
    for p in paths:
        p = p.replace("\\", "/")
        before = len(chosen)
        if any(fnmatch.fnmatch(p, g) for g in cfg["ignore"]):
            continue
        for g, ts in cfg["overrides"].items():
            if fnmatch.fnmatch(p, g):
                for t in ts:
                    add(t, f"override for {p}")
        if p in repo.tests:
            add(p, "changed test file")
        elif p in repo.helpers.values():
            for t in sorted(by_helper.get(p, [])):
                add(t, f"imports {Path(p).stem}")
        elif p in repo.path2mod:
            m = repo.path2mod[p]
            mods = sorted(repo.closure(m)) if deep else [m]
            for mm in mods:
                for t in sorted(by_helper.get(mm, [])):
                    if t not in cfg["opt_in"]:
                        add(t, f"uses {mm}" + ("" if mm == m else f" (imports {m})"))
            mod_key = m.split(".", 1)[1] if "." in m else m
            for node in repo.doc_nodes("mod", mod_key):
                add(node, f"doctests of {m}")
            if len(chosen) == before or all(c.split("::")[0] in cfg["doc_tests"] for c in chosen[before:]):
                # no test uses it directly: a small sample of the nearest importers' tests (greedy cover of the
                # importers, at most FALLBACK files); --deep runs every test that reaches it
                level, seen = [m], {m}
                while level:
                    nxt = []
                    for x in level:
                        for k in repo.importers(x):
                            if k not in seen:
                                seen.add(k)
                                nxt.append(k)
                    cover = {}
                    for k in nxt:
                        for t in by_helper.get(k, []):
                            if t not in cfg["opt_in"]:
                                cover.setdefault(t, set()).add(k)
                    left, picked = set().union(*cover.values()) if cover else set(), 0
                    while left and picked < FALLBACK:
                        t = max(sorted(cover), key=lambda x: len(cover[x] & left))
                        add(t, f"via {', '.join(sorted(cover[t] & left))} (no test uses {m} directly)")
                        left -= cover[t]
                        picked += 1
                    if cover:
                        break
                    level = nxt
        elif Path(p).suffix in DOC_EXT and repo.doc_nodes("doc", p):
            for node in repo.doc_nodes("doc", p):
                add(node, f"python blocks of {p}")
        else:  # any other file: the tests that name it
            name = Path(p).name
            for t in repo.tests:
                if t == p or t in cfg["doc_tests"] or t in cfg["opt_in"]:
                    continue
                txt = repo.texts[t]
                if p in txt or (len(name) > 6 and re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])", txt)):
                    add(t, f"names {p}")
        if len(chosen) == before:
            uncovered.append(p)
    return chosen, why, uncovered


def commands(targets, cfg=CONFIG):
    """[(cwd, [targets relative to cwd])], one per suite."""
    out = []
    for s in cfg["suites"]:
        pre = "" if s["dir"] == "." else s["dir"].rstrip("/") + "/"
        mine = [t[len(pre):] for t in targets if t.startswith(s["tests"].rstrip("/") + "/")]
        # a whole file selected: drop its narrower node ids
        files = {t for t in mine if "::" not in t}
        mine = [t for t in mine if "::" not in t or t.split("::")[0] not in files]
        if mine:
            out.append((s["dir"], mine))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--base")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--why", action="store_true")
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args(argv)
    paths = a.paths or changed(a.base)
    if not paths:
        print("no changed files: nothing to test (--base <ref> to compare with another commit)")
        return 0
    targets, why, uncovered = select(paths, deep=a.deep)
    print(f"{len(paths)} changed path(s); {len(targets)} test target(s)")
    if a.why:
        for t in targets:
            print(f"  {t}    <- {why[t]}")
    if uncovered:
        print("no test reads: " + ", ".join(uncovered[:20]) + (" ..." if len(uncovered) > 20 else ""))
    rc = 0
    for cwd, ts in commands(targets):
        cmd = [sys.executable, "-m", "pytest", "-q", *ts, *extra]
        print(("(cd " + cwd + " && " if cwd != "." else "") + "python -m pytest -q " + " ".join(ts + extra)
              + (")" if cwd != "." else ""), flush=True)
        if not a.list:
            rc = subprocess.run(cmd, cwd=str(ROOT / cwd)).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
