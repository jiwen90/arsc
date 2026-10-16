"""Docstring coverage and doctests for the public API.

(1) Every public object (each module's ``__all__`` plus the public callables of the top-level ``arsc``
    namespace) has a numpy-style docstring: a one-line summary, a ``Parameters`` section naming every
    parameter, a ``Returns`` (or ``Attributes`` / ``Yields``) section and an ``Examples`` section with ``>>>``.
(2) The ``>>>`` examples of every module run (doctest), so the documented examples can never drift.
"""
from __future__ import annotations

import doctest
import importlib
import inspect
import os
import re

import pytest

import arsc

MODULES = ["preprocessing", "io", "formats", "markers", "pseudobulk", "scores", "neighbors", "cluster", "harmony", "symphony",
           "clones", "clonotrace", "genesets", "stats", "de", "scanpy_extra", "seurat_extra", "datasets"]


def _public_objects():
    seen, out = set(), []
    for m in MODULES:
        mod = importlib.import_module(f"arsc.{m}")
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue  # constants (gene lists, dicts) are documented in the module docstring
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            out.append((f"arsc.{m}.{name}", obj))
    for name, obj in vars(arsc).items():
        if name.startswith("_") or inspect.ismodule(obj) or id(obj) in seen:
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            seen.add(id(obj))
            out.append((f"arsc.{name}", obj))
    return out


def _params(obj):
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return []
    names = []
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        names.append(p.name)
    return names


def _problems(obj) -> list[str]:
    doc = inspect.getdoc(obj) or ""
    if not doc.strip():
        return ["no docstring"]
    probs = []
    if not doc.splitlines()[0].strip():
        probs.append("empty summary line")
    params = _params(obj)
    if inspect.isclass(obj):
        if not re.search(r"^(Attributes|Parameters)\s*\n-{3,}", doc, re.M):
            probs.append("no Attributes/Parameters section")
    else:
        if params:
            m = re.search(r"^Parameters\s*\n-{3,}\n(.*?)(?=^\w[\w ]*\n-{3,}|\Z)", doc, re.M | re.S)
            if not m:
                probs.append("no Parameters section")
            else:
                body = m.group(1)
                missing = [p for p in params if not re.search(rf"(^|[\s,`*]){re.escape(p)}\b", body)]
                if missing:
                    probs.append(f"Parameters missing {missing}")
        if not re.search(r"^(Returns|Yields)\s*\n-{3,}", doc, re.M):
            probs.append("no Returns section")
    if not re.search(r"^Examples?\s*\n-{3,}", doc, re.M) or ">>>" not in doc:
        probs.append("no Examples section with >>>")
    return probs


def test_public_docstrings_complete():
    bad = {name: p for name, obj in _public_objects() if (p := _problems(obj))}
    msg = "\n".join(f"  {k}: {'; '.join(v)}" for k, v in sorted(bad.items()))
    assert not bad, f"{len(bad)} public objects with incomplete docstrings:\n{msg}"


@pytest.mark.parametrize("module", MODULES)
def test_doctests(module, tmp_path):
    mod = importlib.import_module(f"arsc.{module}")
    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for t in finder.find(mod, mod.__name__, globs={"arsc": arsc}):
            if t.examples:
                runner.run(t, out=lambda s: None)
    finally:
        os.chdir(cwd)
    res = runner.summarize(verbose=False)
    if res.failed:
        # re-run verbosely for a readable message
        out = []
        os.chdir(tmp_path)
        try:
            r2 = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
            for t in finder.find(mod, mod.__name__, globs={"arsc": arsc}):
                if t.examples:
                    r2.run(t, out=out.append)
        finally:
            os.chdir(cwd)
        pytest.fail(f"{res.failed} doctest failures in arsc.{module}:\n" + "".join(out)[-6000:])
