"""``arsc.help()`` and ``arsc.recipes()``: the task map and the copy-paste recipes, from Python."""
from __future__ import annotations

import re

from ._recipes import RECIPES, TASKS

_STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "my", "do", "does", "is", "are", "i", "want",
         "how", "what", "which", "with", "for", "between", "these", "this", "me", "show", "find", "get", "after",
         "before", "per", "by", "from", "cells", "cell"}


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP}


def help(topic: str | None = None, *, show: bool = True) -> str:  # noqa: A001 - deliberate API name
    """Print the task -> function map ("I want to ..." -> the arsc call).

    Parameters
    ----------
    topic : optional filter; only rows whose section, task or call contain this text (case-insensitive)
        are shown, e.g. ``"clone"``, ``"DE"``, ``"markers"``.
    show : print the table (default). The text is returned either way.

    Returns
    -------
    str, the formatted table.

    Examples
    --------
    >>> import arsc
    >>> txt = arsc.help("pseudobulk", show=False)
    >>> "arsc.pseudobulk" in txt
    True
    """
    rows = TASKS
    if topic:
        t = topic.lower()
        rows = [r for r in TASKS if t in r[0].lower() or t in r[1].lower() or t in r[2].lower()]
    lines = []
    sec = None
    for s, task, call in rows:
        if s != sec:
            lines.append(f"\n[{s}]")
            sec = s
        lines.append(f"  {task}\n      {call}")
    if not rows:
        lines.append(f"no task matches {topic!r}; call arsc.help() for the full map or arsc.recipes() for recipes")
    lines.append("\nRecipes (complete, runnable): arsc.recipes() lists them; arsc.recipes('<id or a question>') prints one.")
    out = "\n".join(lines).lstrip("\n")
    if show:
        print(out)
    return out


def recipes(query: str | None = None, *, show: bool = True) -> str:
    """List the recipes, or print the one matching an id or a plain-English question.

    Parameters
    ----------
    query : ``None`` lists every recipe (id, title, example questions). A recipe id (e.g. ``"de_paired"``)
        prints that recipe's code. Any other text is matched against the titles and example questions
        (word overlap), e.g. ``"compare responders and non-responders in CD8 T cells"``.
    show : print the result (default). The text is returned either way.

    Returns
    -------
    str, the listing or the recipe code (runnable on ``arsc.datasets.toy()``; swap in your data and column names).

    Examples
    --------
    >>> import arsc
    >>> code = arsc.recipes("do these clones expand after treatment", show=False)
    >>> "clone_dynamics" in code
    True
    >>> "de_paired" in arsc.recipes(show=False)
    True
    """
    if query is None:
        out = "\n".join(f"{r['id']:<18} {r['title']}\n{'':<18} e.g. " + "; ".join(f'"{a}"' for a in r["asks"][:3])
                        for r in RECIPES)
        out += "\n\narsc.recipes('<id>') prints the code; arsc.help() shows the task -> function map."
    else:
        by_id = {r["id"]: r for r in RECIPES}
        r = by_id.get(query.strip())
        if r is None:
            q = _words(query)
            scored = []
            for i, rr in enumerate(RECIPES):
                vocab = [_words(rr["title"])] + [_words(a) for a in rr["asks"]]
                score = max(len(q & v) / (len(v) ** 0.5 or 1) for v in vocab)
                scored.append((score, -i, rr))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if not scored or scored[0][0] == 0:
                out = f"no recipe matches {query!r}; arsc.recipes() lists them all"
                if show:
                    print(out)
                return out
            r = scored[0][2]
        out = f"# {r['title']}  (recipe '{r['id']}')\n{r['code']}"
    if show:
        print(out)
    return out
