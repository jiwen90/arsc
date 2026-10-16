"""Record the seed every stochastic arsc call used (see SEEDS.md, "Recording the seed").

``records_seed(param)`` wraps a public stochastic function. After a successful call it

* sets ``result.attrs["seed"]`` (and ``attrs["seed_function"]``) when the result is a pandas
  DataFrame or Series, so a saved table can say which seed produced it;
* adds the call to a per-process log, ``arsc.seeds_used()``. Only the outermost recorded call is
  logged: ``paired_test`` calling ``boot_ci``, or ``clonotrace.run`` calling ``propagate_clones``,
  is one entry (the seed passed down is the same, SEEDS.md rule 5). Repeated calls with the same
  function and seed are counted, not listed again, so the log stays small inside loops.

The log never touches any RNG and never writes to disk; a caller (e.g. a run-config writer) reads it.
"""
from __future__ import annotations

import functools
import inspect
import threading

_lock = threading.Lock()
_log: dict = {}  # (function, param, seed) -> number of outermost calls, insertion-ordered
_state = threading.local()


def _name(fn) -> str:
    mod = fn.__module__.split(".", 1)[1] if fn.__module__.startswith("arsc.") else fn.__module__
    return {"preprocessing": "pp"}.get(mod, mod) + "." + fn.__name__


def _as_seed(v):
    try:
        import numpy as np
        if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
            return int(v)
    except Exception:  # pragma: no cover
        pass
    return repr(v)


def records_seed(param: str):
    """Decorator: record the value of the seed parameter ``param`` of each call (module docstring)."""
    def deco(fn):
        sig = inspect.signature(fn)
        if param not in sig.parameters:
            raise TypeError(f"{fn.__name__} has no parameter {param!r}")
        default = sig.parameters[param].default
        name = _name(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            depth = getattr(_state, "depth", 0)
            if param in kwargs:
                seed = kwargs[param]
            else:
                try:
                    seed = sig.bind(*args, **kwargs).arguments.get(param, default)
                except TypeError:  # bad call: let the function raise its own error
                    return fn(*args, **kwargs)
            _state.depth = depth + 1
            try:
                out = fn(*args, **kwargs)
            finally:
                _state.depth = depth
            seed = _as_seed(seed)
            if hasattr(out, "attrs") and hasattr(out, "iloc"):  # pandas DataFrame / Series
                try:
                    out.attrs["seed"] = seed
                    out.attrs["seed_function"] = name
                except Exception:  # pragma: no cover
                    pass
            if depth == 0:
                key = (name, param, seed)
                with _lock:
                    _log[key] = _log.get(key, 0) + 1
            return out

        wrapper.__arsc_seed_param__ = param
        return wrapper
    return deco


def seeds_used(reset: bool = False) -> list:
    """Seeds used by stochastic arsc calls in this process (outermost calls only).

    Parameters
    ----------
    reset : clear the log after reading it.

    Returns
    -------
    list of dicts ``{"function", "param", "seed", "calls"}`` in first-call order.

    Examples
    --------
    >>> import arsc
    >>> arsc.reset_seeds_used()
    >>> _ = arsc.stats.boot_ci([1.0, 2.0, 3.0, 4.0], seed=7)
    >>> arsc.seeds_used()
    [{'function': 'stats.boot_ci', 'param': 'seed', 'seed': 7, 'calls': 1}]
    """
    with _lock:
        out = [{"function": f, "param": p, "seed": s, "calls": n} for (f, p, s), n in _log.items()]
        if reset:
            _log.clear()
    return out


def reset_seeds_used() -> None:
    """Clear the seed log of this process (``seeds_used``).

    Returns
    -------
    None

    Examples
    --------
    >>> import arsc
    >>> arsc.reset_seeds_used()
    >>> arsc.seeds_used()
    []
    """
    with _lock:
        _log.clear()
