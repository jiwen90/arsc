"""Synthetic inputs for the R parity of arsc.stats resampling / rank helpers (STATS_PARITY.md, 'Rank, randomisation
and stratified helpers'). Writes CSVs to $ARSC_SCRATCH/stats_resampling/; then run
``Rscript benchmarks/r_reference_stats_resampling.R <that dir>`` and ``pytest tests/test_stats_resampling.py``.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    cases, vals = [], []
    cid = 0

    def add(kind, x, y=None, alternative="two-sided", conf=0.95, exact="NA", correct=True):
        nonlocal cid
        cid += 1
        cases.append(dict(case=cid, kind=kind, alternative=alternative, conf_level=conf, exact=exact,
                          correct=correct))
        vals.extend(dict(case=cid, grp="x", i=i, value=v) for i, v in enumerate(x))
        if y is not None:
            vals.extend(dict(case=cid, grp="y", i=i, value=v) for i, v in enumerate(y))

    alts = ["two-sided", "greater", "less"]
    for rep in range(90):
        n = int(rng.integers(3, 31))
        x = rng.normal(0.4, 1, n)
        style = rep % 3
        if style == 1:
            x = np.round(x, 1)            # ties (normal branch)
        if style == 2 and n > 4:
            x[:2] = 0.0                   # zeros (normal branch)
        add("one", x, alternative=alts[rep % 3], conf=[0.9, 0.95, 0.99][rep // 30])
    for rep in range(12):
        n = int(rng.integers(50, 70))
        add("one", rng.normal(0.2, 1, n), alternative=alts[rep % 3], correct=bool(rep % 2))
    for rep in range(30):
        n = int(rng.integers(3, 20))
        a = rng.normal(0, 1, n); b = a + rng.normal(0.3, 0.7, n)
        add("paired", b, a, alternative=alts[rep % 3])
    for rep in range(90):
        nx, ny = int(rng.integers(2, 21)), int(rng.integers(2, 21))
        x = rng.normal(0.5, 1, nx); y = rng.normal(0, 1, ny)
        if rep % 3 == 1:
            x, y = np.round(x, 1), np.round(y, 1)
        add("two", x, y, alternative=alts[rep % 3], conf=[0.9, 0.95, 0.99][rep // 30])
    for rep in range(9):
        add("two", rng.normal(0.3, 1, 55), rng.normal(0, 1, 52), alternative=alts[rep % 3], correct=bool(rep % 2))
    pd.DataFrame(cases).to_csv(out / "hl_cases.csv", index=False)
    pd.DataFrame(vals).to_csv(out / "hl_values.csv", index=False)

    # Mantel-Haenszel: K strata of 2x2 tables
    mh = []
    for case in range(1, 41):
        K = int(rng.integers(2, 9))
        lor = rng.normal(0.5, 0.6)
        for k in range(K):
            n1, n0 = int(rng.integers(3, 40)), int(rng.integers(3, 40))
            p0 = rng.uniform(0.1, 0.6)
            p1 = 1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + lor)))
            a = int(rng.binomial(n1, p1)); c = int(rng.binomial(n0, p0))
            mh.append(dict(case=case, stratum=k, a=a, b=n1 - a, c=c, d=n0 - c, correct=bool(case % 2)))
    pd.DataFrame(mh).to_csv(out / "mh_tables.csv", index=False)

    # rarefaction: clone-size vectors and depths
    rf = []
    for case in range(1, 41):
        S = int(rng.integers(5, 300))
        sizes = rng.geometric(rng.uniform(0.05, 0.8), S)
        N = int(sizes.sum())
        depth = int(rng.integers(1, N + 1))
        rf.extend(dict(case=case, depth=depth, clone=i, size=int(v)) for i, v in enumerate(sizes))
    pd.DataFrame(rf).to_csv(out / "rarefy_counts.csv", index=False)

    # ICC: balanced one-way data (k groups x m) with between-group SD > 0
    ic = []
    for case in range(1, 31):
        k, m = int(rng.integers(4, 30)), int(rng.integers(2, 8))
        mu = rng.normal(0, rng.uniform(0.5, 2), k)
        for g in range(k):
            for v in mu[g] + rng.normal(0, 1, m):
                ic.append(dict(case=case, group=f"g{g}", stratum="s0", y=v))
    # nested in strata (balanced within strata)
    for case in range(31, 46):
        m = int(rng.integers(2, 6))      # equal group size: ANOVA = REML (when > 0) also with strata
        for s_ in range(int(rng.integers(2, 6))):
            k = int(rng.integers(3, 12))
            off = rng.normal(0, 3)
            mu = rng.normal(0, 1.2, k)
            for g in range(k):
                for v in off + mu[g] + rng.normal(0, 1, m):
                    ic.append(dict(case=case, group=f"s{s_}g{g}", stratum=f"s{s_}", y=v))
    pd.DataFrame(ic).to_csv(out / "icc_values.csv", index=False)

    # OLS t for Freedman-Lane observed statistic
    fl = []
    for case in range(1, 21):
        n = int(rng.integers(6, 40)); q = int(rng.integers(0, 3))
        Z = rng.normal(size=(n, q)); x = Z.sum(1) * 0.7 + rng.normal(size=n); y = Z.sum(1) + 0.3 * x + rng.normal(size=n)
        for i in range(n):
            fl.append(dict(case=case, i=i, y=y[i], x=x[i], **{f"z{j}": Z[i, j] for j in range(2)} if q == 2 else
                           ({"z0": Z[i, 0], "z1": np.nan} if q == 1 else {"z0": np.nan, "z1": np.nan})))
    pd.DataFrame(fl).to_csv(out / "fl_values.csv", index=False)


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["ARSC_SCRATCH"]) / "stats_resampling"
    main(base)
    print("wrote", base)
