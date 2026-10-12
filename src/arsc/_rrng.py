"""R-compatible random numbers (R >= 3.6 defaults: Mersenne-Twister, Inversion, Rejection).

Lets ports of R functions that draw random numbers (e.g. Seurat's
``AddModuleScore``, which jitters gene means with ``rnorm`` and picks control
genes with ``sample``) reproduce R's output exactly for the same ``set.seed``.

Implements ``set.seed`` scrambling (``RNG_Init``), ``unif_rand`` (MT19937 with
R's ``fixup``), ``norm_rand`` by inversion (``BIG = 2^27`` two-draw scheme) and
``R_unif_index`` rejection sampling used by ``sample``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

_M32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)
_BIG = 134217728  # 2^27


class RRandom:
    """Stateful R RNG. ``RRandom(1)`` == ``set.seed(1)`` in R."""

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        s = int(seed) & _M32
        for _ in range(50):
            s = (69069 * s + 1) & _M32
        seeds = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & _M32
            seeds[j] = s
        # i_seed[0] is mti (set to 624 by FixupSeeds), i_seed[1:] is the MT state
        self._bg = np.random.MT19937()
        self._bg.state = {"bit_generator": "MT19937", "state": {"key": seeds[1:].copy(), "pos": 624}}

    def unif_rand(self, n: int = 1) -> np.ndarray:
        y = self._bg.random_raw(n).astype(np.float64)
        u = y * 2.3283064365386963e-10
        u = np.where(u <= 0.0, 0.5 * _I2_32M1, u)
        u = np.where(1.0 - u <= 0.0, 1.0 - 0.5 * _I2_32M1, u)
        return u

    def rnorm(self, n: int) -> np.ndarray:
        u = self.unif_rand(2 * n).reshape(n, 2)
        v = np.floor(_BIG * u[:, 0]) + u[:, 1]  # (int)(BIG*u) + unif_rand()
        return ndtri(v / _BIG)

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand(1)[0] * 65536))
            v = 65536 * v + v1
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """R_unif_index (rejection sampling) -> integer in [0, dn)."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int) -> np.ndarray:
        """``sample.int(n, size)`` without replacement, returned 0-based."""
        if size > n:
            raise ValueError("cannot take a sample larger than the population when 'replace = FALSE'")
        if size < 2:
            return np.array([self.unif_index(n) for _ in range(size)], dtype=np.int64)
        x = list(range(n))
        out = np.empty(size, dtype=np.int64)
        m = n
        for i in range(size):
            j = self.unif_index(m)
            out[i] = x[j]
            m -= 1
            x[j] = x[m]
        return out
