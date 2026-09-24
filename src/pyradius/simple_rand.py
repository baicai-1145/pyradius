"""pyradius.simple_rand — MSVC LCG + noise template (port of simple_rand.c).

state = state*214013 + 2531011 (u64 wraparound); out = (state>>16) & 0x7FFF.
Template fill: u = next/32767.0f; dst = u*2pi - pi (float32 math).
"""
from __future__ import annotations

import numpy as np

MUL = 214013
ADD = 2531011
MASK64 = (1 << 64) - 1
TWO_PI_F = np.float32(6.283185307179586)
PI_F = np.float32(3.141592653589793)


class SimpleRand:
    __slots__ = ("state",)

    def __init__(self, seed: int = 1):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state * MUL + ADD) & MASK64
        return (self.state >> 16) & 0x7FFF

    def skip(self, k: int) -> None:
        """Advance the LCG by k steps in O(log k), bitwise-equal to k next().

        state_{n+k} = state_n*M^k + A*S(k),  S(k) = (M^k-1)/(M-1) (integer).
        Binary composition:  S(a+b) = S(a) + M^a*S(b);  S(2m) = S(m)*(1+M^m).
        Maintains (M^(2^j), A*S(2^j)) and folds the set bits of k into s.
        """
        M, A, mask = MUL, ADD, MASK64
        mk = M                      # M^(2^0)
        ask = A                     # A * S(2^0) = A
        s = self.state
        e = k
        while e > 0:
            if e & 1:
                s = (s * mk + ask) & mask
            ask = (ask * (mk + 1)) & mask
            mk = (mk * mk) & mask
            e >>= 1
        self.state = s


_noise_fill_nb = None
try:
    from numba import njit

    @njit(cache=True)
    def _noise_fill_nb(dst, seed, mul, add, mask, two_pi, pi):
        s = seed
        n = dst.shape[0]
        for i in range(n):
            s = (s * mul + add) & mask
            out15 = (s >> 16) & np.uint64(0x7FFF)
            u = np.float32(np.float64(out15) / 32767.0)
            dst[i] = np.float32(u * two_pi) - pi
        return s
except Exception:
    _noise_fill_nb = None


def noise_template_fill(dst: np.ndarray, rng: SimpleRand) -> None:
    """Vectorized equivalent of C per-element loop (float32 rounding identical)."""
    n = dst.shape[0]
    if _noise_fill_nb is not None:
        rng.state = int(_noise_fill_nb(dst, np.uint64(rng.state),
                                       np.uint64(MUL), np.uint64(ADD),
                                       np.uint64(MASK64),
                                       TWO_PI_F, PI_F))
        return
    # batch generate LCG outputs
    s = np.uint64(rng.state)
    states = np.empty(n, dtype=np.uint64)
    mul = np.uint64(MUL)
    add = np.uint64(ADD)
    for i in range(n):
        states[i] = (s * mul + add) & MASK64
        s = states[i]
    rng.state = int(s)
    outs = ((states >> np.uint64(16)) & np.uint64(0x7FFF)).astype(np.float32)
    u = (outs / np.float32(32767.0)).astype(np.float32)
    np.multiply(u, TWO_PI_F, out=dst[:n], dtype=np.float32)
    dst[:n] -= PI_F
