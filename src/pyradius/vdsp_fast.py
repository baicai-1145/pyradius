"""Route-B fast paths for pyradius.vdsp wrappers (single-allocation).

The python wrappers in vdsp.py pay ~10us of numpy/ctypes marshalling per
call; the render loop calls them ~30k times.  This module keeps persistent
scratch buffers + prebuilt ctypes structs keyed by size so a call is just:
ctoz -> fft_zrip -> ztoc + 3 scalar fixups, all raw vDSP.
"""
from __future__ import annotations

import ctypes

import numpy as np

from . import vdsp as _vd

_lib = _vd._lib
_Split = _vd._Split
FFT_FORWARD = _vd.FFT_FORWARD
FFT_INVERSE = _vd.FFT_INVERSE


class _RPlan:
    __slots__ = ("n", "m", "log2n", "setup", "z", "zb", "re", "im",
                 "half", "inv_n")

    def __init__(self, n: int):
        self.n = n
        self.m = n >> 1
        self.log2n = n.bit_length() - 1
        self.setup = _vd._setup(self.log2n)
        self.re = np.empty(self.m, dtype=np.float32)
        self.im = np.empty(self.m, dtype=np.float32)
        self.z = _vd._split(self.re, self.im)
        self.zb = ctypes.byref(self.z)
        self.half = np.float32(0.5)
        self.inv_n = np.float32(1.0 / n)


_plans: dict[int, _RPlan] = {}


def _plan(n: int) -> _RPlan:
    p = _plans.get(n)
    if p is None:
        p = _RPlan(n)
        _plans[n] = p
    return p


def fwd_r(src: np.ndarray, out_cart: np.ndarray, n: int) -> None:
    """Real [n] src -> engine cart [n+2] (contiguous f32, same layout).

    Bit-identical to vdsp.fft_fwd_r(src, out_cart).
    """
    p = _plan(n)
    _lib.vDSP_ctoz(_vd._cast_dsp(src), 2, p.zb, 1, p.m)
    _lib.vDSP_fft_zrip(p.setup, p.zb, 1, p.log2n, FFT_FORWARD)
    _lib.vDSP_ztoc(p.zb, 1, _vd._cast_dsp(out_cart), 2, p.m)
    im = p.im
    out_cart[1] = 0.0
    out_cart[n] = im[0]
    out_cart[n + 1] = 0.0
    out_cart *= p.half              # zrip fwd = 2x DFT
    out_cart[1] = 0.0
    out_cart[n + 1] = 0.0


def inv_r(cart: np.ndarray, out: np.ndarray, n: int) -> None:
    """Engine cart [n+2] -> real [n] * (1/n).

    Bit-identical to vdsp.fft_inv_r(cart, out).
    """
    p = _plan(n)
    _lib.vDSP_ctoz(_vd._cast_dsp(cart), 2, p.zb, 1, p.m)
    # ctoz wrote re[k]=cart[2k], im[k]=cart[2k+1]; fix DC/Nyquist slots:
    p.re[0] = cart[0]
    p.im[0] = cart[n]
    _lib.vDSP_fft_zrip(p.setup, p.zb, 1, p.log2n, FFT_INVERSE)
    _lib.vDSP_ztoc(p.zb, 1, _vd._cast_dsp(out), 2, p.m)
    out *= p.inv_n                  # zrip inv of DFT units = N*signal


class _ZPlan:
    __slots__ = ("log2n", "setup", "z", "zb")

    def __init__(self, re: np.ndarray, im: np.ndarray):
        n = re.size
        self.log2n = n.bit_length() - 1
        self.setup = _vd._setup(self.log2n)
        self.z = _vd._split(re, im)
        self.zb = ctypes.byref(self.z)


def zcall(re: np.ndarray, im: np.ndarray, log2n: int, inverse: bool) -> None:
    """In-place zip FFT without per-call struct/byref builds (when the
    arrays are the SAME objects each call, the _ZPlan cache hits)."""
    key = (id(re), id(im), log2n)
    p = _zplans.get(key)
    if p is None:
        p = _ZPlan(re, im)
        _zplans[key] = p
    _lib.vDSP_fft_zip(p.setup, p.zb, 1, log2n,
                      FFT_INVERSE if inverse else FFT_FORWARD)


_zplans: dict = {}
