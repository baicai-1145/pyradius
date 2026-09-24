"""pyradius.vdsp — ctypes bindings for Apple Accelerate vDSP (route B: fast).

FFT (zrip real-packed) and a few vector ops, used by the fast paths.  Values
are ULP-level different from the C engine's self-written radix-2 kernel —
this module exists for the speed route where "error negligible, not
bit-exact" is the acceptance bar.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import numpy as np

_lib = None
_checked = False


def _load():
    global _lib, _checked
    if _checked:
        return _lib
    _checked = True
    try:
        _lib = ctypes.CDLL(ctypes.util.find_library("Accelerate") or
                           "/System/Library/Frameworks/Accelerate.framework/Accelerate")
    except OSError:
        _lib = None
    return _lib


HAVE_VDSP = False
try:
    _load()
    HAVE_VDSP = _lib is not None and hasattr(_lib, "vDSP_create_fftsetup")
except Exception:
    HAVE_VDSP = False

if HAVE_VDSP:
    _FT = ctypes.c_void_p
    _lib.vDSP_create_fftsetup.restype = _FT
    _lib.vDSP_create_fftsetup.argtypes = [ctypes.c_uint, ctypes.c_int]
    _lib.vDSP_destroy_fftsetup.argtypes = [_FT]
    # vDSP_fft_zrip(setup, z*, stride, log2n, direction)
    _lib.vDSP_fft_zrip.argtypes = [_FT, ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.c_uint, ctypes.c_int]
    _lib.vDSP_fft_zip.argtypes = [_FT, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_uint, ctypes.c_int]
    # vDSP_ctoz / ztoc take DSPSplitComplex* (two float*)
    _lib.vDSP_ctoz.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    _lib.vDSP_ztoc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]

FFT_FORWARD = 1
FFT_INVERSE = -1

_setups: dict[int, ctypes.c_void_p] = {}


def _setup(log2n: int):
    h = _setups.get(log2n)
    if h is None:
        h = _lib.vDSP_create_fftsetup(ctypes.c_uint(log2n), 1)  # kFFTRadix2
        if not h:
            raise RuntimeError("vDSP_create_fftsetup failed")
        _setups[log2n] = h
    return h


class _Split(ctypes.Structure):
    _fields_ = [("real", ctypes.POINTER(ctypes.c_float)),
                ("imag", ctypes.POINTER(ctypes.c_float))]


def _cast_dsp(a: np.ndarray) -> ctypes.c_void_p:
    return ctypes.cast(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       ctypes.c_void_p)


def _split(re: np.ndarray, im: np.ndarray) -> _Split:
    return _Split(re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


def fft_fwd_r(src: np.ndarray, out_cart: np.ndarray | None = None) -> np.ndarray:
    """Real N -> engine cart packing [N+2] via vDSP zrip.

    cart[0]=DC, cart[1]=0, cart[2k]=Re(k), cart[2k+1]=Im(k), k=1..N/2-1,
    cart[N]=Nyquist, cart[N+1]=0.  Same layout as rx_fft_fwd / FFTPlan.fwd.
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    n = src.size
    log2n = n.bit_length() - 1
    m = n >> 1
    re = np.empty(m, dtype=np.float32)
    im = np.empty(m, dtype=np.float32)
    z = _split(re, im)
    _lib.vDSP_ctoz(_cast_dsp(src), 2, ctypes.byref(z), 1, m)
    _lib.vDSP_fft_zrip(_setup(log2n), ctypes.byref(z), 1, log2n, FFT_FORWARD)
    if out_cart is None:
        out_cart = np.empty(n + 2, dtype=np.float32)
    # interleave re/im -> cart[0..N-1]; then fix DC/Nyquist slots
    c = out_cart[:n].reshape(m, 2)
    np.multiply(re, 1.0, out=c[:, 0])
    np.multiply(im, 1.0, out=c[:, 1])
    out_cart[1] = 0.0
    nyq = im[0]
    out_cart[n] = nyq
    out_cart[n + 1] = 0.0
    out_cart *= np.float32(0.5)        # zrip fwd = 2x DFT
    out_cart[1] = 0.0
    out_cart[n + 1] = 0.0
    return out_cart


def fft_inv_r(cart: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Engine cart [N+2] -> real [N] * (1/N) via vDSP zrip inverse.

    Reassembles the packed half-spectrum (Nyquist from cart[N] into zrip's
    im[0]) and scales by 1/N: zrip inverse of a DFT-units spectrum returns
    N * signal; the engine's rx_fft_inv also multiplies by 1/N.
    """
    cart = np.ascontiguousarray(cart, dtype=np.float32)
    n = cart.size - 2
    log2n = n.bit_length() - 1
    m = n >> 1
    re = np.empty(m, dtype=np.float32)
    im = np.empty(m, dtype=np.float32)
    c = cart[:n].reshape(m, 2)
    np.multiply(c[:, 0], 1.0, out=re)
    np.multiply(c[:, 1], 1.0, out=im)
    re[0] = cart[0]
    im[0] = cart[n]        # Nyquist
    z = _split(re, im)
    _lib.vDSP_fft_zrip(_setup(log2n), ctypes.byref(z), 1, log2n, FFT_INVERSE)
    if out is None:
        out = np.empty(n, dtype=np.float32)
    _lib.vDSP_ztoc(ctypes.byref(z), 1, _cast_dsp(out), 2, m)
    out *= np.float32(1.0 / n)
    return out


def fft_zip(re: np.ndarray, im: np.ndarray, log2n: int, inverse: bool) -> None:
    """In-place complex FFT on split arrays (vDSP_fft_zip, unscaled)."""
    z = _split(re, im)
    _lib.vDSP_fft_zip(_setup(log2n), ctypes.byref(z), 1, log2n,
                      FFT_INVERSE if inverse else FFT_FORWARD)
