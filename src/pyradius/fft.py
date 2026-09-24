"""pyradius.fft — mirror of libradius self-written radix-2 FFT kernels.

Two paths:
  - exact: mirrors rx_fft_cfft_kern/rx_fft_fwd/rx_fft_inv bit-for-bit
    (same twiddle values from double sin/cos cast to f32; same butterfly op
    order; separated mul/add). Vectorized over stages in numpy but elementwise
    op sequence identical per butterfly.
  - fast: scipy pocketfft with f32; same semantics (cart packing, 1/N scale,
    Nyquist slots) but ULP-level differences. Chosen per-call by tolerance.

FFTInv = variant A (hermitian full-spectrum expand -> complex IFFT(N) -> real
* 1/N). FFTFwd = real -> canonical DFT packed as cart [N+2] with dst[1]=0,
dst[N]=Nyquist, dst[N+1]=0.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.fft import rfft, irfft

_PI = 3.14159265358979323846  # RX_PI source constant (f32 suffix in C)
# The compiled lib constant-folds -2.0 * RX_PI into double(2 * float(RX_PI)) =
# 0xc01921fb60000000 — verify against C build once at import; fallback exact.
import struct as _struct
_NEG2PI_F64 = _struct.unpack(">d", (0xC01921FB60000000).to_bytes(8, "big"))[0]


class _Twiddles:
    """Twiddle cache: W[k] = exp(-2pi i k / n), f32 from double sin/cos."""

    def __init__(self):
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, n: int):
        if n in self.cache:
            return self.cache[n]
        # math.cos/sin route through libm double — identical to C (float)cos(ang);
        # np.cos uses SIMD variants that can differ by 1 ULP.
        wr = np.empty(n // 2, dtype=np.float32)
        wi = np.empty(n // 2, dtype=np.float32)
        for k in range(n // 2):
            ang = k * _NEG2PI_F64 / n
            wr[k] = math.cos(ang)
            wi[k] = math.sin(ang)
        self.cache[n] = (wr, wi)
        return wr, wi


_tw = _Twiddles()


def _bit_reverse_idx(n: int) -> np.ndarray:
    idx = np.arange(n)
    j = np.zeros(n, dtype=np.int64)
    bits = n.bit_length() - 1
    for b in range(bits):
        j |= ((idx >> b) & 1) << (bits - 1 - b)
    return j


class FFTPlan:
    """Precomputed plan for exact kernel at size n (power of two)."""

    def __init__(self, n: int):
        self.n = n
        self.wr, self.wi = _tw.get(n)
        self.rev = _bit_reverse_idx(n)
        # Per-stage twiddle slices, precomputed once (the C kernel re-indexes
        # wr[t]/wi[t] with t = j*step every stage; identical values).
        self._stages = []
        ln = 2
        while ln <= n:
            half = ln >> 1
            step = n // ln
            self._stages.append((ln, half, self.wr[0:half * step:step].copy(),
                                 self.wi[0:half * step:step].copy()))
            ln <<= 1
        # Scratch for the butterfly temporaries, reused across stages/calls.
        # float32 on purpose: the C kernel keeps every intermediate in float and
        # -ffp-contract=off forbids fusing, so each product/sum rounds on its own.
        self._buf = {}
        # Half-spectrum gather index for inv() — constant per plan.
        m = n >> 1
        self._k = np.arange(1, m, dtype=np.int64)
        self._rev64 = np.asarray(self.rev, dtype=np.int64)
        # bit-reversal scratch (out-of-place gather permutation)
        self._bre = np.empty(n, dtype=np.float32)
        self._bim = np.empty(n, dtype=np.float32)
        self._has_im = True

    def _tmp(self, key, shape):
        b = self._buf.get(key)
        if b is None or b.shape != shape:
            b = np.empty(shape, dtype=np.float32)
            self._buf[key] = b
        return b

    def cfft(self, re: np.ndarray, im: np.ndarray, inverse: bool = False) -> None:
        """In-place radix-2 DIT, unscaled; mirrors rx_fft_cfft_kern bit for bit.

        Dispatches to the numba kernel when it is importable AND was verified
        bit-identical to the numpy kernel at this plan size (see _numba_verified);
        otherwise runs the numpy kernel below.
        """
        if _numba_verified(self):
            self._kern(re, im, inverse)
            return
        self._cfft_numpy(re, im, inverse)

    def _kern(self, re: np.ndarray, im: np.ndarray, inverse: bool) -> None:
        _numba_kern()(re, im, self._rev64, self.wr, self.wi, inverse)

    def _cfft_numpy(self, re: np.ndarray, im: np.ndarray,
                    inverse: bool = False) -> None:
        """Reference numpy kernel, scalar-op-exact against the C kernel.

        The C kernel is scalar (float c = wr[t]; tr = xr*c - xi*s; +,-), so every
        arithmetic op rounds to float32 on its own.  numpy float32 arrays give
        the same per-op rounding as long as the temporaries also stay float32
        (an out= buffer of a wider dtype would silently promote the operation),
        which is why _tmp allocates float32.
        """
        n = self.n
        rev = self.rev
        # bit-reversal is a permutation, so out-of-place into scratch buffers is
        # equivalent to the C kernel's in-place swap loop (which only swaps i<j;
        # an aliased gather would read already-permuted slots).
        np.take(re, rev, out=self._bre)
        re[:] = self._bre
        if self._has_im:
            np.take(im, rev, out=self._bim)
            im[:] = self._bim
        for ln, half, c0, s0 in self._stages:
            s = -s0 if inverse else s0
            R = re.reshape(-1, ln)
            I = im.reshape(-1, ln)
            vl = R[:, :half]
            vr = R[:, half:]
            ul = I[:, :half]
            ur = I[:, half:]
            shape = vl.shape
            t0 = self._tmp((ln, 0), shape)
            t1 = self._tmp((ln, 1), shape)
            t2 = self._tmp((ln, 2), shape)
            t3 = self._tmp((ln, 3), shape)
            t4 = self._tmp((ln, 4), shape)
            np.multiply(vr, c0, out=t0)
            np.multiply(ur, s, out=t1)
            t0 -= t1                       # tr = xr*c - xi*s
            np.multiply(vr, s, out=t1)
            np.multiply(ur, c0, out=t2)
            t1 += t2                       # ti = xr*s + xi*c
            np.add(vl, t0, out=t2)
            np.subtract(vl, t0, out=t3)
            np.add(ul, t1, out=t0)
            np.subtract(ul, t1, out=t4)
            vl[:] = t2
            vr[:] = t3
            ul[:] = t0
            ur[:] = t4

    def fwd(self, src: np.ndarray) -> np.ndarray:
        """rx_fft_fwd: real N -> cart [N+2]."""
        n = self.n
        re = np.array(src, dtype=np.float32, copy=True)
        im = np.zeros(n, dtype=np.float32)
        self.cfft(re, im, inverse=False)
        m = n >> 1
        dst = np.zeros(n + 2, dtype=np.float32)
        dst[0] = re[0]
        dst[1] = 0.0
        dst[2:n:2] = re[1:m]
        dst[3:n:2] = im[1:m]
        dst[n] = re[m]
        dst[n + 1] = 0.0
        return dst

    def inv(self, cart: np.ndarray) -> np.ndarray:
        """rx_fft_inv variant A: cart [N+2] -> real [N], x1/N scale.

        Expands the packed cart spectrum into the hermitian full spectrum
        (re/m only touch the odd slots for h and the conjugate mirror for n-h,
        so the even slots are written by the pair loop) and runs the same
        kernel, then scales by 1/N exactly like the C wrapper.
        """
        n = self.n
        m = n >> 1
        k = self._k
        h = n - k
        re = np.zeros(n, dtype=np.float32)
        im = np.zeros(n, dtype=np.float32)
        ck = cart[2 * k]
        dk = cart[2 * k + 1]
        re[k] = ck
        im[k] = dk
        re[h] = ck
        im[h] = -dk          # k and h are disjoint, so no aliasing to worry about
        re[0] = cart[0]
        re[m] = cart[n]
        self.cfft(re, im, inverse=True)
        s = np.float32(1.0 / n)
        return (re * s).astype(np.float32)


# Optional compiled kernel.  The numpy path below is the reference; numba is
# used only after a one-time bit-exact self-check at the plan's own size, so a
# numba/libm/fastmath regression can never silently change results — it just
# falls back to numpy.
_NUMBA = None
_NUMBA_OK: dict[int, bool] = {}
_NUMBA_CHECKED: set[int] = set()


def _numba_kern():
    global _NUMBA
    if _NUMBA is None:
        try:
            from . import _fft_numba
            _NUMBA = _fft_numba.cfft_kern if _fft_numba._HAVE_NUMBA else False
        except Exception:
            _NUMBA = False
    return _NUMBA


def _numba_verified(pl: "FFTPlan") -> bool:
    """Run the compiled kernel once against the numpy kernel at this size."""
    kern = _numba_kern()
    if not kern:
        return False
    n = pl.n
    ok = _NUMBA_OK.get(n)
    if ok is not None:
        return ok
    rng = np.random.default_rng(0x5AD10)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    ok = True
    for inverse in (False, True):
        want_r = a.copy()
        want_i = b.copy()
        pl._cfft_numpy(want_r, want_i, inverse)
        got_r = a.copy()
        got_i = b.copy()
        try:
            pl._kern(got_r, got_i, inverse)
        except Exception:
            ok = False
            break
        if not (bool((want_r == got_r).all()) and bool((want_i == got_i).all())):
            ok = False
            break
    _NUMBA_OK[n] = ok
    _NUMBA_CHECKED.add(n)
    return ok


_plans: dict[int, FFTPlan] = {}


def plan(n: int) -> FFTPlan:
    p = _plans.get(n)
    if p is None:
        p = FFTPlan(n)
        _plans[n] = p
    return p


# ---- fast path (scipy pocketfft), same cart packing / scaling semantics ----

_VDSP = None
_VDSP_TRIED = False


def _vdsp():
    global _VDSP, _VDSP_TRIED
    if not _VDSP_TRIED:
        _VDSP_TRIED = True
        try:
            from . import vdsp
            _VDSP = vdsp if vdsp.HAVE_VDSP else None
        except Exception:
            _VDSP = None
    return _VDSP


def fwd_fast(src: np.ndarray) -> np.ndarray:
    n = len(src)
    m = n >> 1
    z = rfft(src.astype(np.float32), n=n).astype(np.complex64)
    dst = np.zeros(n + 2, dtype=np.float32)
    dst[0] = z.real[0]
    dst[1] = 0.0
    dst[2:n:2] = z.real[1:m]
    dst[3:n:2] = z.imag[1:m]
    dst[n] = z.real[m]
    dst[n + 1] = 0.0
    return dst


def inv_fast(cart: np.ndarray) -> np.ndarray:
    n = len(cart) - 2
    m = n >> 1
    z = np.zeros(m + 1, dtype=np.complex64)
    z.real[0] = cart[0]
    k = np.arange(1, m)
    z.real[1:m] = cart[2 * k]
    z.imag[1:m] = cart[2 * k + 1]
    z.real[m] = cart[n]
    return irfft(z, n=n).astype(np.float32)
