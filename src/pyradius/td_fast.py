"""pyradius.td_fast — the TD_FAST=1 approximate/high-throughput tier.

The default TD path (:mod:`pyradius.td_core`) is bit-exact against libradius and
must stay that way; everything in this module is only reachable when
``TD_FAST=1`` and each piece falls back to the exact implementation when its
backend (Accelerate vDSP / numba) is unavailable, exactly like
:mod:`pyradius.neon` does for route B.

Pieces, ordered by precision risk
---------------------------------
1. :class:`VPlan` — pitch-search FFTs (N=8192, 4 transforms per granule) run
   through Accelerate vDSP instead of the self-written radix-2 numba kernel.
   Measured 109/146us -> 16/16us per transform.  The consequence is ULU-level
   (2e-7 rel) noise in the whitened ACF; the argmax that picks the pitch period
   is unaffected on the corpus (period and win sequences identical, output
   bit-equal — see the task report).
2. :func:`interp_nsamples_fast` — the drain resampler fuse: phases, sub-band
   selection, wrap fix and the dot product in one numba kernel.  Bit-exact with
   :func:`pyradius.sampler.interp_nsamples` (verified at import on random data
   incl. the wrap cases; falls back if the check fails).
3. :func:`do_ola_fast` — the overlap-add write batched over channels and over
   the three segments of ``DoOla`` (one gather + one fancy-index write instead
   of 3 x nch passes).  Same float op order, so bit-exact.
4. :class:`DecimatedPitch` — 2x/4x decimated pitch search with a full-rate
   refinement around the coarse peak (lossy; opt-in via ``TD_FAST_DS``).
"""
from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# 1. vDSP FFT plan (rotating buffers, same call contract as fft.FFTPlan)
# ---------------------------------------------------------------------------

_VDSP = None
_VDSP_TRIED = False
_NBUF = 4                       # pitch holds sa/sb/x concurrently; 4 is ample


def _vdsp_fast():
    global _VDSP, _VDSP_TRIED
    if not _VDSP_TRIED:
        _VDSP_TRIED = True
        try:
            from . import vdsp, vdsp_fast
            _VDSP = vdsp_fast if vdsp.HAVE_VDSP else None
        except Exception:
            _VDSP = None
    return _VDSP


class VPlan:
    """vDSP-backed stand-in for :class:`pyradius.fft.FFTPlan` (fwd/inv only).

    Returns a *rotating* scratch buffer per call because the callers keep two
    spectra live at once (pitch: ``sa``/``sb``) and a single shared buffer would
    clobber them.  Bit-exactness is not claimed: Accelerate rounds differently
    (measured 2.2e-7 relative on the 8192-point inverse), which is what makes
    this a TD_FAST-only component.
    """

    def __init__(self, n: int):
        self.n = int(n)
        self._fb = [np.empty(self.n + 2, dtype=np.float32) for _ in range(_NBUF)]
        self._ib = [np.empty(self.n, dtype=np.float32) for _ in range(_NBUF)]
        self._fi = 0
        self._ii = 0

    def fwd(self, src: np.ndarray) -> np.ndarray:
        b = self._fb[self._fi]
        self._fi = (self._fi + 1) % _NBUF
        _vdsp_fast().fwd_r(np.ascontiguousarray(src, dtype=np.float32), b, self.n)
        return b

    def inv(self, cart: np.ndarray) -> np.ndarray:
        b = self._ib[self._ii]
        self._ii = (self._ii + 1) % _NBUF
        _vdsp_fast().inv_r(np.ascontiguousarray(cart, dtype=np.float32), b, self.n)
        return b


_PLANS: dict[int, object] = {}


def make_plan(n: int):
    """FFT plan (fwd/inv) for n: vDSP when available, else the exact one.

    Cached per size so the rotating scratch pool is shared across calls — the
    only requirement is that no more than ``_NBUF`` results from the same plan
    are live at once (the pitch front-end keeps 2).
    """
    p = _PLANS.get(n)
    if p is not None:
        return p
    vf = _vdsp_fast()
    p = None
    if vf is not None:
        try:
            p = VPlan(n)
        except Exception:
            p = None
    if p is None:
        from .fft import plan as _exact_plan
        p = _exact_plan(n)
    _PLANS[n] = p
    return p


# ---------------------------------------------------------------------------
# 2. fused drain resampler
# ---------------------------------------------------------------------------

_interp_fused = None
_INTERP_OK = False
_INTERP_CHECKED = False


def _build_interp():
    global _interp_fused
    if _interp_fused is not None:
        return _interp_fused
    try:
        from numba import njit

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern(src, dst, pool, taps_arr, offs, nsubs, rs, nch, count, ph0, rate):
            """Phases + band select + dot, one pass.

            Mirrors sampler.interp_nsamples exactly:
              sub = (int)fmaf(frac, nsubs, 0.5)  == f32(frac_f*subs_f + 0.5) trunc
              base = iph - offs[sub]; the C wrap fix is a *single* conditional
              step, and the un-wrapped case (base<0 or base+taps>rs) is handled
              the same way the numpy fallback does it, per sample.
            """
            ph = ph0
            for i in range(count):
                iph = int(ph)
                frac = ph - iph
                sub = int(np.float32(np.float32(frac) * np.float32(nsubs) + np.float32(0.5)))
                t = taps_arr[sub]
                b = iph - offs[sub]
                for ch in range(nch):
                    acc = np.float32(0.0)
                    for jj in range(t):
                        idx = b + jj
                        if idx < 0:
                            idx += rs
                        elif idx >= rs:
                            idx -= rs
                        acc = np.float32(acc + np.float32(src[ch, idx] * pool[sub, jj]))
                    dst[ch, i] = acc
                ph = rate + ph

        _interp_fused = _kern
    except Exception:
        _interp_fused = False
    return _interp_fused


def interp_nsamples_fast(tbl, src, ring_size, phase, dst_off, count, rate, quality):
    """Drop-in replacement for :func:`pyradius.sampler.interp_nsamples`.

    The ``dst_off`` argument is unused by the exact implementation too (the C
    appends at 0 and the caller offsets by slice).  ``src`` must be the
    ``[nch, ring]`` view: both call sites pass ``out_ring[:, :out_len]``, which
    is a strided view, so the kernel gets a contiguous copy only when needed.
    """
    from . import sampler as _s
    k = _s.interp_kidx(quality, tbl.n_levels)
    ns = np.float32(float(tbl.nsubs[k]))
    nch = src.shape[0]
    dst = np.empty((nch, count), dtype=np.float32)
    kern = _build_interp()
    if kern is False or not _interp_ok():
        return _s.interp_nsamples(tbl, src, ring_size, phase, dst_off, count, rate, quality)
    kern(np.ascontiguousarray(src, dtype=np.float32), dst,
         np.ascontiguousarray(tbl.pools[k]), tbl.taps[k], tbl.offsets[k],
         ns, int(ring_size), int(nch), int(count), float(phase), float(rate))
    return dst


def _interp_ok() -> bool:
    """One-time bit-exactness check of the fused resampler vs sampler.

    Covers both the direct path and the ring-wrap path (phases placed so the
    window straddles the ring end, which forces sampler's per-sample wrap fix),
    across rates above/below 1 and both corpus ring sizes.  Geometry is kept
    inside the C contract (``base + taps <= ring`` is *not* required by the C
    code, but a ring shorter than the resampled span is out of contract in
    sampler's own fallback, so it is not exercised here).  Requires uint32
    equality against :func:`pyradius.sampler.interp_nsamples`.
    """
    global _INTERP_OK, _INTERP_CHECKED
    if _INTERP_CHECKED:
        return _INTERP_OK
    _INTERP_CHECKED = True
    try:
        from . import sampler as _s
        kern = _build_interp()
        if kern is False:
            return False
        tbl = _s.InterpTable(1024, 6, 12.0)
        rng = np.random.default_rng(0x7DFA57)
        for quality, rate, ring, count in ((1.0, 1.189207115002721, 8192, 4096),
                                           (1.0, 0.8408964152537145, 8192, 4096),
                                           (1.0, 1.0, 4096, 2048),
                                           (1.0, 0.5, 4096, 4096),
                                           (1.0, 1.189207115002721, 1 << 20, 8192)):
            k = _s.interp_kidx(quality, tbl.n_levels)
            ns = np.float32(float(tbl.nsubs[k]))
            src = (rng.standard_normal((2, ring)) * 0.3).astype(np.float32)
            # ph=ring-taps straddles the ring end; ph=0 is the plain path
            for ph_i in (0, max(ring - int(tbl.taps[k].max()) - 2, 0),
                         int(rng.integers(0, max(ring - count - 8, 1)))):
                ph = float(ph_i)
                ref = _s.interp_nsamples(tbl, src, ring, ph, 0, count, rate, quality)
                got = np.empty((2, count), dtype=np.float32)
                kern(np.ascontiguousarray(src), got,
                     np.ascontiguousarray(tbl.pools[k]), tbl.taps[k],
                     tbl.offsets[k], ns, int(ring), 2, count, ph, rate)
                if not np.array_equal(ref.view(np.uint32), got.view(np.uint32)):
                    return False
        _INTERP_OK = True
    except Exception:
        _INTERP_OK = False
    return _INTERP_OK


# ---------------------------------------------------------------------------
# 2b. TransientsInfo post-FFT kernels (bit-exact, replaces the python loops)
# ---------------------------------------------------------------------------

_ti_kern = None
_TI_OK = False
_TI_CHECKED = False


def _build_ti():
    global _ti_kern
    if _ti_kern is not None:
        return _ti_kern
    try:
        from numba import njit

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern_big(cart, scale, pw, r9, band_raw, bounds):
            """TIState._big post-FFT: pw/scale, r9 = |F|^(1/4), 24-band + total sum.

            Every op is float32 with the same association the numpy port uses
            (separate mul/add, sequential f32 accumulation per band).
            """
            nh = pw.shape[0]
            for k in range(nh):
                x = np.float32(cart[2 * k] * cart[2 * k])
                x = np.float32(x + np.float32(cart[2 * k + 1] * cart[2 * k + 1]))
                pw[k] = np.float32(x * scale)
            for k in range(nh):
                v = pw[k]
                r9[k] = np.float32(np.sqrt(np.float32(np.sqrt(np.float32(np.sqrt(v))))))
            for b in range(bounds.shape[0] - 1):
                s = np.float32(0.0)
                for k in range(bounds[b], bounds[b + 1]):
                    s = np.float32(s + pw[k])
                band_raw[b] = np.float32(np.sqrt(np.float32(np.sqrt(np.float32(np.sqrt(s))))))
            acc = np.float32(0.0)
            for k in range(nh):
                acc = np.float32(acc + pw[k])
            if acc > np.float32(0.0):
                return np.float32(np.sqrt(np.float32(np.sqrt(acc))))
            return np.float32(0.0)

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern_sub(cart, band65, scale):
            """TIState._sub post-FFT: band65 = scale * pw^(1/8) where pw > 0."""
            n = band65.shape[0]
            for b in range(n):
                x = np.float32(cart[2 * b] * cart[2 * b])
                x = np.float32(x + np.float32(cart[2 * b + 1] * cart[2 * b + 1]))
                if x > np.float32(0.0):
                    s = np.float32(np.sqrt(np.float32(np.sqrt(np.float32(np.sqrt(x))))))
                    band65[b] = np.float32(scale * s)
                else:
                    band65[b] = np.float32(0.0)

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern_synth(r24, pool, r9, w228, mag, c, purge, nh, n114, nb9):
            """TIState._synth: band/shape/pool/vsum reductions + w228/mag stores.

            The 8-slot left-to-right association and the ascending f32
            accumulation order are the numpy port's (and the C's), so the
            reducs are bit-equal.  Returns (pool, s0, wi, mi).
            """
            wb = c - 4
            i_m4 = ((wb - 4) if (wb - 4) >= 0 else 0) % nb9
            i_m3 = ((wb - 3) if (wb - 3) >= 0 else 0) % nb9
            i_m2 = ((wb - 2) if (wb - 2) >= 0 else 0) % nb9
            i_m1 = ((wb - 1) if (wb - 1) >= 0 else 0) % nb9
            i_p1 = ((wb + 1) if (wb + 1) >= 0 else 0) % nb9
            i_p2 = ((wb + 2) if (wb + 2) >= 0 else 0) % nb9
            i_p3 = ((wb + 3) if (wb + 3) >= 0 else 0) % nb9
            i_p4 = ((wb + 4) if (wb + 4) >= 0 else 0) % nb9
            # shape over 24 bands.  The accumulator is *seeded with the first
            # term* (not 0.0f): numpy's np.add.accumulate has no identity, so
            # this is what makes the +/-0 results match bit-for-bit.  For
            # non-zero data the two are identical (0.0f + a == a exactly).
            shape_acc = np.float32(0.0)
            for b in range(24):
                v = np.float32(r24[i_p1, b] + r24[i_p2, b])
                v = np.float32(v + r24[i_p3, b])
                v = np.float32(v + r24[i_p4, b])
                v = np.float32(v - r24[i_m1, b])
                v = np.float32(v - r24[i_m2, b])
                v = np.float32(v - r24[i_m3, b])
                v = np.float32(v - r24[i_m4, b])
                if v <= np.float32(0.0):
                    v = np.float32(np.float32(-0.1) * v)
                shape_acc = v if b == 0 else np.float32(shape_acc + v)
            shape = np.float32(shape_acc / np.float32(24.0))
            # pool over the 65 subbands (two-slot difference, rectify, mean)
            pool_acc = np.float32(0.0)
            for b in range(65):
                m = np.float32(pool[i_p1, b] - pool[i_m1, b])
                if m <= np.float32(0.0):
                    m = np.float32(np.float32(-0.1) * m)
                pool_acc = m if b == 0 else np.float32(pool_acc + m)
            poolv = np.float32(pool_acc / np.float32(n114))
            # vsum over the 257 big-window bins
            vsum_acc = np.float32(0.0)
            for k in range(nh):
                v = np.float32(r9[i_p1, k] + r9[i_p2, k])
                v = np.float32(v + r9[i_p3, k])
                v = np.float32(v + r9[i_p4, k])
                v = np.float32(v - r9[i_m1, k])
                v = np.float32(v - r9[i_m2, k])
                v = np.float32(v - r9[i_m3, k])
                v = np.float32(v - r9[i_m4, k])
                if v <= np.float32(0.0):
                    v = np.float32(np.float32(-0.1) * v)
                vsum_acc = v if k == 0 else np.float32(vsum_acc + v)
            vsum = np.float32(vsum_acc / np.float32(nh))
            s0 = np.float32(np.float32(2.0)
                            * (np.float32(vsum + shape)
                               + np.float32(np.float32(0.5) * poolv)))
            wi = (c - 4) - purge
            mi = wi
            if wi >= 0:
                w228[wi] = poolv
            if mi >= 0:
                mag[mi] = s0
            return poolv, s0, wi, mi

        _ti_kern = (_kern_big, _kern_sub, _kern_synth)
    except Exception:
        _ti_kern = False
    return _ti_kern


def _ti_ok() -> bool:
    """One-time bit-exactness check of the TI post-FFT kernels vs numpy."""
    global _TI_OK, _TI_CHECKED
    if _TI_CHECKED:
        return _TI_OK
    _TI_CHECKED = True
    try:
        from .td_core import _TI_BANDS, _TI_WIN_SCALE2, TIState
        k = _build_ti()
        if k is False:
            return False
        kbig, ksub, ksynth = k
        rng = np.random.default_rng(0x71CFA5)
        bounds = np.asarray(_TI_BANDS, dtype=np.int64)
        for trial in range(8):
            st = TIState(2, 44100, 1.0)
            cartB = (rng.standard_normal(st.N + 2) * (10.0 ** (trial - 3))).astype(np.float32)
            pw = np.empty(st.Nh, dtype=np.float32)
            r9 = np.empty(st.Nh, dtype=np.float32)
            br = np.empty(24, dtype=np.float32)
            w = kbig(cartB, np.float32(_TI_WIN_SCALE2), pw, r9, br, bounds)
            re = cartB[0:2 * st.Nh:2]
            im = cartB[1:2 * st.Nh:2]
            x = re * re
            x = x + im * im
            x = x * np.float32(_TI_WIN_SCALE2)
            if not np.array_equal(x.view(np.uint32), pw.view(np.uint32)):
                return False
            ref_r9 = np.sqrt(np.sqrt(np.sqrt(x)))
            if not np.array_equal(ref_r9.view(np.uint32), r9.view(np.uint32)):
                return False
            widths = bounds[1:25] - bounds[0:24]
            bmat = np.zeros((24, int(widths.max())), dtype=np.float32)
            for b in range(24):
                lo, hi = int(bounds[b]), int(bounds[b + 1])
                bmat[b, :hi - lo] = x[lo:hi]
            s = np.add.accumulate(bmat, axis=1, dtype=np.float32)[:, -1]
            ref_br = np.sqrt(np.sqrt(np.sqrt(s)))
            if not np.array_equal(ref_br.view(np.uint32), br.view(np.uint32)):
                return False
            accf = np.add.accumulate(x, dtype=np.float32)[-1]
            ref_w = np.float32(np.sqrt(np.sqrt(accf))) if accf > 0 else np.float32(0.0)
            if np.float32(w).view(np.uint32) != np.float32(ref_w).view(np.uint32):
                return False
            cartS = (rng.standard_normal(st.Nm + 2) * (10.0 ** (trial - 3))).astype(np.float32)
            got = np.empty(65, dtype=np.float32)
            ksub(cartS, got, np.float32(1.16609546))
            r2 = cartS[0:130:2]
            i2 = cartS[1:130:2]
            p2 = r2 * r2
            p2 = p2 + i2 * i2
            ref2 = np.where(p2 > 0, np.float32(1.16609546)
                            * np.sqrt(np.sqrt(np.sqrt(p2))), np.float32(0.0)).astype(np.float32)
            if not np.array_equal(ref2.view(np.uint32), got.view(np.uint32)):
                return False

            # synth kernel: drive ring r24/pool/r9 with random data over 3 steps
            st2 = TIState(2, 44100, 1.0)
            for _ in range(8):
                st2.ring_r9[:] = (rng.standard_normal(st2.ring_r9.shape)
                                  * (10.0 ** (trial - 4))).astype(np.float32)
                st2.ring_r24[:] = (rng.standard_normal(st2.ring_r24.shape)
                                   * (10.0 ** (trial - 4))).astype(np.float32)
                st2.ring_pool[:] = (rng.standard_normal(st2.ring_pool.shape)
                                    * (10.0 ** (trial - 4))).astype(np.float32)
                c = int(rng.integers(0, 40))
                purge = 8
                w228_ref = st2.w228.copy()
                mag_ref = st2.mag.copy()
                pool_ref, s0_ref, wi_ref, mi_ref = _synth_numpy_ref(
                    st2, c, purge, w228_ref, mag_ref)
                w228_got = st2.w228.copy()
                mag_got = st2.mag.copy()
                poolv, s0, wi, mi = ksynth(st2.ring_r24, st2.ring_pool, st2.ring_r9,
                                           w228_got, mag_got, c, purge, st2.Nh,
                                           st2.n114, st2.NBAND9)
                if int(wi) != int(wi_ref) or int(mi) != int(mi_ref):
                    return False
                for a, b in ((np.float32(poolv), np.float32(pool_ref)),
                             (np.float32(s0), np.float32(s0_ref))):
                    if np.float32(a).view(np.uint32) != np.float32(b).view(np.uint32):
                        return False
                if not np.array_equal(w228_ref.view(np.uint32), w228_got.view(np.uint32)):
                    return False
                if not np.array_equal(mag_ref.view(np.uint32), mag_got.view(np.uint32)):
                    return False
        _TI_OK = True
    except Exception:
        _TI_OK = False
    return _TI_OK


def _synth_numpy_ref(st, c, purge, w228, mag):
    """The numpy body of TIState._synth (reference for the kernel check)."""
    nb9 = st.NBAND9
    wb = c - 4
    ix = [(wb + k) if (wb + k) >= 0 else 0 for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    i_m4, i_m3, i_m2, i_m1, i_p1, i_p2, i_p3, i_p4 = [i % nb9 for i in ix]
    v = ((st.ring_r24[i_p1] + st.ring_r24[i_p2]) + st.ring_r24[i_p3]) + st.ring_r24[i_p4]
    v = (v - st.ring_r24[i_m1] - st.ring_r24[i_m2]) - st.ring_r24[i_m3] - st.ring_r24[i_m4]
    v = np.where(v <= 0.0, np.float32(-0.1) * v, v)
    shape = np.float32(np.add.accumulate(v, dtype=np.float32)[-1] / np.float32(24.0))
    m = st.ring_pool[i_p1] - st.ring_pool[i_m1]
    m = np.where(m <= 0.0, np.float32(-0.1) * m, m)
    pool = np.float32(np.add.accumulate(m, dtype=np.float32)[-1] / np.float32(st.n114))
    wi = (c - 4) - purge
    if wi >= 0:
        w228[wi] = pool
    v2 = ((st.ring_r9[i_p1] + st.ring_r9[i_p2]) + st.ring_r9[i_p3]) + st.ring_r9[i_p4]
    v2 = (v2 - st.ring_r9[i_m1] - st.ring_r9[i_m2]) - st.ring_r9[i_m3] - st.ring_r9[i_m4]
    v2 = np.where(v2 <= 0.0, np.float32(-0.1) * v2, v2)
    vsum = np.float32(np.add.accumulate(v2, dtype=np.float32)[-1] / np.float32(st.Nh))
    s0 = np.float32(2.0) * (vsum + shape + np.float32(0.5) * pool)
    mi = (c - 4) - purge
    if mi >= 0:
        mag[mi] = s0
    return pool, s0, wi, mi


def ti_ok() -> bool:
    return _ti_ok()


def ti_big_post(cart, scale, pw, r9, band_raw, bounds):
    """TIState._big post-FFT (cart spectrum in, f32 tables out)."""
    return _build_ti()[0](cart, np.float32(scale), pw, r9, band_raw, bounds)


def ti_sub_post(cart, band65, scale):
    return _build_ti()[1](cart, band65, np.float32(scale))


def ti_synth_post(r24, pool, r9, w228, mag, c, purge, nh, n114, nb9):
    """TIState._synth reductions; returns (pool, s0, wi, mi) and stores w228/mag."""
    return _build_ti()[2](r24, pool, r9, w228, mag, int(c), int(purge),
                          int(nh), int(n114), int(nb9))


# ---------------------------------------------------------------------------
# 2c. pitch-search front-end (bit-exact kernels + a fast-path front-end)
# ---------------------------------------------------------------------------

_pitch_kern = None
_PITCH_OK = False
_PITCH_CHECKED = False
#: last build/verify failure reason (diagnostics only, never raised).
LAST_ERROR = ""


def _build_pitch():
    global _pitch_kern
    if _pitch_kern is not None:
        return _pitch_kern
    try:
        from numba import njit

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern_gather(ring, out, off, l1, nch, mod):
            """Analysis window gather: out[i] = sum_ch ring[ch][mod(in_pos+off[i])].

            `off` is the precomputed (absolute) index vector and `mod` the ring
            size.  The `vis` mask of the numpy version (samples past
            `fed_visible` read as zero) is folded in by the caller, which only
            takes this path when the whole window is visible.
            """
            rcap = ring.shape[1]
            for i in range(l1):
                j = off[i] % mod
                v = ring[0, j]
                for c in range(1, nch):
                    v = np.float32(v + ring[c, j])
                out[i] = v

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern_acf_win(acf, nb, glitch):
            """acf *= win_lin, then the -1000 leading-run suppression."""
            for i in range(nb):
                acf[i] = np.float32(acf[i] * glitch[i])
            i = 0
            while i + 1 < nb and acf[i] > acf[i + 1]:
                acf[i] = np.float32(-1000.0)
                i += 1

        @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
        def _kern_clarity(acf, nb, p, lim_l, lim_r):
            """The clarity accumulation (comp) for a chosen peak p.

            Returns comp (the max of the two candidate regions); `clarity`
            itself is then computed in python to keep the powf call identical.
            """
            comp = 0.0
            if p - 4 >= 1:
                g = p - 4
                while g > lim_l and acf[g] < acf[g - 1]:
                    g -= 1
                if g >= 1 and float(acf[g]) > comp:
                    comp = float(acf[g])
            if p + 4 < nb - 1:
                g = p + 4
                while g < lim_r and g < nb - 2 and acf[g] < acf[g + 1]:
                    g += 1
                if float(acf[g]) > comp:
                    comp = float(acf[g])
            return comp

        # NOTE: a numba energy kernel was tried and DROPPED: numpy's np.sum on
        # float64 is pairwise, so a sequential f32->f64 accumulation cannot
        # match it (measured 1.7e-12 absolute), and a numba `mag ** f32`
        # whitening kernel was DROPPED for the same reason (it did not even
        # reproduce np.power: produced NaN/28x errors).  np.sum/np.power stay.
        _pitch_kern = (_kern_gather, _kern_acf_win, _kern_clarity)
    except Exception as exc:  # pragma: no cover - environment dependent
        global LAST_ERROR
        LAST_ERROR = f"pitch build: {type(exc).__name__}: {exc}"
        _pitch_kern = False
    return _pitch_kern


def _pitch_ok() -> bool:
    """One-time bit-exactness check of the pitch front-end kernels."""
    global _PITCH_OK, _PITCH_CHECKED
    if _PITCH_CHECKED:
        return _PITCH_OK
    _PITCH_CHECKED = True
    try:
        k = _build_pitch()
        if k is False:
            return False
        kg, ka, kc = k
        rng = np.random.default_rng(0x91CC4)
        same = lambda a, b: np.array_equal(np.asarray(a).view(np.uint32),
                                           np.asarray(b).view(np.uint32))
        for _ in range(4):
            mod, l1, nch = 8192, 3060, 2
            ring = (rng.standard_normal((nch, mod)) * 0.4).astype(np.float32)
            off = rng.integers(-l1, mod, size=l1).astype(np.int64)
            got = np.zeros(l1, dtype=np.float32)
            kg(ring, got, off, l1, nch, mod)
            idx = off % mod
            ref = ring[0][idx].copy()
            ref = ref + ring[1][idx]
            if not same(ref, got):
                return False
            # acf window + leading suppression
            acf = (rng.standard_normal(4097) * 0.1).astype(np.float32)
            glitch = np.abs(rng.standard_normal(4097)).astype(np.float32)
            ref = acf * glitch
            i = 0
            while i + 1 < 4097 and ref[i] > ref[i + 1]:
                ref[i] = np.float32(-1000.0)
                i += 1
            ka(acf, 4097, glitch)
            if not same(ref, acf):
                return False
            # clarity
            acf2 = (rng.standard_normal(4097) * 0.5).astype(np.float32)
            for p in (30, 200, 900):
                lim_l = int(0.6 * float(p))
                lim_r = int(1.8 * float(p) + 0.5)
                comp = 0.0
                if p - 4 >= 1:
                    g = p - 4
                    while g > lim_l and acf2[g] < acf2[g - 1]:
                        g -= 1
                    if g >= 1 and float(acf2[g]) > comp:
                        comp = float(acf2[g])
                if p + 4 < 4096:
                    g = p + 4
                    while g < lim_r and g < 4095 and acf2[g] < acf2[g + 1]:
                        g += 1
                    if float(acf2[g]) > comp:
                        comp = float(acf2[g])
                if float(kc(acf2, 4097, p, lim_l, lim_r)) != comp:
                    return False
        _PITCH_OK = True
    except Exception as exc:  # pragma: no cover - environment dependent
        global LAST_ERROR
        import traceback
        LAST_ERROR = (f"pitch verify: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc())
        _PITCH_OK = False
    return _PITCH_OK


def pitch_ok() -> bool:
    return _pitch_ok()


def pitch_gather(ring, out, off, l1, nch, mod):
    return _build_pitch()[0](ring, out, off, int(l1), int(nch), int(mod))


def pitch_acf_win(acf, nb, glitch):
    return _build_pitch()[1](acf, int(nb), glitch)


def pitch_clarity(acf, nb, p, lim_l, lim_r):
    return float(_build_pitch()[2](acf, int(nb), int(p), int(lim_l), int(lim_r)))


# ---------------------------------------------------------------------------
# 3. batched overlap-add (bit-exact, pure scheduling)
# ---------------------------------------------------------------------------

def do_ola_fast(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain):
    """Overlap-add for the TD_FAST tier.

    Prefers the NEON ``td_ola`` kernel (pyradius.neon, certified bit-exact
    against :func:`td_core._do_ola_exact` over a wide parameter grid) and falls
    back to the numpy 2-D batched version below when the C extension is not
    available.  Both take ``ring`` as ``[nch, ring_tot]`` and ``out_ring`` as
    ``[nch, out_len]``.
    """
    from . import neon as _neon
    if _neon.td_ola is not None and _neon.td_ola_ok():
        _neon.td_ola_run(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain)
        return
    do_ola_numpy(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain)


def do_ola_numpy(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain):
    """``_do_ola_exact`` with the nch loop folded into 2-D fancy indexing.

    Identical arithmetic: same gather, same mask, same per-element op order
    (``d*s1 + (gainf*sv)*s0``).
    """
    nch = st.nch
    if nch <= 0 or a9 == 0:
        return
    rcap = st.ring_tot
    rstart = st.ring_start
    outlen = st.out_len
    gainf = np.float32(gain)
    fvis = st.fed_visible
    if (wr | a4) == 0:
        gi = _wrap_ring_idx(a9, rstart, rcap)
        sv = np.where(gi < fvis, ring[:, gi], np.float32(0.0))
        out_ring[:, 0:a9] = gainf * sv
    env, half, w10 = st.pick_env(a8, a9, a10, boolean)
    if half == 0 or env is None:
        return
    rbase = a4 + w10
    wbase = wr + w10
    w11 = a10 if boolean else (a10 if a10 > a9 else a9)
    # one gather for both segments (segment 2 is empty when w11 <= half)
    n_all = w11 if w11 > half else half
    i_all = np.arange(n_all, dtype=np.int64)
    gi = _wrap_ring_idx(n_all, rstart, rcap, rbase)
    sv = np.where(gi < fvis, ring[:, gi], np.float32(0.0))
    wi = (wbase + i_all) % outlen
    # segment 1: crossfade accumulate
    s0 = env[0:half]
    s1 = env[half:2 * half]
    wiA = wi[:half]
    out_ring[:, wiA] = out_ring[:, wiA] * s1 + gainf * sv[:, :half] * s0
    # segment 2: plain copy tail
    if w11 > half:
        out_ring[:, wi[half:]] = gainf * sv[:, half:]


def _wrap_ring_idx(n, rstart, rtot, base=0):
    """``_wrap_ring`` on arange(n)+base (avoids the where/asarray pair)."""
    i = np.arange(n, dtype=np.int64) + base
    span = (rtot - rstart) if (rtot > rstart) else 1
    ok = (i >= 0) & (i < rtot)
    if ok.all():
        return i
    return np.where(ok, i, rstart + ((i - rtot) % span))


# ---------------------------------------------------------------------------
# 4. decimated pitch search (lossy, TD_FAST_DS>1)
# ---------------------------------------------------------------------------

class DecimatedPitch:
    """Front-end for a D-times decimated ``analyze_pitch`` (TD_FAST_DS > 1).

    MEASURED RESULT — DO NOT ENABLE THIS EXPECTING 0.99: it does not reach the
    TD_FAST acceptance bar and is left in as an opt-in experiment only.  On
    audio_test/4.wav the render correlates 0.001-0.08 against the exact TD
    output (D=2/3/4), i.e. decorrelated.

    Root cause (diagnosed with .tmp/diag_ds3.py): the coarse ACF peak is not
    the exact peak shifted by <=D samples, it is a *different* peak (e.g.
    839.8 vs 966.6).  Decimation is not a similarity transform of this
    front-end: ``analyze_pitch`` whitens by |X|^-0.95 with Hz-derived limits
    (maxbin / taper_len / v387 / v386) and a noise floor scaled by
    ``sr/44100``, so at 1/D the sample rate the whitened spectrum — and hence
    which ACF peak wins — changes; the analysis window is not a sufficient
    anti-alias filter for that.  A correct lossy pitch search would have to
    downsample the *whitening decisions* or search the full-rate ACF with a
    cheaper method (e.g. FFT-based ACF at reduced precision), not the input.

    Only the *front-end* (analysis window, windowed FFT power ACF, whitening,
    final ACF) runs at the decimated rate; :func:`refine_lag` then re-searches
    the neighbourhood at full rate, which fixes the D-quantisation of the
    fractional lag but cannot recover a wrong coarse peak.
    """

    DS_REFINE = 3

    def __init__(self, st, d: int):
        d = int(d)
        N = st.pitch_N // d
        L1 = st.pitch_L1 // d
        while N < 2 * L1:
            N <<= 1
        self.d = d
        self.N = N
        self.L1 = L1
        self.half = (L1 * d) >> 1
        self.M = N >> 1
        self.nb = (N >> 1) | 1
        self.plan = make_plan(N)
        self.maxbin = max(1, st.pitch_maxbin // d)
        self.taper_len = max(1, st.pitch_taper_len // d)
        self.lo = max(1, st.pitch_lo // d)
        self.hi = max(1, st.pitch_hi // d)
        PI = 3.14159265358979323846
        ia = np.arange(L1, dtype=np.float64)
        self.win_an = np.array(0.5 - 0.5 * np.cos(2.0 * PI * ia / float(L1)),
                               dtype=np.float32)
        R = L1 >> 3
        acf_w = np.zeros(N, dtype=np.float32)
        ir = np.arange(R, dtype=np.float64)
        acf_w[:R] = 0.5 + 0.5 * np.cos(PI * ir / float(R))
        for k in range(1, R):
            acf_w[N - k] = acf_w[k]
        self.win_acf = acf_w
        il = np.arange(N, dtype=np.float64)
        self.win_lin = np.array(np.maximum(1.0 - il / float(N // 2 + 1), 0.0),
                                dtype=np.float32)
        it = np.arange(self.taper_len, dtype=np.float64)
        self.win_taper = np.array(0.5 + 0.5 * np.cos(PI * it / float(self.taper_len)),
                                  dtype=np.float32)
        self._x = np.zeros(N, dtype=np.float32)
        self._out = np.zeros(self.M, dtype=np.float32)
        self._x2 = np.zeros(N + 2, dtype=np.float32)
        self.gather_idx = np.arange(0, L1 * d, d, dtype=np.int64) - self.half
        # full-rate refinement state (the engine's own L1 / half, at rate 1)
        self.fL1 = st.pitch_L1
        self.fhalf = st.pitch_L1 >> 1
        self.foff = np.arange(self.fL1, dtype=np.int64) - self.fhalf
        self._xa = np.zeros(self.fL1, dtype=np.float32)


def refine_lag(c, st, ring, in_pos, lag_c):
    """Full-rate refinement of a coarse (D-quantised) lag.

    The decimated ACF can only place the period on a multiple of D, and the OLA
    write pointer integrates that error over thousands of granules (measured:
    corr 0.04 without refinement).  Here the analysis window is gathered at the
    *full* rate, windowed with the engine's ``win_an``, and a plain time-domain
    correlation is evaluated over ``lag_c +- D*DS_REFINE``; the engine's
    parabola is then applied at the refined integer peak, which restores the
    sub-sample lag.

    This is a lossy approximation by construction (the coarse ACF is whitened,
    this one is not) — it lives in the TD_FAST tier only.
    """
    d = c.d
    L1 = c.fL1
    mod = st.ring_tot if st.ring_tot else st.ring_len
    off = c.foff + in_pos
    if off[0] < 0 or off[-1] >= mod:
        idx = off % mod
    else:
        idx = off
    vis = idx < st.fed_visible
    xa = c._xa
    xi = np.where(vis, ring[0][idx], np.float32(0.0))
    if st.nch > 1:
        xi = xi + np.where(vis, ring[1][idx], np.float32(0.0))
    np.multiply(xi, st.pitch_win_an, out=xa)
    # ACF over lag_c +- d*DS_REFINE at full rate (dot in the same f32 order as
    # the engine's clarity region walks: strict ascending index).
    lo = int(lag_c) - d * c.DS_REFINE
    hi = int(lag_c) + d * c.DS_REFINE
    if lo < 1:
        lo = 1
    hi = min(hi, L1 - 1)
    nlag = hi - lo + 1
    if nlag < 3:
        return float(lag_c)
    # Vectorised normalised cross-correlation: c(L) = sum_i xa[i]*xa[i+L] /
    # sqrt(sum_i xa[i]^2 * sum_i xa[i+L]^2).  Normalising matters: the raw dot
    # is maximised by the shortest lag (window overlap), which made the refined
    # lag *worse* than the coarse one (measured corr 0.15 -> -0.02).  The
    # accumulation order per row is the same strict-ascending f32 one the
    # engine's clarity region uses.
    cols = np.arange(L1, dtype=np.int64)
    lags = np.arange(lo, hi + 1, dtype=np.int64)
    valid = cols[None, :] < (L1 - lags)[:, None]
    w_idx = np.where(valid, cols[None, :] + lags[:, None], 0)
    prod = (xa[w_idx] * xa[cols][None, :]).astype(np.float32)
    np.copyto(prod, np.float32(0.0), where=~valid)
    num = np.add.accumulate(prod, axis=1, dtype=np.float32)[:, -1].astype(np.float64)
    e_a = np.add.accumulate((xa[cols] * xa[cols]).astype(np.float32),
                            dtype=np.float32)[-1].astype(np.float64)
    ex = (xa[w_idx] * xa[w_idx]).astype(np.float32)
    np.copyto(ex, np.float32(0.0), where=~valid)
    e_b = np.add.accumulate(ex, axis=1, dtype=np.float32)[:, -1].astype(np.float64)
    den = np.sqrt(e_a * e_b)
    ac = np.where(den > 0.0, num / np.maximum(den, 1e-30), -2.0)
    k = int(np.argmax(ac))
    best_lag = float(lags[k])
    if 0 < k < ac.shape[0] - 1:
        y0 = float(ac[k - 1]); y1 = float(ac[k]); y2 = float(ac[k + 1])
        den2 = 2.0 * y1 - y2 - y0
        if den2 != 0.0:
            dd = (y2 - y0) / (2.0 * den2)
            if dd < -1.0:
                dd = -1.0
            if dd > 1.0:
                dd = 1.0
            best_lag = best_lag + dd
    return best_lag


def analyze_pitch_ds(st, ring, in_pos: int, c: "DecimatedPitch"):
    """Decimated front-end of :meth:`td_core.TDState.analyze_pitch` (TD_FAST_DS).

    The analysis window / windowed power ACF / whitening / final ACF all run at
    1/D of the sample rate (D decimation, no pre-filter — the analysis window
    is the anti-alias filter).  Lags, the octave search range and the clarity
    ranges are all scaled by D, so the returned lag is comparable to the exact
    one; the coarse peak is then refined at full rate in :func:`refine_lag`.
    Float discipline mirrors the exact path (f32 everywhere, op order
    preserved); only the sample grid is coarser.
    """
    import math

    from .td_core import PI, _octave_correct_peak

    N = c.N
    L1 = c.L1
    M = c.M
    nb = c.nb
    x = c._x
    x[:] = 0.0
    mod = st.ring_tot if st.ring_tot else st.ring_len
    off = c.gather_idx
    if off[0] + in_pos < 0 or off[-1] + in_pos >= mod:
        idx = (in_pos + off) % mod
    else:
        idx = in_pos + off
    vis = idx < st.fed_visible
    xi = np.where(vis, ring[0][idx], np.float32(0.0))
    if st.nch > 1:
        xi = xi + np.where(vis, ring[1][idx], np.float32(0.0))
    x[:L1] = xi
    energy_raw = float(np.sum(x[:L1].astype(np.float64) ** 2))
    rms = np.float32(np.sqrt(energy_raw / float(L1)))

    plan = c.plan
    x[:L1] *= c.win_an
    sa = plan.fwd(x)
    sa[1] = 0.0
    re = sa[0:2 * M:2]
    im = sa[1:2 * M:2]
    np.multiply(re, re, out=re)
    re += im * im
    sa[1:2 * M:2] = 0.0
    sa[N + 1] = 0.0
    x = plan.inv(sa)
    x *= c.win_acf
    sb = plan.fwd(x)

    sr = float(st.sr)
    v387 = int(0.5 + 20.0 / sr * float(N))
    v386 = int(0.5 + 150.0 / sr * float(N)) + 1
    e02 = energy_raw * 0.2
    noise = (e02 if e02 > 1e-6 else 1e-6) * (sr / 44100.0)
    noisef = np.float32(noise)
    kw = c.maxbin + c.taper_len
    if kw > M:
        kw = M
    out = c._out
    out[:] = 0.0
    if kw > v387:
        ks = np.arange(max(v387, 0), kw)
        mag = np.abs(sb[2 * ks])
        den = noisef + np.power(mag, np.float32(0.95))
        o = sa[2 * ks] / den
        tap = ks >= c.maxbin
        if np.any(tap):
            ti = np.where(tap, ks - c.maxbin, 0)
            ti = np.minimum(ti, c.taper_len - 1)
            o = np.where(tap, (o.astype(np.float64)
                               * c.win_taper[ti].astype(np.float64)).astype(np.float32), o)
        ramp = ks < v386
        if np.any(ramp):
            t = (ks[ramp].astype(np.float64) - float(v387)) / float(v386 - v387)
            fac = 0.5 - 0.5 * np.cos(PI * t)
            o = o.copy()
            o[ramp] = (o[ramp].astype(np.float64) * fac).astype(np.float32)
        out[ks] = o
    x2 = c._x2
    x2[:] = 0.0
    x2[0:2 * M:2] = out
    acf = plan.inv(x2)
    acf[:nb] = acf[:nb] * c.win_lin[:nb]
    i = 0
    while i + 1 < nb and acf[i] > acf[i + 1]:
        acf[i] = np.float32(-1000.0)
        i += 1
    v160 = int(c.lo)
    v138 = int(c.hi)
    if v160 < 1:
        v160 = 1
    if v138 > nb - 2:
        v138 = nb - 2
    if v138 <= v160:
        return 0.0, 0, float(rms)
    seg = acf[v160:v138 + 1]
    p = v160 + int(np.argmax(seg))
    run_oct = (st.pitch_octave_en == 0)
    res = _octave_correct_peak(acf, nb, v160, v138, run_oct)
    if res is not None and run_oct:
        p = res[0]
    st.pitch_octave_en = 0
    d = c.d
    lag_c = float(p * d)
    # Full-rate refinement of the D-quantised peak (see refine_lag).
    lag = refine_lag(c, st, ring, in_pos, lag_c)
    y1 = float(acf[p])
    v14 = y1
    comp = 0.0
    lb = int(0.6 * float(p)); rb = int(1.8 * float(p) + 0.5)
    if p - 4 >= 1:
        g = p - 4
        while g > lb and acf[g] < acf[g - 1]:
            g -= 1
        if g >= 1 and float(acf[g]) > comp:
            comp = float(acf[g])
    if p + 4 < nb - 1:
        g = p + 4
        while g < rb and g < nb - 2 and acf[g] < acf[g + 1]:
            g += 1
        if float(acf[g]) > comp:
            comp = float(acf[g])
    l22 = int(2.2 * float(p)); r28 = int(2.8 * float(p) + 0.5)
    hi = min(r28, nb - 2)
    if hi >= l22:
        base = max(l22, 1)
        if hi >= base:
            mx = float(acf[base:hi + 1].max())
            if mx > comp:
                comp = mx
    if comp < 0.0:
        comp = 0.0
    clarity = (1.0 - comp / v14) if v14 != 0.0 else 0.25
    if clarity < 0.0:
        clarity = 0.0
    w = int((float(st.win_max) + 0.4) * math.pow(clarity, 1.1) + 0.5)
    if w > st.win_max - 1:
        w = st.win_max - 1
    if w < 0:
        w = 0
    return lag, w, float(rms)


# ---------------------------------------------------------------------------
# installer
# ---------------------------------------------------------------------------

def avail() -> dict:
    return {"vdsp": _vdsp_fast() is not None, "interp": _interp_ok(),
            "ti": _ti_ok(), "pitch": _pitch_ok()}


def ds_factor() -> int:
    """Decimation factor for the pitch front-end (1 = disabled).

    Default 1: the decimated search is a *failed experiment* (see
    :class:`DecimatedPitch`) and must be opted into explicitly with
    ``TD_FAST_DS``.
    """
    try:
        d = int(os.environ.get("TD_FAST_DS", "1"))
    except ValueError:
        return 1
    return d if 2 <= d <= 8 else 1
