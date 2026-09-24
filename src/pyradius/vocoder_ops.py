"""pyradius.vocoder_ops — operator-level ports of the libradius vocoder chain.

One entry point per C operator (see libradius/src/ops/*.c), with the argument
shapes used by tests/test_ops_parity.py.  Floating-point discipline follows the
C sources: float32 throughout, C truncation semantics, fused C ops (fmaf /
NEON vfmaq) reproduced with :func:`_fma` (float64 accumulate — f32*f32 is
exact in f64, so only the final add can double-round), separated mul/add kept
separated, and `math.*` (libm double) for transcendentals cast back to f32.

Covered:
  fill_granule                 (fill_granule.c: FillGranule / FillGranuleWin)
  time_to_iir_a, acs_spectrum1, acs_spectrum2, cart_to_polar,
  threshold_lt_inplace         (analyze_channel_spectrum.c)
  unwrap_phase                 (unwrap_phase.c)
  apply_pitch_coherence        (apply_pitch_coherence.c)
  reset_phases_for_transients  (reset_phases.c)
  synchronize_stereo_phases, adjust_multiphase_diff, ampd_pull_to_peak
                               (synchronize_stereo_phases.c / adjust_multiphase_diff.c)
  FormantState                 (formant.c)
  overlap_add_channel          (overlap_add.c)
  randomize_phases, substitute_noisy_phases
                               (randomize_phases.c / substitute_noisy_phases.c)
  crossover_process1, crossover_process
                               (crossover.c — float64 FFT convolution; the C is a
                                f32 NEON 4-accumulator fma tree, so 1-2 ULP class
                                differences are expected and documented)
"""
from __future__ import annotations

import math
import os as _os
import struct as _struct

import numpy as np
from scipy.signal import fftconvolve, lfilter

from .fft import plan as _fft_plan
from .simple_rand import SimpleRand
from .tables import get_tables

F32 = np.float32

# ---------------------------------------------------------------------------
# bit-pattern constants (per the C `mov/movk` immediates; never decimal)
# ---------------------------------------------------------------------------


def _bits(u: int) -> np.float32:
    return np.frombuffer(np.uint32(u & 0xFFFFFFFF).tobytes(), dtype="<f4")[0]


PI_F = _bits(0x40490FDB)          # f32(pi)
PI_2_F = _bits(0x3FC90FDB)        # f32(pi/2)
TWO_PI_F = _bits(0x40C90FDB)      # f32(2pi)
NEG_2PI_F = _bits(0xC0C90FDB)
INV_2PI_F = _bits(0x3E22F983)     # f32(1/2pi)
EPS1E6_F = _bits(0x358637BD)      # 1e-6f
K07_F = _bits(0x3F333333)         # 0.7f
FLT_MAX_F = _bits(0x7F7FFFFF)


def _fma(a, b, c):
    """Correctly-rounded fmaf on f32 inputs.

    f32*f32 is exact in f64, but the f64 add can round twice when the f32 sum
    lands near a f32 midpoint, so the residual is recovered with TwoSum and
    applied as a one-ulp correction.  Verified against an exact Fraction
    reference over 50k random triples spanning 1e-30..1e6 (0 mismatches, vs
    ~30% mismatches for the naive f64-then-round form).
    """
    ad = float(a)
    bd = float(b)
    cd = float(c)
    p = ad * bd
    s = p + cd
    if not math.isfinite(s):
        return F32(s)
    bb = s - p
    err = (p - (s - bb)) + (cd - bb)      # exact residual of the f64 add
    r = F32(s)
    e = (s - float(r)) + err
    if e != 0.0:
        ulp = math.ulp(float(r))
        if abs(e) > 0.5 * ulp:
            r = F32(math.nextafter(float(r),
                                   math.inf if e > 0 else -math.inf))
    return r


try:
    from numba import njit as _njit
except ImportError:                       # pragma: no cover - numba optional
    _njit = None


def _fma_arr_fast(a, b, c):
    """Route-B fma: plain vectorized multiply-add (1-ulp vs fmaf)."""
    if type(a) is np.ndarray and type(b) is np.ndarray and type(c) is np.ndarray:
        r = (a.astype(np.float64) if a.dtype != np.float64 else a)
        r = r * (b.astype(np.float64) if b.dtype != np.float64 else b)
        r = r + (c.astype(np.float64) if c.dtype != np.float64 else c)
        return r.astype(np.float32)
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64)
            + np.asarray(c, np.float64)).astype(np.float32)


def _fma_arr(a, b, c):
    """Vectorised fmaf (f32 arrays/scalars) -> f32 array.

    The f64 product is exact; the residual of the f64 add is recovered with the
    same TwoSum trick as :func:`_fma` and applied as a one-ulp correction, so
    each element equals the correctly-rounded fmaf.

    numba kernel (same algorithm, compiled) is bit-identical and ~5x faster;
    the numpy fallback below reproduces it exactly when numba is absent.
    """
    a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
    if _route_b():
        return _fma_arr_fast(a, b, c)
    if type(a) is np.ndarray and a.dtype == np.float32 and a.flags.c_contiguous \
            and type(b) is np.ndarray and b.dtype == np.float32 and b.flags.c_contiguous \
            and type(c) is np.ndarray and c.dtype == np.float32 and c.flags.c_contiguous \
            and a.shape == b.shape == c.shape and a.ndim >= 1 \
            and _fma_arr_nb is not None:
        return _fma_arr_nb(a.reshape(-1), b.reshape(-1), c.reshape(-1)).reshape(a.shape)
    if _fma_arr_bs_nb is not None and a.ndim == 1 and np.isscalar(b) \
            and type(a) is np.ndarray and a.dtype == np.float32 and a.flags.c_contiguous \
            and type(c) is np.ndarray and c.dtype == np.float32 and c.flags.c_contiguous \
            and a.shape == c.shape:
        return _fma_arr_bs_nb(a, float(b), c)
    if _fma_arr_cs_nb is not None and a.ndim == 1 and np.isscalar(b) and np.isscalar(c) \
            and type(a) is np.ndarray and a.dtype == np.float32 and a.flags.c_contiguous:
        return _fma_arr_cs_nb(a, float(b), float(c))
    if a.ndim == b.ndim == c.ndim == 0:
        p = float(a) * float(b)
        s = p + float(c)
        bb = s - p
        err = (p - (s - bb)) + (float(c) - bb)
        r = F32(s)
        e = (s - float(r)) + err
        ulp = float(np.nextafter(r, np.float32(np.inf))) - float(r)
        dulp = float(r) - float(np.nextafter(r, np.float32(-np.inf)))
        if e > 0.5 * ulp:
            return F32(np.nextafter(r, np.float32(np.inf)))
        if e < -0.5 * dulp:
            return F32(np.nextafter(r, np.float32(-np.inf)))
        return r
    shape = np.broadcast_shapes(a.shape, b.shape, c.shape)
    a = np.ascontiguousarray(np.broadcast_to(a, shape), np.float32)
    b = np.ascontiguousarray(np.broadcast_to(b, shape), np.float32)
    c = np.ascontiguousarray(np.broadcast_to(c, shape), np.float32)
    if _fma_arr_nb is not None:
        out = _fma_arr_nb(a.reshape(-1), b.reshape(-1), c.reshape(-1))
        return out.reshape(shape)
    a6 = a.astype(np.float64); b6 = b.astype(np.float64); c6 = c.astype(np.float64)
    p = a6 * b6
    s = p + c6
    bb = s - p
    err = (p - (s - bb)) + (c6 - bb)
    r = s.astype(np.float32)
    e = (s - r.astype(np.float64)) + err
    up = np.nextafter(r, np.float32(np.inf))
    dn = np.nextafter(r, np.float32(-np.inf))
    ulp = (up.astype(np.float64) - r.astype(np.float64))
    step = np.where(e > 0.5 * ulp, up, np.where(e < -0.5 * ulp, dn, r))
    return np.where(np.isfinite(s), step, s).astype(np.float32)


if _njit is not None:
    @_njit(cache=True, fastmath=False)
    def _fma_arr_bs_nb(a, b, c):
        """(f32 arr, py float scalar, f32 arr) variant — most common call shape."""
        n = a.shape[0]
        out = np.empty(n, dtype=np.float32)
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        for i in range(n):
            ad = np.float64(a[i]); bd = b; cd = np.float64(c[i])
            p = ad * bd
            s = p + cd
            bb = s - p
            err = (p - (s - bb)) + (cd - bb)
            r = np.float32(s)
            e = (s - np.float64(r)) + err
            if np.isfinite(s):
                f32v[0] = r
                rb = bits[0]
                neg = (rb >> 31) != 0
                if neg:
                    bits[1] = rb - 1
                else:
                    bits[1] = rb + 1
                up = f32v[1]
                ulp = np.float64(up) - np.float64(r)
                if e > half * ulp:
                    r = up
                else:
                    if neg:
                        bits[1] = rb + 1
                    else:
                        bits[1] = rb - 1
                    dn = f32v[1]
                    dulp = np.float64(r) - np.float64(dn)
                    if e < -half * dulp:
                        r = dn
            else:
                r = np.float32(s)
            out[i] = r
        return out

    @_njit(cache=True, fastmath=False)
    def _fma_arr_cs_nb(a, b, c):
        """(f32 arr, py float scalar, py float scalar) variant."""
        n = a.shape[0]
        out = np.empty(n, dtype=np.float32)
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        for i in range(n):
            ad = np.float64(a[i])
            p = ad * b
            s = p + c
            bb = s - p
            err = (p - (s - bb)) + (c - bb)
            r = np.float32(s)
            e = (s - np.float64(r)) + err
            if np.isfinite(s):
                f32v[0] = r
                rb = bits[0]
                neg = (rb >> 31) != 0
                if neg:
                    bits[1] = rb - 1
                else:
                    bits[1] = rb + 1
                up = f32v[1]
                ulp = np.float64(up) - np.float64(r)
                if e > half * ulp:
                    r = up
                else:
                    if neg:
                        bits[1] = rb + 1
                    else:
                        bits[1] = rb - 1
                    dn = f32v[1]
                    dulp = np.float64(r) - np.float64(dn)
                    if e < -half * dulp:
                        r = dn
            else:
                r = np.float32(s)
            out[i] = r
        return out
else:
    _fma_arr_bs_nb = None
    _fma_arr_cs_nb = None

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _fma1(a, b, c, bits, f32v, half):
        """Scalar fmaf emulation (TwoSum), workspace views passed in."""
        ad = np.float64(a); bd = np.float64(b); cd = np.float64(c)
        p = ad * bd
        s = p + cd
        bb = s - p
        err = (p - (s - bb)) + (cd - bb)
        r = np.float32(s)
        e = (s - np.float64(r)) + err
        if np.isfinite(s):
            f32v[0] = r
            rb = bits[0]
            neg = (rb >> 31) != 0
            if neg:
                bits[1] = rb - 1
            else:
                bits[1] = rb + 1
            up = f32v[1]
            ulp = np.float64(up) - np.float64(r)
            if e > half * ulp:
                return up
            if neg:
                bits[1] = rb + 1
            else:
                bits[1] = rb - 1
            dn = f32v[1]
            dulp = np.float64(r) - np.float64(dn)
            if e < -half * dulp:
                return dn
        return r
else:
    _fma1 = None

_VDSP_REF = None
_ROUTE_B_CACHE = None


def _route_b() -> bool:
    """Live read of vocoder_core.ROUTE_B_FAST (circular-import safe)."""
    global _ROUTE_B_CACHE
    if _ROUTE_B_CACHE is None:
        try:
            from . import vocoder_core as _vc
            _ROUTE_B_CACHE = bool(_vc.ROUTE_B_FAST)
        except Exception:
            _ROUTE_B_CACHE = False
    return _ROUTE_B_CACHE


_NEON_DATA_CACHE = None


def _neon_data():
    """Lazy pyradius.neon handle, only while route B is on and the C kernels
    have passed their bit-exactness self-check (see neon.data_ok).

    Returns the module, or None so callers fall back to numba/numpy.  Import is
    deferred because neon.py's certification imports this module.
    """
    global _NEON_DATA_CACHE
    if _NEON_DATA_CACHE is None:
        if _route_b():
            try:
                from . import neon as _neon
                # False (not None) caches the miss, so the non-route-B and
                # no-NEON paths pay this check only once.
                _NEON_DATA_CACHE = _neon if _neon.data_ok() else False
            except Exception:
                _NEON_DATA_CACHE = False
        else:
            _NEON_DATA_CACHE = False
    return _NEON_DATA_CACHE or None


def _vdsp_mod_ref():
    """Lazy vdsp module (route B only)."""
    global _VDSP_REF
    if _VDSP_REF is None:
        try:
            from . import vdsp
            _VDSP_REF = vdsp if vdsp.HAVE_VDSP else False
        except Exception:
            _VDSP_REF = False
    return _VDSP_REF or None

if _njit is not None:

    @_njit(cache=True, fastmath=True)
    def _ctp_fast_nb(re, im):
        """Route-B magnitude: plain f32 hypot-style sqrt(re^2+im^2)."""
        n = re.size
        m = np.empty(n, dtype=np.float32)
        for i in range(n):
            m[i] = np.sqrt(re[i] * re[i] + im[i] * im[i])
        return m

    @_njit(cache=True, fastmath=True)
    def _ctp_fast_nb_ph(re, im):
        """Route-B phase: atan2 via the engine polynomial, fastmath f32."""
        n = re.size
        ph = np.empty(n, dtype=np.float32)
        c0 = np.float32(_CTP_COEF[0]); c1 = np.float32(_CTP_COEF[1])
        c2 = np.float32(_CTP_COEF[2]); c3 = np.float32(_CTP_COEF[3])
        c4 = np.float32(_CTP_COEF[4]); c5 = np.float32(_CTP_COEF[5])
        c6 = np.float32(_CTP_COEF[6]); c7 = np.float32(_CTP_COEF[7])
        for i in range(n):
            ar = re[i]; ai = im[i]
            are = abs(ar); aim = abs(ai)
            mx = are if are > aim else aim
            mx0 = np.float32(1.0) if mx == np.float32(0.0) else mx
            mn = are if aim > are else aim
            r = mn / mx0
            r2 = r * r
            p = c0 * r2 + c1
            p = p * r2 + c2
            p = p * r2 + c3
            p = p * r2 + c4
            p = p * r2 + c5
            p = p * r2 + c6
            p = p * r2 + c7
            p = p * r2 + np.float32(1.0)
            ang = r * p
            if aim > are:
                ang = np.float32(PI_2_F) - ang
            if ar < np.float32(0.0):
                ang = np.float32(PI_F) - ang
            if np.signbit(ai):
                ang = -ang
            ph[i] = ang
        return ph
else:
    _ctp_fast_nb = None
    _ctp_fast_nb_ph = None


if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _unwrap_nb(mask, mcref, mag, reg_start, reg_end, reg_prev_peak,
                   reg_offset, scratch, phase, f1f, f3f, f2f, inv_f1, v9,
                   nreg, copy_len, max_bin):
        """Route-B unwrap: scalar region loops (region count is small)."""
        mb = max_bin - 1
        for r in range(nreg):
            rs = reg_start[r]; re_ = reg_end[r]
            if re_ < rs:
                continue
            peak = rs
            best = mag[rs]
            i = rs + 1
            while i <= re_:
                if mag[i] > best:
                    best = mag[i]
                    peak = i
                i += 1
            v16 = np.float32(peak)
            if peak != 0 and peak != mb:
                ap = mag[peak + 1]; am = mag[peak]; amm = mag[peak - 1]
                v20 = ap - np.float32(2.0) * am - amm
                if v20 != np.float32(0.0):
                    v21 = ((ap - amm) / (v20 + v20)) + v16
                else:
                    v21 = v16
                t = v16 + np.float32(0.7)
                if not (t < v21):
                    t = v21
                lo = v16 - np.float32(0.7)
                if t < lo:
                    t = lo
                v16 = t
            v37 = v9 * (v16 + reg_offset[r])
            dlt = peak - reg_prev_peak[r]
            for b in range(rs, re_):
                idx = b + dlt
                if idx < 0:
                    idx = 0
                elif idx > mb:
                    idx = mb
                v38 = (mask[b] - mcref[idx]) - v37
                q = np.rint(v38 * np.float32(0.15915494))
                wrap = np.float32(np.float64(q) * -6.2831854820251465 + np.float64(v38))
                term = inv_f1 * (v37 + wrap)
                scratch[b] = np.float32(np.float64(f2f) * np.float64(term)
                                        + np.float64(phase[idx]))
        for i in range(copy_len):
            phase[i] = scratch[i]
        return phase
else:
    _unwrap_nb = None

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _fgw_nb(ring, win, acc, band, off, hop, N5):
        """Route-B fgw one band: two windowed fma wings (f64 mul-add)."""
        r = ring[band]
        w = win[band]
        for i in range(hop):
            acc[i] = np.float32(np.float64(r[off + hop + i]) * np.float64(w[hop + i])
                                + np.float64(acc[i]))
            j = N5 - hop + i
            acc[j] = np.float32(np.float64(r[off + i]) * np.float64(w[i])
                                + np.float64(acc[j]))

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _apc_region_nb(phase, phase_mod, dir_prev, dir_cur, rrs, ree, v130):
        """Route-B apc region update (non-vec semantics: separate mul/add)."""
        for i in range(rrs, ree):
            b = np.float32(i)
            dd = (phase[i] + (b * v130)) - phase_mod[i]
            nn = np.rint(dd * np.float32(0.15915494))
            phase_mod[i] = phase_mod[i] + (dd + (nn * np.float32(-6.2831854820251465)))
            dir_prev[i] = dir_cur[i]
            dir_cur[i] = np.float32(0.0)

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _fm_db_nb(mag, db, NB):
        """Formant step 1: dB curve (bit-equal to the numpy path)."""
        thr = 9.999999999988105e-21
        for i in range(NB):
            mv = float(mag[i])
            if mv >= thr:
                db[i] = np.float32(np.log(mv if mv > 1e-300 else 1e-300)
                                   * 8.68588924407959)
            else:
                db[i] = np.float32(-391.0)

    @_njit(cache=True, fastmath=False)
    def _fm_env_nb(spec, env, MB, a1):
        """Formant step 3: envelope IIR (fma single-round)."""
        for k in range(MB):
            re = spec[2 * k]
            im = spec[2 * k + 1]
            e = re * re + im * im
            env[k] = np.float32(a1 * np.float64(e - env[k]) + np.float64(env[k]))


    @_njit(cache=True, fastmath=False)
    def _fm_gain_nb(db, gscr, ges, mag, NB, ratio, strength):
        """Formant step 9 gain (sticky d) + db2amp, bit-equal."""
        last_d = np.float32(0.0)
        for v in range(NB):
            k2f = np.float32(ratio) * np.float32(v)
            if k2f < 0.0:
                k2 = int(k2f - 0.5)
            else:
                k2 = int(k2f + 0.5)
            if k2 < NB:
                d = db[k2] - db[v]
                last_d = d
            else:
                d = last_d
            if d > np.float32(20.0):
                d = np.float32(20.0)
            elif d < np.float32(-40.0):
                d = np.float32(-40.0)
            gscr[v] = np.float32(np.exp(np.float64(d) * np.float64(strength)
                                        * 0.115129254758358))

    @_njit(cache=True, fastmath=False)
    def _fm_env2_nb(ffr, ffi, env, MB, a1):
        """Step 3 reading ffr/ffi directly (skips the spec pack)."""
        for k in range(MB):
            re = ffr[k]
            im = ffi[k]
            e = re * re + im * im
            env[k] = np.float32(a1 * np.float64(e - env[k]) + np.float64(env[k]))

    @_njit(cache=True, fastmath=False)
    def _fm_clip_mirror_nb(ffr, ffi, MB, M, c4):
        """Steps 5-tail+6-prep on ffr/ffi directly (no spec array).

        spec pairs (2k, 2k+1) map to (ffr[k], ffi[k]); the C clip touches
        complex bins cut-2..cut-1 with 0.75/0.75/0.25/0.25 and zeroes bins
        cut.. — here applied pre-mirror, then hermitian mirror built.
        """
        for j in range(4):
            k = (c4 >> 1) - 2 + j
            f = np.float32(0.75) if j < 2 else np.float32(0.25)
            if 0 <= k < MB:
                ffr[k] = np.float32(ffr[k] * f)
                ffi[k] = np.float32(ffi[k] * f)
        for k in range((c4 >> 1), MB):
            ffr[k] = np.float32(0.0)
            ffi[k] = np.float32(0.0)
        # hermitian mirror for the inverse FFT (builds a full M spectrum
        # from the packed MB half)
        ffi[0] = np.float32(0.0)
        ffi[MB - 1] = np.float32(0.0)
        for k in range(1, MB - 1):
            ffr[M - k] = ffr[k]
            ffi[M - k] = -ffi[k]

    @_njit(cache=True, fastmath=False)
    def _fm_ker_scale_nb(ffr, ker, M, inv_scale):
        """Step 6-post: ker = ffr * inv_scale (single pass)."""
        for i in range(M):
            ker[i] = np.float32(ffr[i] * inv_scale)

    @_njit(cache=True, fastmath=False)
    def _fm_ker_nb(spec, ffr, ffi, ker, MB, M, inv_scale):
        """Step 6: unpack spec into ffr/ffi (stride-2 reads), mirror, scale."""
        for k in range(MB):
            ffr[k] = spec[2 * k]
            ffi[k] = spec[2 * k + 1]
        ffi[0] = np.float32(0.0)
        ffi[MB - 1] = np.float32(0.0)
        for k in range(1, MB - 1):
            ffr[M - k] = ffr[k]
            ffi[M - k] = -ffi[k]

    @_njit(cache=True, fastmath=False)
    def _fm_prep_nb(mag, db, ffr, ffi, NB, M):
        """Steps 1+2-prep fused: dB curve into db AND ffr, zero ffi.

        db beyond NB (up to M) is zero-filled; bit-equal to the numpy path.
        """
        thr = 9.999999999988105e-21
        for i in range(NB):
            mv = float(mag[i])
            if mv >= thr:
                dbv = np.log(mv if mv > 1e-300 else 1e-300) * 8.68588924407959
                db[i] = np.float32(dbv)
            else:
                db[i] = np.float32(-391.0)
        # db[NB:M] keeps the kermix residue from the previous call (the C's
        # persistent scratch) — ffr copies whatever db holds.
        for i in range(M):
            ffr[i] = db[i]
            ffi[i] = np.float32(0.0)

    @_njit(cache=True, fastmath=False)
    def _fm_gain2_nb(db, gscr, NB, ratio, strength):
        """Step 9 fused: sticky-d gain (bit-equal to _fm_gain_nb)."""
        last_d = np.float32(0.0)
        for v in range(NB):
            k2f = np.float32(ratio) * np.float32(v)
            if k2f < 0.0:
                k2 = int(k2f - 0.5)
            else:
                k2 = int(k2f + 0.5)
            if k2 < NB:
                d = db[k2] - db[v]
                last_d = d
            else:
                d = last_d
            if d > np.float32(20.0):
                d = np.float32(20.0)
            elif d < np.float32(-40.0):
                d = np.float32(-40.0)
            gscr[v] = np.float32(np.exp(np.float64(d) * np.float64(strength)
                                        * 0.115129254758358))

    @_njit(cache=True, fastmath=False)
    def _fm_ges_tail_nb(gscr, ges, mag, NB, a3, fC, N, sr):
        """Steps 11+12 fused: gain envelope IIR + tail multiply."""
        for v in range(NB):
            ges[v] = np.float32(a3 * np.float64(gscr[v] - ges[v]) + np.float64(ges[v]))
        vt = np.float32(np.float32(fC * np.float32(0.8)) * np.float32(N) / np.float32(sr))
        tail = int(vt + (0.5 if vt >= 0 else -0.5))
        if tail > NB - 1:
            tail = NB - 1
        for v in range(tail, NB):
            mag[v] = np.float32(ges[v] * mag[v])

    @_njit(cache=True, fastmath=False)
    def _fm_env2_nb(ffr, ffi, env, MB, a1):
        """Step 3 reading ffr/ffi directly (skips the spec pack)."""
        for k in range(MB):
            re = ffr[k]
            im = ffi[k]
            e = re * re + im * im
            env[k] = np.float32(a1 * np.float64(e - env[k]) + np.float64(env[k]))

    @_njit(cache=True, fastmath=False)
    def _fm_spec_clip_nb(ffr, ffi, MB, c4):
        """Steps 5-tail+6-prep: clip scalars + zero tail, all in ffr/ffi."""
        for j in range(4):
            v = ffr[c4 - 4 + j] if (c4 - 4 + j) < MB else np.float32(0.0)
        # clip applies to spec (=ffr/ffi pairs) with 0.75/0.75/0.25/0.25 on
        # complex samples c4/2-2 .. c4/2-1 — see caller for exact mapping.
        for k in range(c4 >> 1, MB):
            ffr[k] = np.float32(0.0)
            ffi[k] = np.float32(0.0)

    @_njit(cache=True, fastmath=False)
    def _fm_clip_mirror_nb(ffr, ffi, MB, M, c4):
        """Steps 5-tail+6-prep on ffr/ffi directly (no spec array).

        spec pairs (2k, 2k+1) map to (ffr[k], ffi[k]); the C clip touches
        complex bins cut-2..cut-1 with 0.75/0.75/0.25/0.25 and zeroes bins
        cut.. — here applied pre-mirror, then hermitian mirror built.
        """
        for j in range(4):
            k = (c4 >> 1) - 2 + j
            f = np.float32(0.75) if j < 2 else np.float32(0.25)
            if 0 <= k < MB:
                ffr[k] = np.float32(ffr[k] * f)
                ffi[k] = np.float32(ffi[k] * f)
        for k in range((c4 >> 1), MB):
            ffr[k] = np.float32(0.0)
            ffi[k] = np.float32(0.0)
        # hermitian mirror for the inverse FFT (builds a full M spectrum
        # from the packed MB half)
        ffi[0] = np.float32(0.0)
        ffi[MB - 1] = np.float32(0.0)
        for k in range(1, MB - 1):
            ffr[M - k] = ffr[k]
            ffi[M - k] = -ffi[k]

    @_njit(cache=True, fastmath=False)
    def _fm_ker_scale_nb(ffr, ker, M, inv_scale):
        """Step 6-post: ker = ffr * inv_scale (single pass)."""
        for i in range(M):
            ker[i] = np.float32(ffr[i] * inv_scale)

    @_njit(cache=True, fastmath=False)
    def _fm_ker_nb(spec, ffr, ffi, ker, MB, M, inv_scale):
        """Step 6: unpack spec into ffr/ffi (stride-2 reads), mirror, scale."""
        for k in range(MB):
            ffr[k] = spec[2 * k]
            ffi[k] = spec[2 * k + 1]
        ffi[0] = np.float32(0.0)
        ffi[MB - 1] = np.float32(0.0)
        for k in range(1, MB - 1):
            ffr[M - k] = ffr[k]
            ffi[M - k] = -ffi[k]

    @_njit(cache=True, fastmath=False)
    def _fm_prep_nb(mag, db, ffr, ffi, NB, M):
        """Steps 1+2-prep fused: dB curve into db AND ffr, zero ffi.

        db beyond NB (up to M) is zero-filled; bit-equal to the numpy path.
        """
        thr = 9.999999999988105e-21
        for i in range(NB):
            mv = float(mag[i])
            if mv >= thr:
                dbv = np.log(mv if mv > 1e-300 else 1e-300) * 8.68588924407959
                db[i] = np.float32(dbv)
            else:
                db[i] = np.float32(-391.0)
        # db[NB:M] keeps the kermix residue from the previous call (the C's
        # persistent scratch) — ffr copies whatever db holds.
        for i in range(M):
            ffr[i] = db[i]
            ffi[i] = np.float32(0.0)

    @_njit(cache=True, fastmath=False)
    def _fm_tail_nb(db, gscr, ges, mag, NB, ratio, strength, a3, fC,
                    N, sr, mode_rms):
        """Steps 9+10+11+12 fused (bit-equal per-step f32 semantics)."""
        # step 9: sticky-d gain
        last_d = np.float32(0.0)
        for v in range(NB):
            k2f = np.float32(ratio) * np.float32(v)
            if k2f < 0.0:
                k2 = int(k2f - 0.5)
            else:
                k2 = int(k2f + 0.5)
            if k2 < NB:
                d = db[k2] - db[v]
                last_d = d
            else:
                d = last_d
            if d > np.float32(20.0):
                d = np.float32(20.0)
            elif d < np.float32(-40.0):
                d = np.float32(-40.0)
            gscr[v] = np.float32(np.exp(np.float64(d) * np.float64(strength)
                                        * 0.115129254758358))
        # step 10: RMS normalisation
        if mode_rms != 0:
            den = 0.0
            num = 0.0
            for v in range(NB):
                mv = float(mag[v])
                gv = float(gscr[v])
                den += mv * mv
                num += gv * gv * mv * mv
            ss = np.float32(math.sqrt(den / (num + 1.000029594723506e-12)))
            for v in range(NB):
                gscr[v] = np.float32(ss * gscr[v])
        # step 11: gain envelope IIR
        for v in range(NB):
            ges[v] = np.float32(a3 * np.float64(gscr[v] - ges[v]) + np.float64(ges[v]))
        # step 12: tail multiply
        vt = np.float32(np.float32(fC * np.float32(0.8)) * np.float32(N) / np.float32(sr))
        tail = int(vt + (0.5 if vt >= 0 else -0.5))
        if tail > NB - 1:
            tail = NB - 1
        for v in range(tail, NB):
            mag[v] = np.float32(ges[v] * mag[v])

    @_njit(cache=True, fastmath=False)
    def _fm_peak_nb(env, MB, v19, freq_hi, freq_lo, mode_freq, thr_score):
        """Formant step 4+5: peak search + peak freq/clip scalars.

        Returns (best, fC, cut) — fC already clipped to [150, 800].
        Bit-equal to the numpy chain (f32 math, same comparisons).
        """
        hi = v19 / freq_hi
        hi2 = int(hi + (0.5 if hi >= 0 else -0.5))
        if hi2 > MB - 2:
            hi2 = MB - 2
        if hi2 < 2:
            hi2 = 2
        lo_ = v19 / freq_lo
        lo2 = int(lo_ + (0.5 if lo_ >= 0 else -0.5))
        if lo2 > MB - 2:
            lo2 = MB - 2
        best = 0
        if lo2 > hi2:
            # vectorised score over (hi2, lo2)
            n_sc = lo2 - hi2
            sc = np.empty(n_sc, dtype=np.float32)
            for j in range(n_sc):
                k = hi2 + j
                v = env[k]
                den1 = (env[k - 1] + env[k + 1]) + np.float32(1e-6)
                r1 = v / den1
                if r1 <= np.float32(0.5):
                    w1 = np.float32(0.2)
                elif r1 < np.float32(2.0):
                    w1 = (r1 - np.float32(0.5)) / np.float32(1.5) + np.float32(0.2)
                else:
                    w1 = np.float32(1.2)
                r2 = v / (env[k >> 1] + np.float32(1e-6))
                if r2 > np.float32(0.3):
                    if r2 < np.float32(5.0):
                        w2 = (r2 - np.float32(0.3)) / np.float32(4.7)
                    else:
                        w2 = np.float32(1.0)
                else:
                    w2 = np.float32(0.0)
                sc[j] = (v * w1) * w2
            bi = 0
            bmax = sc[0]
            for j in range(1, n_sc):
                if sc[j] > bmax:
                    bmax = sc[j]
                    bi = j
            if bmax > thr_score:
                best = hi2 + bi
        if mode_freq:
            pk = (v19 / best) if best else np.float32(0.0)
            peak = best
        else:
            pk = np.float32(500.0)
            peak = int(v19 / pk + (0.5 if v19 / pk >= 0 else -0.5))
        fC = pk if pk <= np.float32(800.0) else np.float32(800.0)
        fC = fC if fC >= np.float32(150.0) else np.float32(150.0)
        return best, fC, peak

    @_njit(cache=True, fastmath=False)
    def _fm_kermix_nb(db, ker, M, h1, h2):
        """Formant step 8: ker mix-in (fma single-round)."""
        if h1 > 0:
            for i in range(h1):
                db[i] = ker[i]
        if h2 > h1:
            span = np.float32(h2 - h1)
            for k in range(h1, h2):
                w = np.float32(h2 - k) / span
                db[k] = np.float32(np.float64(w) * np.float64(ker[k] - db[k])
                                   + np.float64(db[k]))

    @_njit(cache=True, fastmath=False)
    def _fm_ges_nb(gscr, ges, NB, a3):
        """Formant step 11: gain envelope IIR."""
        for k in range(NB):
            ges[k] = np.float32(a3 * np.float64(gscr[k] - ges[k]) + np.float64(ges[k]))

if _njit is not None:

    @_njit(cache=True, fastmath=False, nogil=True)
    def _iir16_2_nb(y0, y1, a2, a1m):
        """Two independent 8x fwd+bwd first-order IIR chains interleaved.

        Identical f32 round-trip math to _iir16_nb per chain; interleaving
        only adds instruction-level parallelism to hide the recurrence
        latency.  Both chains must have the same length.
        """
        n = y0.size
        b0 = a2
        for _ in range(8):
            p0 = b0 * float(y0[0]) + (1.0 - a2) * float(y0[0])
            p1 = b0 * float(y1[0]) + (1.0 - a2) * float(y1[0])
            y0[0] = np.float32(p0)
            y1[0] = np.float32(p1)
            for i in range(1, n):
                p0 = b0 * float(y0[i]) + a1m * p0
                p1 = b0 * float(y1[i]) + a1m * p1
                y0[i] = np.float32(p0)
                y1[i] = np.float32(p1)
            p0 = b0 * float(y0[n - 1]) + (1.0 - a2) * float(y0[n - 1])
            p1 = b0 * float(y1[n - 1]) + (1.0 - a2) * float(y1[n - 1])
            y0[n - 1] = np.float32(p0)
            y1[n - 1] = np.float32(p1)
            for i in range(n - 2, -1, -1):
                p0 = b0 * float(y0[i]) + a1m * p0
                p1 = b0 * float(y1[i]) + a1m * p1
                y0[i] = np.float32(p0)
                y1[i] = np.float32(p1)
        return y0, y1

    @_njit(cache=True, fastmath=False, nogil=True)
    def _iir16_nb(y, a2, a1m):
        """8x forward+backward first-order IIR with f32 round-trips per pass.

        Mirrors the scipy fallback: y starts f64-valued but each double-pass
        result is rounded to f32 then back (the C's float storage), lfilter's
        f64 recurrence per pass, zi=(1-a2)*y[0].
        """
        n = y.size
        b0 = a2
        for _ in range(8):
            # forward: y[i] = b0*x[i] + a1m*y[i-1], y[0] = b0*x[0] + zi
            zi = (1.0 - a2) * float(y[0])
            prev = b0 * float(y[0]) + zi
            y[0] = np.float32(prev)
            for i in range(1, n):
                prev = b0 * float(y[i]) + a1m * prev
                y[i] = np.float32(prev)
            # backward
            zi = (1.0 - a2) * float(y[n - 1])
            prev = b0 * float(y[n - 1]) + zi
            y[n - 1] = np.float32(prev)
            for i in range(n - 2, -1, -1):
                prev = b0 * float(y[i]) + a1m * prev
                y[i] = np.float32(prev)
        return y
else:
    _iir16_nb = None
    _fm_gain_nb = None
    _apc_region_nb = None
    _fgw_nb = None

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _xover_nb(hp, taps_rev, out, n, N):
        """Crossover fast path: plain f32(f64 mul + add) lanes (vectorizable).

        Followed by _xover_fix_nb which re-rounds the (rare) elements where
        the f64 intermediate changed the correctly-rounded fmaf result.
        """
        TB = 128
        s = np.zeros((16, TB), dtype=np.float32)
        for t0 in range(0, n, TB):
            tb = min(TB, n - t0)
            for j in range(16):
                for t in range(tb):
                    s[j, t] = np.float32(0.0)
            for k in range(0, N, 16):
                for j in range(16):
                    tap = taps_rev[k + j]
                    for t in range(tb):
                        s[j, t] = np.float32(np.float64(hp[t0 + t, k + j]) * np.float64(tap)
                                             + np.float64(s[j, t]))
            for t in range(tb):
                u0 = np.float32(np.float64(s[0, t]) + np.float64(s[4, t]))
                u1 = np.float32(np.float64(s[1, t]) + np.float64(s[5, t]))
                u2 = np.float32(np.float64(s[2, t]) + np.float64(s[6, t]))
                u3 = np.float32(np.float64(s[3, t]) + np.float64(s[7, t]))
                u4 = np.float32(np.float64(s[8, t]) + np.float64(s[12, t]))
                u5 = np.float32(np.float64(s[9, t]) + np.float64(s[13, t]))
                u6 = np.float32(np.float64(s[10, t]) + np.float64(s[14, t]))
                u7 = np.float32(np.float64(s[11, t]) + np.float64(s[15, t]))
                v0 = np.float32(np.float64(u0) + np.float64(u4))
                v1 = np.float32(np.float64(u1) + np.float64(u5))
                v2 = np.float32(np.float64(u2) + np.float64(u6))
                v3 = np.float32(np.float64(u3) + np.float64(u7))
                out[t0 + t] = np.float32(np.float64(np.float32(np.float64(v0) + np.float64(v1)))
                                         + np.float64(np.float32(np.float64(v2) + np.float64(v3))))
        return out

if _njit is not None:
    _prange = __import__("numba").prange

    @_njit(cache=True, fastmath=True, parallel=True)
    def _xover_zp_nb(z, taps_rev, out, n, N, n_bands):
        """Route-B crossover, all bands in parallel (prange over 4 P-cores)."""
        TB = 256
        for b in _prange(n_bands):
            s = np.zeros((TB, 16), dtype=np.float32)
            tr = taps_rev[b]
            ob = out[b]
            for t0 in range(0, n, TB):
                tb = min(TB, n - t0)
                for t in range(tb):
                    for j in range(16):
                        s[t, j] = np.float32(0.0)
                for k in range(0, N, 16):
                    for t in range(tb):
                        row = t0 + t
                        base = k
                        for j in range(16):
                            s[t, j] = z[row + base + j] * tr[base + j] + s[t, j]
                for t in range(tb):
                    u0 = s[t, 0] + s[t, 4]
                    u1 = s[t, 1] + s[t, 5]
                    u2 = s[t, 2] + s[t, 6]
                    u3 = s[t, 3] + s[t, 7]
                    u4 = s[t, 8] + s[t, 12]
                    u5 = s[t, 9] + s[t, 13]
                    u6 = s[t, 10] + s[t, 14]
                    u7 = s[t, 11] + s[t, 15]
                    ob[t0 + t] = (u0 + u4 + u1 + u5) + (u2 + u6 + u3 + u7)

    @_njit(cache=True, fastmath=True)
    def _xover_z_nb(z, taps_rev, out, n, N):
        """Route-B crossover on the raw z buffer (no sliding-window copy).

        Row t reads z[t .. t+N) contiguously; same math as _xover_fast_nb.
        """
        TB = 256
        s = np.zeros((TB, 16), dtype=np.float32)
        for t0 in range(0, n, TB):
            tb = min(TB, n - t0)
            for t in range(tb):
                for j in range(16):
                    s[t, j] = np.float32(0.0)
            for k in range(0, N, 16):
                for t in range(tb):
                    row = t0 + t
                    base = k
                    for j in range(16):
                        s[t, j] = z[row + base + j] * taps_rev[base + j] + s[t, j]
            for t in range(tb):
                u0 = s[t, 0] + s[t, 4]
                u1 = s[t, 1] + s[t, 5]
                u2 = s[t, 2] + s[t, 6]
                u3 = s[t, 3] + s[t, 7]
                u4 = s[t, 8] + s[t, 12]
                u5 = s[t, 9] + s[t, 13]
                u6 = s[t, 10] + s[t, 14]
                u7 = s[t, 11] + s[t, 15]
                out[t0 + t] = (u0 + u4 + u1 + u5) + (u2 + u6 + u3 + u7)
        return out

    @_njit(cache=True, fastmath=True)
    def _xover_fast_nb(hp, taps_rev, out, n, N):
        """Route-B crossover: row-contiguous f32 lanes (SIMD), relaxed tree.

        hp[t, k:k+16] is contiguous so the j loop vectorizes; s is (TB, 16).
        Reduction groups as (u0+u4+u1+u5)+(u2+u6+u3+u7) — same lane set,
        1-ulp-level differences vs the engine tree (route B accepts this).
        """
        TB = 128
        s = np.zeros((TB, 16), dtype=np.float32)
        for t0 in range(0, n, TB):
            tb = min(TB, n - t0)
            for t in range(tb):
                for j in range(16):
                    s[t, j] = np.float32(0.0)
            for k in range(0, N, 16):
                for t in range(tb):
                    row = t0 + t
                    base = k
                    for j in range(16):
                        s[t, j] = hp[row, base + j] * taps_rev[base + j] + s[t, j]
            for t in range(tb):
                u0 = s[t, 0] + s[t, 4]
                u1 = s[t, 1] + s[t, 5]
                u2 = s[t, 2] + s[t, 6]
                u3 = s[t, 3] + s[t, 7]
                u4 = s[t, 8] + s[t, 12]
                u5 = s[t, 9] + s[t, 13]
                u6 = s[t, 10] + s[t, 14]
                u7 = s[t, 11] + s[t, 15]
                out[t0 + t] = (u0 + u4 + u1 + u5) + (u2 + u6 + u3 + u7)
        return out

    @_njit(cache=True, fastmath=False)
    def _xover_fix_nb(hp, taps_rev, out, n, N):
        """Exactness repair for _xover_nb: recompute lanes with full TwoSum
        fmaf emulation ONLY where the plain lane chain may have mis-rounded.

        A lane element needs repair when, at some accumulation step, the f64
        rounding of p+c differed from exact addition (double-rounding). The
        cheap detector: recompute each step's TwoSum residual e and flag when
        e != 0 AND the f32 rounding was a near-tie. To stay honest and simple,
        this kernel recomputes the whole chain per flagged t; flagging uses
        the same criterion the exact kernel's correction branch would use.
        """
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        for t in range(n):
            # detect: run exact chain, compare with out only if any step had
            # a nonzero TwoSum error affecting rounding
            need = False
            lanes = np.zeros(16, dtype=np.float32)
            for k in range(0, N, 16):
                for j in range(16):
                    ad = np.float64(hp[t, k + j]); bd = np.float64(taps_rev[k + j])
                    cd = np.float64(lanes[j])
                    p = ad * bd
                    s2 = p + cd
                    bb = s2 - p
                    err = (p - (s2 - bb)) + (cd - bb)
                    r = np.float32(s2)
                    e = (s2 - np.float64(r)) + err
                    if e != 0.0 and np.isfinite(s2):
                        f32v[0] = r
                        rb = bits[0]
                        neg = (rb >> 31) != 0
                        if neg:
                            bits[1] = rb - 1
                        else:
                            bits[1] = rb + 1
                        up = f32v[1]
                        ulp = np.float64(up) - np.float64(r)
                        if e > half * ulp:
                            r = up
                            need = True
                        else:
                            if neg:
                                bits[1] = rb + 1
                            else:
                                bits[1] = rb - 1
                            dn = f32v[1]
                            dulp = np.float64(r) - np.float64(dn)
                            if e < -half * dulp:
                                r = dn
                                need = True
                    lanes[j] = r
            if need:
                u0 = np.float32(np.float64(lanes[0]) + np.float64(lanes[4]))
                u1 = np.float32(np.float64(lanes[1]) + np.float64(lanes[5]))
                u2 = np.float32(np.float64(lanes[2]) + np.float64(lanes[6]))
                u3 = np.float32(np.float64(lanes[3]) + np.float64(lanes[7]))
                u4 = np.float32(np.float64(lanes[8]) + np.float64(lanes[12]))
                u5 = np.float32(np.float64(lanes[9]) + np.float64(lanes[13]))
                u6 = np.float32(np.float64(lanes[10]) + np.float64(lanes[14]))
                u7 = np.float32(np.float64(lanes[11]) + np.float64(lanes[15]))
                v0 = np.float32(np.float64(u0) + np.float64(u4))
                v1 = np.float32(np.float64(u1) + np.float64(u5))
                v2 = np.float32(np.float64(u2) + np.float64(u6))
                v3 = np.float32(np.float64(u3) + np.float64(u7))
                out[t] = np.float32(np.float64(np.float32(np.float64(v0) + np.float64(v1)))
                                    + np.float64(np.float32(np.float64(v2) + np.float64(v3))))
        return out
else:
    _xover_fix_nb = None
    _xover_nb = None
    _xover_fast_nb = None
    _xover_z_nb = None
    _xover_zp_nb = None

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _ctp_nb(re, im, n):
        """Fused CartToPolar: polynomial atan2 + sqrt(mag), fmaf-exact, stepwise f32."""
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        c0 = np.float32(_CTP_COEF[0]); c1 = np.float32(_CTP_COEF[1])
        c2 = np.float32(_CTP_COEF[2]); c3 = np.float32(_CTP_COEF[3])
        c4 = np.float32(_CTP_COEF[4]); c5 = np.float32(_CTP_COEF[5])
        c6 = np.float32(_CTP_COEF[6]); c7 = np.float32(_CTP_COEF[7])
        pi_f = np.float32(PI_F)
        pi2_f = np.float32(PI_2_F)
        m = np.empty(n, dtype=np.float32)
        ph = np.empty(n, dtype=np.float32)
        for i in range(n):
            ar = np.float32(re[i]); ai = np.float32(im[i])
            are = np.float32(abs(ar)); aim = np.float32(abs(ai))
            if are > aim:
                mx = are
            else:
                mx = aim
            mx0 = mx
            if mx0 == 0.0:
                mx0 = np.float32(1.0)
            if aim > are:
                mn = are
            else:
                mn = aim
            r = np.float32(np.float64(mn) / np.float64(mx0))
            r2 = np.float32(np.float64(r) * np.float64(r))
            # poly: p = fma chain (cs kernels: _fma_arr semantics per element)
            p = _fma1(c0, r2, c1, bits, f32v, half)
            p = _fma1(p, r2, c2, bits, f32v, half)
            p = _fma1(p, r2, c3, bits, f32v, half)
            p = _fma1(p, r2, c4, bits, f32v, half)
            p = _fma1(p, r2, c5, bits, f32v, half)
            p = _fma1(p, r2, c6, bits, f32v, half)
            p = _fma1(p, r2, c7, bits, f32v, half)
            p = _fma1(p, r2, np.float32(1.0), bits, f32v, half)
            re2 = np.float32(np.float64(ar) * np.float64(ar))
            m[i] = np.float32(np.sqrt(np.float64(_fma1(ai, ai, re2, bits, f32v, half))))
            ang = np.float32(np.float64(r) * np.float64(p))
            if aim > are:
                ang = np.float32(np.float64(pi2_f) - np.float64(ang))
            if ar < 0.0:
                ang = np.float32(np.float64(pi_f) - np.float64(ang))
            if np.signbit(ai):
                ang = -ang
            ph[i] = ang
        return m, ph
else:
    _ctp_nb = None

if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _wrap_pi_nb(x):
        """Fused phase wrap: r = x - 2pi*rint(x*inv2pi) with fmaf semantics.

        -rint(x*inv2pi) is the 'a' of fma(a, 2pi, x): rint rounds to nearest-even
        on f32 (matches np.rint on the f32 product), product in f64 (exact for
        f32 inputs), then the TwoSum-emulated fma correction.
        """
        n = x.shape[0]
        out = np.empty(n, dtype=np.float32)
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        ib = np.zeros(1, dtype=np.uint32)
        if32 = ib.view(np.float32)
        ib[0] = 0x3E22F983
        inv2pi_f = if32[0]
        ib[0] = 0x40C90FDB
        two_pi_f = if32[0]
        for i in range(n):
            xf = np.float32(x[i])
            prod = np.float32(np.float64(xf) * np.float64(inv2pi_f))
            # np.rint on f32: round-half-to-even
            rv = np.rint(np.float64(prod))
            rint_f = np.float32(rv)
            ad = -np.float64(rint_f)
            bd = np.float64(two_pi_f)
            cd = np.float64(xf)
            p = ad * bd
            s = p + cd
            bb = s - p
            err = (p - (s - bb)) + (cd - bb)
            r = np.float32(s)
            e = (s - np.float64(r)) + err
            if np.isfinite(s):
                f32v[0] = r
                rb = bits[0]
                neg = (rb >> 31) != 0
                if neg:
                    bits[1] = rb - 1
                else:
                    bits[1] = rb + 1
                up = f32v[1]
                ulp = np.float64(up) - np.float64(r)
                if e > half * ulp:
                    r = up
                else:
                    if neg:
                        bits[1] = rb + 1
                    else:
                        bits[1] = rb - 1
                    dn = f32v[1]
                    dulp = np.float64(r) - np.float64(dn)
                    if e < -half * dulp:
                        r = dn
            else:
                r = np.float32(s)
            out[i] = r
        return out
else:
    _wrap_pi_nb = None



if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _oac_nb(out, win1, win2, synth_a, synth_b, R, N, pos0, v35, v36, wh,
                cursor, hop, v197, v198):
        """Fused overlap-add: t2/t window chain + ring accumulation (see _fma_arr)."""
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        wrap = cursor < hop or N + pos0 > R
        pcur = pos0
        if pcur < 0:
            pcur = 0
        fvi = 0
        if cursor - hop < 0:
            fvi = hop - cursor
        for i in range(N):
            t2 = np.float32(np.float64(v198) * np.float64(np.float32(
                np.float64(win2[wh + i]) * np.float64(synth_b[i]))))
            ad = np.float64(win1[wh + i]); bd = np.float64(synth_a[i]); cd = np.float64(t2)
            p = ad * bd
            s = p + cd
            bb = s - p
            err = (p - (s - bb)) + (cd - bb)
            r = np.float32(s)
            e = (s - np.float64(r)) + err
            if np.isfinite(s):
                f32v[0] = r
                rb = bits[0]
                neg = (rb >> 31) != 0
                if neg:
                    bits[1] = rb - 1
                else:
                    bits[1] = rb + 1
                up = f32v[1]
                ulp = np.float64(up) - np.float64(r)
                if e > half * ulp:
                    r = up
                else:
                    if neg:
                        bits[1] = rb + 1
                    else:
                        bits[1] = rb - 1
                    dn = f32v[1]
                    dulp = np.float64(r) - np.float64(dn)
                    if e < -half * dulp:
                        r = dn
            else:
                r = np.float32(s)
            t = r
            if wrap:
                if i < fvi:
                    continue
                pidx = (pcur + (i - fvi)) % R
                if i < v35:
                    ad = np.float64(t); bd = np.float64(v197); cd = np.float64(out[pidx])
                    p = ad * bd
                    s = p + cd
                    bb = s - p
                    err = (p - (s - bb)) + (cd - bb)
                    rr = np.float32(s)
                    e = (s - np.float64(rr)) + err
                    if np.isfinite(s):
                        f32v[0] = rr
                        rb = bits[0]
                        neg = (rb >> 31) != 0
                        if neg:
                            bits[1] = rb - 1
                        else:
                            bits[1] = rb + 1
                        up = f32v[1]
                        ulp = np.float64(up) - np.float64(rr)
                        if e > half * ulp:
                            rr = up
                        else:
                            if neg:
                                bits[1] = rb + 1
                            else:
                                bits[1] = rb - 1
                            dn = f32v[1]
                            dulp = np.float64(rr) - np.float64(dn)
                            if e < -half * dulp:
                                rr = dn
                    else:
                        rr = np.float32(s)
                    out[pidx] = rr
                elif i >= v36 and i >= v35:
                    out[pidx] = np.float32(np.float64(t) * np.float64(v197))
            else:
                pidx = pos0 + i
                if pidx >= R:
                    continue
                if i < v36:
                    ad = np.float64(t); bd = np.float64(v197); cd = np.float64(out[pidx])
                    p = ad * bd
                    s = p + cd
                    bb = s - p
                    err = (p - (s - bb)) + (cd - bb)
                    rr = np.float32(s)
                    e = (s - np.float64(rr)) + err
                    if np.isfinite(s):
                        f32v[0] = rr
                        rb = bits[0]
                        neg = (rb >> 31) != 0
                        if neg:
                            bits[1] = rb - 1
                        else:
                            bits[1] = rb + 1
                        up = f32v[1]
                        ulp = np.float64(up) - np.float64(rr)
                        if e > half * ulp:
                            rr = up
                        else:
                            if neg:
                                bits[1] = rb + 1
                            else:
                                bits[1] = rb - 1
                            dn = f32v[1]
                            dulp = np.float64(rr) - np.float64(dn)
                            if e < -half * dulp:
                                rr = dn
                    else:
                        rr = np.float32(s)
                    out[pidx] = rr
                else:
                    out[pidx] = np.float32(np.float64(t) * np.float64(v197))
        return out
else:
    _oac_nb = None



if _njit is not None:
    @_njit(cache=True, fastmath=False)
    def _fma_arr_nb(a, b, c):
        n = a.shape[0]
        out = np.empty(n, dtype=np.float32)
        bits = np.zeros(2, dtype=np.uint32)
        f32v = bits.view(np.float32)
        half = 0.5
        for i in range(n):
            ad = np.float64(a[i]); bd = np.float64(b[i]); cd = np.float64(c[i])
            p = ad * bd
            s = p + cd
            bb = s - p
            err = (p - (s - bb)) + (cd - bb)
            r = np.float32(s)
            e = (s - np.float64(r)) + err
            if np.isfinite(s):
                f32v[0] = r
                rb = bits[0]
                neg = (rb >> 31) != 0
                if neg:
                    bits[1] = rb - 1
                else:
                    bits[1] = rb + 1
                up = f32v[1]
                ulp = np.float64(up) - np.float64(r)
                if e > half * ulp:
                    r = up
                else:
                    if neg:
                        bits[1] = rb + 1
                    else:
                        bits[1] = rb - 1
                    dn = f32v[1]
                    dulp = np.float64(r) - np.float64(dn)
                    if e < -half * dulp:
                        r = dn
            else:
                r = np.float32(s)
            out[i] = r
        return out
else:
    _fma_arr_nb = None


def _rint(x):
    """C rintf / AArch64 FRINTI with the default round-to-nearest mode."""
    return F32(np.rint(float(x)))


def _round_i(x):
    """C `(int)(x + (x<0 ? -0.5f : 0.5f))` (fcvtzs truncation)."""
    x = float(x)
    return int(x + (-0.5 if x < 0.0 else 0.5))


def _wrap_pi(x):
    """fmsub wrap: phase - 2pi*rint(phase/2pi) (adjust_multiphase_diff.c)."""
    if _wrap_pi_nb is not None:
        scalar = np.isscalar(x) or (type(x) is np.ndarray and x.ndim == 0)
        if type(x) is np.ndarray and x.dtype == np.float32 and x.flags.c_contiguous:
            return _wrap_pi_nb(x)
        a = np.ascontiguousarray(x, np.float32)
        r = _wrap_pi_nb(a)
        if scalar:                           # scalar in -> scalar out (old numpy path)
            return r.reshape(-1)[0]
        return r
    return _fma_arr(-np.rint((np.asarray(x, np.float32) * INV_2PI_F).astype(np.float32)),
                    np.float64(TWO_PI_F), x)


# ===========================================================================
# fill_granule.c — FillGranule / FillGranuleWin
# ===========================================================================

def _floormod(d: int, cap: int) -> int:
    m = d % cap
    return m + cap if m < 0 else m


_fg_bandsum_nb = None
try:
    from numba import njit as _njit_fgb

    @_njit_fgb(cache=True, fastmath=False)
    def _fg_bandsum_nb(ring, acc, off, N, cap, nb):
        """mode=fg: sum bands with ring wraparound into acc[:N]."""
        for i in range(N):
            j = off + i
            if j >= cap:
                j -= cap
            s = ring[0, j]
            for k in range(1, nb):
                s = np.float32(s + ring[k, j])
            acc[i] = s
except Exception:
    _fg_bandsum_nb = None


def fill_granule(ring, win, acc, mode="fg", hop=0, n_write=0, n_bands=0, n5d0=0,
                 cursor=0, buffered=True, cap_scalar=0, ring_cap0_a=0,
                 ring_cap0_b=0, gain=1.0):
    """Port of rx_fill_granule / rx_fill_granule_win (modes fg / fgw / chain).

    ``ring`` is [n_bands, cap]; ``acc`` is the accumulator (length >= n_write).
    Returns the accumulator (f32, length n_write).
    """
    acc = np.asarray(acc, dtype=np.float32)
    hop = int(hop)
    N = int(n_write)
    nb = int(n_bands)
    cap = int(ring_cap0_a) + int(ring_cap0_b)
    d = int(cursor) - hop

    if mode == "chain":
        # DataTail1D::Reset (full n_fft zero) then FillGranule then FGWin x n_bands
        acc[:] = 0.0
        fill_granule(ring, win, acc, mode="fg", hop=hop, n_write=N, n_bands=nb,
                     n5d0=n5d0, cursor=cursor, buffered=buffered,
                     cap_scalar=cap_scalar, ring_cap0_a=ring_cap0_a,
                     ring_cap0_b=ring_cap0_b, gain=gain)
        fill_granule(ring, win, acc, mode="fgw", hop=hop, n_write=N, n_bands=nb,
                     n5d0=n5d0, cursor=cursor, buffered=buffered,
                     cap_scalar=cap_scalar, ring_cap0_a=ring_cap0_a,
                     ring_cap0_b=ring_cap0_b, gain=gain)
        return acc[:N]

    if mode == "fg":
        if buffered:
            off = _floormod(d, cap)
            _neon = _neon_data()
            # Both kernels write acc in place, so only take the C path when the
            # buffers really are C-contiguous f32 (guaranteed for the engine's
            # self.acc / self.ring, but not for arbitrary callers).
            if (_neon is not None
                    and acc.flags.c_contiguous and acc.dtype == np.float32
                    and ring.flags.c_contiguous and ring.dtype == np.float32):
                _neon.fg_bandsum_run(ring, acc, off, N, cap, nb)
                return acc[:N]
            if _fg_bandsum_nb is not None and _route_b():
                _fg_bandsum_nb(ring, acc, off, N, cap, nb)
                return acc[:N]
            if N + off <= cap:
                a = ring[0, off:off + N].copy()
                for k in range(1, nb):
                    a = (ring[k, off:off + N] + a).astype(np.float32)
                acc[:N] = a
            else:
                idx = (np.arange(N, dtype=np.int64) + off) % cap
                a = ring[0][idx].copy()
                for k in range(1, nb):
                    a = (ring[k][idx] + a).astype(np.float32)
                acc[:N] = a
            return acc[:N]
        if d >= 0 and int(cursor) < int(cap_scalar) - hop:
            if nb < 1 or N < 1:
                return acc[:N]
            a = ring[0, d:d + N].copy()
            for k in range(1, nb):
                a = (ring[k, d:d + N] + a).astype(np.float32)
            acc[:N] = a
            return acc[:N]
        if nb < 1 or N < 1:
            return acc[:N]
        idx = np.clip(np.arange(N, dtype=np.int64) + d, 0, int(cap_scalar) - 1)
        a = ring[0][idx].copy()
        for k in range(1, nb):
            a = (ring[k][idx] + a).astype(np.float32)
        acc[:N] = a
        return acc[:N]

    # ---- mode == "fgw": FillGranuleWin over all bands ----
    g = F32(gain)
    # Wing destination bases.  The C source is *not* self-consistent here (see the
    # fill_granule.c header note about the engine's head/tail mismatch):
    #   buffered branch      -> lower wing at acc + N5 - hop
    #   clamp branch         -> lower wing at acc + N5 - hop
    #   direct branch (NEON)  -> lower wing at acc + N - hop   <-- vector body
    #   direct branch (tail)  -> lower wing at acc + N5 - hop  (unreachable for
    #                            hop % 4 == 0, which is the case in the corpus)
    # So for the direct branch the lower wing targets n_write - hop; the corpus
    # (fill_granule/{direct,direct_gain}: changed segments [0,hop) and
    # [n_write-hop, n_write)) confirms the vector body is authoritative.
    N5 = int(n5d0)
    lo_direct = N - hop
    for band in range(nb):
        r = ring[band]
        w = win[band]
        if buffered:
            off = _floormod(d, cap)
            if N + off <= cap:
                if g == 1.0:
                    if _fgw_nb is not None:
                        _fgw_nb(ring, win, acc, band, off, hop, N5)
                        continue
                    acc[:hop] = _fma_arr(r[off + hop:off + 2 * hop], w[hop:2 * hop],
                                         acc[:hop])
                    acc[N5 - hop:N5] = _fma_arr(r[off:off + hop], w[:hop],
                                                acc[N5 - hop:N5])
                elif hop >= 1:
                    prod = (r[off + hop:off + 2 * hop] * w[hop:2 * hop]).astype(np.float32)
                    acc[:hop] = _fma_arr(prod, g, acc[:hop])
                    prod = (r[off:off + hop] * w[:hop]).astype(np.float32)
                    acc[N5 - hop:N5] = _fma_arr(prod, g, acc[N5 - hop:N5])
            elif hop >= 1:
                i = np.arange(hop, dtype=np.int64)
                i1 = off + hop + i
                i1 = np.where(i1 >= cap, i1 - cap, i1)
                i2 = off + i
                i2 = np.where(i2 >= cap, i2 - cap, i2)
                p1 = ((r[i1] * w[hop:2 * hop]).astype(np.float32) * g).astype(np.float32)
                p2 = ((r[i2] * w[:hop]).astype(np.float32) * g).astype(np.float32)
                acc[:hop] = (acc[:hop] + p1).astype(np.float32)
                acc[N5 - hop:N5] = (acc[N5 - hop:N5] + p2).astype(np.float32)
            continue
        if hop < 1:
            continue
        capm1 = int(cap_scalar) - 1
        if int(cursor) < hop or int(cursor) >= int(cap_scalar) - hop:
            i = np.arange(hop, dtype=np.int64)
            c1 = np.clip(int(cursor) + i, 0, capm1)
            c2 = np.clip(d + i, 0, capm1)
            p1 = ((r[c1] * w[hop:2 * hop]).astype(np.float32) * g).astype(np.float32)
            p2 = ((r[c2] * w[:hop]).astype(np.float32) * g).astype(np.float32)
            acc[:hop] = (acc[:hop] + p1).astype(np.float32)
            acc[N5 - hop:N5] = (acc[N5 - hop:N5] + p2).astype(np.float32)
            continue
        p1 = (r[int(cursor):int(cursor) + hop] * w[hop:2 * hop]).astype(np.float32)
        p2 = (r[d:d + hop] * w[:hop]).astype(np.float32)
        if g == 1.0:
            acc[:hop] = (acc[:hop] + p1).astype(np.float32)
            acc[lo_direct:lo_direct + hop] = (acc[lo_direct:lo_direct + hop] + p2).astype(np.float32)
        else:
            # the direct branch's scalar tail is a non-fused add-product:
            # acc += (r*w)*gain  (the fma shape is only used by the buffered and
            # clamp branches).  Mirroring this is required for bit-exactness —
            # corpus case fill_granule/direct_gain (gain=0.7) is exactly the
            # 1-2 ULP split between the two shapes.
            acc[:hop] = (acc[:hop] + (p1 * g).astype(np.float32)).astype(np.float32)
            acc[lo_direct:lo_direct + hop] = (
                acc[lo_direct:lo_direct + hop]
                + (p2 * g).astype(np.float32)).astype(np.float32)
    return acc[:N]


# ===========================================================================
# analyze_channel_spectrum.c
# ===========================================================================

def time_to_iir_a(tau, rate):
    """AudioProcessor::TimeToIirA @dtk 0x19EAB8 (f32)."""
    tau = F32(tau)
    if tau == 0.0:
        return F32(1.0)
    tr = F32(tau * F32(rate))
    inv = F32(F32(-1.0) / tr)
    return F32(F32(1.0) - F32(math.exp(float(inv))))


def _t2a_fast(tau, rate):
    """float-only time_to_iir_a (route B fast path; f32-round identical)."""
    tau = np.float32(tau)
    if tau == 0.0:
        return 1.0
    tr = np.float32(tau * np.float32(rate))
    inv = np.float32(np.float32(-1.0) / tr)
    return float(np.float32(np.float32(1.0) - np.float32(math.exp(float(inv)))))


def _f32f(x):
    """Round a python float to f32 (via struct; cache-free)."""
    return _struct.unpack("<f", _struct.pack("<f", x))[0]


_VDSP_F = None
_VDSP_F_TRIED = False


def _vdsp_fwd():
    global _VDSP_F, _VDSP_F_TRIED
    if not _VDSP_F_TRIED:
        _VDSP_F_TRIED = True
        try:
            from . import vdsp
            from . import vocoder_core as _vc
            _VDSP_F = (vdsp.fft_fwd_r if (vdsp.HAVE_VDSP and _vc.ROUTE_B_FAST)
                       else None)
        except Exception:
            _VDSP_F = None
    return _VDSP_F


def _fft_fwd(src, n):
    """transform::FFTFwd — cart packing [n+2] (pyradius.fft / rx_fft_fwd)."""
    v = _vdsp_fwd()
    if v is not None:
        return v(np.asarray(src, dtype=np.float32))
    return _fft_plan(int(n)).fwd(np.asarray(src, dtype=np.float32))


def acs_spectrum1(time, win, env, n_fft=0, n_bins=0, iir_a=1.0):
    """rx_acs_spectrum1: window multiply, FFT, cross-granule IIR envelope."""
    t_io = np.array(time, dtype=np.float32, copy=True)
    win = np.asarray(win, dtype=np.float32)
    env = np.array(env, dtype=np.float32, copy=True)
    n_fft = int(n_fft)
    n_bins = int(n_bins)
    n_win = min(t_io.size, win.size)
    t_io[:n_win] = (t_io[:n_win] * win[:n_win]).astype(np.float32)
    cart = _fft_fwd(t_io, n_fft)
    re = cart[0:2 * n_bins:2]
    im = cart[1:2 * n_bins:2]
    m = np.sqrt((re * re + im * im).astype(np.float32)).astype(np.float32)
    env[:n_bins] = _fma_arr((m - env[:n_bins]).astype(np.float32), F32(iir_a),
                            env[:n_bins])
    return {"t1_io": t_io, "t1_cart": cart, "t1_env": env}


def acs_spectrum2(time2, n_fft=0, n_polar=0, n_maxbin=0):
    """rx_acs_spectrum2: FFT, CartToPolar, Threshold(1e-12), tail zeroing."""
    n_fft = int(n_fft)
    n_polar = int(n_polar)
    n_maxbin = int(n_maxbin)
    cart = _fft_fwd(np.asarray(time2, dtype=np.float32), n_fft)
    mag = np.zeros(n_maxbin, dtype=np.float32)
    phase = np.zeros(n_maxbin, dtype=np.float32)
    ctp = cart_to_polar(cart)
    mag[:n_polar] = ctp["mag"][:n_polar]
    phase[:n_polar] = ctp["phase"][:n_polar]
    mag = threshold_lt_inplace(mag, F32(1e-12))
    # [n_polar, n_maxbin) is already zero (fresh buffers, as in the C harness)
    return {"t2_cart": cart, "t2_mag": mag, "t2_phase": phase}


#: CartToPolar Horner coefficients (mov/movk immediates @0x2BC58-0x2BCCC)
_CTP_COEF = tuple(_bits(u) for u in (0x3B390CCD, 0xBC82B80D, 0x3D2E19B6,
                                     0xBD995FFA, 0x3DD9CCF2, 0xBE116F9F,
                                     0x3E4CB9A7, 0xBEAAAA5D))


def cart_to_polar(cart):
    """dvaaccelerate CartToPolar (NEON polynomial atan2) — vectorised, same order.

    Returns dict with f32 ``mag`` / ``phase`` of length ``(len(cart)-1)//2``
    (the C reads pairs from the cart buffer; the caller sizes the output).
    """
    cart = np.asarray(cart, dtype=np.float32)
    # the C reads (re, im) pairs for k = 0..n-1 from a [N+2] cart buffer,
    # i.e. it consumes 2n = N+2 values (the last pair is bin N/2 + the pad slot)
    n = cart.size // 2
    re = np.ascontiguousarray(cart[0:2 * n:2])
    im = np.ascontiguousarray(cart[1:2 * n:2])
    if _route_b() and _ctp_fast_nb is not None and _ctp_fast_nb_ph is not None:
        m = _ctp_fast_nb(re, im)
        ph = _ctp_fast_nb_ph(re, im)
        return {"mag": m, "phase": ph}
    are = np.abs(re)
    aim = np.abs(im)
    mx = np.where(are > aim, are, aim)
    mx = np.where(mx == 0.0, F32(1.0), mx)
    mn = np.where(aim > are, are, aim)
    r = (mn / mx).astype(np.float32)
    r2 = (r * r).astype(np.float32)
    if _ctp_nb is not None:
        m, ph = _ctp_nb(re, im, n)
        return {"mag": m, "phase": ph}
    p = _fma_arr(_CTP_COEF[0], r2, _CTP_COEF[1])
    for c in _CTP_COEF[2:]:
        p = _fma_arr(p, r2, c)
    p = _fma_arr(p, r2, F32(1.0))
    m = np.sqrt(_fma_arr(im, im, (re * re).astype(np.float32))).astype(np.float32)
    ang = (r * p).astype(np.float32)
    ang = np.where(aim > are, (PI_2_F - ang).astype(np.float32), ang)
    ang = np.where(re < 0.0, (PI_F - ang).astype(np.float32), ang)
    ph = np.where(np.signbit(im), -ang, ang).astype(np.float32)
    return {"mag": m, "phase": ph}


def threshold_lt_inplace(v, thr):
    """Threshold_LT_InPlace (vDSP_vthr): lower clamp (not zeroing)."""
    v = np.asarray(v, dtype=np.float32)
    np.copyto(v, np.maximum(v, F32(thr)))
    return v


# ===========================================================================
# unwrap_phase.c
# ===========================================================================

def unwrap_phase(mask, mask_copy, mag_copy, reg_start, reg_end, reg_prev_peak,
                 reg_offset, scratch, phase, f1=0, f2=0, f3=0, max_bin=0,
                 copy_len=0):
    """rx_unwrap_phase @0x16B1C8 — region phase unwrap (vectorised per bin)."""
    mask = np.asarray(mask, dtype=np.float32)
    mcref = np.asarray(mask_copy, dtype=np.float32)
    mag = np.asarray(mag_copy, dtype=np.float32)
    reg_start = np.asarray(reg_start, dtype=np.int64)
    reg_end = np.asarray(reg_end, dtype=np.int64)
    reg_prev_peak = np.asarray(reg_prev_peak, dtype=np.int64)
    reg_offset = np.asarray(reg_offset, dtype=np.float32)
    scratch = np.array(scratch, dtype=np.float32, copy=True)
    phase = np.array(phase, dtype=np.float32, copy=True)
    nreg = reg_start.size
    copy_len = int(copy_len)
    max_bin = int(max_bin)
    mb = max_bin - 1

    f1f = F32(f1)
    inv_f1 = F32(F32(1.0) / f1f)
    v9 = F32(F32(F32(F32(f1f * TWO_PI_F) / F32(f3)) * F32(0.5)))
    f2f = F32(f2)

    if _unwrap_nb is not None and _route_b():
        F = np.float32
        got = _unwrap_nb(mask, mcref, mag, reg_start, reg_end, reg_prev_peak,
                         reg_offset, scratch, phase, f1f, F(f3), f2f, inv_f1, v9,
                         nreg, copy_len, max_bin)
        return {"scratch": scratch, "phase": got}
    if nreg >= 1:
        # ---- per-region peak (MaxIndex, first strict maximum) ----
        # The C scans [reg_start, reg_end] inclusive for each region; a region
        # with reg_end < reg_start runs no iterations at all (regions produced by
        # ev_find_peaks are never empty, but unwrap's contract allows it and the
        # cnt==0 fallback path depends on the bound being honoured).
        lens = np.maximum(reg_end - reg_start + 1, 0)
        live = lens > 0
        # work with a >=1 repeat, then drop the dummy slot of empty regions
        rep = np.where(live, lens, 1)
        region_of = np.repeat(np.arange(nreg, dtype=np.int64), rep)
        posall = np.clip(
            np.arange(int(rep.sum()), dtype=np.int64)
            + np.repeat(reg_start - np.concatenate([[0], np.cumsum(rep)[:-1]]), rep),
            0, mag.size - 1)
        valid = np.repeat(live, rep)
        vals = mag[posall][valid]
        region_live = region_of[valid]
        mg = np.full(nreg, -np.inf, dtype=np.float32)
        np.maximum.at(mg, region_live, vals)
        # first index attaining each region max (strict max, lowest index wins)
        eq = np.flatnonzero(vals == np.repeat(mg[live], lens[live]))
        pos_live = posall[valid]
        got = region_live[eq]
        keep = np.concatenate([[True], np.diff(got) != 0])
        first = np.full(nreg, -1, dtype=np.int64)
        first[got[keep]] = pos_live[eq[keep]]
        peak = np.where(first >= 0, first, reg_start)
        v16 = peak.astype(np.float32)

        inner = (peak != 0) & (peak != mb)
        if inner.any():
            p = peak[inner]
            ap = mag[p + 1].astype(np.float32)
            am = mag[p].astype(np.float32)
            amm = mag[p - 1].astype(np.float32)
            v20 = ((ap - (F32(2.0) * am).astype(np.float32)).astype(np.float32) - amm).astype(np.float32)
            v21 = v16[inner].copy()
            nz = v20 != 0.0
            v21[nz] = (((ap[nz] - amm[nz]).astype(np.float32)
                        / (v20[nz] + v20[nz]).astype(np.float32)).astype(np.float32)
                       + v16[inner][nz]).astype(np.float32)
            t = (v16[inner] + K07_F).astype(np.float32)
            t = np.where(t < v21, t, v21)
            lo = (v16[inner] - K07_F).astype(np.float32)
            t = np.where(t < lo, lo, t)
            v16 = v16.copy()
            v16[inner] = t

        v37 = (v9 * (v16 + reg_offset[:nreg]).astype(np.float32)).astype(np.float32)
        lens2 = reg_end - reg_start
        nz2 = lens2 > 0
        if nz2.any():
            s = reg_start[nz2]
            Lb = lens2[nz2]
            bins = np.repeat(s, Lb) + (np.arange(int(Lb.sum()), dtype=np.int64)
                                       - np.repeat(np.cumsum(Lb) - Lb, Lb))
            dlt = np.repeat((peak - reg_prev_peak)[nz2], Lb)
            v37b = np.repeat(v37[nz2], Lb)
            idx = np.clip(bins + dlt, 0, mb)
            v38 = ((mask[bins] - mcref[idx]).astype(np.float32) - v37b).astype(np.float32)
            wrap = _fma_arr(np.rint((v38 * INV_2PI_F).astype(np.float32)),
                            np.float64(NEG_2PI_F), v38)
            scratch[bins] = _fma_arr(f2f, (inv_f1 * (v37b + wrap).astype(np.float32)).astype(np.float32),
                                     phase[idx])
    phase[:copy_len] = scratch[:copy_len]
    return {"scratch": scratch, "phase": phase}


# ===========================================================================
# apply_pitch_coherence.c
# ===========================================================================

#: the six 5-entry per-segment tables (@0x217B28, stride 0x14)
_APC_T_LO = (F32(1.5), F32(1.35), F32(1.07000005), F32(1.03999996), F32(1.02999997))
_APC_T_HI = (F32(1.89999998), F32(1.70000005), F32(1.20000005), F32(1.12), F32(1.10000002))
_APC_T_GMIX = (F32(0.69999999), F32(0.69999999), F32(0.60000002), F32(0.40000001), F32(0.2))
_APC_T_LIN = (F32(0.60000002), F32(0.40000001), F32(0.2), F32(0.15000001), F32(0.1))
_APC_T_EA = (F32(1.60000002), F32(1.70000005), F32(0.40000001), F32(0.51999998), F32(0.55000001))
_APC_T_EB = (F32(2.5), F32(2.0), F32(0.80000001), F32(0.55000001), F32(0.55000001))


_apc_full_nb = None
try:
    from numba import njit as _njit_apc

    @_njit_apc(cache=True, fastmath=False)
    def _apc_full_nb(phase, phase_mod, dir_cur, dir_prev, env,
                     peak_bins, reg_start, reg_end, seg_bound,
                     region_gain, noise_wt,
                     apc_t_lo, apc_t_hi, apc_t_gmix, apc_t_lin,
                     apc_t_ea, apc_t_eb,
                     v22_init, v133, v23, v24u, v130, v135, v136,
                     coh_center, a4f, nseg, peak_cnt, transient_state,
                     sr):
        """Full rx_apply_pitch_coherence scalar path (route B).

        Single-kernel port of the seg/peak loops with np.float32 rounds at
        the same places as the python F32() chain.  Returns the updated v22.
        """
        PI = np.float32(3.1415927410125732)
        INV_2PI = np.float32(0.15915494)
        NEG_2PI = np.float32(-6.2831854820251465)
        v22 = v22_init
        cursor = 0
        for s in range(nseg):
            c0 = coh_center
            v27 = np.float32(math.pow(float(c0), 1.7)
                             + float(np.float32(np.float32(c0 * np.float32(1.5))
                                                * np.float32(a4f - np.float32(0.5)))))
            if v27 > 1.0:
                v27 = np.float32(1.0)
            e = env[s]
            lo = apc_t_lo[s]
            hi = apc_t_hi[s]
            v31 = np.float32(np.float32(e - lo) / np.float32(hi - lo))
            if not (e < hi):
                v31 = np.float32(1.0)
            ramp = v31 if (e > lo) else np.float32(0.0)
            gm = apc_t_gmix[s]
            v33 = np.float32(np.float32(np.float32(1.0) - gm)
                             * np.float32(ramp + np.float32(np.float32(v27 - ramp)
                                                            * apc_t_lin[s]))
                             + np.float32(gm * np.float32(math.sqrt(float(np.float32(v27 * ramp))))))
            if v33 > 1.0:
                v33 = np.float32(1.0)
            if v33 < 0.0:
                v33 = np.float32(0.0)
            wLin = np.float32(math.pow(float(v33),
                                       float(np.float32(apc_t_ea[s] / v133))))
            if v23 < wLin:
                wLin = v23
            if wLin < 0.0:
                wLin = np.float32(0.0)
            wExp = np.float32(math.pow(float(v33),
                                       float(np.float32(apc_t_eb[s] / v133))))
            if v23 < wExp:
                wExp = v23
            if wExp < 0.0:
                wExp = np.float32(0.0)

            if cursor >= peak_cnt:
                continue
            v40 = np.float32(wExp * PI)
            bound = seg_bound[s + 1]
            while cursor < peak_cnt and peak_bins[cursor] < bound:
                p = cursor
                cursor += 1
                bin_ = peak_bins[p]
                if (bin_ & 0xFFFFFFFF) < v24u:
                    continue
                bf = np.float32(bin_)
                h = int(np.rint(np.float32(bf / v22)))
                thr = v40
                if h != 0:
                    center = np.float32(v22 * np.float32(h))
                    outside = ((center - np.float32(2.0)) >= bf) or ((center + np.float32(2.0)) <= bf)
                    if not outside:
                        ratio = np.float32(bf / np.float32(h))
                        v22 = np.float32(ratio + np.float32(np.float32(v136 - ratio)
                                                            / np.float32(math.sqrt(math.sqrt(float(np.float32(h)))))))
                        dd = np.float32(abs(float(np.float32(bf - center))))
                        v53 = np.float32(np.float32(dd - np.float32(0.7)) / np.float32(1.3))
                        if dd >= 2.0:
                            v53 = np.float32(1.0)
                        v54 = v53 if (dd > 0.7) else np.float32(0.0)
                        thr = np.float32(np.float32(wLin * PI)
                                         + np.float32(np.float32(v40 - np.float32(wLin * PI)) * v54))
                dph = np.float32(np.float32(phase[bin_] + np.float32(bf * v130)) - phase_mod[bin_])
                wrapped = np.float32(dph + np.float32(np.rint(np.float32(dph * INV_2PI)) * NEG_2PI))
                dp = dir_prev[bin_]
                dc = dir_cur[bin_]
                if ((dp > dc) and (wrapped > 0.0) and (dc > wrapped)) or \
                   ((dp < dc) and (wrapped < 0.0) and (dc < wrapped)):
                    v64 = np.float32(dp / wrapped)
                    v65 = np.float32(0.0)
                    if v64 > 1.0:
                        v65 = np.float32(1.0)
                        if v64 < 4.0:
                            v65 = np.float32(np.float32(v64 - np.float32(1.0)) / np.float32(3.0))
                    thr = np.float32(thr * np.float32(np.float32(v65 * np.float32(0.75)) + np.float32(0.25)))
                if dp == 0.0 and dc == 0.0:
                    m = np.float32(PI - thr)
                    if thr < m:
                        m = thr
                    thr = np.float32(thr + np.float32(m * np.float32(0.2)))
                else:
                    m = np.float32(PI - thr)
                    if thr < m:
                        m = thr
                    t68 = np.float32(thr + np.float32(m * np.float32(0.15)))
                    if dc == 0.0:
                        thr = t68
                rg = region_gain[bin_]
                v71 = np.float32(0.0)
                if rg > 2.0:
                    v71 = np.float32(1.0)
                    if rg < 6.0:
                        v71 = np.float32(np.float32(rg - np.float32(2.0)) * np.float32(0.25))
                nw = noise_wt[bin_]
                v70 = np.float32(0.0)
                if nw > 0.4:
                    v70 = np.float32(1.0)
                    if nw < 0.7:
                        v70 = np.float32(np.float32(nw - np.float32(0.4)) / np.float32(0.3))
                if (np.float32(np.float32(np.float32(1.0) - v71) + v70) > 1.0
                        and v135 < np.float32((sr * bin_) & 0xFFFFFFFF)
                        and transient_state != 2):
                    thr = np.float32(10.0)
                rrs = reg_start[p]
                ree = reg_end[p]
                if rrs < ree:
                    if abs(float(wrapped)) > float(thr):
                        dir_prev[rrs:ree] = dir_cur[rrs:ree]
                        dir_cur[rrs:ree] = wrapped
                    else:
                        for i in range(rrs, ree):
                            b = np.float32(i)
                            dd = (phase[i] + (b * v130)) - phase_mod[i]
                            nn = np.rint(dd * INV_2PI)
                            phase_mod[i] = phase_mod[i] + (dd + (nn * NEG_2PI))
                            dir_prev[i] = dir_cur[i]
                            dir_cur[i] = np.float32(0.0)
        return v22
except Exception:
    _apc_full_nb = None


def apply_pitch_coherence(args, env, peak_bins, reg_start, reg_end, seg_bound,
                          phase, phase_mod, dir_cur, dir_prev, region_gain,
                          noise_wt, a3=0.0, a4=0.0):
    """rx_apply_pitch_coherence @0x16B444 (scalar path, A-VECT-1 knob honoured)."""
    precision = int(args["precision"])
    trans_sens = F32(args["trans_sens"])
    phase_mod = np.array(phase_mod, dtype=np.float32, copy=True)
    dir_cur = np.array(dir_cur, dtype=np.float32, copy=True)
    dir_prev = np.array(dir_prev, dtype=np.float32, copy=True)
    advance = F32(-12345.0)

    if precision > 9 or trans_sens == 0.0:
        return {"phase_mod": phase_mod, "dir_cur": dir_cur,
                "dir_prev": dir_prev, "advance": advance}

    total_ratio = float(args["total_ratio"])
    r = total_ratio
    inv = 1.0 / r
    if r < inv:
        r = inv
    v8 = F32(r)
    v13 = F32(math.exp(float(F32(F32(v8 - F32(1.0)) * F32(-5.0)))))
    transient_state = int(args["transient_state"])
    v15 = F32(0.5) if transient_state == 2 else F32(1.0)
    N = int(args["n_fft"])
    a3f = F32(a3)
    a4f = F32(a4)
    f580 = int(args["f580"])
    f584 = int(args["f584"])
    acc_fc0 = F32(args["acc_fc0"])

    # phase advance (written through *out_advance by the C)
    v17 = F32(F32(acc_fc0 + F32(f580 - f584)) / a3f)
    advance = F32(a3f * F32(v17 - _rint(v17)))

    nseg = int(args["seg_count"])
    if nseg == 0:
        return {"phase_mod": phase_mod, "dir_cur": dir_cur,
                "dir_prev": dir_prev, "advance": advance}

    v22 = F32(F32(N) / a3f)
    v133 = F32(v15 * F32(F32(trans_sens * F32(1.4)) * F32(v13 + F32(1.0))))
    v23 = F32(max(float(F32(F32(1.0) - F32(F32(0.2) / v133))), 0.2))
    arg = F32(F32(F32(N) * F32(0.9)) / a3f)
    if np.isfinite(float(arg)):
        iv = int(np.rint(float(arg)))
        iv = max(min(iv, (1 << 63) - 1), -(1 << 63))
        v24u = iv & 0xFFFFFFFF           # (int)(int64_t)rintf + unsigned truncation
    else:
        v24u = 0xFFFFFFFF                # fcvtzs saturates to INT64_MAX -> -1
    env = np.asarray(env, dtype=np.float32)
    peak_bins = np.asarray(peak_bins, dtype=np.int64)
    reg_start = np.asarray(reg_start, dtype=np.int64)
    reg_end = np.asarray(reg_end, dtype=np.int64)
    seg_bound = np.asarray(seg_bound, dtype=np.int64)
    phase = np.asarray(phase, dtype=np.float32)
    region_gain = np.asarray(region_gain, dtype=np.float32)
    noise_wt = np.asarray(noise_wt, dtype=np.float32)
    peak_cnt = int(args["peak_count"])
    vec = bool(args["vector_fmaf_region"])
    sr = int(args["sr"])
    coh_center = F32(args["coh_center"])
    v136 = v22
    v130 = F32(F32(F32(advance * F32(-2.0)) * PI_F) / F32(N))
    v135 = F32(F32(N) * F32(4000.0))

    if (_apc_full_nb is not None and _route_b()
            and int(args["precision"]) <= 9 and F32(args["trans_sens"]) != 0.0):
        _apc_full_nb(phase, phase_mod, dir_cur, dir_prev, env,
                     peak_bins, reg_start, reg_end, seg_bound,
                     region_gain, noise_wt,
                     _APC_T_LO, _APC_T_HI, _APC_T_GMIX, _APC_T_LIN,
                     _APC_T_EA, _APC_T_EB,
                     v22, v133, v23, v24u, v130, v135, v136,
                     coh_center, a4f, nseg, peak_cnt, transient_state,
                     sr)
        return {"phase_mod": phase_mod, "dir_cur": dir_cur,
                "dir_prev": dir_prev, "advance": advance}

    cursor = 0
    for s in range(nseg):
        c0 = coh_center
        v27 = F32(math.pow(float(c0), 1.7) + float(F32(F32(c0 * F32(1.5)) * F32(a4f - F32(0.5)))))
        if v27 > 1.0:
            v27 = F32(1.0)
        e = F32(env[s])
        lo = _APC_T_LO[s]
        hi = _APC_T_HI[s]
        v31 = F32(F32(e - lo) / F32(hi - lo))
        if not (e < hi):
            v31 = F32(1.0)
        ramp = v31 if (e > lo) else F32(0.0)
        gm = _APC_T_GMIX[s]
        v33 = F32(F32(F32(1.0) - gm) * F32(ramp + F32(F32(v27 - ramp) * _APC_T_LIN[s]))
                  + F32(gm * F32(math.sqrt(float(F32(v27 * ramp))))))
        if v33 > 1.0:
            v33 = F32(1.0)
        if v33 < 0.0:
            v33 = F32(0.0)
        wLin = F32(math.pow(float(v33), float(F32(_APC_T_EA[s] / v133))))
        if v23 < wLin:
            wLin = v23
        if wLin < 0.0:
            wLin = F32(0.0)
        wExp = F32(math.pow(float(v33), float(F32(_APC_T_EB[s] / v133))))
        if v23 < wExp:
            wExp = v23
        if wExp < 0.0:
            wExp = F32(0.0)

        if cursor >= peak_cnt:
            continue
        v40 = F32(wExp * PI_F)
        bound = int(seg_bound[s + 1])
        while cursor < peak_cnt and int(peak_bins[cursor]) < bound:
            p = cursor
            cursor += 1
            bin_ = int(peak_bins[p])
            if (bin_ & 0xFFFFFFFF) < v24u:
                continue
            bf = F32(bin_)
            h = int(_rint(F32(bf / v22)))
            thr = v40
            if h != 0:
                center = F32(v22 * F32(h))
                outside = ((center - F32(2.0)) >= bf) or ((center + F32(2.0)) <= bf)
                if not outside:
                    ratio = F32(bf / F32(h))
                    v22 = F32(ratio + F32(F32(v136 - ratio)
                                          / F32(math.sqrt(math.sqrt(float(F32(h)))))))
                    dd = F32(abs(float(F32(bf - center))))
                    v53 = F32(F32(dd - F32(0.7)) / F32(1.3))
                    if dd >= 2.0:
                        v53 = F32(1.0)
                    v54 = v53 if (dd > 0.7) else F32(0.0)
                    thr = F32(F32(wLin * PI_F) + F32(F32(v40 - F32(wLin * PI_F)) * v54))
            dph = F32(F32(phase[bin_] + F32(bf * v130)) - phase_mod[bin_])
            wrapped = F32(dph + F32(_rint(F32(dph * INV_2PI_F)) * NEG_2PI_F))
            dp = F32(dir_prev[bin_])
            dc = F32(dir_cur[bin_])
            if ((dp > dc) and (wrapped > 0.0) and (dc > wrapped)) or \
               ((dp < dc) and (wrapped < 0.0) and (dc < wrapped)):
                v64 = F32(dp / wrapped)
                v65 = F32(0.0)
                if v64 > 1.0:
                    v65 = F32(1.0)
                    if v64 < 4.0:
                        v65 = F32(F32(v64 - F32(1.0)) / F32(3.0))
                thr = F32(thr * F32(F32(v65 * F32(0.75)) + F32(0.25)))
            if dp == 0.0 and dc == 0.0:
                m = F32(PI_F - thr)
                if thr < m:
                    m = thr
                thr = F32(thr + F32(m * F32(0.2)))
            else:
                m = F32(PI_F - thr)
                if thr < m:
                    m = thr
                t68 = F32(thr + F32(m * F32(0.15)))
                if dc == 0.0:
                    thr = t68
            rg = F32(region_gain[bin_])
            v71 = F32(0.0)
            if rg > 2.0:
                v71 = F32(1.0)
                if rg < 6.0:
                    v71 = F32(F32(rg - F32(2.0)) * F32(0.25))
            nw = F32(noise_wt[bin_])
            v70 = F32(0.0)
            if nw > 0.4:
                v70 = F32(1.0)
                if nw < 0.7:
                    v70 = F32(F32(nw - F32(0.4)) / F32(0.3))
            if (F32(F32(F32(1.0) - v71) + v70) > 1.0
                    and v135 < F32((sr * bin_) & 0xFFFFFFFF)
                    and transient_state != 2):
                thr = F32(10.0)
            rrs, ree = int(reg_start[p]), int(reg_end[p])
            if rrs < ree:
                if abs(float(wrapped)) > float(thr):
                    dir_prev[rrs:ree] = dir_cur[rrs:ree]
                    dir_cur[rrs:ree] = wrapped
                else:
                    if _apc_region_nb is not None:
                        _apc_region_nb(phase, phase_mod, dir_prev, dir_cur,
                                       rrs, ree, v130)
                    else:
                        b = np.arange(rrs, ree, dtype=np.float32)
                        pm = phase_mod[rrs:ree]
                        if vec:
                            dd = (_fma_arr(b, np.float64(v130), phase[rrs:ree]) - pm).astype(np.float32)
                        else:
                            dd = ((phase[rrs:ree] + (b * v130).astype(np.float32)).astype(np.float32)
                                  - pm).astype(np.float32)
                        nn = np.rint((dd * INV_2PI_F).astype(np.float32))
                        if vec:
                            phase_mod[rrs:ree] = (pm + _fma_arr(nn, np.float64(NEG_2PI_F), dd)).astype(np.float32)
                        else:
                            phase_mod[rrs:ree] = (pm + (dd + (nn * NEG_2PI_F).astype(np.float32)).astype(np.float32)).astype(np.float32)
                        dir_prev[rrs:ree] = dir_cur[rrs:ree]
                        dir_cur[rrs:ree] = 0.0
    return {"phase_mod": phase_mod, "dir_cur": dir_cur, "dir_prev": dir_prev,
            "advance": advance}


# ===========================================================================
# reset_phases.c
# ===========================================================================

#: Σa² block accumulation order @0x16d588 (D1 in the C source)
_RPT_A2ORD = (0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 11, 15, 12, 13, 14, 10)


def reset_phases_for_transients(mode_4b8=0, proc_mode_9b8=0, scale_20=0.0,
                                len_594=0, n9bc=0, n9c0=0, u_70=0, n578=0,
                                n590=0, n5d0=0, n5d4=0, avg_m1=0.0, avg_0=0.0,
                                avg_p1=0.0, use_p568=False, mag=None, b_710=None,
                                mask=None, r_start=None, r_end=None, r_bin=None,
                                phase=None, mask_table=None, n_out=0):
    """rx_reset_phases_for_transients @0x16D354 (three branches)."""
    phase = np.array(phase, dtype=np.float32, copy=True)
    mask_table = np.array(mask_table, dtype=np.float32, copy=True)
    e08 = np.float64(0.0)
    n_out = int(n_out)

    # ---- [B1] this[0x4B8] == 1 ----
    if int(mode_4b8) == 1:
        n = int(len_594)
        if n >= 1:
            phase[:n] = (mask[:n].astype(np.float64) * float(scale_20)).astype(np.float32)
        return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}

    # ---- [B2] this[0x9B8] == 2 ----
    if int(proc_mode_9b8) == 2:
        v12 = F32(math.sqrt(float(F32(avg_0))))
        v13 = F32(0.0)
        if v12 > 0.8:
            v13 = F32(1.0)
            if v12 < 1.5:
                v13 = F32(F32(v12 - F32(0.8)) / F32(0.7))
        v14 = _fma(v13, F32(0.7), F32(0.5))
        if int(n9c0) < int(n9bc):
            v14 = F32(v14 * F32(math.sqrt(float(F32(n9bc) / F32(n9c0)))))
        r_count = 0 if r_start is None else np.asarray(r_start).size
        mag = np.asarray(mag, dtype=np.float32)
        b_710 = np.asarray(b_710, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        r_start = np.asarray(r_start, dtype=np.int64)
        r_end = np.asarray(r_end, dtype=np.int64)
        r_bin = np.asarray(r_bin, dtype=np.int64)
        if r_count < 1:
            return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}
        for r in range(r_count):
            s_, e_ = int(r_start[r]), int(r_end[r])
            sa2 = F32(0.0)
            sb2 = F32(0.0)
            sa = F32(0.0)
            if s_ < e_:
                n = e_ - s_
                i = 0
                while i + 16 <= n:
                    a = mag[s_ + i:s_ + i + 16].astype(np.float32)
                    b = b_710[s_ + i:s_ + i + 16].astype(np.float32)
                    a2 = (a * a).astype(np.float32)
                    b2 = (b * b).astype(np.float32)
                    for k in range(16):
                        sa2 = F32(sa2 + a2[_RPT_A2ORD[k]])
                    for k in range(16):
                        sb2 = F32(sb2 + b2[k])
                    for k in range(16):
                        sa = F32(sa + a[k])
                    i += 16
                while i + 4 <= (n & ~3):
                    for k in range(4):
                        av = F32(mag[s_ + i + k])
                        bv = F32(b_710[s_ + i + k])
                        sa2 = F32(sa2 + F32(av * av))
                        sb2 = F32(sb2 + F32(bv * bv))
                        sa = F32(sa + av)
                    i += 4
                while i < n:
                    av = F32(mag[s_ + i])
                    bv = F32(b_710[s_ + i])
                    sa2 = F32(sa2 + F32(av * av))
                    sb2 = F32(sb2 + F32(bv * bv))
                    sa = F32(sa + av)
                    i += 1
            sa = F32(sa + F32(1e-9))
            v88 = F32(F32(F32(F32(F32(r_bin[r]) * F32(u_70)) / F32(n5d0))
                          + F32(-1500.0)) / F32(2000.0))
            v88 = _fma(v88, F32(-23.0), F32(25.0))
            if v88 > 25.0:
                v88 = F32(25.0)
            if v88 < 2.0:
                v88 = F32(2.0)
            v89 = F32(v88 / F32(n590))
            v27 = F32(0.0)
            thr = F32(sb2 * F32(v89 + F32(1.0)))
            if sa2 > thr:
                base = _fma(v89, F32(1.5), F32(1.0))
                v91 = _fma(base, sb2, F32(1e-9))
                v27 = F32(1.0)
                if sa2 < v91:
                    v27 = F32(F32(sa2 - thr) / F32(v91 - thr))
            v92 = F32(F32(math.sqrt(float(F32(sa2 * F32(e_ - s_))))) / sa)
            v93 = F32(0.0)
            if v92 > 1.0:
                v93 = F32(1.0)
                if v92 < 1.5:
                    v93 = F32(F32(v92 - F32(1.0)) + F32(v92 - F32(1.0)))
            score = _fma(F32(v14 * F32(F32(1.0) - v93)), F32(0.5), F32(v14 * v27))
            if score <= 0.75:
                if s_ < e_:
                    if use_p568:
                        idx = np.nonzero(mask_table[s_:e_] != 0.0)[0] + s_
                    else:
                        idx = np.arange(s_, e_)
                    phase[idx] = mask[idx]
                    mask_table[idx] = F32(1.0)
            elif e_ > s_:
                phase[s_:e_] = mask[s_:e_]
                mask_table[s_:e_] = F32(1.0)
        return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}

    # ---- [B3] else ----
    a_m1, a_0, a_p1 = F32(avg_m1), F32(avg_0), F32(avg_p1)
    if a_m1 >= a_0:
        if a_0 <= 2.0:
            return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}
    else:
        w8 = (a_0 > a_p1) and (a_0 > 1.2)
        if a_0 <= 2.0 and not w8:
            return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}
    u70f = F32(u_70)
    lim = F32(u70f * _bits(0x3B449BA6))
    if lim >= F32(float(int(e08))):
        return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}
    v130 = F32(F32(F32(n5d0) * F32(1500.0)) / u70f)
    half = F32(0.5) if v130 >= 0.0 else F32(-0.5)
    t = F32(v130 + half)
    start = 0 if t < 0.0 else int(t)
    if int(n5d4) <= start:
        return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}
    mask = np.asarray(mask, dtype=np.float32)
    phase[start:int(n5d4)] = mask[start:int(n5d4)]
    return {"phase": phase[:n_out], "mask_table": mask_table[:n_out]}


# ===========================================================================
# synchronize_stereo_phases.c / adjust_multiphase_diff.c
# ===========================================================================

def adjust_multiphase_diff(src, dst, src_bin=0, dst_bin=0, w=0.0, nch=2):
    """rx_adjust_multiphase_diff @0x16BED4 (nch==2 main path + generic path)."""
    src = np.asarray(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32, copy=True)
    w = F32(w)
    nch = int(nch)
    if nch == 2:
        s0 = F32(src[0][src_bin])
        s1 = F32(src[1][src_bin])
        d0 = F32(dst[0][dst_bin])
        d1 = F32(dst[1][dst_bin])
        a0 = _wrap_pi(F32(d0 - s0))
        a1 = _wrap_pi(F32(d1 - s1))
        m = F32(F32(a0 + a1) * F32(0.5))
        if not (abs(float(F32(a1 - a0))) < float(PI_F)):
            m = F32(m + PI_F)
        m = _wrap_pi(m)
        t0 = _wrap_pi(F32(F32(s0 + m) - d0))
        dst[0][dst_bin] = _wrap_pi(_fma(w, t0, d0))
        t1 = _wrap_pi(F32(F32(s1 + m) - d1))
        dst[1][dst_bin] = _wrap_pi(_fma(w, t1, d1))
        return dst
    if nch <= 0:
        return dst
    # generic (nch > 2) circular-mean rotation search
    inv = F32(F32(1.0) / F32(nch))
    step = F32(inv * TWO_PI_F)
    scratch_f = np.zeros(nch, dtype=np.float32)
    scratch_i = np.zeros(nch, dtype=np.int32)
    ssum = F32(0.0)
    for ch in range(nch):
        d = _wrap_pi(F32(dst[ch][dst_bin]))
        dst[ch][dst_bin] = d
        scratch_f[ch] = _wrap_pi(F32(d - F32(src[ch][src_bin])))
        ssum = F32(ssum + scratch_f[ch])
    mean = F32(inv * ssum)
    best = F32(1000000.0)
    bestk = 0
    for k in range(nch):
        arg = int(np.argmax(scratch_f))
        scratch_i[arg] = k
        scratch_f[arg] = F32(scratch_f[arg] + TWO_PI_F)
        mean = _fma(inv, TWO_PI_F, mean)
        disp = F32(0.0)
        for ch in range(nch):
            t = F32(scratch_f[ch] - mean)
            tw = _fma(_rint(F32(t * INV_2PI_F)), -TWO_PI_F, t)
            disp = F32(disp + F32(tw * tw))
        if best > disp:
            best = disp
            bestk = k
    for ch in range(nch):
        if scratch_i[ch] > bestk:
            scratch_f[ch] = F32(scratch_f[ch] - TWO_PI_F)
            mean = F32(mean - step)
    m = _wrap_pi(mean)
    for ch in range(nch):
        d = F32(dst[ch][dst_bin])
        t = _wrap_pi(F32(F32(m + F32(src[ch][src_bin])) - d))
        dst[ch][dst_bin] = _wrap_pi(_fma(w, t, d))
    return dst


def ampd_pull_to_peak(pm, mask_ch, bin=0, peak_bin=0, w=0.0):
    """rx_ampd_pull_to_peak @0x168B0C (in-place on ``pm``)."""
    pm = np.array(pm, dtype=np.float32, copy=True)
    pm_bin = F32(pm[bin])
    pm_pk = F32(pm[peak_bin])
    t0 = _wrap_pi(F32(pm_pk - pm_bin))
    t1 = _wrap_pi(F32(mask_ch[bin] - mask_ch[peak_bin]))
    s = _wrap_pi(F32(t0 + t1))
    pm[bin] = _fma(F32(w), s, pm_bin)
    return pm


def _adjust_diff_nch2_vec(src0, src1, dst0, dst1, w):
    """Vectorised nch==2 body of rx_adjust_multiphase_diff (elementwise per bin).

    Identical arithmetic to the nch==2 branch of :func:`adjust_multiphase_diff`
    with a scalar `w`: the C calls that function once per bin inside a loop, this
    runs the whole slice at once.  Safe because the peak regions handed in by
    ev_find_peaks are disjoint, so the per-bin loop never observes its own
    earlier writes.
    """
    a0 = _wrap_pi((dst0 - src0).astype(np.float32))
    a1 = _wrap_pi((dst1 - src1).astype(np.float32))
    m = ((a0 + a1).astype(np.float32) * F32(0.5)).astype(np.float32)
    flip = ~(np.abs((a1 - a0).astype(np.float32)) < PI_F)
    m = np.where(flip, (m + PI_F).astype(np.float32), m).astype(np.float32)
    m = _wrap_pi(m)
    t0 = _wrap_pi(((src0 + m).astype(np.float32) - dst0).astype(np.float32))
    o0 = _wrap_pi(_fma_arr(F32(w), t0, dst0))
    t1 = _wrap_pi(((src1 + m).astype(np.float32) - dst1).astype(np.float32))
    o1 = _wrap_pi(_fma_arr(F32(w), t1, dst1))
    return o0, o1


def synchronize_stereo_phases_nch2_fast(mag, src, dst, peaks=None,
                                        pk_start=None, pk_end=None,
                                        peak_count=0, sens=0.0):
    """All-regions-at-once nch==2 sync — identical result, far fewer calls.

    :func:`synchronize_stereo_phases` runs its body once per peak (the C shape);
    for the vocoder path that is ~2000 Python iterations per granule and it
    dominated the render (91 s of 93 s in the profile).  ``w`` depends only on the
    peak's mag ratio and the per-bin body is a pure elementwise function of
    ``(src, dst, w)``, so flattening every region into one index set gives the
    same answer for disjoint regions — which is what ev_find_peaks produces.
    """
    mag = np.asarray(mag, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32, copy=True)
    weight = np.zeros(dst.shape[1], dtype=np.float32)
    pk = int(peak_count)
    if pk < 1:
        return {"dst": dst, "weight": weight}
    if False and _sync2_nb is not None and _route_b():  # slower than numpy path
        pass
    st = np.asarray(pk_start[:pk], dtype=np.int64)
    en = np.asarray(pk_end[:pk], dtype=np.int64)
    lens = en - st
    live = lens > 0
    st, lens = st[live], lens[live]
    if st.size == 0:
        return {"dst": dst, "weight": weight}
    idx = np.repeat(st, lens) + (np.arange(int(lens.sum()), dtype=np.int64)
                                 - np.repeat(np.cumsum(lens) - lens, lens))
    pb = np.asarray(peaks[:pk], dtype=np.int64)[live]
    m0, m1 = mag[0][pb], mag[1][pb]
    v9 = (np.abs((m0 - m1).astype(np.float32))
          / ((m0 + m1).astype(np.float32) + EPS1E6_F)).astype(np.float32)
    v10 = (F32(F32(1.0) / F32(F32(sens) + EPS1E6_F)) * v9).astype(np.float32)
    v11 = np.where(v10 > 0.0,
                   np.where(v10 < K07_F, (v10 / K07_F).astype(np.float32),
                            np.float32(1.0)),
                   np.float32(0.0)).astype(np.float32)
    w = np.sqrt((F32(1.0) - v11).astype(np.float32)).astype(np.float32)
    wt = np.repeat(w, lens)
    weight[idx] = wt
    s0, s1 = src[0][idx], src[1][idx]
    d0, d1 = dst[0][idx], dst[1][idx]
    a0 = _wrap_pi((d0 - s0).astype(np.float32))
    a1 = _wrap_pi((d1 - s1).astype(np.float32))
    mm = ((a0 + a1).astype(np.float32) * F32(0.5)).astype(np.float32)
    flip = ~(np.abs((a1 - a0).astype(np.float32)) < PI_F)
    mm = np.where(flip, (mm + PI_F).astype(np.float32), mm).astype(np.float32)
    mm = _wrap_pi(mm)
    t0 = _wrap_pi(((s0 + mm).astype(np.float32) - d0).astype(np.float32))
    t1 = _wrap_pi(((s1 + mm).astype(np.float32) - d1).astype(np.float32))
    dst[0][idx] = _wrap_pi(_fma_arr(wt, t0, d0))
    dst[1][idx] = _wrap_pi(_fma_arr(wt, t1, d1))
    return {"dst": dst, "weight": weight}


def synchronize_stereo_phases(mag, src, dst, peaks=None, pk_start=None,
                              pk_end=None, peak_count=0, sens=0.0, nch=2):
    """rx_synchronize_stereo_phases @0x16BD64."""
    mag = np.asarray(mag, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32, copy=True)
    weight = np.zeros(dst.shape[1], dtype=np.float32)
    nch = int(nch)
    if nch < 2 or int(peak_count) < 1:
        return {"dst": dst, "weight": weight}
    inv = F32(F32(1.0) / F32(F32(sens) + EPS1E6_F))
    for pk in range(int(peak_count)):
        b = int(peaks[pk])
        m0 = F32(mag[0][b])
        m1 = F32(mag[1][b])
        v9 = F32(F32(abs(float(F32(m0 - m1)))) / F32(F32(F32(m0 + m1) + EPS1E6_F)))
        if nch > 2:
            v9 = F32(0.5)
        v10 = F32(inv * v9)
        v11 = F32(0.0)
        if v10 > 0.0:
            v11 = F32(1.0)
            if v10 < K07_F:
                v11 = F32(v10 / K07_F)
        w = F32(math.sqrt(float(F32(F32(1.0) - v11))))
        st, en = int(pk_start[pk]), int(pk_end[pk])
        weight[st:en] = w
        if nch == 2:
            # `w` is the same for every bin of this region, so the whole region
            # goes through one elementwise pass (the per-bin C loop is only
            # observable through writes, and regions are disjoint).
            d0, d1 = _adjust_diff_nch2_vec(src[0][st:en], src[1][st:en],
                                           dst[0][st:en], dst[1][st:en], w)
            dst[0][st:en] = d0
            dst[1][st:en] = d1
        else:
            for bin_ in range(st, en):
                dst = adjust_multiphase_diff(src, dst, bin_, bin_, w, nch)
    return {"dst": dst, "weight": weight}


def _fm_h8_opt() -> bool:
    """True when the H8 frequency-domain smoothing is explicitly requested.

    That variant is NOT bit-exact: its residual is the reference's own
    finite-domain end conditions (~1e1 dB over the first ~100 bins of a real db
    curve, see .tmp/h8_split2.py), so it is off unless PYR_FM_H8=1 is set.
    """
    return _os.environ.get("PYR_FM_H8", "0") not in ("", "0", "false", "False")


# ===========================================================================
# formant.c — ApplyFormantCorrection
# ===========================================================================

class FormantState:
    """rx_formant_state + rx_formant_apply (persistent env / gain envelope)."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.nb_bins = int(cfg["nb_bins"])
        self.n_fft = int(cfg["n_fft"])
        self.m_fft = int(cfg["m_fft"])
        self.m_bins = int(cfg["m_bins"])
        # NOTE: f580 is NOT cached here.  The engine recomputes it before every
        # rx_formant_apply call (`fc->f580 = prev_granule_1408 > 0 ?
        # prev_granule_1408 : step_base`), and it genuinely changes between
        # granules; reading a snapshot taken at construction pinned it to the
        # 222 default after the first call, which skewed the formant envelope
        # time-constant and so every AFC gain.  Read it via self._f580() below.
        self.sr = float(cfg["sr"])
        self.prec_mode = int(cfg["prec_mode"])
        self.nb_bands = int(cfg["nb_bands"])
        self.env = np.zeros(self.m_bins + 8, dtype=np.float32)
        self.gain_env = np.ones((self.nb_bands, self.nb_bins + 8), dtype=np.float32)
        dbn = max(self.nb_bins, self.m_fft)
        self.db = np.zeros(dbn + 8, dtype=np.float32)
        self.gscr = np.zeros(self.nb_bins + 8, dtype=np.float32)
        self.spec = np.zeros(2 * self.m_bins + 8, dtype=np.float32)
        self.ker = np.zeros(self.m_fft + 8, dtype=np.float32)
        self.ffr = np.zeros(self.m_fft, dtype=np.float32)
        self.ffi = np.zeros(self.m_fft, dtype=np.float32)
        self.inv_scale = F32(1.0 / self.m_fft)
        self._inv_scale = float(self.inv_scale)
        self._thr_score = float(_bits(0xC9742400))
        self._rms_eps = float(_bits(0x2B8CBDDD))
        h1 = _round_i(F32(F32(self.n_fft) * F32(4.0) / F32(self.sr)))
        self._h1 = min(h1, self.m_fft - 1)
        h2 = _round_i(F32(F32(self.n_fft) * F32(9.0) / F32(self.sr)))
        self._h2 = min(h2, self.m_fft)
        self._build_tw()

    def _build_tw(self):
        n = self.m_fft
        idx = np.arange(n, dtype=np.int64)
        bits = n.bit_length() - 1
        rev = np.zeros(n, dtype=np.int64)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        self._rev = rev
        self._tw = {}
        for inverse in (0, 1):
            stages = []
            ln = 2
            while ln <= n:
                half = ln >> 1
                ang = F32((F32(2.0) if inverse else F32(-2.0)) * F32(math.pi) / F32(ln))
                k = np.arange(half, dtype=np.float32)
                arg = (ang * k).astype(np.float32)
                # C uses cosf/sinf on the f32 argument
                c = np.cos(arg.astype(np.float64)).astype(np.float32)[None, :]
                s = np.sin(arg.astype(np.float64)).astype(np.float32)[None, :]
                stages.append((ln, half, c, s))
                ln <<= 1
            self._tw[inverse] = stages

    def _fft(self, ffr, ffi, inverse):
        """formant.c rx_fft_c: radix-2 DIT, per-stage cosf/sinf twiddles."""
        if _route_b() and _vdsp_mod_ref() is not None:
            _vdsp_mod_ref().fft_zip(ffr, ffi, self.m_fft.bit_length() - 1,
                                    bool(inverse))
            return
        n = self.m_fft
        ffr[:] = ffr[self._rev]
        ffi[:] = ffi[self._rev]
        for ln, half, c, s in self._tw[inverse]:
            nb = n // ln
            r = ffr.reshape(nb, ln)
            i = ffi.reshape(nb, ln)
            xr = r[:, half:]
            xi = i[:, half:]
            ur = r[:, :half].copy()
            ui = i[:, :half].copy()
            vr = (xr * c - xi * s).astype(np.float32)
            vi = (xr * s + xi * c).astype(np.float32)
            r[:, :half] = (ur + vr).astype(np.float32)
            i[:, :half] = (ui + vi).astype(np.float32)
            r[:, half:] = (ur - vr).astype(np.float32)
            i[:, half:] = (ui - vi).astype(np.float32)

    @staticmethod
    def _amp2db(x):
        if float(x) >= 9.999999999988105e-21:
            return F32(math.log(float(x)) * 8.68588924407959)
        return F32(-391.0)

    @staticmethod
    def _db2amp(x):
        return F32(math.exp(float(x) * 0.115129254758358))

    @staticmethod
    def _db2amp_arr(x):
        """Vectorised _db2amp (same libm exp, same f32 round)."""
        return np.exp(np.asarray(x, np.float64) * 0.115129254758358).astype(np.float32)

    @staticmethod
    def _round_away_arr(v):
        """Vectorised C `(int)(v + (v<0 ? -0.5 : 0.5))` = round-half-away-from-0."""
        v = np.asarray(v, np.float32)
        return np.trunc((v + np.copysign(F32(0.5), v)).astype(np.float64))

    def _f580(self) -> int:
        """Live +0x580 (prev_granule_1408, else step_base) — re-read every call."""
        return int(self.cfg["f580"]) if int(self.cfg["f580"]) > 0 else 222

    def apply(self, mag, band=0):
        """rx_formant_apply — in-place conceptually; returns the mag buffer."""
        if (_fm_prep_nb is not None and _fm_env2_nb is not None
                and _fm_peak_nb is not None and _fm_clip_mirror_nb is not None
                and _fm_ker_scale_nb is not None and _fm_kermix_nb is not None
                and _fm_gain2_nb is not None and _fm_ges_tail_nb is not None
                and _iir16_nb is not None and _route_b()):
            return self._apply_fast(mag, band)
        return self._apply_full(mag, band)

    def _apply_fast(self, mag, band=0):
        """Route-B fast path: the C formant kernel once it passed its gate,
        otherwise the numba chain."""
        _neon = _neon_data()
        if (_neon is not None and getattr(self, "_fm_car", None) is not None
                and not _fm_h8_opt() and _neon.formant_ok()):
            out = np.array(mag, dtype=np.float32, copy=True)
            c = self.cfg
            if int(c["active"]) == 1 and _neon.formant_run(self, out, band):
                return out
        return self._apply_numba(mag, band)

    def _apply_numba(self, mag, band=0):
        """Route-B fast path: numba kernels only, scalar cache, no numpy
        micro-ops beyond the two RMS sums."""
        mag = np.array(mag, dtype=np.float32, copy=True)
        c = self.cfg
        if int(c["active"]) != 1:
            return mag
        ratio = float(c["ratio"]); strength = float(c["strength"])
        if ratio == 1.0 or strength == 0.0:
            return mag
        NB, M, MB, N = self.nb_bins, self.m_fft, self.m_bins, self.n_fft
        db, ker, gscr = self.db, self.ker, self.gscr
        env = self.env
        ges = self.gain_env[band]
        ffr, ffi = self.ffr, self.ffi
        sr = self.sr

        # ---- 1+2 dB -> FFT ----
        _fm_prep_nb(mag, db, ffr, ffi, NB, M)
        self._fft(ffr, ffi, 0)

        # ---- 3 envelope ----
        tconst = 0.005 if self.prec_mode == 2 else 0.01
        a1 = _t2a_fast(tconst, _f32f(sr) / _f32f(self._f580()))
        _fm_env2_nb(ffr, ffi, env, MB, a1)

        # ---- 4+5 peak ----
        v19 = _f32f(_f32f(sr) * _f32f(M) / _f32f(N))
        best, fC, peak = _fm_peak_nb(env, MB, v19,
                                     _f32f(c["freq_hi"]), _f32f(c["freq_lo"]),
                                     int(c["mode_freq"]), self._thr_score)

        # ---- clip + mirror + inverse FFT -> ker ----
        r_ = ratio if ratio < 1.0 else 1.0
        cut = int(_f32f(_f32f((peak - 2) / _f32f(c["width"])) * _f32f(r_))
                  + (0.5 if _f32f(_f32f((peak - 2) / _f32f(c["width"])) * _f32f(r_)) >= 0 else -0.5))
        if cut < 2:
            cut = 2
        _fm_clip_mirror_nb(ffr, ffi, MB, M, 2 * cut)
        self._fft(ffr, ffi, 1)
        _fm_ker_scale_nb(ffr, ker, M, self._inv_scale)

        # ---- 7 iir16 smoothing ----
        a2 = _t2a_fast(_f32f(fC * _f32f(c["width"] * 0.12)), _f32f(N) / _f32f(sr))
        _neon = _neon_data()
        if _neon is not None:
            _neon.iir16_run(db[:NB], a2, _f32f(1.0 - a2))
        else:
            _iir16_nb(db[:NB], a2, _f32f(1.0 - a2))

        # ---- 8 ker mix ----
        _fm_kermix_nb(db, ker, M, self._h1, self._h2)

        # ---- 9..11+12 ----
        a3 = _t2a_fast(tconst, _f32f(sr) / _f32f(self._f580()))
        _fm_gain2_nb(db, gscr, NB, ratio, strength)
        if int(c["mode_rms"]):
            mm = mag[:NB].astype(np.float64)
            gg = gscr[:NB].astype(np.float64)
            den = float(np.sum(mm ** 2))
            num = float(np.sum(gg * gg * mm * mm))
            ss = math.sqrt(den / (num + self._rms_eps))
            gscr[:NB] = (np.float32(ss) * gscr[:NB]).astype(np.float32)
        _fm_ges_tail_nb(gscr, ges, mag, NB, a3, fC, N, sr)
        return mag

    def _apply_full(self, mag, band=0):
        mag = np.array(mag, dtype=np.float32, copy=True)
        c = self.cfg
        NB, M, MB, N = self.nb_bins, self.m_fft, self.m_bins, self.n_fft
        db, spec, ker, gscr = self.db, self.spec, self.ker, self.gscr
        env = self.env
        ges = self.gain_env[band]
        if int(c["active"]) != 1:
            return mag
        ratio = F32(c["ratio"])
        strength = F32(c["strength"])
        width = F32(c["width"])
        if ratio == 1.0 or strength == 0.0:
            return mag
        if NB < 1 or MB < 1:
            return mag

        # ---- 1 dB + 2 formant FFT prep ----
        if _fm_prep_nb is not None:
            _fm_prep_nb(mag, db, self.ffr, self.ffi, NB, M)
        else:
            thr = 9.999999999988105e-21
            if _fm_db_nb is not None:
                _fm_db_nb(mag, db, NB)
            else:
                mv = mag[:NB].astype(np.float64)
                db[:NB] = np.where(mv >= thr,
                                   (np.log(np.maximum(mv, 1e-300)) * 8.68588924407959).astype(np.float32),
                                   F32(-391.0)).astype(np.float32)
            self.ffr[:] = db[:M]
            self.ffi[:] = 0.0
        self._fft(self.ffr, self.ffi, 0)

        # ---- 3 envelope IIR ----
        tconst = F32(0.005) if self.prec_mode == 2 else F32(0.01)
        a1 = time_to_iir_a(tconst, F32(F32(self.sr) / F32(self._f580())))
        if _fm_env2_nb is not None:
            _fm_env2_nb(self.ffr, self.ffi, env, MB, float(a1))
        elif _fm_env_nb is not None:
            spec[0:2 * MB:2] = self.ffr[:MB]
            spec[1:2 * MB:2] = self.ffi[:MB]
            _fm_env_nb(spec, env, MB, float(a1))
        else:
            spec[0:2 * MB:2] = self.ffr[:MB]
            spec[1:2 * MB:2] = self.ffi[:MB]
            e = ((spec[0:2 * MB:2].astype(np.float32) * spec[0:2 * MB:2].astype(np.float32)).astype(np.float32)
                 + (spec[1:2 * MB:2].astype(np.float32) * spec[1:2 * MB:2].astype(np.float32)).astype(np.float32)).astype(np.float32)
            env[:MB] = _fma_arr(a1, (e - env[:MB]).astype(np.float32), env[:MB])

        # ---- 4 peak search + 5 peak freq/clip ----
        v19 = F32(F32(self.sr) * F32(M) / F32(N))
        if _fm_peak_nb is not None:
            best, fC, peak = _fm_peak_nb(env, MB, float(v19),
                                         float(F32(c["freq_hi"])),
                                         float(F32(c["freq_lo"])),
                                         int(c["mode_freq"]),
                                         float(_bits(0xC9742400)))
        else:
            hi = _round_i(F32(v19 / F32(c["freq_hi"])))
            hi = min(hi, MB - 2)
            hi = max(hi, 2)
            lo_ = _round_i(F32(v19 / F32(c["freq_lo"])))
            lo_ = min(lo_, MB - 2)
            best = 0
            if lo_ > hi:
                k = np.arange(hi, lo_)
                v = env[k]
                r1 = (v / ((env[k - 1] + env[k + 1]).astype(np.float32) + EPS1E6_F)).astype(np.float32)
                w1 = np.where(r1 <= F32(0.5), F32(0.2),
                              np.where(r1 < F32(2.0),
                                       ((r1 - F32(0.5)) / F32(1.5)).astype(np.float32) + F32(0.2),
                                       F32(1.2))).astype(np.float32)
                r2 = (v / (env[k >> 1] + EPS1E6_F)).astype(np.float32)
                w2 = np.where(r2 > F32(0.3),
                              np.where(r2 < F32(5.0),
                                       ((r2 - F32(0.3)) / F32(4.7)).astype(np.float32),
                                       F32(1.0)),
                              np.float32(0.0)).astype(np.float32)
                score = ((v * w1).astype(np.float32) * w2).astype(np.float32)
                bi = int(np.argmax(score))
                if float(score[bi]) > float(_bits(0xC9742400)):
                    best = hi + bi
            if int(c["mode_freq"]):
                pk = F32(v19 / F32(best)) if best else F32(0.0)
                peak = best
            else:
                pk = F32(500.0)
                peak = _round_i(F32(v19 / pk))
            fC = pk if pk <= F32(800.0) else F32(800.0)
            fC = fC if fC >= F32(150.0) else F32(150.0)
        cut = _round_i(F32(F32(F32(peak - 2) / width) * F32(min(float(ratio), 1.0))))
        if cut < 2:
            cut = 2
        c4 = 2 * cut

        # ---- 6 IFFT -> ker ----
        if _fm_clip_mirror_nb is not None:
            # route B: ffr/ffi already hold the packed spectrum from the
            # forward FFT — clip+zero+mirror in place, no spec round-trip.
            _fm_clip_mirror_nb(self.ffr, self.ffi, MB, M, c4)
        else:
            spec[c4 - 4] = F32(spec[c4 - 4] * F32(0.75))
            spec[c4 - 3] = F32(spec[c4 - 3] * F32(0.75))
            spec[c4 - 2] = F32(spec[c4 - 2] * F32(0.25))
            spec[c4 - 1] = F32(spec[c4 - 1] * F32(0.25))
            spec[c4:2 * MB] = 0.0
            if _fm_ker_nb is not None:
                _fm_ker_nb(spec, self.ffr, self.ffi, ker, MB, M,
                           float(self.inv_scale))
            else:
                self.ffr[:MB] = spec[0:2 * MB:2]
                self.ffi[:MB] = spec[1:2 * MB:2]
                self.ffi[0] = 0.0
                self.ffi[MB - 1] = 0.0
                kk = np.arange(1, MB - 1)
                self.ffr[M - kk] = self.ffr[kk]
                self.ffi[M - kk] = -self.ffi[kk]
        self._fft(self.ffr, self.ffi, 1)
        if _fm_ker_scale_nb is not None:
            _fm_ker_scale_nb(self.ffr, ker, M, float(self.inv_scale))
        else:
            ker[:M] = (self.ffr[:M] * self.inv_scale).astype(np.float32)

        # ---- 7 dB curve 8x forward+backward smoothing ----
        a2 = time_to_iir_a(F32(fC * F32(width * F32(0.12))), F32(F32(N) / F32(self.sr)))
        _neon = _neon_data()
        if _neon is not None:
            _neon.iir16_run(db[:NB], float(a2), float(F32(F32(1.0) - a2)))
        elif _iir16_nb is not None and _route_b():
            db[:NB] = _iir16_nb(db[:NB].astype(np.float32), float(a2),
                                float(F32(F32(1.0) - a2)))
        else:
            y = db[:NB].astype(np.float64)
            bcoef = [float(a2)]
            acoef = [1.0, -float(F32(F32(1.0) - a2))]
            for _ in range(8):
                y, _ = lfilter(bcoef, acoef, y, zi=np.array([(1.0 - float(a2)) * y[0]]))
                y = y[::-1]
                y, _ = lfilter(bcoef, acoef, y, zi=np.array([(1.0 - float(a2)) * y[0]]))
                y = y[::-1]
                y = y.astype(np.float32).astype(np.float64)
            db[:NB] = y.astype(np.float32)

        # ---- 8 ker mix-in ----
        h1 = _round_i(F32(F32(N) * F32(4.0) / F32(self.sr)))
        h1 = min(h1, M - 1)
        h2 = _round_i(F32(F32(N) * F32(9.0) / F32(self.sr)))
        h2 = min(h2, M)
        if _fm_kermix_nb is not None:
            _fm_kermix_nb(db, ker, M, int(h1), int(h2))
        else:
            if h1 > 0:
                db[:h1] = ker[:h1]
            if h2 > h1:
                span = F32(h2 - h1)
                k = np.arange(h1, h2)
                w = (F32(h2 - k) / span).astype(np.float32)
                db[h1:h2] = _fma_arr(w, (ker[h1:h2] - db[h1:h2]).astype(np.float32), db[h1:h2])

        # ---- 9 gain (fused) ----
        a3 = time_to_iir_a(tconst, F32(F32(self.sr) / F32(self._f580())))
        if _fm_gain2_nb is not None:
            _fm_gain2_nb(db, gscr, NB, float(ratio), float(strength))
        elif _fm_gain_nb is not None:
            _fm_gain_nb(db, gscr, ges, mag, NB, float(ratio), float(strength))
        else:
            d = np.zeros(NB, dtype=np.float32)
            # k2[v] = round(ratio * v): f32 product then round-half-away-from-zero
            k2 = self._round_away_arr((ratio * np.arange(NB, dtype=np.float32)).astype(np.float32)
                                      ).astype(np.int64)
            valid = k2 < NB
            kk = np.flatnonzero(valid)
            d[kk] = (db[k2[kk]] - db[kk]).astype(np.float32)
            lastv = np.maximum.accumulate(np.where(valid, np.arange(NB), -1))
            d = d[np.maximum(lastv, 0)]
            d = np.clip(d, F32(-40.0), F32(20.0))
            gscr[:NB] = self._db2amp_arr((d * strength).astype(np.float32))

        # ---- 10 RMS normalisation ----
        if int(c["mode_rms"]):
            mm = mag[:NB]
            gg = gscr[:NB]
            den = float(np.sum(mm.astype(np.float64) ** 2))
            num = float(np.sum((gg * gg).astype(np.float64) * (mm * mm).astype(np.float64)))
            ss = F32(math.sqrt(den / (num + float(_bits(0x2B8CBDDD)))))
            gscr[:NB] = (ss * gscr[:NB]).astype(np.float32)

        # ---- 11+12 gain envelope + tail multiply (fused) ----
        if _fm_ges_tail_nb is not None:
            _fm_ges_tail_nb(gscr, ges, mag, NB, float(a3), float(fC),
                            int(N), float(self.sr))
            return mag
        if _fm_ges_nb is not None:
            _fm_ges_nb(gscr, ges, NB, float(a3))
        else:
            ges[:NB] = _fma_arr(a3, (gscr[:NB] - ges[:NB]).astype(np.float32), ges[:NB])

        # ---- 12 tail multiply ----
        vt = F32(F32(fC * F32(0.8)) * F32(N) / F32(self.sr))
        tail = _round_i(vt)
        tail = min(tail, NB - 1)
        mag[tail:NB] = (ges[tail:NB] * mag[tail:NB]).astype(np.float32)
        return mag


def _bits_nb(u32):
    """Float32 from raw u32 bits (numba-friendly, mirrors _bits)."""
    return np.float32(_struct.unpack("<f", _struct.pack("<I", u32))[0])


def formant_apply(state, mag, band=0):
    """Convenience wrapper (module-level symbol for the parity gate)."""
    return state.apply(mag, band=band)


# ===========================================================================
# overlap_add.c
# ===========================================================================

def overlap_add_channel(win1=None, win2=None, synth_a=None, synth_b=None,
                        out_ring=None, edge_gain=None, ch=0, a3=0, a4=0, a5=0,
                        frame_n=0, p=0, A=0, v8=0, v9=0, f1412=0, cursor=0,
                        hop=0, ring_len=0, f1224=0, n_write=0):
    """rx_overlap_add_channel @0x16EE44 (window fade + ring write)."""
    win1 = np.array(win1, dtype=np.float32, copy=True)
    win2 = np.array(win2, dtype=np.float32, copy=True)
    synth_a = np.asarray(synth_a, dtype=np.float32)
    synth_b = np.asarray(synth_b, dtype=np.float32)
    out = np.array(out_ring, dtype=np.float32, copy=True)
    edge_gain = np.asarray(edge_gain, dtype=np.float32)
    frame_n, hop = int(frame_n), int(hop)
    n_write, ring_len = int(n_write), int(ring_len)
    cursor, f1224 = int(cursor), int(f1224)
    half_n = frame_n >> 1

    den_i = int(A) * int(v8) // (int(A) + int(v8))
    gidx = int(p) if int(p) < 3 else 3
    if int(p) >= 10:
        gidx = 0
    v197 = F32(edge_gain[gidx] * F32(0.4))
    v197 = F32(v197 * F32(int(f1412) + int(a3)))
    v197 = F32(v197 / F32(den_i))
    v198 = F32(1.0)
    g09 = F32(v197 * F32(0.9))
    if int(a4) > 0:
        v198 = F32(0.5)
        v197 = g09

    # ---- window fading (in place on win1 / win2) ----
    half1, half2 = int(v8) // 2, int(v9) // 2
    base = half_n - hop
    if cursor < hop + half1 and hop >= 1:
        pos = cursor - hop
        n = min(2 * hop, half1 - pos)
        if n > 0:
            pp = np.arange(pos, pos + n, dtype=np.float32)
            t = ((pp * PI_F).astype(np.float32) * F32(0.5)).astype(np.float32)
            t = (t / F32(half1)).astype(np.float32)
            t = (F32(2.0) - np.sqrt(np.sin(t.astype(np.float64)).astype(np.float32))).astype(np.float32)
            sl = slice(base, base + n)
            win1[sl] = (t * win1[sl]).astype(np.float32)
    if cursor < hop + half2 and hop > 0:
        pos = cursor - hop
        n = min(2 * hop, half2 - pos)
        if n > 0:
            pp = np.arange(pos, pos + n, dtype=np.float32)
            t = ((pp * PI_F).astype(np.float32) * F32(0.5)).astype(np.float32)
            t = (t / F32(half2)).astype(np.float32)
            t = (F32(2.0) - np.sqrt(np.sin(t.astype(np.float64)).astype(np.float32))).astype(np.float32)
            sl = slice(base, base + n)
            win2[sl] = (t * win2[sl]).astype(np.float32)

    gw = [float(v197), float(v198)]
    if int(a4) != int(a5):
        return {"out_ring": out, "win1": win1, "win2": win2, "gain_weight": np.array(gw, np.float32)}

    # ---- main write ----
    R, N = ring_len, n_write
    pos0 = int(math.fmod(cursor - hop, R))          # C truncating remainder
    v35 = min(f1224 + hop - cursor, N)
    v36 = max(v35, 0)
    wh = base
    _neon = _neon_data()
    if _neon is not None and out.flags.c_contiguous and win1.flags.c_contiguous \
            and win2.flags.c_contiguous and synth_a.flags.c_contiguous \
            and synth_b.flags.c_contiguous:
        _neon.oac_run(out, win1, win2, synth_a, synth_b, R, N, pos0, v35, v36,
                      wh, cursor, hop, float(v197), float(v198))
        return {"out_ring": out, "win1": win1, "win2": win2, "gain_weight": np.array(gw, np.float32)}
    if _oac_nb is not None:
        _oac_nb(out, win1, win2, synth_a, synth_b, R, N, pos0, v35, v36, wh,
                cursor, hop, float(v197), float(v198))
        return {"out_ring": out, "win1": win1, "win2": win2, "gain_weight": np.array(gw, np.float32)}
    t2 = (v198 * (win2[wh:wh + N] * synth_b[:N]).astype(np.float32)).astype(np.float32)
    t = _fma_arr(win1[wh:wh + N], synth_a[:N], t2)
    if cursor < hop or N + pos0 > R:
        pcur = pos0 if pos0 >= 0 else 0
        i = np.arange(N, dtype=np.int64)
        cpos = cursor - hop + i
        valid = cpos >= 0
        k = np.cumsum(valid) - valid
        pidx = (pcur + k) % R
        m1 = valid & (i < v35)
        m2 = valid & (i >= v36) & (i >= v35)
        if m1.any():
            out[pidx[m1]] = _fma_arr(t[m1], np.float64(v197), out[pidx[m1]])
        if m2.any():
            out[pidx[m2]] = (t[m2] * v197).astype(np.float32)
        return {"out_ring": out, "win1": win1, "win2": win2, "gain_weight": np.array(gw, np.float32)}

    i1 = np.arange(0, v36, dtype=np.int64)
    if i1.size:
        out[pos0 + i1] = _fma_arr(t[i1], np.float64(v197), out[pos0 + i1])
    i2 = np.arange(v36, N, dtype=np.int64)
    if i2.size:
        out[pos0 + i2] = (t[i2] * v197).astype(np.float32)
    return {"out_ring": out, "win1": win1, "win2": win2, "gain_weight": np.array(gw, np.float32)}


# ===========================================================================
# randomize_phases.c / substitute_noisy_phases.c
# ===========================================================================

#: f32(2π/32767) bit pattern (0x3949116D; the decimal 0.00019175f is 235 ulp off)
_NP_INV32767 = _bits(0x3949116D)


def _noise_start_bin(f372, u112):
    """Randomize/Substitute preamble: start = (int)(f372*150/u112 +- 0.5)."""
    v = F32(F32(f372) * F32(150.0))
    v = F32(v / F32(np.uint32(u112)))
    v = F32(v + (F32(0.5) if v >= 0.0 else F32(-0.5)))
    return int(v)


def randomize_phases(noise_phase, noise_gain, phase, mag, region_gain,
                     noise_weight, sync_weight, a2=0.0, ramp=0.0, f372=0, u112=0,
                     seed=1, nch=1, max_bin=0):
    """rx_randomize_phases @0x16CE40 (PRNG order: bin outer, channel inner)."""
    noise_phase = np.array(noise_phase, dtype=np.float32, copy=True)
    noise_gain = np.array(noise_gain, dtype=np.float32, copy=True)
    phase = np.array(phase, dtype=np.float32, copy=True)
    mag = np.array(mag, dtype=np.float32, copy=True)
    region_gain = np.asarray(region_gain, dtype=np.float32)
    noise_weight = np.asarray(noise_weight, dtype=np.float32)
    sync_weight = np.asarray(sync_weight, dtype=np.float32)
    gain_mean = np.zeros(int(max_bin), dtype=np.float32)
    nch, max_bin = int(nch), int(max_bin)
    start = _noise_start_bin(f372, u112)
    rng = SimpleRand(int(seed))

    # ---- loop 1: noise generation ----
    if max_bin > start and nch != 0:
        v7 = F32(min(float(F32(a2)), 2.0))
        v8 = F32(ramp)
        lo = F32(v8 + F32(1.0))
        hi = _fma(v8, F32(6.0), F32(2.0))
        span = F32(hi - lo)
        for bin_ in range(start, max_bin):
            w0 = F32(noise_weight[bin_])
            for ch in range(nch):
                g = F32(region_gain[ch][bin_])
                t = F32(0.0)
                if not (g <= lo):
                    t = F32(1.0)
                    if not (g >= hi):
                        t = F32(F32(g - lo) / span)
                v14 = F32(F32(F32(1.0) - t) * F32(v7 * w0))
                n1 = F32(rng.next())
                n2 = F32(rng.next())
                noise_phase[ch][bin_] = F32(F32(v14 * _NP_INV32767) * F32(n1 - n2))
                noise_gain[ch][bin_] = _fma(F32(v14 * v14), F32(6.5), F32(1.0))

    # ---- loop 2: gain mean + per-bin AMPD (nch != 1) ----
    if max_bin > start:
        for bin_ in range(start, max_bin):
            if nch != 0:
                ssum = F32(0.0)
                for ch in range(nch):
                    ssum = F32(ssum + noise_gain[ch][bin_])
                gain_mean[bin_] = F32(ssum / F32(nch))
                if nch != 1:
                    # AMpd hook: pull_to_peak(phase[0], noise_phase[0], bin, bin, w)
                    tmp = ampd_pull_to_peak(phase[0], noise_phase[0], bin_, bin_,
                                            F32(sync_weight[bin_]))
                    phase[0] = tmp
            else:
                gain_mean[bin_] = F32(np.float32(0.0) / np.float32(0.0))

    # ---- loop 3: apply ----
    if nch != 0 and max_bin > start:
        for ch in range(nch):
            b = np.arange(start, max_bin, dtype=np.int64)
            phase[ch][b] = (noise_phase[ch][b] + phase[ch][b]).astype(np.float32)
            gain = noise_gain[ch][b]
            mag[ch][b] = (mag[ch][b] * _fma_arr(sync_weight[b],
                                                (gain_mean[b] - gain).astype(np.float32),
                                                gain)).astype(np.float32)
    return {"phase": phase, "mag": mag, "noise_phase": noise_phase,
            "noise_gain": noise_gain, "gain_mean": gain_mean}


def substitute_noisy_phases(noise_phase, phase, mag, region_gain, noise_template,
                            noise_weight, sync_weight, a2=0.0, ramp=0.0, f372=0,
                            u112=0, slot=0, slot_count=0, tmpl_stride=0,
                            nch=1, max_bin=0):
    """rx_substitute_noisy_phases @0x16C540 (template copy + gain ramp blend)."""
    noise_phase = np.array(noise_phase, dtype=np.float32, copy=True)
    phase = np.array(phase, dtype=np.float32, copy=True)
    mag = np.array(mag, dtype=np.float32, copy=True)
    region_gain = np.asarray(region_gain, dtype=np.float32)
    t0 = np.asarray(noise_template[0], dtype=np.float32)
    t1 = np.asarray(noise_template[1], dtype=np.float32)
    noise_weight = np.asarray(noise_weight, dtype=np.float32)
    sync_weight = np.asarray(sync_weight, dtype=np.float32)
    nch, max_bin = int(nch), int(max_bin)
    start = _noise_start_bin(f372, u112)
    slot = int(slot)
    off = slot * int(tmpl_stride)

    # ---- Part A: template -> noise phase ----
    if max_bin > start:
        b = np.arange(start, max_bin, dtype=np.int64)
        if nch < 2:
            nph = t0[b + off]
            noise_phase[0][b] = nph
        else:
            for ch in range(nch):
                noise_phase[ch][b] = (t1 if (ch & 1) else t0)[b + off]
            for bin_ in range(start, max_bin):
                tmp = ampd_pull_to_peak(phase[0], noise_phase[0], bin_, bin_,
                                        F32(sync_weight[bin_]))
                phase[0] = tmp

    # ---- Part B: gain ramp blend ----
    if nch != 0 and max_bin > start:
        v8 = F32(ramp)
        lo = F32(v8 + F32(1.0))
        hi = _fma(v8, F32(6.0), F32(2.0))
        for bin_ in range(start, max_bin):
            wgt = F32(noise_weight[bin_])
            for ch in range(nch):
                ratio = F32(F32(region_gain[ch][bin_]) / F32(F32(a2) * wgt))
                t = F32(0.0)
                if not (ratio <= lo):
                    t = F32(1.0)
                    if not (ratio >= hi):
                        t = F32(F32(ratio - lo) / F32(hi - lo))
                k = F32(F32(1.0) - t)
                d = F32(noise_phase[ch][bin_] - phase[ch][bin_])
                wrap = _fma(_rint(F32(d * INV_2PI_F)), -TWO_PI_F, d)
                phase[ch][bin_] = _fma(k, wrap, phase[ch][bin_])
                if nch == 1:
                    magfac = F32(F32(k * F32(0.5)) + F32(1.0))
                else:
                    magfac = F32(_fma(F32(sync_weight[bin_]), F32(k * F32(0.2)),
                                      F32(k * F32(0.5))) + F32(1.0))
                mag[ch][bin_] = F32(magfac * mag[ch][bin_])

    # ---- Part C: slot rotation ----
    slot = (slot + 1) if (slot + 1 < int(slot_count)) else 0
    return {"phase": phase, "mag": mag, "slot": slot}


# ===========================================================================
# crossover.c
# ===========================================================================

_XOVER_TAPS = {}


def _xover_taps_rev(sr):
    """taps_rev[b][k] = taps[b][TAPS-1-k] as float32 (rx_crossover_create)."""
    key = int(sr)
    t = _XOVER_TAPS.get(key)
    if t is None:
        fir = np.asarray(get_tables(key)["fir"], dtype=np.float32)
        t = np.ascontiguousarray(fir[:, ::-1])
        _XOVER_TAPS[key] = t
    return t


class Crossover:
    """4-band FIR bank — bit-exact port of rx_crossover_process1.

    Geometry (adjudicated against the C, not inferred from its comments): the
    engine keeps a double-mirrored history buffer, writes ``buf[h] = buf[h+N] =
    x[t]`` and reads ``hp = buf + h + 1``, so ``hp[k] = x[t-N+1+k]``, while the
    coefficient table is pre-reversed via ``taps_rev[b][k] = taps[b][N-1-k]``.
    Combining the two gives the ordinary causal convolution

        y[t] = sum_i taps[b][i] * x[t-i]

    verified bit-exact (max|d| = 0, exact_frac = 1.000000) against the crossover
    parity corpus by .tmp/xo_which.c, which runs BOTH the engine's indexing and
    a plain causal convolution: the engine indexing reproduces the corpus
    exactly, while the plain version (correct algebra, different f32 rounding
    arrangement) is only within 3.7e-09.

    Accumulation must follow the engine's NEON tree verbatim: four lane chains
    ``s{j} = vfmaq(s{j}, hp[k+4j], taps_rev[k+4j])`` for k stepping by 16, then
    ``vaddvq(vaddq(vaddq(s0,s1), vaddq(s2,s3)))``.  This matters far more than
    the ~1e-7 size suggests: the vocoder's peak detector is a discrete decision,
    so a 1e-7 crossover difference flipped the peak count (236 vs 290) and
    decorrelated the entire render.
    """

    def __init__(self, sr, n_bands=4):
        fir = np.asarray(get_tables(int(sr))["fir"], dtype=np.float32)
        # taps_rev[b][k] = taps[b][N-1-k] (rx_crossover_create)
        self.taps_rev = np.ascontiguousarray(fir[:, ::-1])
        self.N = fir.shape[1]
        self.n_bands = n_bands
        # the C's mirrored buffer retains the previous N-1 samples across calls,
        # so the filter is continuous over feed blocks (not block-local).
        self.hist = np.zeros(self.N - 1, dtype=np.float32)

    def process(self, seq):
        """One channel: returns f64 [n_bands, n] (the C yields band values as double)."""
        x = np.asarray(seq, dtype=np.float32)
        n = x.size
        N = self.N
        if n == 0:
            return np.zeros((self.n_bands, 0), dtype=np.float64)
        # hp[t, k] = x_global[t0 + t - N + 1 + k] = z[t + k] with
        # z = [previous N-1 samples, x]; that is the C's buf[head+1 .. head+N]
        # read after writing buf[head] = buf[head+N] = x[t].
        z = np.concatenate([self.hist, x])
        self.hist = z[-(N - 1):].copy()
        if _route_b() and _xover_zp_nb is not None:
            out = np.empty((self.n_bands, n), dtype=np.float32)
            _xover_zp_nb(z, self.taps_rev, out, n, N, self.n_bands)
            return out.astype(np.float64)
        if _route_b() and _xover_z_nb is not None:
            out = np.empty((self.n_bands, n), dtype=np.float32)
            for b in range(self.n_bands):
                _xover_z_nb(z, self.taps_rev[b], out[b], n, N)
            return out.astype(np.float64)
        hp = np.ascontiguousarray(
            np.lib.stride_tricks.sliding_window_view(z, N))
        out = np.empty((self.n_bands, n), dtype=np.float32)
        kern = _xover_fast_nb if (_route_b() and _xover_fast_nb is not None) else _xover_nb
        if kern is not None:
            for b in range(self.n_bands):
                kern(hp, self.taps_rev[b], out[b], n, N)
        else:
            for b in range(self.n_bands):
                tp = self.taps_rev[b].astype(np.float64)
                s = np.zeros((16, n), dtype=np.float32)
                for k in range(0, N, 16):
                    s = _fma_arr(hp[:, k:k + 16].T, tp[k:k + 16, None], s)
                sj = s.reshape(4, 4, n)
                v = ((sj[0] + sj[1]).astype(np.float32)
                     + (sj[2] + sj[3]).astype(np.float32)).astype(np.float32)
                out[b] = ((v[0] + v[1]).astype(np.float32)
                          + (v[2] + v[3]).astype(np.float32)).astype(np.float32)
        return out.astype(np.float64)


def crossover_process1(sr, seq):
    """One-channel fresh-instance run (as in the C op harness)."""
    return Crossover(sr).process(seq)


def crossover_process(sr, seq):
    """Whole-sequence convenience form (per-channel instance, 1 channel)."""
    return Crossover(sr).process(seq)
