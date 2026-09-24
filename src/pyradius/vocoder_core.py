"""pyradius.vocoder_core — vocoder chain orchestration (port of vocoder_core.c).

The DSP operators themselves live in :mod:`pyradius.vocoder_ops`, which is
bit-checked against the C operator corpus by ``tests/test_ops_parity.py``.
This module is only the orchestration layer of libradius/src/engine/vocoder_core.c
plus the CLI driver of libradius/tools/rx_vc_render.c:

  * ``rx_vc_init``            — container topology, persistent state, windows
  * ``rx_vc_start_streaming`` — PRNG Seed(1) + noise template
  * ``rx_vc_process_granule`` — granule schedule, warm-up g0..g7, 4-phase chain
  * ``rx_vc_feed``            — crossover into the band rings + readiness gate
  * ``rx_vc_take_output``     — output-ring drain
  * ``vc_render``             — feed rhythm (448 then 1024) + Resampler drain

Two libradius OOB bugs are fixed here (see ``tools/libradius_vc_oob_fix.md``);
the 48 kHz path is unaffected and the 44.1 kHz path becomes well-defined:

  1. ``copy_len_594 = nb_bins``.  The C hardcodes ``RX_VC_V4_COPY_LEN`` (8193),
     so at 44.1 kHz (nbins = 4097) every Unwrap tail memcpy wrote 4096 floats
     past the end of ``phase[b]``/``scratch`` — a 16 KB heap smash per call.
  2. the ``cnt == 0`` peak fallback sets ``reg_end[0] = nbins - 1``.  The C sets
     ``nbins``, which makes UnwrapPhase's MaxIndex read ``mag[nbins]``.

Floating point: float32 throughout with the C's op order; fused C operations go
through the correctly-rounded fma shim in :mod:`pyradius.vocoder_ops`.
"""
from __future__ import annotations

import math

import ctypes

import numpy as np
from scipy.signal import lfilter

from .fft import plan as _fft_plan, fwd_fast, inv_fast
from .sampler import InterpTable, interp_nsamples
from .simple_rand import SimpleRand, noise_template_fill
from .tables import get_tables, vc_calc_sched
from . import vocoder_ops as _ops
from .vocoder_ops import (F32, PI_F, TWO_PI_F, NEG_2PI_F, INV_2PI_F, K07_F,
                          EPS1E6_F, _bits, _fma_arr, _rint, _round_i, _wrap_pi,
                          time_to_iir_a, FormantState,
                          fill_granule as _fill_op)

#: Use pyradius.fft's bit-exact mirror of rx_fft_fwd/rx_fft_inv rather than
#: scipy's pocketfft.  pyradius.fft is verified bit-identical to the C kernel
#: (max|d| = 0 against rx_fft_fwd at N = 2048/4096/8192/16384, see
#: .tmp/fftcmp2.c).  scipy is a close but inexact substitute: on the smoke render
#: it still tracks (corr 0.99999994) but doubles the residual
#: (mean|d| 1.96e-05 vs 9.50e-06), because a non-exact FFT perturbs the magnitude
#: spectrum that peak detection thresholds on.  Keep True so that internal state
#: stays bit-exact; False is only for speed experiments.
#: Route B (speed target) auto-disables it: set env PYR_FAST=1 to enable the
#: whole fast stack (vDSP/scipy FFT + fastmath kernels) in one move.
EXACT_FFT = True
import os as _os

ROUTE_B_FAST = _os.environ.get("PYR_FAST", "") not in ("", "0", "false", "False")

# Route-B NEON kernels (optional: import never raises, kernels are None on
# failure and everything falls back to numba).
try:
    from . import neon as _neon
except Exception:                                     # pragma: no cover
    _neon = None

_p2c_nb = None
try:
    from numba import njit as _njit2

    @_njit2(cache=True, fastmath=True)
    def _p2c_nb(mag, phase, cart, n):
        """Route-B polar->cart: f32 sincos fused with mag multiply."""
        for i in range(n):
            m = mag[i]
            p = phase[i]
            cart[2 * i] = m * np.cos(p)
            cart[2 * i + 1] = m * np.sin(p)
except Exception:
    _p2c_nb = None

_fold_nb = None
try:
    from numba import njit as _njit3

    @_njit3(cache=True, fastmath=False)
    def _fold_nb(acc, w1, w2, N, hop, alpha, c):
        """Route-B fold: fftshift + fwd/bwd first-order IIR + sub, fused.

        y[0] = alpha*x[0] + c*x[0] (scipy zi semantics); recurrence in f64,
        rounded to f32 on store (closer to the C's float storage).
        """
        half = N >> 1
        for i in range(half):
            w1[half + i] = acc[i]
            w1[i] = acc[half + i]
            w2[half + i] = acc[i]
            w2[i] = acc[half + i]
        x0 = float(w1[0])
        prev = alpha * x0 + c * x0
        w1[0] = np.float32(prev)
        for i in range(1, N):
            prev = alpha * float(w1[i]) + c * prev
            w1[i] = np.float32(prev)
        stop = half - hop
        xn = float(w1[N - 1])
        prev = alpha * xn + c * xn
        w1[N - 1] = np.float32(prev)
        for i in range(N - 2, stop - 1, -1):
            prev = alpha * float(w1[i]) + c * prev
            w1[i] = np.float32(prev)
        for i in range(2 * hop):
            j = stop + i
            w2[j] = w2[j] - w1[j]
except Exception:
    _fold_nb = None

_pull_nb = None
try:
    from numba import njit as _njit4

    @_njit4(cache=True, fastmath=False)
    def _pull_nb(phase, mask, region_gain, reg_start, reg_end, peak_bins,
                 cnt, lo, hi, inv_range):
        """Route-B pull-to-peak: scalar region loops, bit-equal semantics."""
        for r in range(cnt):
            rs = reg_start[r]
            re_ = reg_end[r]
            pk = peak_bins[r]
            for b in range(rs, re_):
                g = region_gain[b]
                if g >= hi:
                    w = np.float32(1.0)
                elif g > lo:
                    w = (g - lo) * inv_range
                else:
                    w = np.float32(0.0)
                if w > np.float32(0.0) and b != pk:
                    pm_bin = phase[b]
                    x0 = phase[pk] - pm_bin
                    q0 = np.rint(x0 * np.float32(0.15915494))
                    t0 = np.float32(np.float64(q0) * -6.2831854820251465
                                    + np.float64(x0))
                    x1 = mask[b] - mask[pk]
                    q1 = np.rint(x1 * np.float32(0.15915494))
                    t1 = np.float32(np.float64(q1) * -6.2831854820251465
                                    + np.float64(x1))
                    s = t0 + t1
                    qs = np.rint(s * np.float32(0.15915494))
                    sw = np.float32(np.float64(qs) * -6.2831854820251465
                                    + np.float64(s))
                    phase[b] = np.float32(np.float64(w) * np.float64(sw)
                                          + np.float64(pm_bin))
except Exception:
    _pull_nb = None


_ring_scatter_nb = None
try:
    from numba import njit as _njit7

    @_njit7(cache=True)
    def _ring_scatter_nb(ring, bands, base, nin, lag, cap):
        """Route-B feed ring write (modulo scatter, all bands)."""
        for i in range(nin):
            j = (base + i - lag) % cap
            for b in range(bands.shape[0]):
                ring[b, j] = bands[b, i]
except Exception:
    _ring_scatter_nb = None

_acs_mag_env_nb = None
try:
    from numba import njit as _njit5

    @_njit5(cache=True, fastmath=False)
    def _acs_mag_env_nb(cart, env, n_bins, iir_a):
        """Route-B acs: |cart| + envelope IIR fused (1ulp vs split numpy)."""
        for k in range(n_bins):
            re = cart[2 * k]
            im = cart[2 * k + 1]
            m = np.sqrt(re * re + im * im)
            env[k] = np.float32(np.float64(iir_a) * np.float64(m - env[k])
                                + np.float64(env[k]))
except Exception:
    _acs_mag_env_nb = None

_find_peaks_nb = None
try:
    from numba import njit as _njit6

    @_njit6(cache=True, fastmath=False)
    def _find_peaks_nb(mag, env, peak_bins, reg_offset, reg_start, reg_end,
                       region_gain, NB):
        """Route-B find_peaks: scalar loops, bit-equal to the numpy path."""
        cnt = 0
        for i in range(1, NB - 1):
            if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]:
                peak_bins[cnt] = i
                cnt += 1
        if cnt == 0:
            peak_bins[0] = NB // 2
            reg_offset[0] = np.float32(NB // 2)
            reg_start[0] = 0
            reg_end[0] = NB - 1
            return 1
        for j in range(cnt):
            i = peak_bins[j]
            ap = mag[i + 1]
            am = mag[i]
            amm = mag[i - 1]
            v20 = (ap - np.float32(2.0) * am) - amm
            if v20 != np.float32(0.0):
                off = (ap - amm) / (v20 + v20)
            else:
                off = np.float32(0.0)
            if off > np.float32(0.7):
                off = np.float32(0.7)
            elif off < np.float32(-0.7):
                off = np.float32(-0.7)
            reg_offset[j] = np.float32(i) + off
        b = 0
        best = mag[0]
        for i in range(1, peak_bins[0] + 1):
            if mag[i] < best:
                best = mag[i]
                b = i
        reg_start[0] = b
        for j in range(cnt - 1):
            s = peak_bins[j]
            e = peak_bins[j + 1]
            b = s
            best = mag[s]
            i = s + 1
            while i < e:
                if mag[i] < best:
                    best = mag[i]
                    b = i
                i += 1
            reg_end[j] = b
            reg_start[j + 1] = b
        pk_last = peak_bins[cnt - 1]
        b = pk_last
        best = mag[pk_last]
        for i in range(pk_last + 1, NB):
            if mag[i] < best:
                best = mag[i]
                b = i
        reg_end[cnt - 1] = b
        for j in range(cnt):
            s = reg_start[j]
            e = reg_end[j]
            if e > s:
                mn = env[s]
                mx = env[s]
                for i in range(s + 1, e):
                    v = env[i]
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                if mn > np.float32(0.0):
                    ratio = mx / mn
                else:
                    ratio = np.float32(1.5)
                for i in range(s, e):
                    region_gain[i] = ratio
        return cnt
except Exception:
    _find_peaks_nb = None

_VDSP_MOD = None
_VDSP_FAST = None
_VDSP_TRIED = False


def _vdsp_mod():
    """Lazy ctypes Accelerate module (route B FFT); None if unavailable."""
    global _VDSP_MOD, _VDSP_TRIED
    if not _VDSP_TRIED:
        _VDSP_TRIED = True
        try:
            from . import vdsp
            _VDSP_MOD = vdsp if vdsp.HAVE_VDSP else None
        except Exception:
            _VDSP_MOD = None
    return _VDSP_MOD


def _vdsp_fast_mod():
    """Lazy vdsp_fast (persistent-scratch vDSP wrappers)."""
    global _VDSP_FAST
    if _VDSP_FAST is None and _vdsp_mod() is not None:
        try:
            from . import vdsp_fast
            _VDSP_FAST = vdsp_fast
        except Exception:
            _VDSP_FAST = False
    return _VDSP_FAST or None


if ROUTE_B_FAST:
    EXACT_FFT = False

#: fold-smoothing alpha (@0x168D.., expf(-1/(1.5e-5*sr)) at 48 kHz)
_FOLD_ALPHA = F32(0.24935225)

#: the renderer builds exactly one InterpTable(8192, 6, 16.0); cache it (build
#: cost is ~3 s) — see pyradius.vocoder_ops for the table semantics.
_INTERP_TBL = {}

#: noise-template cache, keyed by (seed, size): deterministic and rebuilt per
#: render otherwise (2.1M-element LCG loop == 3.3 s).
_NOISE_TMPL = {}


def _interp_table(X, levels, quality):
    key = (int(X), int(levels), float(quality))
    tbl = _INTERP_TBL.get(key)
    if tbl is None:
        tbl = InterpTable(key[0], key[1], key[2])
        _INTERP_TBL[key] = tbl
    return tbl


def _c_mod(a: int, b: int) -> int:
    """C truncating remainder (sign follows the dividend)."""
    return int(math.fmod(a, b))


def _seg_minmax(x, starts, ends):
    """Per-segment min/max over each [start, end) window (C ev_find_peaks).

    Carries the same ``np.minimum.reduceat`` pitfall as :func:`_seg_first_min`:
    reduceat runs its LAST segment to the end of the array, so env[nbins-1]
    leaked into the final region's min/max and skewed its gain ratio
    (observed 1.00836 vs the C's 1.00765 on the last 10 bins).  Segments are
    contiguous, so truncating the input to ends[-1] makes reduceat exact.
    """
    x = np.asarray(x, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if starts.size == 0:
        return x[:0], x[:0]
    if starts.size > 1 and not np.array_equal(starts[1:], ends[:-1]):
        mn = np.array([x[s:e].min() for s, e in zip(starts, ends)],
                      dtype=np.float32)
        mx = np.array([x[s:e].max() for s, e in zip(starts, ends)],
                      dtype=np.float32)
        return mn, mx
    t = x[:int(ends[-1])]
    return np.minimum.reduceat(t, starts), np.maximum.reduceat(t, starts)


def _seg_first_min(x, starts, ends):
    """First index of the minimum over each [start, end) window (C MinIndex).

    MinIndex uses strict ``<`` so the FIRST occurrence wins; the naive
    argmin-per-slice / searchsorted formulation picks the last instead.

    ``np.minimum.reduceat`` runs its LAST segment to the end of the array and
    ignores the final ``end``, so the array is truncated to ``ends[-1]`` first.
    Without that truncation the final peak-to-peak segment was reduced over
    [start, len) instead of [start, end); its minimum then matched no index in
    the segment, the scatter below left the entry at its 0 default, and the
    whole last region silently lost its boundary (and so its gain).
    """
    x = np.asarray(x, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    lengths = np.asarray(ends - starts, dtype=np.int64)
    if (lengths <= 0).any():
        # Degenerate window: MinIndex over n<=0 reads x[0] in the C, but callers
        # only pass peak-to-peak windows, which are never empty.
        raise ValueError("empty segment in _seg_first_min")
    # Segments produced by ev_find_peaks are contiguous (ends[i] == starts[i+1]),
    # so a single truncated reduceat is exact; fall back to per-segment reduction
    # if that ever stops holding.
    if starts.size > 1 and not np.array_equal(starts[1:], ends[:-1]):
        vals = np.array([x[s:e].min() for s, e in zip(starts, ends)],
                        dtype=np.float32)
    else:
        vals = np.minimum.reduceat(x[:int(ends[-1])], starts)
    exp = np.repeat(vals, lengths)
    idx_all = np.repeat(starts, lengths) + (
        np.arange(int(lengths.sum()), dtype=np.int64)
        - np.repeat(np.cumsum(lengths) - lengths, lengths))
    eq = np.flatnonzero(x[idx_all] == exp)
    region_of = np.repeat(np.arange(starts.size), lengths)
    got = region_of[eq]
    keep = np.concatenate([[True], np.diff(got) != 0])
    out = np.zeros(starts.size, dtype=np.int64)
    out[got[keep]] = idx_all[eq[keep]]
    return out


class VocoderState:
    """Container/persistent state mirroring ``rx_vc_state``."""

    def __init__(self, sr: int, nch: int, precision: int = 2):
        if sr == 44100:
            self.N, self.nb_bins, self.m_fft, self.mb_bins = 8192, 4097, 2048, 1025
            self.n_write, self.hop = 6599, 3299
        else:
            self.N, self.nb_bins, self.m_fft, self.mb_bins = 16384, 8193, 4096, 2049
            self.n_write, self.hop = 7184, 3592
        self.sr = int(sr)
        self.nch = int(nch)
        self.precision = int(precision)
        self.nbands = 4
        self.trans_sens = F32(1.0)
        self.quality = F32(37.0)
        self.ratio = 1.0
        self.step_base = 222.0
        self.ring_out_len_1232 = self.sr + 65536

        nb, N, NB, n_write, hop = self.nbands, self.N, self.nb_bins, self.n_write, self.hop
        cap = 4 * n_write
        self.ring_cap = cap
        self.ring = np.zeros((nb, cap), dtype=np.float32)

        t = get_tables(self.sr)
        if self.sr == 44100:
            self.win_table = t["win"].reshape(4, 6599).copy()
            self.synth_a = np.asarray(t["synth_a"], dtype=np.float32).copy()
            self.synth_b = np.asarray(t["synth_b"], dtype=np.float32).copy()
            self.v8_len_a, self.v9_len_b = 2688, 1152
        else:
            self.win_table = t["win"].reshape(4, 2 * hop).copy()
            self._build_synth_windows()

        # -- persistent cursor / geometry (rx_vc_init) --
        self.cursor_1376 = 0
        self.acc_1440 = 0.0
        self.pos_1384 = 0
        self.writepos_1224 = 0
        self.hop_1420 = hop
        self.granule_1400 = hop
        self.out_step_1404 = hop
        self.prev_granule_1408 = hop
        self.step_1412 = 0
        self.comp_1432 = 0.0
        self.transient_state_2488 = 0
        self.transient_flag_1208 = 0
        self.buffered_1352 = 1
        self.g_index = 0
        self.fed_total = 0
        self.out_total = 0
        self.out_reported = 0
        self.out_read_pos = 0
        # FIX #2 (tools/libradius_vc_oob_fix.md): nb_bins, not RX_VC_V4_COPY_LEN.
        self.copy_len_594 = NB
        self._sched_cdel = -1
        self.noise_slot_958 = 0
        self.ramp_scalar_3580 = F32(0.2)
        # route-B C carrier (built lazily on first use; see _cg_attach)
        self._cg = None
        self._cg_ok = False
        self._cg_vg = None
        self._cg_ptr = None
        self._cg_p = None

        # -- containers --
        self.mag = np.zeros((nb, NB), dtype=np.float32)
        self.magCopy = np.zeros((nb, NB), dtype=np.float32)
        self.mask = np.zeros((nb, NB), dtype=np.float32)
        self.maskCopy = np.zeros((nb, NB), dtype=np.float32)
        self.phase = np.zeros((nb, NB), dtype=np.float32)
        self.cart = np.zeros(N + 2, dtype=np.float32)
        self.frame = np.zeros(N, dtype=np.float32)
        self.acc = np.zeros(N, dtype=np.float32)
        self._vf_cart = None
        self._vf_out = None
        self.acs_env = np.zeros((self.nch, NB), dtype=np.float32)
        self.region_gain = np.zeros((self.nch, NB), dtype=np.float32)
        self.noise_phase = np.zeros((self.nch, NB), dtype=np.float32)
        self.noise_gain = np.zeros((self.nch, NB), dtype=np.float32)
        self.noise_slot_count_959 = 128
        self.noise_template = np.zeros(2 * 128 * NB, dtype=np.float32)
        self.noise_weight_3472 = np.power(
            np.arange(NB, dtype=np.float32) / F32(NB), F32(0.2)).astype(np.float32)
        self.sync_weight_3504 = np.zeros(NB, dtype=np.float32)
        self.gain_mean_3808 = np.zeros(NB, dtype=np.float32)
        self.dir_cur = np.zeros((self.nch, NB), dtype=np.float32)
        self.dir_prev = np.zeros((self.nch, NB), dtype=np.float32)
        self.seg_bound_1830 = np.full(6, NB, dtype=np.int32)
        self.peak_bins = np.zeros(NB, dtype=np.int32)
        self.reg_start = np.zeros(NB, dtype=np.int32)
        self.reg_end = np.zeros(NB, dtype=np.int32)
        self.reg_offset = np.zeros(NB, dtype=np.float32)
        self.scratch_8a8 = np.zeros(NB, dtype=np.float32)
        self.b_710 = np.zeros(NB, dtype=np.float32)
        self.mask_table_8f8 = np.zeros(NB, dtype=np.float32)
        self.sync_weight_buf = np.zeros(NB, dtype=np.float32)
        self.peak_count = 0
        self.sync_sens_3496 = F32(0.0)
        self.pitch_freq_169 = F32(0.0)
        self.pitch_metric_168 = F32(0.0)
        self.scratch_f = np.zeros(8, dtype=np.float32)
        self.scratch_i = np.zeros(8, dtype=np.int32)
        self.win1 = np.zeros(N, dtype=np.float32)
        self.win2 = np.zeros(N, dtype=np.float32)
        self.out_ring = np.zeros((self.nch, self.ring_out_len_1232), dtype=np.float32)
        self.edge_gain = np.array([1.15, 1.30, 1.05, 1.10], dtype=np.float32)
        self.rng = SimpleRand(1)
        self._xo = None
        self._xo_fir = t["fir"]
        # AFC persistent state (rx_formant_state_init: env = 0, gain_env = 1)
        self.fm = FormantState(dict(active=1, mode_freq=0, mode_rms=1,
                                    ratio=F32(1.0), width=F32(1.0),
                                    strength=F32(1.0), freq_lo=F32(40.0),
                                    freq_hi=F32(800.0), nb_bins=self.nb_bins,
                                    n_fft=self.N, m_fft=self.m_fft,
                                    m_bins=self.mb_bins, f580=int(self.step_base),
                                    sr=float(self.sr), prec_mode=2,
                                    nb_bands=nb))

    def _build_synth_windows(self):
        """48 kHz: synth_win_A/B generated by formula (rx_vc_init).

        HanningWindow^0.7 with len_a = 2930, len_b = 1256 and the centre at
        ``n_write//2`` (== hop); the C comment "中心 N/2=3592" pins N to
        this[0x588] = n_write, which the measured non-zero spans
        [2127, 5056) / [2964, 4219) confirm.
        """
        N, n_write = self.N, self.n_write
        len_a, len_b = 2930, 1256
        self.v8_len_a, self.v9_len_b = len_a, len_b
        self.synth_a = np.zeros(n_write, dtype=np.float32)
        self.synth_b = np.zeros(n_write, dtype=np.float32)
        pi_f = F32(math.pi)
        off_b = (n_write // 2) - (len_b // 2)
        off_a = (n_write // 2) - (len_a // 2)
        scale_a = F32(math.sqrt(float(F32(len_b) / F32(len_a))))
        i = np.arange(len_b, dtype=np.float32)
        ang = (F32(2.0) * pi_f * (i + F32(0.5)) / F32(len_b)).astype(np.float32)
        val = (F32(0.5) * (F32(1.0) - np.cos(ang.astype(np.float64)).astype(np.float32))).astype(np.float32)
        self.synth_b[off_b:off_b + len_b] = np.power(
            val.astype(np.float64), 0.7).astype(np.float32)
        i = np.arange(len_a, dtype=np.float32)
        ang = (F32(2.0) * pi_f * (i + F32(0.5)) / F32(len_a)).astype(np.float32)
        val = (F32(0.5) * (F32(1.0) - np.cos(ang.astype(np.float64)).astype(np.float32))).astype(np.float32)
        self.synth_a[off_a:off_a + len_a] = (
            np.power(val.astype(np.float64), 0.7).astype(np.float32) * scale_a)

    # =====================================================================
    # route-B C carrier (granule mega-kernel)
    # =====================================================================
    def _cg_build(self) -> bool:
        """Construct the C-side carrier (vg_state) for this renderer.

        The struct holds raw pointers into this instance's numpy buffers, so it
        must be built *after* every buffer exists and is never rebuilt (the
        arrays are never reallocated).

        Deliberately does NOT consult the bit-exactness gate: the gate itself
        needs a carrier to run its checks against, so routing that through
        _cg_attach would recurse (granule_ok -> _certify_granule -> _cg_attach
        -> granule_ok).
        """
        try:
            n = _neon
            nch = self.nch
            vg = n.VGState()
            # The FFT destination buffers are st.cart (spectrum #1/#2) and
            # st.frame (inverse), matching _fft_fwd / _fft_inv below.
            n.gran_fft_init(ctypes.byref(vg.fwd), self.N, n._fp(self.cart))
            n.gran_fft_init(ctypes.byref(vg.inv), self.N, n._fp(self.frame))
            vg.N, vg.nb_bins = self.N, self.nb_bins
            vg.m_fft, vg.mb_bins = self.m_fft, self.mb_bins
            vg.n_write, vg.hop = self.n_write, self.hop
            vg.nch, vg.nbands, vg.ring_cap = self.nch, self.nbands, self.ring_cap
            vg.buffered_1352 = int(bool(self.buffered_1352))
            vg.ring_out_len, vg.sr, vg.precision = (self.ring_out_len_1232,
                                                    self.sr, self.precision)
            vg.v8_len_a, vg.v9_len_b = self.v8_len_a, self.v9_len_b
            vg.step_base, vg.ratio = float(self.step_base), float(self.ratio)
            vg.fold_alpha = float(_FOLD_ALPHA)
            vg.fold_c = float(F32(F32(1.0) - _FOLD_ALPHA))
            # NOTE: never build a pointer to a temporary array here — ctypes
            # stores only the address, so `_fp(np.ascontiguousarray(x))` would
            # dangle as soon as the temporary is collected.  Every buffer
            # pointed at below is a real attribute of self.
            vg.win0 = n._fp(self.win_table[0])
            vg.synth_a = n._fp(self.synth_a)
            vg.synth_b = n._fp(self.synth_b)
            vg.edge_gain = n._fp(self.edge_gain)
            vg.acc, vg.cart = n._fp(self.acc), n._fp(self.cart)
            vg.mag, vg.mask = n._fp(self.mag), n._fp(self.mask)
            vg.acs_env, vg.region_gain = n._fp(self.acs_env), n._fp(self.region_gain)
            vg.phase = n._fp(self.phase)
            vg.mag_copy, vg.mask_copy = n._fp(self.magCopy), n._fp(self.maskCopy)
            vg.peak_bins = n._ip(self.peak_bins)
            vg.reg_start, vg.reg_end = n._ip(self.reg_start), n._ip(self.reg_end)
            vg.reg_offset = n._fp(self.reg_offset)
            vg.scratch_8a8 = n._fp(self.scratch_8a8)
            vg.win1, vg.win2 = n._fp(self.win1), n._fp(self.win2)
            vg.out_ring = n._fp(self.out_ring)
            vg.sync_weight = n._fp(self.sync_weight_3504)
            self._cg_vg = vg
            self._cg_ptr = ctypes.byref(vg)
            # Cache the raw pointers used by the per-granule entry points:
            # building each one through ctypes casts costs ~2 us and there are
            # ~25k such calls per render.  Two alternating sets are needed
            # because the assembly loop runs per channel (mag/phase/win slices
            # are rebuilt with the current channel).
            self._cg_p = {
                "ring": n._fp(self.ring), "win": n._fp(self.win_table),
                "acc": n._fp(self.acc), "cart": n._fp(self.cart),
                "frame": n._fp(self.frame),
                "win1": n._fp(self.win1), "win2": n._fp(self.win2),
                "mag": [n._fp(self.mag[ch]) for ch in range(nch)],
                "phase": [n._fp(self.phase[ch]) for ch in range(nch)],
            }
            self._cg = vg
            self._cg_ok = True
            self._fm_attach()
            return True
        except Exception:
            self._cg = None
            self._cg_ok = False
            return False

    def _fm_attach(self) -> bool:
        """Build the formant C carrier over the FormantState scratch buffers.

        Only under route B and only once the kernel passed its own
        bit-exactness gate (neon.formant_ok), which is what keeps the struct's
        raw pointers from ever reaching a render that the gate did not cover.
        Idempotent: it creates vDSP setups, so it must NOT run per granule
        (`_cg_attach` is called from every `_ev_acs`).
        """
        try:
            fm = self.fm
            if getattr(fm, "_fm_car", None) is not None:
                return True
            if not _neon.formant_ok():
                return False
            car = _neon.FMState()
            rc = _neon.fm_state_init(
                ctypes.byref(car), fm.nb_bins, fm.n_fft, fm.m_fft, fm.m_bins,
                fm.nb_bands, fm.prec_mode,
                int(fm.gain_env.strides[0] // fm.gain_env.itemsize),
                float(fm.sr), float(fm.inv_scale),
                _neon._fp(fm.db), _neon._fp(fm.gscr), _neon._fp(fm.ker),
                _neon._fp(fm.env), _neon._fp(fm.gain_env),
                _neon._fp(fm.ffr), _neon._fp(fm.ffi))
            if rc != 0:
                return False
            fm._fm_car = car
            return True
        except Exception:
            return False

    def _cg_attach(self) -> bool:
        """Attach the C carrier if the extension passed its bit-exactness gate.

        Returns False when the extension is missing or the gate failed, in which
        case every C stage falls back to the route-B Python/numba code.  Only
        meaningful under route B: without PYR_FAST the default path must stay
        bit-identical to the C engine and never touches these kernels.
        """
        if self._cg is not None:
            if self._cg_ok:
                self._fm_attach()
            return self._cg_ok
        if not ROUTE_B_FAST or _neon is None or not _neon.granule_ok():
            return False
        return self._cg_build()

    @property
    def _cg_fast(self) -> bool:
        """True while the route-B C granule stages are usable."""
        return self._cg_ok and self._cg is not None

    def _cg_acs(self, ch):
        """_ev_acs entirely in C (except the fgwin band sum, still NEON)."""
        n = _neon
        n.gran_acs1(self._cg_ptr, ch, self.step_1412, self.transient_state_2488)
        self._ev_fgwin_x4(ch)
        self.cart[:] = self._fft_fwd(self.acc)
        n.gran_acs2(self._cg_ptr, ch, self._cg_p["cart"])
        self.peak_count = n.gran_find_peaks(self._cg_ptr, ch)
        self._trace("ACS", ch)

    # =====================================================================
    # setup / streaming
    # =====================================================================
    def set_ratio(self, semis: float, quality: float = 100.0):
        """pitch_chain(semis) + exp2 chain of tools/rx_vc_render.c."""
        pr = math.pow(2.0, semis / 12.0)
        s = F32(F32(12.0) * F32(math.log2(float(F32(pr)))))
        self.ratio = float(math.pow(2.0, float(s) / 12.0))
        self.comp_1432 = self.ratio * float(self.hop_1420)
        self.fm.cfg["ratio"] = F32(self.ratio)
        return self.ratio

    def start_streaming(self):
        """rx_vc_start_streaming (0x1668A0)."""
        self.rng = SimpleRand(1)
        self.transient_flag_1208 = 1
        self.noise_slot_958 = 0
        self.ramp_scalar_3580 = F32(0.2)
        # The template is a pure function of (seed=1, size); every render rebuilds
        # it with a fresh SimpleRand(1), so cache it (the LCG loop over 2.1M
        # elements cost 3.3 s per render).
        key = (1, self.noise_template.size)
        cached = _NOISE_TMPL.get(key)
        if cached is None:
            noise_template_fill(self.noise_template, self.rng)
            _NOISE_TMPL[key] = self.noise_template.copy()
        else:
            self.noise_template[:] = cached
            # advance the LCG to the same state the fill loop would leave:
            # O(log n) affine jump-ahead (SimpleRand.skip), bitwise-equal to
            # n next() calls (verified vs sequential in tests).
            self.rng.skip(self.noise_template.size)
        self.acs_env[:] = 0.0
        self.noise_phase[:] = 0.0
        self.noise_gain[:] = 0.0
        self._trace("StartStreaming", -1)

    def _trace(self, event, band):
        if getattr(self, "_tracef", None) is not None:
            self._tracef.write(f"{event} {band}\n")

    # =====================================================================
    # events
    # =====================================================================
    def _ev_fill_granule(self, ch):
        """FillGranule (acc reset + band sum)."""
        if self._cg_fast:
            _neon.gran_fill(self._cg_ptr, 0, self.cursor_1376,
                            self._cg_p["ring"], self._cg_p["win"])
        else:
            self.acc[:] = 0.0
            _fill_op(self.ring, self.win_table, self.acc, mode="fg",
                     hop=self.hop_1420, n_write=self.n_write, n_bands=self.nbands,
                     cursor=self.cursor_1376,
                     buffered=bool(self.buffered_1352) if self.cursor_1376 >= self.hop_1420 else False,
                     cap_scalar=self.ring_cap, ring_cap0_a=self.ring_cap // 2,
                     ring_cap0_b=self.ring_cap // 2)
        self._trace("FillGranule", ch)

    def _ev_fgwin_x4(self, ch):
        """FillGranuleWin x nbands (acc already zeroed by the caller)."""
        if self._cg_fast:
            _neon.gran_fill(self._cg_ptr, 1, self.cursor_1376,
                            self._cg_p["ring"], self._cg_p["win"])
        else:
            _fill_op(self.ring, self.win_table, self.acc, mode="fgw",
                     hop=self.hop_1420, n_write=self.n_write, n_bands=self.nbands,
                     n5d0=self.N, cursor=self.cursor_1376, buffered=True,
                     cap_scalar=self.ring_cap, ring_cap0_a=self.ring_cap // 2,
                     ring_cap0_b=self.ring_cap // 2, gain=1.0)
        for band in range(self.nbands):
            self._trace("FGWin", band)

    def _ev_acs_front(self, ch):
        """_ev_acs steps a-c: win0 multiply, spectrum #1, envelope IIR.

        Split out so the route-B C driver can implement exactly this half in C
        while the Python path keeps calling it (see _ev_acs).
        """
        NB, n_win = self.nb_bins, self.n_write
        win0 = self.win_table[0]
        tau = _bits(0x3D4CCCCD) if self.transient_state_2488 == 2 else _bits(0x3DCCCCCD)
        hopf = F32(self.step_1412) if self.step_1412 > 0 else F32(288.0)
        iir_a = time_to_iir_a(tau, F32(F32(self.sr) / hopf))
        self.acc[:n_win] = (self.acc[:n_win] * win0[:n_win]).astype(np.float32)
        cart = self._fft_fwd(self.acc)
        env = self.acs_env[ch]
        if _acs_mag_env_nb is not None and not EXACT_FFT:
            _acs_mag_env_nb(cart, env, NB, iir_a)
        else:
            re, im = cart[0::2], cart[1::2]
            m = np.sqrt((re * re + im * im).astype(np.float32)).astype(np.float32)
            env[:] = _fma_arr((m - env).astype(np.float32), iir_a, env)
        self.acc[:] = 0.0
        return cart

    def _ev_acs(self, ch):
        """AnalyzeChannelSpectrum (spectrum #1 + envelope, FGWin, spectrum #2).

        Route B: the C driver runs window/FFT/envelope/polar/clamp and the peak
        search in one ctypes call (stages a+b+c of the migration).
        """
        self._cg_attach()
        if self._cg_fast:
            self._cg_acs(ch)
            return
        self._ev_acs_front(ch)
        self._ev_fgwin_x4(ch)
        cart2 = self._fft_fwd(self.acc)
        self.cart[:] = cart2
        NB = self.nb_bins
        mag, mask = self.mag[ch], self.mask[ch]
        ctp = _ops.cart_to_polar(cart2)
        mag[:NB] = ctp["mag"][:NB]
        mask[:NB] = ctp["phase"][:NB]
        # Threshold_LT_InPlace(mag, n_polar, 0x2B8CBCCC) — lower clamp only
        np.copyto(mag, np.maximum(mag, _bits(0x2B8CBCCC)))
        self._ev_find_peaks(ch)
        self._trace("ACS", ch)

    def _ev_find_peaks(self, ch):
        """Peaks + parabolic refinement + region boundaries + region gains."""
        NB = self.nb_bins
        mag = self.mag[ch]
        env = self.acs_env[ch]
        if self._cg_fast:
            self.peak_count = _neon.gran_find_peaks(self._cg_ptr, ch)
            return
        if _find_peaks_nb is not None and not EXACT_FFT:
            self.peak_count = _find_peaks_nb(
                mag, env, self.peak_bins, self.reg_offset,
                self.reg_start, self.reg_end, self.region_gain[ch], NB)
            return
        hits = np.nonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:]))[0] + 1
        cnt = hits.size
        if cnt == 0:
            self.peak_bins[0] = NB // 2
            self.reg_offset[0] = F32(NB // 2)
            self.reg_start[0] = 0
            # FIX #1 (tools/libradius_vc_oob_fix.md): the C sets reg_end[0] =
            # nbins, which makes UnwrapPhase's MaxIndex read mag[nbins].
            self.reg_end[0] = NB - 1
            self.peak_count = 1
            return
        self.peak_bins[:cnt] = hits
        ap, am, amm = mag[hits + 1], mag[hits], mag[hits - 1]
        # v20 = (ap - 2*am) - amm (two separate roundings, no fma);
        # off = (ap - amm)/(v20 + v20), clamped to +-0.7; position = i + off.
        v20 = ((ap - (F32(2.0) * am).astype(np.float32)).astype(np.float32)
               - amm).astype(np.float32)
        off = np.zeros(cnt, dtype=np.float32)
        nz = v20 != 0.0
        off[nz] = ((ap[nz] - amm[nz]) / (v20[nz] + v20[nz])).astype(np.float32)
        off = np.clip(off, F32(-0.7), F32(0.7))
        self.reg_offset[:cnt] = (hits.astype(np.float32) + off).astype(np.float32)
        self.peak_count = cnt

        mids = _seg_first_min(mag, hits[:-1], hits[1:])
        self.reg_start[0] = int(np.argmin(mag[:hits[0] + 1]))
        self.reg_end[:cnt - 1] = mids
        self.reg_start[1:cnt] = mids
        pk_last = int(hits[-1])
        self.reg_end[cnt - 1] = pk_last + int(np.argmin(mag[pk_last:]))

        rs = np.ascontiguousarray(self.reg_start[:cnt])
        re_ = np.ascontiguousarray(self.reg_end[:cnt])
        nonempty = re_ > rs
        if nonempty.any():
            s2, e2 = rs[nonempty], re_[nonempty]
            mn, mx = _seg_minmax(env, s2, e2)
            ratio = np.where(mn > 0.0, (mx / mn).astype(np.float32), F32(1.5))
            lens = e2 - s2
            idx = np.repeat(s2, lens) + (np.arange(int(lens.sum()))
                                         - np.repeat(np.cumsum(lens) - lens, lens))
            self.region_gain[ch][idx] = np.repeat(ratio, lens)

    def _ev_unwrap(self, ch):
        """UnwrapPhase (region phase unwrap)."""
        out = _ops.unwrap_phase(
            mask=self.mask[ch], mask_copy=self.maskCopy[ch],
            mag_copy=self.magCopy[ch], reg_start=self.reg_start[:self.peak_count],
            reg_end=self.reg_end[:self.peak_count],
            reg_prev_peak=self.peak_bins[:self.peak_count],
            reg_offset=self.reg_offset[:self.peak_count],
            scratch=self.scratch_8a8, phase=self.phase[ch],
            f1=self.prev_granule_1408, f2=self.step_1412, f3=self.N,
            max_bin=self.nb_bins, copy_len=self.copy_len_594)
        self.scratch_8a8[:] = out["scratch"]
        self.phase[ch][:] = out["phase"]
        self._trace("Unwrap", ch)

    def _ev_apc(self, ch):
        """ApplyPitchCoherence (gated on precision <= 9 and transSens != 0)."""
        if self.precision > 9 or self.trans_sens == 0.0:
            return
        args = dict(precision=self.precision, trans_sens=self.trans_sens,
                    total_ratio=self.ratio, transient_state=self.transient_state_2488,
                    n_fft=self.N, sr=self.sr, f580=self.hop_1420,
                    f584=self.hop_1420, acc_fc0=F32(0.0),
                    coh_center=F32(0.0), seg_count=5,
                    peak_count=self.peak_count, vector_fmaf_region=False)
        out = _ops.apply_pitch_coherence(
            args=args, env=self.acs_env[ch], peak_bins=self.peak_bins,
            reg_start=self.reg_start, reg_end=self.reg_end,
            seg_bound=self.seg_bound_1830, phase=self.mask[ch],
            phase_mod=self.phase[ch], dir_cur=self.dir_cur[ch],
            dir_prev=self.dir_prev[ch], region_gain=self.region_gain[ch],
            noise_wt=self.noise_weight_3472,
            a3=F32(self.pitch_freq_169), a4=F32(self.pitch_metric_168))
        self.phase[ch][:] = out["phase_mod"]
        self.dir_cur[ch][:] = out["dir_cur"]
        self.dir_prev[ch][:] = out["dir_prev"]
        self._trace("APC", ch)

    def _ev_pull_to_peak(self, ch):
        """Loose phase locking (inline @0x168B0C)."""
        cnt = self.peak_count
        if cnt < 1:
            return
        if self._cg_fast:
            _neon.gran_pull(self._cg_ptr, ch, self.prev_granule_1408,
                            self.step_1412)
            self._trace("PullToPeak", ch)
            return
        # Every step rounds to float32 on its own in the C (0.5f*v74 then
        # 0.5f+..., not one double evaluation rounded once).  Collapsing them to
        # double arithmetic changed lo/hi/inv_range by an ULP, which shifted the
        # interpolated lock weight w and so the phase of ~1 bin per granule.
        f1 = F32(float(self.prev_granule_1408))
        f2 = F32(float(self.step_1412))
        if f1 <= 0.0:
            return
        ratio = F32(f2 / f1)
        v74 = F32(0.0)
        if ratio > 1.0:
            r_clamp = F32(1.0) if ratio >= 4.0 else F32(F32(ratio - F32(1.0)) / F32(3.0))
            v74 = F32(math.sqrt(float(r_clamp)))
        lo = F32(F32(0.5) + F32(F32(0.5) * v74))
        hi = F32(F32(0.5) + F32(F32(4.0) * v74))
        inv_range = F32(F32(1.0) / F32(hi - lo)) if hi > lo else F32(0.0)
        rs = np.ascontiguousarray(self.reg_start[:cnt])
        re_ = np.ascontiguousarray(self.reg_end[:cnt])
        if _pull_nb is not None and not EXACT_FFT:
            _pull_nb(self.phase[ch], self.mask[ch], self.region_gain[ch],
                     rs, re_,
                     np.ascontiguousarray(self.peak_bins[:cnt]),
                     cnt, lo, hi, inv_range)
            self._trace("PullToPeak", ch)
            return
        lens = re_ - rs
        nz = lens > 0
        if not nz.any():
            self._trace("PullToPeak", ch)
            return
        Lb = lens[nz]
        bins = np.repeat(rs[nz], Lb) + (np.arange(int(Lb.sum()))
                                        - np.repeat(np.cumsum(Lb) - Lb, Lb))
        pk = np.repeat(np.ascontiguousarray(self.peak_bins[:cnt])[nz], Lb)
        g = self.region_gain[ch][bins]
        w = np.where(g >= hi, F32(1.0), np.zeros_like(g))
        mid = (g > lo) & (g < hi)
        w = np.where(mid, (F32(g - lo) * inv_range).astype(np.float32), w)
        act = (w > 0.0) & (bins != pk)
        if act.any():
            phase = self.phase[ch]
            mask = self.mask[ch]
            b, p, ww = bins[act], pk[act], w[act]
            pm_bin = phase[b].copy()
            t0 = _wrap_pi((phase[p] - pm_bin).astype(np.float32))
            t1 = _wrap_pi((mask[b] - mask[p]).astype(np.float32))
            phase[b] = _fma_arr(ww, _wrap_pi((t0 + t1).astype(np.float32)), pm_bin)
        self._trace("PullToPeak", ch)

    def _ev_sync(self):
        """SynchronizeStereoPhases."""
        if self.nch < 2 or self.peak_count < 1:
            return
        cnt = self.peak_count
        if self._cg_fast and self.nch == 2:
            _neon.gran_sync2(self._cg_ptr, float(self.sync_sens_3496))
            self._trace("Sync", -1)
            return
        if self.nch == 2:
            # fast path: identical output to the per-peak C shape (see
            # vocoder_ops.synchronize_stereo_phases_nch2_fast), but one pass
            # instead of ~2000 Python iterations per granule.
            out = _ops.synchronize_stereo_phases_nch2_fast(
                np.stack([self.mag[0], self.mag[1]]),
                np.stack([self.mask[0], self.mask[1]]),
                np.stack([self.phase[0], self.phase[1]]),
                peaks=self.peak_bins[:cnt], pk_start=self.reg_start[:cnt],
                pk_end=self.reg_end[:cnt], peak_count=cnt,
                sens=self.sync_sens_3496)
        else:
            out = _ops.synchronize_stereo_phases(
                np.stack([self.mag[c] for c in range(self.nch)]),
                np.stack([self.mask[c] for c in range(self.nch)]),
                np.stack([self.phase[c] for c in range(self.nch)]),
                peaks=self.peak_bins[:cnt], pk_start=self.reg_start[:cnt],
                pk_end=self.reg_end[:cnt], peak_count=cnt,
                sens=self.sync_sens_3496, nch=self.nch)
        for c in range(self.nch):
            self.phase[c][:] = out["dst"][c]
        self.sync_weight_buf[:] = out["weight"]
        self._trace("Sync", -1)

    def _ev_noise_ops(self):
        """Substitute|Randomize gate (empty body in the C skeleton)."""
        return

    def _ev_rpt(self, b):
        """ResetPhasesForTransients."""
        if self._cg_fast:
            _neon.gran_rpt(self._cg_ptr, b, self.transient_flag_1208,
                           self.transient_state_2488, float(self.ratio))
            self._trace("RPT", b)
            return
        out = _ops.reset_phases_for_transients(
            mode_4b8=self.transient_flag_1208,
            proc_mode_9b8=self.transient_state_2488,
            scale_20=self.ratio, len_594=self.nb_bins,
            n9bc=0, n9c0=0, u_70=self.sr, n578=self.granule_1400,
            n590=self.hop_1420, n5d0=self.N, n5d4=self.nb_bins,
            avg_m1=0.0, avg_0=0.0, avg_p1=0.0, use_p568=False,
            mag=self.mag[b], b_710=self.b_710, mask=self.mask[b],
            r_start=self.reg_start, r_end=self.reg_end, r_bin=self.peak_bins,
            phase=self.phase[b], mask_table=self.mask_table_8f8,
            n_out=self.nb_bins)
        self.phase[b][:] = out["phase"]
        self.mask_table_8f8[:] = out["mask_table"]
        self._trace("RPT", b)

    def _ev_afc(self, b):
        """ApplyFormantCorrection."""
        self.fm.cfg["f580"] = (int(self.prev_granule_1408) if self.prev_granule_1408 > 0
                               else int(self.step_base))
        self.mag[b][:] = self.fm.apply(self.mag[b], band=b)
        self._trace("AFC", b)

    def _ev_fold_iir_sub(self):
        """fftshift + forward/backward IIR smoothing + Sub_InPlace.

        The C runs the recurrences sample-serially in f32; scipy's lfilter is
        used here with the identical transfer function (alpha = 0.24935225,
        forward over the whole frame, backward from N-2 down to half-hop).
        Route B: _fold_nb (Python) or the bit-equal C driver.
        """
        N, hop = self.N, self.hop_1420
        w1, w2 = self.win1, self.win2
        if self._cg_fast:
            n = _neon
            n.gran_fold(self._cg_p["acc"], self._cg_p["win1"],
                        self._cg_p["win2"], N, hop, float(_FOLD_ALPHA),
                        float(F32(F32(1.0) - _FOLD_ALPHA)))
            return
        if _fold_nb is not None and not EXACT_FFT:
            _fold_nb(self.acc, w1, w2, N, hop,
                     _FOLD_ALPHA, float(F32(F32(1.0) - _FOLD_ALPHA)))
            return
        half = N >> 1
        src = self.acc
        w1[half:N] = src[:half]
        w1[:half] = src[half:N]
        w2[half:N] = src[:half]
        w2[:half] = src[half:N]
        c = float(F32(F32(1.0) - _FOLD_ALPHA))
        bcoef, acoef = [float(_FOLD_ALPHA)], [1.0, -c]
        y, _ = lfilter(bcoef, acoef, w1.astype(np.float64),
                       zi=np.array([c * float(w1[0])]))
        w1[:] = y.astype(np.float32)
        stop = half - hop
        seg = w1[stop:]
        yr, _ = lfilter(bcoef, acoef, seg[::-1].astype(np.float64),
                        zi=np.array([c * float(seg[-1])]))
        w1[stop:] = yr[::-1].astype(np.float32)
        i = np.arange(2 * hop)
        w2[stop + i] = (w2[stop + i] - w1[stop + i]).astype(np.float32)

    def _ev_oac(self, ch):
        """OverlapAddChannel."""
        a3 = int(vc_calc_sched(self.g_index + 1, self.ratio, self.step_base)[0]
                 - vc_calc_sched(self.g_index, self.ratio, self.step_base)[0])
        if a3 <= 0:
            a3 = self.step_1412
        f1412 = max(self.step_1412, 0)
        out = _ops.overlap_add_channel(
            win1=self.win1, win2=self.win2, synth_a=self.synth_a,
            synth_b=self.synth_b, out_ring=self.out_ring[ch],
            edge_gain=self.edge_gain, ch=ch, a3=a3, a4=0, a5=0,
            frame_n=self.N, p=self.precision,
            A=3300 if self.sr == 44100 else 3592,
            v8=self.v8_len_a, v9=self.v9_len_b, f1412=f1412,
            cursor=self.pos_1384, hop=self.hop_1420,
            ring_len=self.ring_out_len_1232,
            f1224=self.pos_1384 - a3 + self.hop_1420, n_write=self.n_write)
        self.win1[:] = out["win1"]
        self.win2[:] = out["win2"]
        self.out_ring[ch][:] = out["out_ring"]
        self._trace("OAC", ch)

    def _ev_assembly(self):
        """RPT -> Copy1 -> Copy2 -> AFC -> P2C -> FFTInv -> fold -> OAC per channel."""
        for r in range(self.nch):
            self._ev_rpt(r)
            self.maskCopy[r][:self.nb_bins] = self.mask[r][:self.nb_bins]
            self._trace("Copy1", r)
            self.magCopy[r][:self.nb_bins] = self.mag[r][:self.nb_bins]
            self._trace("Copy2", r)
            self._ev_afc(r)
            mag, phase, n = self.mag[r], self.phase[r], self.nb_bins
            if self._cg_fast:
                _neon.gran_p2c(self._cg_p["mag"][r], self._cg_p["phase"][r],
                               self._cg_p["cart"], n)
            elif _p2c_nb is not None and not EXACT_FFT:
                _p2c_nb(mag, phase, self.cart, n)
            else:
                cosv = np.cos(phase[:n].astype(np.float64)).astype(np.float32)
                sinv = np.sin(phase[:n].astype(np.float64)).astype(np.float32)
                self.cart[0:2 * n:2] = (mag[:n] * cosv).astype(np.float32)
                self.cart[1:2 * n:2] = (mag[:n] * sinv).astype(np.float32)
            self._trace("P2C", r)
            self.acc[:] = self._fft_inv(self.cart)
            self._trace("FFTInv", r)
            self._ev_fold_iir_sub()
            self._ev_oac(r)

    def _ev_cursor_advance(self):
        """LABEL_144 cursor advance."""
        pos_old = self.pos_1384
        if self._sched_cdel >= 0:
            self.cursor_1376 += self._sched_cdel
            self.transient_flag_1208 = 0
            self.out_total += self._sched_cdel
            self._sched_cdel = -1
            self._trace("ADV", -1)
            return
        self.writepos_1224 = self.pos_1384 + self.hop_1420
        self.cursor_1376 += self.granule_1400
        v158 = self.out_step_1404 if self.out_step_1404 > 1 else 1
        v157 = self.granule_1400 if self.granule_1400 > 1 else 1
        v9 = self.comp_1432 if v158 == self.out_step_1404 else float(v158)
        self.acc_1440 += v9
        self.pos_1384 = int(self.acc_1440 + (-0.5 if self.acc_1440 < 0.0 else 0.5))
        self.step_1412 = int(self.pos_1384 - pos_old)
        self.prev_granule_1408 = v157
        self.transient_flag_1208 = 0
        self.out_total += v157
        self._trace("ADV", -1)

    # =====================================================================
    # FFT wrappers
    # =====================================================================
    def _fft_fwd(self, src):
        if self._cg_fast:
            n = _neon
            n.gran_fft_run(ctypes.byref(self._cg_vg.fwd), n._fp(src),
                           self._cg_p["cart"])
            return self.cart
        if EXACT_FFT:
            return _fft_plan(self.N).fwd(src)
        vf = _vdsp_fast_mod()
        if vf is not None:
            if self._vf_cart is None or self._vf_cart.size != self.N + 2:
                self._vf_cart = np.empty(self.N + 2, dtype=np.float32)
            src32 = np.ascontiguousarray(src, dtype=np.float32)
            vf.fwd_r(src32, self._vf_cart, self.N)
            return self._vf_cart
        v = _vdsp_mod()
        if v is not None:
            return v.fft_fwd_r(np.asarray(src, dtype=np.float32))
        return fwd_fast(src)

    def _fft_inv(self, cart):
        if self._cg_fast:
            n = _neon
            n.gran_fft_inv_run(ctypes.byref(self._cg_vg.inv), n._fp(cart),
                               self._cg_p["frame"])
            return self.frame
        if EXACT_FFT:
            return _fft_plan(self.N).inv(cart)
        vf = _vdsp_fast_mod()
        if vf is not None:
            if self._vf_out is None or self._vf_out.size != self.N:
                self._vf_out = np.empty(self.N, dtype=np.float32)
            vf.inv_r(np.ascontiguousarray(cart, dtype=np.float32),
                     self._vf_out, self.N)
            return self._vf_out
        v = _vdsp_mod()
        if v is not None:
            return v.fft_inv_r(cart)
        return inv_fast(cart)

    # =====================================================================
    # granule pump / feed
    # =====================================================================
    def process_granule(self):
        """rx_vc_process_granule — schedule, warm-up g0..g7, 4-phase chain."""
        se_curr = vc_calc_sched(self.g_index, self.ratio, self.step_base)
        se_next = vc_calc_sched(self.g_index + 1, self.ratio, self.step_base)
        self.pos_1384 = int(se_curr[0])
        self.writepos_1224 = self.pos_1384 + self.hop_1420
        self._sched_cdel = int(se_curr[1])
        if se_curr[1] > 0:
            self.granule_1400 = int(se_curr[1])
        if se_curr[2] > 0:
            self.prev_granule_1408 = int(se_curr[2])
        if se_curr[3] > 0:
            self.step_1412 = int(se_curr[3])
        self.out_step_1404 = int(se_next[0]) - int(se_curr[0])
        ph = self.g_index & 3
        nch = self.nch
        ch1 = 1 % nch

        if self.g_index < 8:
            # warm-up (engine g0..g7):
            #  g0 empty | g1 FillG | g2 FillG+ACS(ch0) | g3 FillG+ACS(ch1)
            #  g4 FillG+Sync+assembly(no Unwrap) | g5 FillG | g6 FillG+ACS(ch0)
            #  g7 FillG+Unwrap+APC+ACS(ch1); steady state starts at g8
            g = self.g_index
            if g >= 1:
                self._ev_fill_granule(0)
            if g in (2, 3, 6):
                self._ev_acs(ch1 if g == 3 else 0)
            if g == 4:
                self._ev_sync()
                self._ev_noise_ops()
                self._ev_assembly()
            if g == 7:
                self._ev_unwrap(0)
                self._ev_apc(0)
                self._ev_pull_to_peak(0)
                self._ev_acs(ch1)
            self.g_index += 1
            self._sched_cdel = -1
            return

        if ph == 1:                                   # bare granule
            self._ev_fill_granule(0)
        elif ph == 2:
            self._ev_fill_granule(0)
            self._ev_acs(0)
        elif ph == 3:
            self._ev_fill_granule(0)
            self._ev_unwrap(0)
            self._ev_apc(0)
            self._ev_pull_to_peak(0)
            self._ev_acs(ch1)
        else:                                         # big granule (cycle end)
            self._ev_fill_granule(0)
            self._ev_fgwin_x4(ch1)
            self._ev_unwrap(ch1)
            self._ev_apc(ch1)
            self._ev_pull_to_peak(ch1)
            self._ev_sync()
            self._ev_noise_ops()
            self._ev_assembly()
        self.g_index += 1
        self._trace("PTS/CPM(skel)", -1)
        if ph == 0:
            self._sched_cdel = int(vc_calc_sched(self.g_index, self.ratio,
                                                 self.step_base)[1])
            self._ev_cursor_advance()
        else:
            self._sched_cdel = -1

    def feed(self, x, nin):
        """rx_vc_feed — crossover into the band rings, then pump granules."""
        if self._xo is None:
            self._xo = _ops.Crossover(self.sr, self.nbands)
        if (_ops._xover_zp_nb is not None and not EXACT_FFT):
            xo = self._xo
            x0 = np.ascontiguousarray(x[:nin, 0])
            z = np.concatenate([xo.hist, x0])
            xo.hist = z[-(xo.N - 1):]
            bands = np.empty((self.nbands, nin), dtype=np.float32)
            # Route B: the NEON xo_tree kernel is bit-identical to
            # _xover_zp_nb (self-certified in pyradius.neon.xo_tree_ok(),
            # which re-checks max|d| == 0 against the live numba kernel), so it
            # is a drop-in replacement.  If the extension is unavailable or
            # fails certification we stay on numba.
            if _neon is not None and _neon.xo_tree is not None and \
                    _neon.xo_tree_ok():
                _neon.xo_tree_run(z, xo.taps_rev, bands, nin, xo.N,
                                  self.nbands)
            else:
                _ops._xover_zp_nb(z, xo.taps_rev, bands, nin, xo.N,
                                  self.nbands)
            _ring_scatter_nb(self.ring, bands, self.fed_total, nin, 1023,
                             self.ring_cap)
        else:
            ch0 = np.ascontiguousarray(x[:nin, 0]).astype(np.float64)
            bands = self._xo.process(ch0)
            _ring_scatter_nb(self.ring, bands, self.fed_total, nin, 1023,
                             self.ring_cap)
        self.fed_total += nin
        while self.cursor_1376 + self.hop_1420 < self.fed_total:
            self.process_granule()
        made = self.out_total - self.out_reported
        self.out_reported = self.out_total
        return made

    def take_output(self, dst, max_frames):
        """rx_vc_take_output — drain the settled region of the output ring.

        The read window lags one granule step behind ``pos_1384 - hop`` so the
        next granule's overlap-add cannot race the read.
        """
        safe_end = self.pos_1384 - self.hop_1420 - self.out_step_1404
        avail = safe_end - self.out_read_pos
        if avail > max_frames:
            avail = max_frames
        if avail <= 0:
            return 0
        ring = self.ring_out_len_1232
        gpos = self.out_read_pos + np.arange(avail, dtype=np.int64)
        dst[:avail] = self.out_ring[:, gpos % ring].T
        self.out_read_pos += avail
        return int(avail)


# ---------------------------------------------------------------------------
# CLI driver (tools/rx_vc_render.c main)
# ---------------------------------------------------------------------------

def vc_render(st: VocoderState, x, trace=None):
    """Render ``x`` [nframes, nch] f32 through the vocoder chain.

    Mirrors the C driver: feed rhythm (first block 448 frames, then 1024; zero
    pad at EOS until ``out_n >= nframes``) and the Resampler drain loop
    (``writepos = max(pos - hop, 0)``, ``head = writepos - 100``,
    ``target_out = round(out_n + (head - drain_phase)/ratio + 0.5)``,
    chunks <= 4096, ``ph = fmod(drain_phase, ring_len)``,
    ``drain_phase += chunk_n*ratio``).

    Returns ``(out [nframes, nch] f32, stats dict)``.
    """
    if trace is not None:
        if isinstance(trace, str):
            st._tracef = open(trace, "w")
        else:
            st._tracef = trace
    st.start_streaming()
    st._cg_attach()
    nframes = int(x.shape[0])
    nch = st.nch
    ring_len = st.ring_out_len_1232
    ratio = st.ratio
    out = np.zeros((nframes, nch), dtype=np.float32)
    tbl = _interp_table(8192, 6, 16.0)
    drain_phase = 0.0
    out_n = 0
    pos = 0
    CH = 1024
    chunk = np.zeros((CH, nch), dtype=np.float32)
    while out_n < nframes:
        n = 448 if pos == 0 else 1024
        if pos >= nframes:
            chunk[:n] = 0.0
        elif pos + n > nframes:
            chunk[:nframes - pos] = x[pos:nframes]
            chunk[nframes - pos:n] = 0.0
        else:
            chunk[:n] = x[pos:pos + n]
        pos += n
        st.feed(chunk, n)
        while True:
            writepos = st.pos_1384 - st.hop_1420 if st.pos_1384 > st.hop_1420 else 0
            head = float(writepos - 100)
            if head <= drain_phase:
                break
            target_out = int(out_n + (head - drain_phase) / ratio + 0.5)
            chunk_n = target_out - out_n
            if chunk_n > 4096:
                chunk_n = 4096
            if out_n + chunk_n > nframes:
                chunk_n = nframes - out_n
            if chunk_n <= 0:
                break
            ph = math.fmod(drain_phase, ring_len)
            if ph < 0:
                ph += ring_len
            plane = interp_nsamples(tbl, st.out_ring, ring_len, ph, 0, chunk_n,
                                    ratio, F32(ratio))
            out[out_n:out_n + chunk_n] = plane.T
            out_n += chunk_n
            drain_phase += chunk_n * ratio
            if out_n >= nframes:
                break
    stats = {"out_n": out_n, "cursor": st.cursor_1376, "pos": st.pos_1384,
             "made": st.out_total, "drain_phase": drain_phase}
    if getattr(st, "_tracef", None) is not None and isinstance(trace, str):
        st._tracef.close()
        st._tracef = None
    return out, stats
