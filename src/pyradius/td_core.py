"""pyradius.td_core — RadiusImplTD (time-domain) pitch/time renderer port.

Mirrors libradius (single source of truth = the C sources):
  - src/engine/td_core.c        rx_td_init / build_env_tables / pick_env /
                                do_ola_engine / rx_td_render main loop + drain
  - src/ops/td_pitch.c          rx_td_pitch_set_options / analyze (FFT ACF +
                                whitening + peak search + parabola + clarity)
  - src/ops/td_octave.c         octave / sub-harmonic correction
  - src/ops/transients_info.c   TransientsInfo (Kaiser STFT + IIR + argmax)
  - src/engine/td_transient.c   rx_td_transient_pos wrapper

Float discipline
----------------
Everything is float32 unless the C source explicitly computes in double.
Transcendentals: numpy float32 ufuncs (np.exp/np.log/np.power) are
bit-identical to libm expf/logf/powf on this platform (verified against
ctypes-libm: 0/200000 mismatches), and numpy's float64 pow/cos/sin match
libm double exactly.  ``math.exp`` going through double does *not* match
expf, so expf calls use ``np.exp`` on a float32 scalar.

``a * b + c`` patterns are kept as separate numpy multiply/add (no fast-math
fusing) to mirror -ffp-contract=off.  fmaf appears only inside the C
resampler (pyradius.sampler, already bit-exact); the C TD chain has no other
explicit fmaf.
"""
from __future__ import annotations

import math
import os
import sys as _sys

import numpy as np

from .fft import plan as _fft_plan
from .sampler import InterpTable, interp_nsamples

PI = 3.14159265358979323846
_DBG = bool(os.environ.get("PYR_TD_DBG"))
_PDBG = bool(os.environ.get("PYR_TD_PDBG"))

#: TD_FAST=1 enables the approximate/high-throughput tier (pyradius.td_fast).
#: The default path (TD_FAST unset) is byte-for-byte unchanged and remains the
#: bit-exact reference; every approximation below is gated on this flag *and*
#: on the corresponding backend being available (see td_fast.avail()).
TD_FAST = os.environ.get("TD_FAST", "") == "1"
_td_fast_mod = None


def _fast_on(name: str) -> bool:
    """TD_FAST sub-switch: enabled unless explicitly set to 0/false/no."""
    v = os.environ.get("TD_FAST_" + name)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off")


if TD_FAST:
    try:
        from . import td_fast as _td_fast_mod
    except Exception:
        _td_fast_mod = None

# ---------------------------------------------------------------------------
# small helpers mirroring C integer semantics
# ---------------------------------------------------------------------------


def _c_u32(v: int) -> int:
    """(unsigned)v — 32-bit wraparound."""
    return v & 0xFFFFFFFF


def _wrap_ring(pos, rstart: int, rtot: int):
    """rx_td_wrap_ring for scalars/arrays (int64 math).

    C: pos in [0,rtot) -> pos; else rstart + ((pos - rtot) mod span).
    Python's % is already non-negative so both branches collapse.
    """
    a = np.asarray(pos, dtype=np.int64)
    span = (rtot - rstart) if (rtot > rstart) else 1
    ok = (a >= 0) & (a < rtot)
    return np.where(ok, a, rstart + ((a - rtot) % span)).astype(np.int64)


def _wrap_out(pos, out_len: int):
    return np.asarray(pos, dtype=np.int64) % out_len


def _ring2d_view(st):
    """Contiguous [nch, ring_tot] view for the pitch gather kernel, or None.

    Under TD_FAST the ring is already one 2-D array; every other path passes the
    per-channel row-view list, which is not contiguous as a matrix, so the
    gather kernel is skipped and the numpy gather path is used instead.
    """
    r = getattr(st, "_ring2d", None)
    return r




# ---------------------------------------------------------------------------
# TransientsInfo (src/ops/transients_info.c + src/engine/td_transient.c)
# ---------------------------------------------------------------------------

_TI_BANDS = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17, 20, 23, 27, 31,
             37, 43, 51, 62, 74, 89, 110, 139, 180]
_TI_BOUNDS = np.asarray(_TI_BANDS, dtype=np.int64)
_TI_WIN_SCALE2 = np.float32(2.7996967 * 2.7996967)
#: TD_FAST sub-switches.  Resolved once, at the *end* of this module — the
#: kernel certifications need TIState/TDState to exist (and td_fast re-imports
#: pyradius.td_core lazily), so they cannot run during this module's own import.
_FAST_TI = False
_FAST_INTERP = False
_FAST_OLA = False
_FAST_VDSP = False
_FAST_PITCH = False
_FAST_DS = 1
#: ACF/search index vector: N/2+1 for the largest pitch N (8192) plus slack,
#: sliced per call instead of allocating a fresh arange per granule.
_PITCH_ARANGE = np.arange(4096, dtype=np.int64)


def _ti_i0(z: float) -> float:
    s = 1.0
    t = 1.0
    for k in range(1, 60):
        t *= (z / 2.0 / k) * (z / 2.0 / k)
        s += t
    return s


def _ti_kaiser(i: int, n: int, beta: float) -> float:
    if n < 2:
        return 1.0
    a = (2.0 * float(i) / float(n - 1)) - 1.0
    r = 1.0 - a * a
    return _ti_i0(beta * math.sqrt(r if r > 0.0 else 0.0)) / _ti_i0(beta)


class TIState:
    """TransientsInfo port.  ``mix`` (the summed input channels) is supplied
    per chunk; the engine's DataTail semantics make the full source readable,
    which is what ``src`` does in the C port."""

    NBAND9 = 9

    def __init__(self, nch: int, sr: int, sens: float):
        self.nch = nch
        self.sr = sr
        self.sens = float(sens)
        v = 16
        need = int(np.float32(np.float32(sr) * np.float32(0.008)) + np.float32(0.5))
        if need >= 17:
            while v < need:
                v *= 2
        self.v5 = v
        self.N = v
        self.Nm = v >> 2
        self.Nh = v // 2 + 1
        self.n114 = ((v >> 2) >> 2) + 1
        self.win1 = np.array([_ti_kaiser(i, self.N, 12.0) for i in range(self.N)],
                             dtype=np.float32)
        self.win2 = np.array([_ti_kaiser(i, self.Nm, 5.0) for i in range(self.Nm)],
                             dtype=np.float32)
        self.fft_in = np.zeros(self.N, dtype=np.float32)
        self.sub_in = np.zeros(self.Nm, dtype=np.float32)
        self.spec_re = np.zeros(2 * self.N, dtype=np.float32)
        self.r9 = np.zeros(self.Nh, dtype=np.float32)
        self.pw = np.zeros(self.Nh, dtype=np.float32)
        self.band65 = np.zeros(65, dtype=np.float32)
        self.band_raw = np.zeros(24, dtype=np.float32)
        # scratch for the exact (numpy) 24-band accumulation; the TD_FAST kernel
        # accumulates each band directly and never touches it.
        self._bmat = np.zeros((24, int((_TI_BOUNDS[1:25] - _TI_BOUNDS[0:24]).max())),
                              dtype=np.float32)
        self.ring_r9 = np.zeros((self.NBAND9, self.Nh), dtype=np.float32)
        self.ring_pool = np.zeros((self.NBAND9, 65), dtype=np.float32)
        self.ring_r24 = np.zeros((self.NBAND9, 24), dtype=np.float32)
        # frame-indexed working buffers (purge is always 0 in the C port)
        self._cap = 4096
        self.w240 = np.zeros(self._cap, dtype=np.float32)
        self.w228 = np.zeros(self._cap, dtype=np.float32)
        self.mag = np.zeros(self._cap, dtype=np.float32)
        self.v00 = np.zeros(self._cap, dtype=np.float32)
        self.w240_n = 0
        self.w228_n = 0
        self.mag_n = 0
        self.v00_n = 0
        self.lp_state = 0.0
        self.last_idx = 0
        self.last398 = 0
        self.c390 = 0
        self.c394 = 0
        self.purge = 0
        # w22 = w9 + 4*w21 (see transients_info.c struct comment)
        s = float(self.sr) if self.sr else 44100.0
        nm = float(self.Nm) if self.Nm else 128.0
        wa = int(0.01 * s / nm + 0.5)
        w21 = int(0.1 * s / nm + 0.5)
        a2 = int(0.01 * s / nm + 0.5)
        b2 = int(0.025 * s / nm + 0.5)
        c2 = int(0.05 * s / nm + 0.5)
        self.w22 = wa + a2 + 6 + b2 + c2 + 4 * w21

    # -- growth ------------------------------------------------------------
    def _ensure(self, want: int) -> None:
        if want <= self._cap:
            return
        nc = self._cap
        while nc < want:
            nc *= 2
        for name in ("w240", "w228", "mag", "v00"):
            a = getattr(self, name)
            b = np.zeros(nc, dtype=np.float32)
            b[: a.shape[0]] = a
            setattr(self, name, b)
        self._cap = nc

    # -- ProcessStreaming @0x1620AC ---------------------------------------
    def process(self, count: int, mix: np.ndarray) -> None:
        stride = self.Nm
        nframes = count // stride
        nsrc = mix.shape[0]
        fftN = _fft_plan(self.N)
        fftM = _fft_plan(self.Nm)
        for _ in range(nframes):
            c = self.c394 + 1
            g9 = stride * c - 384
            g6 = stride * c - 192
            if g9 >= 0:
                self._big(fftN, g9, c, mix, nsrc)
            if g6 >= 0:
                self._sub(fftM, g6, c, mix, nsrc)
            self._synth(c)
            self.c390 = c - 4
            self.c394 = c

    def _gather(self, mix, nsrc, start, n):
        out = np.zeros(n, dtype=np.float32)
        if start < nsrc:
            end = min(start + n, nsrc)
            k = end - start
            if k > 0:
                out[:k] = mix[start:end]
        return out

    def _big(self, fftN, g9, c, mix, nsrc):
        v = self._gather(mix, nsrc, g9, self.N)
        # C: (float)((double)win * (double)v) — double intermediate, f32 store
        self.fft_in[:] = self.win1.astype(np.float64) * v.astype(np.float64)
        sa = fftN.fwd(self.fft_in)
        if _FAST_TI:
            # 2b: bit-exact numba post-FFT (pw/r9/bands/total in one pass).
            w = _td_fast_mod.ti_big_post(sa, _TI_WIN_SCALE2, self.pw, self.r9,
                                        self.band_raw, _TI_BOUNDS)
        else:
            re = sa[0:2 * self.Nh:2]
            im = sa[1:2 * self.Nh:2]
            np.multiply(re, re, out=self.pw)
            self.pw += im * im
            self.pw *= _TI_WIN_SCALE2
            # |FFT|^(1/4) = sqrt(sqrt(sqrt(power)))
            self.r9[:] = np.sqrt(np.sqrt(np.sqrt(self.pw)))
            # 24 log bands: C accumulates each band's power sequentially in f32
            # starting from 0.0f, then ^(1/8).  Pad-and-accumulate keeps that exact
            # association (a global cumulative sum would not: restarts at 0 round
            # differently from continuing an accumulator).
            bmat = self._bmat
            for b in range(24):
                lo, hi = int(_TI_BOUNDS[b]), int(_TI_BOUNDS[b + 1])
                bmat[b, :hi - lo] = self.pw[lo:hi]
            s = np.add.accumulate(bmat, axis=1, dtype=np.float32)[:, -1]
            self.band_raw[:] = np.sqrt(np.sqrt(np.sqrt(s)))
            # w240: full power ^(1/4) (sequential f32 accumulation as in C)
            accf = np.add.accumulate(self.pw, dtype=np.float32)[-1]
            w = np.float32(np.sqrt(np.sqrt(accf))) if accf > 0 else np.float32(0.0)
        wi = c - self.purge
        if wi >= 0:
            self._ensure(wi + 1)
            self.w240[wi] = w
            if wi + 1 > self.w240_n:
                self.w240_n = wi + 1
        slot = c % self.NBAND9
        np.copyto(self.ring_r9[slot], self.r9)
        np.copyto(self.ring_r24[slot], self.band_raw)

    def _sub(self, fftM, g6, c, mix, nsrc):
        v = self._gather(mix, nsrc, g6, self.Nm)
        # C: (float)((double)win * (double)v) — double intermediate, f32 store
        self.sub_in[:] = self.win2.astype(np.float64) * v.astype(np.float64)
        sa = fftM.fwd(self.sub_in)
        if _FAST_TI:
            _td_fast_mod.ti_sub_post(sa, self.band65, np.float32(1.16609546))
        else:
            re = sa[0:130:2]
            im = sa[1:130:2]
            pw = re * re
            pw += im * im
            self.band65[:] = np.where(pw > 0, np.float32(1.16609546) * np.sqrt(np.sqrt(np.sqrt(pw))),
                                      np.float32(0.0))
        slot = c % self.NBAND9
        np.copyto(self.ring_pool[slot], self.band65)

    def _synth(self, c):
        if _FAST_TI:
            # 2b: reductions + the w228/mag stores in one numba pass.  The
            # 8-slot association and ascending f32 accumulation order are the
            # numpy port's, so the result is bit-equal (certified in td_fast).
            _poolv, _s0, wi, mi = _td_fast_mod.ti_synth_post(
                self.ring_r24, self.ring_pool, self.ring_r9, self.w228, self.mag,
                c, self.purge, self.Nh, self.n114, self.NBAND9)
            if wi >= 0 and wi + 1 > self.w228_n:
                self.w228_n = wi + 1
            if mi >= 0 and mi + 1 > self.mag_n:
                self.mag_n = mi + 1
            return
        wb = c - 4
        ix = [(wb + k) if (wb + k) >= 0 else 0 for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        i_m4, i_m3, i_m2, i_m1, i_p1, i_p2, i_p3, i_p4 = [i % self.NBAND9 for i in ix]
        # shape over 24 bands (left-to-right association as in C)
        v = ((self.ring_r24[i_p1] + self.ring_r24[i_p2]) + self.ring_r24[i_p3]) + self.ring_r24[i_p4]
        v = (v - self.ring_r24[i_m1] - self.ring_r24[i_m2]) - self.ring_r24[i_m3] - self.ring_r24[i_m4]
        v = np.where(v <= 0.0, np.float32(-0.1) * v, v)
        shape = np.float32(np.add.accumulate(v, dtype=np.float32)[-1] / np.float32(24.0))
        # pool over 65 subbands: two-slot difference, rectify, mean
        m = self.ring_pool[i_p1] - self.ring_pool[i_m1]
        m = np.where(m <= 0.0, np.float32(-0.1) * m, m)
        pool = np.float32(np.add.accumulate(m, dtype=np.float32)[-1] / np.float32(self.n114))
        wi = (c - 4) - self.purge
        if wi >= 0:
            self._ensure(wi + 1)
            self.w228[wi] = pool
            if wi + 1 > self.w228_n:
                self.w228_n = wi + 1
        # vsum over 257 bins (big-window ring)
        v2 = ((self.ring_r9[i_p1] + self.ring_r9[i_p2]) + self.ring_r9[i_p3]) + self.ring_r9[i_p4]
        v2 = (v2 - self.ring_r9[i_m1] - self.ring_r9[i_m2]) - self.ring_r9[i_m3] - self.ring_r9[i_m4]
        v2 = np.where(v2 <= 0.0, np.float32(-0.1) * v2, v2)
        vsum = np.float32(np.add.accumulate(v2, dtype=np.float32)[-1] / np.float32(self.Nh))
        s0 = np.float32(2.0) * (vsum + shape + np.float32(0.5) * pool)
        mi = (c - 4) - self.purge
        if mi >= 0:
            self._ensure(mi + 1)
            self.mag[mi] = s0
            if mi + 1 > self.mag_n:
                self.mag_n = mi + 1

    # -- AnalyzeStreaming @0x162FB8 ---------------------------------------
    def analyze(self) -> None:
        n = self.mag_n
        if not n:
            return
        alpha = np.float32(self.iir_alpha())
        k_lo = self.last_idx
        k_hi = n - 1
        if k_lo > k_hi:
            k_lo = k_hi
        lp = np.float32(self.lp_state)
        mag = self.mag
        for k in range(k_lo, k_hi + 1):
            x = mag[k]
            lp = lp + alpha * (x - lp)
            y = x - np.float32(0.8) * lp
            mag[k] = y
            lp = y
        self.lp_state = float(lp)
        self.last_idx = k_hi + 1

        thr = np.float32(1.0 / self.sens) if self.sens > 0.0 else np.float32(1.0)
        last398 = self.last398
        purge = self.purge
        nrec = (n - 1) + purge                     # = c390
        # threshold cleanup over [last398, c390)
        a0 = last398 if last398 > purge else purge
        a1 = n - 1 + purge
        if a1 > purge + n:
            a1 = purge + n
        if a1 > a0:
            lo = a0 - purge
            hi = a1 - purge
            np.copyto(mag[lo:hi], np.where(mag[lo:hi] < thr, np.float32(0.0), mag[lo:hi]))

        sr = np.float32(self.sr)
        W = self.w240
        P = self.w228
        wa = int((sr * np.float32(0.01)) / np.float32(self.Nm) + np.float32(0.5))

        # ---- T1: weighted max-filter kill (0x1630D4-0x1631B4) ----
        k10 = nrec - wa
        if k10 > 0 and last398 < nrec:
            k_start = last398 - wa
            if k_start < 0:
                k_start = 0
            w15 = nrec - 1
            for k in range(k_start, k10 + 1):
                kr = k - purge
                if not (0 <= kr < n):
                    continue
                vk = mag[kr]
                if not (vk >= thr):
                    continue
                w3 = k - wa
                if w3 < 0:
                    w3 = 0
                w2 = k + wa
                if w15 < w2:
                    w2 = w15
                if w3 > w2:
                    continue
                keyk = vk * (W[kr] if kr < self.w240_n else np.float32(0.0))
                killed = False
                for j in range(w3, w2):
                    jr = j - purge
                    if not (0 <= jr < self.w240_n):
                        continue
                    if mag[jr] * W[jr] > keyk:
                        killed = True
                        break
                if killed:
                    mag[kr] = np.float32(0.0)

        # ---- A2/B2/C2 progressive max-filters (0x1632BC-0x16353C) ----
        gstep = (0.01, 0.025, 0.05)
        gain = (np.float32(1.0), np.float32(2.7), np.float32(6.0))
        w9 = wa
        for p in range(3):
            w10 = int((sr * np.float32(gstep[p])) / np.float32(self.Nm) + np.float32(0.5))
            w9 += w10
            if p == 0:
                w9 += 6
            k_hi2 = nrec - w9
            if k_hi2 < 1:
                continue
            k_start = last398 - w9
            if k_start < 0:
                k_start = 0
            w15 = nrec - 1
            for k in range(k_start, k_hi2):
                kr = k - purge
                if not (0 <= kr < n):
                    continue
                vk = mag[kr]
                if not (vk >= thr):
                    continue
                w4 = k - w10
                if w4 < 0:
                    w4 = 0
                w3 = k + w10
                if w3 > w15:
                    w3 = w15
                if w4 > w3:
                    continue
                keyk = vk * gain[p] * (W[kr] if kr < self.w240_n else np.float32(0.0))
                killed = False
                for j in range(w4, w3):
                    jr = j - purge
                    if not (0 <= jr < self.w240_n):
                        continue
                    if mag[jr] * W[jr] > keyk:
                        killed = True
                        break
                if killed:
                    mag[kr] = np.float32(0.0)

        # ---- T2: 7-point argmax on w228, swap mag (0x1631B8-0x1632B8) ----
        lo = last398 - wa - 3
        if lo < 3:
            lo = 3
        hi = n - 3
        for k in range(lo, max(lo, hi)):
            vk = mag[k - purge]
            if vk > thr:
                best = np.float32(0.0)
                bj = k
                for q in range(-3, 4):
                    j = k + q
                    jr = j - purge
                    if j < 0 or not (0 <= jr < self.w228_n):
                        continue
                    key = P[jr]
                    if key > best:
                        best = key
                        bj = j
                if bj != k:
                    a = k - purge
                    b = bj - purge
                    t1 = mag[b]
                    mag[b] = vk
                    mag[a] = t1

        # ---- final pass: v00 (0x163658-0x1636A4) ----
        w22 = self.w22
        i0 = last398 - w22
        if i0 < purge:
            i0 = purge
        i1 = n - 1 + purge - w22
        i1c = self._cap + purge
        if i1 > i1c:
            i1 = i1c
        if i1 > i0:
            lo = i0 - purge
            hi = i1 - purge
            hi = min(hi, n)
            if hi > lo:
                seg = mag[lo:hi]
                np.copyto(self.v00[lo:hi], np.where(seg > thr, seg, np.float32(0.0)))
        ln = i1 - purge
        if ln < 0:
            ln = 0
        if ln > self.v00_n:
            self.v00_n = ln

        self.last_idx = n
        self.last398 = n - 1 + purge

    def iir_alpha(self) -> float:
        a = 0.1
        b = (float(self.sr) / float(self.Nm)) if self.Nm else 0.0
        if a == 0.0 or b == 0.0:
            return 1.0
        return 1.0 - math.exp(-1.0 / (a * b))

    # -- GetTransientPos @0x163E4C ----------------------------------------
    def get_transient_pos(self, frm: int, length: int) -> int:
        if self.mag_n == 0:
            return -1
        stride = self.Nm
        origin = -self.Nm
        purge = self.purge
        k0 = int((frm - origin + (stride >> 1)) / stride)
        k1 = int((frm + length - origin + (stride >> 1)) / stride)
        if k0 < purge:
            k0 = purge
        k_hi_max = purge + self.mag_n - 1
        if k1 > k_hi_max:
            k1 = k_hi_max
        if k0 > k1:
            return -1
        lo = k0 - purge
        hi = k1 - purge + 1
        seg = self.v00[lo:hi] if (self.v00_n and hi <= self.v00.shape[0]) else None
        # the C reads v00[idx] when idx < v00_n, else mag[idx]
        vals = np.empty(hi - lo, dtype=np.float32)
        for q, idx in enumerate(range(lo, hi)):
            if self.v00_n and idx < self.v00_n:
                vals[q] = self.v00[idx]
            else:
                vals[q] = self.mag[idx]
        best = vals.max()
        if best <= 0.0:
            return -1
        bk = k0 + int(np.argmax(vals))
        T = origin + stride * bk
        if T >= frm + length:
            return -1
        if T < frm:
            return -1
        return int(T)

    def stride(self) -> int:
        return self.Nm


# ---------------------------------------------------------------------------
# Pitch::SetOptions / AnalyzePitchPrecise (src/ops/td_pitch.c)
# ---------------------------------------------------------------------------

def _octave_correct_peak(acf: np.ndarray, acf_len: int, search_start: int,
                         search_end: int, octave_enable: bool):
    """Port of td_octave_correct_peak (src/ops/td_octave.c).

    Returns (p, subharmonic_suspect, subharmonic_ratio)."""
    f32 = np.float32
    if search_start < 1:
        search_start = 1
    if search_end > acf_len - 2:
        search_end = acf_len - 2
    if search_start >= search_end:
        return None
    p = search_start
    max_val = acf[search_start]
    for i in range(search_start + 1, search_end):
        if acf[i] > max_val:
            max_val = acf[i]
            p = i
    sub_sus = 0
    sub_ratio = f32(0.0)
    if octave_enable:
        cand_p2 = (p >> 1) - 1
        if cand_p2 >= search_start:
            idx_p2 = cand_p2
            val_p2 = acf[idx_p2]
            if cand_p2 + 1 < acf_len and acf[cand_p2 + 1] > val_p2:
                idx_p2 = p >> 1
                val_p2 = acf[idx_p2]
            if cand_p2 + 2 < acf_len and acf[cand_p2 + 2] > val_p2:
                idx_p2 = cand_p2 + 2
                val_p2 = acf[idx_p2]
            p15 = (3 * p) >> 1
            idx_p15 = p15 - 2
            val_p15 = acf[idx_p15] if 0 <= idx_p15 < acf_len else f32(-1000.0)
            for k in range(p15 - 1, p15 + 3):
                if 0 <= k < acf_len and acf[k] > val_p15:
                    idx_p15 = k
                    val_p15 = acf[k]
            r_p2 = np.sqrt(np.sqrt(np.maximum(val_p2, f32(0.0))))
            r_p15 = np.sqrt(np.sqrt(np.maximum(val_p15, f32(0.0))))
            sum_r = r_p2 + r_p15
            sum_r2 = sum_r * sum_r
            crit = f32(0.1) * (sum_r2 * sum_r2)
            if crit > acf[p]:
                p = idx_p2
            if acf[p] > f32(1e-12):
                ratio = crit / acf[p]
                if ratio > f32(0.1) and ratio < f32(1.0):
                    sub_sus = 1
                    sub_ratio = ratio
            cand_p2 = (p >> 1) - 1
        if cand_p2 >= search_start:
            p_div3 = p // 3
            idx_p3 = p_div3 - 1
            val_p3 = acf[idx_p3] if 0 <= idx_p3 < acf_len else f32(-1000.0)
            if p_div3 < acf_len and acf[p_div3] > val_p3:
                idx_p3 = p_div3
                val_p3 = acf[idx_p3]
            if p_div3 + 1 < acf_len and acf[p_div3 + 1] > val_p3:
                idx_p3 = p_div3 + 1
                val_p3 = acf[idx_p3]
            p_2div3 = (2 * p) // 3
            idx_2p3 = p_2div3 - 1
            val_2p3 = acf[idx_2p3] if 0 <= idx_2p3 < acf_len else f32(-1000.0)
            if p_2div3 < acf_len and acf[p_2div3] > val_2p3:
                idx_2p3 = p_2div3
                val_2p3 = acf[idx_2p3]
            if p_2div3 + 1 < acf_len and acf[p_2div3 + 1] > val_2p3:
                idx_2p3 = p_2div3 + 1
                val_2p3 = acf[idx_2p3]
            r_p3 = np.sqrt(np.sqrt(np.maximum(val_p3, f32(0.0))))
            r_2p3 = np.sqrt(np.sqrt(np.maximum(val_2p3, f32(0.0))))
            sum_r3 = r_p3 + r_2p3
            sum_r3_2 = sum_r3 * sum_r3
            crit3 = f32(0.1) * (sum_r3_2 * sum_r3_2)
            if crit3 > acf[p]:
                p = idx_p3
    return (p, sub_sus, sub_ratio)


# ---------------------------------------------------------------------------
# TD state (rx_td_state subset that the port needs)
# ---------------------------------------------------------------------------

RX_TD_NTAB = 32
RX_TD_NBLK = 5


class TDState:
    def __init__(self, sr: int, quality: int = 37, solo: int = 0, nch: int = 2):
        self.sr = int(sr)
        self.nch = int(nch)
        self.solo = int(solo)
        self.quality = int(quality)
        self._fast_ds = 0
        # asm 0x15B58C..0x15B5C0
        s8 = np.float32(quality)
        s1 = np.float32(1.5) * s8
        s2 = np.float32(sr) * np.float32(0.001)
        v = s2 * s1
        self.hop = int(v + np.float32(0.5))
        self.f28 = _c_u32(int(np.float32(np.float32(sr) * np.float32(0.1)) + np.float32(0.5))) & ~7
        self.win_max = 4
        self.state_330 = _c_u32(self.solo ^ 1)
        self.last_transient = 0xFFFFFFFF
        self.pitch_lo = (self.hop // 40) if self.solo else (self.hop >> 2)
        self.pitch_hi = self.hop >> 1
        self.stretch = 1.0
        self.total_ratio = 1.0
        self.pitch_ratio = 1.0
        self.ring_tot = 0
        self.ring_len = 0
        self.ring_start = 0
        self.fed_visible = 0
        self.cursor = 0
        self.cursor_prev = 0
        self.acc = 0.0
        self.skip_next_318 = 0
        self.dec_flag = 0
        self.advanced = 0
        self.forced_T = 0xFFFFFFFF
        self.write_ptr_acc = 0.0
        self.wrap_cnt = 0
        self.n_granule = 0
        self.n_transient = 0
        self.dedup_hist: list[int] = []
        self.trans: TIState | None = None
        self._build_env_tables()
        self._pitch_set_options()
        self.ring_tot = self.f28 * 16
        self.out_len = self.ring_tot * 8 + 65536
        if self.out_len < self.hop * 4 + 8192:
            self.out_len = self.hop * 4 + 8192
        self.dedup_cap = max(self.ring_tot // 32 + 64, 64)

    # -- rx_td_build_env_tables -------------------------------------------
    def _build_env_tables(self):
        C = RX_TD_NTAB
        f28 = self.f28
        s8 = np.log(np.float32(f28 // 10))
        s9 = np.log(np.float32(f28))
        self.env_span = np.zeros((RX_TD_NBLK, C), dtype=np.int64)
        self.env_tab = [[None] * C for _ in range(RX_TD_NBLK)]
        self.env_gain_tab = np.zeros(RX_TD_NBLK, dtype=np.float32)
        for a8 in range(RX_TD_NBLK):
            g = np.float32(1.0) if a8 >= self.win_max else np.float32(0.5625 + 0.125 * a8)
            for j in range(C):
                v = (s8 * np.float32(C - 1 - j) + s9 * np.float32(j)) / np.float32(C - 1)
                L = int(np.exp(v) + np.float32(0.5))
                if L < 2:
                    L = 2
                self.env_span[a8, j] = L
                self.env_tab[a8][j] = _hann_pow(L, float(g))
            self.env_gain_tab[a8] = g

    # -- rx_td_pitch_set_options ------------------------------------------
    def _pitch_set_options(self):
        sr = self.sr
        a4 = self.pitch_hi
        N = 128
        self.pitch_L1 = int(2.5 * float(a4) + 0.5)
        while N < 5 * a4:
            N <<= 1
        self.pitch_N = N
        self._fast_ds = 0
        m = int(4000.0 / float(sr) * float(N) + 0.5)
        cap = N // 2 - 1
        self.pitch_maxbin = m if m < cap else cap
        if self.pitch_maxbin < 1:
            self.pitch_maxbin = 1
        v58 = int(10000.0 / float(sr) * float(N) + 0.5) + 1
        if v58 > N - 1:
            v58 = N - 1
        vlen = v58 - self.pitch_maxbin
        if vlen < 1:
            vlen = 1
        self.pitch_taper_len = vlen
        i = np.arange(vlen, dtype=np.float64)
        self.pitch_win_taper = np.array(0.5 + 0.5 * np.cos(PI * i / float(vlen)),
                                        dtype=np.float32)
        L1 = self.pitch_L1
        ia = np.arange(L1, dtype=np.float64)
        self.pitch_win_an = np.array(0.5 - 0.5 * np.cos(2.0 * PI * ia / float(L1)),
                                     dtype=np.float32)
        R = L1 >> 3
        self.pitch_win_acf = np.zeros(N, dtype=np.float32)
        ir = np.arange(R, dtype=np.float64)
        self.pitch_win_acf[:R] = 0.5 + 0.5 * np.cos(PI * ir / float(R))
        for k in range(1, R):
            self.pitch_win_acf[N - k] = self.pitch_win_acf[k]
        il = np.arange(N, dtype=np.float64)
        lin = 1.0 - il / float(N // 2 + 1)
        self.pitch_win_lin = np.array(np.maximum(lin, 0.0), dtype=np.float32)
        self.pitch_octave_en = 1
        self.pitch_plan = _fft_plan(N)
        if _td_fast_mod is not None and self.solo == 0:
            # 1) vDSP FFT plan for the pitch front-end (approximate tier).
            if _FAST_VDSP:
                self.pitch_plan = _td_fast_mod.make_plan(N)
            # 2) decimated front-end (lossy, opt-in via TD_FAST_DS).
            d = _FAST_DS
            if d > 1 and N // d >= 256:
                try:
                    self._coarse = _td_fast_mod.DecimatedPitch(self, d)
                    self._fast_ds = d
                except Exception:
                    self._fast_ds = 0

    def set_ratio(self, semis: float, tempo: float) -> None:
        self.stretch = tempo / 100.0
        self.pitch_ratio = math.pow(2.0, semis / 12.0)
        self.total_ratio = self.pitch_ratio * self.stretch

    # -- rx_td_analyze_pitch ----------------------------------------------
    def analyze_pitch(self, ring, in_pos: int):
        """Returns (lag, win, rms) — rx_td_pitch_analyze."""
        st = self
        ds = getattr(st, "_fast_ds", 0)
        if ds:
            from .td_fast import analyze_pitch_ds as _analyze_pitch_ds
            return _analyze_pitch_ds(st, ring, in_pos, st._coarse)
        N = st.pitch_N
        L1 = st.pitch_L1
        nb = (N >> 1) | 1
        x = np.zeros(N, dtype=np.float32)
        half = L1 >> 1
        mod = st.ring_tot if st.ring_tot else st.ring_len
        if _FAST_PITCH:
            # 2c: gather from the numba front-end.  The kernel reads a
            # contiguous [nch, ring_tot] matrix, so it is only usable when the
            # 2-D ring is present (TD_FAST).  The fed_visible mask is checked
            # per call: in_pos advances differently per granule (transient
            # jumps), so it cannot be hoisted.
            r2 = _ring2d_view(st)
            if r2 is not None and in_pos - half >= 0 \
                    and in_pos - half + L1 <= st.fed_visible \
                    and in_pos - half + L1 <= mod:
                off = _PITCH_ARANGE[:L1] + (in_pos - half)
                _td_fast_mod.pitch_gather(r2, x, off, L1, st.nch, mod)
                energy_raw = float(np.sum(x[:L1].astype(np.float64) ** 2))
            else:
                idx = (in_pos - half + _PITCH_ARANGE[:L1]) % mod
                vis = idx < st.fed_visible
                xi = np.where(vis, ring[0][idx], np.float32(0.0))
                if st.nch > 1:
                    xi = xi + np.where(vis, ring[1][idx], np.float32(0.0))
                x[:L1] = xi
                energy_raw = float(np.sum(x[:L1].astype(np.float64) ** 2))
        else:
            idx = (in_pos - half + _PITCH_ARANGE[:L1]) % mod
            vis = idx < st.fed_visible
            xi = np.where(vis, ring[0][idx], np.float32(0.0))
            if st.nch > 1:
                xi = xi + np.where(vis, ring[1][idx], np.float32(0.0))
            x[:L1] = xi
            energy_raw = float(np.sum(x[:L1].astype(np.float64) ** 2))
        rms = np.float32(math.sqrt(energy_raw / float(L1)))
        # analysis window
        x[:L1] = x[:L1] * st.pitch_win_an
        sa = st.pitch_plan.fwd(x)
        M = N >> 1
        sa[1] = 0.0
        re = sa[0:2 * M:2]
        im = sa[1:2 * M:2]
        np.multiply(re, re, out=re)
        re += im * im
        sa[1:2 * M:2] = 0.0
        sa[N + 1] = 0.0
        x = st.pitch_plan.inv(sa)
        x *= st.pitch_win_acf
        sb = st.pitch_plan.fwd(x)
        # whitening
        v387 = int(0.5 + 20.0 / float(st.sr) * float(N))
        v386 = int(0.5 + 150.0 / float(st.sr) * float(N)) + 1
        e02 = energy_raw * 0.2
        noise = (e02 if e02 > 1e-6 else 1e-6) * (float(st.sr) / 44100.0)
        noisef = np.float32(noise)
        kw = st.pitch_maxbin + st.pitch_taper_len
        if kw > M:
            kw = M
        out = np.zeros(M, dtype=np.float32)
        if kw > v387:
            ks = np.arange(max(v387, 0), kw)
            mag = np.abs(sb[2 * ks])
            # NOTE: np.power (not a numba `mag ** f32`) — the whitening power is
            # one of the places where the numpy float32 ufunc is bit-identical
            # to libm powf and a numba kernel is not (see td_fast._build_pitch).
            den = noisef + np.power(mag, np.float32(0.95))
            o = sa[2 * ks] / den
            # taper / ramp products are computed in double then cast (C source)
            tap = ks >= st.pitch_maxbin
            if np.any(tap):
                ti = np.where(tap, ks - st.pitch_maxbin, 0)
                ti = np.minimum(ti, st.pitch_taper_len - 1)
                o = np.where(tap, (o.astype(np.float64) * st.pitch_win_taper[ti].astype(np.float64)).astype(np.float32), o)
            ramp = ks < v386
            if np.any(ramp):
                t = (ks[ramp].astype(np.float64) - float(v387)) / float(v386 - v387)
                fac = 0.5 - 0.5 * np.cos(PI * t)
                o = o.copy()
                o[ramp] = (o[ramp].astype(np.float64) * fac).astype(np.float32)
            out[ks] = o
        x2 = np.zeros(N + 2, dtype=np.float32)
        # cart layout as written by the C whitening loop: even slot = bin value,
        # odd slot = 0 (bin k at index 2k), Nyquist at index N, N+1 zeroed.
        x2[0:2 * M:2] = out
        x2[1:2 * M:2] = 0.0
        x2[N] = 0.0
        x2[N + 1] = 0.0
        acf = st.pitch_plan.inv(x2)
        if _FAST_PITCH:
            # 2c: acf *= win_lin + the -1000 leading-run suppression in C order.
            acf = np.ascontiguousarray(acf)
            _td_fast_mod.pitch_acf_win(acf, nb, st.pitch_win_lin)
        else:
            acf[:nb] = acf[:nb] * st.pitch_win_lin[:nb]
            # leading decreasing suppression
            i = 0
            while i + 1 < nb and acf[i] > acf[i + 1]:
                acf[i] = np.float32(-1000.0)
                i += 1
        v160 = int(st.pitch_lo)
        v138 = int(st.pitch_hi)
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
        lag = float(p)
        y0 = float(acf[p - 1]); y1 = float(acf[p]); y2 = float(acf[p + 1])
        den = 2.0 * y1 - y2 - y0
        if den != 0.0:
            d = (y2 - y0) / (2.0 * den)
            if d < -1.0:
                d = -1.0
            if d > 1.0:
                d = 1.0
            lag = float(p) + d
        # clarity -> win
        y1 = float(acf[p])
        v14 = y1
        comp = 0.0
        lb = int(0.6 * float(p)); rb = int(1.8 * float(p) + 0.5)
        if _FAST_PITCH:
            comp = _td_fast_mod.pitch_clarity(acf, nb, p, lb, rb)
        else:
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
        if _PDBG:
            print(f"P in={in_pos} lo={v160} hi={v138} p={p} win={w} lag={lag:.9f}",
                  file=_sys.stderr)
        return lag, w, float(rms)

    # -- rx_td_pick_env ----------------------------------------------------
    def pick_env(self, a8_in: int, a9: int, a10: int, boolean: int):
        a8 = a8_in if a8_in < RX_TD_NBLK else RX_TD_NBLK - 1
        v12 = (2 * a10) // 3 if boolean else a9
        j = RX_TD_NTAB - 1
        for k in range(RX_TD_NTAB):
            if v12 < self.env_span[a8, k]:
                j = 0 if k == 0 else k - 1
                break
        span = int(self.env_span[a8, j])
        env = self.env_tab[a8][j]
        self._last_pick_j = int(j)
        half = span >> 1
        # C: (unsigned)((int)(a9>>1) - (int)half).  Negative w10 is legal
        # (ring reads wrap backwards via _wrap_ring).  NOTE: a9 < span occurs
        # for quality=2 pitch detection on near-silence; that configuration
        # has a known divergence vs the C engine (see docs) — the acceptance
        # corpus uses quality=37, where this never triggers.
        w10 = 0 if boolean else ((a9 >> 1) - half)
        return env, half, int(w10)

    # -- rx_td_transient_pos ----------------------------------------------
    def transient_pos(self, frm: int, length: int) -> int:
        if self.trans is None or length == 0:
            return 0xFFFFFFFF
        T = self.trans.get_transient_pos(frm, length)
        return 0xFFFFFFFF if T < 0 else _c_u32(T)


def _hann_pow(length: int, gain: float) -> np.ndarray:
    """hann_pow (td_core.c): record = 2*len, both halves = Hann^gain."""
    i = np.arange(length, dtype=np.float64)
    u = (i + 0.5) / float(length)
    h = 0.5 - 0.5 * np.cos(2.0 * PI * u)
    v = np.power(h, gain).astype(np.float32)
    return np.concatenate([v, v])


# ---------------------------------------------------------------------------
# DoOla (asm 0x15D070) — additive COLA with window lookup
# ---------------------------------------------------------------------------

def _do_ola(st: TDState, ring, out_ring, a4: int, wr: int, a8: int, a9: int,
            a10: int, boolean: int, gain: float) -> None:
    """Overlap-add dispatch: TD_FAST NEON/batched variant when gated in.

    The fast kernel works on the 2-D ring (``st._ring2d``); ``ring`` is the
    channel-view list the exact path needs.
    """
    if _FAST_OLA and st._ring2d is not None:
        _td_fast_mod.do_ola_fast(st, st._ring2d, out_ring, a4, wr, a8, a9, a10,
                                 boolean, gain)
        return
    _do_ola_exact(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain)


def _do_ola_exact(st: TDState, ring, out_ring, a4: int, wr: int, a8: int, a9: int,
                  a10: int, boolean: int, gain: float) -> None:
    nch = st.nch
    if nch <= 0 or a9 == 0:
        return
    rcap = st.ring_tot
    rstart = st.ring_start
    outlen = st.out_len
    gainf = np.float32(gain)
    if (wr | a4) == 0:
        ri = _wrap_ring(np.arange(a9, dtype=np.int64), rstart, rcap)
        for c in range(nch):
            sv = np.where(ri < st.fed_visible, ring[c][ri], np.float32(0.0))
            out_ring[c, 0:a9] = gainf * sv
    env, half, w10 = st.pick_env(a8, a9, a10, boolean)
    if half == 0 or env is None:
        return
    rbase = a4 + w10
    wbase = wr + w10
    # segment 1: crossfade accumulate
    i1 = np.arange(half, dtype=np.int64)
    ri = _wrap_ring(rbase + i1, rstart, rcap)
    wi = _wrap_out(wbase + i1, outlen)
    s0 = env[0:half]
    s1 = env[half:2 * half]
    for c in range(nch):
        sv = np.where(ri < st.fed_visible, ring[c][ri], np.float32(0.0))
        d = out_ring[c]
        d[wi] = d[wi] * s1 + gainf * sv * s0
    # segment 2: plain copy tail
    w11 = a10 if boolean else (a10 if a10 > a9 else a9)
    if w11 > half:
        i2 = np.arange(half, w11, dtype=np.int64)
        ri = _wrap_ring(rbase + i2, rstart, rcap)
        wi = _wrap_out(wbase + i2, outlen)
        for c in range(nch):
            sv = np.where(ri < st.fed_visible, ring[c][ri], np.float32(0.0))
            out_ring[c, wi] = gainf * sv


# ---------------------------------------------------------------------------
# rx_td_render (src/engine/td_core.c main loop)
# ---------------------------------------------------------------------------

def td_render(st: TDState, x: np.ndarray):
    """x: interleaved float32 [nframes, nch].  Returns (out [n, nch], stats)."""
    hop = st.hop
    win_max = st.win_max
    nch = st.nch
    nframes = int(x.shape[0])
    ring_tot = st.ring_tot
    out_len = st.out_len
    if _FAST_OLA:
        # TD_FAST: one 2-D ring, exposed to the rest of the loop as per-channel
        # ROW VIEWS so the exact `_do_ola_exact` and the pitch/feed code keep
        # working unchanged (passing the 2-D array itself as `ring` would make
        # `ring[c]` the c-th row for every c and silently drop channels 1+).
        ring2d = np.zeros((nch, ring_tot), dtype=np.float32)
        ring = [ring2d[c] for c in range(nch)]
        st._ring2d = ring2d
    else:
        ring = [np.zeros(ring_tot, dtype=np.float32) for _ in range(nch)]
        st._ring2d = None
    st.fed_visible = 0
    st.ring_start = 0
    st.ring_len = ring_tot
    tbl = InterpTable(1024, 6, 12.0)
    out_ring = np.zeros((nch, out_len + 16), dtype=np.float32)

    # mixed mono view used by TransientsInfo (ch0+ch1, f32 add)
    if nch >= 2:
        mix = x[:, 0] + x[:, 1]
    else:
        mix = np.ascontiguousarray(x[:, 0])

    if st.trans is None:
        st.trans = TIState(nch, st.sr, 1.0)

    out_chunks: list[np.ndarray] = []
    out_n = 0
    wp = 0.9499998092651367
    cursor = 0
    in_pos = 0
    fed_pos = 0
    acc = 0.0
    st.write_ptr_acc = wp
    st.cursor = 0
    st.cursor_prev = 0
    st.in_pos = 0
    st.fed_pos = 0
    st.acc = 0.0
    st.advanced = 0
    st.skip_next_318 = 0
    st.last_transient = 0xFFFFFFFF
    st.n_granule = 0
    st.n_transient = 0

    gate = hop + ((hop >> 1) << (3 if st.state_330 else 0)) + 100
    nstop = _c_u32(nframes)
    total_ext = _c_u32(nframes + 4 * gate + 65536)
    feed_n = 0
    cursor_append = 0
    fed_total = 0
    loop_bound = _c_u32(nframes + gate + 8 * hop)

    while cursor_append < total_ext and cursor_append < loop_bound:
        feed = 1024 if feed_n < 14 else 0x2000
        if cursor_append + feed > total_ext:
            feed = total_ext - cursor_append
        if _DBG:
            print(f"F n={feed_n} feed={feed} ca={cursor_append} gate={gate}", file=_sys.stderr)
        g = cursor_append + np.arange(feed, dtype=np.int64)
        gi = g % ring_tot
        idx_src = np.minimum(g, nframes - 1) if nframes > 0 else np.zeros(feed, np.int64)
        ok = g < nframes
        for c in range(nch):
            vals = np.where(ok, x[idx_src, c], np.float32(0.0))
            ring[c][gi] = vals
        cursor_append += feed
        fed_total = cursor_append
        st.fed_visible = fed_total
        # TransientsInfo: 1024-sample chunks
        off = 0
        while off < feed:
            cl = feed - off
            if cl > 1024:
                cl = 1024
            st.trans.process(cl, mix)
            st.trans.analyze()
            off += cl
        feed_n += 1

        gate_target = (fed_total - gate) if fed_total > gate else 0
        if gate_target > nstop:
            gate_target = nstop

        guard = 0
        while fed_pos < gate_target and guard < 400000:
            guard += 1
            rms = 0.0
            if st.state_330 == 1:
                lag, win, rms = st.analyze_pitch(ring, in_pos + (3 * hop) // 8)
                period = int(lag + 0.5)
                if period < 1:
                    period = 1
            else:
                period = hop >> 2
                win = period
            if win > win_max - 1:
                win = win_max - 1
            istep = period
            if istep < 1 or istep > 65536:
                istep = period
            st.dec_flag = 0
            st.forced_T = 0xFFFFFFFF
            skip = st.skip_next_318
            st.skip_next_318 = 0
            s11 = np.float32(np.float32(istep) / np.float32(st.sr))
            d13 = float(s11)
            T_prescan = 0xFFFFFFFF
            tr = st.trans
            if tr is not None:
                wf0 = np.float32(2 * period)
                lo0 = _c_u32(int(wf0 * np.float32(0.1)))
                tl0 = _c_u32(int(wf0 * np.float32(0.9)))
                T_prescan = st.transient_pos(in_pos + lo0, tl0)
            if not skip:
                h2 = hop >> 1
                if tr is not None:
                    T = T_prescan
                    if T != 0xFFFFFFFF and T != st.last_transient:
                        st.forced_T = T
                        stp = tr.stride()
                        fr = T // (stp if stp else 1)
                        dup = False
                        if st.dedup_hist and len(st.dedup_hist):
                            if fr in st.dedup_hist:
                                dup = True
                            else:
                                if len(st.dedup_hist) >= st.dedup_cap:
                                    oi = int(np.argmin(st.dedup_hist))
                                    st.dedup_hist[oi] = fr
                                else:
                                    st.dedup_hist.append(fr)
                        else:
                            st.dedup_hist.append(fr)
                        if not dup:
                            d1 = acc * st.sr + (st.total_ratio - 1.0) * (float(T) - float(in_pos))
                            v37 = int(d1 + (0.5 if d1 >= 0.0 else -0.5))
                            av = float(v37) if v37 >= 0 else float(-v37)
                            if float(np.float32(st.sr) * np.float32(0.005)) <= av:
                                st.dec_flag = 1
                                st.n_transient += 1
                            else:
                                st.last_transient = T
                if tr is not None and st.dec_flag:
                    T = st.forced_T
                    if T == 0xFFFFFFFF:
                        wf2 = np.float32(2 * period)
                        T = st.transient_pos(in_pos + _c_u32(int(wf2 * np.float32(0.1))),
                                             _c_u32(int(wf2 * np.float32(0.9))))
                    if T != 0xFFFFFFFF:
                        d1 = acc * st.sr + (st.total_ratio - 1.0) * (float(T) - float(in_pos))
                        v22 = int(d1 + (0.5 if d1 >= 0.0 else -0.5))
                        rmin = st.total_ratio if st.total_ratio < 1.0 else 1.0
                        extra = int(float(np.float32(h2) * np.float32(rmin)))
                        span = 2 * period + extra
                        a9v = span
                        delta = span - h2
                        wr_off = 0
                        prev_abs = cursor
                        v22e = v22
                        if v22 < 1:
                            cand = cursor + v22
                            x12 = st.cursor_prev
                            w11b = (x12 - cursor) if cand < x12 else v22
                            if not (w11b + delta > 0):
                                w11b = -delta
                            wr_off = w11b
                            v22e = w11b
                            prev_abs = cursor + w11b
                        wpv = cursor + wr_off
                        m = out_len
                        if m > 0:
                            wpv %= m
                        a4t = in_pos - (v22 if v22 >= 0 else -v22)
                        _do_ola(st, ring, out_ring, a4t if a4t > 0 else 0,
                                int(wpv), 0, a9v, hop, 1, 1.0)
                        acc = acc + (st.total_ratio - 1.0) * (float(delta) / float(st.sr)) \
                            - float(v22e) / float(st.sr)
                        st.cursor_prev = prev_abs
                        cursor = cursor + delta + v22e
                        st.last_transient = T
                        in_pos = (in_pos + delta) % ring_tot
                        fed_pos = fed_pos + delta
                        st.advanced = 1
                if not st.advanced:
                    if st.dec_flag:
                        a9t = 2 * period + int(float(h2) *
                                               (st.total_ratio if st.total_ratio < 1.0 else 1.0))
                        _do_ola(st, ring, out_ring, in_pos, cursor % out_len,
                                0, a9t, hop, 1, 1.0)
                    else:
                        _do_ola(st, ring, out_ring, in_pos, cursor % out_len,
                                win_max, 2 * period, hop, 0, 1.0)
                    acc = acc + (st.total_ratio - 1.0) * d13
                    st.cursor_prev = cursor
                    cursor += istep
                if (not st.advanced) and st.total_ratio >= 1.0:
                    w9 = 0
                    if np.float32(2.0) * np.float32(acc) > s11:
                        while True:
                            if not (2.0 * st.total_ratio > float(w9)):
                                break
                            _do_ola(st, ring, out_ring, in_pos, cursor % out_len,
                                    win_max, 2 * period, hop, 0, 1.0)
                            acc -= d13
                            w9 += 1
                            st.cursor_prev = cursor
                            cursor += istep
                            if w9 > 64:
                                break
                            if not (np.float32(2.0) * np.float32(acc) > s11):
                                break
            else:
                acc = acc + st.total_ratio * d13
            if st.total_ratio < 1.0:
                st.skip_next_318 = 1 if (np.float32(2.0) * np.float32(acc) < -s11) else 0
            if _DBG:
                print(f"G {st.n_granule} in={in_pos} cursor={cursor} curprev={st.cursor_prev} "
                      f"fed={fed_pos} acc={acc:.9f} per={period} win={win} skip={skip} "
                      f"dec={st.dec_flag} adv={st.advanced} lt={st.last_transient} "
                      f"rms={rms:.6f} T={st.forced_T}", file=_sys.stderr)
            st.n_granule += 1
            if not st.advanced:
                in_pos = (in_pos + istep) % ring_tot
                fed_pos += istep
            st.advanced = 0
            st.cursor = cursor
            st.in_pos = in_pos
            st.acc = acc
            st.fed_pos = fed_pos

        # ---- drain ----
        cp_s = int(np.int32(st.cursor_prev & 0xFFFFFFFF))
        head = (cp_s - 100) if cp_s > 100 else 0
        wp_acc = st.write_ptr_acc
        dd0 = float(head) - wp_acc
        w9 = int(dd0) if dd0 > 0.0 else 0
        if _DBG:
            print(f"D cp={st.cursor_prev} head={head} wp={wp_acc:.6f} w9={w9} "
                  f"feedn={feed_n} fedtot={fed_total} outn={out_n}", file=_sys.stderr)
        if w9:
            count = int(float(w9) / st.pitch_ratio)
            if count:
                ph = math.fmod(wp_acc, float(out_len))
                if ph < 0:
                    ph += float(out_len)
                if _FAST_INTERP:
                    dst = _td_fast_mod.interp_nsamples_fast(
                        tbl, out_ring[:, :out_len], out_len, ph, 0, count,
                        st.pitch_ratio, float(st.pitch_ratio))
                else:
                    dst = interp_nsamples(tbl, out_ring[:, :out_len], out_len, ph, 0,
                                          count, st.pitch_ratio, float(st.pitch_ratio))
                out_chunks.append(np.ascontiguousarray(dst[:, :count].T))
                out_n += count
                wp_acc += st.total_ratio * float(count)
                st.write_ptr_acc = wp_acc

        if cursor_append >= _c_u32(nframes + gate + 8 * hop):
            break
        if fed_pos >= nstop and cursor_append >= nframes:
            break

    out = np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0, nch), np.float32)
    stats = {"n_granule": st.n_granule, "n_transient": st.n_transient,
             "wrap_cnt": st.wrap_cnt}
    return out, stats


# ---------------------------------------------------------------------------
# TD_FAST tier activation (must run last: the kernel certifications below call
# back into TIState/TDState and td_fast imports this module lazily).
# ---------------------------------------------------------------------------

if _td_fast_mod is not None:  # pragma: no cover - env dependent
    _FAST_TI = bool(_fast_on("TI") and _td_fast_mod.ti_ok())
    _FAST_INTERP = bool(_fast_on("INTERP") and _td_fast_mod._interp_ok())
    _FAST_OLA = bool(_fast_on("OLA"))
    _FAST_VDSP = bool(_fast_on("VDSP"))
    _FAST_PITCH = bool(_fast_on("PITCH") and _td_fast_mod.pitch_ok())
    if _fast_on("DS"):
        _FAST_DS = _td_fast_mod.ds_factor()


# ---------------------------------------------------------------------------
# Operator-level adapters used by tests/test_ops_parity.py (task-3 harness).
# Each mirrors the corresponding C entry point with a plain functional API so
# the parity suite can drive it directly against tools/gen_ref_ops.py output.
# ---------------------------------------------------------------------------

def geometry(sr: int, quality: int, solo: bool) -> dict:
    """rx_td_init derived geometry (hop/f28/pitch N,L1,maxbin,taper,lo,hi)."""
    st = TDState(int(sr), int(quality), int(bool(solo)), 2)
    return {
        "hop": st.hop,
        "f28": st.f28,
        "pitch_N": st.pitch_N,
        "pitch_L1": st.pitch_L1,
        "pitch_maxbin": st.pitch_maxbin,
        "pitch_taper_len": st.pitch_taper_len,
        "pitch_lo": st.pitch_lo,
        "pitch_hi": st.pitch_hi,
    }


def pick_env(sr: int, quality: int, solo: bool, picks, env_len: int = 64) -> dict:
    """rx_td_pick_env over a (a8, a9, a10, boolean) grid.

    ``picks`` is the flat int32 payload from gen_ref_ops.py.  Returns the same
    four outputs the C harness dumps (j / half / w10 / first env_len samples).
    """
    st = TDState(int(sr), int(quality), int(bool(solo)), 2)
    p = np.asarray(picks, dtype=np.int64).ravel()
    n = p.size // 4
    j = np.zeros(n, dtype=np.int32)
    half = np.zeros(n, dtype=np.int32)
    w10 = np.zeros(n, dtype=np.int32)
    env = np.zeros((n, env_len), dtype=np.float32)
    for i in range(n):
        tbl, h, w = st.pick_env(int(p[4 * i]), int(p[4 * i + 1]),
                                int(p[4 * i + 2]), int(p[4 * i + 3]))
        j[i] = st._last_pick_j
        half[i] = h
        w10[i] = w
        k = min(int(env_len), tbl.shape[0])
        env[i, :k] = tbl[:k]
    return {"pick_j": j, "pick_half": half, "pick_w10": w10, "pick_env": env}


def transient_positions(src, sr: int, nch: int, sens: float, chunks: int,
                        queries) -> np.ndarray:
    """rx_ti_create/process/analyze + rx_td_transient_pos over canned queries."""
    s = np.asarray(src, dtype=np.float32)
    if s.ndim == 2 or int(nch) == 1:
        inter = s.reshape(-1)
    else:
        inter = s.ravel()
    frames = inter.size // max(int(nch), 1)
    if int(nch) >= 2:
        plane = inter[:frames * int(nch)].reshape(frames, int(nch))
        mix = plane[:, 0].copy()
        for c in range(1, int(nch)):
            mix = mix + plane[:, c]
    else:
        mix = inter[:frames].copy()
    ti = TIState(int(nch), int(sr), float(sens))
    for _ in range(int(chunks)):
        ti.process(1024, mix)
        ti.analyze()
    q = np.asarray(queries, dtype=np.int64).ravel()
    out = np.empty(q.size // 2, dtype=np.float32)
    for i in range(q.size // 2):
        T = ti.get_transient_pos(int(q[2 * i]), int(q[2 * i + 1]))
        out[i] = -1.0 if T < 0 else float(T)
    return out


def pitch_analyze(ring, sr: int, quality: int, solo: bool, nch: int,
                  fed_visible: int, positions) -> dict:
    """rx_td_pitch_analyze over canned ring positions."""
    st = TDState(int(sr), int(quality), int(bool(solo)), int(nch))
    r = np.asarray(ring, dtype=np.float32)
    rt = st.f28 * 16
    if r.ndim == 2 and r.shape[0] == int(nch):
        channels = [np.ascontiguousarray(r[c]) for c in range(int(nch))]
    else:
        flat = r.ravel()
        channels = [np.ascontiguousarray(flat[c * rt:(c + 1) * rt])
                    for c in range(int(nch))]
    st.fed_visible = int(fed_visible)
    pos = np.asarray(positions, dtype=np.int64).ravel()
    lag = np.zeros(pos.size, dtype=np.float32)
    win = np.zeros(pos.size, dtype=np.int32)
    rms = np.zeros(pos.size, dtype=np.float32)
    for i, p in enumerate(pos):
        lg, w, rm = st.analyze_pitch(channels, int(p))
        lag[i] = np.float32(lg)
        win[i] = int(w)
        rms[i] = np.float32(rm)
    return {"lag": lag, "win": win, "rms": rms}


def transients_info_run(src, nch: int, sr: float, sens: float, chunks: int,
                        queries) -> dict:
    """TransientsInfo driver mirroring the gen_ref_ops.py td_ti harness.

    Returns vector/meta/iir_alpha plus rx_ti_get_transient_pos query answers.
    """
    s = np.asarray(src, dtype=np.float32).ravel()
    frames = s.size // max(int(nch), 1)
    if int(nch) >= 2:
        mix = (s[0:frames * int(nch):int(nch)]
               + s[1:frames * int(nch):int(nch)]).astype(np.float32, copy=True)
    else:
        mix = s[:frames].copy()
    ti = TIState(int(nch), int(sr), float(sens))
    for _ in range(int(chunks)):
        ti.process(1024, mix)
        ti.analyze()
    q = np.asarray(queries, dtype=np.int64).ravel()
    tpos = np.empty(q.size // 2, dtype=np.float32)
    for i in range(q.size // 2):
        tpos[i] = float(ti.get_transient_pos(int(q[2 * i]), int(q[2 * i + 1])))
    meta = np.array([ti.mag_n, ti.purge, -ti.Nm, ti.Nm, int(chunks)], dtype=np.int32)
    return {"vector": np.asarray(ti.mag[:ti.mag_n], dtype=np.float32),
            "meta": meta,
            "iir_alpha": float(ti.iir_alpha()),
            "transient_pos": tpos}
