"""pyradius.sampler — Resampler::InterpolateNSamples port (interp_nsamples.c).

Table build (rx_interp_tbl_build): master kernel = sinc(j*pi/X) (double sin/div,
cast f32; x==0→1.0f), Kaiser beta=12 (I0 series, double), strict-sequential f32
sum normalize; per level: pitch = 1 + 2*level/n_levels (f32), ns = int(X/pitch+0.5)
when pitch>1 else X, ratio = ns/X (f32), sub-phase coefficients ratio*master[v24+j*ns].

Runtime (rx_interp_nsamples): per output sample i:
  frac = ph - floor-ish (C trunc toward zero; ph >= 0 here)
  sub  = int(fmaf(frac, nsubs, 0.5))   (double, trunc)
  base = int(ph) - offsets[sub]
  direct fast path: base >= 0 and base + taps <= ring: dot(strict j ascending)
  else per-index single-step wrap correction (one pass, non-wrapped re-check)
  taps == 0 → write zeros
  ph += rate (double)
Vectorized over output samples with np.take / reduceat; wrap path handled
per-sample for the (rare) boundary cases; kernel inner products via matmul-free
reduceat on gathered source windows.
"""
from __future__ import annotations

import math
import os
import numpy as np


_TABLE_CACHE_DIR = ".cache"


def _table_cache_path(X, n_levels, quality):
    import hashlib
    h = hashlib.sha256(f"{X}:{n_levels}:{quality}".encode()).hexdigest()[:16]
    return os.path.join(_TABLE_CACHE_DIR, f"itbl_{h}.npz")


def _table_cache_load(X, n_levels, quality):
    try:
        p = _table_cache_path(X, n_levels, quality)
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=False)
        Xc = int(z["X"]); nlc = int(z["n_levels"]); qc = float(z["quality"])
        if (Xc, nlc, qc) != (int(X), int(n_levels), float(quality)):
            return None
        nsubs = z["nsubs"]
        offsets = [z[f"off_{i}"] for i in range(nlc)]
        pools = [z[f"pool_{i}"] for i in range(nlc)]
        taps = [z[f"taps_{i}"] for i in range(nlc)]
        return (Xc, nlc, int(z["len"]), int(z["half"]), nsubs,
                offsets, pools, taps)
    except Exception:
        return None


def _table_cache_save(tbl, X, n_levels, quality):
    try:
        os.makedirs(_TABLE_CACHE_DIR, exist_ok=True)
        p = _table_cache_path(X, n_levels, quality)
        data = {"X": X, "n_levels": n_levels, "quality": quality,
                "len": tbl.len, "half": tbl.half, "nsubs": tbl.nsubs}
        for i in range(tbl.n_levels):
            data[f"off_{i}"] = tbl.offsets[i]
            data[f"pool_{i}"] = tbl.pools[i]
            data[f"taps_{i}"] = tbl.taps[i]
        np.savez(p, **data)
    except Exception:
        pass



def _i0(x: float) -> float:
    s = 1.0
    term = 1.0
    x2 = x * x * 0.25
    for k in range(1, 64):
        term *= x2 / (k * k)
        s += term
        if term < 1e-18 * s:
            break
    return s


def _kaiser(k: np.ndarray, beta: float) -> np.ndarray:
    n = len(k)
    if n < 2:
        if n == 1:
            k[0] = np.float32(1.0)
        return k
    denom = _i0(beta)
    n1 = float(n - 1)
    i = np.arange(n, dtype=np.float64)
    x = 2.0 * i / n1 - 1.0
    t = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    vals = np.array([_i0(beta * tv) for tv in t]) / denom
    return (k * vals.astype(np.float32)).astype(np.float32)


class InterpTable:
    def __init__(self, X: int, n_levels: int, quality: float):
        cached = _table_cache_load(X, n_levels, quality)
        if cached is not None:
            (self.X, self.n_levels, self.len, self.half, self.nsubs,
             self.offsets, self.pools, self.taps) = cached
            return
        xf = np.float32(X)
        lq = np.float32(xf * np.float32(quality))
        # C: len = (int)(lq + lq) | 1 — f32 add then trunc
        ln = int(np.float32(lq + lq)) | 1
        half = ln >> 1
        assert ln >= 1

        # master kernel: x = (j*pi)/X computed in double from integer j
        pi = 3.14159265358979323846
        j = np.arange(-half, -half + ln, dtype=np.float64)
        x = (j * pi) / float(X)
        master = np.where(x == 0.0, 1.0, np.sin(x) / np.where(x == 0.0, 1.0, x))
        m = master.astype(np.float32)
        m = _kaiser(m, 12.0)
        # strict sequential f32 sum
        ssum = np.float32(0.0)
        for v in m:
            ssum = np.float32(ssum + v)
        scale = np.float32(xf / ssum)
        m = (scale * m).astype(np.float32)

        self.X = X
        self.n_levels = n_levels
        self.len = ln
        self.half = half
        self.nsubs = np.empty(n_levels, dtype=np.int64)
        self.offsets: list[np.ndarray] = []
        # per level: pool of coefficient rows (taps_padded), row lengths
        self.pools: list[np.ndarray] = []      # [ns+1, max_taps] f32, zero-padded
        self.taps: list[np.ndarray] = []       # [ns+1] int

        for level in range(n_levels):
            pitch = np.float32(np.float32(level + level) / np.float32(n_levels) + np.float32(1.0))
            ns = X
            ratio = np.float32(1.0)
            if pitch > np.float32(1.0):
                ns = int(np.float32(np.float32(X) / pitch) + np.float32(0.5))
                ratio = np.float32(np.float32(ns) / np.float32(X))
            assert ns >= 1 and half >= ns
            self.nsubs[level] = ns
            offs = np.empty(ns + 1, dtype=np.int64)
            v24s = np.empty(ns + 1, dtype=np.int64)
            for s in range(ns + 1):
                v24 = half - s
                o = 0
                if v24 >= ns:
                    while True:
                        o += 1
                        v24 -= ns
                        if v24 < ns:
                            break
                offs[s] = o
                v24s[s] = v24
            taps = (ln - v24s) // ns + 1
            maxt = int(taps.max())
            pool = np.zeros((ns + 1, maxt), dtype=np.float32)
            for s in range(ns + 1):
                v24 = v24s[s]
                ncoef = 0
                jj = v24
                # coef = ratio * master[v24 + j*ns] while v24+j*ns < len
                idx = v24
                col = 0
                while idx < ln:
                    pool[s, col] = np.float32(ratio * m[idx])
                    idx += ns
                    col += 1
            self.offsets.append(offs)
            self.pools.append(pool)
            self.taps.append(taps)
        _table_cache_save(self, X, n_levels, quality)


def interp_kidx(quality: float, n_levels: int) -> int:
    t = np.float32((np.float32(quality) - np.float32(1.0)) * np.float32(0.5))
    k = int(np.float32(t * np.float32(n_levels) + np.float32(0.5)))
    if k > n_levels - 1:
        k = n_levels - 1
    if k < 0:
        k = 0
    return k



_interp_nb = None
try:
    from numba import njit

    @njit(cache=True)
    def _interp_nb(src, dst, base, taps, pool, sub, rs, nch, count):
        """Route-B resampler dot (strict-ascending f32 adds, wrap fix)."""
        for i in range(count):
            t = taps[i]
            if t <= 0:
                continue
            b = base[i]
            row = pool[sub[i]]
            for ch in range(nch):
                acc = np.float32(0.0)
                for jj in range(t):
                    idx = b + jj
                    if idx < 0:
                        idx += rs
                    elif idx >= rs:
                        idx -= rs
                    acc = np.float32(acc + np.float32(src[ch, idx] * row[jj]))
                dst[ch, i] = acc
except Exception:
    _interp_nb = None


def interp_nsamples(tbl: InterpTable, src: np.ndarray, ring_size: int,
                    phase: float, dst_off: int, count: int,
                    rate: float, quality: float) -> np.ndarray:
    """src: [nch, ring] float32; returns dst [nch, count] appended at logical offset.

    C semantics preserved: sub computed in double via fmaf(frac, nsubs, 0.5) then
    truncated; direct dot strict ascending j; out-of-range single-step wrap fix
    (idx<0 +=rs, >=rs -=rs, once); empty kernel → 0; ph += rate double.
    """
    if count == 0:
        return src  # unreachable in practice

    kidx = interp_kidx(quality, tbl.n_levels)
    dnsubs = float(tbl.nsubs[kidx])
    offs = tbl.offsets[kidx]
    pool = tbl.pools[kidx]
    taps_arr = tbl.taps[kidx]
    rs = ring_size

    ph = float(phase)
    phases = np.empty(count, dtype=np.float64)
    for i in range(count):
        phases[i] = ph
        ph = rate + ph
    iph = phases.astype(np.int64)  # C (int) trunc, phase >= 0
    frac = phases - iph
    # C: `int sub = (int)fmaf(frac, dnsubs, 0.5)` — frac (double) and dnsubs are
    # converted to float32 and the multiply-add is fused with a single f32
    # rounding.  float64 holds frac_f*subs_f exactly (24x24 bits), so rounding the
    # double sum back to f32 reproduces the fmaf result; the plain double route
    # picks a different sub index on exact-.5 boundaries (rare, but visible as
    # isolated samples in the TD output).
    frac_f = frac.astype(np.float32)
    subs_f = np.float32(dnsubs)
    sub = (frac_f.astype(np.float64) * float(subs_f) + 0.5)
    sub = sub.astype(np.float32).astype(np.int64)  # single f32 rounding, then trunc
    base = iph - offs[sub]
    taps = taps_arr[sub]

    nch = src.shape[0]
    dst = np.zeros((nch, count), dtype=np.float32)
    bad_precheck = bool(((taps > 0) & ((base < 0) | (base + taps > rs))).any())

    if _interp_nb is not None and not bad_precheck:
        _interp_nb(src, dst, base, taps, pool, sub, rs, nch, count)
        return dst
    ok_mask = (taps > 0) & (base >= 0) & (base + taps <= rs)
    ok_idx = np.nonzero(ok_mask)[0]
    if ok_idx.size:
        # gather windows and dot with kernel rows via einsum per channel
        b = base[ok_idx]
        t = taps[ok_idx]
        # pool row width (never smaller than the selected taps); padding rows are
        # masked below so the strict-ascending dot is unchanged.
        maxt = int(pool.shape[1])
        cols = np.arange(maxt)
        win_idx = b[:, None] + cols[None, :]
        valid = cols[None, :] < t[:, None]
        win_idx = np.where(valid, win_idx, 0)
        kern = pool[sub[ok_idx]]
        kern = np.where(valid, kern, np.float32(0.0))
        for ch in range(nch):
            windows = src[ch][win_idx]
            # strict ascending j: separate mul (single rounding) then sequential f32 adds
            prod = (windows * kern).astype(np.float32)
            acc = np.add.accumulate(prod, axis=1, dtype=np.float32)[:, -1]
            dst[ch, ok_idx] = acc.astype(np.float32)

    bad_mask = (taps > 0) & ~ok_mask
    bad_idx = np.nonzero(bad_mask)[0]
    for i in bad_idx:
        b = int(base[i])
        t = int(taps[i])
        row = pool[sub[i]]
        for ch in range(nch):
            acc = np.float32(0.0)
            for jj in range(t):
                idx = b + jj
                if idx < 0:
                    idx += rs
                elif idx >= rs:
                    idx -= rs
                acc = np.float32(acc + np.float32(src[ch, idx] * row[jj]))
            dst[ch, i] = acc
    return dst
