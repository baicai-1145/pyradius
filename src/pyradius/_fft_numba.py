"""Optional numba accelerator for pyradius.fft's exact radix-2 kernel.

The C kernel (`rx_fft_cfft_kern`) is scalar float32 with `-ffp-contract=off`, so
every product and sum rounds to float32 on its own.  A numpy port can only
reproduce that by keeping the butterflies in float32 (float64 intermediates
diverge once cancellation is rough); numba reproduces it directly, ~4x faster.

Import is best-effort: pyradius.fft verifies the compiled kernel against the
numpy reference on first use and falls back if it disagrees (or if numba is not
installed / cannot compile in this environment).
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - environment dependent
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=False, boundscheck=False, nogil=True)
    def cfft_kern(re, im, rev, wr, wi, inverse):
        """Exact mirror of rx_fft_cfft_kern (float32 butterflies, no fma).

        `wr`/`wi` are float32 twiddle tables, so each `xr * c` is a float32
        multiply and each add is a float32 add — the same rounding sequence the
        C compiler emits for the scalar kernel.
        """
        n = re.shape[0]
        for i in range(n):
            j = rev[i]
            if i < j:
                t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
        ln = 2
        while ln <= n:
            half = ln >> 1
            step = n // ln
            for i0 in range(0, n, ln):
                for j in range(half):
                    t = j * step
                    c = wr[t]
                    s = wi[t]
                    if inverse:
                        s = -s
                    xr = re[i0 + j + half]
                    xi = im[i0 + j + half]
                    tr = np.float32(np.float32(xr * c) - np.float32(xi * s))
                    ti = np.float32(np.float32(xr * s) + np.float32(xi * c))
                    ur = re[i0 + j]
                    ui = im[i0 + j]
                    re[i0 + j] = np.float32(ur + tr)
                    im[i0 + j] = np.float32(ui + ti)
                    re[i0 + j + half] = np.float32(ur - tr)
                    im[i0 + j + half] = np.float32(ui - ti)
            ln <<= 1
