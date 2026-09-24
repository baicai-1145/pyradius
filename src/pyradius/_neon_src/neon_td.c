/* pyradius NEON TD kernels — compiled at import time by pyradius/neon.py.
 *
 * Build: clang -O3 -mcpu=apple-m4 -shared -fPIC -ffp-contract=off (see neon.py)
 * Loader: ctypes with EXPLICIT argtypes (see pyradius/vdsp.py for the lesson).
 *
 * All kernels here are TD_FAST=1 only (see pyradius/td_fast.py).
 *
 * td_ola — one call of the TD engine's DoOla (td_core._do_ola_exact), with
 *
 *   * the per-channel loop fused into the two segments + the leading plain
 *     window (numpy paid 2-3 fancy-index gather/scatter passes per channel),
 *     and
 *   * the ring / out-ring wrapping done by increment + reset instead of a
 *     np.where + mod over an index vector per call.
 *
 * Bit-exactness: same three f32 operations in the same association numpy uses
 *
 *     out[wi] = (f32)((f32)(out[wi] * s1) + (f32)((f32)(gain * sv) * s0))
 *
 * with separate mul/add — the build's -ffp-contract=off forbids folding them
 * into an fma, and the gather mask is the same `idx < fed_visible` test.  The
 * env table selection reproduces td_core.TDState.pick_env exactly (including
 * the truncating (2*a10)/3 for `boolean`).
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

int neon_td_abi(void) { return 1; }

/* pos -> in-range index: rx_td_wrap_ring in C integer semantics.
 * Only non-negative overshoot and negative p coming from `w10` occur; the
 * floor-mod form matches numpy's non-negative `%`. */
static inline long td_wrap(long p, long rstart, long rcap, long span)
{
    if (p >= 0 && p < rcap)
        return p;
    long m = (p - rcap) % span;
    if (m < 0)
        m += span;
    return rstart + m;
}

void td_ola(const float *ring, float *out, int nch, int rcap, int rstart,
            int outlen, int ostride, int fvis, int a4, int wr, int a8, int a9,
            int a10, int boolean, float gain, const float *env_flat,
            const int *env_off, const int *env_span, int ntab)
{
    if (nch <= 0 || a9 == 0)
        return;
    const long span_all = (rcap > rstart) ? (rcap - rstart) : 1;
    /* out was allocated with `outlen + 16` columns, so the per-channel row
     * stride is NOT outlen (the numpy path uses out[c] on a 2-D array and
     * therefore gets the real stride for free). */
    const size_t ostr = (size_t)ostride;

    /* ---- pick_env ---- */
    int v12 = boolean ? (2 * a10) / 3 : a9;
    int j = ntab - 1;
    for (int k = 0; k < ntab; k++) {
        if (v12 < env_span[a8 * ntab + k]) {
            j = (k == 0) ? 0 : k - 1;
            break;
        }
    }
    const int half = env_span[a8 * ntab + j] >> 1;
    const float *env = env_flat + env_off[a8 * ntab + j];

    /* ---- leading plain window (only when a4 == 0 && wr == 0) ---- */
    if ((wr | a4) == 0) {
        for (int i = 0; i < a9; i++) {
            long idx = td_wrap(i, rstart, rcap, span_all);
            const int vis = (idx < fvis);
            for (int c = 0; c < nch; c++) {
                float v = vis ? ring[(size_t)c * (size_t)rcap + (size_t)idx] : 0.0f;
                out[(size_t)c * ostr + (size_t)i] = (float)(gain * v);
            }
        }
    }

    if (half == 0)
        return;
    const int w10 = boolean ? 0 : ((a9 >> 1) - half);
    const long rbase = (long)a4 + (long)w10;
    const long wbase = (long)wr + (long)w10;
    const int w11 = boolean ? a10 : (a10 > a9 ? a10 : a9);

    long ri = td_wrap(rbase, rstart, rcap, span_all);
    long wi = wbase % outlen;
    if (wi < 0)
        wi += outlen;

    /* ---- segment 1: crossfade accumulate over [0, half) ---- */
    for (int i = 0; i < half; i++) {
        const float s0 = env[i];
        const float s1 = env[half + i];
        const int vis = ((long)ri < fvis);
        for (int c = 0; c < nch; c++) {
            const float sv = vis ? ring[(size_t)c * (size_t)rcap + (size_t)ri] : 0.0f;
            size_t o = (size_t)c * ostr + (size_t)wi;
            float t2 = (float)((float)(gain * sv) * s0);
            out[o] = (float)((float)(out[o] * s1) + t2);
        }
        if (++ri >= rcap)
            ri = rstart;
        if (++wi >= outlen)
            wi = 0;
    }

    /* ---- segment 2: plain copy tail over [half, w11) ---- */
    for (int i = half; i < w11; i++) {
        const int vis = ((long)ri < fvis);
        for (int c = 0; c < nch; c++) {
            const float sv = vis ? ring[(size_t)c * (size_t)rcap + (size_t)ri] : 0.0f;
            out[(size_t)c * ostr + (size_t)wi] = (float)(gain * sv);
        }
        if (++ri >= rcap)
            ri = rstart;
        if (++wi >= outlen)
            wi = 0;
    }
}
