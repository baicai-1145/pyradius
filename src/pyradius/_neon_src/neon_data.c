/* pyradius NEON data-movement kernels — compiled at import time by neon.py.
 *
 * Build: clang -O3 -mcpu=apple-m4 -shared -fPIC -ffp-contract=off (see neon.py)
 * Loader: ctypes with EXPLICIT argtypes (see pyradius/vdsp.py for the lesson).
 *
 * All kernels here are route-B (PYR_FAST=1) only and must be BIT-EXACT with
 * their numba counterparts (pyradius.vocoder_ops._iir16_nb / _fg_bandsum_nb /
 * _oac_nb); pyradius.neon.data_ok() re-verifies that on first use.
 *
 * BIT-EXACTNESS NOTES (each one was recovered empirically, not guessed):
 *
 *  iir16   — numba's signature is (float32[:], float64, float64): the caller
 *            passes *python floats*, so a2/a1m stay f64 inside the loop.  The
 *            generated arm64 is:  fmul d2, a2, y[i] ; fmul d1, a1m, prev ;
 *            fadd d1, d1, d2 ; fcvt s2, d1 ; str  — i.e. NO contraction.
 *            clang's default -ffp-contract=fast would fold that into a single
 *            fma and change the rounding, so the build uses
 *            -ffp-contract=off (verified: xo_tree's explicit vfmaq/fmaf are
 *            unaffected by that flag).
 *
 *  fg_bandsum — f32 accumulation in strict band order (k ascending) with an
 *            f32 round after every add, and a *single* conditional subtract
 *            for the ring wrap (reproduced literally, even though a single
 *            subtract is only sufficient while off + N <= 2*cap).
 *
 *  oac     — the t2/t chain is "correctly-rounded fmaf" implemented in f64 via
 *            TwoSum + one-ulp correction (_fma_arr in vocoder_ops).  The f64
 *            steps must stay un-contracted, hence the same build flag.
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------------------
 * iir16 — 8x forward+backward first-order IIR, bit-exact _iir16_nb.
 *
 * Per pass:
 *   fwd: prev = a2*y[0] + (1-a2)*y[0]; write back f32; then
 *        prev = a2*y[i] + a1m*prev   (i ascending, f32 write-back each step)
 *   bwd: same from n-1 downward.
 * prev is a *double*; y[] is read as f32->f64 (before being overwritten) and
 * written back as f32.  Only the f64 add is on the true dependency chain, so
 * this is latency-bound: ~8 cycles/step, ~1.5ns/step on M4.
 * ------------------------------------------------------------------------- */
void iir16(float *y, int n, double a2, double a1m)
{
    if (n < 1)
        return;
    double one_minus = 1.0 - a2;
    for (int p = 0; p < 8; p++) {
        double zi = one_minus * (double)y[0];
        double prev = a2 * (double)y[0] + zi;
        y[0] = (float)prev;
        for (int i = 1; i < n; i++) {
            prev = a2 * (double)y[i] + a1m * prev;
            y[i] = (float)prev;
        }
        zi = one_minus * (double)y[n - 1];
        prev = a2 * (double)y[n - 1] + zi;
        y[n - 1] = (float)prev;
        for (int i = n - 2; i >= 0; i--) {
            prev = a2 * (double)y[i] + a1m * prev;
            y[i] = (float)prev;
        }
    }
}

/* ---------------------------------------------------------------------------
 * fg_bandsum — mode=fg band sum with ring wraparound, bit-exact _fg_bandsum_nb.
 *
 * ring: [nb][cap] f32 C-contiguous; acc: N f32.
 *   acc[i] = sum_{k=0}^{nb-1} ring[k][wrap(off + i)]   (f32 add, band order)
 * ------------------------------------------------------------------------- */
void fg_bandsum(const float *ring, float *acc, int off, int N, int cap, int nb)
{
    if (nb < 1 || N < 1 || cap < 1)
        return;
    for (int i = 0; i < N; i++) {
        int j = off + i;
        if (j >= cap)
            j -= cap;                       /* literally one subtract */
        float s = ring[j];
        for (int k = 1; k < nb; k++)
            s = (float)(s + ring[(size_t)k * (size_t)cap + (size_t)j]);
        acc[i] = s;
    }
}

/* ---------------------------------------------------------------------------
 * Correctly-rounded fmaf, bit-exact replica of _fma_arr / _oac_nb's inline
 * TwoSum + one-ulp correction.  All intermediate math is f64 and must not be
 * contracted.
 * ------------------------------------------------------------------------- */
static inline float f32_of_bits(uint32_t b)
{
    float f;
    __builtin_memcpy(&f, &b, sizeof f);
    return f;
}

static inline uint32_t bits_of_f32(float f)
{
    uint32_t b;
    __builtin_memcpy(&b, &f, sizeof b);
    return b;
}

static inline float fma_rn(double ad, double bd, double cd)
{
    double p = ad * bd;
    double s = p + cd;
    double bb = s - p;
    double err = (p - (s - bb)) + (cd - bb);
    float r = (float)s;
    double e = (s - (double)r) + err;
    if (isfinite(s)) {
        uint32_t rb = bits_of_f32(r);
        int neg = (rb >> 31) != 0;
        uint32_t ub = neg ? rb - 1u : rb + 1u;
        float up = f32_of_bits(ub);
        double ulp = (double)up - (double)r;
        if (e > 0.5 * ulp)
            return up;
        uint32_t db = neg ? rb + 1u : rb - 1u;
        float dn = f32_of_bits(db);
        double dulp = (double)r - (double)dn;
        if (e < -0.5 * dulp)
            return dn;
        return r;
    }
    return (float)s;
}

/* ---------------------------------------------------------------------------
 * oac — fused overlap-add ring write, bit-exact _oac_nb.
 *
 * win1/win2 are read at [wh .. wh+N); synth_a/synth_b at [0 .. N); out is the
 * per-channel ring of length R.  v197/v198 are f32 values passed as python
 * floats by the caller (so they are exact f32 in f64 form here too).
 *
 *   t2 = f32(v198 * f32(win2[wh+i] * synth_b[i]))
 *   t  = fmaf(win1[wh+i], synth_a[i], t2)      (correctly rounded)
 *   wrap = cursor < hop || N + pos0 > R
 *   pidx = wrap ? (max(pos0,0) + i - fvi) % R : pos0 + i   (fvi = hop-cursor if
 *          cursor < hop else 0; i < fvi is skipped; non-wrap skips pidx >= R)
 *   out[pidx] = i < v35 (wrap) / i < v36 (no-wrap)
 *                 ? fmaf(t, v197, out[pidx]) : f32(t * v197)
 *
 * v197/v198 arrive as *python floats* (f64) — the numba signature is float64,
 * so e.g. 0.7125 keeps all 52 mantissa bits.  In the fmaf(t, v197, ...) step
 * that makes the f64 product inexact, so the TwoSum correction is only an
 * approximation of a correctly-rounded fmaf — reproduce it literally rather
 * than "fixing" it with a real fmaf.
 * ------------------------------------------------------------------------- */
void oac(float *out, const float *win1, const float *win2,
         const float *synth_a, const float *synth_b,
         int R, int N, int pos0, int v35, int v36, int wh,
         int cursor, int hop, double v197, double v198)
{
    int wrap = (cursor < hop) || (N + pos0 > R);
    int pcur = pos0 < 0 ? 0 : pos0;
    int fvi = (cursor - hop < 0) ? (hop - cursor) : 0;
    for (int i = 0; i < N; i++) {
        double inner = (double)win2[wh + i] * (double)synth_b[i];
        float t2 = (float)(v198 * (double)(float)inner);
        float t = fma_rn((double)win1[wh + i], (double)synth_a[i], (double)t2);
        if (wrap) {
            if (i < fvi)
                continue;
            int pidx = (pcur + (i - fvi)) % R;
            if (i < v35)
                out[pidx] = fma_rn((double)t, v197, (double)out[pidx]);
            else if (i >= v36 && i >= v35)
                out[pidx] = (float)(v197 * (double)t);
        } else {
            int pidx = pos0 + i;
            if (pidx >= R)
                continue;
            if (pidx < 0)
                pidx += R;      /* numba wraps negative indices (unreachable in
                                 * the real path: pos0<0 implies cursor<hop and
                                 * hence wrap==true, but keep it faithful) */
            if (i < v36)
                out[pidx] = fma_rn((double)t, v197, (double)out[pidx]);
            else
                out[pidx] = (float)(v197 * (double)t);
        }
    }
}
