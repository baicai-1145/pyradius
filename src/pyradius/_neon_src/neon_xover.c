/* pyradius NEON kernels — compiled at import time by pyradius/neon.py.
 *
 * Build: clang -O3 -mcpu=apple-m4 -shared -fPIC  (see pyradius/neon.py)
 * Loader: ctypes with EXPLICIT argtypes (see pyradius/vdsp.py for the lesson).
 *
 * All kernels are route-B (PYR_FAST=1) only.  The TD path never touches them.
 */
#include <arm_neon.h>
#include <math.h>
#include <stddef.h>

/* ---------------------------------------------------------------------------
 * xo_tree — 4-band crossover FIR, bit-exact replica of vocoder_ops._xover_zp_nb
 *
 * Geometry (matching Crossover.process / _xover_zp_nb):
 *   z  : [N - 1 + n] f32, z[t + k] = x_global[t - N + 1 + k]   (hist || x)
 *   tr : [n_bands][N] f32, taps_rev[b][k] = taps[b][N - 1 - k]
 *   out: [n_bands][n] f32, out[b][t] = sum_k z[t + k] * tr[b][k]
 *
 * The accumulation tree is NOT arbitrary.  The vocoder's peak detector makes
 * discrete decisions, so a 1e-7 crossover difference flips the peak count and
 * decorrelates the whole render (see the Crossover docstring).  This kernel
 * therefore reproduces numba's tree exactly — recovered empirically from the
 * kernel with a 65536-subset oracle (see the report), it is:
 *
 *   leaves : 16 independent fmaf chains over k (k += 16), ascending
 *   a1 = ((L4+L0) + (L1+L5)) + ((L2+L6) + L3)
 *   a2 = ((L7+L8) + L12) + L9
 *   a3 = (((L13+L10) + L14) + L11) + L15
 *   out = (a1 + a2) + a3
 *
 * Verified bit-identical (max|d| == 0, exact_frac == 1.0) to _xover_zp_nb.
 * The 4-row-per-accumulator NEON form keeps the identical per-lane chain and
 * the identical final tree, so it stays bit-exact while running ~2.6x faster
 * than the 4-thread numba kernel on one core.
 * ------------------------------------------------------------------------- */
void xo_tree(const float *z, const float *tr, float *out, int n, int N, int nb)
{
    for (int b = 0; b < nb; b++) {
        const float *tb = tr + (size_t)b * N;
        float *ob = out + (size_t)b * n;
        int t = 0;
        /* 4 output rows per accumulator register (lane = row offset). */
        for (; t + 4 <= n; t += 4) {
            float32x4_t v0 = vdupq_n_f32(0.0f),  v1 = vdupq_n_f32(0.0f);
            float32x4_t v2 = vdupq_n_f32(0.0f),  v3 = vdupq_n_f32(0.0f);
            float32x4_t v4 = vdupq_n_f32(0.0f),  v5 = vdupq_n_f32(0.0f);
            float32x4_t v6 = vdupq_n_f32(0.0f),  v7 = vdupq_n_f32(0.0f);
            float32x4_t v8 = vdupq_n_f32(0.0f),  v9 = vdupq_n_f32(0.0f);
            float32x4_t v10 = vdupq_n_f32(0.0f), v11 = vdupq_n_f32(0.0f);
            float32x4_t v12 = vdupq_n_f32(0.0f), v13 = vdupq_n_f32(0.0f);
            float32x4_t v14 = vdupq_n_f32(0.0f), v15 = vdupq_n_f32(0.0f);
            for (int k = 0; k < N; k += 16) {
                const float *zp = z + t + k;
                const float *tp = tb + k;
                v0  = vfmaq_n_f32(v0,  vld1q_f32(zp),      tp[0]);
                v1  = vfmaq_n_f32(v1,  vld1q_f32(zp + 1),  tp[1]);
                v2  = vfmaq_n_f32(v2,  vld1q_f32(zp + 2),  tp[2]);
                v3  = vfmaq_n_f32(v3,  vld1q_f32(zp + 3),  tp[3]);
                v4  = vfmaq_n_f32(v4,  vld1q_f32(zp + 4),  tp[4]);
                v5  = vfmaq_n_f32(v5,  vld1q_f32(zp + 5),  tp[5]);
                v6  = vfmaq_n_f32(v6,  vld1q_f32(zp + 6),  tp[6]);
                v7  = vfmaq_n_f32(v7,  vld1q_f32(zp + 7),  tp[7]);
                v8  = vfmaq_n_f32(v8,  vld1q_f32(zp + 8),  tp[8]);
                v9  = vfmaq_n_f32(v9,  vld1q_f32(zp + 9),  tp[9]);
                v10 = vfmaq_n_f32(v10, vld1q_f32(zp + 10), tp[10]);
                v11 = vfmaq_n_f32(v11, vld1q_f32(zp + 11), tp[11]);
                v12 = vfmaq_n_f32(v12, vld1q_f32(zp + 12), tp[12]);
                v13 = vfmaq_n_f32(v13, vld1q_f32(zp + 13), tp[13]);
                v14 = vfmaq_n_f32(v14, vld1q_f32(zp + 14), tp[14]);
                v15 = vfmaq_n_f32(v15, vld1q_f32(zp + 15), tp[15]);
            }
            float32x4_t a1 = vaddq_f32(vaddq_f32(vaddq_f32(v4, v0),
                                                 vaddq_f32(v1, v5)),
                                       vaddq_f32(vaddq_f32(v2, v6), v3));
            float32x4_t a2 = vaddq_f32(vaddq_f32(vaddq_f32(v7, v8), v12), v9);
            float32x4_t a3 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(v13, v10),
                                                           v14), v11), v15);
            vst1q_f32(ob + t, vaddq_f32(vaddq_f32(a1, a2), a3));
        }
        /* tail rows: scalar form, identical tree */
        for (; t < n; t++) {
            float s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;
            s0=s1=s2=s3=s4=s5=s6=s7=s8=s9=s10=s11=s12=s13=s14=s15=0.0f;
            for (int k = 0; k < N; k += 16) {
                const float *zp = z + t + k;
                const float *tp = tb + k;
                s0  = fmaf(zp[0],  tp[0],  s0);
                s1  = fmaf(zp[1],  tp[1],  s1);
                s2  = fmaf(zp[2],  tp[2],  s2);
                s3  = fmaf(zp[3],  tp[3],  s3);
                s4  = fmaf(zp[4],  tp[4],  s4);
                s5  = fmaf(zp[5],  tp[5],  s5);
                s6  = fmaf(zp[6],  tp[6],  s6);
                s7  = fmaf(zp[7],  tp[7],  s7);
                s8  = fmaf(zp[8],  tp[8],  s8);
                s9  = fmaf(zp[9],  tp[9],  s9);
                s10 = fmaf(zp[10], tp[10], s10);
                s11 = fmaf(zp[11], tp[11], s11);
                s12 = fmaf(zp[12], tp[12], s12);
                s13 = fmaf(zp[13], tp[13], s13);
                s14 = fmaf(zp[14], tp[14], s14);
                s15 = fmaf(zp[15], tp[15], s15);
            }
            float a1 = ((s4 + s0) + (s1 + s5)) + ((s2 + s6) + s3);
            float a2 = ((s7 + s8) + s12) + s9;
            float a3 = (((s13 + s10) + s14) + s11) + s15;
            ob[t] = (a1 + a2) + a3;
        }
    }
}

/* ---------------------------------------------------------------------------
 * xo_fast — reference/contestant variant, NOT bit-exact with the engine tree.
 *
 * Straight sequential accumulation over k with 4 rows per NEON register
 * (the shape of .tmp/xo_neon.c's xo2).  Different summation order, so its
 * output differs from _xover_zp_nb by ~1e-6; it is provided only for A/B
 * speed experiments and must never be selected on the default path.
 * ------------------------------------------------------------------------- */
void xo_fast(const float *z, const float *tr, float *out, int n, int N, int nb)
{
    for (int b = 0; b < nb; b++) {
        const float *tb = tr + (size_t)b * N;
        float *ob = out + (size_t)b * n;
        int t = 0;
        for (; t + 4 <= n; t += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int k = 0; k < N; k++)
                acc = vfmaq_n_f32(acc, vld1q_f32(z + t + k), tb[k]);
            vst1q_f32(ob + t, acc);
        }
        for (; t < n; t++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++)
                acc = fmaf(z[t + k], tb[k], acc);
            ob[t] = acc;
        }
    }
}

/* Version probe so the loader can reject a stale .dylib from a previous tag. */
int neon_kernels_abi(void) { return 1; }
