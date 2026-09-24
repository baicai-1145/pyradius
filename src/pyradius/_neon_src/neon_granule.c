/* pyradius route-B granule mega-kernel — compiled at import time by neon.py.
 *
 * Build: clang -O3 -mcpu=apple-m4 -shared -fPIC -ffp-contract=off
 *        -fno-math-errno -framework Accelerate       (see pyradius/neon.py)
 * Loader: ctypes with EXPLICIT argtypes (see pyradius/vdsp.py for the lesson).
 *
 * SCOPE (route B / PYR_FAST=1 only; the default path never touches this file):
 * shift the per-granule Python scheduling loop of VOC route B into C stage by
 * stage, so the ~10^5 Python->native conversions and buffer handoffs per render
 * disappear.  Every stage keeps its Python fallback; the caller only takes the C
 * path when the corresponding self-check (.tmp/selfcheck_granule.py) proves the C
 * code bit-equal to the route-B Python/numba code it replaces.
 *
 * NUMERICAL CONTRACT (recovered empirically, not guessed — the source text of a
 * fastmath kernel is NOT a reliable description of what runs; see
 * .tmp/probe2.py and .tmp/probe_phase_shape2.py):
 *
 *  - Route-B numba kernels declared fastmath=True (_ctp_fast_nb,
 *    _ctp_fast_nb_ph, _p2c_nb) DO contract into fused multiply-add:
 *        mag : m = sqrtf( fmaf(im, im, re*re) )      <- im is the multiplicand
 *        ph  : b = 8-term fma Horner; q = b*r2 (separate fmul);
 *              ang = fmaf(q, r, r)                   <- NOT r*(b*r2 + 1)
 *    Verified bit-exact over 8193 random bins spanning 1e-9..1e5 (float and
 *    vector lanes take the same path; the plain forms differ in ~75% of bins).
 *  - Route-B numba kernels declared fastmath=False keep *separate* fmul+fadd,
 *    which is exactly what clang emits under -ffp-contract=off, so those are
 *    written as plain C with the same f64 promotions (probe: 0 mismatches over
 *    4097 bins for the acs mag+envelope fusion).
 *  - f32 constants are emitted as exact hex float literals: decimal spellings
 *    of the engine's mov/movk immediates would double-round.
 *
 * FFT: Accelerate is linked directly (structure, function pointer and scratch
 * live in C), so a granule costs one ctypes call instead of ~20.
 */
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Exact f32 bit constants (never decimal — decimal literals double-round).
 * ------------------------------------------------------------------------- */
#define VG_MAG_CLAMP  0x1.1979980000000p-40f   /* 0x2B8CBCCC */
#define VG_TAU_A      0x1.99999a0000000p-5f    /* 0x3D4CCCCD */
#define VG_TAU_B      0x1.99999a0000000p-4f    /* 0x3DCCCCCD */
#define VG_PI         0x1.921fb60000000p+1f    /* 0x40490FDB */
#define VG_PI_2       0x1.921fb60000000p+0f    /* 0x3FC90FDB */
#define VG_2PI        0x1.921fb60000000p+2f    /* 0x40C90FDB */
#define VG_INV_2PI    0x1.45f3060000000p-3f    /* 0x3E22F983 */
#define VG_K07        0x1.6666660000000p-1f    /* 0x3F333333 */
#define VG_EPS1E6     0x1.0c6f7a0000000p-20f   /* 0x358637BD */
#define VG_CTP0       0x1.72199a0000000p-9f    /* 0x3B390CCD */
#define VG_CTP1      -0x1.05701a0000000p-6f    /* 0xBC82B80D */
#define VG_CTP2       0x1.5c336c0000000p-5f    /* 0x3D2E19B6 */
#define VG_CTP3      -0x1.32bff40000000p-4f    /* 0xBD995FFA */
#define VG_CTP4       0x1.b399e40000000p-4f    /* 0x3DD9CCF2 */
#define VG_CTP5      -0x1.22df3e0000000p-3f    /* 0xBE116F9F */
#define VG_CTP6       0x1.99734e0000000p-3f    /* 0x3E4CB9A7 */
#define VG_CTP7      -0x1.5554ba0000000p-2f    /* 0xBEAAAA5D */

/* ---------------------------------------------------------------------------
 * TimeToIirA (AudioProcessor::TimeToIirA @dtk 0x19EAB8), f32 chain.
 * ------------------------------------------------------------------------- */
static inline float vg_time_to_iir_a(float tau, float rate)
{
    if (tau == 0.0f)
        return 1.0f;
    float tr = tau * rate;
    float inv = -1.0f / tr;
    return 1.0f - (float)exp((double)inv);
}

/* ---------------------------------------------------------------------------
 * vDSP-backed real FFT plan (fwd [n] -> cart [n+2], inv cart -> [n]/n).
 *
 * Reproduces pyradius.vdsp_fast.fwd_r / inv_r call for call (same zrip calls,
 * same DC/Nyquist fixups, same 0.5 / 1/n scaling), but with the setup, split
 * buffers and cart destination owned by C.
 * ------------------------------------------------------------------------- */
typedef struct {
    int n, m, log2n, inv_n;
    FFTSetup setup;
    float *re, *im;
    float *cart;                        /* scratch for the fwd result */
} vg_fft;

void vg_fft_init(vg_fft *f, int n, float *cart)
{
    f->n = n;
    f->m = n >> 1;
    f->log2n = 0;
    for (int t = n; t > 1; t >>= 1)
        f->log2n++;
    f->inv_n = n;
    f->setup = vDSP_create_fftsetup((vDSP_Length)f->log2n, kFFTRadix2);
    f->re = (float *)calloc((size_t)f->m, sizeof(float));
    f->im = (float *)calloc((size_t)f->m, sizeof(float));
    f->cart = cart;
}

void vg_fft_fwd_run(vg_fft *f, const float *src, float *out)
{
    DSPSplitComplex z;
    z.realp = f->re;
    z.imagp = f->im;
    vDSP_ctoz((const DSPComplex *)src, 2, &z, 1, (vDSP_Length)f->m);
    vDSP_fft_zrip(f->setup, &z, 1, (vDSP_Length)f->log2n, FFT_FORWARD);
    vDSP_ztoc(&z, 1, (DSPComplex *)out, 2, (vDSP_Length)f->m);
    out[1] = 0.0f;
    out[f->n] = f->im[0];
    out[f->n + 1] = 0.0f;
    for (int i = 0; i <= f->n + 1; i++)
        out[i] = out[i] * 0.5f;
    out[1] = 0.0f;
    out[f->n + 1] = 0.0f;
}

void vg_fft_inv_run(vg_fft *f, const float *cart, float *out)
{
    DSPSplitComplex z;
    z.realp = f->re;
    z.imagp = f->im;
    vDSP_ctoz((const DSPComplex *)cart, 2, &z, 1, (vDSP_Length)f->m);
    f->re[0] = cart[0];
    f->im[0] = cart[f->n];
    vDSP_fft_zrip(f->setup, &z, 1, (vDSP_Length)f->log2n, FFT_INVERSE);
    vDSP_ztoc(&z, 1, (DSPComplex *)out, 2, (vDSP_Length)f->m);
    float s = 1.0f / (float)f->n;
    for (int i = 0; i < f->n; i++)
        out[i] = out[i] * s;
}

/* ---------------------------------------------------------------------------
 * Per-granule carrier for the whole route-B pipeline.
 *
 * One instance per VocoderState (built once by Python, see
 * pyradius/vocoder_core.py `_cg_attach`).  Every pointer refers to a numpy
 * buffer that outlives the render, so a granule performs no allocation.
 * ------------------------------------------------------------------------- */
typedef struct {
    /* geometry */
    int N, nb_bins, m_fft, mb_bins, n_write, hop, nch, nbands, ring_cap;
    int ring_out_len, sr, precision, buffered_1352;
    int v8_len_a, v9_len_b;
    float step_base, ratio;
    float fold_alpha, fold_c;

    /* FFT + windows + tables */
    vg_fft fwd, inv;
    const float *win0;                  /* win_table[0], n_write */
    const float *synth_a;
    const float *synth_b;
    const float *edge_gain;

    /* containers */
    float *acc;                         /* N */
    float *cart;                        /* N + 2 */
    float *mag;                         /* [nch][nb_bins] */
    float *mask;                        /* [nch][nb_bins] */
    float *acs_env;                     /* [nch][nb_bins] */
    float *region_gain;                 /* [nch][nb_bins] */
    float *phase;                       /* [nch][nb_bins] */
    float *mag_copy;                    /* [nch][nb_bins] */
    float *mask_copy;                   /* [nch][nb_bins] */
    int *peak_bins;                     /* nb_bins */
    int *reg_start;
    int *reg_end;
    float *reg_offset;
    int peak_count;

    /* scratch */
    float *scratch_8a8;                 /* nb_bins */
    float *win1;                        /* N */
    float *win2;                        /* N */
    float *out_ring;                    /* [nch][ring_out_len] */
    float *sync_weight;                 /* nb_bins */
} vg_state;

static inline int vg_imax(int a, int b) { return a > b ? a : b; }

/* ---------------------------------------------------------------------------
 * |cart| -> mag, CartToPolar phase -> mask, Threshold_LT clamp  (ACS tail).
 * ------------------------------------------------------------------------- */
static inline float vg_ctp_phase(float ar, float ai)
{
    float are = fabsf(ar);
    float aim = fabsf(ai);
    float mx = are > aim ? are : aim;
    float mx0 = mx == 0.0f ? 1.0f : mx;
    float mn = aim > are ? are : aim;
    float r = mn / mx0;
    float r2 = r * r;
    float p = fmaf(VG_CTP0, r2, VG_CTP1);
    p = fmaf(p, r2, VG_CTP2);
    p = fmaf(p, r2, VG_CTP3);
    p = fmaf(p, r2, VG_CTP4);
    p = fmaf(p, r2, VG_CTP5);
    p = fmaf(p, r2, VG_CTP6);
    p = fmaf(p, r2, VG_CTP7);
    float q = p * r2;                   /* separate fmul, then folded into r*p */
    float ang = fmaf(q, r, r);
    if (aim > are)
        ang = VG_PI_2 - ang;
    if (ar < 0.0f)
        ang = VG_PI - ang;
    if (signbit(ai))
        ang = -ang;
    return ang;
}

/* ---------------------------------------------------------------------------
 * vg_acs1 — _ev_acs front half: win0 multiply, spectrum #1, envelope IIR.
 * ------------------------------------------------------------------------- */
void vg_acs1(vg_state *st, int ch, int step_1412, int transient_state)
{
    const int N = st->N;
    const int NB = st->nb_bins;
    const int n_win = st->n_write;
    float *acc = st->acc;

    float tau = (transient_state == 2) ? VG_TAU_A : VG_TAU_B;
    float hopf = (step_1412 > 0) ? (float)step_1412 : 288.0f;
    float iir_a = vg_time_to_iir_a(tau, (float)st->sr / hopf);

    const float *win0 = st->win0;
    for (int i = 0; i < n_win; i++)
        acc[i] = acc[i] * win0[i];

    vg_fft_fwd_run(&st->fwd, acc, st->cart);

    /* NOTE the deliberate asymmetry, which route B itself has and which must be
     * copied for bit-equality: here the magnitude is the fastmath=False
     * _acs_mag_env_nb shape (plain re*re + im*im, separate roundings), NOT the
     * fma shape that vg_acs2 / cart_to_polar use for the same |cart|. */
    float *env = st->acs_env + (size_t)ch * NB;
    for (int k = 0; k < NB; k++) {
        float re = st->cart[2 * k];
        float im = st->cart[2 * k + 1];
        float m = sqrtf(re * re + im * im);
        env[k] = (float)((double)iir_a * (double)(m - env[k]) + (double)env[k]);
    }

    memset(acc, 0, (size_t)N * sizeof(float));
}

/* ---------------------------------------------------------------------------
 * vg_acs2 — _ev_acs back half: spectrum #2 -> mag/mask (+ Threshold clamp).
 * ------------------------------------------------------------------------- */
void vg_acs2(vg_state *st, int ch, const float *cart2)
{
    const int NB = st->nb_bins;
    float *mag = st->mag + (size_t)ch * NB;
    float *mask = st->mask + (size_t)ch * NB;
    for (int k = 0; k < NB; k++) {
        float re = cart2[2 * k];
        float im = cart2[2 * k + 1];
        float m = sqrtf(fmaf(im, im, re * re));
        if (m < VG_MAG_CLAMP)           /* np.maximum: NaN stays NaN */
            m = VG_MAG_CLAMP;
        mag[k] = m;
        mask[k] = vg_ctp_phase(re, im);
    }
}

/* ---------------------------------------------------------------------------
 * vg_find_peaks — _ev_find_peaks: bit-equal scalar replica of _find_peaks_nb.
 * ------------------------------------------------------------------------- */
int vg_find_peaks(vg_state *st, int ch)
{
    const int NB = st->nb_bins;
    const float *mag = st->mag + (size_t)ch * NB;
    const float *env = st->acs_env + (size_t)ch * NB;
    float *rg = st->region_gain + (size_t)ch * NB;
    int *peak_bins = st->peak_bins;
    float *reg_offset = st->reg_offset;
    int *reg_start = st->reg_start;
    int *reg_end = st->reg_end;
    int cnt = 0;

    for (int i = 1; i < NB - 1; i++)
        if (mag[i] > mag[i - 1] && mag[i] > mag[i + 1])
            peak_bins[cnt++] = i;

    if (cnt == 0) {
        peak_bins[0] = NB / 2;
        reg_offset[0] = (float)(NB / 2);
        reg_start[0] = 0;
        reg_end[0] = NB - 1;            /* FIX #1: not NB (that reads mag[NB]) */
        st->peak_count = 1;
        return 1;
    }

    for (int j = 0; j < cnt; j++) {
        int pj = st->peak_bins[j];
        float ap = mag[pj + 1];
        float am = mag[pj];
        float amm = mag[pj - 1];
        float v20 = (ap - 2.0f * am) - amm;     /* two roundings, no fma */
        float off = (v20 != 0.0f) ? (ap - amm) / (v20 + v20) : 0.0f;
        if (off > VG_K07)
            off = VG_K07;
        else if (off < -VG_K07)
            off = -VG_K07;
        reg_offset[j] = (float)pj + off;
    }

    int b = 0;
    float best = mag[0];
    for (int i = 1; i <= peak_bins[0]; i++)
        if (mag[i] < best) { best = mag[i]; b = i; }
    reg_start[0] = b;
    for (int j = 0; j < cnt - 1; j++) {
        int s = peak_bins[j];
        int e = peak_bins[j + 1];
        b = s;
        best = mag[s];
        for (int i = s + 1; i < e; i++)
            if (mag[i] < best) { best = mag[i]; b = i; }
        reg_end[j] = b;
        reg_start[j + 1] = b;
    }
    int pk_last = peak_bins[cnt - 1];
    b = pk_last;
    best = mag[pk_last];
    for (int i = pk_last + 1; i < NB; i++)
        if (mag[i] < best) { best = mag[i]; b = i; }
    reg_end[cnt - 1] = b;

    for (int j = 0; j < cnt; j++) {
        int s = reg_start[j];
        int e = reg_end[j];
        if (e > s) {
            float mn = env[s];
            float mx = env[s];
            for (int i = s + 1; i < e; i++) {
                float v = env[i];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float ratio = (mn > 0.0f) ? mx / mn : 1.5f;
            for (int i = s; i < e; i++)
                rg[i] = ratio;
        }
    }
    st->peak_count = cnt;
    return cnt;
}

/* ---------------------------------------------------------------------------
 * vg_pull_to_peak — loose phase locking (inline @0x168B0C), bit-equal replica
 * of _pull_nb.  Each intermediate rounds to f32 on its own.
 *
 * The loops are FLAT and contiguous (bins are visited in ascending order), not
 * per region: the Python side builds one concatenated index set with
 * np.repeat/np.cumsum for all live regions and indexes phase[]/mask[] with it,
 * and the consecutive float ops differ from a per-region walk (verified:
 * a per-region loop mismatches ~2500 of 4097 bins, the flat walk is exact).
 * ------------------------------------------------------------------------- */
void vg_pull_to_peak(vg_state *st, int ch, int prev_granule, int step_1412)
{
    const int cnt = st->peak_count;
    if (cnt < 1)
        return;
    float f1 = (float)prev_granule;
    float f2 = (float)step_1412;
    if (f1 <= 0.0f)
        return;
    float ratio = f2 / f1;
    float v74 = 0.0f;
    if (ratio > 1.0f) {
        float r_clamp = (ratio >= 4.0f) ? 1.0f : (ratio - 1.0f) / 3.0f;
        v74 = sqrtf(r_clamp);
    }
    float lo = 0.5f + 0.5f * v74;
    float hi = 0.5f + 4.0f * v74;
    float inv_range = (hi > lo) ? 1.0f / (hi - lo) : 0.0f;

    const int NB = st->nb_bins;
    float *phase = st->phase + (size_t)ch * NB;
    const float *mask = st->mask + (size_t)ch * NB;
    const float *rg = st->region_gain + (size_t)ch * NB;

    /* Flatten the live regions into one ascending bin walk, carrying the
     * owning peak bin, exactly like the np.repeat construction. */
    int r = 0;
    for (int rr = 0; rr < cnt; rr++) {
        int rs = st->reg_start[rr];
        int re_ = st->reg_end[rr];
        if (re_ <= rs)
            continue;                   /* lens > 0 filter */
        int pk = st->peak_bins[rr];
        for (int b = rs; b < re_; b++, r++) {
            (void)r;
            float g = rg[b];
            float w;
            if (g >= hi)
                w = 1.0f;
            else if (g > lo)
                w = (g - lo) * inv_range;
            else
                w = 0.0f;
            if (!(w > 0.0f) || b == pk)
                continue;
            float pm_bin = phase[b];
            float x0 = phase[pk] - pm_bin;
            float q0 = rintf(x0 * VG_INV_2PI);
            float t0 = (float)((double)q0 * -6.2831854820251465 + (double)x0);
            float x1 = mask[b] - mask[pk];
            float q1 = rintf(x1 * VG_INV_2PI);
            float t1 = (float)((double)q1 * -6.2831854820251465 + (double)x1);
            float s = t0 + t1;
            float qs = rintf(s * VG_INV_2PI);
            float sw = (float)((double)qs * -6.2831854820251465 + (double)s);
            phase[b] = (float)((double)w * (double)sw + (double)pm_bin);
        }
    }
}

/* ---------------------------------------------------------------------------
 * vg_p2c — polar -> cartesian, bit-equal replica of _p2c_nb (fastmath=True:
 * f32 cosf/sinf of the f32 phase, plain m*cos product).
 * ------------------------------------------------------------------------- */
void vg_p2c(const float *mag, const float *phase, float *cart, int n)
{
    cart[1] = 0.0f;
    cart[2 * n + 1] = 0.0f;
    for (int i = 0; i < n; i++) {
        float m = mag[i];
        float pv = phase[i];
        /* cosf/sinf, NOT (float)cos((double)pv): numba's fastmath=True libm
         * maps to the f32 math functions, which differ from the f64 result by
         * an ulp in ~0.05% of bins (probe: 2 mismatches over 8193 bins). */
        cart[2 * i] = m * cosf(pv);
        cart[2 * i + 1] = m * sinf(pv);
    }
}

/* ---------------------------------------------------------------------------
 * vg_fold_iir_sub — fftshift + fwd/bwd first-order IIR + sub, bit-equal
 * replica of _fold_nb.
 *
 * The recurrence shape is NOT the obvious double one.  numba's signature is
 * (f32[:], f32[:], f32[:], int64, int64, float32, float64) — `alpha` was
 * specialised to f32 while `c` stayed f64 — and LLVM then contracts the
 * f32*f32 multiply to f32 BEFORE the add is promoted to f64.  An exact-rational
 * probe (.tmp/probe_fold_form.py) pins it:
 *
 *     prev = (float)(alpha * x)   [f32 rounding]
 *          + c * prev            [f64, single rounding]
 *
 * i.e. `(double)(float)(alpha*(float)x) + c*prev`.  Writing the natural
 * f64 multiplication instead matches only ~55% of the samples (verified over
 * N = 8/64/100/2048/4096/8192 with several hop values: this form gives 0
 * mismatches, the f64 form thousands).
 * ------------------------------------------------------------------------- */
void vg_fold_iir_sub(const float *acc, float *w1, float *w2, int N, int hop,
                     float alpha, double c)
{
    const int half = N >> 1;
    for (int i = 0; i < half; i++) {
        w1[half + i] = acc[i];
        w1[i] = acc[half + i];
        w2[half + i] = acc[i];
        w2[i] = acc[half + i];
    }
    double x0 = (double)w1[0];
    double prev = (double)(float)(alpha * (float)x0) + c * x0;
    w1[0] = (float)prev;
    for (int i = 1; i < N; i++) {
        prev = (double)(float)(alpha * w1[i]) + c * prev;
        w1[i] = (float)prev;
    }
    const int stop = half - hop;
    double xn = (double)w1[N - 1];
    prev = (double)(float)(alpha * (float)xn) + c * xn;
    w1[N - 1] = (float)prev;
    for (int i = N - 2; i > stop - 1; i--) {
        prev = (double)(float)(alpha * w1[i]) + c * prev;
        w1[i] = (float)prev;
    }
    for (int i = 0; i < 2 * hop; i++) {
        int j = stop + i;
        w2[j] = w2[j] - w1[j];
    }
}

/* ---------------------------------------------------------------------------
 * vg_sync_nch2 — _ev_sync for nch == 2, bit-equal replica of
 * synchronize_stereo_phases_nch2_fast (all regions flattened, one pass).
 * ------------------------------------------------------------------------- */
static inline float vg_wrap_pi(float x)
{
    float r = rintf(x * VG_INV_2PI);
    return fmaf(-r, VG_2PI, x);          /* fmsub wrap, correctly rounded */
}

void vg_sync_nch2(vg_state *st, float sens)
{
    const int NB = st->nb_bins;
    const int cnt = st->peak_count;
    float *weight = st->sync_weight;
    memset(weight, 0, (size_t)NB * sizeof(float));
    if (st->nch < 2 || cnt < 1)
        return;

    float *dest0 = st->phase, *dest1 = st->phase + NB;
    const float *src0 = st->mask, *src1 = st->mask + NB;
    const float *mag0 = st->mag, *mag1 = st->mag + NB;
    const float inv_sens = 1.0f / (sens + VG_EPS1E6);

    for (int r = 0; r < cnt; r++) {
        int rs = st->reg_start[r];
        int re_ = st->reg_end[r];
        if (re_ <= rs)
            continue;
        int pk = st->peak_bins[r];
        float m0 = mag0[pk];
        float m1 = mag1[pk];
        float v9 = fabsf(m0 - m1) / ((m0 + m1) + VG_EPS1E6);
        float v10 = inv_sens * v9;
        float v11;
        if (v10 > 0.0f)
            v11 = (v10 < VG_K07) ? (v10 / VG_K07) : 1.0f;
        else
            v11 = 0.0f;
        float w = sqrtf(1.0f - v11);
        for (int b = rs; b < re_; b++) {
            weight[b] = w;
            float s0 = src0[b], s1 = src1[b];
            float d0 = dest0[b], d1 = dest1[b];
            float a0 = vg_wrap_pi(d0 - s0);
            float a1 = vg_wrap_pi(d1 - s1);
            float mm = (a0 + a1) * 0.5f;
            if (!(fabsf(a1 - a0) < VG_PI))
                mm = mm + VG_PI;
            mm = vg_wrap_pi(mm);
            float t0 = vg_wrap_pi((s0 + mm) - d0);
            float t1 = vg_wrap_pi((s1 + mm) - d1);
            /* _fma_arr in route B is _fma_arr_fast: a plain f64 multiply-add
             * rounded once to f32, NOT a correctly-rounded fmaf. */
            dest0[b] = vg_wrap_pi((float)((double)w * (double)t0 + (double)d0));
            dest1[b] = vg_wrap_pi((float)((double)w * (double)t1 + (double)d1));
        }
    }
}

/* ---------------------------------------------------------------------------
 * vg_rpt — _ev_rpt (ResetPhasesForTransients).
 *
 * With the vocoder's arguments this function has exactly two outcomes: while
 * the transient flag is set (mode_4b8 == 1) the whole phase row is replaced by
 * mask * scale (an f64 product rounded to f32); otherwise branch B3 returns
 * immediately because avg_m1 == avg_0 == 0 <= 2.0, i.e. it is a no-op copy.
 * mask_table is never modified by either path.
 * ------------------------------------------------------------------------- */
void vg_rpt(vg_state *st, int ch, int transient_flag, int transient_state,
            double scale)
{
    (void)transient_state;
    if (transient_flag != 1)
        return;
    const int NB = st->nb_bins;
    float *phase = st->phase + (size_t)ch * NB;
    const float *mask = st->mask + (size_t)ch * NB;
    /* scale_20 is a python float (f64) in the Python path, so the product must
     * stay f64 until the single f32 store. */
    for (int i = 0; i < NB; i++)
        phase[i] = (float)((double)mask[i] * scale);
}

/* ---------------------------------------------------------------------------
 * vg_fill_granule — FillGranule / FillGranuleWin (rx_fill_granule.c), modes
 * "fg" and "fgw" in the buffered branch (the only one the vocoder takes).
 *
 * mode 0 ("fg", _ev_fill_granule):
 *     acc[:] = 0;  off = floormod(cursor - hop, cap);
 *     acc[i] = sum_{k<nb} ring[k][wrap(off + i)]      (f32 add, band order;
 *              exactly one conditional subtract for the wrap, as in the C)
 * mode 1 ("fgw", _ev_fgwin_x4): for every band, two windowed wings with an
 *     f64 multiply-add rounded once to f32:
 *     acc[i]          += r[off + hop + i] * w[hop + i]     i < hop
 *     acc[N5 - hop + i] += r[off + i] * w[i]               i < hop
 *
 * Both are bit-equal to _fg_bandsum_nb / _fgw_nb (verified against the live
 * numba kernels at 44.1 and 48 kHz).
 * ------------------------------------------------------------------------- */
void vg_fill_granule(vg_state *st, int mode, int cursor,
                     const float *ring, const float *win)
{
    const int nb = st->nbands;
    const int cap = st->ring_cap;
    const int N = st->n_write;
    const int hop = st->hop;
    const int N5 = st->N;
    const int buffered = st->buffered_1352 && cursor >= hop;
    float *acc = st->acc;    int d = cursor - hop;
    int off = d % cap;
    if (off < 0)
        off += cap;                     /* floormod */

    if (mode == 0) {
        memset(acc, 0, (size_t)N5 * sizeof(float));
        if (buffered) {
            /* buffered branch (cursor >= hop): single-subtract ring wrap */
            for (int i = 0; i < N; i++) {
                int j = off + i;
                if (j >= cap)
                    j -= cap;               /* literally one subtract */
                float s = ring[j];
                for (int k = 1; k < nb; k++)
                    s = (float)(s + ring[(size_t)k * (size_t)cap + (size_t)j]);
                acc[i] = s;
            }
            return;
        }
        if (d >= 0 && cursor < cap - hop) {
            /* direct branch: contiguous read at d */
            for (int i = 0; i < N; i++) {
                float s = ring[d + i];
                for (int k = 1; k < nb; k++)
                    s = (float)(s + ring[(size_t)k * (size_t)cap + (size_t)(d + i)]);
                acc[i] = s;
            }
            return;
        }
        /* cursor < hop (or cursor >= cap - hop): clamp the indices to [0, cap-1]
         * (the np.clip form) */
        for (int i = 0; i < N; i++) {
            int j = d + i;
            if (j < 0)
                j = 0;
            else if (j > cap - 1)
                j = cap - 1;
            float s = ring[j];
            for (int k = 1; k < nb; k++)
                s = (float)(s + ring[(size_t)k * (size_t)cap + (size_t)j]);
            acc[i] = s;
        }
        return;
    }

    /* mode 1: FillGranuleWin x nbands, gain == 1.0, hop >= 1.
     *
     * The buffered flag is forced on here: _ev_fgwin_x4 calls fill_granule with
     * buffered=True unconditionally, while _ev_fill_granule passes
     * `bool(self.buffered_1352) if cursor >= hop else False`.  Reusing the mode-0
     * value would send the first granule (cursor < hop) down the clamp branch and
     * diverge from the Python path.
     *
     * Both wings land at N5 - hop.  The two shapes are NOT the same arithmetic:
     *   no ring wrap -> f64 multiply-add rounded once to f32
     *   ring wrap    -> f64 product rounded to f32 first (`r[i]*w` promotes to
     *                   f64 in numpy), then a plain f32 add into acc
     */
    if (N + off <= cap) {
        for (int band = 0; band < nb; band++) {
            const float *r = ring + (size_t)band * (size_t)cap;
            const float *w = win + (size_t)band * (size_t)N;
            for (int i = 0; i < hop; i++) {
                acc[i] = (float)((double)r[off + hop + i] * (double)w[hop + i]
                                 + (double)acc[i]);
                const int j = N5 - hop + i;
                acc[j] = (float)((double)r[off + i] * (double)w[i]
                                 + (double)acc[j]);
            }
        }
        return;
    }
    for (int band = 0; band < nb; band++) {
        const float *r = ring + (size_t)band * (size_t)cap;
        const float *w = win + (size_t)band * (size_t)N;
        for (int i = 0; i < hop; i++) {
            int i1 = off + hop + i;
            if (i1 >= cap)
                i1 -= cap;
            int i2 = off + i;
            if (i2 >= cap)
                i2 -= cap;
            const float p1 = (float)((double)r[i1] * (double)w[hop + i]);
            const float p2 = (float)((double)r[i2] * (double)w[i]);
            acc[i] = (float)(acc[i] + p1);
            const int j = N5 - hop + i;
            acc[j] = (float)(acc[j] + p2);
        }
    }
}

/* ABI probe so the loader can reject a stale .dylib from a previous tag. */
int neon_granule_abi(void) { return 1; }
