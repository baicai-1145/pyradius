/* pyradius route-B formant kernel — C port of pyradius.vocoder_ops.FormantState.
 *
 * Build: clang -O3 -mcpu=apple-m4 -shared -fPIC -ffp-contract=off
 *        -fno-math-errno -framework Accelerate      (see pyradius/neon.py)
 * Loader: ctypes with EXPLICIT argtypes.
 *
 * SCOPE (route B / PYR_FAST=1 only).  Replaces the per-granule numba chain
 *   _fm_prep_nb -> _fft -> _fm_env2_nb -> _fm_peak_nb -> _fm_clip_mirror_nb ->
 *   _fft -> _fm_ker_scale_nb -> iir16 -> _fm_kermix_nb -> _fm_gain2_nb ->
 *   [rms] -> _fm_ges_tail_nb
 * with one ctypes call per (granule, channel).
 *
 * NUMERICAL CONTRACT — recovered by TRACING the live kernels on a real render
 * (.tmp/fm_stage_trace.py), never by reading their source text.  The traps:
 *
 *  - dB curve: DOUBLE log, one f64 multiply by the f64 literal 8.68588924407959,
 *    one f32 store.  A float logf path agrees with numba in only ~82% of bins.
 *  - numpy/numba promote a *python float* argument to f64.  `_fm_peak_nb` takes
 *    (v19, freq_hi, freq_lo, thr_score) as python floats, so its divisions,
 *    `+0.5` roundings and the w1/w2 weights are all f64 until the f32 store.
 *    `_fm_gain2_nb`'s `int(k2f - 0.5)` is likewise trunc(f64(k2f) - 0.5).
 *  - env / ges / kermix are fma single-rounding.
 *  - gain2's exponent is exp((double)d * (double)strength * 0.115129254758358):
 *    a THREE-factor f64 product, then exp, then one f32 store.  Computing
 *    d*strength in f32 first double-rounds differently.
 *  - the RMS branch uses numpy's *pairwise* summation (blocks of 128, 8-way
 *    unrolled, recursive halves), not a linear sum.  A 1.7e-12 relative shift
 *    here was once enough to move a render's correlation.
 *  - `_t2a_fast` rounds exp() to f32 BEFORE the 1.0 - x subtraction.
 *  - the two internal FFTs are vDSP fft_zip on split re/im, matching the
 *    existing route-B ctypes call in FormantState._fft.
 *
 * fm_apply_algo selects the smoothing implementation:
 *   0 = exact 16-pass IIR, bit-identical (the verified default)
 *   1 = H8 frequency-domain approximation: NOT bit-identical (see the header of
 *       .tmp/h8_split2.py; the residual is the reference's own finite-domain end
 *       conditions, ~1e1 dB on the first ~100 bins of a real db curve).  Only
 *       reachable through PYR_FM_H8=1 and rejected by the bit-exactness gate.
 */
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

/* the already-certified bit-exact IIR (neon_data.c) */
extern void iir16(float *y, int n, double a2, double a1m);

/* --------------------------------------------------------------------------
 * exact f32 bit constants
 * ------------------------------------------------------------------------ */
#define FM_AMP2DB_K   8.68588924407959      /* f64 literal (numba source) */
#define FM_DB2AMP_K   0.115129254758358     /* f64 literal (numba source) */
#define FM_T_CONST2   0x3BA3D70Au           /* 0.005f (prec_mode==2) */
#define FM_T_CONST1   0x3C23D70Au           /* 0.01f  */
#define FM_W012       0x3DF5C28Fu           /* 0.12f  */
#define FM_C_0P5      0x3F000000u
#define FM_C_0P2      0x3E4CCCCDu
#define FM_C_1P5      0x3FC00000u
#define FM_C_1P2      0x3F99999Au
#define FM_C_2P0      0x40000000u
#define FM_C_0P3      0x3E99999Au
#define FM_C_4P7      0x40966666u
#define FM_C_5P0      0x40A00000u
#define FM_C_EPS1E6   0x358637BDu
#define FM_C_20       0x41A00000u
#define FM_C_M40      0xC2200000u
#define FM_C_500      0x43FA0000u
#define FM_C_800      0x44480000u
#define FM_C_150      0x43160000u
#define FM_C_THRSC    0xC9742400u           /* -1000000.0f */
#define FM_C_0P8      0x3F4CCCCDu
#define FM_C_0P75     0x3F400000u
#define FM_C_0P25     0x3E800000u
#define FM_C_1P0      0x3F800000u

static inline float fm_f(uint32_t u)
{
    float f;
    __builtin_memcpy(&f, &u, sizeof f);
    return f;
}

/* Amp2DB threshold: double bits 0x3BC79CA10C922342 = 9.999999999988105e-21
 * (a decimal spelling double-rounds, so the bit pattern is used verbatim). */
static inline double fm_amp2db_thresh(void)
{
    uint64_t u = 0x3BC79CA10C922342ULL;
    double d;
    __builtin_memcpy(&d, &u, sizeof d);
    return d;
}

/* asm round chain used by the python path: int(v + (v<0 ? -0.5 : 0.5)) */
static inline int fm_round_d(double v)
{
    return (int)(v + (v < 0.0 ? -0.5 : 0.5));
}

/* --------------------------------------------------------------------------
 * _t2a_fast: f32 chain, exp in f64 rounded to f32 BEFORE the subtraction
 * ------------------------------------------------------------------------ */
static inline float fm_time_to_iir_a(float tau, float rate)
{
    if (tau == 0.0f)
        return 1.0f;
    float tr = tau * rate;
    float inv = -1.0f / tr;
    float e = (float)exp((double)inv);
    return 1.0f - e;
}

/* --------------------------------------------------------------------------
 * numpy pairwise summation (_pairwise_sum for float64)
 *   n < 8      -> linear
 *   n <= 128   -> 8 partial sums + a fixed reduction tree
 *   else       -> split at n2 = (n/2) rounded down to a multiple of 8
 * ------------------------------------------------------------------------ */
#define FM_PW_BLOCK 128

static double fm_pairwise(const double *a, size_t n)
{
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; i++)
            res += a[i];
        return res;
    }
    if (n <= FM_PW_BLOCK) {
        double r[8];
        for (int j = 0; j < 8; j++)
            r[j] = a[j];
        size_t i = 8;
        size_t stop = n - (n % 8);
        for (; i < stop; i += 8) {
            r[0] += a[i + 0]; r[1] += a[i + 1];
            r[2] += a[i + 2]; r[3] += a[i + 3];
            r[4] += a[i + 4]; r[5] += a[i + 5];
            r[6] += a[i + 6]; r[7] += a[i + 7];
        }
        for (; i < n; i++)
            r[0] += a[i];
        return ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return fm_pairwise(a, n2) + fm_pairwise(a + n2, n - n2);
}

/* --------------------------------------------------------------------------
 * vDSP fft_zip carrier
 * ------------------------------------------------------------------------ */
typedef struct {
    int n, m, log2n, ok;
    FFTSetup setup;
    float *re, *im;
} fm_fft;

static void fm_fft_init(fm_fft *f, int n)
{
    memset(f, 0, sizeof *f);
    f->n = n;
    f->m = n >> 1;
    for (int t = n; t > 1; t >>= 1)
        f->log2n++;
    f->setup = vDSP_create_fftsetup((vDSP_Length)f->log2n, kFFTRadix2);
    f->re = (float *)calloc((size_t)f->m, sizeof(float));
    f->im = (float *)calloc((size_t)f->m, sizeof(float));
    f->ok = (f->setup && f->re && f->im) ? 1 : 0;
}

static inline void fm_fft_zip(const fm_fft *f, float *re, float *im, int inverse)
{
    DSPSplitComplex z;
    z.realp = re;
    z.imagp = im;
    vDSP_fft_zip(f->setup, &z, 1, (vDSP_Length)f->log2n,
                 inverse ? FFT_INVERSE : FFT_FORWARD);
}

/* --------------------------------------------------------------------------
 * stage functions — each is an exact replica of the numba kernel / python
 * expression it replaces (types follow the traced argument promotion).
 * ------------------------------------------------------------------------ */

/* steps 1+2-prep */
void fm_prep(const float *mag, float *db, float *ffr, float *ffi, int NB, int M)
{
    const double thr = fm_amp2db_thresh();
    for (int i = 0; i < NB; i++) {
        double mv = (double)mag[i];
        if (mv >= thr) {
            double a = (mv > 1e-300) ? mv : 1e-300;
            db[i] = (float)(log(a) * FM_AMP2DB_K);
        } else {
            db[i] = (float)(-391.0);
        }
    }
    /* db[NB:M] keeps the kermix residue of the previous call; ffr copies it. */
    for (int i = 0; i < M; i++) {
        ffr[i] = db[i];
        ffi[i] = 0.0f;
    }
}

/* step 3 */
void fm_env2(const float *ffr, const float *ffi, float *env, int MB, double a1)
{
    for (int k = 0; k < MB; k++) {
        float re = ffr[k], im = ffi[k];
        float e = re * re + im * im;
        env[k] = (float)(a1 * (double)(e - env[k]) + (double)env[k]);
    }
}

/* steps 4+5 — all scalar args are python floats, hence f64 */
void fm_peak(const float *env, int MB, double v19, double freq_hi, double freq_lo,
             int mode_freq, double thr_score,
             int *best_out, double *fC_out, int *peak_out)
{
    const float eps = fm_f(FM_C_EPS1E6);
    int hi = fm_round_d(v19 / freq_hi);
    if (hi > MB - 2) hi = MB - 2;
    if (hi < 2) hi = 2;
    int lo = fm_round_d(v19 / freq_lo);
    if (lo > MB - 2) lo = MB - 2;
    int best = 0;
    if (lo > hi) {
        float bmax = 0.0f;
        int bi = 0;
        for (int k = hi; k < lo; k++) {
            float v = env[k];
            float den1 = (env[k - 1] + env[k + 1]) + eps;
            float r1 = v / den1;
            /* the weights are np.float32-expressed in the kernel, so every
             * operation here stays f32 (traced; f64 would differ) */
            float w1;
            if (r1 <= fm_f(FM_C_0P5))
                w1 = fm_f(FM_C_0P2);
            else if (r1 < fm_f(FM_C_2P0))
                w1 = (r1 - fm_f(FM_C_0P5)) / fm_f(FM_C_1P5) + fm_f(FM_C_0P2);
            else
                w1 = fm_f(FM_C_1P2);
            float r2 = v / (env[k >> 1] + eps);
            float w2;
            if (r2 > fm_f(FM_C_0P3)) {
                if (r2 < fm_f(FM_C_5P0))
                    w2 = (r2 - fm_f(FM_C_0P3)) / fm_f(FM_C_4P7);
                else
                    w2 = fm_f(FM_C_1P0);
            } else {
                w2 = 0.0f;
            }
            float sc = (v * w1) * w2;
            if (k == hi) { bmax = sc; bi = 0; }
            else if (sc > bmax) { bmax = sc; bi = k - hi; }
        }
        if ((double)bmax > thr_score)
            best = hi + bi;
    }
    double pk;
    int peak;
    if (mode_freq) {
        pk = best ? (v19 / (double)best) : 0.0;
        peak = best;
    } else {
        pk = 500.0;
        peak = fm_round_d(v19 / pk);
    }
    double fC = (pk <= 800.0) ? pk : 800.0;
    fC = (fC >= 150.0) ? fC : 150.0;
    *best_out = best; *fC_out = fC; *peak_out = peak;
}

/* step 5-tail + 6-prep */
void fm_clip_mirror(float *ffr, float *ffi, int MB, int M, int c4)
{
    for (int j = 0; j < 4; j++) {
        int k = (c4 >> 1) - 2 + j;
        float f = (j < 2) ? fm_f(FM_C_0P75) : fm_f(FM_C_0P25);
        if (k >= 0 && k < MB) {
            ffr[k] = ffr[k] * f;
            ffi[k] = ffi[k] * f;
        }
    }
    for (int k = (c4 >> 1); k < MB; k++) {
        ffr[k] = 0.0f;
        ffi[k] = 0.0f;
    }
    ffi[0] = 0.0f;
    ffi[MB - 1] = 0.0f;
    for (int k = 1; k < MB - 1; k++) {
        ffr[M - k] = ffr[k];
        ffi[M - k] = -ffi[k];
    }
}

/* step 6-post */
void fm_ker_scale(const float *ffr, float *ker, int M, double inv_scale)
{
    for (int i = 0; i < M; i++)
        ker[i] = (float)((double)ffr[i] * inv_scale);
}

/* step 8 */
void fm_kermix(float *db, const float *ker, int M, int h1, int h2)
{
    if (h1 > 0) {
        for (int i = 0; i < h1; i++)
            db[i] = ker[i];
    }
    if (h2 > h1) {
        float span = (float)(h2 - h1);
        for (int k = h1; k < h2; k++) {
            float w = (float)(h2 - k) / span;
            db[k] = (float)((double)w * (double)(ker[k] - db[k]) + (double)db[k]);
        }
    }
}

/* step 9 */
void fm_gain2(const float *db, float *gscr, int NB, double ratio, double strength)
{
    float last_d = 0.0f;
    for (int v = 0; v < NB; v++) {
        float k2f = (float)ratio * (float)v;   /* np.float32(ratio)*np.float32(v) */
        int k2 = (k2f < 0.0f) ? (int)((double)k2f - 0.5) : (int)((double)k2f + 0.5);
        float d;
        if (k2 < NB) {
            d = db[k2] - db[v];
            last_d = d;
        } else {
            d = last_d;
        }
        if (d > fm_f(FM_C_20)) d = fm_f(FM_C_20);
        else if (d < fm_f(FM_C_M40)) d = fm_f(FM_C_M40);
        gscr[v] = (float)exp((double)d * strength * FM_DB2AMP_K);
    }
}

/* step 10 (scratch must hold 2*NB doubles) */
void fm_rms(const float *mag, float *gscr, int NB, double rms_eps, double *scr)
{
    double *mm = scr;
    double *gg = scr + NB;
    for (int i = 0; i < NB; i++) {
        double m = (double)mag[i];
        mm[i] = m * m;
        gg[i] = (double)gscr[i];
    }
    double den = fm_pairwise(mm, (size_t)NB);
    for (int i = 0; i < NB; i++)
        gg[i] = (gg[i] * gg[i]) * mm[i];
    double num = fm_pairwise(gg, (size_t)NB);
    float ss = (float)sqrt(den / (num + rms_eps));
    for (int i = 0; i < NB; i++)
        gscr[i] = ss * gscr[i];
}

/* steps 11+12 */
void fm_ges_tail(const float *gscr, float *ges, float *mag, int NB, double a3,
                 double fC, int N, double sr)
{
    for (int v = 0; v < NB; v++)
        ges[v] = (float)(a3 * (double)(gscr[v] - ges[v]) + (double)ges[v]);
    /* np.float32(fC * np.float32(0.8)) * np.float32(N) / np.float32(sr) */
    float vt = (float)((float)((double)fC * (double)fm_f(FM_C_0P8))
                       * (float)N / (float)sr);
    int tail = fm_round_d((double)vt);
    if (tail > NB - 1)
        tail = NB - 1;
    for (int v = tail; v < NB; v++)
        mag[v] = ges[v] * mag[v];
}

/* --------------------------------------------------------------------------
 * exact 16-pass smoothing: delegates to the certified iir16 kernel
 * ------------------------------------------------------------------------ */
void fm_iir16(float *y, int n, double a2, double a1m)
{
    iir16(y, n, a2, a1m);
}

/* --------------------------------------------------------------------------
 * H8 frequency-domain smoothing (opt-in approximation, NOT bit-exact).
 *
 * The 16 passes are a zero-phase 8th-power low-pass with analytic response
 *     H8(k) = (a2^2 / (1 - 2*a1m*cos(2*pi*k/K) + a1m^2))^8.
 * Applied through a real FFT on a constant-extended pad it matches the
 * reference to ~1e-5 dB in the interior, but the reference's own finite-domain
 * end conditions survive: measured |d| ~1e1 dB over the first ~100 bins and
 * ~1e-1 dB at the far end of a real db curve.  Hence OFF by default.
 * ------------------------------------------------------------------------ */
static float *fm_h8_tab = NULL;
static int fm_h8_k = 0;
static float fm_h8_a2 = -1.0f;

static void fm_h8_tab_build(int K, float a2)
{
    int m = K >> 1;
    if (fm_h8_tab == NULL || fm_h8_k != K) {
        free(fm_h8_tab);
        fm_h8_tab = (float *)malloc((size_t)(m + 1) * sizeof(float));
        fm_h8_k = K;
        fm_h8_a2 = -1.0f;
    }
    if (fm_h8_tab == NULL)
        return;
    if (fm_h8_a2 == a2)
        return;
    const double a2d = (double)a2;
    const double a1m = 1.0 - a2d;
    for (int k = 0; k <= m; k++) {
        double den = 1.0 - 2.0 * a1m * cos(2.0 * M_PI * (double)k / (double)K)
                     + a1m * a1m;
        fm_h8_tab[k] = (float)pow(a2d * a2d / den, 8.0);
    }
    fm_h8_a2 = a2;
}

/* pad/scratch must hold K floats; fft must be an fm_fft of size K */
static void fm_h8_run(float *y, int n, float a2, float *pad, fm_fft *fft)
{
    const int K = fft->n, m = fft->m;
    const int reach = (int)(8.0 / (double)(1.0f - a2)) + 20;
    if (n + 2 * reach > K)
        return;                                  /* caller sized it wrong */
    fm_h8_tab_build(K, a2);
    if (fm_h8_tab == NULL)
        return;
    for (int i = 0; i < reach; i++)
        pad[i] = y[0];
    memcpy(pad + reach, y, (size_t)n * sizeof(float));
    for (int i = reach + n; i < K; i++)
        pad[i] = y[n - 1];
    DSPSplitComplex z;
    z.realp = fft->re;
    z.imagp = fft->im;
    vDSP_ctoz((const DSPComplex *)pad, 2, &z, 1, (vDSP_Length)m);
    vDSP_fft_zrip(fft->setup, &z, 1, (vDSP_Length)fft->log2n, FFT_FORWARD);
    vDSP_vmul(fft->re, 1, fm_h8_tab, 1, fft->re, 1, (vDSP_Length)m);
    vDSP_vmul(fft->im, 1, fm_h8_tab, 1, fft->im, 1, (vDSP_Length)m);
    fft->im[0] = 0.0f;                           /* the zrip pack's Nyquist slot */
    vDSP_fft_zrip(fft->setup, &z, 1, (vDSP_Length)fft->log2n, FFT_INVERSE);
    vDSP_ztoc(&z, 1, (DSPComplex *)pad, 2, (vDSP_Length)m);
    float sc = 1.0f / (float)K;
    vDSP_vsmul(pad, 1, &sc, pad, 1, (vDSP_Length)K);
    memcpy(y, pad + reach, (size_t)n * sizeof(float));
}

/* --------------------------------------------------------------------------
 * carrier: pointers into the python-owned FormantState buffers
 * ------------------------------------------------------------------------ */
typedef struct {
    int NB, N, M, MB, nb_bands, prec_mode;
    int ges_stride;             /* row stride of gain_env (python: NB+8) */
    float sr, inv_scale_f;
    float *db, *gscr, *ker, *env, *gain_env, *ffr, *ffi;
    double *rms_scr;            /* [2*NB] */
    float *h8_pad;              /* [h8_k] */
    fm_fft fft_m, fft_h8;
    int h8_ready;
} fm_state;

int fm_state_init(fm_state *s, int NB, int N, int M, int MB, int nb_bands,
                  int prec_mode, int ges_stride, float sr, float inv_scale,
                  float *db, float *gscr, float *ker, float *env,
                  float *gain_env, float *ffr, float *ffi)
{
    memset(s, 0, sizeof *s);
    s->NB = NB; s->N = N; s->M = M; s->MB = MB;
    s->nb_bands = nb_bands; s->prec_mode = prec_mode;
    s->ges_stride = ges_stride > 0 ? ges_stride : NB;
    s->sr = sr; s->inv_scale_f = inv_scale;
    s->db = db; s->gscr = gscr; s->ker = ker; s->env = env;
    s->gain_env = gain_env; s->ffr = ffr; s->ffi = ffi;
    fm_fft_init(&s->fft_m, M);
    s->rms_scr = (double *)calloc((size_t)(2 * NB > 8 ? 2 * NB : 8), sizeof(double));
    if (!s->fft_m.ok || !s->rms_scr)
        return -1;
    return 0;
}

/* one band, whole ApplyFormantCorrection; mag is modified in place */
int fm_apply(fm_state *s, float *mag, int band, double ratio, double strength,
             double width, double freq_hi, double freq_lo, int mode_freq,
             int mode_rms, int f580, int active, int algo)
{
    const int NB = s->NB, M = s->M, N = s->N;
    float *ges;
    if (active != 1 || ratio == 1.0 || strength == 0.0)
        return 0;
    if (NB < 1 || s->MB < 1)
        return 0;
    if (band < 0 || band >= s->nb_bands)
        return -1;
    ges = s->gain_env + (size_t)band * (size_t)s->ges_stride;

    /* 1+2 */
    fm_prep(mag, s->db, s->ffr, s->ffi, NB, M);
    fm_fft_zip(&s->fft_m, s->ffr, s->ffi, 0);

    /* 3 */
    float tconst = (s->prec_mode == 2) ? fm_f(FM_T_CONST2) : fm_f(FM_T_CONST1);
    float rate1 = (float)((double)s->sr / (double)(float)f580);
    double a1 = (double)fm_time_to_iir_a(tconst, rate1);
    fm_env2(s->ffr, s->ffi, s->env, s->MB, a1);

    /* 4+5 */
    /* _f32f(_f32f(sr) * _f32f(M) / _f32f(N)): an f64 product/quotient of
     * f32-rounded operands, then one f32 store */
    double v19 = (double)(float)(((double)s->sr * (double)M) / (double)N);
    int best, peak; double fC;
    fm_peak(s->env, s->MB, v19, freq_hi, freq_lo, mode_freq,
            fm_f(FM_C_THRSC), &best, &fC, &peak);
    (void)best;

    /* cut: f32((peak-2)/f32(width)) then f32(that * f32(min(ratio,1))) */
    float widthf = (float)width;
    double r_ = (ratio < 1.0) ? ratio : 1.0;
    float q = (float)((double)(peak - 2) / (double)widthf);
    float q2 = (float)((double)q * (double)(float)r_);
    int cut = fm_round_d((double)q2);
    if (cut < 2) cut = 2;
    int c4 = 2 * cut;

    /* 5-tail + 6 */
    fm_clip_mirror(s->ffr, s->ffi, s->MB, M, c4);
    fm_fft_zip(&s->fft_m, s->ffr, s->ffi, 1);
    fm_ker_scale(s->ffr, s->ker, M, (double)s->inv_scale_f);

    /* 7 */
    float w012 = (float)((double)widthf * 0.12);   /* _f32f(width * 0.12) */
    float tau2 = (float)(fC * (double)w012);       /* _f32f(fC * w012) */
    float rate2 = (float)((double)(float)N / (double)s->sr);
    float a2 = fm_time_to_iir_a(tau2, rate2);
    if (algo == 1) {
        if (!s->h8_ready) {
            int need = 1;
            while (need < NB + 2 * ((int)(8.0 / (double)(1.0f - a2)) + 20) ||
                   need < 4096)
                need <<= 1;
            fm_fft_init(&s->fft_h8, need);
            if (s->fft_h8.ok) {
                s->h8_pad = (float *)calloc((size_t)need, sizeof(float));
                s->h8_ready = s->h8_pad ? 1 : 0;
            }
        }
        if (s->h8_ready)
            fm_h8_run(s->db, NB, a2, s->h8_pad, &s->fft_h8);
    } else {
        fm_iir16(s->db, NB, (double)a2, (double)(float)(1.0 - (double)a2));
    }

    /* 8 */
    /* F32(N)*F32(4.0)/F32(sr) is an f32 chain in the python path */
    int h1 = fm_round_d((double)(((float)N * 4.0f) / s->sr));
    if (h1 > M - 1) h1 = M - 1;
    int h2 = fm_round_d((double)(((float)N * 9.0f) / s->sr));
    if (h2 > M) h2 = M;
    fm_kermix(s->db, s->ker, M, h1, h2);

    /* 9 + 10 + 11 + 12 */
    float a3 = fm_time_to_iir_a(tconst, (float)((double)s->sr / (double)(float)f580));
    fm_gain2(s->db, s->gscr, NB, ratio, strength);
    if (mode_rms != 0)
        fm_rms(mag, s->gscr, NB, 1.000029594723506e-12, s->rms_scr);
    fm_ges_tail(s->gscr, ges, mag, NB, (double)a3, fC, N, (double)s->sr);
    return 0;
}

int neon_formant_abi(void) { return 1; }
