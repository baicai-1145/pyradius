# pyradius — Pure-Python/NumPy Time-Domain & Phase-Vocoder Pitch Shifter

> ⚠️ **Research & Academic Disclaimer**: This project is a clean-room Python/NumPy DSP reimplementation of time-domain (TD) and phase-vocoder pitch-shifting algorithms for research and interoperability purposes. All trademarks belong to their respective owners.

* **TD** (time domain) — `pyradius/td_core.py`: OLA granule engine with
  transient-aware envelope selection and FFT-autocorrelation pitch analysis.
  Reproduces the C renderer **bit for bit** (see below).
* **VOC** (vocoder) — `pyradius/vocoder_core.py`: phase-vocoder chain
  (analysis FFT → peak search → unwrap → pitch coherence → formant compensation →
  overlap-add synthesis) driven by the C event schedule.

The C sources under `libradius/` are the specification and are treated as
read-only. Every number in this README is produced by the harnesses in `tools/`
and can be reproduced from a clean checkout.

---

## Quick start

```bash
pip install -e .                       # numpy, scipy, soundfile
pip install -e '.[fast]'               # + numba (recommended, see Performance)

# TD pitch shift, +3 semitones, quality 37 (solo 0)
pyr-td audio_test/1.wav out_td.wav 3 37 0

# Vocoder pitch shift, -3 semitones, precision 2
pyr-vc audio_test/1.wav out_vc.wav -3 2

# equivalent module entry points
python3 -m pyradius.cli_td audio_test/1.wav out_td.wav 3 37 0
python3 -m pyradius.cli_vc audio_test/1.wav out_vc.wav -3 2
```

Both CLIs mirror the C drivers (`libradius/tools/rx_td_render.c`,
`rx_vc_render.c`) argument for argument, including the optional trailing
`truth.wav` correlation report for TD.

---

## numba is an optional accelerator, not a dependency

The FFT kernel (`pyradius/fft.py`) has two interchangeable implementations:

* a **NumPy** implementation, which is the reference for both correctness and
  behaviour;
* an optional **numba** implementation (`pyradius/_fft_numba.py`, exposed by the
  `fast` extra).

The numba path is enabled per FFT plan size only after a one-time self-check
that renders random input through **both** kernels (forward and inverse) and
requires them to be bit-identical; on any mismatch — or when numba is not
importable — it silently falls back to NumPy. So:

> **Installing without numba changes speed only. Functionality is complete and
> the output is bit-for-bit identical, by construction of the self-check.**

Practical impact: TD is roughly **2.5× slower without numba** (measured 4.18 s →
11.4 s on `4.wav`; the 239.6 s `2.wav` goes from 41.8 s to ≈108 s, over the 90 s
perf-gate leg). The acceptance/perf harnesses record which path produced a given
measurement (`numba: on/off`), so a timing is never read without knowing the
backend. The speed gate (`py_seconds ≤ max(6 × c_seconds, 90)`) is defined for
the **numba** build; the NumPy fallback is recorded as a reference figure, not a
pass/fail criterion.

---


## Verification

Everything is checked against the C engine; the Python code is never its own
reference.

```bash
tools/run_acceptance.sh              # full 16-case matrix (slow: 2.wav vocoder)
tools/run_acceptance.sh --quick      # 4.wav, both modes, both semitones
tools/run_acceptance.sh --td         # TD cases only
```

Individual harnesses:

| command | what it does |
|---|---|
| `python3 tools/gen_c_refs.py` | renders the C reference corpus into `.tmp/`, writes `refs/c_ref_manifest.json` |
| `python3 tools/acceptance.py` | 16-case matrix → console + `.tmp/acceptance.json` + markdown |
| `python3 tools/selftest_matrix.py` | proves the acceptance gate is not vacuous |
| `python3 tools/perf.py --c-only` | reproducible timing of one case |
| `python3 -m pytest tests/ -q` | unit tests (add `-m "not slow"` for the fast set) |
| `python3 tools/render_readme.py` | regenerates the tables below; `--check` fails if stale |

### Acceptance口径

| level | criterion |
|---|---|
| target | `max\|d\| < 1e-6` or `corr > 0.999999` |
| floor (must hold) | `max\|d\| < 5e-4` **and** `corr > 0.9999` |
| perf | `py_seconds ≤ max(6 × c_seconds, 90)`, numba build (`numba: on`); the NumPy fallback is recorded alongside but is not gated |

Errors are computed in float64 over the common prefix of the two renders, after
both have been read back from f32 WAV — i.e. through the same path a user gets.
A reference that is digital silence leaves `corr` undefined; there the error
magnitude alone decides, so an exact match is never failed by an undefined
correlation.

### Accuracy vs C reference

<!-- ACCEPTANCE:ERROR-TABLE:BEGIN -->
_measured 2026-09-21 13:44 · numba 0.67.0_
| case | mode | semis | len (py/C ref) | max&nbsp;\|d\| | mean&nbsp;\|d\| | RMS err | SNR (dB) | corr | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `1_td3` | td | +3 | 1393118 / 1393118 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `1_td-3` | td | -3 | 1393182 / 1393182 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `1_vc3` | vc | +3 | 1393598 / 1393598 | 8.285e-06 | 5.219e-07 | 8.051e-07 | 113.96 | 1.00000000 | **FAIL-PERF** |
| `1_vc-3` | vc | -3 | 1393598 / 1393598 | 1.725e-03 | 1.491e-06 | 1.827e-05 | 87.12 | 1.00000000 | **PASS(NEG-CLASS)** |
| `2_td3` | td | +3 | 11499345 / 11499345 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `2_td-3` | td | -3 | 11498974 / 11498974 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `2_vc3` | vc | +3 | 11499488 / 11499488 | 4.216e-04 | 3.771e-07 | 3.369e-06 | 99.23 | 1.00000000 | **FAIL-PERF** |
| `2_vc-3` | vc | -3 | 11499488 / 11499488 | 2.230e-02 | 1.907e-05 | 1.669e-04 | 65.66 | 0.99999986 | **FAIL-ACC** |
| `3_td3` | td | +3 | 1745773 / 1745773 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `3_td-3` | td | -3 | 1745630 / 1745630 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `3_vc3` | vc | +3 | 1745920 / 1745920 | 1.254e-04 | 8.535e-08 | 5.824e-07 | 112.82 | 1.00000000 | **FAIL-PERF** |
| `3_vc-3` | vc | -3 | 1745920 / 1745920 | 7.451e-04 | 3.959e-07 | 4.395e-06 | 95.69 | 1.00000000 | **PASS(NEG-CLASS)** |
| `4_td3` | td | +3 | 1019056 / 1019056 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `4_td-3` | td | -3 | 1018734 / 1018734 | 0.000e+00 | 0.000e+00 | 0.000e+00 | inf | 1.00000000 | **TARGET** |
| `4_vc3` | vc | +3 | 1019728 / 1019728 | 2.384e-06 | 8.349e-08 | 1.430e-07 | 122.81 | 1.00000000 | **FAIL-PERF** |
| `4_vc-3` | vc | -3 | 1019728 / 1019728 | 1.089e-05 | 2.528e-07 | 5.587e-07 | 111.48 | 1.00000000 | **FAIL-PERF** |
<!-- ACCEPTANCE:ERROR-TABLE:END -->

Both engines are **duration preserving**: `semis` sets the pitch ratio
`total_ratio = 2^(semis/12)` used to advance the read cursor, it does not
resample the timeline. TD trims a partial granule at the end
(measured: −0.001 % … −0.10 % of the input length), the vocoder drain loop
lands exactly on the input length (measured: 0.000 %).

### Performance

<!-- ACCEPTANCE:PERF-TABLE:BEGIN -->
_measured 2026-09-21 13:44 · numba 0.67.0_
| case | input | py s | C s | speedup | py × realtime | C × realtime | perf gate |
|---|---|---|---|---|---|---|---|
| `1_td3` | 29.0 s | 3.97 | 0.51 | 0.13× | 7.3× | 57.4× | ok |
| `1_td-3` | 29.0 s | 3.40 | 0.50 | 0.15× | 8.5× | 57.6× | ok |
| `1_vc3` | 29.0 s | 310.62 | 22.47 | 0.07× | 0.1× | 1.3× | **limit 134.8s** |
| `1_vc-3` | 29.0 s | 233.32 | 18.70 | 0.08× | 0.1× | 1.6× | **limit 112.2s** |
| `2_td3` | 239.6 s | 20.68 | 4.23 | 0.20× | 11.6× | 56.6× | ok |
| `2_td-3` | 239.6 s | 19.15 | 4.16 | 0.22× | 12.5× | 57.6× | ok |
| `2_vc3` | 239.6 s | 2193.59 | 233.83 | 0.11× | 0.1× | 1.0× | **limit 1403.0s** |
| `2_vc-3` | 239.6 s | 1935.68 | 147.62 | 0.08× | 0.1× | 1.6× | **limit 885.7s** |
| `3_td3` | 39.6 s | 3.50 | 0.67 | 0.19× | 11.3× | 59.4× | ok |
| `3_td-3` | 39.6 s | 3.11 | 0.66 | 0.21× | 12.7× | 60.0× | ok |
| `3_vc3` | 39.6 s | 292.82 | 15.32 | 0.05× | 0.1× | 2.6× | **limit 91.9s** |
| `3_vc-3` | 39.6 s | 256.42 | 13.27 | 0.05× | 0.2× | 3.0× | **limit 90.0s** |
| `4_td3` | 23.1 s | 2.18 | 0.44 | 0.20× | 10.6× | 52.9× | ok |
| `4_td-3` | 23.1 s | 2.13 | 0.40 | 0.19× | 10.8× | 58.0× | ok |
| `4_vc3` | 23.1 s | 158.68 | 9.85 | 0.06× | 0.1× | 2.3× | **limit 90.0s** |
| `4_vc-3` | 23.1 s | 150.92 | 7.32 | 0.05× | 0.2× | 3.2× | **limit 90.0s** |
<!-- ACCEPTANCE:PERF-TABLE:END -->

Python timings are medians of repeated runs after a warm-up; the harness reports
min/max/spread and warns when the machine is too loaded for the comparison to
mean anything (see `tools/perf.py`).

### C baseline

Measured with the reference binaries on the same machine, for context on the
six-fold performance allowance. Regenerate with `python3 tools/gen_c_refs.py`.

<!-- ACCEPTANCE:C-BASELINE:BEGIN -->
_measured 2026-09-21 13:44 · numba 0.67.0_
| case | input | C s | C × realtime | quality / precision |
|---|---|---|---|---|
| `1_td3` | 29.0 s | 0.51 | 57.4× | quality=37 solo=0 |
| `1_td-3` | 29.0 s | 0.50 | 57.6× | quality=37 solo=0 |
| `1_vc3` | 29.0 s | 22.47 | 1.3× | precision=2 |
| `1_vc-3` | 29.0 s | 18.70 | 1.6× | precision=2 |
| `2_td3` | 239.6 s | 4.23 | 56.6× | quality=37 solo=0 |
| `2_td-3` | 239.6 s | 4.16 | 57.6× | quality=37 solo=0 |
| `2_vc3` | 239.6 s | 233.83 | 1.0× | precision=2 |
| `2_vc-3` | 239.6 s | 147.62 | 1.6× | precision=2 |
| `3_td3` | 39.6 s | 0.67 | 59.4× | quality=37 solo=0 |
| `3_td-3` | 39.6 s | 0.66 | 60.0× | quality=37 solo=0 |
| `3_vc3` | 39.6 s | 15.32 | 2.6× | precision=2 |
| `3_vc-3` | 39.6 s | 13.27 | 3.0× | precision=2 |
| `4_td3` | 23.1 s | 0.44 | 52.9× | quality=37 solo=0 |
| `4_td-3` | 23.1 s | 0.40 | 58.0× | quality=37 solo=0 |
| `4_vc3` | 23.1 s | 9.85 | 2.3× | precision=2 |
| `4_vc-3` | 23.1 s | 7.32 | 3.2× | precision=2 |
<!-- ACCEPTANCE:C-BASELINE:END -->

---

## Layout

```
pyradius/
  wavio.py           RIFF/WAVE I/O, bit-exact vs the C reader
  sampler.py         Resampler (InterpTable + drain), bit-exact vs C
  fft.py             radix-2 FFT, bit-exact vs C (fwd + inv); numba/NumPy backends
  _fft_numba.py      optional numba kernel (fast extra), enabled after a bit-exact self-check
  tables.py          FIR/analysis/synthesis window tables parsed from the C headers
  simple_rand.py     LCG, bit-exact vs C
  td_core.py         TD chain                       (TDState, td_render)
  td_fast.py         TD_FAST=1 approximate tier     (vDSP/numba kernels + gating)
  _neon_src/neon_td.c  NEON td_ola (TD_FAST overlap-add, certified bit-exact)
  vocoder_core.py    vocoder chain                  (VocoderState, vc_render)
  cli_td.py          pyr-td CLI
  cli_vc.py          pyr-vc CLI
tools/
  refs.py            canonical reference-corpus naming + manifest
  gen_c_refs.py      renders the C reference corpus
  gen_td_refs.py     renders the TD-only reference corpus into .tmp/refs_td/
  acceptance.py      16-case accuracy/perf matrix
  metrics.py         shared metrics and verdict thresholds
  perf.py            timing methodology for the perf gate
  td_fast_ab.py      TD_FAST A/B matrix (per-kernel corr + timing, subprocesses)
  td_fast_bench.py   clean best-of-N bench: default vs TD_FAST vs C engine
  selftest_matrix.py proves the acceptance gate is sound
  render_readme.py   regenerates the tables above
  verify.py          per-case comparison helper
tests/
  test_wavio.py                I/O bit-exactness vs an independent RIFF parser
  test_reference_corpus.py     reference integrity (size/sha256/±3 distinctness)
  test_acceptance_gate.py      gate thresholds, error and degenerate cases
  test_ops_parity.py           per-operator parity against the C harness corpus         
```

## TD_FAST=1 (approximate high-throughput tier)

`TD_FAST=1` switches `td_render` to `pyradius/td_fast.py`; the default path (flag
unset) is byte-for-byte unchanged and remains bit-exact.  Every component is
gated twice — on `TD_FAST` and on its backend's availability/certification (and
each can be disabled individually with `TD_FAST_<NAME>=0`):

| switch        | what it does                                            | exact? |
|-------------|------------------------------------------------------------|--------|
| `VDSP`      | Accelerate vDSP FFT for the 8192-pt pitch front-end (4/granule) | ULU (~2e-7 rel), no decision change on the corpus |
| `PITCH`     | numba kernels: analysis-window gather, `acf*win_lin` + leading-run suppression, clarity accumulation | yes |
| `TI`        | numba TransientsInfo post-FFT: `pw`/r9/bands/total and `_synth` reductions + stores | yes |
| `INTERP`    | fused drain resampler (phases + band select + wrap + dot) | yes |
| `OLA`       | NEON `td_ola` (`pyradius/_neon_src/neon_td.c`) | yes |
| `DS`        | decimated pitch search — **failed experiment, off by default** | no (corr 0.001-0.08) |

Measured on `audio_test/4.wav` (23.12 s, 44100 Hz, ±3 semitones, M4 Air):

| variant | time | RT | vs default | max abs err | corr |
|---------|------|----|-----------|-------------|------|
| default | 2.16 s | 10.7x | 1.00x | 0 | 1.000000 |
| TI | 2.07 s | 11.2x | 1.09x | 0 | 1.000000 |
| INTERP | 1.92 s | 12.1x | 1.18x | 0 | 1.000000 |
| OLA | 2.06 s | 11.2x | 1.05x | 0 | 1.000000 |
| VDSP | 1.48 s | 15.6x | 1.52x | 0 | 1.000000 |
| VDSP+PITCH | 1.44 s | 16.1x | 1.78x | 0 | 1.000000 |
| all (TD_FAST=1) | 0.79-1.13 s | 20-29x | 2.8-3.3x | **0** | 1.000000 |

`tools/td_fast_ab.py` regenerates the table (it renders each variant in its own
subprocess because the switches are read at import); `tools/td_fast_bench.py`
compares the default path, `TD_FAST=1` and the C engine and prints the load
average, since this box is shared.

### Why the decimated pitch search was rejected

The coarse ACF peak is not the exact peak shifted by ≤ D samples — it is a
different peak (839.8 vs 966.6 on one 4.wav granule).  Decimation is not a
similarity transform of this front-end: `analyze_pitch` whitens by `|X|^-0.95`
with Hz-derived limits (`maxbin`, `taper_len`, `v387`, `v386`) and a noise
floor scaled by `sr/44100`, so at 1/D the sample rate the whitened spectrum —
and therefore which peak wins — changes.  Removing the period's D-quantisation
is necessary but not sufficient: even with a full-rate normalised-correlation
refinement the render decorrelates (0.001-0.08 at D=2/3/4), and the refinement
costs more than the exact search it replaces (it is 4x more expensive than the
whole current FAST pitch path).  A workable lossy variant would have to
cheapen the *full-rate* ACF (e.g. FFT-ACF at reduced precision or a shorter
search range), not the input samples.  The code is kept, documented and
defaulted off.

## Notes on floating point

The ports aim for bit-exactness with the C engine, which is compiled with
`-ffp-contract=off` (no FMA contraction) and uses `libm` `sinf`/`cosf`/`expf`.
Accordingly the Python side uses `math.*` (libm) followed by `np.float32()`
where the C result must match to the last bit, rather than NumPy's SIMD
transcendentals, which can differ by an ULP. Vectorised kernels
(`numpy`/`scipy`) are used for hot loops, and any point where exact repro of the
C rounding is not achievable is called out in a comment at that site.

Three traps that are easy to reintroduce, recorded here so they are not
"cleaned up" later:
1. **A wider-dtype `out=` buffer silently promotes the operation.** `fft.py`
   keeps its butterfly scratch in `float32` on purpose: with a `float64` `out=`
   buffer NumPy would compute the product in double and the result no longer
   reproduces C's per-op float32 rounding.
2. **`np.negative(x, out=fancy_index)` is a silent no-op** (verified
   experimentally). It does not error; the target is simply left unchanged.
3. **`math.exp`/`math.log` are not `expf`/`logf`.** Computing in double and
   casting once differs from the C `float` function on ~1 % of inputs. NumPy's
   float32 `exp`/`log`/`power` *are* bit-identical to libm here (verified against
   `ctypes`-loaded libm over 200 k samples).

The resampler additionally needs the C `fmaf(frac, nsubs, 0.5)` single rounding:
the product is formed in double, rounded once to float32, then truncated. Plain
double arithmetic picks a different sub-phase index on exact-`.5` boundaries,
which shows up as isolated single-sample differences in a TD render.
