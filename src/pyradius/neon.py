"""pyradius.neon — route-B NEON kernel loader (compiled at import time).

Route B (``PYR_FAST=1``) trades bit-exactness for speed everywhere *except*
where a discrete decision depends on it.  The crossover is one such place: the
vocoder's peak detector thresholds on band magnitudes, so a 1e-7 crossover
difference can flip the peak count and decorrelate the whole render (see the
``Crossover`` docstring in :mod:`pyradius.vocoder_ops`).  The kernels here are
therefore required to be **bit-identical** to their numba counterparts, and
:func:`xo_tree_ok` re-checks that property against the live numba kernel before
route B is allowed to use them.

Build model (same disk-cache idea as :class:`pyradius.sampler.InterpTable`):
the C source in ``pyradius/_neon_src/`` is hashed, and the first import compiles
it with ``clang -O3 -mcpu=apple-m4 -shared -fPIC`` into ``.cache/`` under a name
carrying that hash.  Nothing is built or written when the source is unchanged.

Failure policy: **never raise**.  If clang is missing, the platform is not
arm64, the build fails, or ctypes cannot resolve a symbol, every kernel stays
``None`` and route B silently falls back to numba.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "_neon_src")
_CACHE_DIR = os.path.join(os.path.dirname(_HERE), ".cache")

#: xo_tree is selected only after xo_tree_ok() confirms bit-exactness.
xo_tree = None
xo_fast = None
#: data-movement kernels, selected only after data_ok() confirms bit-exactness.
iir16 = None
fg_bandsum = None
oac = None
#: TD_FAST overlap-add kernel, selected only after td_ola_ok().
td_ola = None
#: granule driver, selected only after granule_ok() confirms bit-exactness.
gran_fft_init = None
gran_acs1 = None
gran_acs2 = None
gran_find_peaks = None
gran_pull = None
gran_p2c = None
gran_fold = None
gran_sync2 = None
gran_rpt = None
gran_fill = None
#: formant driver, selected only after formant_ok() confirms bit-exactness.
fm_state_init = None
fm_apply = None
fm_prep = None
fm_env2 = None
fm_peak = None
fm_clip_mirror = None
fm_ker_scale = None
fm_kermix = None
fm_gain2 = None
fm_rms = None
fm_ges_tail = None
fm_iir16 = None
HAVE_NEON = False
BUILD_LOG = ""

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)
_VOID_P = ctypes.c_void_p


class VGFFT(ctypes.Structure):
    """Mirror of ``vg_fft`` in _neon_src/neon_granule.c (vDSP zrip plan)."""
    _fields_ = [("n", ctypes.c_int), ("m", ctypes.c_int),
                ("log2n", ctypes.c_int), ("inv_n", ctypes.c_int),
                ("setup", _VOID_P), ("re", _FLOAT_P), ("im", _FLOAT_P),
                ("cart", _FLOAT_P)]


class VGState(ctypes.Structure):
    """Mirror of ``vg_state`` in _neon_src/neon_granule.c.

    Field order MUST match the C declaration exactly; ctypes reproduces the
    platform ABI layout, so Python can fill the pointers once and then hand
    ``byref(vg)`` to every per-granule entry point.
    """
    _fields_ = [
        ("N", ctypes.c_int), ("nb_bins", ctypes.c_int),
        ("m_fft", ctypes.c_int), ("mb_bins", ctypes.c_int),
        ("n_write", ctypes.c_int), ("hop", ctypes.c_int),
        ("nch", ctypes.c_int), ("nbands", ctypes.c_int),
        ("ring_cap", ctypes.c_int), ("ring_out_len", ctypes.c_int),
        ("sr", ctypes.c_int), ("precision", ctypes.c_int),
        ("buffered_1352", ctypes.c_int),
        ("v8_len_a", ctypes.c_int), ("v9_len_b", ctypes.c_int),
        ("step_base", ctypes.c_float), ("ratio", ctypes.c_float),
        ("fold_alpha", ctypes.c_float), ("fold_c", ctypes.c_float),
        ("fwd", VGFFT), ("inv", VGFFT),
        ("win0", _FLOAT_P), ("synth_a", _FLOAT_P), ("synth_b", _FLOAT_P),
        ("edge_gain", _FLOAT_P),
        ("acc", _FLOAT_P), ("cart", _FLOAT_P),
        ("mag", _FLOAT_P), ("mask", _FLOAT_P), ("acs_env", _FLOAT_P),
        ("region_gain", _FLOAT_P), ("phase", _FLOAT_P),
        ("mag_copy", _FLOAT_P), ("mask_copy", _FLOAT_P),
        ("peak_bins", _INT_P), ("reg_start", _INT_P), ("reg_end", _INT_P),
        ("reg_offset", _FLOAT_P), ("peak_count", ctypes.c_int),
        ("scratch_8a8", _FLOAT_P), ("win1", _FLOAT_P), ("win2", _FLOAT_P),
        ("out_ring", _FLOAT_P), ("sync_weight", _FLOAT_P),
    ]


def _fp(arr):
    """Raw float pointer to a C-contiguous f32 numpy array."""
    return arr.ctypes.data_as(_FLOAT_P)


def _ip(arr):
    """Raw int pointer to a C-contiguous int32 numpy array."""
    return arr.ctypes.data_as(_INT_P)


def _source_files():
    out = []
    try:
        for name in sorted(os.listdir(_SRC_DIR)):
            if name.endswith((".c", ".h")):
                out.append(os.path.join(_SRC_DIR, name))
    except OSError:
        pass
    return out


def _src_hash(paths):
    h = hashlib.sha256()
    h.update(platform.machine().encode())
    h.update(sys.platform.encode())
    for p in paths:
        try:
            with open(p, "rb") as f:
                h.update(os.path.basename(p).encode())
                h.update(f.read())
        except OSError:
            return None
    return h.hexdigest()[:16]


def _cache_path(tag):
    return os.path.join(_CACHE_DIR, f"neon_{tag}.dylib")


def _build(paths, tag):
    """Compile the sources to .cache/; returns the .dylib path or None."""
    so = _cache_path(tag)
    if os.path.exists(so):
        return so
    if platform.machine() != "arm64" or sys.platform != "darwin":
        return None
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = so + f".{os.getpid()}.tmp"
    # -mcpu=apple-m4 targets the host ISA directly; -march=arm64e is rejected by
    # clang on macOS and must not be used.
    # -ffp-contract=off is REQUIRED: numba's LLVM emits *separate* fmul/fadd for
    # the f64 recurrences in iir16/oac, while clang's default (fast) contraction
    # fuses them into an fma and changes the rounding (verified: iir16 matched
    # only with contract=off).  Explicit fmaf/vfmaq intrinsics in neon_xover.c
    # are unaffected by this flag.
    # -framework Accelerate: neon_granule.c calls vDSP directly (setup, split
    # buffers and the FFT structure live in C), so a granule costs one ctypes
    # call instead of one per vDSP op.  Linking the framework here is what the
    # "vDSP in C, not through ctypes" requirement asks for.
    cmd = ["clang", "-O3", "-mcpu=apple-m4", "-shared", "-fPIC",
           "-fno-math-errno", "-ffp-contract=off", "-o", tmp,
           "-framework", "Accelerate"] + list(paths)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not os.path.exists(tmp):
        global BUILD_LOG
        BUILD_LOG = (r.stderr or b"").decode("utf-8", "replace")[-2000:]
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    os.replace(tmp, so)
    return so


_I = ctypes.c_int
_D = ctypes.c_double


class FMState(ctypes.Structure):
    """Mirror of ``fm_state`` in _neon_src/neon_formant.c.

    Field order MUST match the C declaration exactly (ctypes reproduces the
    platform ABI layout).  The pointers are filled once from buffers that are
    real attributes of FormantState and never reallocated.
    """
    _fields_ = [("NB", _I), ("N", _I), ("M", _I), ("MB", _I),
                ("nb_bands", _I), ("prec_mode", _I), ("ges_stride", _I),
                ("sr", ctypes.c_float), ("inv_scale_f", ctypes.c_float),
                ("db", _FLOAT_P), ("gscr", _FLOAT_P), ("ker", _FLOAT_P),
                ("env", _FLOAT_P), ("gain_env", _FLOAT_P),
                ("ffr", _FLOAT_P), ("ffi", _FLOAT_P),
                ("rms_scr", ctypes.POINTER(ctypes.c_double)),
                ("h8_pad", _FLOAT_P),
                ("fft_m", ctypes.c_byte * 64), ("fft_h8", ctypes.c_byte * 64),
                ("h8_ready", _I)]


def _bind_formant(lib):
    """Resolve the formant-driver entry points (separate ABI version)."""
    global fm_state_init, fm_apply, fm_prep, fm_env2, fm_peak, fm_clip_mirror
    global fm_ker_scale, fm_kermix, fm_gain2, fm_rms, fm_ges_tail, fm_iir16
    try:
        if lib.neon_formant_abi() != 1:
            return
    except Exception:
        return
    try:
        lib.fm_state_init.restype = _I
        lib.fm_state_init.argtypes = [ctypes.POINTER(FMState), _I, _I, _I, _I,
                                     _I, _I, _I, ctypes.c_float,
                                     ctypes.c_float, _FLOAT_P, _FLOAT_P,
                                     _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
                                     _FLOAT_P]
        fm_state_init = lib.fm_state_init
        lib.fm_apply.restype = _I
        lib.fm_apply.argtypes = [ctypes.POINTER(FMState), _FLOAT_P, _I, _D, _D,
                                _D, _D, _D, _I, _I, _I, _I, _I]
        fm_apply = lib.fm_apply
        lib.fm_prep.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _I, _I]
        fm_prep = lib.fm_prep
        lib.fm_env2.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _I, _D]
        fm_env2 = lib.fm_env2
        lib.fm_peak.argtypes = [_FLOAT_P, _I, _D, _D, _D, _I, _D,
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_int)]
        fm_peak = lib.fm_peak
        lib.fm_clip_mirror.argtypes = [_FLOAT_P, _FLOAT_P, _I, _I, _I]
        fm_clip_mirror = lib.fm_clip_mirror
        lib.fm_ker_scale.argtypes = [_FLOAT_P, _FLOAT_P, _I, _D]
        fm_ker_scale = lib.fm_ker_scale
        lib.fm_kermix.argtypes = [_FLOAT_P, _FLOAT_P, _I, _I, _I]
        fm_kermix = lib.fm_kermix
        lib.fm_gain2.argtypes = [_FLOAT_P, _FLOAT_P, _I, _D, _D]
        fm_gain2 = lib.fm_gain2
        lib.fm_rms.argtypes = [_FLOAT_P, _FLOAT_P, _I, _D,
                              ctypes.POINTER(ctypes.c_double)]
        fm_rms = lib.fm_rms
        lib.fm_ges_tail.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _I, _D, _D,
                                   _I, _D]
        fm_ges_tail = lib.fm_ges_tail
        lib.fm_iir16.argtypes = [_FLOAT_P, _I, _D, _D]
        fm_iir16 = lib.fm_iir16
    except AttributeError:
        fm_state_init = fm_apply = fm_prep = fm_env2 = fm_peak = None
        fm_clip_mirror = fm_ker_scale = fm_kermix = None
        fm_gain2 = fm_rms = fm_ges_tail = fm_iir16 = None


def _bind_granule(lib):
    """Resolve the granule-driver entry points (separate ABI version)."""
    global gran_fft_init, gran_acs1, gran_acs2, gran_find_peaks, gran_pull
    global gran_p2c, gran_fold, gran_sync2, gran_rpt, gran_fill
    try:
        if lib.neon_granule_abi() != 1:
            return
    except Exception:
        return
    try:
        lib.vg_fft_init.argtypes = [ctypes.POINTER(VGFFT), ctypes.c_int,
                                    _FLOAT_P]
        lib.vg_fft_init.restype = None
        gran_fft_init = lib.vg_fft_init
        lib.vg_acs1.argtypes = [ctypes.POINTER(VGState), ctypes.c_int,
                                ctypes.c_int, ctypes.c_int]
        lib.vg_acs1.restype = None
        gran_acs1 = lib.vg_acs1
        lib.vg_acs2.argtypes = [ctypes.POINTER(VGState), ctypes.c_int, _FLOAT_P]
        lib.vg_acs2.restype = None
        gran_acs2 = lib.vg_acs2
        lib.vg_find_peaks.argtypes = [ctypes.POINTER(VGState), ctypes.c_int]
        lib.vg_find_peaks.restype = ctypes.c_int
        gran_find_peaks = lib.vg_find_peaks
        lib.vg_pull_to_peak.argtypes = [ctypes.POINTER(VGState), ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int]
        lib.vg_pull_to_peak.restype = None
        gran_pull = lib.vg_pull_to_peak
        lib.vg_p2c.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, ctypes.c_int]
        lib.vg_p2c.restype = None
        gran_p2c = lib.vg_p2c
        lib.vg_fold_iir_sub.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.c_float, ctypes.c_double]
        lib.vg_fold_iir_sub.restype = None
        gran_fold = lib.vg_fold_iir_sub
        lib.vg_sync_nch2.argtypes = [ctypes.POINTER(VGState), ctypes.c_float]
        lib.vg_sync_nch2.restype = None
        gran_sync2 = lib.vg_sync_nch2
        lib.vg_rpt.argtypes = [ctypes.POINTER(VGState), ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.vg_rpt.restype = None
        gran_rpt = lib.vg_rpt
        lib.vg_fill_granule.argtypes = [ctypes.POINTER(VGState), ctypes.c_int,
                                        ctypes.c_int, _FLOAT_P, _FLOAT_P]
        lib.vg_fill_granule.restype = None
        gran_fill = lib.vg_fill_granule
    except AttributeError:
        gran_fft_init = gran_acs1 = gran_acs2 = None
        gran_find_peaks = gran_pull = gran_p2c = None
        gran_fold = gran_sync2 = gran_rpt = gran_fill = None


#: granule-driver bindings resolved lazily (needs the vg_fft/vg_state mirrors).
gran_fft_run = None
gran_fft_inv_run = None


def _bind_granule_fft(lib):
    global gran_fft_run, gran_fft_inv_run
    try:
        lib.vg_fft_fwd_run.argtypes = [ctypes.POINTER(VGFFT), _FLOAT_P, _FLOAT_P]
        lib.vg_fft_fwd_run.restype = None
        gran_fft_run = lib.vg_fft_fwd_run
        lib.vg_fft_inv_run.argtypes = [ctypes.POINTER(VGFFT), _FLOAT_P, _FLOAT_P]
        lib.vg_fft_inv_run.restype = None
        gran_fft_inv_run = lib.vg_fft_inv_run
    except AttributeError:
        gran_fft_run = gran_fft_inv_run = None


def _bind(lib):
    """Resolve kernels with explicit argtypes (mandatory — see vdsp.py)."""
    global xo_tree, xo_fast, iir16, fg_bandsum, oac, td_ola, HAVE_NEON
    try:
        if lib.neon_kernels_abi() != 1:
            return
    except Exception:
        return
    _bind_granule(lib)
    _bind_granule_fft(lib)
    _bind_formant(lib)
    try:
        lib.xo_tree.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _I, _I, _I]
        lib.xo_tree.restype = None
        xo_tree = lib.xo_tree
        lib.xo_fast.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _I, _I, _I]
        lib.xo_fast.restype = None
        xo_fast = lib.xo_fast
        # iir16: a2/a1m are python floats (f64) in the numba signature and the
        # caller passes f32-rounded values, so c_double reproduces them exactly.
        lib.iir16.argtypes = [_FLOAT_P, _I, _D, _D]
        lib.iir16.restype = None
        iir16 = lib.iir16
        lib.fg_bandsum.argtypes = [_FLOAT_P, _FLOAT_P, _I, _I, _I, _I]
        lib.fg_bandsum.restype = None
        fg_bandsum = lib.fg_bandsum
        # oac: v197/v198 are python floats in the numba signature (f64), and the
        # TwoSum correction is only exact against that full-precision value.
        lib.oac.argtypes = [_FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
                            _I, _I, _I, _I, _I, _I, _I, _I, _D, _D]
        lib.oac.restype = None
        oac = lib.oac
        # td_ola (TD_FAST): env_flat is f32, env_off/env_span are int32 tables.
        lib.td_ola.argtypes = [_FLOAT_P, _FLOAT_P, _I, _I, _I, _I, _I, _I, _I,
                              _I, _I, _I, _I, _I, ctypes.c_float, _FLOAT_P,
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int), _I]
        lib.td_ola.restype = None
        td_ola = lib.td_ola
        HAVE_NEON = True
    except AttributeError:
        xo_tree = xo_fast = None
        iir16 = fg_bandsum = oac = None
        td_ola = None
        HAVE_NEON = False


def _load():
    try:
        paths = _source_files()
        if not paths:
            return
        tag = _src_hash(paths)
        if tag is None:
            return
        so = _build(paths, tag)
        if not so:
            return
        _bind(ctypes.CDLL(so))
    except Exception:
        # Any failure means "no NEON": route B falls back to numba.
        pass


_load()


# ---------------------------------------------------------------------------
# Granule driver (route B): per-stage entry points + safety gate
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------
_CERT = None


def _certify():
    """Check xo_tree against _xover_zp_nb on adversarial inputs; cache the result.

    Tests a range of n (including n < 4 so the scalar tail is exercised) and
    N, plus a deliberately extreme input set, and requires max|d| == 0.
    """
    global _CERT
    if _CERT is not None:
        return _CERT
    if xo_tree is None:
        _CERT = False
        return _CERT
    try:
        from . import vocoder_ops as ops
        from .tables import get_tables
        nb_ref = getattr(ops, "_xover_zp_nb", None)
        if nb_ref is None:
            _CERT = False
            return _CERT
        N = int(np.asarray(get_tables(44100)["fir"]).shape[1])
        tr = np.ascontiguousarray(
            np.asarray(get_tables(44100)["fir"], dtype=np.float32)[:, ::-1])
        rng = np.random.default_rng(0xC0FFEE)
        for n in (1, 2, 3, 4, 5, 7, 64, 448, 1024):
            for scale in (0.0, 1e-3, 0.5, 1e3):
                z = (rng.standard_normal(N - 1 + n) * scale).astype(np.float32)
                if not z.flags.c_contiguous:
                    z = np.ascontiguousarray(z)
                ref = np.empty((4, n), dtype=np.float32)
                nb_ref(z, tr, ref, n, N, 4)
                got = np.empty((4, n), dtype=np.float32)
                xo_tree(z.ctypes.data_as(_FLOAT_P),
                        tr.ctypes.data_as(_FLOAT_P),
                        got.ctypes.data_as(_FLOAT_P), n, N, 4)
                if not np.array_equal(got.view(np.uint32), ref.view(np.uint32)):
                    _CERT = False
                    return _CERT
        _CERT = True
    except Exception:
        _CERT = False
    return _CERT


def xo_tree_ok() -> bool:
    """True iff xo_tree is loaded and verified bit-identical to numba."""
    return _certify()


def xo_tree_run(z, taps_rev, out, n, N, nb):
    """Call the C xo_tree on numpy arrays (all must be C-contiguous f32)."""
    xo_tree(z.ctypes.data_as(_FLOAT_P), taps_rev.ctypes.data_as(_FLOAT_P),
             out.ctypes.data_as(_FLOAT_P), int(n), int(N), int(nb))


# ---------------------------------------------------------------------------
# Data-movement kernels (iir16 / fg_bandsum / oac)
# ---------------------------------------------------------------------------
_DATA_CERT = None


def _certify_data():
    """Check iir16/fg_bandsum/oac against their numba kernels; cache the result.

    Every kernel must be bit-identical (uint32 view equality), across a range of
    sizes (including the n<2 and single-row tails each kernel has) and, for oac,
    all four branch combinations: wrap x (fma row / scale row).
    """
    global _DATA_CERT
    if _DATA_CERT is not None:
        return _DATA_CERT
    _DATA_CERT = False
    if iir16 is None or fg_bandsum is None or oac is None:
        return _DATA_CERT
    try:
        from . import vocoder_ops as ops
        rng = np.random.default_rng(0xDA7A)
        same = lambda a, b: np.array_equal(a.view(np.uint32), b.view(np.uint32))

        # ---- iir16 ----
        nb_ref = getattr(ops, "_iir16_nb", None)
        if nb_ref is None:
            return _DATA_CERT
        for a2f in (0.058, 0.474, 0.9, 0.9999, 0.5):
            a2 = float(np.float32(a2f))
            a1m = float(np.float32(1.0 - np.float32(a2f)))
            for n in (1, 2, 3, 7, 64, 448, 1024, 8193):
                x = (rng.standard_normal(n) * 0.5).astype(np.float32)
                ref = x.copy()
                nb_ref(ref, a2, a1m)
                got = x.copy()
                iir16(got.ctypes.data_as(_FLOAT_P), n, a2, a1m)
                if not same(got, ref):
                    return _DATA_CERT

        # ---- fg_bandsum ----
        nb_ref = getattr(ops, "_fg_bandsum_nb", None)
        if nb_ref is None:
            return _DATA_CERT
        for (N, nb, cap) in ((4, 4, 256), (1, 1, 8), (7, 3, 16),
                             (1024, 4, 7000), (64, 4, 65)):
            ring = rng.standard_normal((nb, cap)).astype(np.float32)
            for off in (0, 1, cap - 1, cap // 2, max(cap - N, 0)):
                ref = np.zeros(N, dtype=np.float32)
                nb_ref(ring, ref, off, N, cap, nb)
                got = np.zeros(N, dtype=np.float32)
                fg_bandsum(ring.ctypes.data_as(_FLOAT_P),
                           got.ctypes.data_as(_FLOAT_P), off, N, cap, nb)
                if not same(got, ref):
                    return _DATA_CERT

        # ---- oac ----
        nb_ref = getattr(ops, "_oac_nb", None)
        if nb_ref is None:
            return _DATA_CERT
        R, hs = 64, 4
        for N in (1, 3, hs, 16, R, 2 * R):
            for pos0 in (0, 1, R - 1, R, -2):
                for cursor in (0, 3, hs, hs + 1, 40):
                    for v35, v36 in ((N, N), (N // 2, N // 2),
                                     (N, 0), (N // 4, 3 * N // 4)):
                        w1 = rng.standard_normal(N + 8).astype(np.float32)
                        w2 = rng.standard_normal(N + 8).astype(np.float32)
                        sa = rng.standard_normal(N).astype(np.float32)
                        sb = rng.standard_normal(N).astype(np.float32)
                        out0 = rng.standard_normal(R).astype(np.float32)
                        v197, v198 = 0.7125, 1.0
                        ref = out0.copy()
                        nb_ref(ref, w1, w2, sa, sb, R, N, pos0, v35, v36,
                               hs, cursor, hs, v197, v198)
                        got = out0.copy()
                        oac(got.ctypes.data_as(_FLOAT_P),
                            w1.ctypes.data_as(_FLOAT_P),
                            w2.ctypes.data_as(_FLOAT_P),
                            sa.ctypes.data_as(_FLOAT_P),
                            sb.ctypes.data_as(_FLOAT_P),
                            R, N, pos0, v35, v36, hs, cursor, hs, v197, v198)
                        if not same(got, ref):
                            return _DATA_CERT
        _DATA_CERT = True
    except Exception:
        _DATA_CERT = False
    return _DATA_CERT


def data_ok() -> bool:
    """True iff iir16/fg_bandsum/oac are loaded and verified bit-identical."""
    return _certify_data()


def iir16_run(y, a2, a1m):
    """In-place 8x IIR on a C-contiguous f32 array (scalars are python floats)."""
    iir16(y.ctypes.data_as(_FLOAT_P), int(y.size), float(a2), float(a1m))


def fg_bandsum_run(ring, acc, off, N, cap, nb):
    """Band sum with ring wraparound (both arrays C-contiguous f32)."""
    fg_bandsum(ring.ctypes.data_as(_FLOAT_P), acc.ctypes.data_as(_FLOAT_P),
               int(off), int(N), int(cap), int(nb))


def oac_run(out, win1, win2, synth_a, synth_b, R, N, pos0, v35, v36, wh,
            cursor, hop, v197, v198):
    """Fused overlap-add ring write (all arrays C-contiguous f32)."""
    oac(out.ctypes.data_as(_FLOAT_P), win1.ctypes.data_as(_FLOAT_P),
        win2.ctypes.data_as(_FLOAT_P), synth_a.ctypes.data_as(_FLOAT_P),
        synth_b.ctypes.data_as(_FLOAT_P), int(R), int(N), int(pos0),
        int(v35), int(v36), int(wh), int(cursor), int(hop),
        float(v197), float(v198))


# ---------------------------------------------------------------------------
# TD_FAST overlap-add kernel (td_ola)
# ---------------------------------------------------------------------------
_TD_CERT = None


def _certify_td():
    """Check td_ola against td_core._do_ola_exact over a wide parameter grid.

    Requires uint32 equality of the touched out-ring region and covers: both
    branch shapes (boolean 0/1 -> one vs two env segments), w10 > 0 (negative
    ring base), the segment-2 tail, crossfade with a non-empty destination, and
    nch 1/2.  The ring is filled with non-trivial values so the "not yet
    written" state of the out ring is exercised too (a batch that a batched
    numpy version would get wrong if it reordered the two segments).

    The out ring is allocated exactly like td_render does it — ``[nch,
    out_len + 16]``, i.e. a row stride LARGER than ``out_len`` — because a
    kernel that assumes stride == out_len is bit-exact on a densely packed
    array and silently wrong in the render (this bug was live once: corr
    0.578 with the kernel "certified").
    """
    global _TD_CERT
    if _TD_CERT is not None:
        return _TD_CERT
    _TD_CERT = False
    if td_ola is None:
        return _TD_CERT
    try:
        import numpy as _np

        from .td_core import TDState, _do_ola_exact, RX_TD_NTAB
        rng = _np.random.default_rng(0x0DDBA11)
        same = lambda a, b: _np.array_equal(a.view(_np.uint32), b.view(_np.uint32))
        st = TDState(48000, 37, 0, 2)
        st.ring_tot = 1 << 12
        st.ring_start = 0
        st.out_len = 1 << 13
        for a8 in (0, 1, 3, 4):
            for boolean in (0, 1):
                for a9 in (1, 128, 1000):
                    for a10 in (0, a9, a9 + 300):
                        for a4 in (0, 777, 4000):
                            for wr in (0, 12345):
                                for nch in (1, 2):
                                    st.nch = nch
                                    rcap = st.ring_tot
                                    ring = [
                                        (rng.standard_normal(rcap) * 0.3).astype(_np.float32)
                                        for _ in range(nch)]
                                    o0 = (rng.standard_normal((nch, st.out_len + 16)) * 0.3).astype(_np.float32)
                                    o1 = o0.copy()
                                    st.fed_visible = int(rng.integers(0, rcap))
                                    _do_ola_exact(st, ring, o0, a4, wr, a8, a9, a10,
                                                  boolean, 1.0)
                                    _td_ola_call(st, ring, o1, a4, wr, a8, a9, a10,
                                                 boolean, 1.0)
                                    if not same(o0, o1):
                                        return _TD_CERT
        _TD_CERT = True
    except Exception:
        _TD_CERT = False
    return _TD_CERT


def _td_ola_call(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain):
    """Call the C td_ola with the TDState env tables (int32 views)."""
    import numpy as _np

    from .td_core import RX_TD_NTAB, RX_TD_NBLK
    nch = st.nch
    ring2 = _np.ascontiguousarray(_np.asarray(ring, dtype=_np.float32))
    flat, off, span = _env_flat(st)
    # out_ring is allocated [nch, out_len + 16] in td_render, so the kernel needs
    # its real row stride (numpy's out[c] gets it implicitly; ctypes does not).
    ostride = int(out_ring.strides[0] // out_ring.itemsize) if out_ring.ndim == 2 \
        else int(st.out_len)
    td_ola(ring2.ctypes.data_as(_FLOAT_P), out_ring.ctypes.data_as(_FLOAT_P),
           int(nch), int(st.ring_tot), int(st.ring_start), int(st.out_len),
           ostride, int(st.fed_visible), int(a4), int(wr), int(a8), int(a9),
           int(a10), int(boolean), float(gain), flat.ctypes.data_as(_FLOAT_P),
           off.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           span.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), int(RX_TD_NTAB))


_ENV_CACHE: dict = {}


def _env_flat(st):
    """Flatten the TDState env tables into (env_flat, env_off, env_span).

    Cached per state object — the tables are built once in __init__ and never
    mutated, and the flattening is O(#entries) only on the first call.
    """
    import numpy as _np

    key = id(st)
    got = _ENV_CACHE.get(key)
    if got is not None and got[3] is st:
        return got[0], got[1], got[2]
    from .td_core import RX_TD_NTAB, RX_TD_NBLK
    spans = _np.ascontiguousarray(st.env_span, dtype=_np.int32).ravel()
    offs = _np.zeros(RX_TD_NBLK * RX_TD_NTAB, dtype=_np.int32)
    tabs = []
    n = 0
    for a8 in range(RX_TD_NBLK):
        for j in range(RX_TD_NTAB):
            offs[a8 * RX_TD_NTAB + j] = n
            tabs.append(st.env_tab[a8][j])
            n += st.env_tab[a8][j].shape[0]
    flat = _np.ascontiguousarray(_np.concatenate(tabs), dtype=_np.float32)
    _ENV_CACHE.clear()
    _ENV_CACHE[key] = (flat, offs, spans, st)
    return flat, offs, spans


def td_ola_ok() -> bool:
    return _certify_td()


def td_ola_run(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain):
    """In-place TD overlap-add on 2-D ring / out_ring (see td_fast.do_ola_fast)."""
    _td_ola_call(st, ring, out_ring, a4, wr, a8, a9, a10, boolean, gain)


_GRAN_CERT = None


def _certify_granule():
    """Bit-exactness self-check for every migrated granule stage.

    Certifies under the same route-B semantics as the render (the reference
    chain dispatches on _route_b() for its fft/kernels — see the matching
    comment on _certify_formant for the import-order trap this avoids).
    """
    global _GRAN_CERT
    if _GRAN_CERT is not None:
        return _GRAN_CERT
    _GRAN_CERT = False
    from . import vocoder_ops as _vo
    _saved = _vo._ROUTE_B_CACHE
    _vo._ROUTE_B_CACHE = True
    try:
        _certify_granule_inner()
        _GRAN_CERT = True
    except Exception:
        _GRAN_CERT = False
    finally:
        _vo._ROUTE_B_CACHE = _saved
    return _GRAN_CERT


def _certify_granule_inner():
    """Bit-exactness self-check for every migrated granule stage.

    Runs each C entry point against the route-B Python/numba path it replaces on
    identical adversarial inputs and requires exact uint32 equality (the
    standalone harness of the same checks is .tmp/selfcheck_granule.py).

    This is the gate: if anything fails, the C stage stays unused and route B
    keeps its numba/numpy implementation, so a partially-built extension can
    never change the render.  It builds its own private carrier with
    VocoderState._cg_build rather than _cg_attach, so it cannot recurse.
    """
    global _GRAN_CERT
    if _GRAN_CERT is not None:
        return _GRAN_CERT
    _GRAN_CERT = False
    if None in (gran_acs1, gran_acs2, gran_find_peaks, gran_pull, gran_p2c,
                gran_fold, gran_sync2, gran_rpt, gran_fill, gran_fft_run,
                gran_fft_inv_run):
        return
    try:
        from . import vocoder_core as _vc
        from . import vdsp_fast as _vf
        from .vocoder_core import (_find_peaks_nb, _pull_nb, _p2c_nb, _fold_nb,
                                   _FOLD_ALPHA)
        from .vocoder_ops import (cart_to_polar, fill_granule,
                                  reset_phases_for_transients,
                                  synchronize_stereo_phases_nch2_fast)
        rng = np.random.default_rng(0x5EED)
        same = lambda a, b: np.array_equal(  # noqa: E731
            np.ascontiguousarray(a).view(np.uint32),
            np.ascontiguousarray(b).view(np.uint32))
        # the route-B clamp constant, written as its exact f32 bit pattern
        mag_floor = np.frombuffer(np.uint32(0x2B8CBCCC).tobytes(),
                                  dtype="<f4")[0]

        for sr in (44100, 48000):
            st = _vc.VocoderState(sr, 2, 2)
            N, NB = st.N, st.nb_bins
            if not st._cg_build():
                return _GRAN_CERT
            vg, ptr = st._cg_vg, st._cg_ptr

            # ---- stage a: acs1 (window + FFT + envelope IIR) ----
            for tstate, step in ((0, 0), (0, st.hop), (2, st.hop), (1, 222)):
                acc0 = (rng.standard_normal(N) * 0.3).astype(np.float32)
                env0 = (rng.standard_normal(NB) * 0.05).astype(np.float32)
                st.acc[:] = acc0
                st.acs_env[0][:] = env0
                gran_acs1(ptr, 0, step, tstate)
                ref = _vc.VocoderState(sr, 2, 2)
                ref.acc[:] = acc0
                ref.acs_env[0][:] = env0
                ref.transient_state_2488 = tstate
                ref.step_1412 = step
                cart_ref = ref._ev_acs_front(0)
                # _ev_acs_front returns its FFT destination (_vf_cart on the vDSP
                # path, ref.cart under EXACT_FFT) — not necessarily ref.cart.
                if not (same(st.cart, cart_ref)
                        and same(st.acs_env[0], ref.acs_env[0])
                        and same(st.acc, ref.acc)):
                    return _GRAN_CERT

            # ---- stage b: acs2 (cart_to_polar + lower clamp) ----
            cart2 = (rng.standard_normal(N + 2) * 0.3).astype(np.float32)
            # gran_acs2 transforms the array its pointer argument names, which in
            # the renderer is st.cart; put the data there and reference the same.
            st.cart[:] = cart2
            st.mag[:] = 0.0
            st.mask[:] = 0.0
            gran_acs2(ptr, 0, _fp(st.cart))
            ctp = cart_to_polar(cart2)
            refmag = np.maximum(ctp["mag"][:NB], mag_floor)
            if not (same(st.mag[0][:NB], refmag)
                    and same(st.mask[0][:NB], ctp["phase"][:NB])):
                return _GRAN_CERT

            # ---- stage c: find_peaks ----
            cnt = 0
            for trial in range(3):
                if trial == 1:
                    st.mag[0][:] = np.float32(0.5)      # cnt == 0 fallback
                else:
                    st.mag[0][:] = (rng.random(NB).astype(np.float32) * 0.2
                                    + 0.01)
                    if trial == 0:
                        st.mag[0][100:110] = 5.0
                        st.mag[0][500] = 9.0
                st.acs_env[0][:] = (rng.random(NB).astype(np.float32) + 0.001)
                # the cnt == 0 fallback returns before touching region_gain, so
                # both sides must start from the same (zeroed) buffer
                st.region_gain[0][:] = 0.0
                ref = _vc.VocoderState(sr, 2, 2)
                ref.mag[0][:] = st.mag[0]
                ref.acs_env[0][:] = st.acs_env[0]
                ref.region_gain[0][:] = 0.0
                pk = np.zeros(NB, np.int32)
                ro = np.zeros(NB, np.float32)
                rs = np.zeros(NB, np.int32)
                re_ = np.zeros(NB, np.int32)
                rg = np.zeros(NB, np.float32)
                cnt = _find_peaks_nb(ref.mag[0], ref.acs_env[0], pk, ro, rs,
                                     re_, rg, NB)
                if gran_find_peaks(ptr, 0) != cnt:
                    return _GRAN_CERT
                for a, b in ((st.peak_bins[:cnt], pk[:cnt]),
                             (st.reg_offset[:cnt], ro[:cnt]),
                             (st.reg_start[:cnt], rs[:cnt]),
                             (st.reg_end[:cnt], re_[:cnt])):
                    if not same(a, b):
                        return _GRAN_CERT
                s0, e0 = int(st.reg_start[0]), int(st.reg_end[cnt - 1])
                if not same(st.region_gain[0][s0:e0], rg[s0:e0]):
                    return _GRAN_CERT
            st.peak_count = cnt

            # ---- stage d: pull_to_peak ----
            for pg, sp in ((st.hop, st.hop), (st.hop, st.hop + 28),
                           (2048, 3592), (0, 100)):
                phase0 = (rng.standard_normal((2, NB)) * 3.0).astype(np.float32)
                mask0 = (rng.standard_normal((2, NB)) * 3.0).astype(np.float32)
                rg0 = rng.random((2, NB)).astype(np.float32) * 8.0
                st.phase[:2] = phase0
                st.mask[:2] = mask0
                st.region_gain[:2] = rg0
                ref = _vc.VocoderState(sr, 2, 2)
                ref.phase[:2] = phase0
                ref.mask[:2] = mask0
                ref.region_gain[:2] = rg0
                ref.peak_count = st.peak_count
                ref.peak_bins[:] = st.peak_bins
                ref.reg_start[:] = st.reg_start
                ref.reg_end[:] = st.reg_end
                ref.prev_granule_1408 = pg
                ref.step_1412 = sp
                ref._ev_pull_to_peak(0)
                gran_pull(ptr, 0, pg, sp)
                if not same(st.phase[0], ref.phase[0]):
                    return _GRAN_CERT

            # ---- stage e: p2c ----
            mag0 = (np.abs(rng.standard_normal(NB)) * 0.1).astype(np.float32)
            ph0 = (rng.standard_normal(NB) * 4.0).astype(np.float32)
            ref_cart = np.zeros(N + 2, np.float32)
            _p2c_nb(mag0, ph0, ref_cart, NB)
            cart_c = np.zeros(N + 2, np.float32)
            gran_p2c(_fp(mag0), _fp(ph0), _fp(cart_c), NB)
            if not same(cart_c, ref_cart):
                return _GRAN_CERT

            # ---- stage f: fold_iir_sub ----
            for _ in range(2):
                a = (rng.standard_normal(N) * 0.3).astype(np.float32)
                w1r = np.zeros(N, np.float32)
                w2r = np.zeros(N, np.float32)
                _fold_nb(a, w1r, w2r, N, st.hop, _FOLD_ALPHA,
                         float(np.float32(1.0) - _FOLD_ALPHA))
                w1c = np.zeros(N, np.float32)
                w2c = np.zeros(N, np.float32)
                gran_fold(_fp(a), _fp(w1c), _fp(w2c), N, st.hop,
                          float(_FOLD_ALPHA),
                          float(np.float32(1.0) - _FOLD_ALPHA))
                if not (same(w1c, w1r) and same(w2c, w2r)):
                    return _GRAN_CERT

            # ---- stage g: sync nch2 ----
            for _ in range(2):
                m0 = (rng.random((2, NB)).astype(np.float32) + 0.001)
                s0 = (rng.standard_normal((2, NB)) * 3.0).astype(np.float32)
                d0 = (rng.standard_normal((2, NB)) * 3.0).astype(np.float32)
                st.mag[:2] = m0
                st.mask[:2] = s0
                st.phase[:2] = d0
                st.sync_sens_3496 = np.float32(0.0)
                out = synchronize_stereo_phases_nch2_fast(
                    m0, s0, d0, peaks=st.peak_bins[:st.peak_count],
                    pk_start=st.reg_start[:st.peak_count],
                    pk_end=st.reg_end[:st.peak_count],
                    peak_count=st.peak_count, sens=np.float32(0.0))
                gran_sync2(ptr, 0.0)
                if not (same(st.phase[:2], out["dst"])
                        and same(st.sync_weight_3504, out["weight"])):
                    return _GRAN_CERT

            # ---- stage h: rpt (ResetPhasesForTransients) ----
            for flag in (0, 1):
                mask0 = (rng.standard_normal((2, NB)) * 2.0).astype(np.float32)
                ph0 = (rng.standard_normal((2, NB)) * 2.0).astype(np.float32)
                st.mask[:2] = mask0
                st.phase[:2] = ph0
                ref = _vc.VocoderState(sr, 2, 2)
                ref.mask[:2] = mask0
                ref.phase[:2] = ph0
                ref.transient_flag_1208 = flag
                ref.transient_state_2488 = 0
                ref.ratio = st.ratio
                for ch in range(2):
                    o = reset_phases_for_transients(
                        mode_4b8=flag, proc_mode_9b8=0, scale_20=ref.ratio,
                        len_594=NB, n9bc=0, n9c0=0, u_70=ref.sr,
                        n578=ref.granule_1400, n590=ref.hop_1420, n5d0=N,
                        n5d4=NB, avg_m1=0.0, avg_0=0.0, avg_p1=0.0,
                        use_p568=False, mag=ref.mag[ch], b_710=ref.b_710,
                        mask=ref.mask[ch], r_start=ref.reg_start,
                        r_end=ref.reg_end, r_bin=ref.peak_bins,
                        phase=ref.phase[ch], mask_table=ref.mask_table_8f8,
                        n_out=NB)
                    ref.phase[ch][:] = o["phase"]
                gran_rpt(ptr, 0, flag, 0, float(st.ratio))
                gran_rpt(ptr, 1, flag, 0, float(st.ratio))
                if not same(st.phase[:2], ref.phase[:2]):
                    return _GRAN_CERT

            # ---- stage j: fill_granule / fgwin_x4 ----
            # The reference is the route-B fill_granule dispatcher itself, which
            # covers the buffered / direct / clamped branches plus the ring wrap,
            # driven on a non-zero ring so the band sum is not trivially zero.
            st.ring[:] = (rng.standard_normal(st.ring.shape) * 0.1).astype(np.float32)
            for cursor in (0, st.hop // 2, st.hop, 3 * st.hop, st.ring_cap // 2,
                           st.ring_cap - st.hop, st.ring_cap,
                           st.ring_cap + 2 * st.hop, 2 * st.ring_cap + 7):
                for mode, kw in ((0, {"mode": "fg"}), (1, {"mode": "fgw"})):
                    ref = np.zeros(st.N, np.float32)
                    fill_granule(
                        st.ring, st.win_table, ref, hop=st.hop,
                        n_write=st.n_write, n_bands=st.nbands, n5d0=st.N,
                        cursor=cursor,
                        # _ev_fgwin_x4 passes buffered=True unconditionally;
                        # _ev_fill_granule only when cursor >= hop
                        buffered=True if mode else (
                            bool(st.buffered_1352) if cursor >= st.hop else False),
                        cap_scalar=st.ring_cap, ring_cap0_a=st.ring_cap // 2,
                        ring_cap0_b=st.ring_cap // 2, gain=1.0, **kw)
                    # mode 1 appends to acc (the renderer zeroes it once per
                    # granule), so the reference is reset the same way
                    st.acc[:] = 0.0
                    gran_fill(ptr, mode, cursor, _fp(st.ring), _fp(st.win_table))
                    if not same(st.acc[:st.N], ref):
                        return _GRAN_CERT

            # ---- stage i: FFT plans (fwd + inv) vs vdsp_fast ----
            src = (rng.standard_normal(N) * 0.3).astype(np.float32)
            c_fwd = np.zeros(N + 2, np.float32)
            r_fwd = np.zeros(N + 2, np.float32)
            gran_fft_run(ctypes.byref(vg.fwd), _fp(src), _fp(c_fwd))
            _vf.fwd_r(src, r_fwd, N)
            cin = (rng.standard_normal(N + 2) * 0.3).astype(np.float32)
            c_out = np.zeros(N, np.float32)
            r_out = np.zeros(N, np.float32)
            gran_fft_inv_run(ctypes.byref(vg.inv), _fp(cin), _fp(c_out))
            _vf.inv_r(cin, r_out, N)
            if not (same(c_fwd, r_fwd) and same(c_out, r_out)):
                return
        return True
    except Exception:
        return False


def granule_ok() -> bool:
    """True iff the granule driver stages are loaded and verified bit-exact."""
    return _certify_granule()


# ---------------------------------------------------------------------------
# Formant driver (route B): fm_apply + its stage entry points
# ---------------------------------------------------------------------------
_FM_CERT = None


def _certify_formant():
    """Bit-exactness self-check for the formant C port.

    Certifies against ``FormantState._apply_numba`` *as route B runs it*: the
    reference chain's ``FormantState._fft`` dispatches to vDSP ``fft_zip`` only
    under route B, and the C kernel embeds the vDSP plans — so without this
    override a non-route-B import order (any module importing vocoder_ops
    before PYR_FAST is seen, e.g. pytest running TestACS ahead of TestFormantC)
    would compare the C kernel against the numpy radix-2 fallback, fail on
    ffi/env, cache ``False``, and silently pin route B to numba for the whole
    process.  The certification itself is import-order-independent: it forces
    the same fft backend the C kernel assumes.
    """
    global _FM_CERT
    if _FM_CERT is not None:
        return _FM_CERT
    _FM_CERT = False
    if None in (fm_state_init, fm_apply):
        return _FM_CERT
    from . import vocoder_ops as _vo
    _saved_cache = _vo._ROUTE_B_CACHE
    _vo._ROUTE_B_CACHE = True
    try:
        if _certify_formant_inner():
            _FM_CERT = True
    except Exception:
        pass
    finally:
        _vo._ROUTE_B_CACHE = _saved_cache
    return _FM_CERT


def _certify_formant_inner():
    """Body of _certify_formant: runs under _ROUTE_B_CACHE=True.

    Runs ``fm_apply`` against the numba-only chain (FormantState._apply_numba)
    it replaces, on inputs that exercise every branch: the peak-search window
    (best==0 and best>0), the RMS normalisation, the sticky-d gain clamp, the
    hermitean mirror, the kermix crossfade, both prec_mode time constants, and
    both bands of the shared persistent ``env``.  Requires uint32 equality of
    mag AND of all persistent/scratch buffers (db, ker, gscr, env,
    gain_env[band]) — that shared state is what makes this operator
    order-sensitive.

    This is the gate: if anything fails, route B keeps the numba chain, so a
    partially-built extension can never change the render.  The standalone
    harness for the same property (against a real render's traced data) is
    .tmp/fm_apply_pin.py.
    """
    from .vocoder_ops import FormantState
    rng = np.random.default_rng(0xF0A17)
    for sr, geom in _FM_GEOM.items():
        for prec in (2, 1):
            for ratio in (0.8408964276313782, 1.189207115002721,
                          0.5946035575013605):
                for band in (0, 1):
                    for trial in range(4):
                        if not _fm_one_trial(FormantState, rng, sr, geom,
                                             prec, ratio, band, trial):
                            return False
    return True


def formant_ok() -> bool:
    """True iff the formant C kernel is loaded and certified bit-exact."""
    return _certify_formant()


_FM_GEOM = {44100: (8192, 4097, 2048, 1025),
            48000: (16384, 8193, 4096, 2049)}


def _fm_make_state(FormantState, sr, geom, prec):
    n, nb, m, mb = geom
    return FormantState(dict(active=1, mode_freq=0, mode_rms=1, ratio=1.0,
                             width=1.0, strength=1.0, freq_lo=40.0,
                             freq_hi=800.0, nb_bins=nb, n_fft=n, m_fft=m,
                             m_bins=mb, f580=314, sr=float(sr),
                             prec_mode=prec, nb_bands=2))


def _fm_one_trial(FormantState, rng, sr, geom, prec, ratio, band, trial):
    nb = geom[1]
    ref = _fm_make_state(FormantState, sr, geom, prec)
    got = _fm_make_state(FormantState, sr, geom, prec)
    # give the two states identical non-trivial persistent content so the
    # shared env / per-band gain-envelope rows are really compared (the arrays
    # must be drawn ONCE and copied, or the two states start out different and
    # the comparison is meaningless)
    ges1 = (rng.standard_normal(ref.gain_env.shape[1]) * 0.1 + 0.9).astype(np.float32)
    env0 = (rng.random(ref.env.size) * 0.05).astype(np.float32)
    db0 = (rng.standard_normal(ref.db.size) * 20).astype(np.float32)
    ker0 = (rng.standard_normal(ref.ker.size) * 8).astype(np.float32)
    for st in (ref, got):
        st.gain_env[1] = ges1
        st.env[:] = env0
        st.db[:] = db0
        st.ker[:] = ker0
    m = rng.random(nb).astype(np.float64)
    if trial % 2 == 0:
        m = np.full(nb, 0.3)
    else:
        m = m * 10.0 ** rng.integers(-12, 3, nb).astype(np.float64)
    m[rng.integers(0, nb, 40)] = 0.0
    mag = m.astype(np.float32)

    cfg = dict(ratio=np.float32(ratio),
               f580=int(rng.integers(1, 3000)),
               mode_freq=int(rng.integers(0, 2)), mode_rms=1)
    ref.cfg.update(cfg)
    got.cfg.update(cfg)

    m1 = np.array(mag, np.float32, copy=True)
    out_ref = FormantState._apply_numba(ref, m1, band)

    car = FMState()
    if fm_state_init(ctypes.byref(car), got.nb_bins, got.n_fft, got.m_fft,
                     got.m_bins, got.nb_bands, got.prec_mode,
                     int(got.gain_env.strides[0] // 4), float(got.sr),
                     float(got.inv_scale), _fp(got.db), _fp(got.gscr),
                     _fp(got.ker), _fp(got.env), _fp(got.gain_env),
                     _fp(got.ffr), _fp(got.ffi)) != 0:
        return False
    m2 = np.array(mag, np.float32, copy=True)
    if fm_apply(ctypes.byref(car), _fp(m2), band, float(cfg["ratio"]), 1.0, 1.0,
                800.0, 40.0, cfg["mode_freq"], 1, cfg["f580"], 1, 0) != 0:
        return False
    pairs = [(out_ref, m2), (ref.db, got.db), (ref.ker, got.ker),
             (ref.gscr, got.gscr), (ref.env, got.env),
             (ref.gain_env[band], got.gain_env[band])]
    return all(np.array_equal(np.ascontiguousarray(a).view(np.uint32),
                              np.ascontiguousarray(b)[:np.size(a)].view(np.uint32))
               for a, b in pairs)


def formant_ok() -> bool:
    """True iff the formant C port is loaded and verified bit-exact."""
    return _certify_formant()


def formant_run(st, mag, band) -> bool:
    """Run the C ApplyFormantCorrection in place; False if unavailable."""
    car = getattr(st, "_fm_car", None)
    if car is None or fm_apply is None:
        return False
    c = st.cfg
    return fm_apply(ctypes.byref(car), _fp(mag), int(band),
                    float(c["ratio"]), float(c["strength"]),
                    float(c["width"]), float(c["freq_hi"]),
                    float(c["freq_lo"]), int(c["mode_freq"]),
                    int(c["mode_rms"]), int(c["f580"]), 1,
                    int(os.environ.get("PYR_FM_H8", "0") or 0)) == 0
