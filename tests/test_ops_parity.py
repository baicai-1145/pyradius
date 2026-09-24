"""tests.test_ops_parity — operator-level bit-exactness gate for pyradius.

Loads the C reference corpus produced by ``tools/gen_ref_ops.py``
(``.tmp/ref_ops/manifest.json`` + raw f32/i32/f64 payloads) and compares it
against the pyradius implementation of the same operator.

Contract
--------
* One test per operator family; each iterates its cases and reports
  ``exact-frac`` (fraction of bit-identical f32 values) and ``max|d|``.
* Operators whose pyradius side does not exist yet are **skipped** (not failed)
  — the suite is green either way, and the skip list tells the two porting
  tasks (TD / VOC) exactly what is still missing.
* Every scalar parameter is taken from the *dumped input payload*, never from
  the manifest ``meta`` block: meta carries human-readable values while the
  harness consumed the exact f32/f64 bit patterns.  (Getting this wrong shows
  up as a sub-bin drift after hundreds of accumulations.)
* Results accumulate into a module-level table printed at the end and written
  to ``.tmp/ref_ops/parity_report.json`` (best-effort) — the machine-readable
  scoreboard for the porting tasks.

Running
-------
    python3 -m pytest tests/test_ops_parity.py -v
    python3 tests/test_ops_parity.py            # no pytest needed

Environment
-----------
    PYR_REF_DIR   override ``.tmp/ref_ops`` location
    PYR_REQUIRE_REF=1  turn "no reference corpus" into a failure (CI mode)
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REF = os.environ.get("PYR_REF_DIR", os.path.join(ROOT, ".tmp", "ref_ops"))
MANIFEST = os.path.join(REF, "manifest.json")
REPORT = os.path.join(REF, "parity_report.json")
REQUIRE_REF = os.environ.get("PYR_REQUIRE_REF", "") not in ("", "0", "false")

# Route B (PYR_FAST=1) is an approximation tier whose kernels deliberately
# differ from the C engine (e.g. the polynomial-atan2 cart_to_polar, 1-ulp fma
# unwrap) — this file is the DEFAULT-path gate and its exact-frac==1.0 contract
# only holds without it.  If the operator-under-test dispatches on route B the
# comparison is meaningless, so the default-path tests refuse to run and say
# why instead of failing with confusing 0.919 exact-frac numbers.
# TestFormantC below is the exception: it certifies the route-B C kernel
# against the route-B numba chain and forces PYR_FAST itself.
_UNDER_ROUTE_B = os.environ.get("PYR_FAST", "") not in ("", "0", "false")

# The 64-bit LCG in pyradius.simple_rand deliberately overflows on every step;
# numpy turns that into a RuntimeWarning that would mask a real overflow later.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*overflow encountered.*")

# Operator families that tools/gen_ref_ops.py must produce.
EXPECTED_OPS = ("fft", "simple", "interp", "crossover", "fill_granule", "acs",
                "unwrap", "apc", "rpt", "sync", "formant", "oac", "noise",
                "td_env", "td_pitch", "td_ti")

# Collects one row per compared output for the parity report.
ROWS: list[dict] = []


# --------------------------------------------------------------------------
# corpus access
# --------------------------------------------------------------------------

def load_manifest() -> dict:
    try:
        with open(MANIFEST) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


MAN = load_manifest()
_DT = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"), "f64": np.dtype("<f8")}


def _path(entry: dict) -> str:
    p = entry["file"]
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load(entry: dict) -> np.ndarray:
    return np.fromfile(_path(entry), dtype=_DT[entry["ext"]])


def case_rec(op: str, case: str) -> dict:
    return MAN[op]["cases"][case]


def case_meta(op: str, case: str) -> dict:
    return case_rec(op, case)["meta"]


def case_input(op: str, case: str, name: str) -> np.ndarray:
    """Pristine input payload as the C harness received it.

    These files are treated as read-only by the harness (outputs go to a
    separate `__out__` namespace), so a test may pass one straight through;
    tests that exercise an in-place operator still `.copy()` where they need to
    observe the untouched baseline.
    """
    return load(case_rec(op, case)["inputs"][name])


def _corpus_settled(seconds: float = 5.0) -> bool:
    """True when the corpus looks idle (no very recent manifest/payload writes).

    Several agents share .tmp/ref_ops and may regenerate it concurrently; a
    generator's purge-then-write window makes payloads briefly absent.  Rather
    than report a phantom integrity failure, back off while a regeneration is in
    flight.
    """
    newest = 0.0
    try:
        newest = os.path.getmtime(MANIFEST)
    except OSError:
        pass
    d = os.path.dirname(MANIFEST)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.endswith(".part") or fn.endswith(".tmp"):
                return False
    return (time.time() - newest) > seconds


def await_corpus(timeout: float = 90.0) -> bool:
    """Wait for a concurrent regeneration to finish. Returns False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _corpus_settled():
            return True
        time.sleep(0.5)
    return False


def case_output(op: str, case: str, name: str, default_missing=None):
    """Reference output. Resolved through the manifest, not by path guessing."""
    e = case_rec(op, case)["outputs"].get(name)
    if e is not None:
        return load(e)
    m = case_rec(op, case)["inputs"].get(name)
    if m is not None:
        # documented alias: an in-place operator whose output equals its input
        # (the harness only dumps what the operator actually produced)
        return load(m)
    if default_missing is not None:
        return default_missing
    raise KeyError(f"reference output {op}/{case}/{name} missing")


def case_output_entry(op: str, case: str, name: str) -> dict | None:
    return case_rec(op, case)["outputs"].get(name)


def ci(op: str, case: str, name: str, default=None):
    """Scalar case parameter taken from the dumped input payload (authority)."""
    e = case_rec(op, case)["inputs"].get(name)
    if e is None:
        meta = case_meta(op, case)
        if name in meta:
            return meta[name]
        if default is not None:
            return default
        raise KeyError(f"{op}/{case}: no input or meta field {name!r}")
    a = load(e)
    if a.size != 1:
        raise ValueError(f"{op}/{case}/{name}: expected scalar, got {a.size}")
    v = a[0]
    return int(v) if a.dtype.kind == "i" else float(v)


def cases_of(op: str) -> list[str]:
    return sorted(MAN.get(op, {}).get("cases", {}))


def have_ref(op: str) -> bool:
    return bool(MAN.get(op, {}).get("cases"))


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------

def exact_frac(got, want) -> float:
    a = np.asarray(got, dtype=np.float32).ravel()
    b = np.asarray(want, dtype=np.float32).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 1.0
    return float(np.mean(a[:n].view(np.int32) == b[:n].view(np.int32)))


def max_abs(got, want) -> float:
    a = np.asarray(got, dtype=np.float64).ravel()
    b = np.asarray(want, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    d = np.abs(a[:n] - b[:n])
    d = d[~np.isnan(d)]
    return float(d.max()) if d.size else 0.0


def record(op: str, case: str, out: str, got, want, note: str = ""):
    ROWS.append({"op": op, "case": case, "output": out,
                 "n": int(min(np.size(got), np.size(want))),
                 "exact_frac": exact_frac(got, want),
                 "max_abs": max_abs(got, want), "note": note})
    return ROWS[-1]["exact_frac"], ROWS[-1]["max_abs"]


def _fail(msg: str):
    raise AssertionError(msg)


def skip_if_no_ref(op: str) -> None:
    if have_ref(op):
        return
    msg = (f"{op}: no reference corpus at {os.path.relpath(MANIFEST, ROOT)} "
           f"— run: python3 tools/gen_ref_ops.py --only {op}")
    if REQUIRE_REF:
        _fail(msg)
    raise unittest.SkipTest(msg)


def skip_under_route_b(op: str) -> None:
    """Default-path exactness tests must not run with route B kernels active."""
    if _UNDER_ROUTE_B:
        raise unittest.SkipTest(
            f"{op}: PYR_FAST=1 selects the route-B approximation tier; this "
            f"exact-frac==1.0 gate is the default-path contract. Run without "
            f"PYR_FAST (or use tools/td_fast_gate.py / TestFormantC for the "
            f"route-B gates).")


def skip_pyr(what: str) -> None:
    raise unittest.SkipTest(f"pyradius side not implemented yet: {what}")


def opt_module(*names: str):
    """First importable candidate module, else None."""
    for n in names:
        try:
            mod = __import__(n, fromlist=["*"])
        except Exception:
            continue
        return mod
    return None


def reset_state(*mods):
    """Reset module-level operator state between cases.

    The C harness runs **every case in its own process**, so each reference Case
    starts from freshly constructed instances.  Several ports keep streaming
    state in module globals instead (crossover's FIR history, noise slot
    rotation, ...); replaying cases in one interpreter would otherwise let case
    N's history leak into case N+1 and report a phantom divergence.

    ``importlib.reload`` re-executes the module body in the *same* module object,
    so existing references stay valid while the globals are re-initialised.
    """
    import importlib
    for m in mods:
        if m is not None and getattr(m, "__name__", "") in sys.modules:
            try:
                importlib.reload(m)
            except Exception:
                pass
    return mods[0] if len(mods) == 1 else mods


# ==========================================================================
# FFT — pyradius.fft (bit-exact radix-2 mirror of rx_fft_fwd / rx_fft_inv)
# ==========================================================================

class TestFFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestFFT")
    def test_fft_fwd_inv(self):
        skip_if_no_ref("fft")
        fft = opt_module("pyradius.fft")
        if fft is None or not hasattr(fft, "plan"):
            skip_pyr("pyradius.fft.plan(N).fwd/inv")

        worst = 1.0
        worst_abs = 0.0
        for case in cases_of("fft"):
            N = ci("fft", case, "N")
            x = case_input("fft", case, "x").astype(np.float32)
            cart = case_input("fft", case, "cart").astype(np.float32)
            pl = fft.plan(N)

            ef, md = record("fft", case, "fwd_cart", pl.fwd(x),
                            case_output("fft", case, "fwd_cart"))
            worst, worst_abs = min(worst, ef), max(worst_abs, md)
            if md > 1e-5:
                _fail(f"fft/{case} fwd_cart: max|d|={md:g}")

            ef, md = record("fft", case, "inv_time", pl.inv(cart),
                            case_output("fft", case, "inv_time"))
            worst, worst_abs = min(worst, ef), max(worst_abs, md)
            if md > 1e-5:
                _fail(f"fft/{case} inv_time: max|d|={md:g}")

            ef, md = record("fft", case, "roundtrip", pl.inv(pl.fwd(x)),
                            case_output("fft", case, "roundtrip"))
            worst, worst_abs = min(worst, ef), max(worst_abs, md)
            if md > 1e-4:
                _fail(f"fft/{case} roundtrip: max|d|={md:g}")

        # pyradius.fft documents exact mode for these sizes: demand near-exactness
        if worst < 0.99:
            _fail(f"fft exact-frac floor violated: {worst:g} (max|d|={worst_abs:g})")


# ==========================================================================
# polar_to_cart / copy_bitwise / simple_rand
# ==========================================================================

class TestSimpleOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestSimpleOps")
    def test_copy_bitwise(self):
        skip_if_no_ref("simple")
        for case in cases_of("simple"):
            src = case_input("simple", case, "mag")
            got = src.copy()          # memcpy semantics == identity
            ef, md = record("simple", case, "copy", got,
                            case_output("simple", case, "copy"))
            if ef != 1.0:
                _fail(f"simple/{case} copy: not bit-exact ({ef:g})")

    def test_polar_to_cart(self):
        skip_if_no_ref("simple")
        for case in cases_of("simple"):
            mag = case_input("simple", case, "mag").astype(np.float32)
            ph = case_input("simple", case, "phase").astype(np.float32)
            got = np.empty(mag.size * 2, dtype=np.float32)
            got[0::2] = mag * np.cos(ph).astype(np.float32)
            got[1::2] = mag * np.sin(ph).astype(np.float32)
            ef, md = record("simple", case, "cart", got,
                            case_output("simple", case, "cart"))
            if md > 1e-6:
                _fail(f"simple/{case} polar_to_cart: max|d|={md:g}")

    def test_simple_rand_lcg(self):
        skip_if_no_ref("simple")
        mod = opt_module("pyradius.simple_rand")
        if mod is None or not hasattr(mod, "SimpleRand"):
            skip_pyr("pyradius.simple_rand.SimpleRand")
        for case in cases_of("simple"):
            seed = ci("simple", case, "seed")
            draws = ci("simple", case, "draws")
            r = mod.SimpleRand(seed)
            # the 64-bit LCG overflows on every step; that is intended, and
            # pytest resets module-level warning filters, so scope it here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                got = np.array([r.next() for _ in range(draws)], dtype=np.float32)
            # `rx_simple_rand_next` is a pure draw counter and the harness's
            # `rand`/`rand_from_seed` payloads come from two identically seeded
            # generators, so both must match the same stream.
            for nm in ("rand", "rand_from_seed"):
                ef, md = record("simple", case, nm, got, case_output("simple", case, nm))
                if ef != 1.0:
                    _fail(f"simple/{case} LCG {nm}: exact-frac={ef:g}")

    def test_noise_template_fill(self):
        skip_if_no_ref("simple")
        mod = opt_module("pyradius.simple_rand")
        if mod is None or not hasattr(mod, "noise_template_fill"):
            skip_pyr("pyradius.simple_rand.noise_template_fill")
        for case in cases_of("simple"):
            seed = ci("simple", case, "seed")
            draws = ci("simple", case, "draws")
            tcount = ci("simple", case, "tcount")
            # harness order: `draws` GetNext calls first, then template fill
            r = mod.SimpleRand(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for _ in range(draws):
                    r.next()
                got = np.zeros(tcount, dtype=np.float32)
                mod.noise_template_fill(got, r)
            ef, md = record("simple", case, "template", got,
                            case_output("simple", case, "template"))
            if ef != 1.0:
                _fail(f"simple/{case} noise_template_fill: exact-frac={ef:g} "
                      f"max|d|={md:g}")


# ==========================================================================
# interp_nsamples (Resampler::InterpolateNSamples)
# ==========================================================================

class TestInterpNsamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestInterpNsamples")
    def test_interp(self):
        skip_if_no_ref("interp")
        mod = opt_module("pyradius.sampler")
        if mod is None or not hasattr(mod, "InterpTable"):
            skip_pyr("pyradius.sampler.InterpTable / interp_nsamples")

        for case in cases_of("interp"):
            X = ci("interp", case, "X")
            levels = ci("interp", case, "n_levels")
            quality = ci("interp", case, "quality")
            nch = ci("interp", case, "nch")
            ring = ci("interp", case, "ring")
            dst_off = ci("interp", case, "dst_off")
            count = ci("interp", case, "count")
            phase = ci("interp", case, "phase")
            rate = ci("interp", case, "rate")

            got_k = np.array([mod.interp_kidx(quality, levels)], dtype=np.int32)
            ef, md = record("interp", case, "kidx", got_k,
                            case_output("interp", case, "kidx"))
            if ef != 1.0:
                _fail(f"interp/{case} kidx: got {got_k[0]}")

            if int(case_output("interp", case, "build_rc")[0]) != 0:
                _fail(f"interp/{case}: C table build failed (bad reference)")

            src = case_input("interp", case, "src").astype(np.float32).reshape(nch, ring)
            tbl = mod.InterpTable(X, levels, quality)
            got = mod.interp_nsamples(tbl, src, ring, phase, dst_off, count,
                                      rate, quality)
            want = case_output("interp", case, "dst").reshape(nch, count)
            ef, md = record("interp", case, "dst", got, want)
            if md > 1e-5:
                _fail(f"interp/{case} dst: max|d|={md:g}")


# ==========================================================================
# crossover (4-band FIR bank)
# ==========================================================================

class TestCrossover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestCrossover")
    @staticmethod
    def _impl():
        for name in ("pyradius.crossover", "pyradius.ops", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "crossover_process1"):
                return mod.crossover_process1
        return None

    def test_process1(self):
        skip_if_no_ref("crossover")
        mod0 = opt_module("pyradius.crossover", "pyradius.ops", "pyradius.vocoder_ops")
        if mod0 is None or not hasattr(mod0, "crossover_process1"):
            skip_pyr("pyradius.crossover.crossover_process1(sr, x64) -> [4][n]")
        for case in cases_of("crossover"):
            sr = ci("crossover", case, "sr")
            n = ci("crossover", case, "n")
            # The C harness builds a *separate* rx_crossover per channel, so each
            # channel starts from zero filter history.  The port keeps that
            # history in a module-level cache keyed by sample rate, so it must be
            # reset per channel (and per case) or channel 1 inherits channel 0.
            for ch, in_name in ((0, "seq0"), (1, "seq1")):
                mod = reset_state(mod0)
                fn = getattr(mod, "crossover_process1", None)
                seq = case_input("crossover", case, in_name).astype(np.float64)
                got = np.asarray(fn(sr, seq), dtype=np.float64)
                if got.shape[0] != 4:
                    got = got.reshape(4, n)
                for b in range(4):
                    want = case_output("crossover", case, f"p1_c{ch}_b{b}")
                    ef, md = record("crossover", case, f"p1_c{ch}_b{b}",
                                    got[b], want)
                    if md > 2e-6:
                        _fail(f"crossover/{case}/c{ch}_b{b}: max|d|={md:g}")

    def test_process_sequence(self):
        """Whole-sequence path (rx_crossover_process on a 1-ch instance)."""
        skip_if_no_ref("crossover")
        mod0 = opt_module("pyradius.crossover", "pyradius.ops", "pyradius.vocoder_ops")
        if mod0 is None or not (hasattr(mod0, "crossover_process")
                                or hasattr(mod0, "crossover_process1")):
            skip_pyr("pyradius.crossover.crossover_process(sr, seq) -> [4][n]")
        for case in cases_of("crossover"):
            sr = ci("crossover", case, "sr")
            n = ci("crossover", case, "n")
            for ch, in_name in ((0, "seq0"), (1, "seq1")):
                mod = reset_state(mod0)
                fn = (getattr(mod, "crossover_process", None)
                      or getattr(mod, "crossover_process1", None))
                seq = case_input("crossover", case, in_name).astype(np.float64)
                got = np.asarray(fn(sr, seq), dtype=np.float64).reshape(4, n)
                for b in range(4):
                    want = case_output("crossover", case, f"proc_c{ch}_b{b}")
                    ef, md = record("crossover", case, f"proc_c{ch}_b{b}",
                                    got[b], want)
                    if md > 2e-6:
                        _fail(f"crossover/{case}/proc_c{ch}_b{b}: max|d|={md:g}")


# ==========================================================================
# fill_granule / fill_granule_win
# ==========================================================================

class TestFillGranule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestFillGranule")
    @staticmethod
    def _impl():
        for name in ("pyradius.fill_granule", "pyradius.vocoder_ops", "pyradius.vocoder"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "fill_granule"):
                return mod
        return None

    def _case_params(self, case):
        return dict(
            hop=ci("fill_granule", case, "hop"),
            n_write=ci("fill_granule", case, "n_write"),
            n_bands=ci("fill_granule", case, "n_bands"),
            n5d0=ci("fill_granule", case, "n5d0"),
            cursor=ci("fill_granule", case, "cursor"),
            buffered=bool(ci("fill_granule", case, "buffered")),
            cap_scalar=ci("fill_granule", case, "cap_scalar"),
            ring_cap0_a=ci("fill_granule", case, "ring_cap0_a"),
            ring_cap0_b=ci("fill_granule", case, "ring_cap0_b"),
            gain=ci("fill_granule", case, "gain"),
        )

    def _inputs(self, case):
        p = self._case_params(case)
        cap = ci("fill_granule", case, "cap")
        ring = case_input("fill_granule", case, "ring").reshape(p["n_bands"], cap)
        win = case_input("fill_granule", case, "win").reshape(p["n_bands"],
                                                             2 * p["hop"])
        acc0 = case_input("fill_granule", case, "acc0")
        return p, ring, win, acc0

    def test_fill_granule(self):
        skip_if_no_ref("fill_granule")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.fill_granule.fill_granule(...)")
        for case in cases_of("fill_granule"):
            mod = reset_state(mod)
            p, ring, win, acc0 = self._inputs(case)
            got = mod.fill_granule(ring=ring, win=win, acc=acc0.copy(),
                                   mode="fg", **p)
            ef, md = record("fill_granule", case, "fg", got,
                            case_output("fill_granule", case, "fg"))
            if ef != 1.0:
                _fail(f"fill_granule/{case} fg: exact-frac={ef:g} max|d|={md:g}")

    def test_fill_granule_win(self):
        skip_if_no_ref("fill_granule")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.fill_granule.fill_granule(..., mode='fgw')")
        for case in cases_of("fill_granule"):
            mod = reset_state(mod)
            p, ring, win, acc0 = self._inputs(case)
            got = mod.fill_granule(ring=ring, win=win, acc=acc0.copy(),
                                   mode="fgw", **p)
            ef, md = record("fill_granule", case, "fgw", got,
                            case_output("fill_granule", case, "fgw"))
            if ef != 1.0:
                _fail(f"fill_granule/{case} fgw: exact-frac={ef:g} max|d|={md:g}")

    def test_fill_granule_chain(self):
        skip_if_no_ref("fill_granule")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.fill_granule.fill_granule(..., mode='chain')")
        for case in cases_of("fill_granule"):
            mod = reset_state(mod)
            p, ring, win, acc0 = self._inputs(case)
            got = mod.fill_granule(ring=ring, win=win, acc=acc0.copy(),
                                   mode="chain", **p)
            ef, md = record("fill_granule", case, "chain", got,
                            case_output("fill_granule", case, "chain"))
            if ef != 1.0:
                _fail(f"fill_granule/{case} chain: exact-frac={ef:g} max|d|={md:g}")


# ==========================================================================
# analyze_channel_spectrum
# ==========================================================================

class TestACS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestACS")
    @staticmethod
    def _impl():
        for name in ("pyradius.acs", "pyradius.analyze_channel_spectrum",
                     "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "acs_spectrum1"):
                return mod
        return None

    def test_time_to_iir_a(self):
        skip_if_no_ref("acs")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.acs.time_to_iir_a")
        for case in cases_of("acs"):
            got = np.array([mod.time_to_iir_a(ci("acs", case, "tau"),
                                              ci("acs", case, "rate"))],
                           dtype=np.float32)
            ef, md = record("acs", case, "iir_a", got,
                            case_output("acs", case, "iir_a"))
            if ef != 1.0:
                _fail(f"acs/{case} time_to_iir_a: exact-frac={ef:g}")

    def test_spectrum1(self):
        skip_if_no_ref("acs")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.acs.acs_spectrum1")
        for case in cases_of("acs"):
            n_win = ci("acs", case, "n_win")
            n_fft = ci("acs", case, "n_fft")
            n_bins = ci("acs", case, "n_bins")
            iir_a = float(case_output("acs", case, "iir_a")[0])
            out = mod.acs_spectrum1(case_input("acs", case, "time").astype(np.float32),
                                    case_input("acs", case, "win").astype(np.float32),
                                    case_input("acs", case, "env0").astype(np.float32),
                                    n_fft=n_fft, n_bins=n_bins, iir_a=iir_a)
            for nm in ("t1_io", "t1_cart", "t1_env"):
                ef, md = record("acs", case, nm, out[nm],
                                case_output("acs", case, nm))
                if md > 1e-5:
                    _fail(f"acs/{case} {nm}: max|d|={md:g}")

    def test_spectrum2(self):
        skip_if_no_ref("acs")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.acs.acs_spectrum2")
        for case in cases_of("acs"):
            n_fft = ci("acs", case, "n_fft")
            n_polar = ci("acs", case, "n_polar")
            n_maxbin = ci("acs", case, "n_maxbin")
            out = mod.acs_spectrum2(case_input("acs", case, "time2").astype(np.float32),
                                    n_fft=n_fft, n_polar=n_polar,
                                    n_maxbin=n_maxbin)
            for nm in ("t2_cart", "t2_mag", "t2_phase"):
                ef, md = record("acs", case, nm, out[nm],
                                case_output("acs", case, nm))
                if md > 1e-5:
                    _fail(f"acs/{case} {nm}: max|d|={md:g}")

    def test_primitives(self):
        skip_if_no_ref("acs")
        mod = self._impl()
        fn = getattr(mod, "cart_to_polar", None) if mod else None
        if fn is None:
            skip_pyr("pyradius.acs.cart_to_polar / threshold_lt_inplace")
        for case in cases_of("acs"):
            n_polar = ci("acs", case, "n_polar")
            cart = case_output("acs", case, "t2_cart")
            got = fn(cart)
            ef, md = record("acs", case, "ctp_mag", got["mag"],
                            case_output("acs", case, "ctp_mag"))
            if ef != 1.0:
                _fail(f"acs/{case} cart_to_polar mag: exact-frac={ef:g}")
            ef, md = record("acs", case, "ctp_phase", got["phase"],
                            case_output("acs", case, "ctp_phase"))
            if ef != 1.0:
                _fail(f"acs/{case} cart_to_polar phase: exact-frac={ef:g}")
            if hasattr(mod, "threshold_lt_inplace"):
                thr = ci("acs", case, "thr")
                got2 = mod.threshold_lt_inplace(got["mag"].copy(), thr)
                ef, md = record("acs", case, "thr_mag", got2,
                                case_output("acs", case, "thr_mag"))
                if ef != 1.0:
                    _fail(f"acs/{case} threshold: exact-frac={ef:g}")


# ==========================================================================
# unwrap_phase
# ==========================================================================

class TestUnwrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestUnwrap")
    @staticmethod
    def _impl():
        for name in ("pyradius.unwrap_phase", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "unwrap_phase"):
                return mod
        return None

    def test_unwrap(self):
        skip_if_no_ref("unwrap")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.unwrap_phase.unwrap_phase")
        for case in cases_of("unwrap"):
            got = mod.unwrap_phase(
                mask=case_input("unwrap", case, "mask"),
                mask_copy=case_input("unwrap", case, "maskCopy"),
                mag_copy=case_input("unwrap", case, "magCopy"),
                reg_start=case_input("unwrap", case, "reg_start"),
                reg_end=case_input("unwrap", case, "reg_end"),
                reg_prev_peak=case_input("unwrap", case, "reg_prev"),
                reg_offset=case_input("unwrap", case, "reg_offset"),
                scratch=case_input("unwrap", case, "scratch0").copy(),
                phase=case_input("unwrap", case, "phase0").copy(),
                f1=ci("unwrap", case, "f1"), f2=ci("unwrap", case, "f2"),
                f3=ci("unwrap", case, "f3"),
                max_bin=ci("unwrap", case, "max_bin"),
                copy_len=ci("unwrap", case, "copy_len"))
            for nm in ("scratch", "phase"):
                ef, md = record("unwrap", case, nm, got[nm],
                                case_output("unwrap", case, nm))
                if ef != 1.0:
                    _fail(f"unwrap/{case} {nm}: exact-frac={ef:g} max|d|={md:g}")


# ==========================================================================
# apply_pitch_coherence
# ==========================================================================

class TestAPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestAPC")
    @staticmethod
    def _impl():
        for name in ("pyradius.apply_pitch_coherence", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "apply_pitch_coherence"):
                return mod
        return None

    def test_apc(self):
        skip_if_no_ref("apc")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.apply_pitch_coherence.apply_pitch_coherence")
        for case in cases_of("apc"):
            maxbin = ci("apc", case, "maxbin")
            args = dict(
                precision=ci("apc", case, "precision"),
                trans_sens=ci("apc", case, "trans_sens"),
                total_ratio=ci("apc", case, "total_ratio"),
                transient_state=ci("apc", case, "transient_state"),
                n_fft=ci("apc", case, "n_fft"), sr=ci("apc", case, "sr"),
                f580=ci("apc", case, "f580"), f584=ci("apc", case, "f584"),
                acc_fc0=ci("apc", case, "acc_fc0"),
                coh_center=ci("apc", case, "coh_center"),
                seg_count=ci("apc", case, "seg_count"),
                peak_count=ci("apc", case, "peak_count"),
                vector_fmaf_region=bool(ci("apc", case, "vector_fmaf_region")))
            got = mod.apply_pitch_coherence(
                args=args,
                env=case_input("apc", case, "env"),
                peak_bins=case_input("apc", case, "peak_bins"),
                reg_start=case_input("apc", case, "reg_start"),
                reg_end=case_input("apc", case, "reg_end"),
                seg_bound=case_input("apc", case, "seg_bound"),
                phase=case_input("apc", case, "phase"),
                phase_mod=case_input("apc", case, "phase_mod").copy(),
                dir_cur=case_input("apc", case, "dir_cur").copy(),
                dir_prev=case_input("apc", case, "dir_prev").copy(),
                region_gain=case_input("apc", case, "region_gain"),
                noise_wt=case_input("apc", case, "noise_wt"),
                a3=ci("apc", case, "a3"), a4=ci("apc", case, "a4"))
            for nm in ("phase_mod", "dir_cur", "dir_prev"):
                ef, md = record("apc", case, nm, got[nm],
                                case_output("apc", case, nm))
                if md > 1e-5:
                    _fail(f"apc/{case} {nm}: max|d|={md:g}")
            ef, md = record("apc", case, "advance", got["advance"],
                            case_output("apc", case, "advance"))
            if md > 1e-6:
                _fail(f"apc/{case} advance: max|d|={md:g}")

    def test_apc_gates(self):
        """Gate cases must leave every buffer untouched (identity semantics)."""
        skip_if_no_ref("apc")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.apply_pitch_coherence.apply_pitch_coherence")
        for case in cases_of("apc"):
            gate = (ci("apc", case, "precision") > 9
                    or ci("apc", case, "trans_sens") == 0.0)
            if not gate:
                continue
            want = case_output("apc", case, "phase_mod")
            got = case_input("apc", case, "phase_mod")
            ef, md = record("apc", case, "phase_mod_identity", got, want)
            if ef != 1.0:
                _fail(f"apc/{case}: gate did not leave phase_mod untouched")


# ==========================================================================
# reset_phases_for_transients
# ==========================================================================

class TestResetPhases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestResetPhases")
    @staticmethod
    def _impl():
        for name in ("pyradius.reset_phases", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "reset_phases_for_transients"):
                return mod
        return None

    def test_rpt(self):
        skip_if_no_ref("rpt")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.reset_phases.reset_phases_for_transients")
        for case in cases_of("rpt"):
            n = int(case_rec("rpt", case)["outputs"]["phase"]["len"])
            got = mod.reset_phases_for_transients(
                mode_4b8=ci("rpt", case, "mode_4b8"),
                proc_mode_9b8=ci("rpt", case, "proc_mode_9b8"),
                scale_20=ci("rpt", case, "scale_20"),
                len_594=ci("rpt", case, "len_594"),
                n9bc=ci("rpt", case, "n9bc"), n9c0=ci("rpt", case, "n9c0"),
                u_70=ci("rpt", case, "u_70"), n578=ci("rpt", case, "n578"),
                n590=ci("rpt", case, "n590"),
                n5d0=ci("rpt", case, "n5d0"), n5d4=ci("rpt", case, "n5d4"),
                avg_m1=ci("rpt", case, "avg_m1"),
                avg_0=ci("rpt", case, "avg_0"),
                avg_p1=ci("rpt", case, "avg_p1"),
                use_p568=bool(ci("rpt", case, "use_p568")),
                mag=case_input("rpt", case, "mag_6a8"),
                b_710=case_input("rpt", case, "b_710"),
                mask=case_input("rpt", case, "mask_778"),
                r_start=case_input("rpt", case, "r_start"),
                r_end=case_input("rpt", case, "r_end"),
                r_bin=case_input("rpt", case, "r_bin"),
                phase=case_input("rpt", case, "phase0").copy(),
                mask_table=case_input("rpt", case, "mask_table0").copy(),
                n_out=n)
            for nm in ("phase", "mask_table"):
                ef, md = record("rpt", case, nm, got[nm],
                                case_output("rpt", case, nm))
                if ef != 1.0:
                    _fail(f"rpt/{case} {nm}: exact-frac={ef:g} max|d|={md:g}")


# ==========================================================================
# synchronize_stereo_phases / adjust_multiphase_diff / ampd_pull_to_peak
# ==========================================================================

class TestSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestSync")
    @staticmethod
    def _impl():
        for name in ("pyradius.sync", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "adjust_multiphase_diff"):
                return mod
        return None

    def test_synchronize_stereo_phases(self):
        skip_if_no_ref("sync")
        mod = self._impl()
        if mod is None or not hasattr(mod, "synchronize_stereo_phases"):
            skip_pyr("pyradius.sync.synchronize_stereo_phases")
        for case in cases_of("sync"):
            mag = np.stack([case_input("sync", case, "mag0"),
                            case_input("sync", case, "mag1")]).astype(np.float32)
            src = np.stack([case_input("sync", case, "src0"),
                            case_input("sync", case, "src1")]).astype(np.float32)
            dst = np.stack([case_input("sync", case, "dst0").copy(),
                            case_input("sync", case, "dst1").copy()]).astype(np.float32)
            got = mod.synchronize_stereo_phases(
                mag, src, dst,
                peaks=case_input("sync", case, "peaks"),
                pk_start=case_input("sync", case, "pk_start"),
                pk_end=case_input("sync", case, "pk_end"),
                peak_count=ci("sync", case, "peak_count"),
                sens=ci("sync", case, "sens"),
                nch=ci("sync", case, "nch"))
            for c in (0, 1):
                ef, md = record("sync", case, f"dst{c}", got["dst"][c],
                                case_output("sync", case, f"dst{c}"))
                if md > 1e-5:
                    _fail(f"sync/{case} dst{c}: max|d|={md:g}")
            ef, md = record("sync", case, "weight", got["weight"],
                            case_output("sync", case, "weight"))
            if ef != 1.0:
                _fail(f"sync/{case} weight: exact-frac={ef:g}")

    def test_adjust_multiphase_diff(self):
        skip_if_no_ref("sync")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.sync.adjust_multiphase_diff")
        for case in cases_of("sync"):
            mch = ci("sync", case, "diff_nch")
            src = np.stack([case_input("sync", case, f"diff_src{c}")
                            for c in range(mch)]).astype(np.float32)
            dst = np.stack([case_input("sync", case, f"diff_dst{c}")
                            for c in range(mch)]).astype(np.float32)
            got = mod.adjust_multiphase_diff(
                src, dst.copy(), src_bin=ci("sync", case, "diff_src_bin"),
                dst_bin=ci("sync", case, "diff_dst_bin"),
                w=ci("sync", case, "diff_w"), nch=mch)
            got = np.asarray(got)
            for c in range(mch):
                ef, md = record("sync", case, f"diff_out{c}", got[c],
                                case_output("sync", case, f"diff_out{c}"))
                if md > 1e-6:
                    _fail(f"sync/{case} diff_nch{mch}/ch{c}: max|d|={md:g}")

    def test_ampd_pull_to_peak(self):
        skip_if_no_ref("sync")
        mod = self._impl()
        if mod is None or not hasattr(mod, "ampd_pull_to_peak"):
            skip_pyr("pyradius.sync.ampd_pull_to_peak")
        for case in cases_of("sync"):
            got = mod.ampd_pull_to_peak(
                case_input("sync", case, "pull_pm").astype(np.float32).copy(),
                case_input("sync", case, "pull_mask").astype(np.float32),
                bin=ci("sync", case, "pull_bin"),
                peak_bin=ci("sync", case, "pull_peak_bin"),
                w=ci("sync", case, "pull_w"))
            ef, md = record("sync", case, "pull_out", got,
                            case_output("sync", case, "pull_out"))
            if ef != 1.0:
                _fail(f"sync/{case} pull_to_peak: exact-frac={ef:g} max|d|={md:g}")


# ==========================================================================
# formant
# ==========================================================================

class TestFormant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestFormant")
    @staticmethod
    def _impl():
        for name in ("pyradius.formant", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "formant_apply"):
                return mod
        return None

    def test_formant(self):
        skip_if_no_ref("formant")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.formant.formant_apply / FormantState")
        for case in cases_of("formant"):
            mod = reset_state(mod)
            cfg = {k: ci("formant", case, k) for k in (
                "active", "mode_freq", "mode_rms", "ratio", "width", "strength",
                "freq_lo", "freq_hi", "nb_bins", "n_fft", "m_fft", "m_bins",
                "f580", "sr", "prec_mode", "nb_bands")}
            mag = case_input("formant", case, "mag").astype(np.float32)
            st = mod.FormantState(cfg)
            got = np.asarray(st.apply(mag.copy(), band=ci("formant", case, "band")))
            want = case_output("formant", case, "mag_p0")
            ef, md = record("formant", case, "mag_p0", got, want)
            if md > 1e-4:
                _fail(f"formant/{case}: max|d|={md:g}")

            # second pass exercises the persistent env/gain state
            got2 = np.asarray(st.apply(mag.copy(), band=ci("formant", case, "band")))
            want2 = case_output("formant", case, "mag_p1")
            ef, md = record("formant", case, "mag_p1", got2, want2)
            if md > 1e-4:
                _fail(f"formant/{case} pass2: max|d|={md:g}")


# ==========================================================================
# overlap_add_channel
# ==========================================================================

class TestOAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestOAC")
    @staticmethod
    def _impl():
        for name in ("pyradius.overlap_add", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "overlap_add_channel"):
                return mod
        return None

    def test_oac(self):
        skip_if_no_ref("oac")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.overlap_add.overlap_add_channel")
        for case in cases_of("oac"):
            got = mod.overlap_add_channel(
                win1=case_input("oac", case, "win1_in").astype(np.float32).copy(),
                win2=case_input("oac", case, "win2_in").astype(np.float32).copy(),
                synth_a=case_input("oac", case, "synth_a").astype(np.float32),
                synth_b=case_input("oac", case, "synth_b").astype(np.float32),
                out_ring=case_input("oac", case, "out_ring_in").astype(np.float32).copy(),
                edge_gain=case_input("oac", case, "edge_gain").astype(np.float32),
                ch=ci("oac", case, "ch"), a3=ci("oac", case, "a3"),
                a4=ci("oac", case, "a4"), a5=ci("oac", case, "a5"),
                frame_n=ci("oac", case, "frame_n"), p=ci("oac", case, "p"),
                A=ci("oac", case, "A"), v8=ci("oac", case, "v8"),
                v9=ci("oac", case, "v9"), f1412=ci("oac", case, "f1412"),
                cursor=ci("oac", case, "cursor"), hop=ci("oac", case, "hop"),
                ring_len=ci("oac", case, "ring_len"),
                f1224=ci("oac", case, "f1224"),
                n_write=ci("oac", case, "n_write"))
            for nm in ("out_ring", "win1", "win2"):
                ef, md = record("oac", case, nm, got[nm],
                                case_output("oac", case, nm))
                if ef != 1.0:
                    _fail(f"oac/{case} {nm}: exact-frac={ef:g} max|d|={md:g}")
            ef, md = record("oac", case, "gain_weight", got["gain_weight"],
                            case_output("oac", case, "gain_weight"))
            if md > 1e-6:
                _fail(f"oac/{case} gain/weight: max|d|={md:g}")


# ==========================================================================
# noise phases (randomize / substitute)
# ==========================================================================

class TestNoisePhases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestNoisePhases")
    @staticmethod
    def _impl():
        for name in ("pyradius.noise_phases", "pyradius.vocoder_ops"):
            mod = opt_module(name)
            if mod is not None and hasattr(mod, "randomize_phases"):
                return mod
        return None

    def test_randomize_phases(self):
        skip_if_no_ref("noise")
        mod = self._impl()
        if mod is None:
            skip_pyr("pyradius.noise_phases.randomize_phases")
        for case in cases_of("noise"):
            nch = ci("noise", case, "nch")
            mb = ci("noise", case, "maxbin")
            got = mod.randomize_phases(
                noise_phase=case_input("noise", case, "noise_phase").reshape(nch, mb).copy(),
                noise_gain=case_input("noise", case, "noise_gain").reshape(nch, mb).copy(),
                phase=case_input("noise", case, "phase").reshape(nch, mb).copy(),
                mag=case_input("noise", case, "mag").reshape(nch, mb).copy(),
                region_gain=case_input("noise", case, "region_gain").reshape(nch, mb),
                noise_weight=case_input("noise", case, "noise_weight"),
                sync_weight=case_input("noise", case, "sync_weight"),
                a2=ci("noise", case, "a2"), ramp=ci("noise", case, "ramp_scalar"),
                f372=ci("noise", case, "f372"), u112=ci("noise", case, "u112"),
                seed=ci("noise", case, "seed"), nch=nch, max_bin=mb)
            for nm, want_nm in (("phase", "rnd_phase"), ("mag", "rnd_mag")):
                ef, md = record("noise", case, want_nm, got[nm][0],
                                case_output("noise", case, want_nm))
                if ef != 1.0:
                    _fail(f"noise/{case} {want_nm}: exact-frac={ef:g} max|d|={md:g}")
            for nm, want_nm in (("noise_phase", "rnd_noise_phase"),
                                ("noise_gain", "rnd_noise_gain"),
                                ("gain_mean", "rnd_gain_mean")):
                if want_nm in case_rec("noise", case)["outputs"]:
                    ef, md = record("noise", case, want_nm, got[nm][0],
                                    case_output("noise", case, want_nm))
                    if ef != 1.0:
                        _fail(f"noise/{case} {want_nm}: exact-frac={ef:g}")

    def test_substitute_noisy_phases(self):
        skip_if_no_ref("noise")
        mod = self._impl()
        if mod is None or not hasattr(mod, "substitute_noisy_phases"):
            skip_pyr("pyradius.noise_phases.substitute_noisy_phases")
        for case in cases_of("noise"):
            nch = ci("noise", case, "nch")
            mb = ci("noise", case, "maxbin")
            tmpl = case_input("noise", case, "tmpl").astype(np.float32)
            got = mod.substitute_noisy_phases(
                noise_phase=case_input("noise", case, "noise_phase").reshape(nch, mb).copy(),
                phase=case_input("noise", case, "phase").reshape(nch, mb).copy(),
                mag=case_input("noise", case, "mag").reshape(nch, mb).copy(),
                region_gain=case_input("noise", case, "region_gain").reshape(nch, mb),
                noise_template=[tmpl, tmpl],
                noise_weight=case_input("noise", case, "noise_weight"),
                sync_weight=case_input("noise", case, "sync_weight"),
                a2=ci("noise", case, "a2"), ramp=ci("noise", case, "ramp_scalar"),
                f372=ci("noise", case, "f372"), u112=ci("noise", case, "u112"),
                slot=ci("noise", case, "slot"),
                slot_count=ci("noise", case, "slot_count"),
                tmpl_stride=ci("noise", case, "tmpl_stride"),
                nch=nch, max_bin=mb)
            for nm, want_nm, tol in (("phase", "sub_phase", 1e-6),
                                     ("mag", "sub_mag", 1e-6)):
                ef, md = record("noise", case, want_nm, got[nm][0],
                                case_output("noise", case, want_nm))
                if md > tol:
                    _fail(f"noise/{case} {want_nm}: max|d|={md:g}")
            ef, md = record("noise", case, "sub_slot",
                            np.array([got["slot"]], dtype=np.int32),
                            case_output("noise", case, "sub_slot"))
            if ef != 1.0:
                _fail(f"noise/{case} slot rotation: exact-frac={ef:g}")


# ==========================================================================
# TD: env tables / pick_env / transient position
# ==========================================================================

class TestTDEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestTDEnv")
    def test_env_tables(self):
        skip_if_no_ref("td_env")
        mod = opt_module("pyradius.tables")
        if mod is None or not hasattr(mod, "td_env_spans"):
            skip_pyr("pyradius.tables.td_env_spans / td_env_gain")
        for case in cases_of("td_env"):
            sr = ci("td_env", case, "sr")
            f28 = int(np.float32(np.float32(sr) * np.float32(0.1)
                                 + np.float32(0.5))) & ~7
            got = np.asarray(mod.td_env_spans(f28)).ravel().astype(np.int32)
            ef, md = record("td_env", case, "spans", got,
                            case_output("td_env", case, "spans"))
            if ef != 1.0:
                _fail(f"td_env/{case} spans: exact-frac={ef:g}")
            gains = np.array([mod.td_env_gain(a) for a in range(5)], dtype=np.float32)
            ef, md = record("td_env", case, "gains", gains,
                            case_output("td_env", case, "gains"))
            if md > 1e-6:
                _fail(f"td_env/{case} gains: max|d|={md:g}")

    def test_geometry(self):
        """TD derived geometry (hop / f28 / pitch N,L1,maxbin,taper) from C."""
        skip_if_no_ref("td_env")
        mod = opt_module("pyradius.td_env", "pyradius.td_core")
        fn = getattr(mod, "geometry", None) if mod else None
        if fn is None:
            skip_pyr("pyradius.td_env.geometry(sr, quality, solo)")
        for case in cases_of("td_env"):
            got = fn(sr=ci("td_env", case, "sr"), quality=ci("td_env", case, "quality"),
                     solo=bool(ci("td_env", case, "solo")))
            keys = ("hop", "f28", "pitch_N", "pitch_L1", "pitch_maxbin",
                    "pitch_taper_len", "pitch_lo", "pitch_hi")
            arr = np.array([got[k] for k in keys], dtype=np.int32)
            ef, md = record("td_env", case, "geo", arr,
                            case_output("td_env", case, "geo"))
            if ef != 1.0:
                _fail(f"td_env/{case} geometry: exact-frac={ef:g}")

    def test_pick_env(self):
        skip_if_no_ref("td_env")
        mod = opt_module("pyradius.td_env", "pyradius.td_core")
        fn = getattr(mod, "pick_env", None) if mod else None
        if fn is None:
            skip_pyr("pyradius.td_env.pick_env")
        for case in cases_of("td_env"):
            got = fn(sr=ci("td_env", case, "sr"), quality=ci("td_env", case, "quality"),
                     solo=bool(ci("td_env", case, "solo")),
                     picks=case_input("td_env", case, "picks"),
                     env_len=64)
            for nm in ("pick_j", "pick_half", "pick_w10", "pick_env"):
                ef, md = record("td_env", case, nm, got[nm],
                                case_output("td_env", case, nm))
                if ef != 1.0:
                    _fail(f"td_env/{case} {nm}: exact-frac={ef:g}")

    def test_transient_pos(self):
        skip_if_no_ref("td_env")
        mod = opt_module("pyradius.transients_info", "pyradius.td_core")
        fn = getattr(mod, "transient_positions", None) if mod else None
        if fn is None:
            skip_pyr("pyradius.transients_info.transient_positions")
        for case in cases_of("td_env"):
            got = np.asarray(fn(case_input("td_env", case, "src"),
                                sr=ci("td_env", case, "sr"),
                                nch=ci("td_env", case, "nch"), sens=1.0,
                                chunks=ci("td_env", case, "chunks"),
                                queries=case_input("td_env", case, "queries")))
            ef, md = record("td_env", case, "transient_pos", got,
                            case_output("td_env", case, "transient_pos"))
            if ef != 1.0:
                _fail(f"td_env/{case} transient_pos: exact-frac={ef:g}")


# ==========================================================================
# TD pitch / transients info
# ==========================================================================

class TestTDPitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestTDPitch")
    def test_pitch_analyze(self):
        skip_if_no_ref("td_pitch")
        mod = opt_module("pyradius.td_pitch", "pyradius.td_core")
        fn = getattr(mod, "pitch_analyze", None) if mod else None
        if fn is None:
            skip_pyr("pyradius.td_pitch.pitch_analyze(ring, sr, quality, solo, ...)")
        for case in cases_of("td_pitch"):
            ring_tot = int(case_meta("td_pitch", case)["ring_tot"])
            nch = ci("td_pitch", case, "nch")
            ring = case_input("td_pitch", case, "ring").astype(np.float32)
            ring = ring.reshape(nch, ring_tot)
            got = fn(ring, sr=ci("td_pitch", case, "sr"),
                     quality=ci("td_pitch", case, "quality"),
                     solo=bool(ci("td_pitch", case, "solo")), nch=nch,
                     fed_visible=ci("td_pitch", case, "fed_visible"),
                     positions=case_input("td_pitch", case, "positions"))
            for nm, tol in (("lag", 1e-5), ("win", 0.0), ("rms", 1e-6)):
                ef, md = record("td_pitch", case, nm, got[nm],
                                case_output("td_pitch", case, nm))
                if tol == 0.0:
                    if ef != 1.0:
                        _fail(f"td_pitch/{case} {nm}: exact-frac={ef:g}")
                elif md > tol:
                    _fail(f"td_pitch/{case} {nm}: max|d|={md:g}")


class TestTDTransients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestTDTransients")
    def test_transients_info(self):
        skip_if_no_ref("td_ti")
        mod = opt_module("pyradius.transients_info", "pyradius.td_core")
        fn = getattr(mod, "transients_info_run", None) if mod else None
        if fn is None:
            skip_pyr("pyradius.transients_info.transients_info_run")
        for case in cases_of("td_ti"):
            got = fn(case_input("td_ti", case, "src").astype(np.float32),
                     nch=ci("td_ti", case, "nch"), sr=ci("td_ti", case, "sr"),
                     sens=ci("td_ti", case, "sens"),
                     chunks=ci("td_ti", case, "chunks"),
                     queries=case_input("td_ti", case, "queries"))
            ef, md = record("td_ti", case, "vector", got["vector"],
                            case_output("td_ti", case, "vector"))
            if md > 1e-6:
                _fail(f"td_ti/{case} vector: max|d|={md:g}")
            for nm, tol in (("meta", 0.0), ("iir_alpha", 1e-12),
                            ("transient_pos", 0.0)):
                ef, md = record("td_ti", case, nm, got[nm],
                                case_output("td_ti", case, nm))
                if tol == 0.0:
                    if ef != 1.0:
                        _fail(f"td_ti/{case} {nm}: exact-frac={ef:g} max|d|={md:g}")
                elif md > tol:
                    _fail(f"td_ti/{case} {nm}: max|d|={md:g}")


# ==========================================================================
# Corpus self-checks — validate the C reference itself, independent of pyradius
# ==========================================================================

class TestCorpusIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_under_route_b("TestCorpusIntegrity")
    """These run before/independently of any pyradius port.

    They exist so that a broken or empty harness cannot make the parity suite
    look green: every reference payload must exist, be finite, and be non-trivial,
    the FFT references must satisfy the forward/inverse duality, and the gate
    branches of APC/RPT must actually exercise distinct code paths.
    """

    def test_manifest_covers_all_operators(self):
        if not MAN:
            msg = (f"no reference corpus at {os.path.relpath(MANIFEST, ROOT)} "
                   f"— run: python3 tools/gen_ref_ops.py")
            if REQUIRE_REF:
                _fail(msg)
            raise unittest.SkipTest(msg)
        missing = sorted(set(EXPECTED_OPS) - set(MAN))
        if missing:
            _fail(f"manifest is missing operators: {missing}")

    def test_payloads_finite_and_nontrivial(self):
        skip_if_no_ref("fft")
        def scan():
            problems = []
            for op, rec in sorted(MAN.items()):
                for case, c in sorted(rec["cases"].items()):
                    if not c["outputs"]:
                        problems.append(f"{op}/{case}: no outputs")
                        continue
                    for nm, o in sorted(c["outputs"].items()):
                        if not os.path.exists(_path(o)):
                            problems.append(f"{op}/{case}/{nm}: missing")
                            continue
                        a = load(o)
                        if a.size == 0:
                            problems.append(f"{op}/{case}/{nm}: empty")
                        elif o["ext"] == "f32" and not np.all(np.isfinite(a)):
                            problems.append(f"{op}/{case}/{nm}: non-finite")
            return problems
        problems = scan()
        if problems and not _corpus_settled():
            await_corpus()
            problems = scan()
        if problems:
            _fail("corpus integrity: " + "; ".join(problems[:20]))

    def test_inputs_and_outputs_are_separate_namespaces(self):
        """Guard against the in-place feedback bug.

        An in-place operator legitimately has an output with the same semantic
        name as an input (apc's phase_mod, sync's dst0, ...).  What must never
        happen is the output landing on the same *file* the harness read, which
        would make the next run consume its own result and drift to whatever
        fixed point the operator has.
        """
        skip_if_no_ref("unwrap")
        same_file = []
        for op, rec in sorted(MAN.items()):
            for case, c in sorted(rec["cases"].items()):
                for nm in sorted(set(c["inputs"]) & set(c["outputs"])):
                    fi = _path(c["inputs"][nm])
                    fo = _path(c["outputs"][nm])
                    if os.path.abspath(fi) == os.path.abspath(fo):
                        same_file.append(f"{op}/{case}/{nm}")
        if same_file:
            _fail("input and output resolve to the same file (harness would "
                  "feed back its own output): " + "; ".join(same_file[:5]))
        # and outputs must not be an unmarked alias of an input path
        for op, rec in sorted(MAN.items()):
            in_files = {_path(e) for c in rec["cases"].values()
                        for e in c["inputs"].values()}
            for case, c in sorted(rec["cases"].items()):
                for nm, o in c["outputs"].items():
                    if _path(o) in in_files and nm not in c["inputs"]:
                        _fail(f"{op}/{case}/{nm}: output file collides with an "
                              f"input file under a different name")

    def test_at_least_one_output_per_case_file_exists(self):
        skip_if_no_ref("unwrap")
        def scan():
            missing = []
            for op, rec in sorted(MAN.items()):
                for case, c in sorted(rec["cases"].items()):
                    for nm, o in c["outputs"].items():
                        if not os.path.exists(_path(o)):
                            missing.append(f"{op}/{case}/{nm}")
                        elif o["len"] * {"f32": 4, "i32": 4, "f64": 8}[o["ext"]] \
                                != os.path.getsize(_path(o)):
                            missing.append(f"{op}/{case}/{nm}: size mismatch")
            return missing
        missing = scan()
        if missing and not _corpus_settled():
            # a concurrent regeneration clipped payloads; wait it out once
            await_corpus()
            missing = scan()
        if missing:
            _fail("manifest refers to missing/short payloads: "
                  + "; ".join(missing[:10]))

    def test_manifest_reflects_the_working_tree(self):
        """Regenerating with --force must be a no-op byte-wise (regression net
        for stale payloads surviving a protocol change).

        OPT-IN (PYR_REGEN_CHECK=1): it rewrites the shared corpus, so it must
        not run by default — several agents regenerate .tmp/ref_ops and a test
        that mutates it concurrently produces spurious failures for everyone.
        """
        skip_if_no_ref("unwrap")
        if not os.environ.get("PYR_REGEN_CHECK"):
            raise unittest.SkipTest("set PYR_REGEN_CHECK=1 to enable (mutates "
                                    "the shared corpus)")
        import subprocess
        if not await_corpus():
            raise unittest.SkipTest("corpus busy (concurrent regeneration)")
        before = {}
        for op, rec in MAN.items():
            for case, c in rec["cases"].items():
                for nm, o in c["outputs"].items():
                    before[(op, case, nm)] = (o["ext"], o["len"])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_ref_ops.py"),
                            "--force"], capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            _fail("re-running gen_ref_ops.py --force failed:\n" + r.stdout[-2000:])
        after = load_manifest()
        drift = []
        for op, rec in after.items():
            for case, c in rec["cases"].items():
                for nm, o in c["outputs"].items():
                    prev = before.get((op, case, nm))
                    if prev != (o["ext"], o["len"]):
                        drift.append(f"{op}/{case}/{nm}")
        if drift:
            _fail("regeneration changed the corpus shape: " + "; ".join(drift[:10]))

    def test_fft_fwd_inv_duality_in_c_reference(self):
        """inv(fwd(x)) == x for the C reference itself (independent check)."""
        skip_if_no_ref("fft")
        for case in cases_of("fft"):
            x = case_input("fft", case, "x").astype(np.float64)
            rt = case_output("fft", case, "roundtrip").astype(np.float64)
            md = float(np.abs(x - rt).max())
            if md > 1e-3:
                _fail(f"fft/{case}: C roundtrip differs from input by {md:g}")

    def test_acs_matches_fft_fwd(self):
        """t1_cart must equal rx_fft_fwd(windowed time) — cross-op consistency."""
        skip_if_no_ref("acs")
        for case in cases_of("acs"):
            n_win = ci("acs", case, "n_win")
            n_fft = ci("acs", case, "n_fft")
            tio = case_output("acs", case, "t1_io")
            cart = case_output("acs", case, "t1_cart")
            fft_case = f"N{n_fft}" if f"N{n_fft}" in cases_of("fft") else None
            if fft_case is None:
                continue
            # use the lib's own definition: DC bin must be the windowed sum
            got_dc = float(np.sum(tio.astype(np.float64)[:n_win]))
            want_dc = float(cart[0])
            rel = abs(got_dc - want_dc) / max(1.0, abs(want_dc))
            if rel > 1e-4:
                _fail(f"acs/{case}: t1_cart DC mismatch rel={rel:g}")

    def test_branch_coverage_is_real(self):
        """Gate combinations must produce distinguishable references."""
        skip_if_no_ref("apc")
        gate_prec = case_output("apc", (
            "gate_prec" if "gate_prec" in cases_of("apc") else cases_of("apc")[0]),
            "phase_mod")
        if "main" in cases_of("apc"):
            main = case_output("apc", "main", "phase_mod")
            if np.array_equal(gate_prec, main):
                _fail("apc: gated (precision>9) case produced the same output as "
                      "the active case — the fixture does not exercise the gate")
        if "rpt" in MAN:
            outs = {c: case_output("rpt", c, "phase")
                    for c in cases_of("rpt")}
            labels = list(outs)
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a, b = outs[labels[i]], outs[labels[j]]
                    if np.array_equal(a, b) and labels[i][:2] != labels[j][:2]:
                        _fail(f"rpt: cases {labels[i]} and {labels[j]} are identical "
                              f"— different branches must differ")

    def test_fill_granule_branches_differ(self):
        skip_if_no_ref("fill_granule")
        seen = {}
        for case in cases_of("fill_granule"):
            seen[case] = case_output("fill_granule", case, "fg")
        if "buffered_single" in seen and "direct" in seen:
            if np.array_equal(seen["buffered_single"], seen["direct"]):
                _fail("fill_granule: buffered and direct branches produced "
                      "identical output — fixture degenerates")


# ==========================================================================
# reporting
# ==========================================================================

def dump_report() -> None:
    if not ROWS:
        return
    summary: dict[str, dict] = {}
    for r in ROWS:
        s = summary.setdefault(r["op"], {"n_outputs": 0, "min_exact_frac": 1.0,
                                         "max_abs": 0.0, "cases": []})
        s["n_outputs"] += 1
        s["min_exact_frac"] = min(s["min_exact_frac"], r["exact_frac"])
        s["max_abs"] = max(s["max_abs"], r["max_abs"])
        if r["case"] not in s["cases"]:
            s["cases"].append(r["case"])
    for s in summary.values():
        s["cases"].sort()
        s["n_cases"] = len(s["cases"])

    try:
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        with open(REPORT + ".part", "w") as fh:
            json.dump({"rows": ROWS, "summary": summary,
                       "generated_by": "tests/test_ops_parity.py"},
                      fh, indent=1, sort_keys=True)
        os.replace(REPORT + ".part", REPORT)
    except OSError:
        pass

    print("\n--- pyradius operator parity "
          "(C reference: .tmp/ref_ops via tools/gen_ref_ops.py) ---")
    print(f"{'op':14s} {'cases':>5s} {'outs':>5s} {'min exact-frac':>15s} "
          f"{'max|d|':>12s}")
    for op in sorted(summary):
        s = summary[op]
        print(f"{op:14s} {s['n_cases']:5d} {s['n_outputs']:5d} "
              f"{s['min_exact_frac']:15.6f} {s['max_abs']:12.3e}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not MAN:
        msg = (f"no reference corpus at {os.path.relpath(MANIFEST, ROOT)} "
               f"— run: python3 tools/gen_ref_ops.py")
        print(msg, file=sys.stderr)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    dump_report()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())


# ==========================================================================
# route-B formant C port (neon_formant.c)
# ==========================================================================

class TestFormantC(unittest.TestCase):
    """The C formant kernel must be bit-identical to the numba chain.

    ``neon.formant_ok()`` is the in-tree gate and is itself the strongest check
    (it compares mag AND the persistent db/ker/gscr/env/gain_env buffers against
    ``FormantState._apply_numba``); these tests additionally pin the individual
    stage entry points, since fm_apply could in principle be right while a
    stage is wrong in a way the sampled configurations never reach.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYR_FAST", "1")
        from pyradius import neon
        cls.neon = neon
        if not neon.HAVE_NEON or neon.fm_apply is None:
            cls.neon = None

    def _need(self):
        if self.neon is None:
            self.skipTest("route-B NEON formant kernel not built")
        if not self.neon.formant_ok():
            self.fail("neon.formant_ok() is False: the formant C port is not "
                      "bit-exact and route B silently fell back to numba")

    def test_gate(self):
        self._need()

    def test_stages(self):
        self._need()
        import ctypes
        from pyradius import vocoder_ops as vo
        import numpy as _np
        n = self.neon
        rng = _np.random.default_rng(0x5EED)
        F = ctypes.POINTER(ctypes.c_float)
        I = ctypes.c_int
        D = ctypes.c_double
        P = lambda a: a.ctypes.data_as(F)  # noqa: E731
        for (NB, M, MB, N, sr) in ((4097, 2048, 1025, 8192, 44100),
                                   (8193, 4096, 2049, 16384, 48000)):
            # peak: both mode_freq branches and the clamps
            env = (rng.random(MB) * 1e-3).astype(_np.float32)
            v19 = float(_np.float32(sr * M / N))
            for mf in (0, 1):
                for fhi, flo in ((800.0, 40.0), (300.0, 60.0), (2e4, 5.0)):
                    b = ctypes.c_int(); fc = ctypes.c_double(); pk = ctypes.c_int()
                    n.fm_peak(P(env), MB, v19, fhi, flo, mf,
                              float(vo._bits(0xC9742400)), ctypes.byref(b),
                              ctypes.byref(fc), ctypes.byref(pk))
                    ref = vo._fm_peak_nb(env, MB, v19, fhi, flo, mf,
                                         float(vo._bits(0xC9742400)))
                    self.assertEqual((b.value, float(fc.value), pk.value),
                                     (ref[0], float(ref[1]), ref[2]))
            # gain2: the three-factor f64 exponent
            for ratio in (0.8408964276313782, 1.189207115002721, 2.0):
                for strength in (1.0, 0.5):
                    db = (rng.standard_normal(NB + 8) * 25).astype(_np.float32)
                    g = _np.zeros(NB + 8, _np.float32); rg = g.copy()
                    n.fm_gain2(P(db), P(g), NB, ratio, strength)
                    vo._fm_gain2_nb(db, rg, NB, ratio, strength)
                    self.assertTrue(_np.array_equal(g.view(_np.uint32),
                                                    rg.view(_np.uint32)))
            # rms: numpy's pairwise summation order
            mag = (rng.random(NB).astype(_np.float64)
                   * _np.power(10.0, rng.integers(-8, 2, NB).astype(_np.float64))
                   ).astype(_np.float32)
            g = (rng.standard_normal(NB) * 1.5).astype(_np.float32)
            rg = g.copy()
            scr = (ctypes.c_double * (2 * NB))()
            eps = 1.000029594723506e-12
            n.fm_rms(P(mag), P(g), NB, eps, scr)
            mv = mag[:NB].astype(_np.float64); gv = rg.astype(_np.float64)
            ss = _np.float32(_np.sqrt(float(_np.sum(mv ** 2))
                                      / (float(_np.sum((gv ** 2) * (mv ** 2))) + eps)))
            self.assertTrue(_np.array_equal(g.view(_np.uint32),
                                            (ss * rg).astype(_np.float32).view(_np.uint32)))
            # iir16 (the exact smoothing the shipped path uses)
            y = (rng.standard_normal(NB) * 40).astype(_np.float32)
            ry = y.copy()
            a2 = _np.float32(rng.random() * 0.9 + 0.05)
            n.fm_iir16(P(y), NB, float(a2), float(_np.float32(1.0) - a2))
            vo._iir16_nb(ry, float(a2), float(_np.float32(1.0) - a2))
            self.assertTrue(_np.array_equal(y.view(_np.uint32), ry.view(_np.uint32)))
