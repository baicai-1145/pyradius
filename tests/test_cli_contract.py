"""tests/test_cli_contract.py — the interface the ports must expose.

`pyradius/cli_td.py`, `cli_vc.py`, `tools/acceptance.py` and `tools/verify.py`
were all written against an agreed interface. If a port drifts from it (state
constructor arity, `set_ratio` signature, the shape of what `*_render` returns),
the failure shows up as an AttributeError deep inside a 5-minute render.

These tests pin the contract with stub modules, so a port can be checked
without the real implementation existing yet:

    TDState(sr, quality, solo, nch)
    st.set_ratio(semis, cent)              -> sets st.total_ratio
    st.hop / st.f28 / st.win_max           -> ints, used by the CLI banner
    td_render(st, x) -> (y[n, nch], {"n_granule","n_transient","wrap_cnt"})

    VocoderState(sr, nch, precision)
    st.set_ratio(semis, cent)
    vc_render(st, x, trace=None) -> (y[n, nch], {"cursor","pos","made"})

Run: python3 -m pytest tests/test_cli_contract.py -q
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".tmp")

CALLS: dict[str, list] = {"td": [], "vc": []}


def _make_td_stub():
    mod = types.ModuleType("pyradius.td_core")

    class TDState:
        def __init__(self, sr, quality, solo, nch):
            CALLS["td"].append(("TDState", sr, quality, solo, nch))
            self.sr, self.quality, self.solo, self.nch = sr, quality, solo, nch
            self.hop, self.f28, self.win_max = 1024, 2048, 4
            self.total_ratio = 1.0

        def set_ratio(self, semis, cent):
            CALLS["td"].append(("set_ratio", semis, cent))
            self.total_ratio = 2.0 ** (semis / 12.0)

    def td_render(st, x):
        CALLS["td"].append(("td_render", x.shape))
        y = (x * np.float32(0.5)).astype(np.float32)
        return y, {"n_granule": 7, "n_transient": 3, "wrap_cnt": 0}

    mod.TDState, mod.td_render = TDState, td_render
    return mod


def _make_vc_stub():
    mod = types.ModuleType("pyradius.vocoder_core")

    class VocoderState:
        def __init__(self, sr, nch, precision):
            CALLS["vc"].append(("VocoderState", sr, nch, precision))
            self.sr, self.nch, self.precision = sr, nch, precision
            self.total_ratio = 1.0

        def set_ratio(self, semis, cent):
            CALLS["vc"].append(("set_ratio", semis, cent))
            self.total_ratio = 2.0 ** (semis / 12.0)

    def vc_render(st, x, trace=None):
        CALLS["vc"].append(("vc_render", x.shape, trace))
        y = (x * np.float32(0.25)).astype(np.float32)
        return y, {"cursor": 10, "pos": 20, "made": 30}

    mod.VocoderState, mod.vc_render = VocoderState, vc_render
    return mod


@pytest.fixture
def stubs(monkeypatch):
    import pyradius  # noqa: F401  (ensures the package object exists)

    for name in ("pyradius.td_core", "pyradius.vocoder_core"):
        attr = name.split(".")[1]
        monkeypatch.delitem(sys.modules, name, raising=False)
        if hasattr(pyradius, attr):
            monkeypatch.delattr(pyradius, attr, raising=False)
    monkeypatch.setitem(sys.modules, "pyradius.td_core", _make_td_stub())
    monkeypatch.setitem(sys.modules, "pyradius.vocoder_core", _make_vc_stub())
    CALLS["td"].clear()
    CALLS["vc"].clear()
    yield
    for name in ("pyradius.td_core", "pyradius.vocoder_core"):
        sys.modules.pop(name, None)


def _write_input(path: str, sr: int = 48000, n: int = 4096, nch: int = 2) -> np.ndarray:
    from pyradius.wavio import write_wav_f32

    t = np.arange(n) / sr

    def channel(amp, freq):
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    x = np.stack([channel(0.5, 440.0), channel(0.25, 660.0)], axis=1).astype(np.float32)
    write_wav_f32(path, x, sr, nch)
    return x


def test_td_cli_full_path(stubs, tmp_path):
    from pyradius.cli_td import main

    inp = str(tmp_path / "in.wav")
    out = str(tmp_path / "out.wav")
    x = _write_input(inp)

    rc = _run_main(main, ["cli_td", inp, out, "3", "37", "0"])
    assert rc == 0
    assert ("TDState", 48000, 37, 0, 2) in CALLS["td"]
    assert ("set_ratio", 3.0, 100.0) in CALLS["td"]

    from pyradius.wavio import read_wav

    y, sr, nch = read_wav(out)
    assert (sr, nch) == (48000, 2)
    assert y.shape == x.shape
    assert np.allclose(y, x * np.float32(0.5))


def test_td_cli_truth_report_accepts_ref_and_dash(stubs, tmp_path, capsys):
    from pyradius.cli_td import main

    inp = str(tmp_path / "in.wav")
    out = str(tmp_path / "out.wav")
    truth = str(tmp_path / "truth.wav")
    x = _write_input(inp)
    _write_input(truth)

    rc = _run_main(main, ["cli_td", inp, out, "3", "37", "0", truth])
    assert rc == 0
    assert "CORR vs gold" in capsys.readouterr().out

    rc = _run_main(main, ["cli_td", inp, out, "3", "37", "0", "-"])
    assert rc == 0


def test_vc_cli_full_path(stubs, tmp_path):
    from pyradius.cli_vc import main

    inp = str(tmp_path / "in.wav")
    out = str(tmp_path / "out.wav")
    x = _write_input(inp, sr=44100)

    rc = _run_main(main, ["cli_vc", inp, out, "-3", "2"])
    assert rc == 0
    assert ("VocoderState", 44100, 2, 2) in CALLS["vc"]
    assert ("set_ratio", -3.0, 100.0) in CALLS["vc"]
    assert ("vc_render", x.shape, None) in CALLS["vc"]

    from pyradius.wavio import read_wav

    y, sr, nch = read_wav(out)
    assert (sr, nch) == (44100, 2)
    assert y.shape == x.shape


def test_vc_cli_rejects_unsupported_sample_rate(stubs, tmp_path):
    from pyradius.cli_vc import main

    inp = str(tmp_path / "in.wav")
    _write_input(inp, sr=22050)
    rc = _run_main(main, ["cli_vc", inp, str(tmp_path / "o.wav"), "3"])
    assert rc == 1


@pytest.mark.parametrize("mod", ["cli_td", "cli_vc"])
def test_cli_usage_on_missing_args(mod, stubs, capsys):
    import importlib

    m = importlib.import_module(f"pyradius.{mod}")
    assert _run_main(m.main, [mod]) == 2
    assert "usage:" in capsys.readouterr().err


def test_acceptance_run_py_contract(stubs, tmp_path):
    """tools/acceptance.py must drive the same interface as the CLIs."""
    A = pytest.importorskip("tools.acceptance")

    inp = str(tmp_path / "in.wav")
    out = str(tmp_path / "out.wav")
    x = _write_input(inp)

    for mode in ("td", "vc"):
        dt, sr, err = A.run_py(mode, inp, out, 3.0, 37, 0, 2)
        assert err == "" and dt >= 0.0 and sr == 48000
        from pyradius.wavio import read_wav

        y, _, _ = read_wav(out)
        assert y.shape == x.shape


def test_run_py_does_not_reuse_render_state(stubs, tmp_path):
    A = pytest.importorskip("tools.acceptance")
    """Regression: the renderer is STATEFUL, so run_py must not time a repeat over
    one state object.

    I once 'improved' run_py to median-of-N over a single state, which silently
    corrupted every sample after the first (max|d| = 1.77 vs the reference on 4.wav)
    while looking like a harmless timing change. This test fails loudly if the state
    is reused: the stub counts distinct state constructions and the output must match
    a single clean render.
    """
    inp = str(tmp_path / "in.wav")
    out = str(tmp_path / "out.wav")
    x = _write_input(inp)
    before = len(CALLS["td"])
    A.run_py("td", inp, out, 3.0, 37, 0, 2)
    calls = CALLS["td"][before:]
    n_states = sum(1 for c in calls if c[0] == "TDState")
    n_renders = sum(1 for c in calls if c[0] == "td_render")
    assert n_states == n_renders, (
        f"{n_renders} renders but {n_states} state constructions - the state was reused")
    assert n_states >= 2, "expected a warm-up render plus the timed render"
    from pyradius.wavio import read_wav

    y, _, _ = read_wav(out)
    assert np.allclose(y, x * np.float32(0.5)), "output must be one clean render"


def _run_main(main, argv) -> int:
    """Call a CLI entry point with argv patched, restoring it afterwards."""
    saved = sys.argv
    sys.argv = list(argv)
    try:
        return main()
    finally:
        sys.argv = saved
