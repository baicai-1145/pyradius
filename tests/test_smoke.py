"""tests/test_smoke.py — self-contained end-to-end smoke tests for pyradius."""
import os
import numpy as np
import pytest

from pyradius.td_core import TDState, td_render
from pyradius.vocoder_core import VocoderState, vc_render
from pyradius.tables import get_tables
from pyradius.wavio import write_wav_f32, read_wav


def _generate_sine(sr=48000, duration=0.25, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    # 2 channels (stereo)
    return np.column_stack([sig, sig])


def test_tables_load():
    t48 = get_tables(48000)
    assert "fir" in t48 and "win" in t48
    assert t48["fir"].shape == (4, 2048)
    assert t48["win"].shape == (28736,)

    t44 = get_tables(44100)
    assert "fir" in t44 and "win" in t44 and "synth_a" in t44 and "synth_b" in t44
    assert t44["fir"].shape == (4, 2048)
    assert t44["win"].shape == (26396,)


def test_td_render_smoke():
    sr = 48000
    x = _generate_sine(sr=sr, duration=0.1)
    st = TDState(sr=sr, quality=37, solo=0, nch=2)
    st.set_ratio(semis=3.0, tempo=100.0)
    y, info = td_render(st, x)
    assert y.shape[1] == 2
    assert y.shape[0] > 0
    assert not np.isnan(y).any()
    assert not np.isinf(y).any()


def test_vc_render_smoke():
    sr = 48000
    x = _generate_sine(sr=sr, duration=0.1)
    st = VocoderState(sr=sr, nch=2, precision=2)
    st.set_ratio(semis=-3.0, quality=100.0)
    y, info = vc_render(st, x)
    assert y.shape[1] == 2
    assert y.shape[0] > 0
    assert not np.isnan(y).any()
    assert not np.isinf(y).any()


def test_wavio_roundtrip(tmp_path):
    wav_path = str(tmp_path / "test.wav")
    x = _generate_sine(sr=44100, duration=0.1)
    write_wav_f32(wav_path, x, 44100, 2)
    y, sr, nch = read_wav(wav_path)
    assert sr == 44100
    assert nch == 2
    assert np.allclose(x, y, atol=1e-6)
