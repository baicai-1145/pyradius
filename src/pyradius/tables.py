"""pyradius.tables — C constant-table access shims (two modes).

Engine-verified constant tables (TD env spans/gains are *computed*; vocoder
windows/FIR/schedule are *data*) live in the C sources. This module extracts
them once by parsing the C headers, or reads a cached .npz. Mode B later
replaces the windows/schedule with generated equivalents.

Sources:
  - span/gain: computed by rx_td_build_env_tables formula (log/exp), verified
    against measured table; gain blocks 0.5625+0.125*k, block4 = 1.0.
  - fir48/fir44: crossover_fir_table.h / _44k.h [4][2048] f32
  - win48/win44: vc_win_table.h / _44k.h [28736]/[26396] f32 (4 bands x n_write)
  - synth44: vc_synth_win_44k.h (a/b, 6599 each)
  - sched: vc_schedule.h pos/cdel/f1/f2 (g_max+1 = 28993 entries) — reference
    truth; mode B regenerates via rx_vc_calc_sched closed form.
"""
from __future__ import annotations

import os
import re
import numpy as np

_LIBRADIUS = os.environ.get(
    "PYR_LIBRADIUS",
    os.path.join(os.path.dirname(__file__), "..", "libradius"),
)
_CACHE = os.environ.get("PYR_TABLE_CACHE", "")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tables")


def _parse_floats(text: str) -> np.ndarray:
    vals = re.findall(r"([-0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)f", text)
    return np.array([float(v) for v in vals], dtype=np.float32)


def _load_c_table(path: str, names: list[str]) -> dict[str, np.ndarray]:
    src = open(path).read()
    out = {}
    for name in names:
        m = re.search(
            rf"static const float\s+{name}\s*(?:\[[^\]]*\])*\s*=\s*\{{(.*?)\}};",
            src, re.S)
        if m is None:
            raise KeyError(f"{name} not found in {path}")
        out[name] = _parse_floats(m.group(1))
    return out


def get_tables(sr: int) -> dict[str, np.ndarray]:
    """Return dict with fir, win, synth (per samplerate)."""
    npz_name = f"tables_{sr // 1000}k.npz"
    npz_path = os.path.join(_CACHE_DIR, npz_name)
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}

    if sr == 48000:
        fir = _load_c_table(os.path.join(_LIBRADIUS, "src/ops/crossover_fir_table.h"),
                            ["rx_xover_fir_taps"])["rx_xover_fir_taps"].reshape(4, 2048)
        win = _load_c_table(os.path.join(_LIBRADIUS, "src/engine/vc_win_table.h"),
                            ["rx_vc_default_win_table"])["rx_vc_default_win_table"]
        return {"fir": fir, "win": win}
    elif sr == 44100:
        fir = _load_c_table(os.path.join(_LIBRADIUS, "src/ops/crossover_fir_table_44k.h"),
                            ["rx_xover_fir_taps_44k"])["rx_xover_fir_taps_44k"].reshape(4, 2048)
        win = _load_c_table(os.path.join(_LIBRADIUS, "src/engine/vc_win_table_44k.h"),
                            ["rx_vc_default_win_table_44k"])["rx_vc_default_win_table_44k"]
        synth = _load_c_table(os.path.join(_LIBRADIUS, "src/engine/vc_synth_win_44k.h"),
                              ["rx_vc_synth_win_a_44k", "rx_vc_synth_win_b_44k"])
        return {"fir": fir, "win": win, "synth_a": synth["rx_vc_synth_win_a_44k"],
                "synth_b": synth["rx_vc_synth_win_b_44k"]}
    raise ValueError(f"unsupported sr={sr}")


# ---- TD env tables (computed, identical to C formula) ----

def td_env_spans(f28: int) -> np.ndarray:
    """32 spans per block; formula: L_j = int(expf(v)+0.5), v = log-space interp."""
    C = 32
    s8 = np.float32(np.log(np.float32(f28 // 10)))
    s9 = np.float32(np.log(np.float32(f28)))
    out = np.empty((5, C), dtype=np.int64)
    for a8 in range(5):
        for j in range(C):
            v = np.float32(np.float32(s8 * np.float32(C - 1 - j)) + np.float32(s9 * np.float32(j))) / np.float32(C - 1)
            L = int(np.float32(np.exp(v)) + np.float32(0.5))
            if L < 2:
                L = 2
            out[a8, j] = L
    return out


def td_env_gain(a8: int) -> np.float32:
    return np.float32(0.5625 + 0.125 * a8) if a8 < 4 else np.float32(1.0)


def hann_pow(len_: int, gain: float) -> np.ndarray:
    """Full record: env1 = Hann^gain (len), env2 = same (len). As in hann_pow()."""
    i = np.arange(len_, dtype=np.float64)
    u = (i + 0.5) / len_
    h = 0.5 - 0.5 * np.cos(2.0 * 3.14159265358979323846 * u)
    v = np.power(h, gain).astype(np.float32)
    return np.concatenate([v, v])


# ---- vocoder schedule closed form (mode B) ----

def vc_calc_sched(g: int, ratio: float, nominal_step: float) -> tuple[int, int, int, int]:
    """Port of static inline rx_vc_calc_sched (vocoder_core.c)."""
    import math as _m
    if g < 5:
        return (0, 0, 0, 0)
    grp = (g - 1) // 4
    if ratio >= 1:
        in_step = int(round(nominal_step))
        out_step = int(round(nominal_step * ratio))
    else:
        in_step = int(round(nominal_step * _m.sqrt(2)))
        out_step = int(round(in_step * ratio))
    pos = grp * out_step
    cdel = in_step if (g % 4 == 1) else 0
    return (pos, cdel, in_step, out_step)
