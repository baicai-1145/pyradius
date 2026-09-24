"""tests/test_wavio.py — bit-exactness of the WAV I/O path.

Every acceptance number flows through pyradius.wavio, so a silent I/O
discrepancy would contaminate the whole matrix. These tests check the reader
against an *independent* RIFF parser written here (not against pyradius code)
and the writer against raw bytes of a C-rendered reference.

Run: python3 -m pytest tests/test_wavio.py -q
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyradius.wavio import read_wav, write_wav_f32  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".tmp")
os.makedirs(TMP, exist_ok=True)

CORPUS = [os.path.join(ROOT, "audio_test", f"{i}.wav") for i in (1, 2, 3, 4)]
TD_REFS = sorted(
    os.path.join(TMP, f)
    for f in os.listdir(TMP)
    if f.endswith("_td3_ref.wav") or f.endswith("_td-3_ref.wav")
) if os.path.isdir(TMP) else []


def indie_read(path: str) -> tuple[np.ndarray, int, int, int, int]:
    """Minimal RIFF/WAVE reader written from the spec (independent of pyradius)."""
    with open(path, "rb") as fh:
        b = fh.read()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"{path}: not RIFF/WAVE"
    off, fmt, data = 12, None, None
    while off + 8 <= len(b):
        cid = b[off:off + 4]
        sz = struct.unpack("<I", b[off + 4:off + 8])[0]
        body = b[off + 8:off + 8 + sz]
        if cid == b"fmt ":
            tag, chans, rate, _, _, bps = struct.unpack("<HHIIHH", body[:16])
            fmt = (tag, chans, rate, bps)
        elif cid == b"data":
            data = body
            break
        off += 8 + sz + (sz & 1)
    assert fmt is not None and data is not None, f"{path}: missing fmt/data"
    tag, chans, rate, bps = fmt
    if tag == 3 and bps == 32:
        x = np.frombuffer(data, dtype="<f4").reshape(-1, chans).copy()
    elif tag == 1 and bps == 16:
        x = (np.frombuffer(data, dtype="<i2").reshape(-1, chans).astype(np.float32)
             / np.float32(32768.0))
    elif tag == 1 and bps == 32:
        x = (np.frombuffer(data, dtype="<i4").reshape(-1, chans).astype(np.float64)
             / 2147483648.0).astype(np.float32)
    elif tag == 3 and bps == 64:
        x = np.frombuffer(data, dtype="<f8").reshape(-1, chans).astype(np.float32)
    else:
        raise AssertionError(f"{path}: unhandled fmt tag={tag} bps={bps}")
    return x, rate, chans, tag, bps


@pytest.mark.parametrize("path", [p for p in CORPUS + TD_REFS if os.path.exists(p)],
                         ids=lambda p: os.path.basename(p))
def test_read_wav_matches_independent_parser(path):
    a, sr, nch = read_wav(path)
    b, sr2, nch2, _tag, _bps = indie_read(path)
    assert (sr, nch) == (sr2, nch2)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "read_wav differs from the spec parser"


@pytest.mark.parametrize("path", [p for p in CORPUS + TD_REFS if os.path.exists(p)],
                         ids=lambda p: os.path.basename(p))
def test_write_read_roundtrip_is_bitexact(path):
    a, sr, nch = read_wav(path)
    out = os.path.join(TMP, "pytest_roundtrip.wav")
    try:
        write_wav_f32(out, a, sr, nch)
        c, sr2, nch2 = read_wav(out)
        assert (sr, nch) == (sr2, nch2)
        assert np.array_equal(a, c)
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_written_file_is_f32_and_reparses_identically():
    """Header must declare IEEE float 32-bit (what the C reader expects)."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1000, 2)) * 0.5).astype(np.float32)
    out = os.path.join(TMP, "pytest_f32.wav")
    try:
        write_wav_f32(out, x, 48000, 2)
        b, sr, nch, tag, bps = indie_read(out)
        assert (sr, nch, tag, bps) == (48000, 2, 3, 32)
        assert np.array_equal(b, x)
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_wavio_out_of_process_reproducible():
    """Py-written bytes must be stable across processes (no dict/hash order leaks)."""
    import subprocess

    src = CORPUS[3] if os.path.exists(CORPUS[3]) else None
    if src is None:
        pytest.skip("no corpus wav")
    out = os.path.join(TMP, "pytest_repro.wav")
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from pyradius.wavio import read_wav, write_wav_f32;"
        "x, sr, nc = read_wav(%r);"
        "write_wav_f32(%r, x, sr, nc)" % (ROOT, src, out)
    )
    try:
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
        first = open(out, "rb").read()
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
        assert open(out, "rb").read() == first
    finally:
        if os.path.exists(out):
            os.remove(out)
