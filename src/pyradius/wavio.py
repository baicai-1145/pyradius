"""pyradius.wavio — RIFF/WAVE IO matching libradius renderers exactly.

Read: f32(tag3/bps32), i16(tag1/16), i32(tag1/32), f64(tag3/64) → float32.
Vocoder accepts only f32. Output is always f32. Frames counted from data
chunk size (trailing chunks skipped), not EOF.
"""
from __future__ import annotations

import struct
import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int, int]:
    """Return (interleaved float32 [frames, ch], sr, nch). Mirrors C wav_read_f32."""
    with open(path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) != 12 or hdr[0:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise ValueError(f"{path}: not RIFF/WAVE")
        tag = chans = bps = 0
        rate = 0
        doff = -1
        dsz = 0
        while True:
            ch = f.read(8)
            if len(ch) != 8:
                break
            cid = ch[0:4]
            sz = struct.unpack("<I", ch[4:8])[0]
            if cid == b"fmt ":
                b = f.read(min(sz, 40))
                tag, chans, rate, _, _, bps = struct.unpack("<HHIIHH", b[:16])
                if sz > 40:
                    f.seek(sz - 40, 1)
            elif cid == b"data":
                doff = f.tell()
                dsz = sz
                break
            else:
                f.seek(sz + (sz & 1), 1)
        if doff < 0 or chans == 0 or rate == 0:
            raise ValueError(f"{path}: bad fmt")
        frames = dsz // (chans * (bps // 8))
        f.seek(doff)
        n = frames * chans
        if tag == 3 and bps == 32:
            raw = f.read(4 * n)
            x = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        elif tag == 1 and bps == 16:
            raw = f.read(2 * n)
            x = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)).astype(np.float32)
        elif tag == 1 and bps == 32:
            raw = f.read(4 * n)
            x = (np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
        elif tag == 3 and bps == 64:
            raw = f.read(8 * n)
            x = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"{path}: unsupported tag={tag} bps={bps}")
        return x.reshape(frames, chans), rate, chans


def write_wav_f32(path: str, x: np.ndarray, sr: int, nch: int) -> None:
    """Write interleaved float32 WAV. Mirrors C wav_write_f32."""
    if x.ndim == 2:
        data = np.ascontiguousarray(x, dtype="<f4")
        n = data.shape[0]
    else:
        data = np.ascontiguousarray(x.reshape(-1, nch), dtype="<f4")
        n = data.shape[0]
    dsz = n * nch * 4
    hdr = b"RIFF" + struct.pack("<I", 36 + dsz) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 3, nch, sr, sr * nch * 4, nch * 4, 32)
    hdr += b"data" + struct.pack("<I", dsz)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data.tobytes())


def read_wav_f32_only(path: str) -> tuple[np.ndarray, int, int]:
    """Vocoder reader: f32 only, error otherwise."""
    x, sr, nch = read_wav(path)
    return x, sr, nch
