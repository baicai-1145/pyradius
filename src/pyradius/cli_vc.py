"""pyradius.cli_vc — vocoder renderer CLI mirroring tools/rx_vc_render.c.

Usage: python -m pyradius.cli_vc <in.wav> <out.wav> <semis> [precision=2] [tracefile]
"""
from __future__ import annotations

import sys
import time
import numpy as np

from .wavio import read_wav, write_wav_f32
from .vocoder_core import VocoderState, vc_render


def main() -> int:
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <in.wav> <out.wav> <semis> [precision=2] [tracefile]",
              file=sys.stderr)
        return 2
    inpath, outpath = sys.argv[1], sys.argv[2]
    semis = float(sys.argv[3])
    precision = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    tracefile = sys.argv[5] if len(sys.argv) > 5 else None

    x, sr, nch = read_wav(inpath)
    if sr not in (48000, 44100):
        print(f"vocoder: unsupported sr={sr}", file=sys.stderr)
        return 1
    nframes = x.shape[0]
    print(f"in: {sr}Hz {nch}ch {nframes} frames")

    st = VocoderState(sr, nch, precision)
    st.set_ratio(semis, 100.0)

    t0 = time.perf_counter()
    out, stats = vc_render(st, x, trace=tracefile)
    dt = time.perf_counter() - t0
    on = out.shape[0]
    print(f"render: out={on} frames ({on/sr:.3f}s) cursor={stats['cursor']} pos={stats['pos']} "
          f"made={stats['made']}  [{dt*1000:.0f} ms, {nframes/dt:.1f}x RT]")
    write_wav_f32(outpath, out, sr, nch)
    print(f"wrote {outpath} (f32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
