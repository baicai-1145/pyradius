"""pyradius.cli_td — TD renderer CLI mirroring tools/rx_td_render.c.

Usage: python -m pyradius.cli_td <in.wav> <out.wav> <semis> [quality=37] [solo=0] [truth.wav|-]
"""
from __future__ import annotations

import sys
import time
import numpy as np

from .wavio import read_wav, write_wav_f32
from .td_core import TDState, td_render


def main() -> int:
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <in.wav> <out.wav> <semis> [quality=37] [solo=0] [truth.wav|-]",
              file=sys.stderr)
        return 2
    inpath, outpath = sys.argv[1], sys.argv[2]
    semis = float(sys.argv[3])
    quality = int(sys.argv[4]) if len(sys.argv) > 4 else 37
    solo = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    truth = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] not in ("", "-") else None

    x, sr, nch = read_wav(inpath)
    nframes = x.shape[0]
    print(f"in : {inpath}  sr={sr} ch={nch} frames={nframes} ({nframes/sr:.3f}s)")

    st = TDState(sr, quality, solo, nch)
    st.set_ratio(semis, 100.0)
    print(f"cfg: hop={st.hop} f28={st.f28} win_max={st.win_max} ratio={st.total_ratio:.12f} (semis={semis:.7f})")

    t0 = time.perf_counter()
    out, stats = td_render(st, x)
    dt = time.perf_counter() - t0
    on = out.shape[0]
    print(f"render: out={on} frames ({on/sr:.3f}s)  granule={stats['n_granule']} "
          f"transient={stats['n_transient']} wrap={stats['wrap_cnt']}  [{dt*1000:.0f} ms, {nframes/dt:.1f}x RT]")
    write_wav_f32(outpath, out, sr, nch)
    print(f"wrote {outpath}")
    if truth:
        try:
            t, _, tch = read_wav(truth)
            n = min(on, t.shape[0])
            a = out[:n].reshape(-1).astype(np.float64)
            b = t[:n].reshape(-1).astype(np.float64)
            ma, mb = a.mean(), b.mean()
            sa = ((a - ma) ** 2).sum()
            sb = ((b - mb) ** 2).sum()
            sab = ((a - ma) * (b - mb)).sum()
            r = sab / np.sqrt(sa * sb) if sa > 0 and sb > 0 else 0.0
            print(f"CORR vs gold: n={n} (len {on} vs {t.shape[0]})  corr={r:.8f}  "
                  f"rmsA={np.sqrt(sa/n):.6f} rmsB={np.sqrt(sb/n):.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"truth read failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
