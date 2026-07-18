#!/usr/bin/env python3
"""Generate a tipsv .inf for the EXACT 127-layer staircase model our
spectral solver runs (one constant zone per ak135 polynomial zone,
midpoint-evaluated) — apples-to-apples arbitration of the low-k
discrepancy. Header, source, stations, spc paths copied verbatim
from the campaign tipsv_mrr.inf (source line keeps the DSM-internal
distorted latitude)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402

lines = open(bc.INF).read().splitlines()
nzone = int(lines[7].split()[0])
tail = lines[8 + 6 * nzone:]
assert tail[0].split()[0] == "6321", "tail alignment lost"

layers = bc.build_layers(1.0e9)                     # 1 layer/zone
assert len(layers) == nzone
out = lines[:8]
r_lo = 0.0
for e in layers:
    r_hi = e["r_top"] / 1e3
    out.append("  %.6f %.6f %.9f 0 0 0"
               % (r_lo, r_hi, e["rho"] / 1e3))
    for _ in range(2):
        out.append("    %.9f 0 0 0" % (e["vp"] / 1e3))
    for _ in range(2):
        out.append("    %.9f 0 0 0" % (e["vs"] / 1e3))
    out.append("    1 0 0 0 -1 -1")
    r_lo = r_hi
assert abs(r_lo - 6371.0) < 1e-6
out += tail
dst = os.path.join(HERE, "stair_dsm", "stair127.inf")
os.makedirs(os.path.join(HERE, "stair_dsm", "spc"), exist_ok=True)
open(dst, "w").write("\n".join(out) + "\n")
print("wrote", dst, "(%d zones)" % nzone)
