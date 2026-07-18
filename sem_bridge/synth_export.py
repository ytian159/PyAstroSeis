#!/usr/bin/env python3
"""npz spectra -> per-station cartesian displacement traces.

usage: synth_export.py <npz> <outdir> [conj(0|1, default 1)]

Writes Uxyz_spectral_AZnn.txt (t ux uy uz, dt 0.25 s to T_OUT) in
the DFDM Uxyz layout so the bridge comparator can project with the
standard geographic basis. conj=1 maps our conj-of-DSM spectrum
convention into DSM space (the campaign-established relation);
the comparator QC step verifies the choice against SEM.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_common as bc                          # noqa: E402

NPZ, OUTDIR = sys.argv[1], sys.argv[2]
CONJ = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True

z = np.load(NPZ)
u = z["u"][0]                                       # (18, 3, IMAX+1)
os.makedirs(OUTDIR, exist_ok=True)
dt = bc.TLEN / bc.NFFT
nout = int(bc.T_OUT / dt) + 1
stride = max(1, int(round(0.25 / dt)))
for i, sta in enumerate(bc.STATIONS):
    t, wav = bc.synth_time(u[i], CONJ)
    sel = np.arange(0, nout, stride)
    arr = np.column_stack([t[sel], wav[0][sel], wav[1][sel],
                           wav[2][sel]])
    out = os.path.join(OUTDIR, "Uxyz_spectral_%s.txt" % sta)
    np.savetxt(out, arr, fmt="%.6f %.9e %.9e %.9e")
print("wrote %d stations to %s (conj=%d)"
      % (len(bc.STATIONS), OUTDIR, CONJ))
