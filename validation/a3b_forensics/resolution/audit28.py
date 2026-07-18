#!/usr/bin/env python3
"""Audit 28 CAPSTONE: the ORIGINAL gate config (3-layer LAYERS,
Mrt, relief on ALL boundaries C/B/top) with the two reference
fixes: delta=3000 (suppresses J(b) FD-noise) and TOTAL-field
comparison (parity split does not commute with station mapping).
Also shown: per-part errors (the artifact) and delta=30 (noise)."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, B, C, IC, OC, SH, wk

LMAX = 12
LAYERS = [dict(r_top=C, **IC), dict(r_top=B, **OC),
          dict(r_top=A, **SH)]
r0 = A - 637.0e3
w = wk(90.0)
zhat = np.array([0.0, 0.0, 1.0])
Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
th = np.linspace(0.35, np.pi - 0.35, 9)
dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
that = np.stack([np.cos(th), 0.0 * th, -np.sin(th)], axis=1)

for dl in (30.0, 3000.0):
    c10 = dl * np.sqrt(4.0 * np.pi / 3.0)
    du_fd = {}
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * np.sin(th)[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(LAYERS, src - s * dl * zhat,
                                   [Mrt], np.array([w]), dp,
                                   l0=False, **kw)
        for part in ("psv", "sh"):
            du_fd[part] = du_fd.get(part, 0.0) \
                + 0.5 * s * us[part][0, :, :, 0]
    du = tfe.relief_spectra(LAYERS, src, [Mrt], np.array([w]),
                            dirs, relief=[(C, 1, 0, c10),
                                          (B, 1, 0, c10),
                                          ("top", 1, 0, c10)],
                            **kw)
    tot_e = (du["psv"] + du["sh"])[0, :, :, 0]
    tot_f = du_fd["psv"] + du_fd["sh"]
    e_tot = (np.abs(tot_e - tot_f).max() / np.abs(tot_f).max())
    line = "delta=%5.0f  TOTAL rel err %.3e   (per-part:" % (dl,
                                                             e_tot)
    for part in ("psv", "sh"):
        e = (np.abs(du[part][0, :, :, 0] - du_fd[part]).max()
             / np.abs(du_fd[part]).max())
        line += " %s %.2e" % (part, e)
    print(line + ")")
