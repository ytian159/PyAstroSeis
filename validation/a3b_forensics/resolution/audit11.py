#!/usr/bin/env python3
"""Audit stage 11: the ACTUAL gate config (Mrt, m=+-1, both
parities) on the homog ball at delta = 30 vs 3000 m, lmax=12.
Expected if the verdict is right: PSV error collapses by ~2 orders
at delta=3000 (residual limited by leg noise/edge truncation)."""
import os
os.environ["PARADOX_LMAX"] = "12"
import numpy as np
from common import A, LMAX, r0, tfe, w, lay
from pyastroseis import spectral as spsp

zhat = np.array([0.0, 0.0, 1.0])
Mrt = np.zeros((3, 3))
Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
th = np.linspace(0.35, np.pi - 0.35, 9)
dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
that = np.stack([np.cos(th), 0.0 * th, -np.sin(th)], axis=1)

for dl in (30.0, 3000.0):
    hLM = dl * np.sqrt(4.0 * np.pi / 3.0)
    du_fd = {}
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * np.sin(th)[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(lay, src - s * dl * zhat,
                                   [Mrt], np.array([w]), dp,
                                   l0=False, **kw)
        for part in ("psv", "sh"):
            du_fd[part] = du_fd.get(part, 0.0) \
                + 0.5 * s * us[part][0, :, :, 0]
    du = tfe.relief_spectra(lay, src, [Mrt], np.array([w]), dirs,
                            relief=[("top", 1, 0, hLM)], **kw)
    print("delta=%5.0f m:" % dl)
    for part in ("psv", "sh"):
        e = (np.abs(du[part][0, :, :, 0] - du_fd[part]).max()
             / np.abs(du_fd[part]).max())
        print("   %s: engine vs FD rel err %.3e" % (part, e))
