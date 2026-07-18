#!/usr/bin/env python3
"""Audit stage 9: CONFIRMATION.
 (1) The minimal-config PSV translation gate vs FD step delta:
     contamination ~1/delta -> ratios must approach 1 for large
     delta if the crisis is leg-amplitude noise.
 (2) Direct exhibit: metre-scale noise floor of source_jumps(b)
     (through system_matrix's internal h=1e-3 radial FD).
"""
import numpy as np
from common import (A, CS, Jvec, LAUD, LMAX, NT, ct, st, delta,
                    m0, mats, proj, r0, tfe, u_line, w, y_of,
                    solve_l, zgrad_proj, _entries, spheroidal_unit,
                    lay)
from pyastroseis import spectral as spsp

zhat = np.array([0.0, 0.0, 1.0])
Mrr = np.diag([0.0, 0.0, 1.0e18])
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
dirs = np.stack([st, 0.0 * st, ct], axis=1)
that = np.stack([ct, 0.0 * ct, -st], axis=1)

print("== (1) PSV gate ratios vs FD step delta ==")
for dl in (30.0, 300.0, 3000.0):
    hLM = dl * np.sqrt(4.0 * np.pi / 3.0)
    du = tfe.relief_spectra(lay, src, [Mrr], np.array([w]), dirs,
                            relief=[("top", 1, 0, hLM)], **kw)
    du_e = (du["psv"] + du["sh"])[0, :, :, 0]
    du_f = 0.0
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * st[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(lay, src - s * dl * zhat, [Mrr],
                                   np.array([w]), dp, l0=False,
                                   **kw)
        du_f = du_f + 0.5 * s * (us["psv"] + us["sh"])[0, :, :, 0]
    Ue, Ve = proj(np.einsum('nc,nc->n', dirs, du_e),
                  np.einsum('nc,nc->n', that, du_e))
    Uf, Vf = proj(np.einsum('nc,nc->n', dirs, du_f),
                  np.einsum('nc,nc->n', that, du_f))
    rats = ["%d:%.3f<%.0f/%.3f<%.0f"
            % (lp, abs(Ue[lp] / Uf[lp]),
               np.degrees(np.angle(Ue[lp] / Uf[lp])),
               abs(Ve[lp] / Vf[lp]),
               np.degrees(np.angle(Ve[lp] / Vf[lp])))
            for lp in range(1, LAUD + 1)]
    print(" delta=%5.0f  eng/fd U/V: %s" % (dl, "  ".join(rats)))

print("\n== (2) noise floor of J(b) at metre scales (lp=3) ==")
J0 = Jvec(3, r0)
print("   db          |dJ_R/J_R|      |dJ_S/J_S|   (J(b+db)-J(b))")
for db in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
    J1 = Jvec(3, r0 + db)
    dR = abs((J1[2] - J0[2]) / J0[2])
    dS = abs((J1[3] - J0[3]) / J0[3])
    print("%8.3f     %.3e      %.3e" % (db, dR, dS))
print("(smooth family: |dJ/J| ~ db/r0 = db * %.1e; a flat floor "
      "at small db = FD-roundoff noise)" % (1.0 / r0))
