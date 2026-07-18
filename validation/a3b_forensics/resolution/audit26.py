#!/usr/bin/env python3
"""Audit 26: one namespace, all routes, lp=3..5, m=1, delta=3000."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit21.py").read().split("def run_fit")[0])
from pyastroseis import spectral as spsp
from pyastroseis import spectral_tfe as tfe2

DL = 3000.0
PS0 = {l: psv_solve(l, R0) for l in range(1, LMAX + 1)}
TS0 = {l: tor_solve(l, R0) for l in range(1, LMAX + 1)}
PSm = {l: psv_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
TSm = {l: tor_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
PSp = {l: psv_solve(l, R0 + DL) for l in range(1, LMAX + 1)}
TSp = {l: tor_solve(l, R0 + DL) for l in range(1, LMAX + 1)}
rh_g = np.stack([st, 0 * st, ct], axis=1)
th_g = np.stack([ct, 0 * ct, -st], axis=1)

# route 1: du_fd from u_point, theta line
fdl = np.zeros((NT, 3), dtype=complex)
for i in range(NT):
    d = rh_g[i]
    for s, PSx, TSx in ((+1.0, PSm, TSm), (-1.0, PSp, TSp)):
        dp = d + s * (DL / A) * st[i] * th_g[i]
        dp = dp / np.linalg.norm(dp)
        fdl[i] += 0.5 * s * u_point(A * dp, PSx, TSx)
Ufd_l, Vfd_l, Wfd_l = proj_disp(
    np.einsum('nc,nc->n', rh_g, fdl),
    np.einsum('nc,nc->n', th_g, fdl), fdl[:, 1])

# route 2: code legs on the SAME theta line (phi=0 dirs)
Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, R0])
zhat = np.array([0.0, 0.0, 1.0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
fdc = 0.0
for s in (+1.0, -1.0):
    dp = rh_g + s * (DL / A) * st[:, None] * th_g
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra([dict(r_top=A, **SH)],
                               src - s * DL * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    fdc = fdc + 0.5 * s * (us["psv"] + us["sh"])[0, :, :, 0]
Ufd_c, Vfd_c, Wfd_c = proj_disp(
    np.einsum('nc,nc->n', rh_g, fdc),
    np.einsum('nc,nc->n', th_g, fdc), fdc[:, 1])

# route 3: engine on the theta line
du = tfe2.relief_spectra([dict(r_top=A, **SH)], src, [Mrt],
                         np.array([w]), rh_g,
                         relief=[("top", 1, 0,
                                  DL * np.sqrt(4 * np.pi / 3))],
                         **kw)
de = (du["psv"] + du["sh"])[0, :, :, 0]
Ue_l, Ve_l, We_l = proj_disp(
    np.einsum('nc,nc->n', rh_g, de),
    np.einsum('nc,nc->n', th_g, de), de[:, 1])

for lp in (3, 4, 5):
    print("lp=%d U: mypt %12.5e  code %12.5e  eng %12.5e"
          % (lp, abs(Ufd_l[lp]), abs(Ufd_c[lp]), abs(Ue_l[lp])))
    print("      ratios mypt/code %.4f  eng/code %.4f"
          % (abs(Ufd_l[lp] / Ufd_c[lp]),
             abs(Ue_l[lp] / Ufd_c[lp])))
