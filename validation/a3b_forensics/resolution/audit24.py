#!/usr/bin/env python3
"""Audit 24: piecewise comparison of my-du_fd vs X at m=1.
 map-only:   0.5[u0(A dp+) - u0(A dp-)]      vs (w + adv)
 shift-only: 0.5[u_m(A d) - u_p(A d)]        vs vsrc
"""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit21.py").read().split("def run_fit")[0])

DL = 3000.0
PS0 = {l: psv_solve(l, R0) for l in range(1, LMAX + 1)}
TS0 = {l: tor_solve(l, R0) for l in range(1, LMAX + 1)}
PSm = {l: psv_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
TSm = {l: tor_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
PSp = {l: psv_solve(l, R0 + DL) for l in range(1, LMAX + 1)}
TSp = {l: tor_solve(l, R0 + DL) for l in range(1, LMAX + 1)}

rh_g = np.stack([st, 0 * st, ct], axis=1)
th_g = np.stack([ct, 0 * ct, -st], axis=1)
ph_g = np.array([0.0, 1.0, 0.0])

# ---- map-only field on the theta line (phi=0) -------------------
map_f = np.zeros((NT, 3), dtype=complex)
shift_f = np.zeros((NT, 3), dtype=complex)
for i in range(NT):
    d = rh_g[i]
    for s in (+1.0, -1.0):
        dp = d + s * (DL / A) * st[i] * th_g[i]
        dp = dp / np.linalg.norm(dp)
        map_f[i] += 0.5 * s * u_point(A * dp, PS0, TS0)
    shift_f[i] = 0.5 * (u_point(A * d, PSm, TSm)
                        - u_point(A * d, PSp, TSp))

# ---- X pieces ---------------------------------------------------
eps = 2.0
zh = np.array([0.0, 0.0, eps])
wadv_f = np.zeros((NT, 3), dtype=complex)
for i in range(NT):
    x = A * rh_g[i]
    dz = (u_point(x + zh, PS0, TS0)
          - u_point(x - zh, PS0, TS0)) / (2 * eps)
    dr = (u_point(x * (A + eps) / A, PS0, TS0)
          - u_point(x * (A - eps) / A, PS0, TS0)) / (2 * eps)
    wadv_f[i] = -DL * dz + DL * ct[i] * dr

def pj(f):
    return proj_disp(np.einsum('nc,nc->n', rh_g, f),
                     np.einsum('nc,nc->n', th_g, f),
                     f[:, 1])

Uma, Vma, Wma = pj(map_f)
Uxa, Vxa, Wxa = pj(wadv_f)
Ush, Vsh, Wsh = pj(shift_f)
# vsrc coefficient route
Uv = np.zeros(LMAX + 2, dtype=complex)
Vv = np.zeros(LMAX + 2, dtype=complex)
Wv = np.zeros(LMAX + 2, dtype=complex)
for l in range(1, LMAX + 1):
    ym = y4_of(l, PSm[l], A) * CSr[l]
    yp = y4_of(l, PSp[l], A) * CSr[l]
    Uv[l] = 0.5 * (ym[0] - yp[0])
    Vv[l] = 0.5 * (ym[1] - yp[1])
    wm = wt_of(l, TSm[l], A) * CSt[l]
    wp = wt_of(l, TSp[l], A) * CSt[l]
    Wv[l] = 0.5 * (wm[0] - wp[0])

print("lp   map/(w+adv) U   map/(w+adv) W  | shift/vsrc U"
      "   shift/vsrc W")
for lp in range(2, LAUD + 1):
    r1 = Uma[lp] / Uxa[lp]
    r2 = Wma[lp] / Wxa[lp]
    r3 = Ush[lp] / Uv[lp]
    r4 = Wsh[lp] / Wv[lp]
    print("%2d  %8.4f<%5.0f %8.4f<%5.0f | %8.4f<%5.0f"
          " %8.4f<%5.0f"
          % (lp, abs(r1), np.degrees(np.angle(r1)), abs(r2),
             np.degrees(np.angle(r2)), abs(r3),
             np.degrees(np.angle(r3)), abs(r4),
             np.degrees(np.angle(r4))))
