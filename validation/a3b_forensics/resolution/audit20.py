#!/usr/bin/env python3
"""Audit 20: m=1 sigma builder vs pure displacement-FD sigma
(independent: no shape/strain formulas — only pointwise u(x))."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit15.py").read().split("# TRUTH")[0])

def u_point(x):
    """pointwise m=1 displacement (cartesian) at x, PSV+SH."""
    r = np.linalg.norm(x)
    cth = np.clip(x[2] / r, -1, 1)
    sth = np.hypot(x[0], x[1]) / r
    ph_ = np.arctan2(x[1], x[0])
    Sh_ = tfe._shapes(1, LMAX + 1, np.array([cth]),
                      np.array([sth]))
    e1 = np.exp(1j * ph_)
    ur = ut = up = 0j
    for l in range(1, LMAX + 1):
        y4 = y4_at(l, r)
        wt = wt_at(l, r)
        ur += y4[0] * Sh_["Y"][l][0] * e1
        ut += (y4[1] * Sh_["Bt"][l][0]
               + wt[0] * Sh_["Ct"][l][0]) * e1
        up += (y4[1] * Sh_["Bp"][l][0]
               + wt[0] * Sh_["Cp"][l][0]) * e1
    rh = np.array([sth * np.cos(ph_), sth * np.sin(ph_), cth])
    th = np.array([cth * np.cos(ph_), cth * np.sin(ph_), -sth])
    p2 = np.array([-np.sin(ph_), np.cos(ph_), 0.0])
    return ur * rh + ut * th + up * p2

def sigma_fd(x, eps=5.0):
    G = np.zeros((3, 3), dtype=complex)
    for q in range(3):
        dq = np.zeros(3); dq[q] = eps
        G[:, q] = (u_point(x + dq) - u_point(x - dq)) / (2 * eps)
    E = 0.5 * (G + G.T)
    return lam * np.trace(E) * np.eye(3) + 2.0 * mu * E

NTc = 5
print("point  comp   sigma_builder     sigma_dispFD      ratio")
for th_ in (0.7, 1.3, 2.2):
    r = A - 1000.0
    ci, si = np.cos(th_), np.sin(th_)
    x = r * np.array([si, 0.0, ci])
    Shp = tfe._shapes(1, LMAX + 1, np.array([ci]), np.array([si]))
    sigS = sigma_sph(r, Shp, 0)
    rh = np.array([si, 0.0, ci])
    thv_ = np.array([ci, 0.0, -si])
    p2 = np.array([0.0, 1.0, 0.0])
    Q = np.stack([rh, thv_, p2])
    sigB = Q.T @ sigS @ Q          # cartesian from builder
    sigF = sigma_fd(x)
    for (a, b, nm) in ((0, 0, "xx"), (0, 2, "xz"), (1, 1, "yy"),
                       (0, 1, "xy"), (2, 2, "zz"), (1, 2, "yz")):
        rB, rf = sigB[a, b], sigF[a, b]
        rr = rB / rf if rf != 0 else np.nan
        print("th=%.1f  %s  %11.4e%+9.1ej %11.4e%+9.1ej  "
              "%7.4f<%5.1f" % (th_, nm, rB.real, rB.imag,
                               rf.real, rf.imag, abs(rr),
                               np.degrees(np.angle(rr))))
