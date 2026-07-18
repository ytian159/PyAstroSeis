#!/usr/bin/env python3
"""Audit 25: projection-pipeline cross-check on the SAME du_fd
field: theta-line (pure-m assumption) vs (theta,phi)-grid."""
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

# du_fd on the theta line via u_point
fdl = np.zeros((NT, 3), dtype=complex)
for i in range(NT):
    d = rh_g[i]
    for s, PSx, TSx in ((+1.0, PSm, TSm), (-1.0, PSp, TSp)):
        dp = d + s * (DL / A) * st[i] * th_g[i]
        dp = dp / np.linalg.norm(dp)
        fdl[i] += 0.5 * s * u_point(A * dp, PSx, TSx)
U1p, V1p, W1p = proj_disp(np.einsum('nc,nc->n', rh_g, fdl),
                          np.einsum('nc,nc->n', th_g, fdl),
                          fdl[:, 1])

# same field on a (theta,phi) grid, coeffs2-projected
NPH = 8
pg = 2.0 * np.pi * np.arange(NPH) / NPH
TH, PH = np.meshgrid(np.arccos(xg), pg, indexing="ij")
dirs2 = np.stack([np.sin(TH) * np.cos(PH),
                  np.sin(TH) * np.sin(PH), np.cos(TH)],
                 axis=-1).reshape(-1, 3)
thv, phv = TH.ravel(), PH.ravel()
that2 = np.stack([np.cos(thv) * np.cos(phv),
                  np.cos(thv) * np.sin(phv), -np.sin(thv)], axis=1)
phat2 = np.stack([-np.sin(phv), np.cos(phv), 0.0 * phv], axis=1)
wq = np.repeat(wg, NPH) * (2.0 * np.pi / NPH)


def rep2(a):
    return np.repeat(a, NPH, axis=1).reshape(a.shape[0], -1)


Y1g, Bt1g, Bp1g = rep2(Sh["Y"]), rep2(Sh["Bt"]), rep2(Sh["Bp"])
Ct1g, Cp1g = rep2(Sh["Ct"]), rep2(Sh["Cp"])
eip = np.exp(1j * phv)
fd2 = np.zeros((len(dirs2), 3), dtype=complex)
for i in range(len(dirs2)):
    d = dirs2[i]
    for s, PSx, TSx in ((+1.0, PSm, TSm), (-1.0, PSp, TSp)):
        dp = d + s * (DL / A) * np.sin(thv[i]) * that2[i]
        dp = dp / np.linalg.norm(dp)
        fd2[i] += 0.5 * s * u_point(A * dp, PSx, TSx)
ur = np.einsum('nc,nc->n', dirs2, fd2)
ut = np.einsum('nc,nc->n', that2, fd2)
up = np.einsum('nc,nc->n', phat2, fd2)
U2p = np.einsum('ln,n->l', np.conj(Y1g * eip), ur * wq)
V2p = (np.einsum('ln,n->l', np.conj(Bt1g * eip), ut * wq)
       + np.einsum('ln,n->l', np.conj(Bp1g * eip), up * wq)) / Lp
W2p = (np.einsum('ln,n->l', np.conj(Ct1g * eip), ut * wq)
       + np.einsum('ln,n->l', np.conj(Cp1g * eip), up * wq))
print("lp  line/grid U      line/grid V      line/grid W")
for lp in range(2, LAUD + 1):
    r1 = U1p[lp] / U2p[lp]
    r2 = V1p[lp] / V2p[lp]
    r3 = W1p[lp] / W2p[lp]
    print("%2d %8.4f<%5.0f %8.4f<%5.0f %8.4f<%5.0f"
          % (lp, abs(r1), np.degrees(np.angle(r1)), abs(r2),
             np.degrees(np.angle(r2)), abs(r3),
             np.degrees(np.angle(r3))))
