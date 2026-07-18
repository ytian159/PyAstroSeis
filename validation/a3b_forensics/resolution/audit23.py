#!/usr/bin/env python3
"""Audit 23: my m=+1 pointwise legs at the MAPPED directions vs
the code's spectral_spectra legs. If my-du_fd == X != code-du_fd,
the code legs' m=+-1 evaluation differs from the true field of the
same solves."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit21.py").read().split("def run_fit")[0])
from pyastroseis import spectral as spsp

DL = 3000.0
PS = {}
TS = {}
for tag, b in (("0", R0), ("m", R0 - DL), ("p", R0 + DL)):
    PS[tag] = {l: psv_solve(l, b) for l in range(1, LMAX + 1)}
    TS[tag] = {l: tor_solve(l, b) for l in range(1, LMAX + 1)}

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


def coeffs2(u):
    ur = np.einsum('nc,nc->n', dirs2, u)
    ut = np.einsum('nc,nc->n', that2, u)
    upp = np.einsum('nc,nc->n', phat2, u)
    U = np.einsum('ln,n->l', np.conj(Y1g * eip), ur * wq)
    V = (np.einsum('ln,n->l', np.conj(Bt1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Bp1g * eip), upp * wq))
    W = (np.einsum('ln,n->l', np.conj(Ct1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Cp1g * eip), upp * wq))
    return U, V / Lp, W


# my m=+1 pointwise du_fd at the mapped directions
my_fd = np.zeros((len(dirs2), 3), dtype=complex)
for s, tag in ((+1.0, "m"), (-1.0, "p")):
    dp = dirs2 + s * (DL / A) * np.sin(thv)[:, None] * that2
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    for i in range(len(dirs2)):
        my_fd[i] += 0.5 * s * u_point(A * dp[i], PS[tag], TS[tag])
Um, Vm, Wm = coeffs2(my_fd)

# code du_fd
Mrt = np.zeros((3, 3))
Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, R0])
zhat = np.array([0.0, 0.0, 1.0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
code_fd = 0.0
for s in (+1.0, -1.0):
    dp = dirs2 + s * (DL / A) * np.sin(thv)[:, None] * that2
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra([dict(r_top=A, **SH)],
                               src - s * DL * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    code_fd = code_fd + 0.5 * s * (us["psv"]
                                   + us["sh"])[0, :, :, 0]
Uc, Vc, Wc = coeffs2(code_fd)

# base-field sanity: my u0 vs code u0 at the UNMAPPED dirs
u0_code = spsp.spectral_spectra([dict(r_top=A, **SH)], src, [Mrt],
                                np.array([w]), dirs2, l0=False,
                                **kw)
u0c = (u0_code["psv"] + u0_code["sh"])[0, :, :, 0]
u0m = np.zeros((len(dirs2), 3), dtype=complex)
for i in range(len(dirs2)):
    u0m[i] = u_point(A * dirs2[i], PS["0"], TS["0"])
U0m, V0m, W0m = coeffs2(u0m)
U0c, V0c, W0c = coeffs2(u0c)

print("lp   u0 my/code U    my/code W   |  dU my/code"
      "     dW my/code")
for lp in range(2, LAUD + 1):
    r0_ = U0m[lp] / U0c[lp]
    r1_ = W0m[lp] / W0c[lp]
    r2_ = Um[lp] / Uc[lp]
    r3_ = Wm[lp] / Wc[lp]
    print("%2d  %8.5f<%5.1f %8.5f<%5.1f | %8.5f<%5.1f"
          " %8.5f<%5.1f"
          % (lp, abs(r0_), np.degrees(np.angle(r0_)), abs(r1_),
             np.degrees(np.angle(r1_)), abs(r2_),
             np.degrees(np.angle(r2_)), abs(r3_),
             np.degrees(np.angle(r3_))))

print("RAW a23: Um[4]=%r Uc[4]=%r" % (Um[4], Uc[4]))
np.save("a23_myfd.npy", my_fd)
np.save("a23_codefd.npy", code_fd)
