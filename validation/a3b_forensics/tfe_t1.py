#!/usr/bin/env python3
"""Three-way t1 test (mrr on-axis, m=0, homog ball, Y10 top):
  A) engine rows (jR, jS)
  B) t1 = -delta (dz sigma0).rhat via SPATIAL FD of the full
     sigma0 tensor built from solver bundles (no perturbation
     formula involved)
  C) traction reconstructed from the FD-true u1 displacement
"""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from pyastroseis.spectral import (_entries, _mats_at, make_stack,
                                  spheroidal_unit, _Ysolid)
from pyastroseis.spheroidal_ref import source_jumps, spheroidal_pole
from tests.test_spectral import A, SH, wk

QSH = 50.0
LMAX = 20
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
stack = make_stack(lay)
mats = _mats_at(stack, w, QSH, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
src = np.array([0.0, 0.0, r0])
delta = 30.0
DY, DG = spheroidal_pole(LMAX)

# per-l solved coefficient state (y-tag, mrr): keep solver x via
# full=True info closures -> use _yv-style evaluation through
# spheroidal_unit's info? Simpler: re-solve and keep coefficients
# by copying the essential solve here for the homog 2-entry stack.
sols = {}
for l in range(1, LMAX + 1):
    ents = _entries(mats, 0, r0, 0)
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    J = source_jumps(l, w, r0, rho, lam, mu, F0y, F1y)
    # replicate solve to keep x: build system as spheroidal_unit
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        Yb.append(_Ysolid(l, w, e["r_bot"], e["mat"], kinds, None)
                  if i > 0 else None)
        Yt.append(_Ysolid(l, w, e["r_top"], e["mat"], kinds, None))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    Am = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        clo = slice(ofs[i], ofs[i + 1])
        chi = slice(ofs[i + 1], ofs[i + 2])
        Am[row:row + 4, chi] = Yb[i + 1]
        Am[row:row + 4, clo] = -Yt[i]
        if ents[i]["src_top"]:
            b[row:row + 4] = J
        row += 4
    ctop = slice(ofs[-2], ofs[-1])
    Am[row, ctop] = Yt[-1][2]
    Am[row + 1, ctop] = Yt[-1][3]
    sc = np.max(np.abs(Am), axis=0)
    sc[sc == 0] = 1.0
    x = np.linalg.solve(Am / sc, b) / sc
    cu = x[ofs[-2]:ofs[-1]]                      # top-entry coeffs
    sols[l] = cu

CS = {l: 1.0e18 * DY[l] for l in range(1, LMAX + 1)}


def y_at(l, r):
    return _Ysolid(l, w, r, m0, ("j", "y"), None) @ sols[l] * CS[l]


def sigma_cart(x):
    """Full cartesian stress tensor of u0 at point x (m=0 field)."""
    r = np.linalg.norm(x)
    th = np.arccos(x[2] / r)
    ph = np.arctan2(x[1], x[0])
    ct, st = np.cos(th), np.sin(th)
    S = tfe._shapes(0, LMAX, np.array([ct]), np.array([st]))
    sig_sph = np.zeros((3, 3), dtype=complex)   # r,t,p basis
    h = 0.5
    for l in range(1, LMAX + 1):
        U, V, R, Ssurf = y_at(l, r)
        yp = (y_at(l, r + h) - y_at(l, r - h)) / (2.0 * h)
        Y = S["Y"][l][0]
        Bt = S["Bt"][l][0]
        eBtt, eBpp, eBtp = (S["eB"][0][l][0], S["eB"][1][l][0],
                            S["eB"][2][l][0])
        L = l * (l + 1.0)
        div = yp[0] + 2.0 * U / r - L * V / r
        ciso = lam * div + 2.0 * mu * U / r
        cV = mu * V / r
        sig_sph[0, 0] += R * Y
        sig_sph[0, 1] += Ssurf * Bt
        sig_sph[1, 0] += Ssurf * Bt
        sig_sph[1, 1] += ciso * Y + cV * eBtt
        sig_sph[2, 2] += ciso * Y + cV * eBpp
        sig_sph[1, 2] += cV * eBtp
        sig_sph[2, 1] += cV * eBtp
        # sigma_rp, tp have phi-components ~ m=0 -> Bp = 0: fine
    rh = np.array([st * np.cos(ph), st * np.sin(ph), ct])
    that = np.array([ct * np.cos(ph), ct * np.sin(ph), -st])
    phat = np.array([-np.sin(ph), np.cos(ph), 0.0])
    Q = np.stack([rh, that, phat])              # rows = basis
    return Q.T @ sig_sph @ Q


# B) spatial-FD t1 on a theta line (phi = 0)
NT = 24
tg = np.linspace(0.25, np.pi - 0.25, NT)
eps = 2.0
t1B = np.zeros((NT, 3), dtype=complex)
for i, th in enumerate(tg):
    xs = A * np.array([np.sin(th), 0.0, np.cos(th)])
    dz = np.array([0.0, 0.0, eps])
    dsig = (sigma_cart(xs + dz) - sigma_cart(xs - dz)) / (2 * eps)
    rh = xs / A
    t1B[i] = -delta * (dsig @ rh)

# A) engine rows -> pointwise field on the same line
keys = ("Up", "Vp", "Rp", "Sp", "Wp", "Tp", "S", "T", "V", "W",
        "ciso", "cV", "cW")
dc = {k: np.zeros(LMAX + 1, dtype=complex) for k in keys}
for l in range(1, LMAX + 1):
    ents = _entries(mats, 0, r0, 0)
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    J = source_jumps(l, w, r0, rho, lam, mu, F0y, F1y)
    Ua, Va, info = spheroidal_unit(l, w, ents, J, full=True)
    sc_ = tfe._side_coeffs(l, w, A, m0, info["surface"] * CS[l],
                           None, yp4=info["surface_p"] * CS[l])
    for k in keys:
        dc[k][l] = sc_[k]
hLM = delta * np.sqrt(4.0 * np.pi / 3.0)
ang = tfe.ReliefCouplings(1, 0, 0, LMAX)
jU, jV, jR, jS, jW, jT = tfe._jump_rows(hLM, A, ang, dc)

ct, st = np.cos(tg), np.sin(tg)
Sg = tfe._shapes(0, LMAX, ct, st)
t1A_r = np.einsum('l,ln->n', jR[:LMAX + 1], Sg["Y"])
t1A_t = np.einsum('l,ln->n', jS[:LMAX + 1], Sg["Bt"])

rr = np.abs(t1A_r - t1B[:, 0] * 0)  # placeholder
rhat_g = np.stack([st, 0 * st, ct], axis=1)
that_g = np.stack([ct, 0 * ct, -st], axis=1)
t1B_r = np.einsum('nc,nc->n', rhat_g, t1B)
t1B_t = np.einsum('nc,nc->n', that_g, t1B)
print("theta  t1_r engine      t1_r spatial-FD   ratio | t1_t "
      "engine      t1_t FD        ratio")
for i in range(0, NT, 4):
    ra = t1A_r[i] / t1B_r[i]
    rt = t1A_t[i] / t1B_t[i]
    print("%5.2f %11.3e%+9.2ej %11.3e%+9.2ej %6.3f<%4.0f | "
          "%10.2e %10.2e %6.3f<%4.0f"
          % (tg[i], t1A_r[i].real, t1A_r[i].imag, t1B_r[i].real,
             t1B_r[i].imag, abs(ra), np.degrees(np.angle(ra)),
             t1A_t[i].real, t1B_t[i].real, abs(rt),
             np.degrees(np.angle(rt))))

# ---- arbitration: direct bundle sum t1C_r = -d cos(th) Sum Rp Y
t1C_r = -delta * ct * np.einsum('l,ln->n', dc["Rp"][:LMAX + 1],
                                Sg["Y"])
# surface consistency of the sigma probe
sR = np.einsum('l,ln->n', np.array(
    [0] + [(_Ysolid(l, w, A, m0, ("j", "y"), None) @ sols[l]
            * CS[l])[2] for l in range(1, LMAX + 1)],
    dtype=complex), Sg["Y"])
print("\nmax |sigma_rr(A)| residual of probe field: %.2e"
      % np.abs(sR).max())
print("theta   t1_r engine   t1_r spatialFD   t1_r bundleSum")
for i in range(0, NT, 4):
    print("%5.2f %13.4e %13.4e %13.4e"
          % (tg[i], t1A_r[i].real, t1B_r[i].real, t1C_r[i].real))

# ---- FINAL: tangential formula field (pointwise, no resum) vs
# spatial FD
sig_tt_f = np.einsum('l,ln->n', dc["ciso"], Sg["Y"]) \
    + np.einsum('l,ln->n', dc["cV"],
                tfe._shapes(0, LMAX, ct, st)["eB"][0])
Sp_f = np.einsum('l,ln->n', dc["Sp"], Sg["Bt"])
t1D_t = -delta * ct * Sp_f - delta * (st / A) * sig_tt_f
print("\ntheta   t1_t spatialFD    t1_t formula-pointwise  ratio")
for i in range(0, NT, 4):
    r = t1D_t[i] / t1B_t[i]
    print("%5.2f %13.4e %13.4e  %6.3f<%4.0f"
          % (tg[i], t1B_t[i].real, t1D_t[i].real, abs(r),
             np.degrees(np.angle(r))))
