"""Rung A3 MVP: first-order boundary relief (TFE class) for the
TOROIDAL part of the spectral sweep (docs/rung_a3_tfe.md).

Relief r = d + h(theta, phi), h = sum h_LM Y_LM, on the SH-free
boundaries of the outermost solid run (outer surface r = a and/or
the run bottom r = b). First-order transferred condition at r = d
for target (l', m' = m + M):

    T1_{l'm'}(d) = - sum_l h_LM [ dTdr0_l I1(l',l)
                                  - (2 mu W0_l / d^2) I2(l',l) ]

with (free row: T0 = 0 -> W0' = W0/d)
    dTdr0_l = W0_l ( mu (L_l - 2)/d^2 - rho w^2 ),
    I1 = <C_l'm', Y_LM C_lm>          (gradient-Gaunt class),
    I2 = <C_l'm', e(C_lm) . grad_1 Y_LM>,
    e(C) = unit-sphere symmetric tangential gradient of C
    (sigma_tang = 2 mu (W0/d) e(C); toroidal is equivoluminal).

The angular integrals are evaluated NUMERICALLY (Gauss-Legendre in
cos(theta), exact phi selection m' = m + M) with the SAME normalized
C_lm conventions as toroidal_ref.toroidal_reconstruct (C orthonormal,
f = 1/sqrt(l(l+1)), sgn = (-1)^|m| for m < 0), so convention slips
cancel against the reconstruction (momentfit property). They are
k-independent and cached per (L, M, m).

PARITY COMPLETENESS: relief couples parities, but the unperturbed
operator is parity-diagonal, so du^SH at first order is closed under
the toroidal projection implemented here (the spheroidal projection
feeds du^PSV only — rung A3b).

Gates: tests/test_spectral_tfe.py (Y00 == exact radius change;
selection rules; closed-form I1 cross-check) + sympy exact-integral
cross-validation under the pytorch module env.
"""

import numpy as np

from .spectral import (TOL_PRUNE, _mats_at, _tor_entries, make_stack,
                       toroidal_unit)
from .toroidal_modes import _pole_coupling
from .toroidal_ref import source_frame, toroidal_reconstruct


# ----------------------------------------------------------------
# normalized associated Legendre grids (reconstruct conventions)
# ----------------------------------------------------------------
def _plm_grid(m, lmax, ct, st):
    """P̄_l^|m|, dP̄/dtheta, d2P̄/dtheta2 arrays (lmax+1, n) at
    ct = cos(theta) (same recurrences as toroidal_reconstruct; the
    second derivative from the Legendre ODE)."""
    am = abs(m)
    n = len(ct)
    P = np.zeros((lmax + 1, n))
    dP = np.zeros_like(P)
    p = np.full(n, 1.0 / np.sqrt(4.0 * np.pi))
    for mm in range(1, am + 1):
        p = -np.sqrt((2.0 * mm + 1.0) / (2.0 * mm)) * st * p
    pm1 = np.zeros(n)
    pcur = None
    for l in range(am, lmax + 1):
        if l == am:
            pl = p
        elif l == am + 1:
            pl = np.sqrt(2.0 * am + 3.0) * ct * p
        else:
            aa = np.sqrt((4.0 * l * l - 1.0) / (l * l - am * am))
            bb = np.sqrt(((2.0 * l + 1.0) * ((l - 1.0) ** 2
                                             - am * am))
                         / ((2.0 * l - 3.0) * (l * l - am * am)))
            pl = aa * ct * pcur - bb * pm1
        if l > am:
            pm1 = pcur
        pcur = pl
        P[l] = pl
        num = l * ct * pl
        if l > am:
            e = np.sqrt((l * l - am * am) * (2.0 * l + 1.0)
                        / (2.0 * l - 1.0))
            num = num - e * pm1
        dP[l] = num / st
    ll = np.arange(lmax + 1)[:, None]
    d2P = -ct / st * dP + (m * m / (st * st) - ll * (ll + 1.0)) * P
    return P, dP, d2P


class _AngCache:
    """Cached I1/I2 coupling integrals for one relief harmonic
    (L, M) and one source order m: I1[lp, l], I2[lp, l]."""

    def __init__(self, L, M, m, lmax, ngl=None):
        self.L, self.M, self.m, self.lmax = L, M, m, lmax
        mp = m + M
        if ngl is None:
            ngl = 2 * lmax + 2 * L + 60
        x, wgt = np.polynomial.legendre.leggauss(ngl)
        ct, st = x, np.sqrt(1.0 - x * x)
        sgn = 1.0 if m >= 0 else (-1.0) ** abs(m)
        sgp = 1.0 if mp >= 0 else (-1.0) ** abs(mp)
        sgL = 1.0 if M >= 0 else (-1.0) ** abs(M)
        P, dP, d2P = _plm_grid(m, lmax, ct, st)
        Pp, dPp, _ = _plm_grid(mp, lmax, ct, st)
        PL, dPL, _ = _plm_grid(M, L, ct, st)
        YL = sgL * PL[L]
        dYL = sgL * dPL[L]
        ll = np.arange(lmax + 1)
        f = np.zeros(lmax + 1)
        f[1:] = 1.0 / np.sqrt(ll[1:] * (ll[1:] + 1.0))
        # source C_lm components and tangential strain e(C_lm)
        Y = sgn * P
        dY = sgn * dP
        d2Y = sgn * d2P
        Ct = -1j * m * f[:, None] * Y / st
        Cp = f[:, None] * dY
        dCt = -1j * m * f[:, None] * (dY / st - Y * ct / st ** 2)
        dCp = f[:, None] * d2Y
        ett = dCt
        epp = 1j * m / st * Cp + ct / st * Ct
        etp = 0.5 * (dCp - ct / st * Cp + 1j * m / st * Ct)
        # target conj(C_l'm') components
        Yp = sgp * Pp
        dYp = sgp * dPp
        Ctp_c = +1j * mp * f[:, None] * Yp / st      # conj
        Cpp_c = f[:, None] * dYp
        # I1[lp, l] = 2 pi int (C'* . C) Y_LM dx
        I1 = 2.0 * np.pi * (
            np.einsum('an,bn,n->ab', Ctp_c, Ct, YL * wgt)
            + np.einsum('an,bn,n->ab', Cpp_c, Cp, YL * wgt))
        # grad_1 Y_LM components
        Gt = dYL
        Gp = 1j * M * YL / st
        # e(C) . grad1 Y_LM  (theta, phi components)
        vt = ett * Gt + etp * Gp
        vp = etp * Gt + epp * Gp
        I2 = 2.0 * np.pi * (
            np.einsum('an,bn,n->ab', Ctp_c, vt, wgt)
            + np.einsum('an,bn,n->ab', Cpp_c, vp, wgt))
        lp_g, l_g = np.meshgrid(np.arange(lmax + 1),
                                np.arange(lmax + 1), indexing='ij')
        tri = np.abs(lp_g - l_g) <= L        # exact selection rule
        I1[~tri] = 0.0
        I2[~tri] = 0.0
        self.I1, self.I2 = I1, I2


def _ang_cache(store, L, M, m, lmax):
    key = (L, M, m, lmax)
    if key not in store:
        store[key] = _AngCache(L, M, m, lmax)
    return store[key]


# ----------------------------------------------------------------
# first-order relief response (SH part)
# ----------------------------------------------------------------
def toroidal_relief_spectra(layers, src_xyz, M_list, w_arr,
                            station_dirs, relief, Q=None,
                            q_sign=-1.0, lmax=250,
                            tol_prune=TOL_PRUNE):
    """du^SH spectra (nsrc, nst, 3, nw), FIRST ORDER in the relief.

    relief: list of (where, L, M, h_LM) with where in
    {'top', 'bottom'} (outer surface / bottom of the outermost solid
    run) and h_LM the coefficient in meters of fully-normalized
    Y_LM. Mr*-type sources (|m| <= 1)."""
    stack = make_stack(layers)
    r0 = float(np.linalg.norm(src_xyz))
    isrc = [i for i, e in enumerate(stack)
            if e["r_bot"] < r0 < e["r_top"]][0]
    Qrot = source_frame(src_xyz)
    poleT = _pole_coupling(lmax)
    hs = []
    for M in M_list:
        M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
        hs.append(M_sf @ np.array([0.0, 0.0, 1.0]))
    w_arr = np.asarray(w_arr, dtype=complex)
    nsrc, nst, nw = len(M_list), len(station_dirs), len(w_arr)
    dirs_sf = station_dirs @ Qrot.T
    store = {}
    Wt1 = np.zeros((nsrc, nw, lmax + 1, 9), dtype=complex)
    for iw, w in enumerate(w_arr):
        mats = _mats_at(stack, w, Q, q_sign)
        msrc = mats[isrc]
        mu = msrc["mu"]
        rho = msrc["rho"]
        # unperturbed boundary values per source l (unit jump)
        Wa0 = np.zeros(lmax + 1, dtype=complex)
        Wb0 = np.zeros(lmax + 1, dtype=complex)
        entsl, botl = {}, {}
        for l in range(1, lmax + 1):
            ents, bot = _tor_entries(mats, l, r0, isrc, tol_prune)
            entsl[l], botl[l] = ents, bot
            wa, wb = toroidal_unit(l, w, ents, bot,
                                   1.0 / (mu * r0 ** 2), full=True)
            Wa0[l] = wa
            Wb0[l] = 0.0 if wb is None else wb
        a_top = stack[-1]["r_top"]
        for js in range(nsrc):
            hvec = hs[js]
            for where, L, M, hLM in relief:
                for m in (-1, 1):
                    Dm = (poleT[m][:, 0] * hvec[0]
                          + poleT[m][:, 1] * hvec[1])
                    mp = m + M
                    if abs(mp) > 4:
                        continue
                    ang = _ang_cache(store, L, M, m, lmax)
                    d = a_top if where == "top" else None
                    if where == "bottom":
                        ents1, bot1 = entsl[1], botl[1]
                        if bot1 is None:
                            raise ValueError(
                                "bottom relief needs a bottom "
                                "boundary (annulus run)")
                        d = bot1
                    W0 = (Wa0 if where == "top" else Wb0) * Dm
                    ll = np.arange(lmax + 1)
                    dTdr0 = W0 * (mu * (ll * (ll + 1.0) - 2.0)
                                  / d ** 2 - rho * w * w)
                    src_amp = 2.0 * mu * W0 / d ** 2
                    bc_l = -hLM * (ang.I1 @ dTdr0
                                   - ang.I2 @ src_amp)
                    # receiver advection: stations ride the moved
                    # free surface, so the top-relief observable is
                    # u1(a) + h d/dr u0(a) (W0' = W0/a at T = 0)
                    adv = (hLM * (ang.I1 @ (W0 / a_top))
                           if where == "top" else None)
                    for lp in range(max(1, abs(mp)), lmax + 1):
                        if bc_l[lp] == 0.0:
                            continue
                        bc = ((bc_l[lp], 0.0) if where == "bottom"
                              else (0.0, bc_l[lp]))
                        ents, bot = entsl.get(lp), botl.get(lp)
                        if ents is None:
                            continue
                        if where == "bottom" and bot is None:
                            continue
                        W1 = toroidal_unit(lp, w, ents, bot, 0.0,
                                           bc_rhs=bc)
                        if adv is not None:
                            W1 = W1 + adv[lp]
                        Wt1[js, iw, lp, mp + 4] += W1
    out = np.zeros((nsrc, nst, 3, nw), dtype=complex)
    for js in range(nsrc):
        for iw in range(nw):
            u_sf = toroidal_reconstruct(Wt1[js, iw], dirs_sf)
            out[js, :, :, iw] = u_sf @ Qrot
    return out
