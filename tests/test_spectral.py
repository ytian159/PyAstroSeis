"""Rung-A gates for pyastroseis.spectral (docs/rung_a_spectral.md
section 4).

  G-A1 partition invariance: splitting uniform layers into
       identical-material sub-layers changes nothing (exercises
       solid-solid, fluid-fluid AND both mixed interface stencils).
  G-A2 vs mini-tipsv: per-(l,k) unit solves and end-to-end spectra
       equal spheroidal_ref on the corefluid topology.
  G-A3 toroidal static limit vs toroidal_modes.static_factor
       (closed form) and resonance position vs mode_catalog.
  G-A4 toroidal physics anchor vs the independent mode-sum route.
  G-A5 pruning-tolerance insensitivity.

Run: python -m pytest tests/test_spectral.py -q  (sub-minute grids)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))

from pyastroseis import spectral as sp                    # noqa: E402
from pyastroseis.spheroidal_ref import (L_SERIES, forced_surface,     # noqa: E402
                                        q_factor, source_jumps,
                                        spheroidal_spectra)
from pyastroseis.toroidal_modes import (mode_catalog, static_factor,  # noqa: E402
                                        toroidal_mode_spectra)

A, B, C = 6371.0e3, 3480.0e3, 1221.5e3
SH = dict(rho=4400.0, vp=11300.0, vs=6300.0)
OC = dict(rho=11000.0, vp=9000.0)
IC = dict(rho=12900.0, vp=11100.0, vs=3500.0)
LAYERS = [dict(r_top=C, **IC), dict(r_top=B, **OC), dict(r_top=A, **SH)]
TLEN, OMEGAI = 240000.0, 1.0e-5
QSH = 50.0


def wk(k):
    return 2.0 * np.pi * k / TLEN + 1j * OMEGAI


def unit_solves(layers, l, w, r0, Q=None, tol_prune=sp.TOL_PRUNE):
    """(U_rh, V_rh, U_y, V_y, W_t) unit responses for one (l, w)."""
    stack = sp.make_stack(layers)
    mats = sp._mats_at(stack, w, Q, -1.0)
    isrc = [i for i, e in enumerate(stack)
            if e["r_bot"] < r0 < e["r_top"]][0]
    msrc = mats[isrc]
    zr = msrc["r_top"] if l >= L_SERIES else None
    L = l * (l + 1.0)
    icut = sp._icut(stack, l, r0, tol_prune)
    ents = sp._entries(mats, icut, r0, isrc)
    F0v = np.array([0, 0, 0, 1.0 / (L * r0 ** 2)], dtype=complex)
    F1v = np.array([0, 0, -1.0 / r0 ** 3, 3.0 / (L * r0 ** 3)],
                   dtype=complex)
    J = source_jumps(l, w, r0, msrc["rho"], msrc["lam"], msrc["mu"],
                     F0v, F1v, zref_a=zr)
    U1, V1 = sp.spheroidal_unit(l, w, ents, J)
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    Jy = source_jumps(l, w, r0, msrc["rho"], msrc["lam"], msrc["mu"],
                      F0y, F1y, zref_a=zr)
    U2, V2 = sp.spheroidal_unit(l, w, ents, Jy)
    ents_t, bot = sp._tor_entries(mats, l, r0, isrc, tol_prune)
    Wt = sp.toroidal_unit(l, w, ents_t, bot,
                          1.0 / (msrc["mu"] * r0 ** 2))
    return np.array([U1, V1, U2, V2, Wt])


def rel(u1, u0):
    """Per-channel relative error: (U, V) pairs are normalized by
    the pair's own scale, W_t by itself (element-wise normalization
    inflates solve roundoff on near-null components — measured
    2026-07-16: worst offender was an element 120x below its
    vector's scale)."""
    s = np.abs(u0)
    scale = np.array([max(s[0], s[1]), max(s[0], s[1]),
                      max(s[2], s[3]), max(s[2], s[3]), s[4]])
    return np.abs(u1 - u0) / scale


def test_partition_invariance():
    split = [
        dict(r_top=0.6 * C, **IC), dict(r_top=C, **IC),
        dict(r_top=0.5 * (B + C), **OC), dict(r_top=B, **OC),
        dict(r_top=B + 0.30 * (A - B), **SH),
        dict(r_top=B + 0.65 * (A - B), **SH), dict(r_top=A, **SH)]
    worst = 0.0
    for r0 in (A - 50.0e3, A - 637.0e3):
        for k in (3.0, 20.0, 60.0, 138.0):
            for l in (1, 2, 5, 8, 9, 12, 19, 20, 21, 25, 40, 80,
                      150, 250, 400):
                u0 = unit_solves(LAYERS, l, wk(k), r0, Q=QSH)
                u1 = unit_solves(split, l, wk(k), r0, Q=QSH)
                worst = max(worst, np.max(rel(u1, u0)))
    print("G-A1 partition invariance worst rel: %.3e" % worst)
    assert worst < 1.0e-8


def test_vs_minitipsv_units():
    worst = 0.0
    for r0 in (A - 50.0e3, A - 637.0e3):
        for k in (3.0, 20.0, 60.0, 138.0):
            w = wk(k)
            fac2 = q_factor(w, QSH, -1.0) ** 2
            mu_s = SH["rho"] * SH["vs"] ** 2 * fac2
            lam_s = (SH["rho"] * SH["vp"] ** 2
                     - 4.0 / 3.0 * SH["rho"] * SH["vs"] ** 2
                     - 2.0 / 3.0 * mu_s)
            mu_i = IC["rho"] * IC["vs"] ** 2 * fac2
            lam_i = (IC["rho"] * IC["vp"] ** 2
                     - 4.0 / 3.0 * IC["rho"] * IC["vs"] ** 2
                     - 2.0 / 3.0 * mu_i)
            model = dict(a=A, b=B, c=C, sh=(SH["rho"], lam_s, mu_s),
                         oc=(OC["rho"], OC["vp"]),
                         ic=(IC["rho"], lam_i, mu_i))
            for l in (1, 2, 5, 8, 9, 12, 19, 20, 21, 25, 40, 80,
                      150, 250, 400, 900):
                zr = A if l >= L_SERIES else None
                L = l * (l + 1.0)
                for F0, F1 in (
                    (np.array([0, 0, 0, 1.0 / (L * r0 ** 2)],
                              dtype=complex),
                     np.array([0, 0, -1.0 / r0 ** 3,
                               3.0 / (L * r0 ** 3)], dtype=complex)),
                    (np.array([0, 0, 1.0 / r0 ** 2, 0],
                              dtype=complex),
                     np.array([0, 0, 2.0 / r0 ** 3, 0],
                              dtype=complex))):
                    J = source_jumps(l, w, r0, SH["rho"], lam_s,
                                     mu_s, F0, F1, zref_a=zr)
                    Ur, Vr = forced_surface(l, w, model, r0, J)
                    stack = sp.make_stack(LAYERS)
                    mats = sp._mats_at(stack, w, QSH, -1.0)
                    icut = sp._icut(stack, l, r0, sp.TOL_PRUNE)
                    ents = sp._entries(mats, icut, r0, 2)
                    Um, Vm = sp.spheroidal_unit(l, w, ents, J)
                    sc = max(abs(Ur), abs(Vr))
                    worst = max(worst, abs(Um - Ur) / sc,
                                abs(Vm - Vr) / sc)
    print("G-A2 vs mini-tipsv unit solves worst rel: %.3e" % worst)
    # Two measured floors (2026-07-16): (a) mini-tipsv's shell-only
    # branch discards (B/r0)^(2l) fluid coupling at its l=20 seam —
    # 1.7e-7 at 637 km depth, where the seam-arbitration assertion
    # below proves the NEW solver matches mini-tipsv's own full
    # 12x12 branch to 2e-12; (b) conditioning floor ~3e-7 at the
    # extreme (l >= 400, k = 3) corner, where the harmonic weight
    # (r0/a)^l makes the contribution physically negligible.
    assert worst < 1.0e-6


def test_seam_arbitration():
    """At mini-tipsv's l=20 shell-only seam (637-km source) the new
    solver must agree with mini-tipsv's FULL 12x12 branch (forced
    via l_switch), proving the broad-grid G-A2 residual there is the
    reference's truncation, not ours."""
    r0 = A - 637.0e3
    w = wk(20.0)
    fac2 = q_factor(w, QSH, -1.0) ** 2
    mu_s = SH["rho"] * SH["vs"] ** 2 * fac2
    lam_s = (SH["rho"] * SH["vp"] ** 2 - 4.0 / 3.0 * SH["rho"]
             * SH["vs"] ** 2 - 2.0 / 3.0 * mu_s)
    mu_i = IC["rho"] * IC["vs"] ** 2 * fac2
    lam_i = (IC["rho"] * IC["vp"] ** 2 - 4.0 / 3.0 * IC["rho"]
             * IC["vs"] ** 2 - 2.0 / 3.0 * mu_i)
    model = dict(a=A, b=B, c=C, sh=(SH["rho"], lam_s, mu_s),
                 oc=(OC["rho"], OC["vp"]),
                 ic=(IC["rho"], lam_i, mu_i))
    l = 20
    L = l * (l + 1.0)
    F0 = np.array([0, 0, 0, 1.0 / (L * r0 ** 2)], dtype=complex)
    F1 = np.array([0, 0, -1.0 / r0 ** 3, 3.0 / (L * r0 ** 3)],
                  dtype=complex)
    J = source_jumps(l, w, r0, SH["rho"], lam_s, mu_s, F0, F1,
                     zref_a=A)
    full = np.array(forced_surface(l, w, model, r0, J, l_switch=25))
    stack = sp.make_stack(LAYERS)
    mats = sp._mats_at(stack, w, QSH, -1.0)
    ents = sp._entries(mats, sp._icut(stack, l, r0, sp.TOL_PRUNE),
                       r0, 2)
    mine = np.array(sp.spheroidal_unit(l, w, ents, J))
    err = np.abs(mine - full).max() / np.abs(full).max()
    print("G-A2 seam arbitration vs full 12x12: %.3e" % err)
    assert err < 1.0e-10


def test_vs_minitipsv_spectra():
    r0 = A - 50.0e3
    src = np.array([0.0, 0.0, r0])
    M = np.zeros((3, 3))
    M[0, 2] = M[2, 0] = 1.0e18            # Mrt-type at the pole
    My = np.diag([0.0, 0.0, 1.0e18])      # Mrr-type
    th = np.radians([35.0, 90.0, 140.0])
    dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
    w_arr = np.array([wk(20.0), wk(90.0)])
    model0 = dict(a=A, b=B, c=C,
                  sh=(SH["rho"], SH["vp"], SH["vs"]),
                  oc=(OC["rho"], OC["vp"]),
                  ic=(IC["rho"], IC["vp"], IC["vs"]))
    ref = spheroidal_spectra(model0, src, [M, My], w_arr, dirs,
                             Q=QSH, q_sign=-1.0, lmax=60)
    got = sp.spectral_spectra(LAYERS, src, [M, My], w_arr, dirs,
                              Q=QSH, q_sign=-1.0, lmax=60,
                              parts=("psv",), l0=False)["psv"]
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print("G-A2 end-to-end PSV vs spheroidal_spectra: %.3e" % err)
    assert err < 1.0e-6


def test_toroidal_static_limit():
    vs, rho = SH["vs"], SH["rho"]
    mu = rho * vs * vs
    w = (1.0e-4 + 1.0e-6j) * vs / A
    worst = 0.0
    for layers, bstat in (([dict(r_top=A, **SH)], None),
                          (LAYERS[1:], B)):
        stack = sp.make_stack(layers)
        isrc = len(stack) - 1
        for r0 in (A - 50.0e3, A - 637.0e3):
            for l in (2, 3, 5, 8, 20, 60):
                mats = sp._mats_at(stack, w, None, -1.0)
                ents, bot = sp._tor_entries(mats, l, r0, isrc,
                                            sp.TOL_PRUNE)
                Wt = sp.toroidal_unit(l, w, ents, bot,
                                      1.0 / (mu * r0 ** 2))
                Ws = static_factor(l, A, bstat, r0) / mu
                worst = max(worst, abs(Wt - Ws) / abs(Ws))
    print("G-A3 static limit worst rel: %.3e" % worst)
    assert worst < 1.0e-6


def test_toroidal_resonance():
    vs, rho = SH["vs"], SH["rho"]
    mu = rho * vs * vs
    r0 = A - 637.0e3
    md = mode_catalog(2, vs, A, None, fmax=2.0e-3)[0]
    stack = sp.make_stack([dict(r_top=A, **SH)])
    amps = []
    ww = md["w"] * np.linspace(0.98, 1.02, 161)
    for w in ww:
        mats = sp._mats_at(stack, w + 1.0e-6j * md["w"], None, -1.0)
        ents, bot = sp._tor_entries(mats, 2, r0, 0, sp.TOL_PRUNE)
        amps.append(abs(sp.toroidal_unit(
            2, w + 1.0e-6j * md["w"], ents, bot,
            1.0 / (mu * r0 ** 2))))
    wpk = ww[int(np.argmax(amps))]
    err = abs(wpk - md["w"]) / md["w"]
    print("G-A3 0T2 resonance offset: %.3e" % err)
    assert err < 2.0e-3


def test_toroidal_vs_mode_sum():
    r0 = A - 637.0e3
    src = np.array([0.0, 0.0, r0])
    M = np.zeros((3, 3))
    M[0, 2] = M[2, 0] = 1.0e18
    th = np.radians([35.0, 90.0, 140.0])
    dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
    w_arr = np.array([wk(10.0), wk(40.0), wk(90.0)])
    model = dict(vs=SH["vs"], rho=SH["rho"], a=A, b=None)
    got = sp.spectral_spectra([dict(r_top=A, **SH)], src, [M],
                              w_arr, dirs, Q=None, lmax=200,
                              parts=("sh",))["sh"][0]
    # The mode-sum reference's residual is ITS OWN l=1 tail: the
    # static completion covers only l >= 2 (l = 1 is the degenerate
    # rigid-rotation case), so the plain l=1 overtone sum leaves a
    # global absolute deficit ~ 1/w_max — measured 2026-07-16:
    # station-independent abs dev 1.5e-9 -> 7.6e-10 -> 3.8e-10 as
    # fmax doubles 1e-2 -> 2e-2 -> 4e-2, converging TOWARD the
    # forced solve. Gate = that convergence (rate ~ 0.5 per fmax
    # doubling) plus a bound on the deviation at default truncation
    # relative to the dominant-station signal.
    dev = []
    for fmax in (1.0e-2, 2.0e-2):
        ref = toroidal_mode_spectra(model, src, M, w_arr, dirs,
                                    Q=None, lmax=200, fmax=fmax)
        dev.append(np.abs(got - ref).max())
    ratio = dev[1] / dev[0]
    relmax = dev[0] / np.abs(got).max()
    print("G-A4 vs mode sum: abs dev %.2e -> %.2e (ratio %.2f), "
          "rel to dominant signal %.3f" % (dev[0], dev[1], ratio,
                                           relmax))
    assert 0.35 < ratio < 0.70 and relmax < 0.05


def test_l0_radial():
    """l = 0 radial branch (feeds Mrr-Z only; anchored on tipsv:
    mrr-Z k=100-138 median 0.22 -> 0.025 on the 50-km campaign when
    added, 2026-07-17): partition invariance + strict additivity
    (l0 off must not change any l >= 1 content; an Mrt-type source
    has mzz = 0 so its spectra are bitwise unaffected)."""
    r0 = A - 50.0e3
    w = wk(60.0)

    def u0_of(layers):
        st = sp.make_stack(layers)
        mats = sp._mats_at(st, w, QSH, -1.0)
        isrc = [i for i, e in enumerate(st)
                if e["r_bot"] < r0 < e["r_top"]][0]
        ents = sp._entries(mats, 0, r0, isrc)
        return sp.spheroidal_unit_l0(w, ents,
                                     sp._sph0_jump(w, r0,
                                                   mats[isrc]))

    split = [
        dict(r_top=0.6 * C, **IC), dict(r_top=C, **IC),
        dict(r_top=0.5 * (B + C), **OC), dict(r_top=B, **OC),
        dict(r_top=B + 0.40 * (A - B), **SH), dict(r_top=A, **SH)]
    ua, ub = u0_of(LAYERS), u0_of(split)
    err = abs(ua - ub) / abs(ua)
    print("G-A6 l=0 partition invariance: %.3e" % err)
    assert err < 1.0e-12

    src = np.array([0.0, 0.0, r0])
    M = np.zeros((3, 3))
    M[0, 2] = M[2, 0] = 1.0e18
    dirs = np.array([[np.sin(0.7), 0.0, np.cos(0.7)]])
    w_arr = np.array([w])
    kw = dict(Q=QSH, q_sign=-1.0, lmax=8, parts=("psv",))
    u_on = sp.spectral_spectra(LAYERS, src, [M], w_arr, dirs,
                               l0=True, **kw)["psv"]
    u_off = sp.spectral_spectra(LAYERS, src, [M], w_arr, dirs,
                                l0=False, **kw)["psv"]
    assert np.array_equal(u_on, u_off)
    print("G-A6 l=0 additivity: Mrt spectra bitwise unchanged")


def test_pruning_insensitivity():
    r0 = A - 50.0e3
    worst = 0.0
    for k in (20.0, 138.0):
        for l in (21, 24, 27, 30):
            u0 = unit_solves(LAYERS, l, wk(k), r0, Q=QSH,
                             tol_prune=1.0e-12)
            u1 = unit_solves(LAYERS, l, wk(k), r0, Q=QSH,
                             tol_prune=1.0e-16)
            worst = max(worst, np.max(rel(u1, u0)))
    print("G-A5 pruning tol 1e-12 vs 1e-16 worst rel: %.3e" % worst)
    assert worst < 1.0e-8


if __name__ == "__main__":
    test_partition_invariance()
    test_vs_minitipsv_units()
    test_seam_arbitration()
    test_vs_minitipsv_spectra()
    test_toroidal_static_limit()
    test_toroidal_resonance()
    test_toroidal_vs_mode_sum()
    test_l0_radial()
    test_pruning_insensitivity()
    print("all rung-A gates passed")
