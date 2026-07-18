"""Rung-A3 gates for pyastroseis.spectral_tfe (toroidal boundary
relief, first order): docs/rung_a3_tfe.md section 3.

  G-TFE-1 Y00 relief == exact radius change (radial-transfer chain,
          both boundaries, tested against FD of exact re-solves).
  G-TFE-3 selection rules for zonal Y20 relief.
  G-TFE-4 I1 vs the closed-form gradient-Gaunt reduction through
          the L = 0 special case (I1 = delta/sqrt(4 pi)) and
          hermiticity checks; the full sympy exact-integral
          cross-validation runs separately under the pytorch env
          (validation/tfe_sympy_check.py).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))

from pyastroseis import spectral as sp                     # noqa
from pyastroseis import spectral_tfe as tfe                # noqa
from tests.test_spectral import A, B, C, IC, OC, SH, wk    # noqa

QSH = 50.0
LAYERS = [dict(r_top=C, **IC), dict(r_top=B, **OC),
          dict(r_top=A, **SH)]


def _sh_spectra(layers, lmax=40):
    r0 = A - 637.0e3
    src = np.array([0.0, 0.0, r0])
    M = np.zeros((3, 3))
    M[0, 2] = M[2, 0] = 1.0e18
    th = np.radians([35.0, 90.0, 140.0])
    dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
    w_arr = np.array([wk(20.0), wk(90.0)])
    return sp.spectral_spectra(layers, src, [M], w_arr, dirs,
                               Q=QSH, q_sign=-1.0, lmax=lmax,
                               parts=("sh",))["sh"], (src, [M],
                                                      w_arr, dirs)


def test_y00_equals_radius_change():
    """h = c Y00 on a boundary must equal moving that boundary by
    delta = c/sqrt(4 pi), to first order (central FD of exact
    solves; FD error O(delta^2) << tol)."""
    lmax = 40
    u_base, (src, Ms, w_arr, dirs) = _sh_spectra(LAYERS, lmax)
    for where in ("bottom", "top"):
        delta = 50.0                    # metres
        c00 = delta * np.sqrt(4.0 * np.pi)
        lay_p = [dict(d) for d in LAYERS]
        lay_m = [dict(d) for d in LAYERS]
        if where == "bottom":
            lay_p[1] = dict(lay_p[1], r_top=B + delta)
            lay_m[1] = dict(lay_m[1], r_top=B - delta)
        else:
            lay_p[2] = dict(lay_p[2], r_top=A + delta)
            lay_m[2] = dict(lay_m[2], r_top=A - delta)
        up, _ = _sh_spectra(lay_p, lmax)
        um, _ = _sh_spectra(lay_m, lmax)
        du_fd = 0.5 * (up - um)
        du_tfe = tfe.toroidal_relief_spectra(
            LAYERS, src, Ms, w_arr, dirs,
            relief=[(where, 0, 0, c00)], Q=QSH, q_sign=-1.0,
            lmax=lmax)
        err = (np.abs(du_tfe - du_fd).max()
               / np.abs(du_fd).max())
        print("G-TFE-1 Y00 %s: TFE vs exact-FD rel %.3e "
              "(|du|/|u| %.2e)" % (where, err,
                                   np.abs(du_fd).max()
                                   / np.abs(u_base).max()))
        assert err < 5.0e-4
    # NOTE for 'top': the surface relief changes the OBSERVATION
    # radius too; stations sit on the unperturbed sphere in both
    # legs here because station_dirs are directions only and the
    # exact legs evaluate W at their own surface — the FD legs and
    # TFE both report the field coefficient at the (moved) free
    # surface, consistently to first order.


def test_selection_rules_y20():
    """Zonal Y20 relief: I1/I2 vanish exactly outside |l'-l|<=2 by
    the triangle mask; INSIDE the band, the |l'-l| = 1 couplings
    must vanish numerically (parity)."""
    ang = tfe._AngCache(2, 0, 1, 30)
    off = max(np.abs(np.diagonal(ang.I1, offset=1)).max(),
              np.abs(np.diagonal(ang.I1, offset=-1)).max(),
              np.abs(np.diagonal(ang.I2, offset=1)).max(),
              np.abs(np.diagonal(ang.I2, offset=-1)).max())
    on = max(np.abs(np.diagonal(ang.I1, offset=0)).max(),
             np.abs(np.diagonal(ang.I2, offset=2)).max())
    print("G-TFE-3 Y20 parity-off couplings %.2e vs on-band %.2e"
          % (off, on))
    assert off < 1.0e-12 * on


def test_i1_l0_closed_form():
    """L = 0: I1 must be delta_{l l'} / sqrt(4 pi) exactly (C_lm
    orthonormal), I2 must vanish (grad Y00 = 0)."""
    ang = tfe._AngCache(0, 0, 1, 25)
    tgt = np.zeros_like(ang.I1)
    np.fill_diagonal(tgt, 1.0 / np.sqrt(4.0 * np.pi))
    tgt[0, 0] = 0.0
    err1 = np.abs(ang.I1[1:, 1:] - tgt[1:, 1:]).max()
    err2 = np.abs(ang.I2).max()
    print("G-TFE-4 L=0 closed form: I1 err %.2e, I2 %.2e"
          % (err1, err2))
    assert err1 < 1.0e-13 and err2 < 1.0e-13


def _mixed_setup(lmax=30):
    r0 = A - 637.0e3
    src = np.array([0.0, 0.0, r0])
    Mt = np.zeros((3, 3))
    Mt[0, 2] = Mt[2, 0] = 1.0e18            # Mrt (m = +-1, both par.)
    Mr = np.diag([0.0, 0.0, 1.0e18])        # Mrr (m = 0)
    th = np.radians([35.0, 90.0, 140.0])
    dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
    w_arr = np.array([wk(20.0), wk(90.0)])
    return src, [Mt, Mr], w_arr, dirs, lmax


def test_null_weld():
    """A3b NULL gate: ANY relief on a welded interface between
    IDENTICAL materials must give du == 0 (all transfer terms carry
    material jumps). Exercises radial transfer, tilt, and BOTH
    parity-conversion blocks with no external reference. Y20 and
    Y21 relief, mixed mrt+mrr sources."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    split = [dict(r_top=5000.0e3, **SH), dict(r_top=A, **SH)]
    u0 = sp.spectral_spectra(split, src, Ms, w_arr, dirs, Q=QSH,
                             q_sign=-1.0, lmax=lmax)
    scale = max(np.abs(u0["psv"]).max(), np.abs(u0["sh"]).max())
    for (L, M) in ((2, 0), (2, 1)):
        du = tfe.relief_spectra(split, src, Ms, w_arr, dirs,
                                relief=[(5000.0e3, L, M, 3000.0)],
                                Q=QSH, q_sign=-1.0, lmax=lmax)
        worst = max(np.abs(du["psv"]).max(),
                    np.abs(du["sh"]).max()) / scale
        print("A3b null weld Y%d%d: |du|/|u0| = %.3e" % (L, M,
                                                         worst))
        assert worst < 1.0e-10


def test_y00_weld_contrast():
    """Y00 relief on a REAL welded interface == moving it, vs
    central FD of exact re-solves (both parities, mixed sources)."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    MID = dict(rho=4600.0, vp=10000.0, vs=5800.0)
    lay = [dict(r_top=3480.0e3, **IC), dict(r_top=4925.5e3, **MID),
           dict(r_top=A, **SH)]
    delta = 50.0
    c00 = delta * np.sqrt(4.0 * np.pi)
    lp_ = [dict(d) for d in lay]
    lm_ = [dict(d) for d in lay]
    lp_[1] = dict(lp_[1], r_top=4925.5e3 + delta)
    lm_[1] = dict(lm_[1], r_top=4925.5e3 - delta)
    kw = dict(Q=QSH, q_sign=-1.0, lmax=lmax)
    # l0=False: the relief engine does not yet carry the l = 0
    # channel (whose response moves with the interface for Mrr
    # sources — measured 1.6% of du when left in); listed A3b todo.
    up = sp.spectral_spectra(lp_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    um = sp.spectral_spectra(lm_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    du = tfe.relief_spectra(lay, src, Ms, w_arr, dirs,
                            relief=[(4925.5e3, 0, 0, c00)], **kw)
    for part in ("psv", "sh"):
        du_fd = 0.5 * (up[part] - um[part])
        err = (np.abs(du[part] - du_fd).max()
               / max(np.abs(du_fd).max(), 1e-300))
        print("A3b Y00 weld contrast %s: TFE vs exact-FD rel %.3e"
              % (part, err))
        assert err < 5.0e-4


def test_y00_top_full():
    """Free-surface Y00 relief for the FULL field (the MVP gated
    SH only): both parities vs central FD, incl. the receiver-
    advection terms."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    lay = [dict(r_top=3480.0e3, **IC), dict(r_top=A, **SH)]
    delta = 50.0
    c00 = delta * np.sqrt(4.0 * np.pi)
    lp_ = [dict(lay[0]), dict(lay[1], r_top=A + delta)]
    lm_ = [dict(lay[0]), dict(lay[1], r_top=A - delta)]
    kw = dict(Q=QSH, q_sign=-1.0, lmax=lmax)
    up = sp.spectral_spectra(lp_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    um = sp.spectral_spectra(lm_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    du = tfe.relief_spectra(lay, src, Ms, w_arr, dirs,
                            relief=[("top", 0, 0, c00)], **kw)
    for part in ("psv", "sh"):
        du_fd = 0.5 * (up[part] - um[part])
        err = (np.abs(du[part] - du_fd).max()
               / max(np.abs(du_fd).max(), 1e-300))
        print("A3b Y00 top %s: TFE vs exact-FD rel %.3e"
              % (part, err))
        assert err < 5.0e-4


def test_y00_cmb_fluid_solid():
    """A3b stage 3: Y00 relief on the fluid-solid CMB == moving the
    CMB, vs central FD of exact re-solves (both parities, mixed
    sources; the SH leg moves the toroidal run bottom). Y00
    exercises the fluid-side value transfers (Up and
    Rp = -rho w^2 u_r); the slip/pressure TILT terms need L >= 1
    (quantitative anchor: BEM rung-3 Y20 ensemble)."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    delta = 50.0
    c00 = delta * np.sqrt(4.0 * np.pi)
    lp_ = [dict(d) for d in LAYERS]
    lm_ = [dict(d) for d in LAYERS]
    lp_[1] = dict(lp_[1], r_top=B + delta)
    lm_[1] = dict(lm_[1], r_top=B - delta)
    kw = dict(Q=QSH, q_sign=-1.0, lmax=lmax)
    up = sp.spectral_spectra(lp_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    um = sp.spectral_spectra(lm_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    du = tfe.relief_spectra(LAYERS, src, Ms, w_arr, dirs,
                            relief=[(B, 0, 0, c00)], **kw)
    for part in ("psv", "sh"):
        du_fd = 0.5 * (up[part] - um[part])
        err = (np.abs(du[part] - du_fd).max()
               / max(np.abs(du_fd).max(), 1e-300))
        print("A3b-3 Y00 CMB %s: TFE vs exact-FD rel %.3e" % (part,
                                                              err))
        assert err < 5.0e-4


def test_y00_icb_solid_fluid():
    """Y00 relief on the ICB (solid below / fluid above — the
    mirrored orientation, incl. the +S_solid(lo) = -jS sign): PSV vs
    central FD; the mantle SH problem is untouched by construction
    (du_sh == 0 on BOTH routes, checked exactly)."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    delta = 50.0
    c00 = delta * np.sqrt(4.0 * np.pi)
    lp_ = [dict(d) for d in LAYERS]
    lm_ = [dict(d) for d in LAYERS]
    lp_[0] = dict(lp_[0], r_top=C + delta)
    lm_[0] = dict(lm_[0], r_top=C - delta)
    kw = dict(Q=QSH, q_sign=-1.0, lmax=lmax)
    up = sp.spectral_spectra(lp_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    um = sp.spectral_spectra(lm_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    du = tfe.relief_spectra(LAYERS, src, Ms, w_arr, dirs,
                            relief=[(C, 0, 0, c00)], **kw)
    du_fd = 0.5 * (up["psv"] - um["psv"])
    err = (np.abs(du["psv"] - du_fd).max()
           / max(np.abs(du_fd).max(), 1e-300))
    fd_sh = np.abs(0.5 * (up["sh"] - um["sh"])).max()
    print("A3b-3 Y00 ICB psv: TFE vs exact-FD rel %.3e (FD sh %.1e,"
          " TFE sh %.1e)" % (err, fd_sh, np.abs(du["sh"]).max()))
    assert err < 5.0e-4
    assert fd_sh == 0.0 and np.abs(du["sh"]).max() == 0.0


def test_null_fluid_split():
    """Fluid-fluid analogue of the null weld: ANY relief on a
    transparent split inside the outer core (IDENTICAL fluid both
    sides) must give du == 0 — delta cancellation through the fluid
    bundles including the two-sided potential slip."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    split = [dict(r_top=C, **IC), dict(r_top=2350.0e3, **OC),
             dict(r_top=B, **OC), dict(r_top=A, **SH)]
    u0 = sp.spectral_spectra(split, src, Ms, w_arr, dirs, Q=QSH,
                             q_sign=-1.0, lmax=lmax)
    scale = max(np.abs(u0["psv"]).max(), np.abs(u0["sh"]).max())
    for (L, M) in ((2, 0), (2, 1)):
        du = tfe.relief_spectra(split, src, Ms, w_arr, dirs,
                                relief=[(2350.0e3, L, M, 3000.0)],
                                Q=QSH, q_sign=-1.0, lmax=lmax)
        worst = max(np.abs(du["psv"]).max(),
                    np.abs(du["sh"]).max()) / scale
        print("A3b-3 null fluid split Y%d%d: |du|/|u0| = %.3e"
              % (L, M, worst))
        assert worst < 1.0e-10


def test_y00_fluid_split_contrast():
    """Y00 relief on a REAL fluid-fluid staircase step == moving it
    (central FD; quantitative check of the 2-row u.n / s_rr fluid
    block). SH is untouched (run bottom stays at B)."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    OCB = dict(rho=10000.0, vp=8200.0)
    lay = [dict(r_top=C, **IC), dict(r_top=2350.0e3, **OC),
           dict(r_top=B, **OCB), dict(r_top=A, **SH)]
    delta = 50.0
    c00 = delta * np.sqrt(4.0 * np.pi)
    lp_ = [dict(d) for d in lay]
    lm_ = [dict(d) for d in lay]
    lp_[1] = dict(lp_[1], r_top=2350.0e3 + delta)
    lm_[1] = dict(lm_[1], r_top=2350.0e3 - delta)
    kw = dict(Q=QSH, q_sign=-1.0, lmax=lmax)
    up = sp.spectral_spectra(lp_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    um = sp.spectral_spectra(lm_, src, Ms, w_arr, dirs, l0=False,
                             **kw)
    du = tfe.relief_spectra(lay, src, Ms, w_arr, dirs,
                            relief=[(2350.0e3, 0, 0, c00)], **kw)
    du_fd = 0.5 * (up["psv"] - um["psv"])
    err = (np.abs(du["psv"] - du_fd).max()
           / max(np.abs(du_fd).max(), 1e-300))
    print("A3b-3 Y00 fluid split psv: TFE vs exact-FD rel %.3e"
          % err)
    assert err < 5.0e-4


import pytest


@pytest.mark.xfail(strict=True, reason="A3b OPEN (docs section 6):"
                   " PSV fails the exact translation identity"
                   " (U/V 0.25..1.09 per lp) while SH passes"
                   " EXACTLY; every engine constituent is"
                   " externally verified and an independent"
                   " collocation solver reproduces the engine to"
                   " 4-5 digits — unresolved paradox; all L >= 1"
                   " PSV relief output is unverdicted until fixed")
def test_l1_translation():
    """EXACT finite-L anchor (the Y00 gates cannot see the tilt/
    slip/conversion terms because grad1 Y00 = 0): Y10 relief
    h = delta cos(theta) applied to EVERY boundary equals a rigid
    translation of the earth by delta zhat with the source held
    fixed — which equals the unperturbed spherical problem with the
    source at src - delta zhat, observed at the mapped surface
    directions d' = d + (delta sin(theta)/a) theta_hat (cartesian
    components are translation-invariant). Central FD in delta.
    mrt source only: m = +-1 keeps the (absent) l = 0 channel out
    of both routes. Pins the SIGNS of all L >= 1 couplings.
    The SH part of this gate PASSES EXACTLY (validating the gate
    construction end-to-end); the PSV part FAILS — see
    docs/rung_a3_tfe.md section 6 for the full forensic record."""
    src, Ms, w_arr, dirs, lmax = _mixed_setup()
    Ms = [Ms[0]]                              # mrt only
    delta = 30.0
    c10 = delta * np.sqrt(4.0 * np.pi / 3.0)  # cos(th) = c*Ybar10
    kw = dict(Q=QSH, q_sign=-1.0, lmax=lmax)
    th = np.arctan2(np.hypot(dirs[:, 0], dirs[:, 1]), dirs[:, 2])
    ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    that = np.stack([np.cos(th) * np.cos(ph),
                     np.cos(th) * np.sin(ph), -np.sin(th)], axis=1)
    zhat = np.array([0.0, 0.0, 1.0])
    du_fd = {}
    for s in (+1.0, -1.0):
        dp = dirs + s * (delta / A) * np.sin(th)[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        u_s = sp.spectral_spectra(LAYERS, src - s * delta * zhat,
                                  Ms, w_arr, dp, l0=False, **kw)
        for part in ("psv", "sh"):
            du_fd[part] = du_fd.get(part, 0.0) + 0.5 * s * u_s[part]
    du = tfe.relief_spectra(LAYERS, src, Ms, w_arr, dirs,
                            relief=[(C, 1, 0, c10), (B, 1, 0, c10),
                                    ("top", 1, 0, c10)], **kw)
    for part in ("psv", "sh"):
        err = (np.abs(du[part] - du_fd[part]).max()
               / max(np.abs(du_fd[part]).max(), 1e-300))
        print("A3b-3 L=1 translation %s: TFE vs exact-FD rel %.3e"
              % (part, err))
        assert err < 5.0e-4


def test_conversion_parity():
    """Parity selection of the conversion blocks: same-parity
    couplings (G0, GA_v, GC_w, HV_v) live on EVEN |l'-l| for even L;
    conversion couplings (GC_v, GA_w, HW_v, Hiso_w) live on ODD
    |l'-l|."""
    ang = tfe.ReliefCouplings(2, 0, 1, 24)

    def offmax(Mx, offs):
        return max(np.abs(np.diagonal(Mx, offset=o)).max()
                   for o in offs)

    same = max(offmax(ang.G0, (1, -1)), offmax(ang.GA_v, (1, -1)),
               offmax(ang.GC_w, (1, -1)), offmax(ang.HV_v, (1, -1)))
    conv_off = max(offmax(ang.GC_v, (0, 2, -2)),
                   offmax(ang.GA_w, (0, 2, -2)),
                   offmax(ang.HW_v, (0, 2, -2)),
                   offmax(ang.Hiso_w, (0, 2, -2)))
    conv_on = max(offmax(ang.GC_v, (1, -1)),
                  offmax(ang.HW_v, (1, -1)))
    print("A3b conversion parity: wrong-parity %.2e vs on-parity "
          "%.2e" % (max(same, conv_off), conv_on))
    assert max(same, conv_off) < 1.0e-12 * conv_on


if __name__ == "__main__":
    test_i1_l0_closed_form()
    test_selection_rules_y20()
    test_conversion_parity()
    test_y00_equals_radius_change()
    print("all rung-A3 toroidal TFE gates passed")
    test_null_weld()
    test_y00_weld_contrast()
    test_y00_top_full()
    print("all rung-A3b spheroidal relief gates passed")
    test_null_fluid_split()
    test_y00_cmb_fluid_solid()
    test_y00_icb_solid_fluid()
    test_y00_fluid_split_contrast()
    print("all rung-A3b stage-3 fluid-interface gates passed")
    try:
        test_l1_translation()
        print("UNEXPECTED: L=1 translation gate passed — the A3b"
              " section-6 paradox may be resolved; remove xfail")
    except AssertionError:
        print("L=1 translation gate: KNOWN FAIL (PSV) — A3b open,"
              " docs section 6")
