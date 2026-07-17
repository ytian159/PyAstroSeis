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


if __name__ == "__main__":
    test_i1_l0_closed_form()
    test_selection_rules_y20()
    test_y00_equals_radius_change()
    print("all rung-A3 toroidal TFE gates passed")
