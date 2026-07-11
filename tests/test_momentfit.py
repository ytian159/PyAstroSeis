"""Self-tests for the moment-fitted-RHS rig (pyastroseis/momentfit.py,
docs/moment_fitted_rhs.md). Pure grid math — login-safe, no meshes, no
solver. Gates 1-3 of the memo:

  M1 basis: scalar + vector orthonormality against an independent
     dense Gauss-Legendre x trapezoid quadrature (validates the
     Legendre recurrence, the dtheta identity, the m/sin(theta) terms,
     the local frames and the P/B/C family orthogonality).
  M2 dP/dtheta against central finite differences.
  M3 exact-moment quadrature: analytic combinations of basis fields on
     the graded source-frame grids return E = R^2 * coeffs, both
     tiers; includes an l = lmax column (panel-resolution check).
  M4 fit: corrected moments match the target to machine precision; a
     smooth low-l trace needs only a tiny correction.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from pyastroseis.momentfit import (MomentFit, basis_labels,        # noqa
                                   norm_legendre, source_frame_grid,
                                   vec_sph_basis)

R = 6371e3
SRC = np.array([0.6, 0.12, 0.79])
SRC = 6321e3 * SRC / np.linalg.norm(SRC)


def dense_grid(nth=64, nphi=128):
    xg, wg = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(xg)
    phi = 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi
    TH, PH = np.meshgrid(th, phi, indexing="ij")
    pts = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                    np.cos(TH)], axis=-1).reshape(-1, 3)
    wt = (wg[:, None] * np.full(nphi, 2.0 * np.pi / nphi)).ravel()
    return pts, wt


def test_gram():
    lmax = 8
    pts, wt = dense_grid()
    Psi = vec_sph_basis(lmax, pts)
    G = (np.conj(Psi) * np.tile(wt, 3)[:, None]).T @ Psi
    err = np.max(np.abs(G - np.eye(G.shape[0])))
    print("M1 vector-basis Gram vs identity: %.3e" % err)
    assert err < 1e-10, err

    # scalar route on its own (independent of the vector assembly)
    P = norm_legendre(lmax, np.clip(pts[:, 2], -1, 1))
    ph = np.arctan2(pts[:, 1], pts[:, 0])
    Y = np.stack([P[l, m] * np.exp(1j * m * ph)
                  for l in range(lmax + 1) for m in range(l + 1)])
    Gs = (np.conj(Y) * wt) @ Y.T
    ers = np.max(np.abs(Gs - np.eye(len(Y))))
    print("M1 scalar Y_lm Gram vs identity:  %.3e" % ers)
    assert ers < 1e-11, ers


def test_dtheta_fd():
    lmax = 12
    rng = np.random.default_rng(7)
    th = rng.uniform(0.05, np.pi - 0.05, 40)
    h = 1e-6
    _, dP = norm_legendre(lmax, np.cos(th), deriv=True)
    Pp = norm_legendre(lmax, np.cos(th + h))
    Pm = norm_legendre(lmax, np.cos(th - h))
    fd = (Pp - Pm) / (2.0 * h)
    scale = np.max(np.abs(dP))
    err = np.max(np.abs(dP - fd)) / scale
    print("M2 dP/dtheta vs central FD: %.3e (rel)" % err)
    assert err < 1e-7, err


def _combo_field(lmax, coeffs):
    def field(pts):
        return vec_sph_basis(lmax, pts) @ coeffs
    return field


def test_exact_moments():
    lmax = 16
    lab = basis_labels(lmax)
    coeffs = np.zeros(len(lab), dtype=complex)
    for key, c in [(("P", 2, 1), 1.3 - 0.4j), (("B", 3, -2), 0.7j),
                   (("C", 2, 1), -2.1 + 0.9j), (("C", 5, 0), 0.35),
                   (("C", lmax, 7), 1.0 + 1.0j)]:
        coeffs[lab.index(key)] = c
    ic, area = dense_grid(48, 96)
    mf = MomentFit(lmax, R * ic, R * R * area, R, SRC, 50e3)
    field = _combo_field(lmax, coeffs)
    for tier in (1, 2):
        E = mf.exact_moments(field, tier=tier)
        err = np.max(np.abs(E - R * R * coeffs)) / (R * R)
        print("M3 tier-%d exact moments vs coeffs: %.3e" % (tier, err))
        assert err < 1e-8, (tier, err)


def test_fit():
    lmax = 6
    # fake "mesh": a moderately dense grid acting as incenters+areas
    ic, area = dense_grid(32, 64)
    mf = MomentFit(lmax, R * ic, R * R * area, R, SRC, 50e3)
    print("M4 gram cond (grid-as-mesh): %.2e" % mf.gram_cond)
    assert mf.gram_cond < 10.0

    lab = basis_labels(lmax)
    rng = np.random.default_rng(3)
    coeffs = (rng.standard_normal(len(lab))
              + 1j * rng.standard_normal(len(lab)))
    # damp towards higher l so the trace is smooth
    for i, (fam, l, m) in enumerate(lab):
        coeffs[i] *= 0.5 ** l
    field = _combo_field(lmax, coeffs)

    b = field(R * ic)                     # pointwise-sampled trace
    bc, dg = mf.correct(b, field)
    resid = np.max(np.abs(mf.MHi @ bc - dg["E"])) / (R * R)
    print("M4 corrected-moment residual: %.3e   |db|/|b| = %.3e   "
          "grid diff = %.3e" % (resid, dg["dbnorm"], dg["griddiff"]))
    assert resid < 1e-9, resid            # moments pinned exactly
    assert dg["dbnorm"] < 1e-6, dg        # smooth trace: tiny fix
    assert dg["griddiff"] < 1e-8, dg


if __name__ == "__main__":
    test_gram()
    test_dtheta_fd()
    test_exact_moments()
    test_fit()
    print("test_momentfit: ALL PASS")
