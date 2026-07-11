"""Unit gates for the semi-analytic toroidal reference
(pyastroseis/toroidal_ref.py; docs/toroidal_reference.md V1 gates).
Login-safe: pure grid math at small lmax.

T1 fast projection == machine-precision-gated vec_sph_basis
   projection (random toroidal field, lmax 8, m <= 2).
T2 Bessel ratio recurrences vs scipy at real argument.
T3 surface factor static limit F_l -> (2l+1)/(l-1).
T4 reconstruction == vec_sph_basis synthesis at random directions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from pyastroseis.momentfit import basis_labels, vec_sph_basis   # noqa
from pyastroseis.toroidal_ref import (bessel_ratios,            # noqa
                                      surface_factor,
                                      toroidal_project,
                                      toroidal_reconstruct)

LMAX = 8
RNG = np.random.default_rng(11)


def _coeffs():
    lab = basis_labels(LMAX)
    c = np.zeros(len(lab), dtype=complex)
    W = np.zeros((LMAX + 1, 9), dtype=complex)
    for i, (fam, l, m) in enumerate(lab):
        if fam == "C" and abs(m) <= 2 and l >= 1:
            v = (RNG.standard_normal() + 1j * RNG.standard_normal()) \
                * 0.6 ** l
            c[i] = v
            W[l, m + 4] = v
    return c, W


def test_projection():
    c, Wref = _coeffs()
    nth, nphi = 48, 32
    xg, wg = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(xg)[::-1]
    wth = wg[::-1]                       # d(cos) weights == sin dtheta
    phi = 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi
    TH, PH = np.meshgrid(th, phi, indexing="ij")
    pts = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                    np.cos(TH)], axis=-1).reshape(-1, 3)
    u = (vec_sph_basis(LMAX, pts) @ c)
    n = len(pts)
    u3 = np.stack([u[0:n], u[n:2 * n], u[2 * n:3 * n]], axis=-1)
    u3 = u3.reshape(nth, nphi, 3)
    ct, st = np.cos(TH), np.sin(TH)
    cp, sp = np.cos(PH), np.sin(PH)
    that = np.stack([ct * cp, ct * sp, -st], axis=-1)
    phat = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    ut = np.einsum("ijk,ijk->ij", u3, that)
    up = np.einsum("ijk,ijk->ij", u3, phat)
    W = toroidal_project(LMAX, th, wth, nphi, (ut, up))
    err = np.max(np.abs(W[:, 2:7] - Wref[:, 2:7]))
    mon = np.max(np.abs(W[:, [0, 1, 7, 8]]))
    print("T1 projection vs basis coeffs: %.3e (monitors %.3e)"
          % (err, mon))
    assert err < 1e-12, err
    assert mon < 1e-12, mon
    return Wref


def test_bessel():
    from scipy.special import spherical_jn, spherical_yn
    z = 3.7
    s, sig = bessel_ratios(20, np.array([z + 0j]))
    for l in (1, 5, 12, 20):
        jr = spherical_jn(l, z) / spherical_jn(l - 1, z)
        h = spherical_jn(l, z) + 1j * spherical_yn(l, z)
        hm = spherical_jn(l - 1, z) + 1j * spherical_yn(l - 1, z)
        assert abs(s[l, 0] - jr) < 1e-12 * abs(jr), (l, s[l, 0], jr)
        assert abs(sig[l, 0] - h / hm) < 1e-12 * abs(h / hm)
    print("T2 bessel ratios vs scipy: ok (z=%.2f, l<=20)" % z)


def test_static_limit():
    z = np.array([1e-4 + 0j])
    F = surface_factor(12, z)
    for l in range(2, 13):
        ref = (2.0 * l + 1.0) / (l - 1.0)
        assert abs(F[l, 0] - ref) < 1e-5 * ref, (l, F[l, 0], ref)
    print("T3 static limit F_l -> (2l+1)/(l-1): ok")


def test_reconstruct():
    c, Wref = _coeffs()
    dirs = RNG.standard_normal((7, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    u = vec_sph_basis(LMAX, dirs) @ c
    n = len(dirs)
    uref = np.stack([u[0:n], u[n:2 * n], u[2 * n:3 * n]], axis=-1)
    ugot = toroidal_reconstruct(Wref, dirs)
    err = np.max(np.abs(ugot - uref)) / np.max(np.abs(uref))
    print("T4 reconstruction vs basis synthesis: %.3e" % err)
    assert err < 1e-12, err


if __name__ == "__main__":
    test_projection()
    test_bessel()
    test_static_limit()
    test_reconstruct()
    print("test_toroidal_ref: ALL PASS")
