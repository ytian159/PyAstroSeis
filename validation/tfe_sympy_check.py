#!/usr/bin/env python3
"""G-TFE-4b: independent SYMBOLIC cross-validation of the TFE
angular coupling integrals I1 and I2 (pyastroseis/spectral_tfe.py)
via sympy exact surface integrals — the house dual-derivation
pattern (rung-C precedent). Needs sympy: run under
`module load pytorch/2.8.0`.

Checks, for relief (L=2, M=0), source order m=1, l, l' in 1..5:
  I1[l',l] = int conj(C_l'm) . C_lm  Y_LM dOmega
  I2[l',l] = int conj(C_l'm) . (e(C_lm) . grad_1 Y_LM) dOmega
against the numeric Gauss-Legendre values of _AngCache.
Convention pin: our Y_lm (m >= 0) must equal sympy Ynm exactly at a
test point before any integral is trusted.
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
from pyastroseis.spectral_tfe import _AngCache, _plm_grid   # noqa

th, ph = sp.symbols("theta phi", real=True)
L, M, m = 2, 0, 1
LMAX = 5


def Y(l, mm):
    return sp.Ynm(l, mm, th, ph).expand(func=True)


def Cvec(l, mm):
    f = 1 / sp.sqrt(sp.Integer(l) * (l + 1))
    y = Y(l, mm)
    Ct = -f * sp.diff(y, ph) / sp.sin(th)      # -(im Y)/sin = -f dY/dphi /sin ... careful: -i m Y f/sin
    Ct = -f * sp.I * mm * y / sp.sin(th)
    Cp = f * sp.diff(y, th)
    return Ct, Cp


# convention pin at a test point
tv = {th: sp.Rational(2, 5), ph: sp.Rational(3, 7)}
ct0 = float(sp.cos(tv[th]))
P, dP, _ = _plm_grid(m, LMAX, np.array([ct0]),
                     np.array([np.sqrt(1 - ct0 ** 2)]))
for l in range(1, LMAX + 1):
    ours = P[l][0] * np.exp(1j * m * float(tv[ph]))
    theirs = complex(Y(l, m).subs(tv).evalf())
    assert abs(ours - theirs) < 1e-12 * max(1.0, abs(theirs)), \
        (l, ours, theirs)
print("convention pin: our Ybar == sympy Ynm for m=%d, l<=%d" %
      (m, LMAX))

YL = Y(L, M)
GtL = sp.diff(YL, th)
GpL = sp.diff(YL, ph) / sp.sin(th)

ang = _AngCache(L, M, m, LMAX)
worst1 = worst2 = 0.0
for lp in range(1, LMAX + 1):
    Ctp, Cpp = Cvec(lp, m + M)
    Ctp_c, Cpp_c = sp.conjugate(Ctp), sp.conjugate(Cpp)
    for l in range(1, LMAX + 1):
        if abs(lp - l) > L:
            continue
        Ct, Cp = Cvec(l, m)
        ett = sp.diff(Ct, th)
        epp = sp.diff(Cp, ph) / sp.sin(th) + sp.cos(th) / sp.sin(th) * Ct
        etp = (sp.diff(Cp, th) - sp.cos(th) / sp.sin(th) * Cp
               + sp.diff(Ct, ph) / sp.sin(th)) / 2
        i1 = sp.integrate(sp.integrate(sp.expand_trig(sp.simplify(
            (Ctp_c * Ct + Cpp_c * Cp) * YL)) * sp.sin(th),
            (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))
        vt = ett * GtL + etp * GpL
        vp2 = etp * GtL + epp * GpL
        i2 = sp.integrate(sp.integrate(sp.simplify(
            (Ctp_c * vt + Cpp_c * vp2)) * sp.sin(th),
            (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))
        i1n, i2n = complex(i1.evalf()), complex(i2.evalf())
        d1 = abs(i1n - ang.I1[lp, l])
        d2 = abs(i2n - ang.I2[lp, l])
        sc1 = max(1e-3, abs(i1n))
        sc2 = max(1e-3, abs(i2n))
        worst1 = max(worst1, d1 / sc1)
        worst2 = max(worst2, d2 / sc2)
        print("l'=%d l=%d: I1 sym %+9.5f num %+9.5f | I2 sym "
              "%+9.5f%+9.5fj num %+9.5f%+9.5fj"
              % (lp, l, i1n.real, ang.I1[lp, l].real, i2n.real,
                 i2n.imag, ang.I2[lp, l].real, ang.I2[lp, l].imag))
print("G-TFE-4b sympy cross-check: worst I1 rel %.2e, I2 rel %.2e"
      % (worst1, worst2))
assert worst1 < 1e-10 and worst2 < 1e-10
print("PASS")

# ---------------------------------------------------------------
# A3b: the NEW coupling matrices (spheroidal + conversion + tilt)
# vs exact symbolic surface integrals, small (l, l') set.
# ---------------------------------------------------------------
from pyastroseis.spectral_tfe import ReliefCouplings   # noqa: E402

ang2 = ReliefCouplings(L, M, m, LMAX)


def Bvec(l, mm):
    y = Y(l, mm)
    return sp.diff(y, th), sp.diff(y, ph) / sp.sin(th)


def surf_strain2(vt, vp):
    """(2 e_tt, 2 e_pp, 2 e_tp) of tangential (vt, vp) on the unit
    sphere — same formulas as spectral_tfe._shapes.strain2."""
    cot = sp.cos(th) / sp.sin(th)
    e_tt = 2 * sp.diff(vt, th)
    e_pp = 2 * (sp.diff(vp, ph) / sp.sin(th) + cot * vt)
    e_tp = sp.diff(vp, th) - cot * vp + sp.diff(vt, ph) / sp.sin(th)
    return e_tt, e_pp, e_tp


def integ(expr):
    return complex(sp.integrate(sp.integrate(
        sp.simplify(expr) * sp.sin(th), (ph, 0, 2 * sp.pi)),
        (th, 0, sp.pi)).evalf())


worst = {}
for lp in range(1, LMAX + 1):
    Lp = sp.Integer(lp) * (lp + 1)
    Ytp = Y(lp, m + M)
    Btp, Bpp = Bvec(lp, m + M)
    Ctp, Cpp = Cvec(lp, m + M)
    cBt, cBp = sp.conjugate(Btp), sp.conjugate(Bpp)
    cCt, cCp = sp.conjugate(Ctp), sp.conjugate(Cpp)
    cY = sp.conjugate(Ytp)
    for l in range(1, LMAX + 1):
        if abs(lp - l) > L:
            continue
        y = Y(l, m)
        Bt, Bp = Bvec(l, m)
        Ct, Cp = Cvec(l, m)
        eB = surf_strain2(Bt, Bp)
        checks = {
            "G0": (integ(cY * YL * y), ang2.G0[lp, l]),
            "GA_v": (integ((cBt * Bt + cBp * Bp) * YL) / Lp,
                     ang2.GA_v[lp, l]),
            "GA_w": (integ((cBt * Ct + cBp * Cp) * YL) / Lp,
                     ang2.GA_w[lp, l]),
            "GC_v": (integ((cCt * Bt + cCp * Bp) * YL),
                     ang2.GC_v[lp, l]),
            "Hiso_v": (integ((cBt * GtL + cBp * GpL) * y) / Lp,
                       ang2.Hiso_v[lp, l]),
            "Hiso_w": (integ((cCt * GtL + cCp * GpL) * y),
                       ang2.Hiso_w[lp, l]),
            "HV_v": (integ(cBt * (eB[0] * GtL + eB[2] * GpL)
                           + cBp * (eB[2] * GtL + eB[1] * GpL))
                     / Lp, ang2.HV_v[lp, l]),
            "HV_w": (integ(cCt * (eB[0] * GtL + eB[2] * GpL)
                           + cCp * (eB[2] * GtL + eB[1] * GpL)),
                     ang2.HV_w[lp, l]),
        }
        for name, (sym, num) in checks.items():
            sym = complex(sym)
            d = abs(sym - num) / max(1e-3, abs(sym))
            worst[name] = max(worst.get(name, 0.0), d)
print("A3b sympy cross-check worst rels:")
for name, dd in sorted(worst.items()):
    print("  %-7s %.2e" % (name, dd))
assert all(d < 1e-10 for d in worst.values())
print("PASS A3b")
