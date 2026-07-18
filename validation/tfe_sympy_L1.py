#!/usr/bin/env python3
"""sympy exact-integral check of ALL ReliefCouplings matrices at
L=1, M=0 (the translation-gate configuration), m in (0, 1)."""
import sys
import numpy as np
import sympy as sp

sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis.spectral_tfe import ReliefCouplings   # noqa

th, ph = sp.symbols("theta phi", real=True)
L, M = 1, 0
LMAX = 4


def Y(l, mm):
    return sp.Ynm(l, mm, th, ph).expand(func=True)


def Bvec(l, mm):
    y = Y(l, mm)
    return sp.diff(y, th), sp.diff(y, ph) / sp.sin(th)


def Cvec(l, mm):
    f = 1 / sp.sqrt(sp.Integer(l) * (l + 1))
    y = Y(l, mm)
    return -f * sp.I * mm * y / sp.sin(th), f * sp.diff(y, th)


def strain2(vt, vp):
    cot = sp.cos(th) / sp.sin(th)
    e_tt = 2 * sp.diff(vt, th)
    e_pp = 2 * (sp.diff(vp, ph) / sp.sin(th) + cot * vt)
    e_tp = sp.diff(vp, th) - cot * vp + sp.diff(vt, ph) / sp.sin(th)
    return e_tt, e_pp, e_tp


def integ(expr):
    return complex(sp.integrate(sp.integrate(
        sp.simplify(expr) * sp.sin(th), (ph, 0, 2 * sp.pi)),
        (th, 0, sp.pi)).evalf())


YL = Y(L, M)
GtL = sp.diff(YL, th)
GpL = sp.diff(YL, ph) / sp.sin(th)

for m in (0, 1):
    ang = ReliefCouplings(L, M, m, LMAX)
    worst = {}
    for lp in range(max(1, abs(m + M)), LMAX + 1):
        Lp = sp.Integer(lp) * (lp + 1)
        Ytp = Y(lp, m + M)
        Btp, Bpp = Bvec(lp, m + M)
        Ctp, Cpp = Cvec(lp, m + M) if lp >= 1 else (0, 0)
        cY = sp.conjugate(Ytp)
        cBt, cBp = sp.conjugate(Btp), sp.conjugate(Bpp)
        cCt, cCp = sp.conjugate(Ctp), sp.conjugate(Cpp)
        for l in range(max(1, abs(m)), LMAX + 1):
            if abs(lp - l) > L:
                continue
            y = Y(l, m)
            Bt, Bp = Bvec(l, m)
            Ct, Cp = Cvec(l, m)
            eB = strain2(Bt, Bp)
            eC = strain2(Ct, Cp)
            BG = Bt * GtL + Bp * GpL
            CG = Ct * GtL + Cp * GpL
            vB = (eB[0] * GtL + eB[2] * GpL,
                  eB[2] * GtL + eB[1] * GpL)
            vC = (eC[0] * GtL + eC[2] * GpL,
                  eC[2] * GtL + eC[1] * GpL)
            checks = {
                "G0": (integ(cY * YL * y), ang.G0[lp, l]),
                "GA_v": (integ((cBt * Bt + cBp * Bp) * YL) / Lp,
                         ang.GA_v[lp, l]),
                "GA_w": (integ((cBt * Ct + cBp * Cp) * YL) / Lp,
                         ang.GA_w[lp, l]),
                "GC_v": (integ((cCt * Bt + cCp * Bp) * YL),
                         ang.GC_v[lp, l]),
                "GC_w": (integ((cCt * Ct + cCp * Cp) * YL),
                         ang.GC_w[lp, l]),
                "H0_s": (integ(cY * BG), ang.H0_s[lp, l]),
                "H0_t": (integ(cY * CG), ang.H0_t[lp, l]),
                "Hiso_v": (integ((cBt * GtL + cBp * GpL) * y) / Lp,
                           ang.Hiso_v[lp, l]),
                "Hiso_w": (integ((cCt * GtL + cCp * GpL) * y),
                           ang.Hiso_w[lp, l]),
                "HV_v": (integ(cBt * vB[0] + cBp * vB[1]) / Lp,
                         ang.HV_v[lp, l]),
                "HV_w": (integ(cCt * vB[0] + cCp * vB[1]),
                         ang.HV_w[lp, l]),
                "HW_v": (integ(cBt * vC[0] + cBp * vC[1]) / Lp,
                         ang.HW_v[lp, l]),
                "HW_w": (integ(cCt * vC[0] + cCp * vC[1]),
                         ang.HW_w[lp, l]),
            }
            for name, (s_, n_) in checks.items():
                s_ = complex(s_)
                d = abs(s_ - n_) / max(1e-3, abs(s_))
                if d > worst.get(name, (0.0, 0, 0))[0]:
                    worst[name] = (d, lp, l, s_, n_)
    print("m=%d worst rels:" % m)
    bad = False
    for name in sorted(worst):
        d, lp, l, s_, n_ = worst[name]
        flag = ""
        if d > 1e-10:
            bad = True
            flag = "   <-- MISMATCH sym %s num %s (lp=%d,l=%d)" \
                % (s_, n_, lp, l)
        print("  %-7s %.2e%s" % (name, d, flag))
    print("  ->", "FAIL" if bad else "ALL PASS")
