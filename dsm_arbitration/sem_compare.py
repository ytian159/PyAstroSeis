#!/usr/bin/env python3
"""Compare solver legs against SPECFEM3D_GLOBE uniform3 seismograms,
in VELOCITY over the metric window (velocity scoring removes the DC
offset that the k >= 1 truncated spectral synthesis cannot carry).

Conventions: SPECFEM ran the CMT erf quasi-Heaviside with half
duration HDUR (its time axis starts at -t0 = -1.5*HDUR, step centered
at t = 0); spectral legs (BEM/mini-tish/tipsv aliases) are synthesized
with the bit-identical erf STF centered at ts = 1.5*HDUR on the
0-based axis (dsm_reference 'heaviside' convention), so
u_leg(t) corresponds to u_sem(t - ts). Conjugation calibrated
globally, frozen, reported.

usage: sem_compare.py <root> <sem_case_dir> <leg_npz> [source] [hdur]
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SEM = os.path.abspath(sys.argv[2])
LEG = sys.argv[3]
SOURCE = sys.argv[4] if len(sys.argv) > 4 else "mrt"
HDUR = float(sys.argv[5]) if len(sys.argv) > 5 else 2000.0
T_WIN = float(sys.argv[6]) if len(sys.argv) > 6 else None

os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402

SOURCE_DECAY_MIMIC_TRIANGLE = 1.628
F_BAND = 5.3e-4     # spectral legs end at k = imax (5.26e-4 Hz):
                    # band-limit BOTH sides for a like-for-like score


def _lp(x, dt):
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), dt) > F_BAND] = 0.0
    return np.fft.irfft(X, len(x))


def leg_velocity(man, u_spec3, syn, nout):
    """Velocity traces (3, nout) from earth-frame Cartesian spectra:
    apply the erf STF spectrum, then differentiate in time."""
    hg = HDUR / SOURCE_DECAY_MIMIC_TRIANGLE
    ts = 1.5 * HDUR
    t = syn.t
    stf = 0.5 * (1.0 + np.vectorize(math.erf)((t - ts) / hg))
    s_spec = np.fft.fft(stf * np.exp(-syn.omegai * t)) * syn.dt
    out = []
    for c in range(3):
        full = np.zeros(sc.NFFT, dtype=complex)
        n = syn.imax
        full[1:n + 1] = u_spec3[c][1:n + 1] * s_spec[1:n + 1]
        full[sc.NFFT - n:] = np.conj(full[1:n + 1])[::-1]
        wav = np.real(np.fft.ifft(full)) * sc.NFFT / syn.tlen
        wav = wav * np.exp(syn.omegai * t)
        out.append(np.gradient(wav, syn.dt)[:nout])
    return out


def sem_velocity(case, st, comp, tgrid, nout, dt):
    """SEM displacement -> velocity, resampled onto the leg's 0-based
    axis (t_leg = t_sem + 1.5*HDUR)."""
    p = os.path.join(case, "OUTPUT_FILES",
                     "DF.%s.MX%s.sem.ascii" % (st, comp))
    if not os.path.exists(p):
        alt = [f for f in os.listdir(os.path.join(case, "OUTPUT_FILES"))
               if f.startswith("DF.%s." % st) and
               f.endswith("%s.sem.ascii" % comp)]
        if not alt:
            return None
        p = os.path.join(case, "OUTPUT_FILES", alt[0])
    d = np.loadtxt(p)
    t_leg = d[:, 0] + 1.5 * HDUR
    v = np.gradient(d[:, 1], d[1, 0] - d[0, 0])
    return np.interp(tgrid[:nout], t_leg, v, left=0.0, right=np.nan)


def main():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int((T_WIN or sc.METRICS_T) / syn.dt) + 1
    z = np.load(os.path.join(ROOT, LEG))
    isrc = [str(s) for s in z["sources"]].index(SOURCE)
    sts = man["stations"]

    best = {}
    for conj in (False, True):
        u = np.conj(z["u"][isrc]) if conj else z["u"][isrc]
        rows = []
        for i, st in enumerate(sts):
            vleg = leg_velocity(man, u[i], syn, nout)
            away, t_hat, north, east = sc.rotation_to_ne(
                man["source"]["lat"], man["source"]["lon"],
                st["lat"], st["lon"])
            xhat = sc.unit_vec(st["lat"], st["lon"])
            b = {"N": sum(vleg[c] * north[c] for c in range(3)),
                 "E": sum(vleg[c] * east[c] for c in range(3)),
                 "Z": sum(vleg[c] * xhat[c] for c in range(3)),
                 "T": sum(vleg[c] * t_hat[c] for c in range(3))}
            snez = {}
            for comp in ("N", "E", "Z"):
                v = sem_velocity(SEM, st["name"], comp, syn.t, nout,
                                 syn.dt)
                if v is None or np.any(np.isnan(v)):
                    snez = None
                    break
                snez[comp] = v
            if snez is None:
                continue
            snez["T"] = (snez["N"] * (t_hat @ north)
                         + snez["E"] * (t_hat @ east))
            for c in list(snez):
                snez[c] = _lp(snez[c], syn.dt)
            for c in list(b):
                b[c] = _lp(b[c], syn.dt)
            peak = max(np.max(np.abs(snez[c])) for c in "NEZ")
            for c in ("N", "E", "Z", "T"):
                nd = np.linalg.norm(snez[c])
                if nd == 0 or np.max(np.abs(snez[c])) < 0.05 * peak:
                    continue
                rows.append(dict(
                    station=st["name"], dist=st["dist_deg"], comp=c,
                    rel=float(np.linalg.norm(b[c] - snez[c]) / nd),
                    corr=float(np.dot(b[c], snez[c])
                               / (np.linalg.norm(b[c]) * nd + 1e-300)),
                    amp=float(np.linalg.norm(b[c]) / nd)))
        med = float(np.median([r["rel"] for r in rows])) if rows \
            else np.inf
        best[conj] = (med, rows)
        print("conj=%s: median rel %.4f (n=%d)"
              % (conj, med, len(rows)))
    conj = min(best, key=lambda k: best[k][0])
    med, rows = best[conj]
    print("frozen conj=%s" % conj)
    print("station dist comp |  rel     corr    amp")
    for r in rows:
        print("%-6s %5.1f  %-2s  | %6.3f  %6.3f  %6.3f"
              % (r["station"], r["dist"], r["comp"],
                 r["rel"], r["corr"], r["amp"]))
    print("VERDICT %s vs SEM (%s): median rel %.4f, min corr %.4f "
          "over %d traces" % (LEG, os.path.basename(SEM), med,
                              min(r["corr"] for r in rows), len(rows)))
    out = os.path.join(ROOT, "sem_compare_%s_%s.json"
                       % (os.path.basename(SEM), SOURCE))
    json.dump(dict(leg=LEG, sem=SEM, conj=conj, hdur=HDUR, rows=rows),
              open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
