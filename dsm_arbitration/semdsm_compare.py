#!/usr/bin/env python3
"""Three-way arbitration on the SPHEROIDAL-ONLY channels (Z up,
R away): tipsv (certified converged, PSV spc) as reference, scoring
BOTH the SEM case and the BEM leg against it with the same erf-STF
velocity synthesis and band-limit as sem_compare.py.  The mrt/mrr Z
and R channels are purely spheroidal, so the unconverged shallow-tish
SH leg never enters.

usage: semdsm_compare.py <root> <sem_case_dir> <leg_npz> [source]
                         [hdur] [t_win]
"""

import json
import math
import os
import sys

import numpy as np

HDUR = 2000.0
BEM_CONJ = True                       # frozen BEM convention
# optional high-pass: excludes the quasi-static near-DC harmonics of
# the gravity-free fluid-core configuration (ill-posed w->0 limit,
# codes legitimately disagree there); 0 = off
F_LO = float(os.environ.get("ARB_F_LO", "0"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402

SOURCE_DECAY_MIMIC_TRIANGLE = 1.628
F_BAND = 5.3e-4


def _lp(x, dt):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), dt)
    X[fr > F_BAND] = 0.0
    if F_LO > 0:
        X[fr < F_LO] = 0.0
    return np.fft.irfft(X, len(x))


def spec_to_vel(u_spec, syn, nout, s_spec):
    full = np.zeros(sc.NFFT, dtype=complex)
    n = syn.imax
    full[1:n + 1] = u_spec[1:n + 1] * s_spec[1:n + 1]
    full[sc.NFFT - n:] = np.conj(full[1:n + 1])[::-1]
    wav = np.real(np.fft.ifft(full)) * sc.NFFT / syn.tlen
    wav = wav * np.exp(syn.omegai * syn.t)
    return np.gradient(wav, syn.dt)[:nout]


def sem_read(case, st, comp, tgrid, nout):
    p = os.path.join(case, "OUTPUT_FILES",
                     "DF.%s.MX%s.sem.ascii" % (st, comp))
    if not os.path.exists(p):
        return None
    d = np.loadtxt(p)
    t_leg = d[:, 0] + 1.5 * HDUR
    v = np.gradient(d[:, 1], d[1, 0] - d[0, 0])
    return np.interp(tgrid[:nout], t_leg, v, left=0.0, right=np.nan)


def metrics(b, d):
    nd = np.linalg.norm(d)
    return (float(np.linalg.norm(b - d) / nd),
            float(np.dot(b, d) / (np.linalg.norm(b) * nd + 1e-300)),
            float(np.linalg.norm(b) / nd))


def main():
    global HDUR
    ROOT = os.path.abspath(sys.argv[1])
    SEM = os.path.abspath(sys.argv[2])
    LEG = sys.argv[3]
    SOURCE = sys.argv[4] if len(sys.argv) > 4 else "mrt"
    HDUR = float(sys.argv[5]) if len(sys.argv) > 5 else 2000.0
    T_WIN = float(sys.argv[6]) if len(sys.argv) > 6 else 9000.0
    os.environ.setdefault("ARB_ROOT", ROOT)
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int(T_WIN / syn.dt) + 1
    hg = HDUR / SOURCE_DECAY_MIMIC_TRIANGLE
    ts = 1.5 * HDUR
    stf = 0.5 * (1.0 + np.vectorize(math.erf)((syn.t - ts) / hg))
    s_spec = np.fft.fft(stf * np.exp(-syn.omegai * syn.t)) * syn.dt

    z = np.load(os.path.join(ROOT, LEG))
    isrc = [str(s) for s in z["sources"]].index(SOURCE)
    u_bem = np.conj(z["u"][isrc]) if BEM_CONJ else z["u"][isrc]

    model = os.path.basename(LEG)[4:-4]   # bem_<model>.npz
    print("reference: tipsv %s %s | window %g s, band %g Hz"
          % (model, SOURCE, T_WIN, F_BAND))
    print("station dist ch | SEMvsDSM rel  corr   amp "
          "| BEMvsDSM rel  corr   amp")
    rows = []
    for i, st in enumerate(man["stations"]):
        base = os.path.join(ROOT, "dsm", model, "spc", st["name"])
        _, _, u_psv = sc.read_spc("%s.%s.PSV.spc" % (base, SOURCE))
        u_dsm = u_psv * 1000.0
        away, t_hat, north, east = sc.rotation_to_ne(
            man["source"]["lat"], man["source"]["lon"],
            st["lat"], st["lon"])
        xhat = sc.unit_vec(st["lat"], st["lon"])
        d = {"Z": _lp(spec_to_vel(u_dsm[0], syn, nout, s_spec), syn.dt),
             "R": _lp(spec_to_vel(u_dsm[1], syn, nout, s_spec), syn.dt)}
        vb = [spec_to_vel(u_bem[i][c], syn, nout, s_spec)
              for c in range(3)]
        b = {"Z": _lp(sum(vb[c] * xhat[c] for c in range(3)), syn.dt),
             "R": _lp(sum(vb[c] * away[c] for c in range(3)), syn.dt)}
        sN = sem_read(SEM, st["name"], "N", syn.t, nout)
        sE = sem_read(SEM, st["name"], "E", syn.t, nout)
        sZ = sem_read(SEM, st["name"], "Z", syn.t, nout)
        if sN is None or np.any(np.isnan(sN)) or np.any(np.isnan(sZ)):
            continue
        s = {"Z": _lp(sZ, syn.dt),
             "R": _lp(sN * (away @ north) + sE * (away @ east),
                      syn.dt)}
        peak = max(np.linalg.norm(d[c]) for c in "ZR")
        for c in ("Z", "R"):
            if np.linalg.norm(d[c]) < 0.05 * peak:
                continue
            m_s = metrics(s[c], d[c])
            m_b = metrics(b[c], d[c])
            rows.append(dict(station=st["name"], dist=st["dist_deg"],
                             comp=c, sem=m_s, bem=m_b))
            print("%-6s %5.1f  %s |     %6.3f %6.3f %6.3f "
                  "|     %6.3f %6.3f %6.3f"
                  % (st["name"], st["dist_deg"], c, *m_s, *m_b))
    for tag, idx in (("SEM-vs-DSM", "sem"), ("BEM-vs-DSM", "bem")):
        rel = [r[idx][0] for r in rows]
        cor = [r[idx][1] for r in rows]
        print("%s: median rel %.4f, median corr %.4f, min corr %.4f "
              "over %d traces" % (tag, np.median(rel), np.median(cor),
                                  min(cor), len(rows)))
    out = os.path.join(ROOT, "semdsm_%s_%s.json"
                       % (os.path.basename(SEM), SOURCE))
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
