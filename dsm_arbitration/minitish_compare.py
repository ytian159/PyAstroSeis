#!/usr/bin/env python3
"""Compare toroidal legs against the DSM tish SH spectra (SH file
ONLY — no PSV), per station/component in the metric window. Used for:
 V2 gate: mini-tish vs healthy tish (637-km source) — conventions +
 physics anchor; and for judging any solver's T channel where tish is
 valid. Conjugation of the candidate leg is calibrated globally like
 the BEM convention (best median), then frozen and reported.

usage: minitish_compare.py <root> <npz_leg> <model> [source]
  <npz_leg>: bem-format npz filename inside <root> (e.g.
             minitish_homog_q50.npz), earth-frame Cartesian spectra.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
LEG = sys.argv[2]
MODEL = sys.argv[3]
SOURCE = sys.argv[4] if len(sys.argv) > 4 else "mrt"

os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402


def main():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int(sc.METRICS_T / syn.dt) + 1
    sts = man["stations"]
    z = np.load(os.path.join(ROOT, LEG))
    isrc = [str(s) for s in z["sources"]].index(SOURCE)

    best = {}
    for conj in (False, True):
        u = np.conj(z["u"][isrc]) if conj else z["u"][isrc]
        rows = []
        for i, st in enumerate(sts):
            d = sc.read_spc(os.path.join(
                ROOT, "dsm", MODEL, "spc",
                "%s.%s.SH.spc" % (st["name"], SOURCE)))[2] * 1000.0
            dz = [syn.to_time(d[c])[:nout] for c in range(3)]
            away, t_hat, north, east = sc.rotation_to_ne(
                man["source"]["lat"], man["source"]["lon"],
                st["lat"], st["lon"])
            dd, bb = sc.to_nez(dz, [syn.to_time(u[i, c])[:nout]
                                    for c in range(3)],
                               man["source"], st)
            for c in ("N", "E", "T"):
                nd = np.linalg.norm(dd[c])
                if nd == 0:
                    continue
                rows.append(dict(
                    station=st["name"], dist=st["dist_deg"], comp=c,
                    rel=float(np.linalg.norm(bb[c] - dd[c]) / nd),
                    corr=float(np.dot(bb[c], dd[c])
                               / (np.linalg.norm(bb[c]) * nd + 1e-300)),
                    amp=float(np.linalg.norm(bb[c]) / nd)))
        med = float(np.median([r["rel"] for r in rows]))
        best[conj] = (med, rows)
        print("conj=%s: median rel %.4f" % (conj, med))
    conj = min(best, key=lambda k: best[k][0])
    med, rows = best[conj]
    print("frozen conj=%s" % conj)
    print("station dist comp |  rel     corr    amp")
    for r in rows:
        print("%-6s %5.1f  %-2s  | %6.3f  %6.3f  %6.3f"
              % (r["station"], r["dist"], r["comp"],
                 r["rel"], r["corr"], r["amp"]))
    print("VERDICT %s vs tish-SH (%s, %s): median rel %.4f, "
          "min corr %.4f, median amp %.4f over %d traces"
          % (LEG, MODEL, SOURCE, med,
             min(r["corr"] for r in rows),
             np.median([r["amp"] for r in rows]), len(rows)))
    out = os.path.join(ROOT, "minitish_compare_%s_%s.json"
                       % (MODEL, SOURCE))
    json.dump(dict(leg=LEG, conj=conj, rows=rows), open(out, "w"),
              indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
