#!/usr/bin/env python3
"""V-a anchor: toroidal mode-sum vs the V2-validated mini-tish forced
solution (same npz layout), T channel at the manifest stations.
Scans the relative conjugation; reports per-station rel/corr/amp.

usage: tormodes_check.py <root> <npz_A(ref=minitish)> <npz_B(modesum)>
                         [source]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
REF, LEG = sys.argv[2], sys.argv[3]
SOURCE = sys.argv[4] if len(sys.argv) > 4 else "mrt"
os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402

man = json.load(open(os.path.join(ROOT, "manifest.json")))
syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
nout = syn.nout


def t_traces(path, conj):
    z = np.load(path)
    isrc = [str(s) for s in z["sources"]].index(SOURCE)
    u = np.conj(z["u"][isrc]) if conj else z["u"][isrc]
    out = {}
    for i, st in enumerate(man["stations"]):
        _, t_hat, _, _ = sc.rotation_to_ne(
            man["source"]["lat"], man["source"]["lon"],
            st["lat"], st["lon"])
        out[st["name"]] = sum(syn.to_time(u[i, c])[:nout] * t_hat[c]
                              for c in range(3))
    return out


best = None
for cr in (False, True):
    ref = t_traces(REF, cr)
    for cl in (False, True):
        leg = t_traces(LEG, cl)
        rels = [np.linalg.norm(leg[n] - ref[n]) / np.linalg.norm(ref[n])
                for n in ref]
        med = float(np.median(rels))
        print("ref_conj=%s leg_conj=%s: median rel %.4f"
              % (cr, cl, med))
        if best is None or med < best[0]:
            best = (med, cr, cl)
med, cr, cl = best
print()
print("BEST ref_conj=%s leg_conj=%s" % (cr, cl))
ref = t_traces(REF, cr)
leg = t_traces(LEG, cl)
print("station dist |  rel     corr    amp")
for st in man["stations"]:
    d, b = ref[st["name"]], leg[st["name"]]
    nd = np.linalg.norm(d)
    print("%-6s %5.1f | %6.3f  %6.3f  %6.3f"
          % (st["name"], st["dist_deg"],
             np.linalg.norm(b - d) / nd,
             np.dot(b, d) / (np.linalg.norm(b) * nd + 1e-300),
             np.linalg.norm(b) / nd))
print("median rel %.4f" % med)
