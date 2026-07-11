#!/usr/bin/env python3
"""T-channel comparison of solver legs against the mini-tish
reference (both bem-format npz, earth-frame Cartesian spectra), in
the metric window. The mini-tish leg is the reference; candidate-leg
conjugation follows the frozen BEM convention (conjugate=True);
reference conjugation passed explicitly (from the V2 calibration).

usage: leg_t_compare.py <root(manifest)> <ref_npz_path> <ref_conj:0|1>
                        <leg_npz_path> [source]
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
REF = sys.argv[2]
REF_CONJ = bool(int(sys.argv[3]))
LEGP = sys.argv[4]
SOURCE = sys.argv[5] if len(sys.argv) > 5 else "mrt"

os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402


def t_traces(npz_path, conj, man, syn, nout):
    z = np.load(npz_path)
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


def main():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int(sc.METRICS_T / syn.dt) + 1
    ref = t_traces(REF, REF_CONJ, man, syn, nout)
    leg = t_traces(LEGP, True, man, syn, nout)
    print("T channel vs mini-tish reference (%s, %s):"
          % (os.path.basename(LEGP), SOURCE))
    print("station dist |  rel     corr    amp")
    rel = []
    for st in man["stations"]:
        d, b = ref[st["name"]], leg[st["name"]]
        nd = np.linalg.norm(d)
        r = np.linalg.norm(b - d) / nd
        rel.append(r)
        print("%-6s %5.1f | %6.3f  %6.3f  %6.3f"
              % (st["name"], st["dist_deg"], r,
                 np.dot(b, d) / (np.linalg.norm(b) * nd + 1e-300),
                 np.linalg.norm(b) / nd))
    print("median rel %.4f" % np.median(rel))


if __name__ == "__main__":
    main()
