#!/usr/bin/env python3
"""ON-OFF alignment analysis for the moment-fitted RHS test
(docs/moment_fitted_rhs.md sec 4b, Codex-recommended observable).

Per station/component over the metric window:

    r  = u_DSM - u_raw    (the missing field)
    du = u_fit - u_raw    (what the fit added; = A^-1 db, exactly
                           linear in the correction)
    C  = |<r,du>| / (||r|| ||du||)    alignment
    g  = <r,du> / <r,r>               recovered fraction of residual
    a  = <r,du> / <du,du>             dose (best-fit scale of du; ~1
                                       means the added field has the
                                       right size, not just direction)

usage: mfit_alignment.py <raw_root> <fit_root> [model]
(<*_root> must each hold manifest.json, dsm/, bem_<model>.npz; the
DSM references must be identical). Writes <fit_root>/mfit_alignment.json.
"""

import json
import os
import sys

import numpy as np

RAW = os.path.abspath(sys.argv[1])
FIT = os.path.abspath(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "homog_q50"

os.environ["ARB_ROOT"] = FIT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402


def main():
    man = json.load(open(os.path.join(FIT, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int(sc.METRICS_T / syn.dt) + 1
    mj = json.load(open(os.path.join(FIT, "metrics.json"))) \
        if os.path.exists(os.path.join(FIT, "metrics.json")) else {}
    conj = bool(mj.get("conjugate", True))
    stations = man["stations"]

    sc.ROOT = FIT
    bem_fit = sc.load_bem(MODEL, stations, syn, conj)
    sc.ROOT = RAW
    bem_raw = sc.load_bem(MODEL, stations, syn, conj)

    rows = []
    for source in man["moment_tensors_1e25dyncm"]:
        dsm = sc.load_dsm(MODEL, source, stations, syn)
        for st in stations:
            d, bf = sc.to_nez(dsm[st["name"]],
                              bem_fit[source][st["name"]],
                              man["source"], st)
            _, br = sc.to_nez(dsm[st["name"]],
                              bem_raw[source][st["name"]],
                              man["source"], st)
            for c in ("N", "E", "Z", "T"):
                dd = d[c][:nout]
                r = dd - br[c][:nout]
                du = bf[c][:nout] - br[c][:nout]
                nr, ndu = np.linalg.norm(r), np.linalg.norm(du)
                nd = np.linalg.norm(dd)
                if nd == 0 or nr == 0:
                    continue
                ip = float(np.dot(r, du))
                rows.append(dict(
                    source=source, station=st["name"],
                    dist_deg=st["dist_deg"], comp=c,
                    C=(abs(ip) / (nr * ndu) if ndu > 0 else 0.0),
                    g=ip / nr ** 2,
                    a=(ip / ndu ** 2 if ndu > 0 else 0.0),
                    resid_frac=nr / nd,
                    rel_raw=float(np.linalg.norm(br[c][:nout] - dd)
                                  / nd),
                    rel_fit=float(np.linalg.norm(bf[c][:nout] - dd)
                                  / nd),
                    amp_raw=float(np.linalg.norm(br[c][:nout]) / nd),
                    amp_fit=float(np.linalg.norm(bf[c][:nout]) / nd)))

    hdr = ("src  station  dist   comp  C      g      a      "
           "resid/d  rel raw->fit     amp raw->fit")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-4s %-8s %5.1f  %-4s %6.3f %6.2f %6.2f  %7.3f  "
              "%6.3f->%6.3f  %6.3f->%6.3f"
              % (r["source"], r["station"], r["dist_deg"], r["comp"],
                 r["C"], r["g"], r["a"], r["resid_frac"],
                 r["rel_raw"], r["rel_fit"],
                 r["amp_raw"], r["amp_fit"]))
    big = [r for r in rows if r["resid_frac"] > 0.5]
    if big:
        print("\nchannels with resid/d > 0.5 (the quasi-static-"
              "dominated set): median C = %.3f, median g = %.2f, "
              "median a = %.2f over %d traces"
              % (np.median([r["C"] for r in big]),
                 np.median([r["g"] for r in big]),
                 np.median([r["a"] for r in big]), len(big)))
    with open(os.path.join(FIT, "mfit_alignment.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote %s" % os.path.join(FIT, "mfit_alignment.json"))


if __name__ == "__main__":
    main()
