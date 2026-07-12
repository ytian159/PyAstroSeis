#!/usr/bin/env python3
"""tipsv dispersion self-test on corefluid_q50:
A) tipsv-Q50 vs tipsv-elastic (same binary, same model geometry):
   isolates the realized physical-dispersion sign.
   causal    -> Q50 slower -> positive lag slope (~ +2.0-2.5 s/deg)
   anti      -> Q50 faster -> negative (~ -2.0-2.5)
   none      -> ~0
B) tipsv-elastic vs SEM-elastic: pure elastic cross-code residual
   (expect ~0 if geometry/conventions are consistent)."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                       else "dsm_arbitration_u3")
os.environ.setdefault("ARB_ROOT", ROOT)
import synthesize_compare as sc                     # noqa: E402
import semdsm_compare as sd                         # noqa: E402

SEM_EL = os.path.join(os.path.dirname(ROOT),
                      "validation/specfem_u3/sem32_elastic")
HDUR = 2000.0

man = json.load(open(os.path.join(ROOT, "manifest.json")))
syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
nout = int(9000.0 / syn.dt) + 1
hg = HDUR / sd.SOURCE_DECAY_MIMIC_TRIANGLE
stf = 0.5 * (1.0 + np.vectorize(math.erf)((syn.t - 1.5 * HDUR) / hg))
s_spec = np.fft.fft(stf * np.exp(-syn.omegai * syn.t)) * syn.dt


def bp(x):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), syn.dt)
    X[(fr < 1e-4) | (fr > 5.3e-4)] = 0.0
    return np.fft.irfft(X, len(x))


def lagscan(a, b):
    nlag = int(1500 / syn.dt)
    lags = np.arange(-nlag, nlag + 1)
    cc = np.array([np.dot(a[max(0, l):nout + min(0, l)],
                          b[max(0, -l):nout - max(0, l)]) for l in lags])
    return lags[np.argmax(cc)] * syn.dt


def dsm_z(subdir, st):
    p = os.path.join(ROOT, "dsm", "corefluid_q50", subdir,
                     "%s.mrt.PSV.spc" % st)
    toks = open(p).read().split()
    vals = [float(t) for t in toks]
    body = vals if len(vals) % 7 == 0 else vals[17:]  # headerless OK
    u = np.zeros((3, 257), dtype=complex)
    for j in range(len(body) // 7):
        b = body[7 * j:7 * j + 7]
        i = int(b[0])
        u[0, i] = b[1] + 1j * b[2]
        u[1, i] = b[3] + 1j * b[4]
        u[2, i] = b[5] + 1j * b[6]
    return bp(sd.spec_to_vel(u[0] * 1000.0, syn, nout, s_spec))


print("A) tipsv-Q50 vs tipsv-elastic (positive = Q50 late = causal)")
resA, resB = [], []
for st in man["stations"]:
    q = dsm_z("spc", st["name"])
    e = dsm_z("spc_el", st["name"])
    lag = lagscan(q, e)
    amp = np.linalg.norm(q) / np.linalg.norm(e)
    resA.append((st["dist_deg"], lag))
    sZ = sd.sem_read(SEM_EL, st["name"], "Z", syn.t, nout)
    line = "%-6s %6.1f  lagA %5.0f  ampQ50/el %.3f" % (
        st["name"], st["dist_deg"], lag, amp)
    if sZ is not None and not np.any(np.isnan(sZ)):
        lagB = lagscan(e, bp(sZ))
        resB.append((st["dist_deg"], lagB))
        line += "   lagB(tipsv_el vs SEM_el) %5.0f" % lagB
    print(line)
for tag, res in (("A tipsv Q50-vs-el", resA),
                 ("B tipsv_el-vs-SEM_el", resB)):
    dd = np.array([r[0] for r in res])
    lg = np.array([r[1] for r in res])
    print("%s slope: %.2f s/deg" % (tag, np.sum(dd * lg) / np.sum(dd * dd)))
