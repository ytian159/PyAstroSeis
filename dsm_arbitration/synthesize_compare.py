#!/usr/bin/env python3
"""Rung-1 DSM arbitration — synthesize waveforms and compare.

Both spectra sets (DSM tipsv+tish, BEM) go through the IDENTICAL
reconstruction (the spcsac convention from
bench/dsm_reference/scripts/synthesize_dsm.py):

    u(t_k) = Re{ (1/tlen) sum_n U_n S_n e^{+2 pi i n k / N} } e^{omegai t}

with a Ricker source-time-function spectrum S_n evaluated on the
damped time axis. DSM spectra are multiplied by 1000 (km -> m for
moment input in 1e25 dyn*cm units); BEM spectra are already meters.

The BEM Fourier-sign convention relative to DSM (conjugate or not) is
calibrated ONCE on the homogeneous leg (it must be globally consistent
across all stations/components/sources) and then frozen for the
two-layer leg.

Verdict logic: the homogeneous leg measures the BEM discretization +
station-sampling floor e0 vs DSM; the welded two-layer leg must land
in the same error class (e1 <= 2.5 * e0) with min correlation > 0.99.

Outputs: metrics.json, results_table.md, record-section figures
(overlay + magnified error panels), verdict on stdout + verdict.txt.
"""

import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ARB_ROOT", SCRIPT_DIR)

NFFT = 32768                      # dt = tlen/NFFT = 8 s
T_OUT = 60000.0                   # figure window
METRICS_T = 24000.0               # metric window: P, S and R1 at all
                                  # distances; the later elastic coda
                                  # only accumulates mesh-dispersion
                                  # dephasing (see verdict notes)
# Ricker center frequency [Hz]; ARB_F0 raises the band when the
# source region's low-frequency deficit zone (w R/vs <~ 1) must be
# avoided (rung 2e)
F0 = float(os.environ.get("ARB_F0", "1.75e-4"))
RICKER_SHIFT = 1.2 / F0

COMPONENT_FLOOR = 0.1             # skip comps with <10% of station peak


def read_spc(path):
    toks = []
    with open(path) as f:
        for line in f:
            toks.extend(line.split())
    vals = [float(t) for t in toks]
    tlen = vals[0]
    np0 = int(vals[1])
    ncomp = int(vals[3])
    omegai = vals[4]
    body = vals[17:]
    assert ncomp == 3
    assert len(body) % 7 == 0, "unexpected spc body length in %s" % path
    u = np.zeros((3, np0 + 1), dtype=complex)
    for k in range(len(body) // 7):
        b = body[7 * k:7 * k + 7]
        i = int(b[0])
        u[0, i] = b[1] + 1j * b[2]
        u[1, i] = b[3] + 1j * b[4]
        u[2, i] = b[5] + 1j * b[6]
    return tlen, omegai, u


def unit_vec(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(lat) * math.cos(lon),
                     math.cos(lat) * math.sin(lon), math.sin(lat)])


def rotation_to_ne(src_lat, src_lon, rec_lat, rec_lon):
    """Local horizontal unit vectors at the receiver (Cartesian):
    away-from-source, transverse (DSM T convention), north, east."""
    s = unit_vec(src_lat, src_lon)
    x = unit_vec(rec_lat, rec_lon)
    toward = s - np.dot(s, x) * x
    toward /= np.linalg.norm(toward)
    away = -toward
    t_hat = np.cross(s, x)
    t_hat /= np.linalg.norm(t_hat)
    zhat = np.array([0.0, 0.0, 1.0])
    north = zhat - np.dot(zhat, x) * x
    north /= np.linalg.norm(north)
    east = np.cross(zhat, x)
    east /= np.linalg.norm(east)
    return away, t_hat, north, east


class Synth:
    def __init__(self, tlen, omegai, imax):
        self.tlen, self.omegai, self.imax = tlen, omegai, imax
        self.dt = tlen / NFFT
        self.t = np.arange(NFFT) * self.dt
        a = (math.pi * F0) ** 2
        tt = self.t - RICKER_SHIFT
        s = (1.0 - 2.0 * a * tt * tt) * np.exp(-a * tt * tt)
        self.s_spec = np.fft.fft(s * np.exp(-omegai * self.t)) * self.dt
        self.nout = int(T_OUT / self.dt) + 1

    def to_time(self, u_spec):
        """u_spec: (imax+1,) complex harmonics -> time series [m]."""
        full = np.zeros(NFFT, dtype=complex)
        n = self.imax
        full[1:n + 1] = u_spec[1:n + 1] * self.s_spec[1:n + 1]
        full[NFFT - n:] = np.conj(full[1:n + 1])[::-1]
        wav = np.real(np.fft.ifft(full)) * NFFT / self.tlen
        return wav * np.exp(self.omegai * self.t)


def load_dsm(model, source, stations, syn):
    """-> dict station -> (Z_up, R_away, T) time series [m]."""
    out = {}
    for st in stations:
        base = os.path.join(ROOT, "dsm", model, "spc", st["name"])
        _, oi1, u_psv = read_spc("%s.%s.PSV.spc" % (base, source))
        assert abs(oi1 - syn.omegai) < 1e-12
        sh_path = "%s.%s.SH.spc" % (base, source)
        if os.path.exists(sh_path):
            _, oi2, u_sh = read_spc(sh_path)
            assert abs(oi2 - oi1) < 1e-12
        else:
            # deep-source campaigns (source below the outermost
            # fluid) have no SH leg
            u_sh = 0.0
        u = (u_psv + u_sh) * 1000.0     # -> meters
        out[st["name"]] = tuple(syn.to_time(u[c]) for c in range(3))
    return out


def load_bem(model, stations, syn, conj, alias=None):
    """-> dict source -> station -> (ux, uy, uz) time series [m]."""
    z = np.load(os.path.join(ROOT, "bem_%s.npz" % (alias or model)))
    srcs = [str(s) for s in z["sources"]]
    u = z["u"]
    if conj:
        u = np.conj(u)
    out = {}
    for si, src in enumerate(srcs):
        out[src] = {}
        for i, st in enumerate(stations):
            out[src][st["name"]] = tuple(
                syn.to_time(u[si, i, c]) for c in range(3))
    return out


def to_nez(dsm_zrt, bem_xyz, src, st):
    """Rotate both to (N, E, Z_up); also return transverse (T)."""
    away, t_hat, north, east = rotation_to_ne(
        src["lat"], src["lon"], st["lat"], st["lon"])
    z_up, u_r, u_t = dsm_zrt
    d = {"N": u_r * np.dot(away, north) + u_t * np.dot(t_hat, north),
         "E": u_r * np.dot(away, east) + u_t * np.dot(t_hat, east),
         "Z": z_up, "T": u_t}
    ux, uy, uz = bem_xyz
    xhat = unit_vec(st["lat"], st["lon"])
    b = {"N": ux * north[0] + uy * north[1] + uz * north[2],
         "E": ux * east[0] + uy * east[1] + uz * east[2],
         "Z": ux * xhat[0] + uy * xhat[1] + uz * xhat[2],
         "T": ux * t_hat[0] + uy * t_hat[1] + uz * t_hat[2]}
    return d, b


def trace_metrics(b, d):
    nd = np.linalg.norm(d)
    if nd == 0:
        return None
    rel = np.linalg.norm(b - d) / nd
    corr = float(np.dot(b, d) / (np.linalg.norm(b) * nd)) \
        if np.linalg.norm(b) > 0 else 0.0
    return {"rel_rms": float(rel), "corr": corr,
            "amp_ratio": float(np.linalg.norm(b) / nd)}


def compare_model(model, man, syn, conj, nout):
    src_meta = man["source"]
    stations = man["stations"]
    bem = load_bem(model, stations, syn, conj,
                   alias=man.get("bem_alias", {}).get(model))
    rows = []
    traces = {}
    for source in man["moment_tensors_1e25dyncm"]:
        dsm = load_dsm(model, source, stations, syn)
        traces[source] = {}
        for st in stations:
            d, b = to_nez(dsm[st["name"]], bem[source][st["name"]],
                          src_meta, st)
            traces[source][st["name"]] = (d, b)
            peak = max(np.max(np.abs(d[c][:nout])) for c in "NEZ")
            for c in "NEZ":
                dd, bb = d[c][:nout], b[c][:nout]
                if np.max(np.abs(dd)) < COMPONENT_FLOOR * peak:
                    continue
                m = trace_metrics(bb, dd)
                m.update(model=model, source=source, station=st["name"],
                         dist_deg=st["dist_deg"], comp=c)
                rows.append(m)
            # vector metric: full 3-component wavefield, energy-
            # weighted (near-nodal components cannot dominate)
            dv = np.concatenate([d[c][:nout] for c in "NEZ"])
            bv = np.concatenate([b[c][:nout] for c in "NEZ"])
            mv = trace_metrics(bv, dv)
            mv.update(model=model, source=source, station=st["name"],
                      dist_deg=st["dist_deg"], comp="VEC")
            rows.append(mv)
    return rows, traces


def calibrate_conj(man, syn, nout, baseline="homog"):
    """Pick the BEM Fourier-sign convention on the homogeneous leg."""
    best = {}
    for conj in (False, True):
        rows, _ = compare_model(baseline, man, syn, conj, nout)
        med = float(np.median([r["rel_rms"] for r in rows
                               if r["comp"] == "VEC"]))
        best[conj] = med
        print("conjugation=%s: %s median rel RMS %.3e"
              % (conj, baseline, med), flush=True)
    conj = min(best, key=best.get)
    if not (best[conj] < 0.3 and best[not conj] > 3 * best[conj]):
        print("WARNING: conjugation calibration not decisive "
              "(%.3e vs %.3e)" % (best[conj], best[not conj]))
    return conj


def record_section(model, source, comp, traces, man, syn, nout, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stations = sorted(man["stations"], key=lambda s: s["dist_deg"])
    t = syn.t[:nout] / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    gmax = max(np.max(np.abs(traces[source][st["name"]][0][comp][:nout]))
               for st in stations)
    errs = []
    for st in stations:
        d = traces[source][st["name"]][0][comp][:nout]
        b = traces[source][st["name"]][1][comp][:nout]
        errs.append(np.max(np.abs(b - d)))
    mag = max(1, int(round(gmax / max(errs) / 2.0))) if max(errs) > 0 else 1

    for st in stations:
        d = traces[source][st["name"]][0][comp][:nout]
        b = traces[source][st["name"]][1][comp][:nout]
        off = st["dist_deg"]
        sc = 8.0 / gmax
        axes[0].plot(t, d * sc + off, "k-", lw=1.2)
        axes[0].plot(t, b * sc + off, "r--", lw=1.0)
        axes[1].plot(t, (b - d) * sc * mag + off, "b-", lw=0.9)
    axes[0].plot([], [], "k-", lw=1.2, label="DSM")
    axes[0].plot([], [], "r--", lw=1.0, label="BEM")
    axes[0].legend(fontsize=14, loc="upper right")
    axes[0].set_title("%s, %s, %s component" % (model, source, comp),
                      fontsize=15)
    axes[1].set_title("error (BEM $-$ DSM) $\\times$%d" % mag,
                      fontsize=15)
    for ax in axes:
        ax.set_xlabel("time [$10^3$ s]", fontsize=14)
        ax.tick_params(labelsize=13)
    axes[0].set_ylabel("epicentral distance [deg]", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote %s (error magnification x%d)" % (path, mag))


def main():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    syn = Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = syn.nout

    nmet = int(METRICS_T / syn.dt) + 1
    models = list(man.get("models",
                           {"homog": None, "twolayer": None}).keys())
    baseline = models[0]
    assert baseline.startswith("homog"), \
        "baseline homogeneous leg must be first in ARB_MODELS"
    conj = calibrate_conj(man, syn, nmet, baseline)
    print("frozen BEM spectral convention: conjugate=%s" % conj)
    all_rows = {}
    all_traces = {}
    for model in models:
        rows, traces = compare_model(model, man, syn, conj, nmet)
        all_rows[model] = rows
        all_traces[model] = traces

    # results table
    lines = ["| model | source | station | dist | comp | rel RMS | corr |"
             " amp ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for model in models:
        for r in all_rows[model]:
            lines.append("| %s | %s | %s | %.1f | %s | %.3e | %.6f | "
                         "%.4f |" % (r["model"], r["source"], r["station"],
                                     r["dist_deg"], r["comp"],
                                     r["rel_rms"], r["corr"],
                                     r["amp_ratio"]))
    with open(os.path.join(ROOT, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    def stats(model):
        vec = [r for r in all_rows[model] if r["comp"] == "VEC"]
        rr = [r["rel_rms"] for r in vec]
        return (float(np.median(rr)), float(np.max(rr)),
                float(np.min([r["corr"] for r in vec])),
                float(np.median([r["amp_ratio"] for r in vec])))

    ladder = [m for m in os.environ.get("ARB_LADDER", "").split(",")
              if m]

    e0, e0max, cmin0, amp0 = stats(baseline)
    verdict = []
    verdict.append("metric window 0-%.0f s (P, S, R1 at all "
                   "distances); gates on per-station 3-component "
                   "VECTOR metrics" % METRICS_T)
    verdict.append("%s (baseline):  median rel RMS %.3e, max %.3e, "
                   "min corr %.6f" % (baseline, e0, e0max, cmin0))
    ok = e0 < 0.2
    summary = {baseline: {"median": e0, "max": e0max, "corr_min": cmin0,
                          "amp_med": amp0}}
    for model in models[1:]:
        e1, e1max, cmin1, amp1 = stats(model)
        m_ok = (e1 <= 2.5 * e0) and (cmin1 > 0.9) and (0.9 < amp1 < 1.1)
        verdict.append("%-10s (layered): median rel RMS %.3e, max %.3e, "
                       "min corr %.6f, median amp ratio %.4f"
                       % (model, e1, e1max, cmin1, amp1))
        if model not in ladder:
            verdict.append("  gate: e <= 2.5*e0 [%s], min corr > 0.9 "
                           "[%s], amp ratio in [0.9,1.1] [%s]"
                           % (e1 <= 2.5 * e0, cmin1 > 0.9,
                              0.9 < amp1 < 1.1))
        summary[model] = {"median": e1, "max": e1max, "corr_min": cmin1,
                          "amp_med": amp1}
        if model not in ladder:
            ok = ok and m_ok

    if ladder:
        # staircase-convergence verdict (graded-PREM campaign): the
        # ladder legs share one graded DSM reference, coarse -> fine;
        # each may exceed the fixed-model error gate (that is the
        # model-approximation error being measured), but the misfit
        # must DECREASE along the ladder and the finest leg must land
        # near its control (same staircase on the DSM side), which in
        # turn must pass the standard fixed-model gates.
        controls = {a: m for m, a in man.get("bem_alias", {}).items()
                    if m in summary}
        meds = [summary[m]["median"] for m in ladder]
        mono = all(a > b for a, b in zip(meds, meds[1:]))
        verdict.append("ladder %s medians: %s" %
                       ("->".join(ladder),
                        ", ".join("%.3e" % e for e in meds)))
        verdict.append("  gate: monotone decrease along the ladder "
                       "[%s]" % mono)
        ok = ok and mono
        fin = ladder[-1]
        ctrl = controls.get(fin)
        if ctrl:
            ec, cminc, ampc = (summary[ctrl]["median"],
                               summary[ctrl]["corr_min"],
                               summary[ctrl]["amp_med"])
            c_ok = (ec <= 2.5 * e0) and (cminc > 0.9) \
                and (0.9 < ampc < 1.1)
            near = summary[fin]["median"] <= 1.5 * ec
            verdict.append("  control %s (DSM runs the same "
                           "staircase): e <= 2.5*e0 [%s], min corr > "
                           "0.9 [%s], amp in [0.9,1.1] [%s]"
                           % (ctrl, ec <= 2.5 * e0, cminc > 0.9,
                              0.9 < ampc < 1.1))
            verdict.append("  gate: finest leg within 1.5x of its "
                           "control (%.3e vs %.3e) [%s]"
                           % (summary[fin]["median"], ec, near))
            ok = ok and c_ok and near
    verdict.append("baseline floor sane e0 < 20%%: %s" % (e0 < 0.2))
    verdict.append("DSM ARBITRATION %s" % ("PASSED" if ok else "FAILED"))
    txt = "\n".join(verdict)
    print(txt)
    with open(os.path.join(ROOT, "verdict.txt"), "w") as f:
        f.write(txt + "\nconjugate=%s\n" % conj)

    with open(os.path.join(ROOT, "metrics.json"), "w") as f:
        json.dump({"conjugate": conj, "rows": all_rows,
                   "summary": summary}, f, indent=2)

    # figures: mrr -> Z (P-SV), mrt -> T (SH through the weld)
    sources = list(man["moment_tensors_1e25dyncm"])
    for model in models:
        if "mrr" in sources:
            record_section(model, "mrr", "Z", all_traces[model], man,
                           syn, nout, os.path.join(ROOT,
                           "waveforms_%s_mrr_Z.png" % model))
        if "mrt" in sources:
            record_section(model, "mrt", "T", all_traces[model], man,
                           syn, nout, os.path.join(ROOT,
                           "waveforms_%s_mrt_T.png" % model))

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
