#!/usr/bin/env python3
"""Score the spectral ak135 hdur80 leg against SEM-Ref (NEX320),
SEM-1 (NEX128) and DSM with the campaign comparator protocol
(compare_moment_f0125_hdur80.py): SEM time offset 120 s, baseline
demean over [0,48] s, DT 0.1 resample, per-station surface-wave
group-velocity windows 4.8-2.9 km/s, mid-range medians AZ04-AZ14.

usage: compare_bridge.py <spectral_uxyz_dir> <tag> [--plots]
writes out_ak135/compare_<tag>.{json,md} (+ waveform figures with
--plots, which needs matplotlib).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_common as bc                          # noqa: E402

SPEC_DIR, TAG = sys.argv[1], sys.argv[2]
PLOTS = "--plots" in sys.argv[3:]
OUTDIR = os.path.join(bc.HERE, "out_ak135")

DT = 0.1
SOURCE_SHIFT = 120.0
BASELINE_END = 48.0
KM_PER_DEG = 2.0 * math.pi * 6371.0 / 360.0
VMAX, VMIN = 4.8, 2.9
MID = set(range(3, 14))


def local_basis(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    radial = np.array([math.cos(lat) * math.cos(lon),
                       math.cos(lat) * math.sin(lon),
                       math.sin(lat)])
    north = np.array([-math.sin(lat) * math.cos(lon),
                      -math.sin(lat) * math.sin(lon),
                      math.cos(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    return east, north, radial


def load_sem(dirname, sta):
    out = {}
    for comp in "ENZ":
        d = np.loadtxt(os.path.join(
            dirname, "DF.%s.BX%s.sem.ascii" % (sta, comp)))
        t = d[:, 0] + SOURCE_SHIFT
        v = d[:, 1].copy()
        pre = (t >= 0.0) & (t <= BASELINE_END)
        v -= float(v[pre].mean())
        out["t"], out[comp.lower()] = t, v
    return out


def load_dsm(sta):
    t, n, e, z = np.loadtxt(os.path.join(bc.DSM_DIR, sta + ".txt"),
                            unpack=True)
    return {"t": t, "n": n, "e": e, "z": z}


def load_spec(sta, lat, lon):
    d = np.loadtxt(os.path.join(SPEC_DIR,
                                "Uxyz_spectral_%s.txt" % sta))
    e_v, n_v, z_v = local_basis(lat, lon)
    xyz = d[:, 1:4]
    return {"t": d[:, 0], "e": xyz @ e_v, "n": xyz @ n_v,
            "z": xyz @ z_v}


def resample(rec, t):
    return {c: np.interp(t, rec["t"], rec[c]) for c in "enz"}


def corr(a, b):
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0.0 else 0.0


def best_lag(a, b, max_lag=30.0):
    nl = int(round(max_lag / DT))
    best, bl = -2.0, 0
    for lag in range(-nl, nl + 1):
        if lag < 0:
            v = corr(a[-lag:], b[:lag])
        elif lag > 0:
            v = corr(a[:-lag], b[lag:])
        else:
            v = corr(a, b)
        if v > best:
            best, bl = v, lag
    return best, bl * DT


def pair_metrics(test, ref, sel):
    tw, rw = test[sel], ref[sel]
    rms_ref = float(np.sqrt(np.mean(rw ** 2)))
    peak_ref = float(np.max(np.abs(rw)))
    blc, blag = best_lag(tw, rw)
    return {
        "corr_win": corr(tw, rw),
        "bestlag_corr_win": blc,
        "bestlag_s": blag,
        "peak_ratio_win": float(np.max(np.abs(tw)) / peak_ref)
        if peak_ref > 0 else math.nan,
        "rms_ratio_win": float(np.sqrt(np.mean(tw ** 2)) / rms_ref)
        if rms_ref > 0 else math.nan,
        "rms_residual_pct": float(
            100.0 * np.sqrt(np.mean((tw - rw) ** 2)) / rms_ref)
        if rms_ref > 0 else math.nan,
    }


PAIRS = [("spec_semref", "SPECTRAL / SEM-Ref"),
         ("spec_dsm", "SPECTRAL / DSM"),
         ("sem1_semref", "SEM-1 / SEM-Ref"),
         ("semref_dsm", "SEM-Ref / DSM")]


def build_rows():
    rows, cache = [], {}
    for i, (sta, lat, lon, dist) in enumerate(zip(
            bc.STATIONS, bc.LATS, bc.LONS, bc.DISTS)):
        legs = {"spec": load_spec(sta, lat, lon),
                "semref": load_sem(bc.SEMREF_DIR, sta),
                "sem1": load_sem(bc.SEM1_DIR, sta),
                "dsm": load_dsm(sta)}
        tmax = min(v["t"][-1] for v in legs.values())
        t = np.arange(0.0, tmax + DT / 2.0, DT)
        li = {k: resample(v, t) for k, v in legs.items()}
        d_km = dist * KM_PER_DEG
        win = (max(SOURCE_SHIFT + d_km / VMAX, t[0]),
               min(SOURCE_SHIFT + d_km / VMIN, t[-1] - 50.0))
        sel = (t >= win[0]) & (t <= win[1])
        row = {"station": sta, "distance_deg": dist,
               "window_s": list(win)}
        for key, _ in PAIRS:
            a, b = key.split("_")
            row[key] = {c: pair_metrics(li[a][c], li[b][c], sel)
                        for c in ("n", "z")}
        rows.append(row)
        cache[sta] = (t, li)
    return rows, cache


def median(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else math.nan


def summarize(rows, pair, comp):
    sub = [r for i, r in enumerate(rows) if i in MID]
    keys = ["corr_win", "bestlag_corr_win", "bestlag_s",
            "peak_ratio_win", "rms_ratio_win", "rms_residual_pct"]
    return {k: median([r[pair][comp][k] for r in sub]) for k in keys}


def write_plots(rows, cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                         "axes.labelsize": 14,
                         "legend.fontsize": 12})
    show = ["AZ02", "AZ06", "AZ10", "AZ13", "AZ15", "AZ18"]
    by = {r["station"]: r for r in rows}
    for comp in ("z", "n"):
        fig, axes = plt.subplots(len(show), 2, figsize=(19, 12.5),
                                 sharex=True)
        for k, sta in enumerate(show):
            i = bc.STATIONS.index(sta)
            t, li = cache[sta]
            r = by[sta]
            ref = li["semref"][comp]
            sc = float(np.max(np.abs(ref))) or 1.0
            mag = 10
            ax = axes[k, 0]
            ax.plot(t, li["semref"][comp] / sc, "k-", lw=1.3,
                    label="SEM-Ref")
            ax.plot(t, li["dsm"][comp] / sc, "-", color="tab:green",
                    lw=0.9, label="DSM")
            ax.plot(t, li["spec"][comp] / sc, "--",
                    color="tab:red", lw=1.0, label="SPECTRAL")
            ax.set_ylabel("%s %gdeg\nU%s" % (sta,
                                             bc.DISTS[i], comp))
            if k == 0:
                ax.set_title("ak135 Mrr hdur80, U%s" % comp)
                ax.legend(loc="upper left", ncol=3, frameon=False)
            ax.grid(alpha=0.18)
            ax = axes[k, 1]
            ax.plot(t, mag * (li["sem1"][comp] - ref) / sc, "-",
                    color="tab:blue", lw=0.9, label="SEM-1 - SEM-Ref")
            ax.plot(t, mag * (li["spec"][comp] - ref) / sc, "--",
                    color="tab:red", lw=1.0,
                    label="SPECTRAL - SEM-Ref")
            ax.axhline(0.0, color="k", lw=0.5)
            spec_rms = r["spec_semref"][comp]["rms_residual_pct"]
            sem1_rms = r["sem1_semref"][comp]["rms_residual_pct"]
            ax.text(0.99, 0.9,
                    "RMS: SPECTRAL %.1f%%  SEM-1 %.1f%%"
                    % (spec_rms, sem1_rms), transform=ax.transAxes,
                    ha="right", va="top", fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75))
            if k == 0:
                ax.set_title("error vs SEM-Ref (x%d)" % mag)
                ax.legend(loc="upper left", ncol=2, frameon=False)
            ax.grid(alpha=0.18)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.tight_layout()
        out = os.path.join(OUTDIR,
                           "wf_%s_U%s.png" % (TAG, comp))
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


def main():
    rows, cache = build_rows()
    summary = {"tag": TAG, "spectral_dir": SPEC_DIR,
               "midrange_40_140deg": {}}
    for comp in ("z", "n"):
        summary["midrange_40_140deg"][comp] = {
            key: summarize(rows, key, comp) for key, _ in PAIRS}
    summary["rows"] = rows
    os.makedirs(OUTDIR, exist_ok=True)
    pj = os.path.join(OUTDIR, "compare_%s.json" % TAG)
    open(pj, "w").write(json.dumps(summary, indent=2) + "\n")
    lines = ["# ak135 Mrr hdur80: spectral leg vs SEM-Ref/SEM-1/DSM",
             "", "tag: " + TAG,
             "Mid-range medians AZ04-AZ14 (40-140 deg).", "",
             "| comp | pair | corr | bestlag corr | lag s "
             "| peak ratio | RMS ratio | RMS resid % |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for comp in ("z", "n"):
        for key, label in PAIRS:
            s = summary["midrange_40_140deg"][comp][key]
            lines.append(
                "| %s | %s | %.5f | %.5f | %.2f | %.5f | %.5f "
                "| %.3f |" % (comp.upper(), label, s["corr_win"],
                              s["bestlag_corr_win"], s["bestlag_s"],
                              s["peak_ratio_win"],
                              s["rms_ratio_win"],
                              s["rms_residual_pct"]))
    pm = os.path.join(OUTDIR, "compare_%s.md" % TAG)
    open(pm, "w").write("\n".join(lines) + "\n")
    print(open(pm).read())
    if PLOTS:
        write_plots(rows, cache)


if __name__ == "__main__":
    main()
