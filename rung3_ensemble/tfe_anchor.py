#!/usr/bin/env python3
"""A3b stage-3 quantitative anchor: first-order TFE Y20-CMB relief
(pyastroseis.spectral_tfe.relief_spectra) vs the BEM rung-3
CMB-topography ensemble (run_ensemble.py members y20_2500 / y20_5000
minus base) on corefluid_q50, 637-km source, ULP band k = 1..138.

This is the first QUANTITATIVE test of the finite-L tilt/slip
machinery (fluid pressure ciso, tangential slip in u.n, HV/HW
strain couplings) — the Y00 FD gates only reach the value-transfer
terms because grad1 Y00 = 0.

Relief normalization: meshgen.relief_ylm PEAK-normalizes over the
evaluated mesh vertices, h = amp * Re(Y20)/max|Re(Y20)|_vertices,
so the TFE coefficient is h20 = amp / peak with
peak = max_vertices |Ybar20| in the fully-normalized convention —
reproduced here from the same seeded mesh construction path.

Metric: complex q = du/u per (source, station, ZRT channel, k) —
the BEM station radius (face centres ~6351 km), mesh amplitude bias
and moment scaling cancel to first order between numerator and
denominator. Conventions: RAW BEM npz and RAW spectral output share
the same time convention (both are conj-of-DSM — the campaign
scorers conjugate EACH of them to reach DSM; verified here by the
base-field phase alignment), so no conj is applied on either side.
The BEM first-order part is Richardson-extracted from the two
members: du_lin(2500) = (4 du(2500) - du(5000)) / 2.

usage: tfe_anchor.py [nproc] [lmax]      (defaults 32, 80)
"""
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "dsm_arbitration"))

import run_bem as rb                                     # noqa: E402
from pyastroseis import spectral as spx                  # noqa: E402
from pyastroseis import spectral_tfe as tfe              # noqa: E402
from pyastroseis.meshgen import (particle_sample_sphere,
                                 subdivide_spherical_mesh)  # noqa: E402

R_CMB = 3480.0e3
NMESH_CMB = 200
LAYERS = [dict(r_top=1221.5e3, rho=6000.0, vp=7000.0, vs=3500.0),
          dict(r_top=R_CMB, rho=5000.0, vp=5500.0),
          dict(r_top=6371.0e3, rho=3000.0, vp=6000.0, vs=3000.0)]
QMU = 50.0
Q_SIGN = -1.0
AMPS = (2500.0, 5000.0)
KMIN_BAND = 20                     # campaign scoring band k=20..138

_W = {}


def cmb_peak():
    """max |Ybar20| over the CMB mesh vertices, exactly reproducing
    the run_ensemble pert_mesh seeded construction path."""
    rng = np.random.default_rng(int(round(R_CMB / 1.0e3)))
    V, Tri, _, _ = particle_sample_sphere(N=NMESH_CMB, rng=rng)
    Tri, V = subdivide_spherical_mesh(Tri, V, 1)
    r_xy = np.linalg.norm(V[:, :2], axis=1)
    theta = np.arctan2(r_xy, V[:, 2])
    ct, st = np.cos(theta), np.sin(theta)
    P, _, _ = tfe._plm_grid(0, 2, ct, st)
    return float(np.max(np.abs(P[2])))


def station_dirs(stations):
    out = []
    for stn in stations:
        th = math.radians(90.0 - stn["lat"])
        ph = math.radians(stn["lon"])
        out.append([math.sin(th) * math.cos(ph),
                    math.sin(th) * math.sin(ph), math.cos(th)])
    return np.array(out)


def zrt_basis(src_dir, dirs):
    """(nst, 3, 3) rows = (Z=rhat, R=in-plane away from source,
    T=transverse) at each station."""
    out = np.empty((len(dirs), 3, 3))
    for i, d in enumerate(dirs):
        t = np.cross(src_dir, d)
        t /= np.linalg.norm(t)
        p = np.cross(t, d)
        out[i] = np.stack([d, p, t])
    return out


def _worker(iw):
    w = _W["w_arr"][iw]
    kw = dict(Q=QMU, q_sign=Q_SIGN, lmax=_W["lmax"])
    u = spx.spectral_spectra(LAYERS, _W["src"], _W["mts"],
                             np.array([w]), _W["dirs"], **kw)
    du = tfe.relief_spectra(LAYERS, _W["src"], _W["mts"],
                            np.array([w]), _W["dirs"],
                            relief=[(R_CMB, 2, 0, _W["h20"])], **kw)
    tot = u["psv"] + u["sh"]
    dtot = du["psv"] + du["sh"]
    return iw, tot[:, :, :, 0], dtot[:, :, :, 0]


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    man = json.load(open(os.path.join(SCRIPT_DIR,
                                      "ens_manifest.json")))
    df = 1.0 / man["tlen"]
    imax = man["imax"]
    omegai = man["omegai_1_per_s"]
    src_d = man["source"]
    ks = np.arange(1, imax + 1)
    w_arr = 2.0 * np.pi * ks * df + 1j * omegai

    if os.environ.get("ANCHOR_LOAD") == "1":
        zt0 = np.load(os.path.join(SCRIPT_DIR,
                                   "tfe_anchor_y20.npz"))
        peak, h20 = float(zt0["peak"]), float(zt0["h20"])
    else:
        peak = cmb_peak()
        h20 = AMPS[0] / peak        # metres of Ybar20 for member 1
    print("CMB vertex peak |Ybar20| = %.6f  ->  h20(2500 m) = "
          "%.1f m of Ybar20" % (peak, h20), flush=True)

    th = math.radians(90.0 - src_d["lat"])
    ph = math.radians(src_d["lon"])
    r0 = src_d["r0_km"] * 1.0e3
    src = np.array([r0 * math.sin(th) * math.cos(ph),
                    r0 * math.sin(th) * math.sin(ph),
                    r0 * math.cos(th)])
    zb = np.load(os.path.join(SCRIPT_DIR, "ens_base.npz"))
    src_names = [str(s) for s in zb["sources"]]
    scale = 1.0e20 / 100.0
    mts = [rb.sph_to_cart_mt(
        dict(mrr=(100.0, 0, 0, 0, 0, 0),
             mrt=(0, 100.0, 0, 0, 0, 0))[s],
        src_d["lat"], src_d["lon"]) * scale for s in src_names]
    dirs = station_dirs(man["stations"])

    _W.update(dict(w_arr=w_arr, src=src, mts=mts, dirs=dirs,
                   h20=h20, lmax=lmax))
    nsrc, nst = len(mts), len(dirs)
    npz_path = os.path.join(SCRIPT_DIR, "tfe_anchor_y20.npz")
    if os.environ.get("ANCHOR_LOAD") == "1":
        zt = np.load(npz_path)
        u_t, du_t = zt["u_tfe"], zt["du_tfe"]
        print("loaded cached TFE sweep (lmax %d)" % zt["lmax"],
              flush=True)
    else:
        u_t = np.zeros((nsrc, nst, 3, imax + 1), dtype=complex)
        du_t = np.zeros_like(u_t)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(nproc, len(ks))) as pool:
            for iw, uu, dd in pool.imap_unordered(_worker,
                                                  range(len(ks))):
                u_t[:, :, :, iw + 1] = uu
                du_t[:, :, :, iw + 1] = dd
        print("TFE sweep done (%d harmonics, lmax %d)"
              % (len(ks), lmax), flush=True)

    # BEM side (raw: same convention as raw spectral, see header)
    u_b = zb["u"]
    du_b = {}
    for amp in AMPS:
        zm = np.load(os.path.join(SCRIPT_DIR,
                                  "ens_y20_%d.npz" % int(amp)))
        du_b[amp] = zm["u"] - u_b
    du_lin = 0.5 * (4.0 * du_b[AMPS[0]] - du_b[AMPS[1]])

    # rotate everything to ZRT
    B = zrt_basis(src / np.linalg.norm(src), dirs)

    def rot(u):
        return np.einsum('sic,ascw->asiw', B, u)

    uz_t, duz_t = rot(u_t), rot(du_t)
    uz_b = rot(u_b)
    duz_b1, duz_lin = rot(du_b[AMPS[0]]), rot(du_lin)

    band = slice(KMIN_BAND, imax + 1)
    chan = "ZRT"
    g_rms = {a: np.sqrt(np.mean(np.abs(duz_lin[a, :, :, band])
                                ** 2)) for a in range(nsrc)}
    print("\n=== base-field sanity (TFE vs BEM base, band k>=%d)"
          % KMIN_BAND)
    for a, sname in enumerate(src_names):
        for c in range(3):
            ub, ut = uz_b[a, :, c, band], uz_t[a, :, c, band]
            msk = np.abs(ub) > 0.1 * np.sqrt(
                np.mean(np.abs(ub) ** 2))
            if not msk.any():
                continue
            dph = np.median(np.abs(np.angle(
                ut[msk] / ub[msk])))
            print("  %s %s base med |dphase| %.1f deg"
                  % (sname, chan[c], math.degrees(dph)))
            rel = np.median(np.abs(ut[msk] - ub[msk])
                            / np.abs(ub[msk]))
            rat = np.median(np.abs(ut[msk]) / np.abs(ub[msk]))
            print("  %s %s: med rel %.3f  med |ratio| %.3f"
                  % (sname, chan[c], rel, rat))

    print("\n=== ANCHOR: du/u, TFE vs BEM (Richardson linear part),"
          " band k>=%d" % KMIN_BAND)
    res = {}
    for a, sname in enumerate(src_names):
        for c in range(3):
            db = duz_lin[a, :, c, band]
            if np.sqrt(np.mean(np.abs(db) ** 2)) \
                    < 1.0e-2 * g_rms[a]:
                continue            # dead channel (e.g. mrr T)
            with np.errstate(divide="ignore", invalid="ignore"):
                qb = db / uz_b[a, :, c, band]
                qt = (duz_t[a, :, c, band]
                      / uz_t[a, :, c, band])
            msk = (np.abs(db) > 0.1 * np.sqrt(
                np.mean(np.abs(db) ** 2))) & np.isfinite(qt) \
                & np.isfinite(qb)
            if not msk.any():
                continue
            rel = np.median(np.abs(qt[msk] - qb[msk])
                            / np.abs(qb[msk]))
            rat = np.median(np.abs(qt[msk]) / np.abs(qb[msk]))
            dph = np.median(np.abs(np.angle(qt[msk] / qb[msk])))
            cc = np.abs(np.vdot(qt[msk], qb[msk])) / (
                np.linalg.norm(qt[msk]) * np.linalg.norm(qb[msk]))
            res[(sname, chan[c])] = (rel, rat, dph, cc)
            print("  %s %s: med rel %.3f  med |ratio| %.3f  med "
                  "|dphase| %.1f deg  corr %.4f (n=%d)"
                  % (sname, chan[c], rel, rat,
                     math.degrees(dph), cc, msk.sum()))
    g_t = np.linalg.norm(duz_t[:, :, :, band])
    g_b = np.linalg.norm(duz_lin[:, :, :, band])
    g_b1 = np.linalg.norm(duz_b1[:, :, :, band])
    print("global ||du||: TFE/BEM-lin = %.4f, TFE/BEM(2500) = %.4f"
          % (g_t / g_b, g_t / g_b1))
    lin = np.linalg.norm(du_b[AMPS[1]]) / np.linalg.norm(
        du_b[AMPS[0]])
    print("BEM member linearity ||du(5k)||/||du(2.5k)|| = %.4f"
          % lin)

    np.savez(os.path.join(SCRIPT_DIR, "tfe_anchor_y20.npz"),
             u_tfe=u_t, du_tfe=du_t, h20=h20, peak=peak, lmax=lmax,
             sources=np.array(src_names), df=df, omegai=omegai)

    # figure: |du| spectra overlays (plain labels)
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib unavailable here — run tfe_anchor_plot.py"
              " under the pytorch env for the figure")
        return
    import matplotlib.pyplot as plt
    show = [0, 3, 5, 9]
    rows = [(a, c) for a in range(nsrc) for c in (0, 2)
            if not (src_names[a] == "mrr" and c == 2)]
    fig, axes = plt.subplots(len(rows), len(show),
                             figsize=(16, 3.0 * len(rows)),
                             sharex=True)
    fhz = ks * df * 1.0e3
    for irow, (a, c) in enumerate(rows):
        for jcol, ist in enumerate(show):
            ax = axes[irow, jcol]
            ax.plot(fhz, np.abs(duz_lin[a, ist, c, 1:]), "k-",
                    lw=2.2, label="BEM")
            ax.plot(fhz, np.abs(duz_t[a, ist, c, 1:]), "r--",
                    lw=1.4, label="TFE")
            ax.set_title("%s %s  ST%02d (%.0f$^\\circ$)"
                         % (src_names[a], chan[c], ist,
                            man["stations"][ist]["dist_deg"]),
                         fontsize=13)
            if irow == 0 and jcol == 0:
                ax.legend(fontsize=12)
            if irow == len(rows) - 1:
                ax.set_xlabel("f [mHz]", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "tfe_anchor_y20.png"),
                dpi=130)
    print("wrote tfe_anchor_y20.npz / tfe_anchor_y20.png")


if __name__ == "__main__":
    main()
