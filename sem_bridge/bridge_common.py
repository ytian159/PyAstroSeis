"""Shared constants + model/station machinery for the ak135 hdur80
SEM bridge (spectral leg vs SEM-Ref NEX320 / SEM-1 NEX128 / DSM).

Model source: the DSM tipsv input that generated the benchmark DSM
leg (validated 0.9962 vs SEM elastic-iso) — 127 zones with linear
polynomials in x = r/6371 km, fluid outer core via vs = 0, elastic
markers. All three reference legs are ELASTIC (SEM ATTENUATION =
.false., DSM zones carry the -1 -1 elastic marker), so the spectral
leg runs with Q = None.

Synthesis conventions mirror synthesize_dsm_hdur80.py exactly
(TLEN 8192, NFFT 131072, omegai = ln(100)/TLEN from the spc header,
erf quasi-Heaviside hdur 80 shifted 1.5*hdur, damped-basis baseline
fit over t <= 48 s).
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INF = ("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
       "dsm_reference/ak135/tipsv_mrr.inf")
SEMREF_DIR = ("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
              "f0125_resolved_ppw/target_nex75_sem320_4800s/"
              "sem_case/OUTPUT_FILES")
SEM1_DIR = ("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
            "f0125_perf_compare/sem_ak135_mrr_hdur80/OUTPUT_FILES")
DSM_DIR = ("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
           "f0125_perf_compare/dsm_hdur80/ak135/synthetics/"
           "heaviside_hdur80/mrr")

TLEN = 8192.0
IMAX = 256
NFFT = 131072
T_OUT = 5040.0
OMEGAI = 5.6215456371924938e-4          # ln(100)/TLEN, spc header
HDUR = 80.0
SHIFT = 1.5 * HDUR
SOURCE_DECAY_MIMIC_TRIANGLE = 1.628

SOURCE_LAT, SOURCE_LON = 6.0, 12.0
R0 = 6321.0e3                            # 50 km depth
A_EARTH = 6371.0e3

STATIONS = ["AZ%02d" % i for i in range(1, 19)]
LATS = [16, 26, 36, 46, 56, 66, 76, 86, 84, 74, 64, 54, 44, 34,
        24, 14, 4, -1]
LONS = [12] * 8 + [-168] * 10
DISTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,
         140, 150, 160, 170, 175]


def parse_zones():
    lines = open(INF).read().splitlines()
    nzone = int(lines[7].split()[0])
    toks = " ".join(lines[8:]).split()
    zones = []
    p = 0
    for _ in range(nzone):
        z = [float(t) for t in toks[p:p + 28]]
        p += 28
        rmin, rmax = z[0], z[1]
        rho = z[2:4]
        vpv, vph = z[6:8], z[10:12]
        vsv, vsh = z[14:16], z[18:20]
        assert vpv == vph and vsv == vsh, "anisotropic zone"
        zones.append((rmin, rmax, rho, vpv, vsv))
    assert abs(float(toks[p]) - 6321.0) < 1e-6, "zone parse drift"
    return zones


def build_layers(h_cap_km):
    """Staircase LAYERS (SI, centre outward): every zone gets
    ceil(thickness/h_cap) equal sublayers, materials at sublayer
    mid-radius. Zone boundaries are always honored exactly."""
    layers = []
    for rmin, rmax, rho, vp, vs in parse_zones():
        nsub = max(1, int(math.ceil((rmax - rmin) / h_cap_km)))
        for j in range(nsub):
            ra = rmin + (rmax - rmin) * j / nsub
            rb = rmin + (rmax - rmin) * (j + 1) / nsub
            x = 0.5 * (ra + rb) / 6371.0
            layers.append(dict(
                r_top=rb * 1e3,
                rho=(rho[0] + rho[1] * x) * 1e3,
                vp=(vp[0] + vp[1] * x) * 1e3,
                vs=(vs[0] + vs[1] * x) * 1e3))
    for e in layers:
        assert abs(e["r_top"] - R0) > 0.5, "layer edge on source"
    return layers


def sph_to_cart_mt(mt6, lat_deg, lon_deg):
    """DSM (Mrr, Mrt, Mrp, Mtt, Mtp, Mpp) -> cartesian 3x3
    (mirrors run_bem.sph_to_cart_mt)."""
    th = math.radians(90.0 - lat_deg)
    ph = math.radians(lon_deg)
    rh = np.array([math.sin(th) * math.cos(ph),
                   math.sin(th) * math.sin(ph), math.cos(th)])
    th_h = np.array([math.cos(th) * math.cos(ph),
                     math.cos(th) * math.sin(ph), -math.sin(th)])
    ph_h = np.array([-math.sin(ph), math.cos(ph), 0.0])
    mrr, mrt, mrp, mtt, mtp, mpp = mt6
    M_sph = np.array([[mrr, mrt, mrp],
                      [mrt, mtt, mtp],
                      [mrp, mtp, mpp]])
    P = np.column_stack([rh, th_h, ph_h])
    return P @ M_sph @ P.T


def unit_vec(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(lat) * math.cos(lon),
                     math.cos(lat) * math.sin(lon),
                     math.sin(lat)])


def src_xyz():
    return R0 * unit_vec(SOURCE_LAT, SOURCE_LON)


def station_dirs():
    return np.array([unit_vec(la, lo)
                     for la, lo in zip(LATS, LONS)])


def mt_cart():
    return sph_to_cart_mt([1.0e20, 0, 0, 0, 0, 0],
                          SOURCE_LAT, SOURCE_LON)


def get_poles(lmax):
    """spectral_poles(lmax) with a disk cache (the lmax-1400
    precompute costs ~13 min single-core; identical across rungs)."""
    import numpy as _np
    from pyastroseis.spectral import spectral_poles
    path = os.path.join(HERE, "out_ak135",
                        "poles_%d.npz" % lmax)
    if os.path.exists(path):
        z = _np.load(path)
        DY = z["DY"]
        DG = {-1: z["DG_m"], 1: z["DG_p"]}
        poleT = {-1: z["T_m"], 1: z["T_p"]}
        return DY, DG, poleT
    DY, DG, poleT = spectral_poles(lmax)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _np.savez(path, DY=DY, DG_m=DG[-1], DG_p=DG[1],
              T_m=poleT[-1], T_p=poleT[1])
    return DY, DG, poleT


def heaviside_stf(t):
    hg = HDUR / SOURCE_DECAY_MIMIC_TRIANGLE
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec((t - SHIFT) / hg))


def synth_time(u_spec, conj):
    """(3, IMAX+1) spectrum -> (3, NFFT) displacement [m] on the DSM
    time grid, DSM-identical synthesis incl. damped-basis baseline.
    conj=True maps our conj-of-DSM npz convention into DSM space."""
    dt = TLEN / NFFT
    t = np.arange(NFFT) * dt
    stf_spec = np.fft.fft(heaviside_stf(t) * np.exp(-OMEGAI * t)) * dt
    if conj:
        u_spec = np.conj(u_spec)
    out = np.zeros((3, NFFT))
    basis = np.exp(OMEGAI * t)
    mask = t <= 48.0
    for c in range(3):
        full = np.zeros(NFFT, dtype=complex)
        full[:IMAX + 1] = u_spec[c, :IMAX + 1] * stf_spec[:IMAX + 1]
        full[NFFT - IMAX:] = np.conj(full[1:IMAX + 1])[::-1]
        wav = np.real(np.fft.ifft(full)) * NFFT / TLEN
        wav *= basis
        coef = (np.dot(wav[mask], basis[mask])
                / np.dot(basis[mask], basis[mask]))
        out[c] = wav - coef * basis
    return t, out
