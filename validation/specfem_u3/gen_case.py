#!/usr/bin/env python3
"""Generate a SPECFEM3D_GLOBE case dir (Par_file/CMTSOLUTION/STATIONS)
for the uniform3_q50 ULP benchmark from the pyastroseis campaign
manifest (stations + source; geocentric lat/lon passed through —
SPECFEM runs purely spherical with ELLIPTICITY off).

usage: gen_case.py <template_DATA> <case_dir> <manifest.json> <source>
         [NEX] [NPROC] [record_min] [hdur_s]
"""

import json
import os
import shutil
import sys

tmpl, case, man_path, source = sys.argv[1:5]
nex = int(sys.argv[5]) if len(sys.argv) > 5 else 64
nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 2
rec_min = float(sys.argv[7]) if len(sys.argv) > 7 else 10.0
hdur = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0

man = json.load(open(man_path))
src = man["source"]

data = os.path.join(case, "DATA")
os.makedirs(case, exist_ok=True)
if os.path.exists(data):
    shutil.rmtree(data)
shutil.copytree(tmpl, data, symlinks=False,
                ignore=shutil.ignore_patterns("semucb_grid"))

# Par_file edits
pf = os.path.join(data, "Par_file")
lines = open(pf).read().split("\n")


def set_par(key, val):
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key) and "=" in ln:
            k = ln.split("=")[0]
            lines[i] = "%s= %s" % (k, val)
            return
    raise KeyError(key)


set_par("MODEL", "uniform3_q50")
set_par("NEX_XI", nex)
set_par("NEX_ETA", nex)
set_par("NPROC_XI", nproc)
set_par("NPROC_ETA", nproc)
set_par("RECORD_LENGTH_IN_MINUTES", "%gd0" % rec_min)
set_par("ATTENUATION", ".true.")
open(pf, "w").write("\n".join(lines))

# CMTSOLUTION: moment tensor in dyn*cm; manifest entries are in units
# of 1e25 dyn*cm (DSM convention, mrr/mrt = 100 -> 1e27 dyn*cm)
mt = man["moment_tensors_1e25dyncm"][source]
mrr, mrt, mrp, mtt, mtp, mpp = [v * 1e25 for v in mt]
cmt = """ PDE 2000  1  1  0  0  0.0  %(lat).4f %(lon).4f %(dep).1f 5.0 5.0 uniform3 benchmark
event name:     uniform3_%(src)s
time shift:     0.0000
half duration:  %(hdur).4f
latitude:       %(lat).4f
longitude:      %(lon).4f
depth:          %(dep).4f
Mrr:            %(mrr).6e
Mtt:            %(mtt).6e
Mpp:            %(mpp).6e
Mrt:            %(mrt).6e
Mrp:            %(mrp).6e
Mtp:            %(mtp).6e
""" % dict(lat=src["lat"], lon=src["lon"], dep=src["depth_km"],
           hdur=hdur, src=source,
           mrr=mrr, mtt=mtt, mpp=mpp, mrt=mrt, mrp=mrp, mtp=mtp)
open(os.path.join(data, "CMTSOLUTION"), "w").write(cmt)

# STATIONS: manifest stations, network DF
with open(os.path.join(data, "STATIONS"), "w") as f:
    for st in man["stations"]:
        f.write("%-6s DF %10.4f %10.4f 0.0 0.0\n"
                % (st["name"], st["lat"], st["lon"]))

print("case %s: MODEL=uniform3_q50 NEX=%d NPROC=%d rec=%g min "
      "hdur=%g src=%s (%d stations)"
      % (case, nex, nproc, rec_min, hdur, source, len(man["stations"])))
