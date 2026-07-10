#!/usr/bin/env python3
"""CLI for the pyastroseis port.

Usage:
    python run_astroseis_py.py BEM_para_Q570 [--nprocs 45]
    python run_astroseis_py.py examples/config_Q570.yml --nprocs 45
    python run_astroseis_py.py inp_sft_lc --liquidcore --nprocs 44

Parameter files may be legacy fixed-line AstroSeis format or YAML
(.yml/.yaml; liquid-core detected from the `core` block).

Defaults: --qp-mode physical (Qp = (3/4)(vp/vs)^2 Qs, community
standard) and --self-scheme polar (accurate fast singular quadrature).
Use `--qp-mode legacy --self-scheme grid` to reproduce original MATLAB
outputs exactly.

Output goes to pyout/<name from the parameter file> unless --out is
given; --plot also writes a record-section PNG next to the output.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyastroseis import run_case, run_case_lc  # noqa: E402
from pyastroseis.params import ParamsLC, resolve_path  # noqa: E402
from pyastroseis.solver import load_params_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("param_file")
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--liquidcore", action="store_true",
                    help="treat a legacy param file as liquid-core "
                         "(YAML configs are auto-detected)")
    ap.add_argument("--qp-mode", choices=["physical", "legacy"],
                    default="physical")
    ap.add_argument("--self-scheme", choices=["polar", "grid"],
                    default="polar")
    ap.add_argument("--quad-mode", choices=["adaptive", "full"],
                    default="adaptive")
    ap.add_argument("--plot", action="store_true",
                    help="write a record-section PNG next to the output")
    args = ap.parse_args()

    p = load_params_any(args.param_file, lc=args.liquidcore)
    is_lc = isinstance(p, ParamsLC)
    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.param_file)),
                           "pyout", os.path.basename(p.output_file))

    if is_lc:
        # pass the original path so mesh paths resolve relative to it
        p.mesh_file = resolve_path(p.mesh_file, args.param_file)
        up, u2, u1, meta = run_case_lc(p, nprocs=args.nprocs, out=out,
                                       qp_mode=args.qp_mode,
                                       self_scheme=args.self_scheme,
                                       quad_mode=args.quad_mode)
        uu_plot = u1
    else:
        p.mesh_file = resolve_path(p.mesh_file, args.param_file)
        uu_plot, meta = run_case(p, nprocs=args.nprocs, out=out,
                                 qp_mode=args.qp_mode,
                                 self_scheme=args.self_scheme,
                                 quad_mode=args.quad_mode)

    if args.plot:
        from pyastroseis.mesh import load_faces_mat
        from pyastroseis.plotting import plot_record_section
        if is_lc:
            from pyastroseis.liquidcore import load_layers_mat
            faces = load_layers_mat(p.mesh_file)[1]
        else:
            faces = load_faces_mat(p.mesh_file)
        png = plot_record_section(uu_plot, faces, p.nt, p.dt, p.f0,
                                  out + ".record_section.png",
                                  title="radial displacement")
        print(f"plot: {png}")


if __name__ == "__main__":
    main()
