# PyAstroSeis

Boundary-element (BEM) seismic wavefield solver for small planetary
bodies — asteroids, comets, moons — in the frequency domain. Python
port and modernization of AstroSeis
([Tian & Zheng, 2020](https://arxiv.org/abs/2004.07410)).

Features:

- **Homogeneous elastic body** with realistic shape/topography;
  single-force or moment-tensor sources
- **Solid body with a liquid core** (coupled elastic/acoustic BEM),
  including off-center ("shifted") cores and pressure sources in the
  fluid
- **Attenuation** via complex velocities; Qp defaults to the standard
  pure-shear-attenuation relation Qp = (3/4)(vp/vs)² Qs
- **Mesh generation**: quasi-uniform sphere sampling (Riesz-energy
  minimization), spherical quadrisection, spherical-harmonic
  topography (Phobos shape of Willner et al. 2014 included); direct
  loading of Wavefront `.obj` asteroid shape models
- **Fast**: canonical closed-form Green kernels, distance-adaptive
  quadrature, polar singular integration, and embarrassingly parallel
  frequency sweeps

## Install / run

```bash
pip install -e .            # numpy + scipy; matplotlib/pyyaml optional

python run_astroseis_py.py examples/config_Q570.yml --nprocs 8 --plot
python run_astroseis_py.py BEM_para_Q570 --nprocs 8          # legacy format
python run_astroseis_py.py examples/config_liquidcore.yml --nprocs 8
```

Outputs are MATLAB-compatible `.mat` files (spectra `uu` or
`up/u2/u1`) plus JSON metadata; `pyastroseis.plotting` reconstructs
time-domain record sections (Ricker source).

## Fidelity to the original MATLAB AstroSeis

Every routine was ported against MATLAB oracle dumps and full
reference runs (see `tests/` and `validation/`). Two switches control
exact reproduction of original results:

```bash
--qp-mode legacy --self-scheme grid --quad-mode full
```

reproduces original MATLAB outputs to ~1e-14 (the bundled
`out_Q*.mat` references; `compare_py_matlab.py` checks this). The
defaults are deliberately better than the original:

| switch | default | legacy (MATLAB-exact) |
|---|---|---|
| `--qp-mode` | `physical`: Qp = (3/4)(vp/vs)²Qs | 2.5 (kernels) / 2.25 (source) |
| `--self-scheme` | `polar`: singular quadrature, 3–4× more accurate, ~13× faster | 300×300 punch-out grid |
| `--quad-mode` | `adaptive`: distance-tiered rules (deg 10/5/2) | degree-10 everywhere |

## Tests

- `tests/test_kernel_equiv.py` — canonical kernels vs verbatim
  Mathematica transcriptions (no fixtures needed)
- `tests/test_oracle.py`, `tests/test_oracle2.py` — every ported
  routine vs MATLAB oracle dumps (fixtures `tests/oracle*.mat`
  included; regenerate with the `.m` scripts against the original
  AstroSeis code)
- `tests/test_self_convergence.py` — singular-quadrature convergence
  study (polar vs legacy grid vs converged reference)
- `tests/test_adaptive_quad.py` — adaptive-quadrature solution-error
  gate
- `tests/test_meshgen.py` — mesh-generation geometry checks
- `compare_py_matlab.py` — full-run comparison against the bundled
  MATLAB reference outputs (run the three `BEM_para_Q*` cases in
  legacy/grid/full mode first)

The liquid-core solver was additionally verified against the current
MATLAB code at machine precision (`validation/lc_arbitration.txt`;
the 2020-era reference output shipped with AstroSeis is accurate only
to ~2% and is not bundled).

## Performance

One 44-frequency case (1584 faces, 4752 unknowns) runs in ~25–40 s on
a single compute node (frequency-parallel workers); the liquid-core
case (10432 unknowns) in ~2–3 min. The original MATLAB code needed
~20 min per case on 32 cores.

## Citation

If you use PyAstroSeis, please cite:
Tian, Y. & Zheng, Y. (2020). AstroSeis — a boundary element method
code for seismic wavefields in asteroids with realistic shapes.
https://arxiv.org/abs/2004.07410

## License

GPL-3.0 (inherits from the original AstroSeis distribution and the
Xiao–Gimbutas quadrature tables).
