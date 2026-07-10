"""pyastroseis — Python port of AstroSeis (boundary element seismic
wavefield solver for small planetary bodies).

Standalone port of the full MATLAB feature set: homogeneous body with
single-force or moment-tensor sources, solid body with a liquid core,
and mesh generation (quasi-uniform sphere sampling + spherical-harmonic
topography). Validated against MATLAB oracles and full reference runs;
see tests/ and compare_*.py.

Qp convention: by default Qp = (3/4)(vp/vs)^2 * Qs (pure shear
attenuation, community standard). Pass qp_mode="legacy" to reproduce
the original MATLAB constants (2.5 / 2.25) exactly.

Original MATLAB code: Yuan Tian (2020), https://arxiv.org/abs/2004.07410
"""

from .mesh import Faces, load_faces_mat, faces_from_vertices
from .params import read_params, read_params_lc
from .solver import run_case, qp_factors
from .liquidcore import run_case_lc, load_layers_mat
from .domains import (Material, Interface, Region, MultiDomainModel,
                      homogeneous_model, liquid_core_model,
                      welded_two_layer_model, nested_shell_model)
from .layered import read_layered_config, build_layered_model

__all__ = [
    "Faces", "load_faces_mat", "faces_from_vertices",
    "read_params", "read_params_lc",
    "run_case", "run_case_lc", "qp_factors", "load_layers_mat",
    "Material", "Interface", "Region", "MultiDomainModel",
    "homogeneous_model", "liquid_core_model",
    "welded_two_layer_model", "nested_shell_model",
    "read_layered_config", "build_layered_model",
]
