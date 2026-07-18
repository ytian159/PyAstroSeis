"""Gates for the PYASTROSEIS_FL_LARGEX large-|z| rescue of the
scaled-series radial basis (spheroidal_ref._fl_scaled).

The small-z series (valid z^2/2 < ~l) catastrophically cancels in
the propagating regime (error ~ eps * e^|z|), reached above ~5 mHz
on Earth-size stacks. The rescue reroutes to the stable recurrence
basis times the exact log-space column-scale constant. Contract:
  * flag off  -> byte-identical old code path (strict no-op);
  * flag on   -> no trigger in the series-healthy regime (bitwise
    there too), rescue only where the series has lost > 8 digits;
  * both branches agree to roundoff in the overlap regime;
  * the rescued (j, y) pair satisfies the exact Wronskian
    fj*fpy - fy*fpj = -(2l+1)*zref/z^2.
"""
import importlib
import os

import numpy as np

import pyastroseis.spheroidal_ref as sr


def _with_flag(on):
    os.environ["PYASTROSEIS_FL_LARGEX"] = "1" if on else ""
    importlib.reload(sr)
    return sr


def teardown_module():
    _with_flag(False)


def test_no_trigger_bitwise():
    z = np.array([8.0 + 0.5j])
    a0 = _with_flag(False)._fl_scaled("j", 40, z, 12.0 + 0.1j)
    a1 = _with_flag(True)._fl_scaled("j", 40, z, 12.0 + 0.1j)
    assert a0[0][0] == a1[0][0] and a0[1][0] == a1[1][0]


def test_overlap_agreement():
    m = _with_flag(True)
    for l, zz in ((40, 15.0 + 0.8j), (60, 18.0 + 0.3j),
                  (30, 12.0 + 1.0j)):
        zr = zz * 1.15
        for kind in ("j", "y"):
            fs = m._fl_scaled(kind, l, np.array([zz]), zr)
            fr = m._fl_scaled_recur(kind, l, np.array([zz]), zr)
            assert abs(fs[0][0] / fr[0][0] - 1) < 1e-12
            assert abs(fs[1][0] / fr[1][0] - 1) < 1e-12


def test_wronskian_rescue_regime():
    m = _with_flag(True)
    for l, zz in ((60, 200.0 + 1.0j), (150, 300.0 + 0.7j),
                  (25, 55.0 + 1.0j)):
        zr = zz * 1.1
        fj, fpj = m._fl_scaled("j", l, np.array([zz]), zr)
        fy, fpy = m._fl_scaled("y", l, np.array([zz]), zr)
        W = fj[0] * fpy[0] - fy[0] * fpj[0]
        Wex = -(2.0 * l + 1.0) * zr / zz ** 2
        assert abs(W / Wex - 1) < 1e-11


def test_rescue_fires_and_is_finite():
    z = np.array([200.0 + 1.0j])
    b0 = _with_flag(False)._fl_scaled("j", 60, z, 220.0 + 1.1j)
    b1 = _with_flag(True)._fl_scaled("j", 60, z, 220.0 + 1.1j)
    assert abs(b0[0][0] / b1[0][0] - 1) > 1.0     # series was junk
    assert np.isfinite(b1[0][0]) and np.isfinite(b1[1][0])
