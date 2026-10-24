"""Tests for unary operations on DnD objects, ported from test_unary_ops.m."""

import numpy as np
import pytest

from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw import d0d, d1d, d2d, d3d, d4d


def _make_dnd_for_unary(cls, val_min: float, val_max: float):
    ndim = cls.NUM_DIMS
    proj = line_proj([1, 0, 0], [0, 1, 0])
    bin_args = []
    dims = []
    for i in range(4):
        if i < ndim:
            bin_args.append([0.0, 0.5, 1.0])
            dims.append(3)
        else:
            bin_args.append([-1.0, 1.0])
    ax = line_axes(*bin_args)
    if ndim == 0:
        sz = (1, 1)
    elif ndim == 1:
        sz = (dims[0], 1)
    else:
        sz = tuple(dims)
    nelem = int(np.prod(sz))
    s = np.linspace(val_min, val_max, nelem).reshape(sz, order="F")
    e = np.linspace(0.01, 0.05, nelem).reshape(sz, order="F")
    npix = np.ones(sz, dtype=float)
    if nelem > 1:
        npix.flat[0] = 0.0
    return cls(ax, proj, s, e, npix)


UNARY_OPS_RANGES = [
    ("acos", 0.1, 0.9),
    ("acosh", 1.5, 3.0),
    ("acot", 0.5, 2.0),
    ("acoth", 1.5, 3.0),
    ("acsc", 1.5, 3.0),
    ("acsch", 0.5, 2.0),
    ("asec", 1.5, 3.0),
    ("asech", 0.2, 0.8),
    ("asin", 0.1, 0.9),
    ("asinh", 0.5, 2.0),
    ("atan", 0.1, 0.9),
    ("atanh", 0.1, 0.8),
    ("cos", 0.1, 0.9),
    ("cosh", 0.1, 1.5),
    ("cot", 0.2, 1.2),
    ("coth", 0.5, 2.0),
    ("csc", 0.2, 1.2),
    ("csch", 0.5, 2.0),
    ("exp", 0.1, 1.5),
    ("log", 0.5, 2.5),
    ("log10", 0.5, 2.5),
    ("sec", 0.1, 1.0),
    ("sech", 0.1, 1.5),
    ("sin", 0.1, 0.9),
    ("sinh", 0.1, 1.5),
    ("sqrt", 0.5, 3.0),
    ("tan", 0.1, 0.9),
    ("tanh", 0.1, 1.5),
    ("uminus", 0.5, 2.5),
    ("uplus", 0.5, 2.5),
]


class TestUnaryOpsDnD:
    @pytest.mark.parametrize("cls", [d0d, d1d, d2d, d3d, d4d])
    @pytest.mark.parametrize("op_name,vmin,vmax", UNARY_OPS_RANGES)
    def test_unary_op_matches_sigvar(self, cls, op_name, vmin, vmax):
        dnd_obj = _make_dnd_for_unary(cls, vmin, vmax)
        sv_in = sigvar(dnd_obj.s, dnd_obj.e)

        res_dnd = getattr(dnd_obj, op_name)()
        res_sv = getattr(sv_in, op_name)()

        expected_s = res_sv.s.copy()
        expected_e = res_sv.e.copy()
        mask = dnd_obj.npix == 0
        expected_s[mask] = 0.0
        expected_e[mask] = 0.0

        np.testing.assert_allclose(res_dnd.s, expected_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_dnd.e, expected_e, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(res_dnd.npix, dnd_obj.npix)

