"""Tests for DnD constructors and properties, ported from test_dnd_constructor.m."""

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.sqw import DnDBase, d0d, d1d, d2d, d3d, d4d, data_sqw_dnd, dnd, dnd_data, dnd_metadata

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


def _sample_axes_proj_data(ndim: int):
    proj = line_proj([1, 1, 0], [-1, 1, 0], offset=[0, 0, 0, 0], alatt=[3.0, 4.0, 5.0], angdeg=[90.0, 90.0, 90.0])
    bin_args = []
    dims = []
    for i in range(4):
        if i < ndim:
            nb = i + 3
            bin_args.append([-1.0, 2.0 / nb, 1.0])
            dims.append(nb + 1)
        else:
            bin_args.append([-0.5, 0.5])
    ax = line_axes(*bin_args)
    if ndim == 0:
        sz = (1, 1)
    elif ndim == 1:
        sz = (dims[0], 1)
    else:
        sz = tuple(dims)
    nelem = int(np.prod(sz))
    s = np.arange(1, nelem + 1, dtype=float).reshape(sz, order="F")
    e = (np.arange(1, nelem + 1, dtype=float) * 0.1).reshape(sz, order="F")
    npix = np.ones(sz, dtype=float)
    return ax, proj, s, e, npix


class TestDnDConstructor:
    @pytest.mark.parametrize("cls,ndim", [(d0d, 0), (d1d, 1), (d2d, 2), (d3d, 3), (d4d, 4)])
    def test_default_constructor(self, cls, ndim):
        obj = cls()
        assert isinstance(obj, cls)
        assert isinstance(obj, DnDBase)
        assert obj.dimensions == ndim
        assert obj.NUM_DIMS == ndim

    def test_dnd_factory_default(self):
        obj = DnDBase.dnd()
        assert isinstance(obj, d0d)
        assert obj.dimensions == 0
        obj2 = dnd()
        assert isinstance(obj2, d0d)

    @pytest.mark.parametrize("cls,ndim", [(d0d, 0), (d1d, 1), (d2d, 2), (d3d, 3), (d4d, 4)])
    def test_axes_proj_constructor(self, cls, ndim):
        ax, proj, s, e, npix = _sample_axes_proj_data(ndim)
        obj = cls(ax, proj)
        assert isinstance(obj, cls)
        assert obj.dimensions == ndim
        assert obj.s.shape == s.shape
        assert np.all(obj.s == 0)

        obj_full = cls(ax, proj, s, e, npix)
        assert isinstance(obj_full, cls)
        np.testing.assert_array_almost_equal(obj_full.s, s)
        np.testing.assert_array_almost_equal(obj_full.e, e)
        np.testing.assert_array_almost_equal(obj_full.npix, npix)

        obj_fac = DnDBase.dnd(ax, proj, s, e, npix)
        assert isinstance(obj_fac, cls)
        assert obj_full.equal_to_tol(obj_fac)

    @pytest.mark.parametrize("cls,ndim", [(d0d, 0), (d1d, 1), (d2d, 2), (d3d, 3), (d4d, 4)])
    def test_wrong_dimension_axes_raises_error(self, cls, ndim):
        other_dim = (ndim + 1) % 5
        ax, proj, _, _, _ = _sample_axes_proj_data(other_dim)
        with pytest.raises(HoraceError):
            cls(ax, proj)

    def test_filename_constructor(self):
        f1d = os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")
        w1 = d1d(f1d)
        assert isinstance(w1, d1d)
        assert w1.dimensions == 1

        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        w2 = d2d(f2d)
        assert isinstance(w2, d2d)
        assert w2.dimensions == 2

        with pytest.raises(HoraceError):
            d1d(f2d)

    def test_spinw_constructor_1d_and_2d(self):
        w1 = d1d(
            [3.0, 4.0, 5.0, 90.0, 90.0, 90.0],
            [1, 0, 0, 0],
            [0, 0.1, 1],
            [0, 1, 0, 0],
            [-0.1, 0.1],
            [0, 0, 1, 0],
            [-0.1, 0.1],
            [0, 0, 0, 1],
            [-1, 1],
        )
        assert isinstance(w1, d1d)
        assert w1.dimensions == 1
        np.testing.assert_array_almost_equal(w1.alatt, [3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(w1.angdeg, [90.0, 90.0, 90.0])

        w2 = d2d(
            [3.0, 4.0, 5.0, 90.0, 90.0, 90.0],
            [1, 0, 0, 0],
            [0, 0.1, 1],
            [0, 0, 0, 1],
            [0, 1, 10],
            [0, 1, 0, 0],
            [-0.1, 0.1],
            [0, 0, 1, 0],
            [-0.1, 0.1],
        )
        assert isinstance(w2, d2d)
        assert w2.dimensions == 2

    def test_modifiable_and_readonly_properties(self):
        ax, proj, s, e, npix = _sample_axes_proj_data(2)
        w2 = d2d(ax, proj, s, e, npix)

        w2.title = "New Title"
        assert w2.title == "New Title"

        w2.filename = "test.sqw"
        assert w2.filename == "test.sqw"

        w2.alatt = [4.0, 4.0, 4.0]
        np.testing.assert_array_almost_equal(w2.alatt, [4.0, 4.0, 4.0])

        w2.angdeg = [90.0, 90.0, 120.0]
        np.testing.assert_array_almost_equal(w2.angdeg, [90.0, 90.0, 120.0])

        w2.dax = [2, 1]
        np.testing.assert_array_equal(w2.dax, [2, 1])

        # Read-only properties
        for prop in ("dimensions", "iax", "iint", "pax", "p", "img_range", "nbins", "offset", "is_filebacked"):
            with pytest.raises((AttributeError, HoraceError)):
                setattr(w2, prop, 1)

    def test_d0d_transforms_into_higher_dim_on_check_combo_arg(self):
        w0 = d0d()
        ax, proj, s, e, npix = _sample_axes_proj_data(2)
        w0.do_check_combo_arg = False
        w0.axes = ax
        w0.proj = proj
        w0.s = s
        w0.e = e
        w0.npix = npix
        w0.do_check_combo_arg = True
        w_out = w0.check_combo_arg()
        assert isinstance(w_out, d2d)
        assert w_out.dimensions == 2

