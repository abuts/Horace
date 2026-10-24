"""Tests for sigvar_set on DnD objects, ported from test_sigvar_set.m."""

import numpy as np
import pytest

from pyhorace.coord_transf import line_axes, line_proj
from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw import d1d, d2d


class TestSigvarSet:
    def test_sigvar_set_raises_error_if_e_not_same_size_as_dnd_object(self):
        ab = line_axes(
            [0, 0.5, 1],
            [0, 1],
            [0, 0.5, 1],
            [0, 1],
            single_bin_defines_iax=[False, True, True, True],
        )
        pr = line_proj(alatt=3, angdeg=90)
        d2d_obj = d2d(ab, pr)
        d2d_obj.npix = np.ones(d2d_obj.axes.dims_as_ssize)

        sigvar_obj = sigvar(np.array([1.0, 2.0, 3.0]), np.array([44.0, 55.0, 66.0]))
        with pytest.raises(HoraceError) as excinfo:
            d2d_obj.sigvar_set(sigvar_obj)
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "size of npix array" in str(excinfo.value)

    def test_sigvar_set_s_and_e_nopix_gives_zero(self):
        ab = line_axes(
            [0, 0.5, 1],
            [0, 1],
            [0, 0.5, 1],
            [0, 1],
            single_bin_defines_iax=[False, True, True, True],
        )
        pr = line_proj(alatt=3, angdeg=90)
        d2d_obj = d2d(ab, pr)
        d2d_obj.npix = np.zeros((3, 3))

        sigvar_obj = sigvar(
            np.arange(1, 10, dtype=float).reshape((3, 3)),
            np.arange(10, 19, dtype=float).reshape((3, 3)),
        )
        result = d2d_obj.sigvar_set(sigvar_obj)
        np.testing.assert_allclose(result.s, np.zeros((3, 3)))
        np.testing.assert_allclose(result.e, np.zeros((3, 3)))

    def test_sigvar_set_updates_s_and_e_values(self):
        ab = line_axes(
            [0, 0.5, 1],
            [0, 1],
            [0, 0.5, 1],
            [0, 1],
            single_bin_defines_iax=[False, True, True, True],
        )
        pr = line_proj(alatt=3, angdeg=90)
        d2d_obj = d2d(ab, pr)
        d2d_obj.npix = np.ones(d2d_obj.axes.dims_as_ssize)

        s = np.arange(1, 10, dtype=float).reshape((3, 3))
        e = np.arange(10, 19, dtype=float).reshape((3, 3))
        sigvar_obj = sigvar(s, e)

        result = d2d_obj.sigvar_set(sigvar_obj)
        np.testing.assert_allclose(result.s, s)
        np.testing.assert_allclose(result.e, e)

    def test_sigvar_set_zero_s_and_e_where_npix_zero(self):
        ab = line_axes([0, 0.5, 1], [0, 1], [0, 1], [0, 1])
        pr = line_proj(alatt=3, angdeg=90)
        d1d_obj = d1d(ab, pr)
        d1d_obj.s = np.ones((3, 1))
        d1d_obj.e = np.ones((3, 1))
        d1d_obj.npix = np.array([3.0, 0.0, 1.0])

        sigvar_obj = sigvar(np.array([1.0, 2.0, 3.0]), np.array([44.0, 55.0, 66.0]))
        expected_s = np.array([[1.0], [0.0], [3.0]])
        expected_e = np.array([[44.0], [0.0], [66.0]])

        result = d1d_obj.sigvar_set(sigvar_obj)
        np.testing.assert_allclose(result.s, expected_s)
        np.testing.assert_allclose(result.e, expected_e)


