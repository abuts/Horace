"""Tests for sigvar_set on DnD objects, ported from test_sigvar_set.m."""

import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw import d0d, d1d, d2d, d3d, d4d


class TestSigvarSet:
    def test_sigvar_set_raises_error_if_s_not_same_size_as_dnd_s(self):
        dnd_2d = d2d()
        dnd_2d.npix = np.ones((2, 3))
        sigvar_obj = sigvar(np.array([1.0, 2.0, 3.0]), np.ones((2, 3)))
        with pytest.raises(HoraceError) as excinfo:
            dnd_2d.sigvar_set(sigvar_obj)
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "Size of signal array: [1, 3] must match size of npix array: [2, 3]" in str(excinfo.value)

    def test_sigvar_set_raises_error_if_e_not_same_size_as_dnd_e(self):
        dnd_2d = d2d()
        dnd_2d.npix = np.ones((2, 3))
        sigvar_obj = sigvar(np.ones((2, 3)), np.array([1.0, 2.0, 3.0]))
        with pytest.raises(HoraceError) as excinfo:
            dnd_2d.sigvar_set(sigvar_obj)
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "Size of variance array: [1, 3] must match size of npix array: [2, 3]" in str(excinfo.value)

    def test_sigvar_set_updates_s_and_e_values_and_zeros_empty_npix(self):
        dnd_2d = d2d()
        dnd_2d.npix = np.array([[1, 0, 2], [0, 3, 0], [4, 5, 0]], dtype=float)
        s = np.arange(1, 10, dtype=float).reshape((3, 3))
        e = np.arange(10, 19, dtype=float).reshape((3, 3))
        sigvar_obj = sigvar(s, e)

        dnd_2d = dnd_2d.sigvar_set(sigvar_obj)

        expected_s = s.copy()
        expected_e = e.copy()
        mask = dnd_2d.npix == 0
        expected_s[mask] = 0.0
        expected_e[mask] = 0.0

        np.testing.assert_array_equal(dnd_2d.s, expected_s)
        np.testing.assert_array_equal(dnd_2d.e, expected_e)

