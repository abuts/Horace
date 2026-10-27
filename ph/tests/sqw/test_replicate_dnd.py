"""Tests for replicate on DnD objects, ported from test_replicate_dnd.m."""

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import d0d, d1d, d2d, d3d, read_dnd, sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


class TestReplicateDnD:
    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w2d_qq_d2d.sqw"))

    @pytest.fixture(scope="class")
    def sqw_2d_obj(self):
        return sqw(os.path.join(COMMON_DATA_DIR, "w2d_qq_sqw.sqw"))

    @pytest.fixture(scope="class")
    def d3d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w3d_d3d.sqw"))

    def test_replicate_d0d_to_d2d(self, d2d_obj):
        w0 = d0d()
        w0.s = np.array([[5.0]])
        w0.e = np.array([[0.25]])
        w0.npix = np.array([[10.0]])

        w_rep = w0.replicate(d2d_obj)
        assert isinstance(w_rep, d2d)
        mask = d2d_obj.npix > 0
        np.testing.assert_allclose(w_rep.s[mask], 5.0)
        np.testing.assert_allclose(w_rep.e[mask], 0.25)
        np.testing.assert_allclose(w_rep.s[~mask], 0.0)

    def test_replicate_d1d_to_d2d_and_sqw(self, d2d_obj, sqw_2d_obj):
        # Integrate along 2nd plot axis of d2d_obj to get a compatible d1d
        p2 = d2d_obj.p[1]
        w1 = d2d_obj.cut([], [float(p2[0]), float(p2[-1])])
        assert isinstance(w1, d1d)

        w_rep = w1.replicate(d2d_obj)
        assert isinstance(w_rep, d2d)
        assert w_rep.s.shape == d2d_obj.s.shape

        w_rep_sqw = w1.replicate(sqw_2d_obj)
        assert isinstance(w_rep_sqw, d2d)
        np.testing.assert_allclose(w_rep.s, w_rep_sqw.s)
        np.testing.assert_allclose(w_rep.e, w_rep_sqw.e)
        np.testing.assert_allclose(w_rep.npix, w_rep_sqw.npix)
        assert w1.replicate(d2d(sqw_2d_obj)).equal_to_tol(w_rep_sqw, ignore_str=True)

    def test_replicate_higher_to_lower_dim_raises_error(self, d2d_obj, d3d_obj):
        with pytest.raises(HoraceError):
            d3d_obj.replicate(d2d_obj)

