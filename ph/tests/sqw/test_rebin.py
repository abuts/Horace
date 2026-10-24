"""Tests for rebin on DnD objects, ported from test_rebin.m."""

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import d0d, d1d, d2d, d3d, read_dnd

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


class TestRebinDnD:
    @pytest.fixture(scope="class")
    def d1d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw"))

    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw"))

    @pytest.fixture(scope="class")
    def d3d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w3d_d3d.sqw"))

    def test_rebin_0d_from_d1d(self, d1d_obj):
        w0 = d0d()
        w0_rebin = w0.rebin(d1d_obj)
        assert isinstance(w0_rebin, d0d)
        expected_npix = float(np.sum(d1d_obj.npix))
        expected_s = float(np.sum(d1d_obj.s * d1d_obj.npix)) / expected_npix
        expected_e = float(np.sum(d1d_obj.e * (d1d_obj.npix**2))) / (expected_npix**2)
        np.testing.assert_allclose(w0_rebin.npix, [[expected_npix]])
        np.testing.assert_allclose(w0_rebin.s, [[expected_s]])
        np.testing.assert_allclose(w0_rebin.e, [[expected_e]])

        # With -keep_contents
        w0_keep = w0_rebin.rebin(d1d_obj, "-keep_contents")
        np.testing.assert_allclose(w0_keep.npix, [[2.0 * expected_npix]])
        np.testing.assert_allclose(w0_keep.s, [[expected_s]])

    def test_rebin_d1d_with_step(self, d1d_obj):
        p0 = d1d_obj.p[0]
        step = float(p0[1] - p0[0]) * 2.0
        w_reb = d1d_obj.rebin([float(p0[0]), step, float(p0[-1])])
        assert isinstance(w_reb, d1d)
        assert w_reb.npix.size < d1d_obj.npix.size

    def test_rebin_d2d_with_template(self, d2d_obj):
        w_reb = d2d_obj.rebin([-0.5, 0.1, 0.5], [])
        assert isinstance(w_reb, d2d)
        w_reb2 = d2d_obj.rebin(w_reb)
        assert w_reb.equal_to_tol(w_reb2)

    def test_rebin_d2d_to_d1d(self, d2d_obj):
        w_reb = d2d_obj.rebin([-0.5, 0.5], [])
        assert isinstance(w_reb, d1d)

    def test_rebin_d3d(self, d3d_obj):
        w_reb = d3d_obj.rebin([-0.8, 0.2, 0.8], [-0.8, 0.2, 0.8], [])
        assert isinstance(w_reb, d3d)

