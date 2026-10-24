"""Tests for func_eval, sqw_eval, dispersion, and disp2sqw_eval on DnD objects, ported from test_dnd_eval.m."""

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import d1d, d2d, d3d, read_dnd

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


def _linear_1d(x, pars):
    return pars[0] + pars[1] * x


def _linear_2d(x, y, pars):
    return pars[0] + pars[1] * x + pars[2] * y


def _linear_3d(x, y, z, pars):
    return pars[0] + pars[1] * x + pars[2] * y + pars[3] * z


def _sqw_model(qh, qk, ql, en, pars):
    return pars[0] + pars[1] * qh + pars[2] * qk + pars[3] * ql + pars[4] * en


def _disp_model(qh, qk, ql, pars):
    w = pars[0] + pars[1] * qh + pars[2] * qk + pars[3] * ql
    s = np.ones_like(qh) * pars[4]
    return w, s


class TestDnDEval:
    @pytest.fixture(scope="class")
    def d1d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw"))

    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw"))

    @pytest.fixture(scope="class")
    def d3d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w3d_d3d.sqw"))

    def test_func_eval_d1d(self, d1d_obj):
        pars = [1.5, 2.0]
        res = d1d_obj.func_eval(_linear_1d, pars)
        xc = 0.5 * (d1d_obj.p[0][:-1] + d1d_obj.p[0][1:])
        expected = _linear_1d(xc, pars).reshape(d1d_obj.s.shape, order="F")
        expected[d1d_obj.npix == 0] = 0.0
        np.testing.assert_allclose(res.s, expected)
        assert np.all(res.e == 0.0)

        res_all = d1d_obj.func_eval(_linear_1d, pars, "all")
        expected_all = _linear_1d(xc, pars).reshape(d1d_obj.s.shape, order="F")
        np.testing.assert_allclose(res_all.s, expected_all)
        assert np.all(res_all.npix > 0)

    def test_func_eval_d2d(self, d2d_obj):
        pars = [1.0, 2.0, -0.5]
        res = d2d_obj.func_eval(_linear_2d, pars)
        assert isinstance(res, d2d)
        assert np.all(res.s[d2d_obj.npix == 0] == 0.0)

        res_all = d2d_obj.func_eval(_linear_2d, pars, "-all")
        assert np.all(res_all.npix > 0)

    def test_func_eval_d3d(self, d3d_obj):
        pars = [1.0, 2.0, -0.5, 0.25]
        res = d3d_obj.func_eval(_linear_3d, pars)
        assert isinstance(res, d3d)
        assert np.all(res.s[d3d_obj.npix == 0] == 0.0)

    def test_sqw_eval_d1d_d2d_d3d(self, d1d_obj, d2d_obj, d3d_obj):
        pars = [10.0, 1.0, 2.0, 3.0, 0.1]
        for obj in (d1d_obj, d2d_obj, d3d_obj):
            res = obj.sqw_eval(_sqw_model, pars)
            assert res.s.shape == obj.s.shape
            assert np.all(res.s[obj.npix == 0] == 0.0)
            assert np.all(res.e == 0.0)

            res_all = obj.sqw_eval(_sqw_model, pars, "-all")
            assert np.all(res_all.npix > 0)

    def test_dispersion_d2d(self, d2d_obj):
        pars = [5.0, 1.0, 2.0, 0.5, 3.0]
        w_disp, s_disp = d2d_obj.dispersion(_disp_model, pars, nargout=2)
        assert isinstance(w_disp, d1d)
        assert isinstance(s_disp, d1d)
        np.testing.assert_allclose(s_disp.s[s_disp.npix > 0], 3.0)

    def test_disp2sqw_eval_d2d(self, d2d_obj):
        pars = [5.0, 1.0, 2.0, 0.5, 3.0]
        res = d2d_obj.disp2sqw_eval(_disp_model, pars, 1.5, "-all")
        assert isinstance(res, d2d)
        assert np.all(res.s >= 0.0)

