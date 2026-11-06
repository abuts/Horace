"""Tests for func_eval, sqw_eval, dispersion, and disp2sqw_eval on DnD and SQW objects.

Ported from ``_test/test_dnd_class/test_dnd_eval.m``.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np
import pytest

from pyhorace.algorithms import disp2sqw_eval, func_eval, sqw_eval
from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import d1d, d2d, d3d, read_dnd, read_sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


def _linear_1d(x: np.ndarray, pars: Any) -> np.ndarray:
    return pars[0] + pars[1] * x


def _linear_2d(x: np.ndarray, y: np.ndarray, pars: Any) -> np.ndarray:
    return pars[0] + pars[1] * x + pars[2] * y


def _linear_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, pars: Any) -> np.ndarray:
    return pars[0] + pars[1] * x + pars[2] * y + pars[3] * z


def _sqw_model(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, en: np.ndarray, pars: Any
) -> np.ndarray:
    return pars[0] + pars[1] * qh + pars[2] * qk + pars[3] * ql + pars[4] * en


def _disp_model(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, pars: Any
) -> Tuple[np.ndarray, np.ndarray]:
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


class TestDndEvalMatlabSuite:
    """Direct port of ``_test/test_dnd_class/test_dnd_eval.m``."""

    @staticmethod
    def sqw_eval_tester(
        h: np.ndarray, k: np.ndarray, l: np.ndarray, en: np.ndarray, par: Any
    ) -> np.ndarray:
        sz = h.shape
        if sz != k.shape or sz != l.shape or sz != en.shape:
            raise RuntimeError("SQW_EVAL_TESTER:runtime_error: unequal size input arrays")
        if h.ndim < 2 or h.shape[1] != 1:
            raise RuntimeError("SQW_EVAL_TESTER:runtime_error: incorrect shape of input arrays")
        dis = np.ones_like(h, dtype=np.float64)
        dis.ravel(order="F")[0] = float(h.size)
        return dis

    @staticmethod
    def disp2sqw_eval_tester2D(
        qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, p: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        w = np.ones_like(qh, dtype=np.float64)
        s = np.ones_like(qh, dtype=np.float64)
        return w, s

    @staticmethod
    def funceval_tester2D(x: np.ndarray, en: np.ndarray, par: Any) -> np.ndarray:
        sz = x.shape
        if sz != en.shape:
            raise RuntimeError("FUNC_EVAL_TESTER:runtime_error: unequal size input arrays")
        if x.ndim < 2 or x.shape[1] != 1:
            raise RuntimeError("FUNC_EVAL_TESTER:runtime_error: incorrect shape of input arrays")
        dis = np.ones_like(x, dtype=np.float64)
        dis.ravel(order="F")[0] = float(x.size)
        return dis

    @pytest.fixture(scope="class")
    def eval_objs(self):
        return {
            "d1d_obj": read_dnd(os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")),
            "d2d_obj": read_dnd(os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")),
            "sqw1d_obj": read_sqw(os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")),
            "sqw2d_obj": read_sqw(os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")),
        }

    def test_disp2sqw_eval_sqw(self, eval_objs):
        ds = disp2sqw_eval(
            eval_objs["sqw2d_obj"],
            self.disp2sqw_eval_tester2D,
            [],
            1.0,
            "-all",
        )
        assert ds.data.s.shape == eval_objs["d2d_obj"].s.shape

    def test_disp2sqw_eval_dnd(self, eval_objs):
        ds = disp2sqw_eval(
            eval_objs["d2d_obj"],
            self.disp2sqw_eval_tester2D,
            [],
            1.0,
            "-all",
        )
        assert ds.s.shape == eval_objs["d2d_obj"].s.shape

    def test_func_eval_dnd(self, eval_objs):
        ds = func_eval(eval_objs["d2d_obj"], self.funceval_tester2D, [])
        sig = ds.s.ravel(order="F")
        assert sig[1] == 1.0
        assert sig[0] == float(sig.size)

    def test_sqw_eval_sqw(self, eval_objs):
        ds = sqw_eval(eval_objs["sqw2d_obj"], self.sqw_eval_tester, [])
        assert ds.data.s.shape == eval_objs["d2d_obj"].s.shape
        expected = np.ones_like(eval_objs["d2d_obj"].s).ravel(order="F")
        np.testing.assert_array_equal(ds.data.s.ravel(order="F")[1:], expected[1:])

    def test_sqw_eval_dnd(self, eval_objs):
        ds = sqw_eval(eval_objs["d2d_obj"], self.sqw_eval_tester, [])
        assert ds.s.shape == eval_objs["d2d_obj"].s.shape
        expected = np.ones_like(eval_objs["d2d_obj"].s).ravel(order="F")
        np.testing.assert_array_equal(ds.s.ravel(order="F")[1:], expected[1:])
