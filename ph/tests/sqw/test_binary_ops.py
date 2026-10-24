"""Tests for binary operations on DnD objects, ported from test_binary_ops.m."""

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw import d1d, d2d, read_dnd, sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)

BINARY_OPS = [
    ("plus", lambda a, b: a + b),
    ("minus", lambda a, b: a - b),
    ("mtimes", lambda a, b: a * b),
    ("mrdivide", lambda a, b: a / b),
    ("mpower", lambda a, b: a ** b),
]


class TestBinaryOpsDnD:
    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w2d_qq_d2d.sqw"))

    @pytest.fixture(scope="class")
    def sqw_obj(self):
        return sqw(os.path.join(COMMON_DATA_DIR, "w2d_qq_sqw.sqw"))

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS)
    def test_dnd_and_double_scalar(self, d2d_obj, op_name, op_func):
        d = d2d_obj.copy()
        d.s = np.abs(d.s) + 1.5
        scalar = 2.0

        res = op_func(d, scalar)
        sv_res = getattr(sigvar(d.s, d.e), op_name)(scalar)
        mask = d.npix == 0
        exp_s = sv_res.s.copy()
        exp_e = sv_res.e.copy()
        exp_s[mask] = 0.0
        exp_e[mask] = 0.0

        np.testing.assert_allclose(res.s, exp_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.e, exp_e, rtol=1e-12, atol=1e-12)

        if op_name != "mpower":
            res_rev = op_func(scalar, d)
            sv_rev = getattr(sigvar(scalar, None), op_name)(sigvar(d.s, d.e))
            exp_s2 = sv_rev.s.copy()
            exp_e2 = sv_rev.e.copy()
            exp_s2[mask] = 0.0
            exp_e2[mask] = 0.0
            np.testing.assert_allclose(res_rev.s, exp_s2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(res_rev.e, exp_e2, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS[:4])
    def test_dnd_and_dnd(self, d2d_obj, op_name, op_func):
        d1 = d2d_obj.copy()
        d1.s = np.abs(d1.s) + 2.0
        d2 = d2d_obj.copy()
        d2.s = np.abs(d2.s) + 1.0

        res = op_func(d1, d2)
        sv_res = getattr(sigvar(d1.s, d1.e), op_name)(sigvar(d2.s, d2.e))
        mask = (d1.npix == 0) | (d2.npix == 0)
        exp_s = sv_res.s.copy()
        exp_e = sv_res.e.copy()
        exp_s[mask] = 0.0
        exp_e[mask] = 0.0

        np.testing.assert_allclose(res.s, exp_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.e, exp_e, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS[:4])
    def test_dnd_and_sigvar(self, d2d_obj, op_name, op_func):
        d = d2d_obj.copy()
        d.s = np.abs(d.s) + 2.0
        sv = sigvar(np.ones_like(d.s) * 3.0, np.ones_like(d.e) * 0.5)

        res = op_func(d, sv)
        sv_res = getattr(sigvar(d.s, d.e), op_name)(sv)
        mask = d.npix == 0
        exp_s = sv_res.s.copy()
        exp_e = sv_res.e.copy()
        exp_s[mask] = 0.0
        exp_e[mask] = 0.0

        np.testing.assert_allclose(res.s, exp_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.e, exp_e, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("op_name,op_func", BINARY_OPS[:4])
    def test_dnd_and_sqw(self, d2d_obj, sqw_obj, op_name, op_func):
        d = d2d_obj.copy()
        d.s = np.abs(d.s) + 2.0
        s_obj = sqw_obj.copy()
        s_obj.data.s = np.abs(s_obj.data.s) + 1.0

        res = op_func(d, s_obj)
        assert isinstance(res, d2d)
        sv_res = getattr(sigvar(d.s, d.e), op_name)(sigvar(s_obj.data.s, s_obj.data.e))
        mask = (d.npix == 0) | (s_obj.data.npix == 0)
        exp_s = sv_res.s.copy()
        exp_e = sv_res.e.copy()
        exp_s[mask] = 0.0
        exp_e[mask] = 0.0

        np.testing.assert_allclose(res.s, exp_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.e, exp_e, rtol=1e-12, atol=1e-12)

    def test_incompatible_dnd_shapes_raise_error(self, d2d_obj):
        d_sec = d2d_obj.section([0, 0.2], [0, 0.2])
        with pytest.raises((HoraceError, ValueError)):
            _ = d2d_obj + d_sec

