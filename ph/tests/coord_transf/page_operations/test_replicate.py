"""Pytest port of ``_test/test_sqw_pageOpMethods/test_replicate.m``."""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.sqw import d2d, sqw


class TestReplicate:
    """Port of ``test_replicate.m``."""

    @pytest.fixture(scope="class")
    def sqw_1d_2d_pair(self):
        w4 = sqw.generate_cube_sqw(10).sqw_eval(
            lambda h, k, l, en, p: np.ones_like(h, dtype=float) * p[0], [1.0]
        )
        w1 = w4.cut([-4.5, 1, 4.5], [-5, 5], [-5, 5], [-5, 5])
        w2 = w4.cut([-4.5, 1, 4.5], [-5, 5], [-4.5, 1, 4.5], [-5, 5])
        return w1, w2

    def test_replicate_sqw_1Dto2D_dnd_warns(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2)
        assert isinstance(s2r, d2d)
        with pytest.warns(UserWarning, match="-set_pix"):
            s2rd = w1.replicate(w2.data, "-set_pix")
        assert isinstance(s2rd, d2d)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_sqw_1Dto2D_sqw_nopix_warns(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2)
        assert isinstance(s2r, d2d)
        w2nop = w2.copy()
        w2nop.pix = PixelDataMemory()
        with pytest.warns(UserWarning, match="-set_pix"):
            s2rd = w1.replicate(w2nop, "-set_pix")
        assert isinstance(s2rd, d2d)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_dnd_1Dto2D_sqw_nopix_warns(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2)
        assert isinstance(s2r, d2d)
        w2nop = w2.copy()
        w2nop.pix = PixelDataMemory()
        with pytest.warns(UserWarning, match="-set_pix"):
            s2rd = w1.data.replicate(w2nop, "-set_pix")
        assert isinstance(s2rd, d2d)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_dnd_1Dto2D_dnd_warns(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2)
        assert isinstance(s2r, d2d)
        with pytest.warns(UserWarning, match="-set_pix"):
            s2rd = w1.data.replicate(w2.data, "-set_pix")
        assert isinstance(s2rd, d2d)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_dnd_1Dto2D_dnd(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2)
        assert isinstance(s2r, d2d)
        s2rd = w1.data.replicate(w2.data)
        assert isinstance(s2rd, d2d)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_dnd_1Dto2D_eq_sqw(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2)
        assert isinstance(s2r, d2d)
        s2rd = w1.data.replicate(w2)
        assert isinstance(s2rd, d2d)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_dnd_1Dto2D_with_pix(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        s2r = w1.replicate(w2, "-set_pix")
        assert isinstance(s2r, sqw)
        s2rd = w1.data.replicate(w2, "-set_pix")
        assert isinstance(s2rd, sqw)
        ok, mess = s2r.equal_to_tol(s2rd)
        assert ok, mess

    def test_replicate_1Dto2D_with_pix(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        d2r = w1.replicate(w2, "-set_pix")
        np.testing.assert_array_equal(w2.data.p[0], d2r.data.p[0])
        np.testing.assert_array_equal(w2.data.p[1], d2r.data.p[1])
        ok, mess = d2r.equal_to_tol(w2)
        assert ok, mess

    def test_replicate_1Dto2D(self, sqw_1d_2d_pair) -> None:
        w1, w2 = sqw_1d_2d_pair
        d2r = w1.replicate(w2)
        assert isinstance(d2r, d2d)
        np.testing.assert_array_equal(w2.data.p[0], d2r.p[0])
        np.testing.assert_array_equal(w2.data.p[1], d2r.p[1])
