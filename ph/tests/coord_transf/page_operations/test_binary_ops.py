"""Tests for binary operations on SQW objects (port of test_binary_ops.m)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels import PixelDataBase
from pyhorace.data.sigvar import SigVar
from pyhorace.sqw.d2d import d2d
from pyhorace.sqw.sqw import sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_SQW_FILE = REPO_ROOT / "_test" / "common_data" / "sqw_2d_1.sqw"


class TestBinaryOps:
    """Port of ``_test/test_sqw_pageOpMethods/test_binary_ops.m``."""

    DOUBLE_REL_TOLERANCE = 10e-6

    @pytest.fixture(scope="class")
    def base_sqw_obj(self) -> sqw:
        s0 = sqw(str(TEST_SQW_FILE))
        return s0.recompute_bin_data()

    @pytest.fixture
    def sqw_and_dnd(self, base_sqw_obj: sqw):
        sqw_obj = base_sqw_obj.copy()
        dnd_obj = d2d(base_sqw_obj)
        return sqw_obj, dnd_obj

    def test_SQW_error_if_operand_is_char(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        with pytest.raises((HoraceError, ValueError, TypeError)):
            _ = "some char" + sqw_obj
        with pytest.raises((HoraceError, ValueError, TypeError)):
            _ = sqw_obj + "some char"

    def test_SQW_error_if_operand_is_cell_array(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        with pytest.raises((HoraceError, ValueError, TypeError)):
            _ = np.array([1, 2, 3], dtype=object) + sqw_obj
        with pytest.raises((HoraceError, ValueError, TypeError)):
            _ = sqw_obj + np.array([0], dtype=object)

    def test_SQW_error_if_operand_is_non_numeric(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        numeric_array = np.ones(sqw_obj.data.npix.shape, dtype=bool)
        with pytest.raises((HoraceError, ValueError, TypeError)):
            _ = sqw_obj + numeric_array
        with pytest.raises((HoraceError, ValueError, TypeError)):
            _ = numeric_array + sqw_obj

    def test_SQW_works_if_both_objects_are_sqw_with_no_pixel_data(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        sqw_obj.pix = PixelDataBase.create()
        res = sqw_obj - sqw_obj
        assert isinstance(res, sqw)
        np.testing.assert_allclose(res.data.s, np.zeros_like(res.data.s))

    def test_subtracting_dnd_from_sqw(self, sqw_and_dnd):
        sqw_obj, dnd_obj = sqw_and_dnd
        res = sqw_obj - dnd_obj
        np.testing.assert_allclose(res.data.s, np.zeros_like(res.data.s), atol=1e-9)
        test_obj = res.recompute_bin_data()
        ok, mess = res.equal_to_tol(test_obj, 1e-9)
        assert ok, mess

        rec = dnd_obj + res
        np.testing.assert_allclose(rec.data.s, sqw_obj.data.s, atol=1e-9)
        np.testing.assert_allclose(rec.pix.signal, sqw_obj.pix.signal, atol=1e-9)

    def test_adding_sqw_and_dnd_objects_1st_operand_is_sqw_returns_sqw(self, sqw_and_dnd):
        sqw_obj, dnd_obj = sqw_and_dnd
        out = sqw_obj + dnd_obj
        assert isinstance(out, sqw)
        expected_signal = sqw_obj.data.s + dnd_obj.s
        np.testing.assert_allclose(out.data.s, expected_signal, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_adding_sqw_and_dnd_objects_2nd_operand_is_sqw_returns_sqw(self, sqw_and_dnd):
        sqw_obj, dnd_obj = sqw_and_dnd
        out = dnd_obj + sqw_obj
        assert isinstance(out, sqw)
        expected_signal = sqw_obj.data.s + dnd_obj.s
        np.testing.assert_allclose(out.data.s, expected_signal, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_dnd_minus_equivalent_sqw_returns_sqw_with_zero_image_data(self, sqw_and_dnd):
        sqw_obj, dnd_obj = sqw_and_dnd
        out = dnd_obj - sqw_obj
        assert isinstance(out, sqw)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_diff = out.data.s / np.maximum(dnd_obj.s, sqw_obj.data.s)
        scaled_diff[np.isnan(scaled_diff)] = 0.0
        np.testing.assert_allclose(scaled_diff, np.zeros_like(sqw_obj.data.s), atol=1e-7)

    def test_sqw_minus_equivalent_dnd_returns_sqw_with_zero_image_data(self, sqw_and_dnd):
        sqw_obj, dnd_obj = sqw_and_dnd
        out = sqw_obj - dnd_obj
        assert isinstance(out, sqw)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_diff = out.data.s / np.maximum(dnd_obj.s, sqw_obj.data.s)
        scaled_diff[np.isnan(scaled_diff)] = 0.0
        np.testing.assert_allclose(scaled_diff, np.zeros_like(sqw_obj.data.s), atol=1e-7)

    def test_subtracting_dnd_from_sqw_returns_sqw(self, sqw_and_dnd):
        sqw_obj, dnd_obj = sqw_and_dnd
        dnd_obj.s = np.ones_like(dnd_obj.npix, dtype=np.float64)
        out = sqw_obj - dnd_obj
        assert isinstance(out, sqw)
        expected_signal = sqw_obj.data.s - 1.0
        expected_signal[sqw_obj.data.npix == 0] = 0.0
        np.testing.assert_allclose(out.data.s, expected_signal, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_subtracting_scalar_from_sqw_returns_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        scalar_operand = 1.3
        out = sqw_obj - scalar_operand
        assert isinstance(out, sqw)
        expected_signal = sqw_obj.data.s - scalar_operand
        expected_signal[sqw_obj.data.npix == 0] = 0.0
        np.testing.assert_allclose(out.data.s, expected_signal, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_subtracting_sqw_from_scalar_returns_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        scalar_operand = 0.3
        out = scalar_operand - sqw_obj
        assert isinstance(out, sqw)
        expected_signal = scalar_operand - sqw_obj.data.s
        expected_signal[sqw_obj.data.npix == 0] = 0.0
        np.testing.assert_allclose(out.data.s, expected_signal, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_subtracting_two_sqw_objects_returns_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        out = sqw_obj - sqw_obj
        assert isinstance(out, sqw)
        np.testing.assert_allclose(out.data.s, np.zeros_like(out.data.s))

    def test_adding_sqw_and_sigvar_with_sqw_1st_operand_returns_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        rng = np.random.default_rng(42)
        signal = 1e5 * (1.0 + rng.random(sqw_obj.data.s.shape))
        variance = 1e5 * (1.0 + rng.random(sqw_obj.data.s.shape))
        out = sqw_obj + SigVar(signal, variance)
        assert isinstance(out, sqw)
        expected_signal = sqw_obj.data.s + signal
        expected_signal[sqw_obj.data.npix == 0] = 0.0
        np.testing.assert_allclose(out.data.s, expected_signal, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_adding_sqw_and_sigvar_with_sqw_2nd_operand_returns_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        rng = np.random.default_rng(42)
        signal = 1.0 + rng.random(sqw_obj.data.s.shape)
        variance = 1.0 + rng.random(sqw_obj.data.s.shape)
        out = SigVar(signal.copy(), variance.copy()) + sqw_obj
        assert isinstance(out, sqw)

        npix = sqw_obj.data.npix
        signal[npix == 0] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = variance / npix
        variance[npix == 0] = 0.0
        expected_sigvar = SigVar(sqw_obj.data.s + signal, sqw_obj.data.e + variance)
        np.testing.assert_allclose(out.data.s, expected_sigvar.s, rtol=self.DOUBLE_REL_TOLERANCE)
        np.testing.assert_allclose(out.data.e, expected_sigvar.e, rtol=self.DOUBLE_REL_TOLERANCE)

    def test_adding_sqw_and_zero_returns_same_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        out = 0 + sqw_obj
        assert isinstance(out, sqw)
        ok, mess = out.equal_to_tol(sqw_obj)
        assert ok, mess

    def test_adding_sqw_and_zero_data_returns_same_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        signal = np.zeros_like(sqw_obj.data.s)
        out = signal + sqw_obj
        assert isinstance(out, sqw)
        ok, mess = out.equal_to_tol(sqw_obj)
        assert ok, mess

    def test_adding_sqw_and_zero_sigvar_returns_same_sqw(self, sqw_and_dnd):
        sqw_obj, _ = sqw_and_dnd
        signal = np.zeros_like(sqw_obj.data.s)
        variance = np.zeros_like(sqw_obj.data.s)
        out = SigVar(signal, variance) + sqw_obj
        assert isinstance(out, sqw)
        ok, mess = out.equal_to_tol(sqw_obj)
        assert ok, mess
