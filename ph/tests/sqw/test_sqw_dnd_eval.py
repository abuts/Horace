"""Tests for user function call interface on SQW and DnD objects.

Ported from ``_test/test_sqw/test_sqw_dnd_eval.m``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from pyhorace.algorithms import func_eval
from pyhorace.sqw import read_dnd, sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)
TEST_SQW_FILE = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")


class TestSqwDndEval:
    """Port of ``_test/test_sqw/test_sqw_dnd_eval.m``."""

    @staticmethod
    def sqw_disp_tester(
        h: np.ndarray, k: np.ndarray, l: np.ndarray, par: Any = None
    ) -> np.ndarray:
        sz = h.shape
        if sz != k.shape or sz != l.shape:
            raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
        if h.ndim < 2 or sz[1] != 1:
            raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
        return 3.0 * np.ones(sz, dtype=np.float64)

    @staticmethod
    def sqw_eval_tester(
        h: np.ndarray, k: np.ndarray, l: np.ndarray, en: np.ndarray, par: Any = None
    ) -> np.ndarray:
        sz = h.shape
        if sz != k.shape or sz != l.shape or sz != en.shape:
            raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
        if h.ndim < 2 or sz[1] != 1:
            raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
        return 3.0 * np.ones(sz, dtype=np.float64)

    @staticmethod
    def funceval_tester2D(x: np.ndarray, en: np.ndarray, par: Any = None) -> np.ndarray:
        sz = x.shape
        if sz != en.shape:
            raise RuntimeError("FUNC_EVAL:runtime_error: incorrect shape of input arrays")
        if x.ndim < 2 or sz[1] != 1:
            raise RuntimeError("FUNC_EVAL:runtime_error: incorrect shape of input arrays")
        return 2.0 * np.ones(sz, dtype=np.float64)

    @pytest.fixture(scope="class")
    def sqw_4_test(self) -> sqw:
        return sqw(TEST_SQW_FILE)

    @pytest.fixture(scope="class")
    def dnd_4_test(self):
        return read_dnd(TEST_SQW_FILE)

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="TODO(port-dependency): Requires set_instrument, set_sample, and tobyfit on sqw",
        strict=True,
    )
    def test_tobyfit(self, sqw_4_test: sqw) -> None:
        w = sqw_4_test.copy()
        w.tobyfit()

    @pytest.mark.skip(reason="New dnd objects to calculate dispersion are not implemented (skipped in MATLAB test_sqw_dnd_eval.m)")
    def test_dispersion_sqw(self, sqw_4_test: sqw) -> None:
        pass

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="TODO(port-dependency): Requires multifit_sqw_sqw on sqw",
        strict=True,
    )
    def test_fit_sqw_sqw(self, sqw_4_test: sqw) -> None:
        sqw_4_test.multifit_sqw_sqw()

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="TODO(port-dependency): Requires multifit_sqw on sqw",
        strict=True,
    )
    def test_fit_sqw(self, sqw_4_test: sqw) -> None:
        sqw_4_test.multifit_sqw()

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="TODO(port-dependency): Requires multifit_func on sqw",
        strict=True,
    )
    def test_fit_func(self, sqw_4_test: sqw) -> None:
        sqw_4_test.multifit_func()

    def test_func_eval_dnd(self, dnd_4_test) -> None:
        ds = func_eval(dnd_4_test, self.funceval_tester2D, [])
        sig = ds.s
        np.testing.assert_array_equal(sig, 2.0 * np.ones_like(sig))

