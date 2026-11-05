"""Pytest port of ``_test/test_sqw_pageOpMethods/test_sqw_funceval.m``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyhorace.sqw.read_dnd import read_dnd
from pyhorace.sqw.sqw import sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_COMMON_DIR = REPO_ROOT / "_test" / "common_data"


def _sqw_disp_tester(h: np.ndarray, k: np.ndarray, l: np.ndarray, par: Any = None) -> np.ndarray:
    sz = h.shape
    if sz != k.shape or sz != l.shape:
        raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
    if len(sz) != 2 or sz[1] != 1:
        raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
    return 3.0 * np.ones(sz, dtype=np.float64)


def _sqw_eval_tester(
    h: np.ndarray, k: np.ndarray, l: np.ndarray, en: np.ndarray, par: Any = None
) -> np.ndarray:
    sz = h.shape
    if sz != k.shape or sz != l.shape or sz != en.shape:
        raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
    if len(sz) != 2 or sz[1] != 1:
        raise RuntimeError("SQW_EVAL:runtime_error: incorrect shape of input arrays")
    return 3.0 * np.ones(sz, dtype=np.float64)


def _funceval_tester2D(x: np.ndarray, en: np.ndarray, par: Any = None) -> np.ndarray:
    sz = x.shape
    if sz != en.shape:
        raise RuntimeError("FUNC_EVAL:runtime_error: incorrect shape of input arrays")
    if len(sz) != 2 or sz[1] != 1:
        raise RuntimeError("FUNC_EVAL:runtime_error: incorrect shape of input arrays")
    return 2.0 * np.ones(sz, dtype=np.float64)


class TestSqwFuncEval:
    """Port of ``test_sqw_funceval.m``."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        test_sqw_file = str(TEST_COMMON_DIR / "sqw_2d_2.sqw")
        self.sqw_4_test = sqw(test_sqw_file)
        self.dnd_4_test = read_dnd(test_sqw_file)

    def test_func_eval_sqw(self) -> None:
        ds = self.sqw_4_test.func_eval(_funceval_tester2D, [], "-all")

        sig = ds.data.s
        pix_sig = ds.pix.signal

        np.testing.assert_array_equal(sig, 2.0 * np.ones_like(sig))
        np.testing.assert_array_equal(pix_sig, 2.0 * np.ones_like(pix_sig))

    def test_sqw_eval_aver(self) -> None:
        ds = self.sqw_4_test.sqw_eval(_sqw_eval_tester, [], "-average")

        sig = ds.data.s
        np.testing.assert_array_equal(sig, 3.0 * np.ones_like(sig))

        pix = ds.pix
        assert float(pix.signal.flat[1]) == float(sig.flat[0])
        assert float(sig.flat[0]) == float(pix.signal.flat[0])

    def test_sqw_eval(self) -> None:
        ds = self.sqw_4_test.sqw_eval(_sqw_eval_tester, [])

        sig = ds.pix.signal
        np.testing.assert_array_equal(sig, 3.0 * np.ones_like(sig))
